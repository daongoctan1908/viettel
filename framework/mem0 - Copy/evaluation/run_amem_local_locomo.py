import argparse
import json
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

try:
    from prompts import ANSWER_PROMPT
except ImportError:
    from evaluation.prompts import ANSWER_PROMPT


LOCOMO_DATETIME_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
)
LOCOMO_TIMEZONE_ASSUMPTION = "UTC"
CHECKPOINT_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run A-MEM on LOCOMO with native A-MEM retrieval defaults while emitting the same "
            "result schema used by the local Mem0 runner."
        )
    )
    parser.add_argument("--dataset", default="evaluation/dataset/locomo10.json")
    parser.add_argument("--amem_src", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--method", default="amem_local_native_fair_eval")

    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--top_k",
        type=int,
        default=30,
        help=(
            "Retrieval K. In global_official mode this is global K; use 30 to match Mem0's "
            "15-per-speaker / 30-total retrieval budget. In per_speaker mode use 15 to match Mem0 directly."
        ),
    )
    parser.add_argument("--search_threshold", type=float, default=None)
    parser.add_argument("--retrieved_memory_token_encoding", default="cl100k_base")
    parser.add_argument("--retrieval_token_budget", type=int, default=None)
    parser.add_argument(
        "--retrieval_budget_strategy",
        choices=["interleave_rank", "balanced_per_speaker", "global_score"],
        default="interleave_rank",
    )
    parser.add_argument(
        "--namespace_mode",
        choices=["per_speaker", "global_official"],
        default="global_official",
        help="global_official preserves the A-MEM repo runner; per_speaker is a Mem0-normalized ablation.",
    )
    parser.add_argument(
        "--retrieval_mode",
        choices=["strict_top_k", "linked_neighbors"],
        default="linked_neighbors",
        help="linked_neighbors preserves A-MEM linked retrieval; strict_top_k is a Mem0-normalized ablation.",
    )
    parser.add_argument(
        "--query_mode",
        choices=["question", "keywords"],
        default="keywords",
        help="keywords preserves the A-MEM repo retrieval pipeline; question is a Mem0-normalized ablation.",
    )
    parser.add_argument(
        "--content_mode",
        choices=["official", "timestamp_context"],
        default="official",
        help="official matches the A-MEM repo turn text; timestamp_context injects Mem0-style timestamp hints into content.",
    )
    parser.add_argument(
        "--answer_prompt_mode",
        choices=["shared_mem0", "amem_official"],
        default="amem_official",
        help="amem_official uses the A-MEM repo's category prompt; shared_mem0 is a prompt-normalized ablation.",
    )

    parser.add_argument("--max_conversations", type=int, default=5)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--max_qa", type=int, default=None)
    parser.add_argument("--skip_category_5", dest="skip_category_5", action="store_true", default=True)
    parser.add_argument("--include_category_5", dest="skip_category_5", action="store_false")

    parser.add_argument("--candidate_extraction_max_tokens", type=int, default=None)
    parser.add_argument("--update_decision_max_tokens", type=int, default=None)
    parser.add_argument("--answer_max_tokens", type=int, default=None)
    parser.add_argument(
        "--summary_max_tokens",
        type=int,
        default=None,
        help="Accepted for Mem0 command compatibility. Native A-MEM has no rolling-summary phase.",
    )
    parser.add_argument("--api_max_retries", type=int, default=10)
    parser.add_argument("--api_timeout_sec", type=float, default=600.0)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--memory_cache_dir", default=None)
    parser.add_argument("--rebuild_memory_cache", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def load_root_env() -> Path:
    cwd = Path.cwd()
    env_path = cwd / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        return cwd

    mem0_root = Path(__file__).resolve().parents[1]
    mem0_env = mem0_root / ".env"
    if mem0_env.exists():
        load_dotenv(mem0_env, override=True)
        return mem0_root

    load_dotenv(override=True)
    return cwd


def get_env_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    value = value.strip() if value else ""
    return value or None


def create_chat_client(api_max_retries: int, api_timeout_sec: float) -> Tuple[OpenAI, Dict[str, Any]]:
    api_key = get_env_value("BEEKNOEE_API_KEY")
    base_url = get_env_value("BEEKNOEE_BASE_URL")
    if not api_key:
        raise RuntimeError("BEEKNOEE_API_KEY is required for chat completions.")
    if not base_url:
        raise RuntimeError("BEEKNOEE_BASE_URL is required for chat completions.")
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=api_max_retries,
        timeout=api_timeout_sec,
    ), {
        "provider": "beeknoee",
        "uses_beeknoee_api_key": True,
        "uses_openai_api_key_for_chat": False,
        "base_url_configured": True,
        "max_retries": api_max_retries,
        "timeout_sec": api_timeout_sec,
    }


def normalize_dataset_path(path_str: str, root_dir: Path) -> Path:
    path = Path(path_str)
    if path.exists():
        return path
    candidate = root_dir / path_str
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Dataset not found: {path_str} or {candidate}")


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    with open(temp_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.flush()
        os.fsync(file_handle.fileno())
    os.replace(temp_path, path)


def parse_locomo_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    normalized = str(value).strip()
    for date_format in LOCOMO_DATETIME_FORMATS:
        try:
            return datetime.strptime(normalized, date_format).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def observation_time_fields(session_date: Optional[str]) -> Dict[str, Any]:
    parsed = parse_locomo_datetime(session_date)
    if session_date and parsed is None:
        raise ValueError(f"Unsupported LOCOMO session timestamp: {session_date}")
    return {
        "session_timestamp_raw": session_date,
        "observed_at": parsed.isoformat().replace("+00:00", "Z") if parsed else None,
        "observed_at_epoch": int(parsed.timestamp()) if parsed else None,
        "observed_at_timezone_assumption": LOCOMO_TIMEZONE_ASSUMPTION if parsed else None,
    }


def get_session_indices(conversation: Dict[str, Any]) -> List[int]:
    indices = []
    for key, value in conversation.items():
        if key.startswith("session_") and key.count("_") == 1 and isinstance(value, list):
            try:
                indices.append(int(key.split("_")[1]))
            except ValueError:
                pass
    return sorted(indices)


def slugify(value: Any) -> str:
    text = str(value or "unknown").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text.strip("_") or "unknown"


def make_user_id(sample_id: str, conversation_index: int, speaker: str, run_id: str) -> str:
    return f"locomo_{slugify(sample_id)}_{conversation_index}_{slugify(speaker)}_{run_id}"


def dialogue_to_content(turn: Dict[str, Any], session_index: int, session_date: Optional[str]) -> str:
    speaker = turn.get("speaker", "Unknown")
    text = turn.get("text", "")

    extra_parts = []
    if turn.get("blip_caption"):
        extra_parts.append(f"Image caption: {turn['blip_caption']}")
    if turn.get("query"):
        extra_parts.append(f"Image query: {turn['query']}")

    extra = ""
    if extra_parts:
        extra = "\n" + "\n".join(extra_parts)

    timestamp_context = (
        f"[Session {session_index} timestamp: {session_date}. "
        "Use this timestamp to resolve relative date references in this utterance.] "
    )
    return f"{timestamp_context}{speaker}: {text}{extra}"


def official_amem_turn_content(turn: Dict[str, Any]) -> str:
    text = turn.get("text", "")
    if turn.get("img_url") and turn.get("blip_caption"):
        caption_text = f"[Image: {turn['blip_caption']}]"
        text = f"{caption_text} {text}" if text else caption_text
    return f"Speaker {turn.get('speaker', 'Unknown')}says : {text}"


def build_add_content(turn: Dict[str, Any], session_index: int, session_date: Optional[str], content_mode: str) -> str:
    if content_mode == "official":
        return official_amem_turn_content(turn)
    return dialogue_to_content(turn, session_index, session_date)


def base_retrieve_k(args: argparse.Namespace) -> int:
    return args.top_k * 2 if args.namespace_mode == "per_speaker" else args.top_k


def build_dia_lookup(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    dia_lookup = {}
    for session_index in get_session_indices(conversation):
        session_key = f"session_{session_index}"
        session_date = conversation.get(f"{session_key}_date_time")
        for turn in conversation.get(session_key) or []:
            dia_id = turn.get("dia_id")
            if not dia_id:
                continue
            dia_lookup[dia_id] = {
                "session": session_key,
                "session_index": session_index,
                "session_date": session_date,
                "dia_id": dia_id,
                "speaker": turn.get("speaker"),
                "text": turn.get("text"),
                "blip_caption": turn.get("blip_caption"),
                "img_url": turn.get("img_url"),
                "query": turn.get("query"),
            }
    return dia_lookup


def get_evidence_texts(evidence_ids: List[str], dia_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    evidence_texts = []
    for evidence_id in evidence_ids or []:
        evidence_texts.append(dia_lookup.get(evidence_id) or {"dia_id": evidence_id, "missing": True})
    return evidence_texts


def memory_observed_at(item: Dict[str, Any]) -> Optional[str]:
    metadata = item.get("metadata") or {}
    return item.get("observed_at") or metadata.get("observed_at") or item.get("timestamp")


def memory_event_at(item: Dict[str, Any]) -> Optional[str]:
    metadata = item.get("metadata") or {}
    return item.get("event_at") or metadata.get("event_at")


def memory_event_time_text(item: Dict[str, Any]) -> Optional[str]:
    metadata = item.get("metadata") or {}
    return item.get("event_time_text") or metadata.get("event_time_text")


def format_memories_for_prompt(memories: List[Dict[str, Any]]) -> List[str]:
    formatted = []
    for item in memories:
        observed_at = memory_observed_at(item) or "unknown time"
        event_at = memory_event_at(item)
        event_time_text = memory_event_time_text(item)
        memory_text = item.get("memory", "")
        temporal_context = f"Observed at {observed_at}"
        if event_at:
            temporal_context += f"; event occurred at {event_at}"
        if event_time_text:
            temporal_context += f"; source time expression: {event_time_text}"
        formatted.append(f"{temporal_context}: {memory_text}")
    return formatted


def count_text_tokens(encoding: tiktoken.Encoding, texts: List[str]) -> int:
    return sum(len(encoding.encode(text)) for text in texts)


def count_memory_tokens(encoding: tiktoken.Encoding, memories: List[Dict[str, Any]]) -> int:
    return count_text_tokens(encoding, format_memories_for_prompt(memories))


def empty_event_counts() -> Dict[str, int]:
    return {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0, "UNKNOWN": 0}


def update_event_counts(counts: Dict[str, int], event: str) -> None:
    normalized = str(event or "UNKNOWN").upper()
    if normalized not in counts:
        normalized = "UNKNOWN"
    counts[normalized] += 1


def safe_usage_dict(usage: Any) -> Dict[str, Any]:
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def answer_question(
    chat_client: OpenAI,
    *,
    llm_model: str,
    question: str,
    speaker_1_name: str,
    speaker_2_name: str,
    speaker_1_memories: List[Dict[str, Any]],
    speaker_2_memories: List[Dict[str, Any]],
    answer_max_tokens: Optional[int],
) -> Tuple[str, Dict[str, Any], str]:
    template = Template(ANSWER_PROMPT)
    prompt = template.render(
        speaker_1_user_id=speaker_1_name,
        speaker_2_user_id=speaker_2_name,
        speaker_1_memories=json.dumps(format_memories_for_prompt(speaker_1_memories), indent=4, ensure_ascii=False),
        speaker_2_memories=json.dumps(format_memories_for_prompt(speaker_2_memories), indent=4, ensure_ascii=False),
        question=question,
    )
    kwargs = {
        "model": llm_model,
        "messages": [{"role": "system", "content": prompt}],
        "temperature": 0,
    }
    if answer_max_tokens is not None:
        kwargs["max_tokens"] = answer_max_tokens

    response = chat_client.chat.completions.create(**kwargs)
    answer = (response.choices[0].message.content or "").strip()
    usage = safe_usage_dict(response.usage)

    if not answer:
        retry_kwargs = {
            "model": llm_model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        "The previous response was empty. Return the answer requested by the system prompt. "
                        "Do not return an empty response. If the memories do not contain enough evidence, "
                        "answer exactly: No information available."
                    ),
                },
            ],
            "temperature": 0,
        }
        if answer_max_tokens is not None:
            retry_kwargs["max_tokens"] = answer_max_tokens
        retry_response = chat_client.chat.completions.create(**retry_kwargs)
        answer = (retry_response.choices[0].message.content or "").strip() or "No information available."
        retry_usage = safe_usage_dict(retry_response.usage)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = int(usage.get(key) or 0) + int(retry_usage.get(key) or 0)
        usage["empty_answer_retry_count"] = 1
    else:
        usage["empty_answer_retry_count"] = 0

    return answer, usage, prompt


def format_amem_official_context(memories: List[Dict[str, Any]]) -> str:
    lines = []
    for item in memories:
        metadata = item.get("metadata") or {}
        lines.append(
            "talk start time:"
            + str(memory_observed_at(item) or metadata.get("timestamp") or "")
            + " memory content: "
            + str(item.get("memory") or "")
        )
    return "\n".join(lines)


def answer_question_amem_official(
    chat_client: OpenAI,
    *,
    llm_model: str,
    question: str,
    category: str,
    ground_truth: str,
    retrieved_memories: List[Dict[str, Any]],
    answer_max_tokens: Optional[int],
) -> Tuple[str, Dict[str, Any], str]:
    context = format_amem_official_context(retrieved_memories)
    if category == "5":
        prompt = (
            f"Based on the context: {context}, answer the following question. {question}\n\n"
            f"Select the correct answer: {ground_truth} or Not mentioned in the conversation  Short answer:"
        )
        temperature = 0.5
    elif category == "2":
        prompt = (
            f"Based on the context: {context}, answer the following question. "
            "Use DATE of CONVERSATION to answer with an approximate date.\n"
            "Please generate the shortest possible answer, using words from the conversation where possible, "
            "and avoid using any subjects.\n\n"
            f"Question: {question} Short answer:"
        )
        temperature = 0.7
    else:
        prompt = (
            f"Based on the context: {context}, write an answer in the form of a short phrase for the following question. "
            "Answer with exact words from the context whenever possible.\n\n"
            f"Question: {question} Short answer:"
        )
        temperature = 0.7

    kwargs = {
        "model": llm_model,
        "messages": [{"role": "system", "content": "Follow the user's instruction exactly."}, {"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if answer_max_tokens is not None:
        kwargs["max_tokens"] = answer_max_tokens
    response = chat_client.chat.completions.create(**kwargs)
    answer = (response.choices[0].message.content or "").strip()
    usage = safe_usage_dict(response.usage)
    usage["empty_answer_retry_count"] = 0
    return answer or "No information available.", usage, prompt


def load_amem_classes(amem_src: Path):
    sys.path.insert(0, str(amem_src))
    from memory_layer_robust import RobustAgenticMemorySystem
    from llm_text_parsers import parse_keywords_response

    return RobustAgenticMemorySystem, parse_keywords_response


def create_amem_system(
    RobustAgenticMemorySystem,
    *,
    embedding_model: str,
    llm_model: str,
    api_max_retries: int,
    api_timeout_sec: float,
    candidate_extraction_max_tokens: Optional[int],
    update_decision_max_tokens: Optional[int],
):
    api_key = get_env_value("BEEKNOEE_API_KEY")
    base_url = get_env_value("BEEKNOEE_BASE_URL")
    if not api_key:
        raise RuntimeError("BEEKNOEE_API_KEY is required for A-MEM LLM calls.")
    if not base_url:
        raise RuntimeError("BEEKNOEE_BASE_URL is required for A-MEM LLM calls.")

    os.environ["OPENAI_MAX_RETRIES"] = str(api_max_retries)
    os.environ["OPENAI_TIMEOUT"] = str(api_timeout_sec)
    return RobustAgenticMemorySystem(
        model_name=embedding_model,
        llm_backend="openai",
        llm_model=llm_model,
        api_key=api_key,
        api_base=base_url,
        candidate_extraction_max_tokens=candidate_extraction_max_tokens,
        update_decision_max_tokens=update_decision_max_tokens,
    )


def save_system_cache(system: Any, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "memories.pkl", "wb") as file_handle:
        pickle.dump(system.memories, file_handle)
    system.retriever.save(str(cache_dir / "retriever.pkl"), str(cache_dir / "retriever_embeddings.npy"))


def load_system_cache(system: Any, cache_dir: Path, model_name: str) -> bool:
    memories_file = cache_dir / "memories.pkl"
    retriever_file = cache_dir / "retriever.pkl"
    embeddings_file = cache_dir / "retriever_embeddings.npy"
    if not memories_file.exists():
        return False
    with open(memories_file, "rb") as file_handle:
        system.memories = pickle.load(file_handle)
    if retriever_file.exists() and embeddings_file.exists():
        try:
            with open(retriever_file, "rb") as file_handle:
                retriever_state = pickle.load(file_handle)
            cached_model_name = retriever_state.get("model_name")
        except Exception:
            cached_model_name = None
        if cached_model_name and cached_model_name != model_name:
            print(
                f"Embedding cache model mismatch ({cached_model_name} != {model_name}); "
                "rebuilding retriever from cached memories."
            )
            system.retriever = system.retriever.load_from_local_memory(system.memories, model_name)
            save_system_cache(system, cache_dir)
            return True
        system.retriever = system.retriever.load(str(retriever_file), str(embeddings_file))
    else:
        system.retriever = system.retriever.load_from_local_memory(system.memories, model_name)
    return True


def sample_cache_dir(memory_cache_dir: Path, sample_id: str, namespace: str) -> Path:
    return memory_cache_dir / slugify(sample_id) / slugify(namespace)


def note_memory_text(note: Any) -> str:
    parts = [f"content: {note.content}"]
    if getattr(note, "context", None):
        parts.append(f"context: {note.context}")
    if getattr(note, "keywords", None):
        parts.append("keywords: " + ", ".join(str(x) for x in note.keywords))
    if getattr(note, "tags", None):
        parts.append("tags: " + ", ".join(str(x) for x in note.tags))
    return " ".join(parts)


def note_to_memory_item(
    note: Any,
    *,
    sample_id: str,
    source_speaker: str,
    user_id: str,
    rank: Optional[int] = None,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    time_fields = observation_time_fields(getattr(note, "timestamp", None))
    metadata = {
        "source": "locomo",
        "sample_id": sample_id,
        "timestamp": getattr(note, "timestamp", None),
        "source_speaker": source_speaker,
        "amem_context": getattr(note, "context", None),
        "amem_keywords": getattr(note, "keywords", []),
        "amem_tags": getattr(note, "tags", []),
        "amem_links": getattr(note, "links", []),
        **time_fields,
    }
    return {
        "id": getattr(note, "id", None),
        "user_id": user_id,
        "memory": note_memory_text(note),
        "metadata": metadata,
        "rank": rank,
        "score": score,
        "source_user_id": user_id,
        "source_speaker": source_speaker,
        **time_fields,
    }


def similarity_scores(system: Any, query: str) -> Dict[int, float]:
    retriever = system.retriever
    if not getattr(retriever, "corpus", None) or getattr(retriever, "embeddings", None) is None:
        return {}
    query_embedding = retriever.model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], retriever.embeddings)[0]
    return {idx: float(score) for idx, score in enumerate(similarities)}


def retrieve_amem_memories(
    system: Any,
    *,
    query: str,
    top_k: int,
    retrieval_mode: str,
    sample_id: str,
    source_speaker: str,
    user_id: str,
) -> Tuple[List[Dict[str, Any]], float]:
    started = time.time()
    if not system.memories:
        return [], time.time() - started

    base_indices = list(system.retriever.search(query, top_k))
    all_notes = list(system.memories.values())
    scores = similarity_scores(system, query)

    selected_indices = []
    seen = set()
    for idx in base_indices:
        idx_int = int(idx)
        if idx_int < 0 or idx_int >= len(all_notes) or idx_int in seen:
            continue
        selected_indices.append(idx_int)
        seen.add(idx_int)
        if retrieval_mode == "linked_neighbors":
            for linked_idx in getattr(all_notes[idx_int], "links", []) or []:
                try:
                    linked_int = int(linked_idx)
                except (TypeError, ValueError):
                    continue
                if 0 <= linked_int < len(all_notes) and linked_int not in seen:
                    selected_indices.append(linked_int)
                    seen.add(linked_int)

    memories = []
    for rank, idx in enumerate(selected_indices, start=1):
        memories.append(
            note_to_memory_item(
                all_notes[idx],
                sample_id=sample_id,
                source_speaker=source_speaker,
                user_id=user_id,
                rank=rank,
                score=scores.get(idx),
            )
        )
    return memories, time.time() - started


def trim_memories_to_token_budget(
    encoding: tiktoken.Encoding,
    speaker_1_memories: List[Dict[str, Any]],
    speaker_2_memories: List[Dict[str, Any]],
    token_budget: Optional[int],
    strategy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pre_1 = count_memory_tokens(encoding, speaker_1_memories)
    pre_2 = count_memory_tokens(encoding, speaker_2_memories)
    info = {
        "retrieval_budget_strategy": strategy,
        "retrieval_token_budget_applied": False,
        "speaker_1_memory_tokens_pre_trim": pre_1,
        "speaker_2_memory_tokens_pre_trim": pre_2,
        "retrieved_memory_tokens_pre_trim": pre_1 + pre_2,
        "num_speaker_1_memories_pre_trim": len(speaker_1_memories),
        "num_speaker_2_memories_pre_trim": len(speaker_2_memories),
        "num_retrieved_memories_pre_trim": len(speaker_1_memories) + len(speaker_2_memories),
        "num_retrieved_memories_dropped_by_budget": 0,
    }
    if token_budget is None or pre_1 + pre_2 <= token_budget:
        return speaker_1_memories, speaker_2_memories, info

    def ranked(memories: List[Dict[str, Any]], speaker_index: int) -> List[Dict[str, Any]]:
        items = []
        for rank, memory in enumerate(memories, start=1):
            items.append({
                "memory": memory,
                "rank": int(memory.get("rank") or rank),
                "speaker_index": speaker_index,
                "score": memory.get("score"),
            })
        return items

    combined = ranked(speaker_1_memories, 1) + ranked(speaker_2_memories, 2)
    if strategy == "global_score":
        combined.sort(key=lambda item: (-(item["score"] if item["score"] is not None else -1.0), item["rank"]))
    elif strategy == "balanced_per_speaker":
        combined.sort(key=lambda item: (item["rank"], item["speaker_index"]))
    else:
        combined.sort(key=lambda item: (item["rank"], item["speaker_index"]))

    kept = []
    total = 0
    for item in combined:
        candidate_tokens = count_memory_tokens(encoding, [item["memory"]])
        if kept and total + candidate_tokens > token_budget:
            continue
        if not kept and candidate_tokens > token_budget:
            kept.append(item)
            total += candidate_tokens
            break
        kept.append(item)
        total += candidate_tokens

    kept_1 = [item["memory"] for item in kept if item["speaker_index"] == 1]
    kept_2 = [item["memory"] for item in kept if item["speaker_index"] == 2]
    info["retrieval_token_budget_applied"] = True
    info["num_retrieved_memories_dropped_by_budget"] = len(combined) - len(kept)
    return kept_1, kept_2, info


def generate_keywords(
    llm: Any,
    parse_keywords_response,
    question: str,
    max_tokens: Optional[int],
) -> str:
    prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
    response = llm.get_completion(prompt, temperature=0.0, max_tokens=max_tokens)
    return parse_keywords_response(response)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_handle:
        checkpoint = json.load(file_handle)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version: {checkpoint.get('schema_version')}")
    return checkpoint


def write_checkpoint(
    path: Path,
    *,
    run_id: str,
    output_path: Path,
    completed_sample_ids: Iterable[str],
    state: Dict[str, Any],
) -> None:
    atomic_write_json(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "output_path": str(output_path),
            "completed_sample_ids": sorted(completed_sample_ids),
            "state": state,
        },
    )


def run() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be greater than 0")
    if args.api_max_retries < 0:
        raise ValueError("--api_max_retries must be >= 0")
    if args.api_timeout_sec <= 0:
        raise ValueError("--api_timeout_sec must be greater than 0")
    for token_arg in (
        "candidate_extraction_max_tokens",
        "update_decision_max_tokens",
        "answer_max_tokens",
        "summary_max_tokens",
    ):
        value = getattr(args, token_arg)
        if value is not None and value <= 0:
            raise ValueError(f"--{token_arg} must be greater than 0")

    root_dir = load_root_env()
    mem0_root = Path(__file__).resolve().parents[1]
    framework_root = Path(__file__).resolve().parents[2]
    dataset_path = normalize_dataset_path(args.dataset, mem0_root)
    amem_src = Path(args.amem_src) if args.amem_src else framework_root / "A-mem-paper-repro-src"
    if not amem_src.exists():
        raise FileNotFoundError(f"A-MEM source folder not found: {amem_src}")

    args.llm_model = args.llm_model or get_env_value("MODEL") or "deepseek-v4-flash"
    RobustAgenticMemorySystem, parse_keywords_response = load_amem_classes(amem_src)

    if args.resume_checkpoint:
        checkpoint = load_checkpoint(Path(args.resume_checkpoint))
        run_id = checkpoint["run_id"]
        output_path = Path(args.output) if args.output else Path(checkpoint["output_path"])
        checkpoint_path = Path(args.resume_checkpoint)
        state = checkpoint.get("state") or {}
        completed_sample_ids = set(checkpoint.get("completed_sample_ids") or [])
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(args.output)
            if args.output
            else mem0_root / "evaluation" / "results" / f"{args.method}_{run_id}.json"
        )
        checkpoint_path = (
            Path(args.checkpoint_path)
            if args.checkpoint_path
            else output_path.with_name(f"{output_path.name}.checkpoint.json")
        )
        state = {}
        completed_sample_ids = set()
    memory_cache_dir = (
        Path(args.memory_cache_dir)
        if args.memory_cache_dir
        else output_path.with_name(f"{output_path.stem}.amem_cache")
    )

    with open(dataset_path, "r", encoding="utf-8") as file_handle:
        dataset = json.load(file_handle)
    if args.max_conversations is not None:
        dataset = dataset[: args.max_conversations]

    if args.dry_run:
        print("Dry run")
        print("Dataset:", dataset_path)
        print("A-MEM src:", amem_src)
        print("Samples:", [sample.get("sample_id") for sample in dataset])
        print("Defaults:", {
            "namespace_mode": args.namespace_mode,
            "query_mode": args.query_mode,
            "retrieval_mode": args.retrieval_mode,
            "content_mode": args.content_mode,
            "answer_prompt_mode": args.answer_prompt_mode,
            "top_k": args.top_k,
            "candidate_extraction_max_tokens": args.candidate_extraction_max_tokens,
            "update_decision_max_tokens": args.update_decision_max_tokens,
            "answer_max_tokens": args.answer_max_tokens,
            "summary_max_tokens": args.summary_max_tokens,
        })
        print("Output:", output_path)
        print("Checkpoint:", checkpoint_path)
        print("Memory cache dir:", memory_cache_dir)
        return

    chat_client, chat_client_config = create_chat_client(args.api_max_retries, args.api_timeout_sec)
    token_encoding = tiktoken.get_encoding(args.retrieved_memory_token_encoding)

    results = state.get("results", [])
    add_logs = state.get("add_logs", [])
    add_event_counts = state.get("add_event_counts", empty_event_counts())
    memory_store_tokens_by_sample = state.get("memory_store_tokens_by_sample", {})
    final_num_memories_by_sample = state.get("final_num_memories_by_sample", {})
    sample_progress = state.get("sample_progress", {})

    print("Run ID:", run_id)
    print("Dataset:", dataset_path)
    print("A-MEM src:", amem_src)
    print("Output:", output_path)
    print("Checkpoint:", checkpoint_path)
    print("Memory cache dir:", memory_cache_dir)
    print("llm_model:", args.llm_model)
    print("embedding_model:", args.embedding_model)
    print("namespace_mode:", args.namespace_mode)
    print("retrieval_mode:", args.retrieval_mode)
    print("query_mode:", args.query_mode)
    print("content_mode:", args.content_mode)
    print("answer_prompt_mode:", args.answer_prompt_mode)
    print("top_k:", args.top_k)
    print("base_retrieve_k:", base_retrieve_k(args))
    print("candidate_extraction_max_tokens:", args.candidate_extraction_max_tokens)
    print("update_decision_max_tokens:", args.update_decision_max_tokens)
    print("answer_max_tokens:", args.answer_max_tokens)
    print("summary_max_tokens:", args.summary_max_tokens, "(accepted for compatibility; native A-MEM has no summary phase)")
    print("skip_category_5:", args.skip_category_5)

    def current_state() -> Dict[str, Any]:
        return {
            "results": results,
            "add_logs": add_logs,
            "add_event_counts": add_event_counts,
            "memory_store_tokens_by_sample": memory_store_tokens_by_sample,
            "final_num_memories_by_sample": final_num_memories_by_sample,
            "sample_progress": sample_progress,
        }

    for conv_idx, sample in enumerate(dataset):
        display_idx = conv_idx + 1
        sample_id = sample.get("sample_id", f"sample_{display_idx}")
        if sample_id in completed_sample_ids:
            print(f"[{display_idx}/{len(dataset)}] Skipping completed sample: {sample_id}")
            continue

        conversation = sample["conversation"]
        dia_lookup = build_dia_lookup(conversation)
        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")
        if not speaker_a or not speaker_b:
            raise ValueError(f"Missing speaker_a/speaker_b for sample {sample_id}")

        speaker_user_ids = {
            speaker_a: make_user_id(sample_id, conv_idx, speaker_a, run_id),
            speaker_b: make_user_id(sample_id, conv_idx, speaker_b, run_id),
        }

        print(f"[{display_idx}/{len(dataset)}] Sample: {sample_id}")
        print(f"Speakers: {speaker_a}, {speaker_b}")

        if args.namespace_mode == "per_speaker":
            systems = {
                speaker_a: create_amem_system(
                    RobustAgenticMemorySystem,
                    embedding_model=args.embedding_model,
                    llm_model=args.llm_model,
                    api_max_retries=args.api_max_retries,
                    api_timeout_sec=args.api_timeout_sec,
                    candidate_extraction_max_tokens=args.candidate_extraction_max_tokens,
                    update_decision_max_tokens=args.update_decision_max_tokens,
                ),
                speaker_b: create_amem_system(
                    RobustAgenticMemorySystem,
                    embedding_model=args.embedding_model,
                    llm_model=args.llm_model,
                    api_max_retries=args.api_max_retries,
                    api_timeout_sec=args.api_timeout_sec,
                    candidate_extraction_max_tokens=args.candidate_extraction_max_tokens,
                    update_decision_max_tokens=args.update_decision_max_tokens,
                ),
            }
        else:
            systems = {
                "global": create_amem_system(
                    RobustAgenticMemorySystem,
                    embedding_model=args.embedding_model,
                    llm_model=args.llm_model,
                    api_max_retries=args.api_max_retries,
                    api_timeout_sec=args.api_timeout_sec,
                    candidate_extraction_max_tokens=args.candidate_extraction_max_tokens,
                    update_decision_max_tokens=args.update_decision_max_tokens,
                )
            }

        session_indices = get_session_indices(conversation)
        if args.max_sessions is not None:
            session_indices = session_indices[: args.max_sessions]

        progress = sample_progress.setdefault(sample_id, {"completed_session_indices": []})
        completed_session_indices = {
            int(value) for value in progress.get("completed_session_indices", []) if value is not None
        }
        cache_files_present = all(
            (sample_cache_dir(memory_cache_dir, sample_id, namespace) / "memories.pkl").exists()
            for namespace in systems
        )
        if args.rebuild_memory_cache:
            completed_session_indices = set()
            progress["completed_session_indices"] = []
        elif completed_session_indices or cache_files_present:
            loaded_all = True
            for namespace, system in systems.items():
                loaded_all = (
                    load_system_cache(
                        system,
                        sample_cache_dir(memory_cache_dir, sample_id, namespace),
                        args.embedding_model,
                    )
                    and loaded_all
                )
            if loaded_all:
                if not completed_session_indices:
                    completed_session_indices = set(session_indices)
                    progress["completed_session_indices"] = sorted(completed_session_indices)
                print(f"Loaded cached A-MEM state for {sample_id}; completed sessions: {sorted(completed_session_indices)}")
            else:
                print(f"Cache incomplete for {sample_id}; rebuilding memory for this sample.")
                completed_session_indices = set()
                progress["completed_session_indices"] = []

        for session_index in session_indices:
            session_key = f"session_{session_index}"
            session_date = conversation.get(f"{session_key}_date_time")
            session = conversation.get(session_key) or []
            if not session:
                continue
            if session_index in completed_session_indices:
                print(f"Skipping cached {session_key}")
                continue

            print(f"Adding {session_key}: {len(session)} turns | timestamp={session_date}")
            for turn_idx, turn in enumerate(session):
                speaker = turn.get("speaker")
                content = build_add_content(turn, session_index, session_date, args.content_mode)
                target_keys = [speaker] if args.namespace_mode == "per_speaker" else ["global"]
                for target_key in target_keys:
                    add_start = time.time()
                    note_id = systems[target_key].add_note(content, time=session_date)
                    add_latency = time.time() - add_start
                    update_event_counts(add_event_counts, "ADD")
                    add_logs.append(
                        {
                            "sample_id": sample_id,
                            "session": session_key,
                            "session_index": session_index,
                            "session_date": session_date,
                            "turn_index": turn_idx,
                            "speaker": speaker,
                            "namespace": target_key,
                            "num_turns": 1,
                            "num_messages_added": 1,
                            "num_target_messages": 1,
                            "num_memories": 1,
                            "latency_sec": add_latency,
                            "add_result": {"results": [{"id": note_id, "event": "ADD"}]},
                        }
                    )
            print(f"  added turns through {session_key}")
            for namespace, system in systems.items():
                save_system_cache(system, sample_cache_dir(memory_cache_dir, sample_id, namespace))
            completed_session_indices.add(session_index)
            progress["completed_session_indices"] = sorted(completed_session_indices)
            write_checkpoint(
                checkpoint_path,
                run_id=run_id,
                output_path=output_path,
                completed_sample_ids=completed_sample_ids,
                state=current_state(),
            )
            print(f"Checkpoint saved after {sample_id}/{session_key}: {checkpoint_path}")

        sample_token_info = {}
        sample_count_info = {}
        if args.namespace_mode == "per_speaker":
            for speaker in (speaker_a, speaker_b):
                notes = list(systems[speaker].memories.values())
                items = [
                    note_to_memory_item(
                        note,
                        sample_id=sample_id,
                        source_speaker=speaker,
                        user_id=speaker_user_ids[speaker],
                    )
                    for note in notes
                ]
                sample_token_info[speaker] = count_memory_tokens(token_encoding, items)
                sample_count_info[speaker] = len(items)
        else:
            notes = list(systems["global"].memories.values())
            items = [
                note_to_memory_item(
                    note,
                    sample_id=sample_id,
                    source_speaker="global",
                    user_id=f"locomo_{slugify(sample_id)}_{conv_idx}_global_{run_id}",
                )
                for note in notes
            ]
            sample_token_info["global"] = count_memory_tokens(token_encoding, items)
            sample_count_info["global"] = len(items)
        sample_token_info["total"] = sum(v for v in sample_token_info.values() if isinstance(v, int))
        sample_token_info["token_encoding"] = args.retrieved_memory_token_encoding
        sample_count_info["total"] = sum(v for v in sample_count_info.values() if isinstance(v, int))
        memory_store_tokens_by_sample[sample_id] = sample_token_info
        final_num_memories_by_sample[sample_id] = sample_count_info

        qa_counter_for_sample = 0
        answered_keys = {
            (str(item.get("sample_id")), int(item.get("qa_index") or -1))
            for item in results
            if item.get("sample_id") is not None
        }
        for qa_idx, qa in enumerate(sample.get("qa", []), start=1):
            category = str(qa.get("category", "unknown"))
            if args.skip_category_5 and category == "5":
                continue

            qa_counter_for_sample += 1
            if (str(sample_id), int(qa_idx)) in answered_keys:
                continue
            if args.max_qa is not None and qa_counter_for_sample > args.max_qa:
                break

            question = qa.get("question", "")
            ground_truth = str(qa.get("adversarial_answer") if category == "5" else qa.get("answer", ""))
            evidence = qa.get("evidence", []) or []
            evidence_texts = get_evidence_texts(evidence, dia_lookup)

            print(f"QA {qa_counter_for_sample}: {question}")

            query = question
            query_latency = 0.0
            if args.query_mode == "keywords":
                keyword_start = time.time()
                any_system = next(iter(systems.values()))
                query = generate_keywords(
                    any_system.llm_controller.llm,
                    parse_keywords_response,
                    question,
                    max_tokens=args.candidate_extraction_max_tokens,
                )
                query_latency = time.time() - keyword_start

            if args.namespace_mode == "per_speaker":
                speaker_1_memories, speaker_1_memory_time = retrieve_amem_memories(
                    systems[speaker_a],
                    query=query,
                    top_k=args.top_k,
                    retrieval_mode=args.retrieval_mode,
                    sample_id=sample_id,
                    source_speaker=speaker_a,
                    user_id=speaker_user_ids[speaker_a],
                )
                speaker_2_memories, speaker_2_memory_time = retrieve_amem_memories(
                    systems[speaker_b],
                    query=query,
                    top_k=args.top_k,
                    retrieval_mode=args.retrieval_mode,
                    sample_id=sample_id,
                    source_speaker=speaker_b,
                    user_id=speaker_user_ids[speaker_b],
                )
                answer_speaker_1 = speaker_a
                answer_speaker_2 = speaker_b
            else:
                global_user_id = f"locomo_{slugify(sample_id)}_{conv_idx}_global_{run_id}"
                global_memories, global_time = retrieve_amem_memories(
                    systems["global"],
                    query=query,
                    top_k=args.top_k,
                    retrieval_mode=args.retrieval_mode,
                    sample_id=sample_id,
                    source_speaker="global",
                    user_id=global_user_id,
                )
                speaker_1_memories = global_memories
                speaker_2_memories = []
                speaker_1_memory_time = global_time
                speaker_2_memory_time = 0.0
                answer_speaker_1 = f"{speaker_a}/{speaker_b}"
                answer_speaker_2 = "empty"

            speaker_1_memories, speaker_2_memories, retrieval_budget_info = trim_memories_to_token_budget(
                token_encoding,
                speaker_1_memories,
                speaker_2_memories,
                args.retrieval_token_budget,
                args.retrieval_budget_strategy,
            )
            speaker_1_memory_tokens = count_memory_tokens(token_encoding, speaker_1_memories)
            speaker_2_memory_tokens = count_memory_tokens(token_encoding, speaker_2_memories)
            retrieved_memory_tokens = speaker_1_memory_tokens + speaker_2_memory_tokens
            retrieved_memories = [
                *speaker_1_memories,
                *speaker_2_memories,
            ]

            gen_start = time.time()
            if args.answer_prompt_mode == "amem_official":
                predicted_answer, usage, answer_prompt = answer_question_amem_official(
                    chat_client,
                    llm_model=args.llm_model,
                    question=question,
                    category=category,
                    ground_truth=ground_truth,
                    retrieved_memories=retrieved_memories,
                    answer_max_tokens=args.answer_max_tokens,
                )
            else:
                predicted_answer, usage, answer_prompt = answer_question(
                    chat_client,
                    llm_model=args.llm_model,
                    question=question,
                    speaker_1_name=answer_speaker_1,
                    speaker_2_name=answer_speaker_2,
                    speaker_1_memories=speaker_1_memories,
                    speaker_2_memories=speaker_2_memories,
                    answer_max_tokens=args.answer_max_tokens,
                )
            generation_latency = time.time() - gen_start
            search_latency = speaker_1_memory_time + speaker_2_memory_time + query_latency

            results.append(
                {
                    "sample_id": sample_id,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "speaker_1_user_id": speaker_user_ids.get(speaker_a),
                    "speaker_2_user_id": speaker_user_ids.get(speaker_b),
                    "qa_index": qa_idx,
                    "qa_index_non_adversarial": qa_counter_for_sample,
                    "category": category,
                    "question": question,
                    "ground_truth": ground_truth,
                    "answer": ground_truth,
                    "predicted_answer": predicted_answer,
                    "response": predicted_answer,
                    "evidence": evidence,
                    "evidence_texts": evidence_texts,
                    "speaker_1_memories": speaker_1_memories,
                    "speaker_2_memories": speaker_2_memories,
                    "retrieved_memories": retrieved_memories,
                    "memory_update_mode": "amem_robust_evolution",
                    "memory_wrapper_store": "amem_simple_embedding",
                    "namespace_mode": args.namespace_mode,
                    "retrieval_mode": args.retrieval_mode,
                    "query_mode": args.query_mode,
                    "content_mode": args.content_mode,
                    "answer_prompt_mode": args.answer_prompt_mode,
                    "retrieval_query": query,
                    "num_speaker_1_memories": len(speaker_1_memories),
                    "num_speaker_2_memories": len(speaker_2_memories),
                    "top_k_per_speaker": args.top_k if args.namespace_mode == "per_speaker" else None,
                    "max_total_retrieved_memories": base_retrieve_k(args),
                    "base_retrieve_k": base_retrieve_k(args),
                    "linked_neighbors_can_expand_retrieval": args.retrieval_mode == "linked_neighbors",
                    "retrieval_token_budget": args.retrieval_token_budget,
                    **retrieval_budget_info,
                    "speaker_1_memory_tokens": speaker_1_memory_tokens,
                    "speaker_2_memory_tokens": speaker_2_memory_tokens,
                    "retrieved_memory_tokens": retrieved_memory_tokens,
                    "retrieved_memory_token_encoding": args.retrieved_memory_token_encoding,
                    "speaker_1_memory_time": speaker_1_memory_time,
                    "speaker_2_memory_time": speaker_2_memory_time,
                    "query_generation_latency_sec": query_latency,
                    "search_latency_sec": search_latency,
                    "generation_latency_sec": generation_latency,
                    "total_latency_sec": search_latency + generation_latency,
                    "usage": usage,
                    "answer_prompt": answer_prompt,
                }
            )
            answered_keys.add((str(sample_id), int(qa_idx)))

            print(f"  pred={predicted_answer}")
            print(
                f"  search={search_latency:.2f}s "
                f"({speaker_a}={speaker_1_memory_time:.2f}s, {speaker_b}={speaker_2_memory_time:.2f}s, query={query_latency:.2f}s) | "
                f"gen={generation_latency:.2f}s | answer_tokens={usage.get('total_tokens')} | "
                f"memory_tokens={retrieved_memory_tokens}"
            )
            write_checkpoint(
                checkpoint_path,
                run_id=run_id,
                output_path=output_path,
                completed_sample_ids=completed_sample_ids,
                state=current_state(),
            )

        completed_sample_ids.add(sample_id)
        write_checkpoint(
            checkpoint_path,
            run_id=run_id,
            output_path=output_path,
            completed_sample_ids=completed_sample_ids,
            state=current_state(),
        )
        print(f"Checkpoint saved after {sample_id}: {checkpoint_path}")

    output = {
        "run_id": run_id,
        "method": args.method,
        "benchmark_protocol": "locomo_amem_native_fair_eval",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "resumed_from_checkpoint": str(Path(args.resume_checkpoint).resolve()) if args.resume_checkpoint else None,
        "completed_sample_ids": sorted(completed_sample_ids),
        "dataset": "locomo10",
        "dataset_path": str(dataset_path),
        "amem_src": str(amem_src),
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "model_env_var": "MODEL",
        "chat_client": chat_client_config,
        "memory_update_mode": "amem_robust_evolution",
        "memory_wrapper_store": "amem_simple_embedding",
        "namespace_mode": args.namespace_mode,
        "retrieval_mode": args.retrieval_mode,
        "query_mode": args.query_mode,
        "content_mode": args.content_mode,
        "answer_prompt_mode": args.answer_prompt_mode,
        "top_k": args.top_k,
        "top_k_per_speaker": args.top_k if args.namespace_mode == "per_speaker" else None,
        "base_retrieve_k": base_retrieve_k(args),
        "max_total_retrieved_memories": base_retrieve_k(args),
        "linked_neighbors_can_expand_retrieval": args.retrieval_mode == "linked_neighbors",
        "search_threshold": args.search_threshold,
        "retrieved_memory_token_encoding": args.retrieved_memory_token_encoding,
        "retrieval_token_budget": args.retrieval_token_budget,
        "retrieval_budget_strategy": args.retrieval_budget_strategy,
        "candidate_extraction_max_tokens": args.candidate_extraction_max_tokens,
        "update_decision_max_tokens": args.update_decision_max_tokens,
        "answer_max_tokens": args.answer_max_tokens,
        "summary_max_tokens": args.summary_max_tokens,
        "summary_max_tokens_note": (
            "Accepted for Mem0 command compatibility only; native A-MEM has no rolling-summary phase."
            if args.summary_max_tokens is not None
            else None
        ),
        "max_conversations": args.max_conversations,
        "max_sessions": args.max_sessions,
        "max_qa": args.max_qa,
        "skip_category_5": args.skip_category_5,
        "add_logs": add_logs,
        "add_event_counts": add_event_counts,
        "memory_add_output_tokens_by_sample": memory_store_tokens_by_sample,
        "memory_store_tokens_by_sample": memory_store_tokens_by_sample,
        "final_memory_store_tokens_by_sample": memory_store_tokens_by_sample,
        "final_num_memories_by_sample": final_num_memories_by_sample,
        "final_store_fetch_supported": True,
        "results": results,
        "limitations_note": (
            "This is an A-MEM robust-layer run adapted to the Mem0 local LOCOMO comparison schema. "
            "Defaults preserve A-MEM-native architecture where it matters: global memory, keyword query generation, "
            "linked-neighbor retrieval, and official A-MEM turn ingestion format. Fairness is applied to shared dataset "
            "subset, answer model, judge model/prompt, embedding model when configured, and output schema. "
            "Remaining paper deviations should be reported explicitly: the robust layer uses plain-text section prompts "
            "instead of the paper/original JSON-schema structured-output prompts; default answer_prompt_mode=amem_official "
            "uses the A-MEM repository's category-aware answer prompts; default skip_category_5=True matches the existing "
            "Mem0 5-conversation run but excludes the paper's adversarial category; top_k defaults to a fair 30-total "
            "budget for DeepSeek because the paper's Table 8 does not specify DeepSeek-v4-flash. "
            "Use --namespace_mode per_speaker --query_mode question --retrieval_mode strict_top_k --content_mode timestamp_context "
            "or --answer_prompt_mode shared_mem0 only as Mem0-normalized ablations, not as the primary A-MEM comparison."
        ),
    }
    atomic_write_json(output_path, output)
    write_checkpoint(
        checkpoint_path,
        run_id=run_id,
        output_path=output_path,
        completed_sample_ids=completed_sample_ids,
        state=current_state(),
    )
    print("\nSaved result JSON:", output_path.resolve())
    print("Total QA answered:", len(results))
    print("Total add operations:", len(add_logs))


if __name__ == "__main__":
    run()
