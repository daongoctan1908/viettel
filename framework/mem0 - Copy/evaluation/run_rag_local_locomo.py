import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI


RAG_PROMPT = """# Context:
{context}

# Question:
{question}

# Instructions:
Answer using only the context. Pay attention to timestamps when resolving relative dates.
Return the shortest possible answer. If the answer is not in the context, answer exactly:
No information available.

# Short answer:
"""


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


def create_embedding_client(api_max_retries: int, api_timeout_sec: float) -> Tuple[OpenAI, Dict[str, Any]]:
    api_key = get_env_value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
    return OpenAI(api_key=api_key, max_retries=api_max_retries, timeout=api_timeout_sec), {
        "provider": "openai",
        "uses_openai_api_key": True,
        "base_url_configured": False,
        "max_retries": api_max_retries,
        "timeout_sec": api_timeout_sec,
    }


def api_call_with_retries(fn, *, max_attempts: int, retry_label: str):
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            sleep_sec = min(2.0 * attempt, 20.0)
            print(f"{retry_label} failed (attempt {attempt}/{max_attempts}): {exc} -- retrying in {sleep_sec:.1f}s")
            time.sleep(sleep_sec)
    raise last_exc


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def get_session_indices(conversation: Dict[str, Any]) -> List[int]:
    indices = []
    for key, value in conversation.items():
        if key.startswith("session_") and key.count("_") == 1 and isinstance(value, list):
            try:
                indices.append(int(key.split("_")[1]))
            except ValueError:
                pass
    return sorted(indices)


def dialogue_turn_text(turn: Dict[str, Any], session_index: int, session_date: Optional[str]) -> str:
    speaker = turn.get("speaker", "Unknown")
    dia_id = turn.get("dia_id") or f"D{session_index}:?"
    text = str(turn.get("text") or "").strip()
    extras = []
    if turn.get("blip_caption"):
        extras.append(f"Image caption: {turn['blip_caption']}")
    if turn.get("query"):
        extras.append(f"Image query: {turn['query']}")
    extra_text = ("\n" + "\n".join(extras)) if extras else ""
    return (
        f"[Session {session_index}; timestamp: {session_date}; dialogue_id: {dia_id}] "
        f"{speaker}: {text}{extra_text}"
    )


def build_dialogue_document(sample: Dict[str, Any], max_sessions: Optional[int]) -> str:
    conversation = sample.get("conversation") or {}
    lines = []
    session_indices = get_session_indices(conversation)
    if max_sessions is not None:
        session_indices = session_indices[:max_sessions]
    for session_index in session_indices:
        session_key = f"session_{session_index}"
        session_date = conversation.get(f"{session_key}_date_time")
        for turn in conversation.get(session_key) or []:
            lines.append(dialogue_turn_text(turn, session_index, session_date))
    return "\n".join(lines)


def chunk_text(
    text: str,
    *,
    encoding: tiktoken.Encoding,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("--chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("--chunk_overlap must be >= 0 and < chunk_size")
    tokens = encoding.encode(text)
    chunks = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        if not chunk_tokens:
            break
        chunks.append(encoding.decode(chunk_tokens))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def embed_text(
    embedding_client: OpenAI,
    *,
    embedding_model: str,
    text: str,
    api_max_retries: int,
) -> List[float]:
    response = api_call_with_retries(
        lambda: embedding_client.embeddings.create(model=embedding_model, input=text),
        max_attempts=api_max_retries,
        retry_label="Embedding call",
    )
    return response.data[0].embedding


def cosine_scores(query_embedding: List[float], embeddings: List[List[float]]) -> np.ndarray:
    query = np.asarray(query_embedding, dtype=np.float32)
    matrix = np.asarray(embeddings, dtype=np.float32)
    query_norm = np.linalg.norm(query) + 1e-12
    matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-12
    return matrix @ query / (matrix_norm * query_norm)


def answer_question(
    chat_client: OpenAI,
    *,
    llm_model: str,
    question: str,
    context: str,
    answer_max_tokens: int,
    api_max_retries: int,
) -> Tuple[str, Dict[str, Any], str]:
    prompt = RAG_PROMPT.format(context=context, question=question)
    response = api_call_with_retries(
        lambda: chat_client.chat.completions.create(
            model=llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise QA assistant. Use only the provided context. "
                        "Use words directly from the context when possible."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=answer_max_tokens,
        ),
        max_attempts=api_max_retries,
        retry_label="Answer call",
    )
    answer = (response.choices[0].message.content or "").strip() or "No information available."
    usage = response.usage.model_dump() if response.usage else {}
    return answer, usage, prompt


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def completed_key(sample_id: str, qa_index: int) -> str:
    return f"{sample_id}:qa_{qa_index}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plain RAG baseline for LOCOMO raw dialogue.")
    parser.add_argument("--dataset", default="evaluation/dataset/locomo10.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--method", default="rag_chunk_top1_500")
    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_conversations", type=int, default=5)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--skip_category_5", action="store_true", default=False)
    parser.add_argument("--answer_max_tokens", type=int, default=2000)
    parser.add_argument("--token_encoding", default="cl100k_base")
    parser.add_argument("--api_max_retries", type=int, default=10)
    parser.add_argument("--api_timeout_sec", type=float, default=600.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    load_dotenv(root_dir / ".env", override=True)
    args.llm_model = args.llm_model or get_env_value("MODEL") or "gpt-4o-mini"
    args.embedding_model = args.embedding_model or get_env_value("EMBEDDING_MODEL") or "text-embedding-3-small"

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        dataset_path = root_dir / args.dataset
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else None
    cache_path = Path(args.cache_path) if args.cache_path else output_path.with_name(f"{output_path.stem}.rag_cache.json")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.max_conversations is not None:
        dataset = dataset[: args.max_conversations]

    encoding = tiktoken.get_encoding(args.token_encoding)
    chat_client, chat_client_config = create_chat_client(args.api_max_retries, args.api_timeout_sec)
    embedding_client, embedding_client_config = create_embedding_client(args.api_max_retries, args.api_timeout_sec)

    checkpoint = load_json(checkpoint_path)
    cache = load_json(cache_path)
    run_id = checkpoint.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results = checkpoint.get("results") or []
    done = set(checkpoint.get("completed_qa_keys") or [])
    total_answered = len(results)

    for conv_idx, sample in enumerate(dataset, start=1):
        sample_id = sample.get("sample_id") or f"conv-{conv_idx}"
        conversation = sample.get("conversation") or {}
        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")
        print(f"\n=== {sample_id}: RAG chunks ===")

        cache_key = f"{sample_id}|chunk={args.chunk_size}|overlap={args.chunk_overlap}|sessions={args.max_sessions}|emb={args.embedding_model}"
        sample_cache = cache.get(cache_key)
        if not sample_cache:
            doc = build_dialogue_document(sample, args.max_sessions)
            chunks = chunk_text(
                doc,
                encoding=encoding,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
            )
            embeddings = []
            for chunk_index, chunk in enumerate(chunks, start=1):
                embeddings.append(
                    embed_text(
                        embedding_client,
                        embedding_model=args.embedding_model,
                        text=chunk,
                        api_max_retries=args.api_max_retries,
                    )
                )
                if chunk_index % 10 == 0 or chunk_index == len(chunks):
                    print(f"Embedded {chunk_index}/{len(chunks)} chunks for {sample_id}")
            sample_cache = {"chunks": chunks, "embeddings": embeddings}
            cache[cache_key] = sample_cache
            atomic_write_json(cache_path, cache)
        else:
            chunks = sample_cache["chunks"]
            embeddings = sample_cache["embeddings"]
            print(f"Loaded cached chunks: {len(chunks)}")

        qa_items = sample.get("qa") or []
        for qa_idx, qa in enumerate(qa_items, start=1):
            if args.max_questions is not None and total_answered >= args.max_questions:
                break
            category = str(qa.get("category", ""))
            if args.skip_category_5 and category == "5":
                continue
            key = completed_key(sample_id, qa_idx)
            if key in done:
                continue

            question = str(qa.get("question", ""))
            ground_truth = str(qa.get("answer", ""))
            evidence = qa.get("evidence") or []

            search_start = time.time()
            query_embedding = embed_text(
                embedding_client,
                embedding_model=args.embedding_model,
                text=question,
                api_max_retries=args.api_max_retries,
            )
            scores = cosine_scores(query_embedding, embeddings)
            top_indices = np.argsort(scores)[-args.top_k :][::-1].tolist()
            retrieved = [
                {
                    "id": f"{sample_id}:chunk:{idx}",
                    "memory": chunks[idx],
                    "score": float(scores[idx]),
                    "metadata": {"source": "rag_chunk", "chunk_index": idx},
                }
                for idx in top_indices
            ]
            context = "\n<->\n".join(item["memory"] for item in retrieved)
            search_latency = time.time() - search_start
            retrieved_tokens = sum(len(encoding.encode(item["memory"])) for item in retrieved)

            gen_start = time.time()
            predicted_answer, usage, answer_prompt = answer_question(
                chat_client,
                llm_model=args.llm_model,
                question=question,
                context=context,
                answer_max_tokens=args.answer_max_tokens,
                api_max_retries=args.api_max_retries,
            )
            generation_latency = time.time() - gen_start
            total_latency = search_latency + generation_latency

            result = {
                "sample_id": sample_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "speaker_1_user_id": speaker_a,
                "speaker_2_user_id": speaker_b,
                "qa_index": qa_idx,
                "qa_index_non_adversarial": qa_idx,
                "category": category,
                "question": question,
                "ground_truth": ground_truth,
                "answer": ground_truth,
                "predicted_answer": predicted_answer,
                "response": predicted_answer,
                "evidence": evidence,
                "speaker_1_memories": retrieved,
                "speaker_2_memories": [],
                "retrieved_memories": retrieved,
                "memory_update_mode": "none",
                "memory_wrapper_store": "none",
                "qa_retrieval_mode": "rag_chunk",
                "num_speaker_1_memories": len(retrieved),
                "num_speaker_2_memories": 0,
                "top_k_per_speaker": None,
                "max_total_retrieved_memories": args.top_k,
                "retrieval_token_budget": None,
                "retrieval_budget_strategy": "global_chunk_cosine",
                "speaker_1_memory_tokens": retrieved_tokens,
                "speaker_2_memory_tokens": 0,
                "retrieved_memory_tokens": retrieved_tokens,
                "speaker_1_memory_tokens_pre_trim": retrieved_tokens,
                "speaker_2_memory_tokens_pre_trim": 0,
                "retrieved_memory_tokens_pre_trim": retrieved_tokens,
                "num_retrieved_memories_pre_trim": len(retrieved),
                "search_latency_sec": search_latency,
                "generation_latency_sec": generation_latency,
                "total_latency_sec": total_latency,
                "answer_usage": usage,
                "answer_prompt": answer_prompt,
            }
            results.append(result)
            done.add(key)
            total_answered += 1
            print(
                f"QA {qa_idx}: {question}\n"
                f"  pred={predicted_answer}\n"
                f"  search={search_latency:.2f}s | gen={generation_latency:.2f}s | "
                f"chunks={len(retrieved)} | context_tokens={retrieved_tokens}"
            )

            if checkpoint_path:
                atomic_write_json(
                    checkpoint_path,
                    {
                        "run_id": run_id,
                        "method": args.method,
                        "output_path": str(output_path),
                        "completed_qa_keys": sorted(done),
                        "results": results,
                    },
                )
                print(f"Checkpoint saved after {sample_id}/qa_{qa_idx}: {checkpoint_path}")

        if args.max_questions is not None and total_answered >= args.max_questions:
            break

    payload = {
        "method": args.method,
        "run_id": run_id,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "memory_update_mode": "none",
        "memory_wrapper_store": "none",
        "qa_retrieval_mode": "rag_chunk",
        "input_file": str(dataset_path),
        "num_questions": len(results),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "max_conversations": args.max_conversations,
        "max_sessions": args.max_sessions,
        "skip_category_5": args.skip_category_5,
        "chat_client": chat_client_config,
        "embedding_client": embedding_client_config,
        "results": results,
    }
    atomic_write_json(output_path, payload)
    print(f"\nSaved result JSON: {output_path}")
    print(f"Total QA answered: {len(results)}")


if __name__ == "__main__":
    main()
