from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from openai import OpenAI
import tiktoken

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for path in (PROJECT_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mem0_question_aware_budget import QASEConfig, QuestionAwareSelectiveEvidence
from paper_update_memory import (
    LocalPaperMemoryStore,
    decide_memory_operation,
    extract_candidate_memories,
)
from run_mem0_local_locomo import (
    chat_completion_with_retries,
    count_text_tokens,
    format_memories_for_prompt,
    load_root_env,
    score_bm25_semantic_candidates,
    update_rolling_summary,
)


DEFAULT_CHECKPOINT: Optional[Path] = None
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_SESSION_ID = "personal-demo"
DEFAULT_USER_NAME = "User"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def memory_text(memory: Dict[str, Any]) -> str:
    return str(memory.get("memory") or memory.get("text") or memory.get("content") or "")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_round(value: Any, digits: int = 3) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def now_observation() -> Tuple[str, str]:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M UTC"), now.isoformat().replace("+00:00", "Z")


def make_qase_controller(max_total_final_memories: int, lexical_weight: float) -> QuestionAwareSelectiveEvidence:
    return QuestionAwareSelectiveEvidence(
        QASEConfig(
            max_total_final_memories=max_total_final_memories,
            budget_allocation_mode="balanced_per_speaker",
            candidate_multi_hop_target=20,
            candidate_multi_hop_other=20,
            final_multi_hop_target=10,
            final_multi_hop_other=10,
            semantic_anchor_enabled=False,
            confidence_fallback_enabled=False,
            diversity_lambda_simple=0.0,
            diversity_lambda_complex=0.0,
            complex_min_fraction_of_cap=0.0,
            temporal_bonus_weight=0.12,
            adaptive_cutoff_mode="largest_gap",
            min_single_hop=2,
            min_temporal=3,
            min_multi_hop=4,
            min_fallback_ambiguous=3,
            lexical_weight=lexical_weight,
        )
    )


def create_demo_embedding_client(api_max_retries: int, api_timeout_sec: float) -> Tuple[OpenAI, Dict[str, Any]]:
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for live demo embeddings.")
    verify_ssl = (os.getenv("DEMO_OPENAI_VERIFY_SSL") or "0").strip().lower() in {"1", "true", "yes"}
    http_client = httpx.Client(verify=verify_ssl, timeout=api_timeout_sec)
    return OpenAI(
        api_key=api_key,
        max_retries=api_max_retries,
        timeout=api_timeout_sec,
        http_client=http_client,
    ), {
        "provider": "openai",
        "uses_openai_api_key": True,
        "base_url_configured": False,
        "max_retries": api_max_retries,
        "timeout_sec": api_timeout_sec,
        "ssl_verify": verify_ssl,
    }


def create_demo_chat_client(api_max_retries: int, api_timeout_sec: float) -> Tuple[OpenAI, Dict[str, Any]]:
    api_key = (os.getenv("BEEKNOEE_API_KEY") or "").strip()
    base_url = (os.getenv("BEEKNOEE_BASE_URL") or "").strip()
    if not api_key:
        raise RuntimeError("BEEKNOEE_API_KEY is required for live demo chat completions.")
    if not base_url:
        raise RuntimeError("BEEKNOEE_BASE_URL is required for live demo chat completions.")
    verify_ssl = (os.getenv("DEMO_BEEKNOEE_VERIFY_SSL") or "0").strip().lower() in {"1", "true", "yes"}
    http_client = httpx.Client(verify=verify_ssl, timeout=api_timeout_sec)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=api_max_retries,
        timeout=api_timeout_sec,
        http_client=http_client,
    ), {
        "provider": "beeknoee",
        "uses_beeknoee_api_key": True,
        "uses_openai_api_key_for_chat": False,
        "base_url_configured": True,
        "max_retries": api_max_retries,
        "timeout_sec": api_timeout_sec,
        "ssl_verify": verify_ssl,
    }


def make_fresh_wrapper_state(embedding_model: str, user_id: str) -> Dict[str, Any]:
    return {
        "embedding_model": embedding_model,
        "memories_by_user": {user_id: []},
        "embedding_cache": {},
        "embedding_recovery_log": [],
    }


def format_demo_memory_evidence(retrieved_memories: List[Dict[str, Any]]) -> str:
    lines = []
    for item in retrieved_memories:
        text = str(item.get("text") or "").strip()
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def format_benchmark_memory_context(retrieved_memories: List[Dict[str, Any]]) -> List[str]:
    return format_memories_for_prompt(retrieved_memories)


def answer_live_chat(
    chat_client: OpenAI,
    *,
    llm_model: str,
    user_message: str,
    profile_name: str,
    conversation_summary: str,
    retrieved_memories: List[Dict[str, Any]],
    answer_max_tokens: int,
) -> Tuple[str, Dict[str, Any], str]:
    memory_evidence = "\n".join(format_benchmark_memory_context(retrieved_memories))
    prompt = f"""
You are a live Mem0 QASE demo assistant.

Goal:
- Reply naturally to the user's latest message.
- Use retrieved memories as the primary stored evidence when recalling facts about the user.
- You may also use the current conversation summary as background memory for the live demo.
- If the user asks a factual question about themselves and the answer is not in retrieved memories or summary, say that the demo memory does not have enough information yet.
- If the user provides a new personal fact, acknowledge it normally. Do not claim the fact is missing just because it was not retrieved before this turn.
- Preserve exact spelling and casing from retrieved memories and the latest user message.
- Do not expand abbreviations or acronyms unless the user explicitly defines them.
- Keep the reply concise and conversational.
- Reply in English unless the user explicitly asks for another language.

User profile name:
{profile_name}

Current conversation summary:
{conversation_summary or "(none)"}

Retrieved memories:
{memory_evidence or "(none)"}

Latest user message:
{user_message}
""".strip()

    response = chat_completion_with_retries(
        chat_client,
        model=llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a memory-augmented chat assistant for a live demo. "
                    "Use supplied memory evidence and summary when recalling stored facts."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=answer_max_tokens,
    )
    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage.model_dump() if response.usage else {}
    if not answer:
        answer = "I have recorded that information."
        usage["empty_answer_fallback"] = 1
    else:
        usage["empty_answer_fallback"] = 0
    return answer, usage, prompt


class LiveDemoState:
    def __init__(
        self,
        *,
        checkpoint_path: Optional[Path],
        user_name: str,
        llm_model: str,
        embedding_model: str,
        answer_max_tokens: int,
        summary_max_tokens: int,
        extraction_max_tokens: int,
        decision_max_tokens: int,
        max_total_final_memories: int,
        lexical_weight: float,
        api_max_retries: int,
        api_timeout_sec: float,
    ):
        load_root_env()
        self.checkpoint_path = checkpoint_path
        self.user_name = user_name
        self.user_id = "demo_personal_user"
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.answer_max_tokens = answer_max_tokens
        self.summary_max_tokens = summary_max_tokens
        self.extraction_max_tokens = extraction_max_tokens
        self.decision_max_tokens = decision_max_tokens
        self.max_total_final_memories = max_total_final_memories
        self.lexical_weight = lexical_weight
        self.api_max_retries = api_max_retries
        self.api_timeout_sec = api_timeout_sec
        self.lock = threading.Lock()

        self.chat_client, self.chat_info = create_demo_chat_client(api_max_retries, api_timeout_sec)
        self.embedding_client, self.embedding_info = create_demo_embedding_client(api_max_retries, api_timeout_sec)
        self.embedding_client_factory = lambda: create_demo_embedding_client(api_max_retries, api_timeout_sec)[0]
        self.qase = make_qase_controller(max_total_final_memories, lexical_weight)
        self.token_encoding = tiktoken.get_encoding("cl100k_base")

        if checkpoint_path:
            checkpoint = read_json(checkpoint_path)
            state = checkpoint.get("state") or {}
            self.initial_wrapper_state = deepcopy(state.get("wrapper_store_state") or {})
            if not (self.initial_wrapper_state.get("memories_by_user") or {}):
                self.initial_wrapper_state = make_fresh_wrapper_state(embedding_model, self.user_id)
        else:
            self.initial_wrapper_state = make_fresh_wrapper_state(embedding_model, self.user_id)
        self.samples = {
            DEFAULT_SESSION_ID: [
                {
                    "speaker": self.user_name,
                    "user_id": self.user_id,
                    "label": "Personal memory",
                }
            ]
        }
        self.sample_id = DEFAULT_SESSION_ID
        self.active_speaker = self.user_name
        self.reset_runtime()

    def _new_store(self) -> LocalPaperMemoryStore:
        store = LocalPaperMemoryStore(
            embedding_model=self.embedding_model,
            embedding_client=self.embedding_client,
            embedding_client_factory=self.embedding_client_factory,
            embedding_recovery_attempts=max(1, self.api_max_retries),
        )
        store.load_state(deepcopy(self.initial_wrapper_state))
        return store

    def reset_runtime(self) -> None:
        self.store = self._new_store()
        self.history: List[Dict[str, str]] = []
        self.summaries: Dict[str, str] = {speaker["speaker"]: "" for speaker in self.samples[self.sample_id]}
        self.last_result: Optional[Dict[str, Any]] = None
        self.pending_ingest: Optional[Dict[str, Any]] = None

    def public_state(self) -> Dict[str, Any]:
        speakers = self.samples[self.sample_id]
        return {
            "sample_id": self.sample_id,
            "session_label": "Personal memory",
            "samples": [
                {
                    "id": sample_id,
                    "label": "Personal memory",
                    "speakers": self.samples[sample_id],
                }
                for sample_id in sorted(self.samples)
            ],
            "speakers": speakers,
            "active_speaker": self.active_speaker,
            "model": self.llm_model,
            "embedding_model": self.embedding_model,
            "retrieval_mode": "QASE semantic + BM25 rerank",
            "memory_counts": {
                item["speaker"]: len(self.store.get_all(item["user_id"]))
                for item in speakers
            },
            "summary": self.summaries.get(self.active_speaker, ""),
            "history": self.history,
            "last_result": self.last_result,
        }

    def chat(self, message: str) -> Dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("Message is empty.")
        with self.lock:
            started = time.time()
            speakers = self.samples[self.sample_id]
            active = next(item for item in speakers if item["speaker"] == self.active_speaker)
            session_date, observed_at = now_observation()

            retrieval_started = time.time()
            plan = self.qase.plan(message, [item["speaker"] for item in speakers])
            candidates_by_speaker: Dict[str, List[Dict[str, Any]]] = {}
            search_logs = []
            semantic_latency = 0.0
            bm25_latency = 0.0
            for item in speakers:
                speaker = item["speaker"]
                user_id = item["user_id"]
                top_k = max(1, int(plan.candidate_top_k_by_speaker.get(speaker) or 10))
                search_start = time.time()
                candidates = self.store.search(user_id, message, top_k)
                semantic_latency += time.time() - search_start
                annotated, speaker_bm25_latency, bm25_log = score_bm25_semantic_candidates(
                    self.store,
                    question=message,
                    user_id=user_id,
                    speaker=speaker,
                    candidates=candidates,
                    k1=1.5,
                    b=0.75,
                )
                bm25_latency += speaker_bm25_latency
                candidates_by_speaker[speaker] = annotated
                search_logs.append({"speaker": speaker, "semantic_count": len(candidates), "bm25": bm25_log})

            selected_by_speaker, selection_log = self.qase.select_global(candidates_by_speaker, plan)
            retrieval_latency = time.time() - retrieval_started

            answer_started = time.time()
            selected_memories = self._flatten_selected(selected_by_speaker)
            formatted_memory_context = format_benchmark_memory_context(selected_memories)
            retrieved_memory_tokens = (
                count_text_tokens(self.token_encoding, formatted_memory_context)
                if formatted_memory_context
                else 0
            )
            summary_before = self.summaries.get(active["speaker"], "")
            answer, answer_usage, answer_prompt = answer_live_chat(
                self.chat_client,
                llm_model=self.llm_model,
                user_message=message,
                profile_name=active["speaker"],
                conversation_summary=summary_before,
                retrieved_memories=selected_memories,
                answer_max_tokens=self.answer_max_tokens,
            )
            answer_latency = time.time() - answer_started

            context_messages = self.history[-10:]

            self.history.append({"role": "user", "speaker": active["speaker"], "content": message})
            self.history.append({"role": "assistant", "speaker": "Mem0 QASE", "content": answer})

            result = {
                "question": message,
                "answer": answer,
                "active_speaker": active["speaker"],
                "sample_id": self.sample_id,
                "plan": plan.to_dict(),
                "selection": selection_log.to_dict(),
                "retrieved_memories": selected_memories,
                "summary_before": summary_before,
                "summary_after": summary_before,
                "summary_log": {},
                "extracted_candidates": [],
                "update_events": [],
                "update_pending": True,
                "search_logs": search_logs,
                "stats": {
                    "retrieval_latency_sec": safe_round(retrieval_latency),
                    "semantic_latency_sec": safe_round(semantic_latency),
                    "bm25_latency_sec": safe_round(bm25_latency),
                    "answer_latency_sec": safe_round(answer_latency),
                    "answer_total_latency_sec": safe_round(time.time() - started),
                    "memory_update_latency_sec": 0.0,
                    "total_latency_sec": safe_round(time.time() - started),
                    "retrieved_memory_tokens": retrieved_memory_tokens,
                    "retrieved_memory_token_encoding": "cl100k_base",
                    "retrieved_memory_token_scope": "benchmark_formatted_memory_context",
                    "answer_prompt_tokens": int(answer_usage.get("prompt_tokens") or 0),
                    "answer_completion_tokens": int(answer_usage.get("completion_tokens") or 0),
                    "answer_total_tokens": int(answer_usage.get("total_tokens") or 0),
                    "retrieved_count": len(selected_memories),
                    "candidate_count": selection_log.total_input_candidates,
                    "memory_counts": {
                        item["speaker"]: len(self.store.get_all(item["user_id"]))
                        for item in speakers
                    },
                },
                "answer_prompt_preview": answer_prompt[:1200],
            }
            self.pending_ingest = {
                "active": active,
                "speakers": deepcopy(speakers),
                "message": message,
                "answer": answer,
                "summary_before": summary_before,
                "context_messages": deepcopy(context_messages),
                "session_date": session_date,
                "observed_at": observed_at,
            }
            self.last_result = result
            return result

    def ingest_pending(self) -> Dict[str, Any]:
        with self.lock:
            if not self.pending_ingest:
                if self.last_result:
                    return self.last_result
                raise ValueError("No pending message to ingest.")
            pending = self.pending_ingest
            self.pending_ingest = None
            extraction_started = time.time()
            active = pending["active"]
            speakers = pending["speakers"]
            message = pending["message"]
            answer = pending["answer"]
            summary_before = pending["summary_before"]
            target_messages = [{"role": "user", "content": f"{active['speaker']}: {message}"}]

            candidates, extraction_log = extract_candidate_memories(
                self.chat_client,
                self.llm_model,
                summary_before,
                pending["context_messages"],
                target_messages,
                pending["session_date"],
                pending["observed_at"],
                active["speaker"],
                self.extraction_max_tokens,
            )
            update_events = []
            for candidate in candidates[:5]:
                similar = self.store.search(active["user_id"], str(candidate.get("memory") or ""), top_k=8)
                decision, decision_log = decide_memory_operation(
                    self.chat_client,
                    self.llm_model,
                    candidate,
                    similar,
                    self.decision_max_tokens,
                )
                operation = str(decision.get("operation") or "NOOP").upper()
                applied = False
                target_memory_id = decision.get("target_memory_id")
                candidate_memory_text = str(candidate.get("memory") or "")
                new_memory_text = str(decision.get("new_memory") or candidate_memory_text)
                target_memory_text = ""
                if target_memory_id:
                    for similar_memory in similar:
                        if str(similar_memory.get("id")) == str(target_memory_id):
                            target_memory_text = str(similar_memory.get("memory") or "")
                            break
                metadata = {
                    "source": "demo_live_chat",
                    "demo_session_id": self.sample_id,
                    "session_timestamp_raw": pending["session_date"],
                    "observed_at": pending["observed_at"],
                    "perspective_speaker": active["speaker"],
                    "profile_name": active["speaker"],
                    "candidate_memory_type": candidate.get("type"),
                    "candidate_memory_importance": candidate.get("importance"),
                    "event_at": candidate.get("event_at"),
                    "event_time_text": candidate.get("event_time_text"),
                }
                if operation == "ADD":
                    target_memory_id = self.store.add_memory(
                        active["user_id"],
                        candidate_memory_text,
                        metadata,
                    )
                    applied = True
                elif operation == "UPDATE" and target_memory_id:
                    applied = self.store.update_memory(
                        active["user_id"],
                        str(target_memory_id),
                        new_memory_text,
                        metadata,
                    )
                elif operation == "DELETE" and target_memory_id:
                    applied = self.store.delete_memory(active["user_id"], str(target_memory_id))

                if operation == "DELETE":
                    display_text = target_memory_text or candidate_memory_text
                elif operation == "UPDATE":
                    display_text = new_memory_text
                else:
                    display_text = candidate_memory_text

                update_events.append(
                    {
                        "candidate": candidate,
                        "operation": operation,
                        "applied": applied,
                        "target_memory_id": target_memory_id,
                        "target_memory_text": target_memory_text,
                        "new_memory": new_memory_text if operation == "UPDATE" else "",
                        "display_text": display_text,
                        "reason": decision.get("reason"),
                        "similar_count": len(similar),
                        "decision_latency_sec": safe_round(decision_log.get("update_decision_latency_sec")),
                    }
                )

            summary_after, summary_log = update_rolling_summary(
                self.chat_client,
                summary_model=self.llm_model,
                summary_max_tokens=self.summary_max_tokens,
                previous_summary=summary_before,
                new_messages=[
                    {"role": "user", "content": f"{active['speaker']}: {message}"},
                    {"role": "assistant", "content": f"Mem0 QASE: {answer}"},
                ],
                perspective_speaker=active["speaker"],
            )
            self.summaries[active["speaker"]] = summary_after
            extraction_latency = time.time() - extraction_started
            if not self.last_result:
                raise ValueError("No chat result to attach ingest result to.")
            self.last_result["summary_after"] = summary_after
            self.last_result["summary_log"] = summary_log
            self.last_result["extracted_candidates"] = candidates
            self.last_result["update_events"] = update_events
            self.last_result["update_pending"] = False
            self.last_result["stats"]["memory_update_latency_sec"] = safe_round(extraction_latency)
            self.last_result["stats"]["update_total_latency_sec"] = safe_round(extraction_latency)
            self.last_result["stats"]["memory_counts"] = {
                item["speaker"]: len(self.store.get_all(item["user_id"]))
                for item in self.samples[self.sample_id]
            }
            self.last_result["extraction_log"] = extraction_log
            return self.last_result

    @staticmethod
    def _flatten_selected(selected_by_speaker: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        memories = []
        for speaker, items in selected_by_speaker.items():
            for memory in items:
                metadata = memory.get("metadata") or {}
                text = memory_text(memory)
                memories.append(
                    {
                        "speaker": speaker,
                        "id": memory.get("id"),
                        "memory": text,
                        "text": text,
                        "metadata": metadata,
                        "observed_at": memory.get("observed_at") or metadata.get("observed_at"),
                        "event_at": memory.get("event_at") or metadata.get("event_at"),
                        "event_time_text": memory.get("event_time_text") or metadata.get("event_time_text"),
                        "score": safe_round(memory.get("qase_score")),
                        "semantic": safe_round(memory.get("qase_semantic_score")),
                        "lexical": safe_round(memory.get("qase_lexical_score")),
                        "time_bonus": safe_round(memory.get("qase_temporal_bonus")),
                        "rank": int(memory.get("qase_source_rank") or 0) + 1,
                    }
                )
        memories.sort(key=lambda item: (-safe_float(item.get("score")), item.get("rank") or 0))
        for index, memory in enumerate(memories, 1):
            memory["display_rank"] = index
        return memories


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Demo Mem0 QASE Live Chat</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #64748b;
      --line: #dfe7f1;
      --soft: #f8fbfd;
      --accent: #008f85;
      --accent-strong: #00756d;
      --accent-soft: #e7f8f5;
      --violet: #7657d8;
      --danger: #b42318;
      --shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, Segoe UI, Arial, sans-serif;
      min-width: 1120px;
    }
    main { display: grid; grid-template-columns: 300px minmax(430px, 1fr) 420px; gap: 14px; padding: 14px; height: 100vh; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      min-height: 0;
    }
    .sidebar { padding: 22px; display: flex; flex-direction: column; gap: 20px; }
    .product { display: grid; grid-template-columns: 62px 1fr; gap: 14px; align-items: center; }
    .cube { width: 62px; height: 62px; border: 1px solid var(--line); border-radius: 16px; display: grid; place-items: center; color: var(--accent); font-size: 28px; font-weight: 900; background: #fff; }
    .status { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 6px 10px; background: var(--accent-soft); color: var(--accent-strong); font-size: 13px; font-weight: 700; margin-top: 8px; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: #17b26a; }
    .desc { color: #526179; line-height: 1.65; margin: 0; }
    .divider { height: 1px; background: var(--line); }
    .meta-row { display: flex; justify-content: space-between; gap: 12px; color: #344054; font-size: 14px; align-items: center; }
    .meta-row b { text-align: right; }
    .run-btn {
      margin-top: auto;
      border: 0;
      border-radius: 14px;
      padding: 16px 18px;
      color: #fff;
      background: linear-gradient(145deg, #00a99d, #00756d);
      font-weight: 800;
      font-size: 16px;
      cursor: pointer;
      box-shadow: 0 16px 36px rgba(0, 143, 133, .28);
    }
    .run-btn:active, .send:active { transform: translateY(1px); }
    .chat { display: grid; grid-template-rows: 56px 1fr 98px; overflow: hidden; }
    .panel-title {
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      font-weight: 800;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .messages {
      padding: 26px 30px;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 18px;
      background: linear-gradient(180deg, #fff, #fbfdff);
    }
    .bubble {
      max-width: 78%;
      border-radius: 16px;
      padding: 16px 18px;
      line-height: 1.55;
      white-space: pre-wrap;
      border: 1px solid var(--line);
      box-shadow: 0 10px 28px rgba(15,23,42,.05);
    }
    .bubble.user { align-self: flex-end; background: #f4f0ff; border-color: #ded5ff; }
    .bubble.assistant { align-self: flex-start; background: #effdfa; border-color: #d3eee9; }
    .composer { padding: 18px 20px; border-top: 1px solid var(--line); background: #fff; }
    .input-wrap { display: grid; grid-template-columns: 1fr 54px; gap: 12px; }
    textarea {
      resize: none;
      width: 100%;
      height: 64px;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      font: inherit;
      outline: none;
      line-height: 1.4;
    }
    textarea:focus, select:focus { border-color: rgba(0,143,133,.55); box-shadow: 0 0 0 4px rgba(0,143,133,.08); }
    .send {
      border: 0;
      border-radius: 14px;
      background: var(--accent);
      color: #fff;
      font-size: 24px;
      font-weight: 900;
      cursor: pointer;
    }
    .right { display: grid; grid-template-rows: minmax(190px, 1.25fr) minmax(145px, .9fr) minmax(135px, .8fr) 96px; gap: 14px; min-height: 0; }
    .card { overflow: hidden; display: flex; flex-direction: column; }
    .card-body { padding: 16px; overflow: auto; min-height: 0; }
    .summary-grid { display: block; min-height: 0; }
    .summary-box { border: 1px solid var(--line); background: var(--soft); border-radius: 14px; padding: 13px; min-height: 0; }
    .summary-box strong { display: block; margin-bottom: 8px; font-size: 13px; }
    .summary-text { white-space: pre-wrap; color: #344054; font-size: 13px; line-height: 1.55; }
    .mem-list, .event-list { display: flex; flex-direction: column; gap: 10px; }
    .memory, .event { border: 1px solid var(--line); border-radius: 13px; padding: 12px; background: #fff; }
    .memory-top, .event-top { display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 6px; }
    .rank { width: 24px; height: 24px; display: inline-grid; place-items: center; border-radius: 50%; background: var(--accent-soft); color: var(--accent-strong); font-weight: 900; font-size: 12px; }
    .score { color: var(--accent-strong); background: var(--accent-soft); border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 800; white-space: nowrap; }
    .speaker { color: var(--violet); font-size: 12px; font-weight: 800; }
    .memory-text, .event-text { color: #344054; font-size: 13px; line-height: 1.45; }
    .event-op { border-radius: 999px; padding: 4px 8px; font-size: 12px; font-weight: 900; }
    .op-ADD { background: #ecfdf3; color: #067647; }
    .op-UPDATE { background: #eff8ff; color: #175cd3; }
    .op-DELETE { background: #fef3f2; color: var(--danger); }
    .op-NOOP { background: #f2f4f7; color: #475467; }
    .stats { padding: 10px; display: block; }
    .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; height: 100%; }
    .metric { padding: 9px 8px; text-align: center; background: #fff; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric b { display: block; font-size: 18px; margin-top: 4px; }
    .metric span { color: #667085; font-size: 11px; }
    .empty { color: #667085; border: 1px dashed var(--line); border-radius: 14px; padding: 16px; font-size: 13px; line-height: 1.5; background: #fff; }
    .loading { opacity: .72; }
  </style>
</head>
<body>
  <main>
    <aside class="panel sidebar">
      <div class="product">
        <div class="cube">M</div>
        <div>
          <h2 style="margin:0">Mem0 QASE</h2>
          <div class="status"><span class="dot"></span><span id="statusText">Starting</span></div>
        </div>
      </div>
      <p class="desc">Query-Aware & Self-Evaluating Memory for LLM agents. Each chat turn retrieves personal memory, answers, then updates memory.</p>
      <div class="divider"></div>
      <div class="meta-row"><span>Answer Model</span><b id="modelName">...</b></div>
      <div class="meta-row"><span>Embedding Model</span><b id="embeddingModelName">...</b></div>
      <button class="run-btn" id="resetBtn">Reset</button>
    </aside>

    <section class="panel chat">
      <div class="panel-title">Conversation <span id="planBadge" style="color:#667085;font-size:13px">planner: ...</span></div>
      <div class="messages" id="messages"></div>
      <div class="composer">
        <div class="input-wrap">
          <textarea id="chatInput" placeholder="Type your message..."></textarea>
          <button class="send" id="sendBtn">➤</button>
        </div>
      </div>
    </section>

    <section class="right">
      <div class="panel card">
        <div class="panel-title">Current Conversation Summary <span id="summaryMeta" style="color:#667085;font-size:13px">current</span></div>
        <div class="card-body">
          <div class="summary-grid">
            <div class="summary-box"><strong>Current state</strong><div class="summary-text" id="summaryCurrent">(empty)</div></div>
          </div>
        </div>
      </div>
      <div class="panel card">
        <div class="panel-title">Updated Memories <span id="updateCount" style="color:#667085;font-size:13px">0 events</span></div>
        <div class="card-body"><div class="event-list" id="updateList"><div class="empty">No memory update yet.</div></div></div>
      </div>
      <div class="panel card">
        <div class="panel-title">Retrieved Memories <span id="memoryCount" style="color:#667085;font-size:13px">0 results</span></div>
        <div class="card-body"><div class="mem-list" id="memoryList"><div class="empty">No chat turns yet.</div></div></div>
      </div>
      <div class="panel stats">
        <div class="metric-grid">
          <div class="metric"><span>Memory Tokens</span><b id="tokensMetric">0</b></div>
          <div class="metric"><span>QASE Retrieval</span><b id="retrievalMetric">0s</b></div>
          <div class="metric"><span>Answer</span><b id="answerMetric">0s</b></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let state = null;
    let busy = false;

    function esc(s) {
      return String(s ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function fmt(s) {
      if (!s) return '(empty)';
      return esc(s);
    }
    function addBubble(role, speaker, content) {
      const box = document.createElement('div');
      box.className = `bubble ${role}`;
      box.innerHTML = esc(content);
      $('messages').appendChild(box);
      $('messages').scrollTop = $('messages').scrollHeight;
    }
    function renderMessages(history) {
      $('messages').innerHTML = '';
      if (!history || history.length === 0) {
        return;
      }
      for (const msg of history) {
        addBubble(msg.role === 'user' ? 'user' : 'assistant', msg.speaker || msg.role, msg.content || '');
      }
    }
    function renderUpdateEvents(result) {
      if (result.update_pending) {
        $('updateCount').textContent = 'pending';
        $('updateList').innerHTML = '<div class="empty">Memory extraction and update will appear here after this turn is processed.</div>';
        return;
      }
      const events = result.update_events || [];
      $('updateCount').textContent = `${events.length} events`;
      $('updateList').innerHTML = events.length ? events.map((event, index) => {
        const op = String(event.operation || 'NOOP').toUpperCase();
        const safeOp = ['ADD', 'UPDATE', 'DELETE', 'NOOP'].includes(op) ? op : 'NOOP';
        const text = event.display_text || event.new_memory || event.target_memory_text || event.candidate?.memory || '(no memory text)';
        const status = event.applied ? 'applied' : 'not applied';
        const reason = event.reason ? `<div style="margin-top:7px;color:#667085;font-size:12px">Reason: ${esc(event.reason)}</div>` : '';
        return `
          <div class="event">
            <div class="event-top">
              <div><span class="rank">${index + 1}</span> <span class="event-op op-${safeOp}">${safeOp}</span></div>
              <span class="score">${status}</span>
            </div>
            <div class="event-text">${esc(text)}</div>
            ${reason}
          </div>`;
      }).join('') : '<div class="empty">No memory operation was produced.</div>';
    }
    function clearResultPanels() {
      $('planBadge').textContent = 'planner: ...';
      $('memoryCount').textContent = '0 results';
      $('memoryList').innerHTML = '<div class="empty">No question has been asked yet.</div>';
      $('updateCount').textContent = '0 events';
      $('updateList').innerHTML = '<div class="empty">No memory update yet.</div>';
      $('tokensMetric').textContent = '0';
      $('retrievalMetric').textContent = '0s';
      $('answerMetric').textContent = '0s';
    }
    function renderConfig(data) {
      $('modelName').textContent = data.model || '...';
      $('embeddingModelName').textContent = data.embedding_model || '...';
      $('statusText').textContent = 'Running';
      $('summaryCurrent').innerHTML = fmt(data.summary);
      $('summaryMeta').textContent = data.summary ? 'updated' : 'empty';
      renderMessages(data.history);
      if (data.last_result) renderResult(data.last_result, false);
      else clearResultPanels();
    }
    function renderResult(result, appendAnswer = true) {
      if (appendAnswer) addBubble('assistant', 'Mem0 QASE', result.answer || '');
      $('summaryCurrent').innerHTML = fmt(result.summary_after || result.summary_before);
      const qtype = result.plan?.question_type || 'unknown';
      $('planBadge').textContent = `planner: ${qtype} · ${result.selection?.total_selected || 0}/${result.selection?.total_input_candidates || 0}`;
      const memories = result.retrieved_memories || [];
      $('memoryCount').textContent = `${memories.length} results`;
      $('memoryList').innerHTML = memories.length ? memories.map(memory => `
        <div class="memory">
          <div class="memory-top">
            <div><span class="rank">${memory.display_rank}</span> <span class="speaker">${esc(memory.speaker)}</span></div>
            <span class="score">Score: ${esc(memory.score)}</span>
          </div>
          <div class="memory-text">${esc(memory.text)}</div>
        </div>`).join('') : '<div class="empty">No memory was selected.</div>';
      renderUpdateEvents(result);
      $('tokensMetric').textContent = result.stats?.retrieved_memory_tokens || 0;
      $('retrievalMetric').textContent = `${result.stats?.retrieval_latency_sec || 0}s`;
      $('answerMetric').textContent = `${result.stats?.answer_latency_sec || 0}s`;
    }
    async function api(path, payload) {
      const options = payload === undefined ? {} : {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      };
      const response = await fetch(path, options);
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || response.statusText);
      return data;
    }
    async function loadState() {
      state = await api('/api/state');
      renderConfig(state);
    }
    async function resetDemo() {
      state = await api('/api/reset', {});
      renderConfig(state);
    }
    async function sendMessage() {
      if (busy) return;
      const text = $('chatInput').value.trim();
      if (!text) return;
      busy = true;
      $('sendBtn').disabled = true;
      $('chatInput').value = '';
      addBubble('user', state?.active_speaker || 'User', text);
      const loading = document.createElement('div');
      loading.className = 'bubble assistant loading';
      loading.textContent = 'Retrieving memories and answering...';
      $('messages').appendChild(loading);
      $('messages').scrollTop = $('messages').scrollHeight;
      try {
        const result = await api('/api/chat', {message: text});
        loading.remove();
        renderResult(result, true);
        state = await api('/api/state');
        if (result.update_pending) {
          $('summaryMeta').textContent = 'extracting/updating...';
          api('/api/ingest_pending', {}).then(updated => {
            renderResult(updated, false);
            $('summaryMeta').textContent = 'updated';
          }).catch(err => {
            $('summaryMeta').textContent = 'update failed';
            addBubble('assistant', 'Update error', err.message || String(err));
          });
        }
      } catch (err) {
        loading.remove();
        addBubble('assistant', 'Demo error', err.message || String(err));
      } finally {
        busy = false;
        $('sendBtn').disabled = false;
      }
    }
    $('sendBtn').addEventListener('click', sendMessage);
    $('chatInput').addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });
    $('resetBtn').addEventListener('click', resetDemo);
    loadState().catch(err => {
      $('statusText').textContent = 'Error';
      addBubble('assistant', 'Demo error', err.message || String(err));
    });
  </script>
</body>
</html>
"""


class DemoHandler(BaseHTTPRequestHandler):
    app_state: LiveDemoState

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        try:
            raw = body.decode("utf-8")
        except UnicodeDecodeError:
            raw = body.decode("utf-8", errors="replace")
        return json.loads(raw or "{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            try:
                self._send_json(self.app_state.public_state())
            except Exception as exc:
                self._send_json({"error": str(exc)}, 500)
            return
        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/chat":
                self._send_json(self.app_state.chat(str(payload.get("message") or "")))
                return
            if path == "/api/ingest_pending":
                self._send_json(self.app_state.ingest_pending())
                return
            if path == "/api/reset":
                with self.app_state.lock:
                    self.app_state.reset_runtime()
                    self._send_json(self.app_state.public_state())
                return
            self._send_json({"error": "not found"}, 404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Mem0 QASE chat demo with retrieved memories and summary updates.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Optional local Mem0 wrapper checkpoint to preload. Omit for a fresh personal memory store.",
    )
    parser.add_argument("--user_name", default=DEFAULT_USER_NAME)
    parser.add_argument("--llm_model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--answer_max_tokens", type=int, default=1000)
    parser.add_argument("--summary_max_tokens", type=int, default=1000)
    parser.add_argument("--extraction_max_tokens", type=int, default=2000)
    parser.add_argument("--decision_max_tokens", type=int, default=1000)
    parser.add_argument("--qase_max_total_final_memories", type=int, default=22)
    parser.add_argument("--qase_lexical_weight", type=float, default=0.30)
    parser.add_argument("--api_max_retries", type=int, default=10)
    parser.add_argument("--api_timeout_sec", type=float, default=600.0)
    args = parser.parse_args()

    state = LiveDemoState(
        checkpoint_path=args.checkpoint,
        user_name=args.user_name,
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
        answer_max_tokens=args.answer_max_tokens,
        summary_max_tokens=args.summary_max_tokens,
        extraction_max_tokens=args.extraction_max_tokens,
        decision_max_tokens=args.decision_max_tokens,
        max_total_final_memories=args.qase_max_total_final_memories,
        lexical_weight=args.qase_lexical_weight,
        api_max_retries=args.api_max_retries,
        api_timeout_sec=args.api_timeout_sec,
    )
    DemoHandler.app_state = state
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Live Mem0 QASE demo: http://{args.host}:{args.port}")
    print("Memory store:", str(args.checkpoint) if args.checkpoint else "fresh personal store")
    server.serve_forever()


if __name__ == "__main__":
    main()
