import json
import re
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


VALID_MEMORY_EVENTS = {"ADD", "UPDATE", "DELETE", "NOOP"}
RETRYABLE_API_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
RETRYABLE_API_ERROR_TYPES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TimeoutException",
}
TEMPORAL_METADATA_FIELDS = (
    "session_timestamp_raw",
    "observed_at",
    "event_at",
    "event_time_text",
)


def messages_to_plain_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in messages)


def temporal_fields_from_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    source = metadata or {}
    return {field: deepcopy(source[field]) for field in TEMPORAL_METADATA_FIELDS if field in source}


def compact_candidate_for_update_prompt(candidate: Dict[str, Any]) -> Dict[str, Any]:
    fields = (
        "memory",
        "event_at",
        "event_time_text",
        "type",
        "importance",
    )
    return {field: candidate[field] for field in fields if candidate.get(field) is not None}


def compact_existing_memory_for_update_prompt(memory: Dict[str, Any]) -> Dict[str, Any]:
    metadata = memory.get("metadata") or {}
    compact = {
        "id": memory.get("id"),
        "memory": memory.get("memory") or memory.get("text"),
        "event_at": memory.get("event_at") or metadata.get("event_at"),
        "observed_at": memory.get("observed_at") or metadata.get("observed_at"),
        "score": memory.get("score"),
    }
    return {field: value for field, value in compact.items() if value is not None}


def normalize_event_at(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown"}:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return None
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_json_object(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def empty_usage(prefix: str) -> Dict[str, Any]:
    return {
        f"{prefix}_prompt_tokens": 0,
        f"{prefix}_completion_tokens": 0,
        f"{prefix}_total_tokens": 0,
        f"{prefix}_reasoning_tokens": 0,
        f"{prefix}_latency_sec": 0.0,
        f"{prefix}_finish_reason": None,
        f"{prefix}_retry_count": 0,
        f"{prefix}_parse_error": None,
        f"{prefix}_raw": None,
    }


def response_usage(response: Any) -> Dict[str, int]:
    usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
    }


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in RETRYABLE_API_STATUS_CODES:
        return True
    error_type = type(exc).__name__
    cause_type = type(getattr(exc, "__cause__", None)).__name__
    return error_type in RETRYABLE_API_ERROR_TYPES or cause_type in RETRYABLE_API_ERROR_TYPES


def json_chat_completion(
    chat_client: OpenAI,
    llm_model: str,
    system_content: str,
    user_content: str,
    max_tokens: int,
) -> Tuple[str, Dict[str, int], Optional[str]]:
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        try:
            response = chat_client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            break
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_api_error(exc):
                raise
            delay = min(2 ** (attempt - 1), 30)
            print(
                "Chat JSON API call failed; retrying "
                f"({attempt}/{max_attempts}) after {delay}s: {type(exc).__name__}"
            )
            time.sleep(delay)
    raw = response.choices[0].message.content or ""
    finish_reason = getattr(response.choices[0], "finish_reason", None)
    return raw, response_usage(response), finish_reason


def add_usage_to_log(log: Dict[str, Any], prefix: str, usage: Dict[str, int]) -> None:
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        log[f"{prefix}_{field}"] += int(usage.get(field) or 0)


def extract_candidate_memories(
    chat_client: OpenAI,
    llm_model: str,
    summary_text: Optional[str],
    context_messages: List[Dict[str, str]],
    target_messages: List[Dict[str, str]],
    session_date: Optional[str],
    observed_at: Optional[str],
    perspective_speaker: str,
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    prompt = f"""
You extract salient candidate long-term memories for one speaker from a conversation.

Focus on concrete information that a future assistant should remember: identities, relationships, dates, plans,
preferences, interests, locations, work or study details, health, family, important events, concrete personal
facts, and time-sensitive commitments. Capture all salient information without turning filler into memories.

Return JSON only in this exact shape:
{{
  "memories": [
    {{
      "memory": "...",
      "event_at": "ISO-8601 date or datetime" or null,
      "event_time_text": "original relative or explicit time expression" or null,
      "type": "fact|preference|event|relationship|identity|plan|other",
      "importance": "low|medium|high"
    }}
  ]
}}

Rules:
- Extract memories only from TARGET messages authored by {perspective_speaker}, represented by role=user.
- TARGET messages with role=assistant belong to the other conversation speaker. Use them only as context; do not
  extract memories from them into {perspective_speaker}'s store.
- Use the summary and context only to resolve ambiguity, pronouns, speaker grounding, and relative time.
- Do not extract a memory directly from the summary or context messages.
- Extract concrete, salient details from TARGET messages, including meaningful one-off events that may matter later.
- Split compound messages when they contain distinct important facts, dates, entities, activities, or plans.
- Each memory must be self-contained and useful for future recall.
- Use real speaker names, especially {perspective_speaker}; do not write "user".
- Preserve exact spelling, casing, and diacritics of names, locations, preferences, and quoted values from TARGET messages.
  Do not transliterate or normalize user-provided names.
- The observation time is supplied by the runner. Do not copy it into event_at unless the event itself occurs at that time.
- Resolve relative event times such as "yesterday" against the observation time when the date is unambiguous.
- Set event_at to null when TARGET messages do not establish when the remembered event occurred.
- Preserve the source wording such as "yesterday" or "last year" in event_time_text when present.
- Do not invent names, dates, locations, relationships, or preferences.
- Skip greetings, acknowledgements, conversational filler, vague reactions, and incidental details with no future
  informational value. Do not skip a one-off event merely because it happened only once.
- Avoid near-duplicate or overly tiny memories; merge tightly related details into one concise memory when that improves quality.
- Use low importance only for specific factual details with clear future value.
- Return [] when TARGET messages contain no salient fact about {perspective_speaker}.
- Keep every memory concise, but return complete valid JSON. Do not stop mid-JSON.

Perspective speaker:
{perspective_speaker}

Session timestamp (raw):
{session_date or "unknown"}

Observation time (canonical UTC, runner supplied when the source timestamp has no timezone):
{observed_at or "unknown"}

Conversation summary (context only):
{summary_text or "(none)"}

Recent messages (context only):
{messages_to_plain_text(context_messages) or "(none)"}

TARGET messages:
{messages_to_plain_text(target_messages)}
""".strip()

    log = empty_usage("candidate_extraction")
    start = time.time()
    system_content = (
        "You are an evidence-bound long-term memory extractor for one specified speaker. "
        "Capture all salient facts about that speaker, including meaningful one-off events, while excluding filler, "
        "duplicates, and assistant-role content belonging to the other speaker. Preserve exact source spelling and "
        "diacritics for names and quoted values. Return complete valid JSON only."
    )
    try:
        raw, usage, finish_reason = json_chat_completion(
            chat_client,
            llm_model,
            system_content,
            prompt,
            max_tokens,
        )
        log["candidate_extraction_latency_sec"] = time.time() - start
        add_usage_to_log(log, "candidate_extraction", usage)
        log["candidate_extraction_finish_reason"] = finish_reason
        log["candidate_extraction_raw"] = raw
        parsed = safe_json_object(raw)
    except Exception as exc:
        first_raw = log.get("candidate_extraction_raw")
        first_finish_reason = log.get("candidate_extraction_finish_reason")
        retry_prompt = f"""
The previous candidate extraction response was invalid, empty, or incomplete.
Return COMPLETE VALID JSON only. Do not include markdown.

Important:
- Extract all salient memories about the specified perspective speaker from TARGET messages.
- Include meaningful one-off events, but exclude filler, duplicates, and assistant-role content.
- Keep each memory short enough that the JSON can finish.
- If there are salient facts about the perspective speaker, do not return an empty list.

Original extraction task:
{prompt}
""".strip()
        retry_start = time.time()
        try:
            raw, usage, finish_reason = json_chat_completion(
                chat_client,
                llm_model,
                system_content,
                retry_prompt,
                max(max_tokens, 4000),
            )
            log["candidate_extraction_latency_sec"] = (time.time() - start)
            add_usage_to_log(log, "candidate_extraction", usage)
            log["candidate_extraction_retry_count"] = 1
            log["candidate_extraction_first_raw"] = first_raw
            log["candidate_extraction_first_finish_reason"] = first_finish_reason
            log["candidate_extraction_retry_latency_sec"] = time.time() - retry_start
            log["candidate_extraction_finish_reason"] = finish_reason
            log["candidate_extraction_raw"] = raw
            parsed = safe_json_object(raw)
        except Exception as retry_exc:
            log["candidate_extraction_latency_sec"] = time.time() - start
            log["candidate_extraction_retry_count"] = 1
            log["candidate_extraction_first_raw"] = first_raw
            log["candidate_extraction_first_finish_reason"] = first_finish_reason
            log["candidate_extraction_parse_error"] = f"{exc}; retry failed: {retry_exc}"
            return [], log

    raw_memories = parsed.get("memories", [])
    if not isinstance(raw_memories, list):
        log["candidate_extraction_parse_error"] = "JSON field memories is not a list."
        return [], log

    memories = []
    for item in raw_memories:
        if not isinstance(item, dict):
            continue
        memory_text = str(item.get("memory") or "").strip()
        if not memory_text:
            continue
        event_at_value = item.get("event_at")
        event_at = normalize_event_at(event_at_value)
        if event_at_value and event_at is None and str(event_at_value).strip().lower() not in {
            "null",
            "none",
            "unknown",
        }:
            log.setdefault("candidate_extraction_invalid_event_at", []).append(str(event_at_value))
        event_time_text_value = item.get("event_time_text")
        event_time_text = str(event_time_text_value).strip() if event_time_text_value else None
        candidate = {
            "memory": memory_text,
            "event_at": event_at,
            "event_time_text": event_time_text,
            "type": str(item.get("type") or "other").strip() or "other",
            "importance": str(item.get("importance") or "medium").strip() or "medium",
        }
        memories.append(candidate)

    return memories, log


def decide_memory_operation(
    chat_client: OpenAI,
    llm_model: str,
    candidate_memory: Dict[str, Any],
    existing_memories: List[Dict[str, Any]],
    max_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    existing_ids = {str(item.get("id")) for item in existing_memories if item.get("id")}
    prompt_candidate = compact_candidate_for_update_prompt(candidate_memory)
    prompt_existing_memories = [compact_existing_memory_for_update_prompt(item) for item in existing_memories]
    prompt = f"""
You decide how a candidate memory should change a local memory store.

Return JSON only in this exact shape:
{{
  "operation": "ADD|UPDATE|DELETE|NOOP",
  "target_memory_id": "..." or null,
  "reason": "..."
}}

Decision rules:
- ADD: choose ADD when no semantically similar memory exists.
- DELETE: choose DELETE when the candidate contradicts an existing memory. target_memory_id must identify the
  contradicted memory.
- UPDATE: choose UPDATE when the candidate augments a related existing memory AND the candidate contains more
  information than that existing memory. target_memory_id must identify the related memory.
- NOOP: choose NOOP when the fact already exists, is irrelevant, does not add information, or an UPDATE candidate
  is not richer than the related memory.
- Apply the classification order from Algorithm 1: no semantic match -> ADD; contradiction -> DELETE;
  richer augmentation -> UPDATE; otherwise -> NOOP.
- For UPDATE or DELETE, target_memory_id must be one of the existing memory ids.

Candidate memory:
{json.dumps(prompt_candidate, ensure_ascii=False, separators=(",", ":"))}

Existing memories:
{json.dumps(prompt_existing_memories, ensure_ascii=False, separators=(",", ":"))}
""".strip()

    log = empty_usage("update_decision")
    start = time.time()
    system_content = (
        "You classify candidate memory operations for a local memory store and return complete valid JSON only. "
        "Follow ADD, DELETE, UPDATE, and NOOP semantics from the supplied decision rules without inventing facts."
    )
    try:
        raw, usage, finish_reason = json_chat_completion(
            chat_client,
            llm_model,
            system_content,
            prompt,
            max_tokens,
        )
        log["update_decision_latency_sec"] = time.time() - start
        add_usage_to_log(log, "update_decision", usage)
        log["update_decision_finish_reason"] = finish_reason
        log["update_decision_raw"] = raw
        parsed = safe_json_object(raw)
    except Exception as exc:
        first_raw = log.get("update_decision_raw")
        first_finish_reason = log.get("update_decision_finish_reason")
        retry_prompt = f"""
The previous update-decision response was invalid, empty, or incomplete.
Return COMPLETE VALID JSON only. Do not include markdown.
Follow Algorithm 1: no semantic match means ADD; contradiction means DELETE; richer augmentation means UPDATE;
otherwise choose NOOP.

Original decision task:
{prompt}
""".strip()
        retry_start = time.time()
        try:
            raw, usage, finish_reason = json_chat_completion(
                chat_client,
                llm_model,
                system_content,
                retry_prompt,
                max(max_tokens, 2000),
            )
            log["update_decision_latency_sec"] = time.time() - start
            add_usage_to_log(log, "update_decision", usage)
            log["update_decision_retry_count"] = 1
            log["update_decision_first_raw"] = first_raw
            log["update_decision_first_finish_reason"] = first_finish_reason
            log["update_decision_retry_latency_sec"] = time.time() - retry_start
            log["update_decision_finish_reason"] = finish_reason
            log["update_decision_raw"] = raw
            parsed = safe_json_object(raw)
        except Exception as retry_exc:
            log["update_decision_latency_sec"] = time.time() - start
            log["update_decision_retry_count"] = 1
            log["update_decision_first_raw"] = first_raw
            log["update_decision_first_finish_reason"] = first_finish_reason
            log["update_decision_parse_error"] = f"{exc}; retry failed: {retry_exc}"
            return fallback_noop_decision("NOOP due to update decision parse/error"), log

    operation = str(parsed.get("operation") or "").upper()
    target_memory_id = parsed.get("target_memory_id")
    target_memory_id = str(target_memory_id) if target_memory_id is not None else None
    reason = str(parsed.get("reason") or "").strip()
    candidate_text = str(candidate_memory.get("memory") or "").strip()

    if operation not in VALID_MEMORY_EVENTS:
        return fallback_noop_decision("NOOP due to invalid operation from update decision"), log
    if operation in {"UPDATE", "DELETE"} and target_memory_id not in existing_ids:
        return fallback_noop_decision("NOOP due to missing/invalid target_memory_id"), log
    if operation == "ADD":
        target_memory_id = None
    new_memory = candidate_text if operation in {"ADD", "UPDATE"} else None

    return {
        "operation": operation,
        "target_memory_id": target_memory_id,
        "new_memory": new_memory,
        "reason": reason,
        "fallback": False,
    }, log


def fallback_noop_decision(reason: str) -> Dict[str, Any]:
    return {
        "operation": "NOOP",
        "target_memory_id": None,
        "new_memory": None,
        "reason": reason,
        "fallback": True,
    }


def verify_update_information_content(
    chat_client: OpenAI,
    llm_model: str,
    candidate_memory: Dict[str, Any],
    existing_memory: Dict[str, Any],
    max_tokens: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply Algorithm 1's information-content guard after UPDATE classification."""
    prompt_candidate = compact_candidate_for_update_prompt(candidate_memory)
    prompt_existing = compact_existing_memory_for_update_prompt(existing_memory)
    prompt = f"""
Algorithm 1 has already classified the candidate as a possible UPDATE. Now independently verify its
InformationContent condition before replacement.

Return JSON only in this exact shape:
{{
  "candidate_is_strictly_richer": true|false,
  "reason": "..."
}}

The candidate is strictly richer only when BOTH conditions hold:
1. It preserves or clearly entails every important factual detail in the existing memory, including identity,
   relationship, activity, date/time, location, status, intent, and uncertainty when present.
2. It adds at least one concrete, useful factual detail not already represented by the existing memory.

Return false when the candidate is merely related, complementary but does not subsume the existing fact, equally
informative, more vague, less specific, missing any important detail, or contradictory. Do not treat longer wording
as greater information content. When uncertain, return false so the existing memory is preserved.

Candidate memory:
{json.dumps(prompt_candidate, ensure_ascii=False, separators=(",", ":"))}

Existing memory that would be replaced:
{json.dumps(prompt_existing, ensure_ascii=False, separators=(",", ":"))}
""".strip()

    log = empty_usage("information_content_guard")
    start = time.time()
    system_content = (
        "You are a conservative information-content verifier for Algorithm 1. Approve replacement only when the "
        "candidate strictly preserves all important existing facts and adds concrete information. Return valid JSON only."
    )
    try:
        raw, usage, finish_reason = json_chat_completion(
            chat_client,
            llm_model,
            system_content,
            prompt,
            max_tokens,
        )
        log["information_content_guard_latency_sec"] = time.time() - start
        add_usage_to_log(log, "information_content_guard", usage)
        log["information_content_guard_finish_reason"] = finish_reason
        log["information_content_guard_raw"] = raw
        parsed = safe_json_object(raw)
    except Exception as exc:
        log["information_content_guard_latency_sec"] = time.time() - start
        log["information_content_guard_parse_error"] = str(exc)
        return {
            "candidate_is_strictly_richer": False,
            "reason": "UPDATE rejected because the information-content guard failed to return valid JSON.",
            "fallback": True,
        }, log

    value = parsed.get("candidate_is_strictly_richer")
    if not isinstance(value, bool):
        log["information_content_guard_parse_error"] = (
            "JSON field candidate_is_strictly_richer must be a boolean."
        )
        return {
            "candidate_is_strictly_richer": False,
            "reason": "UPDATE rejected because the information-content guard returned an invalid boolean.",
            "fallback": True,
        }, log

    return {
        "candidate_is_strictly_richer": value,
        "reason": str(parsed.get("reason") or "").strip(),
        "fallback": False,
    }, log


class LocalPaperMemoryStore:
    def __init__(
        self,
        embedding_model: str,
        embedding_client: Optional[OpenAI] = None,
        embedding_client_factory: Optional[Callable[[], OpenAI]] = None,
        embedding_recovery_attempts: int = 10,
    ):
        self.embedding_model = embedding_model
        self.embedding_client = embedding_client or OpenAI()
        self.embedding_client_factory = embedding_client_factory
        self.embedding_recovery_attempts = embedding_recovery_attempts
        self.embedding_recovery_log: List[Dict[str, Any]] = []
        self.memories_by_user: Dict[str, List[Dict[str, Any]]] = {}
        self.embedding_cache: Dict[str, List[float]] = {}

    def _prepare_metadata(self, memory_text: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        return deepcopy(metadata or {})

    def add_memory(self, user_id: str, memory_text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        now = datetime.now(timezone.utc).isoformat()
        memory_id = f"pmem_{uuid.uuid4().hex}"
        stored_metadata = self._prepare_metadata(memory_text, metadata)
        temporal_fields = temporal_fields_from_metadata(stored_metadata)
        item = {
            "id": memory_id,
            "user_id": user_id,
            "memory": memory_text,
            "embedding": self._embed(memory_text),
            "metadata": stored_metadata,
            **temporal_fields,
            "created_at": now,
            "updated_at": now,
            "history": [
                {
                    "event": "ADD",
                    "memory": memory_text,
                    **temporal_fields,
                    "created_at": now,
                }
            ],
        }
        self.memories_by_user.setdefault(user_id, []).append(item)
        return memory_id

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        new_memory_text: str,
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> bool:
        item = self._find_item(user_id, memory_id)
        if item is None:
            return False
        now = datetime.now(timezone.utc).isoformat()
        previous_memory = item["memory"]
        item["memory"] = new_memory_text
        item["embedding"] = self._embed(new_memory_text)
        item["updated_at"] = now
        merged_metadata = deepcopy(item.get("metadata") or {})
        if metadata_update:
            merged_metadata.update(deepcopy(metadata_update))
        merged_metadata = self._prepare_metadata(new_memory_text, merged_metadata)
        temporal_update = temporal_fields_from_metadata(merged_metadata)
        item["metadata"] = merged_metadata
        item.update(temporal_update)
        item.setdefault("history", []).append(
            {
                "event": "UPDATE",
                "previous_memory": previous_memory,
                "memory": new_memory_text,
                **temporal_update,
                "created_at": now,
            }
        )
        return True

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        items = self.memories_by_user.get(user_id, [])
        for index, item in enumerate(items):
            if item.get("id") == memory_id:
                del items[index]
                return True
        return False

    def search(self, user_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        items = self.memories_by_user.get(user_id, [])
        if not items or top_k <= 0:
            return []
        query_embedding = np.asarray(self._embed(query), dtype=np.float32)
        scored = []
        for item in items:
            memory_embedding = np.asarray(item.get("embedding") or [], dtype=np.float32)
            score = self._cosine(query_embedding, memory_embedding)
            public_item = self._public_item(item)
            public_item["score"] = score
            scored.append(public_item)
        scored.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return scored[:top_k]

    def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        return [self._public_item(item) for item in self.memories_by_user.get(user_id, [])]

    def export_state(self) -> Dict[str, Any]:
        return {
            "embedding_model": self.embedding_model,
            "memories_by_user": deepcopy(self.memories_by_user),
            "embedding_cache": deepcopy(self.embedding_cache),
            "embedding_recovery_log": deepcopy(self.embedding_recovery_log),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        saved_model = state.get("embedding_model")
        if saved_model and saved_model != self.embedding_model:
            raise ValueError(
                f"Checkpoint memory store embedding model {saved_model!r} does not match "
                f"current embedding model {self.embedding_model!r}."
            )
        self.memories_by_user = deepcopy(state.get("memories_by_user") or {})
        self.embedding_cache = deepcopy(state.get("embedding_cache") or {})
        self.embedding_recovery_log = deepcopy(state.get("embedding_recovery_log") or [])

    def _find_item(self, user_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        for item in self.memories_by_user.get(user_id, []):
            if item.get("id") == memory_id:
                return item
        return None

    def _public_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        public_item = {key: deepcopy(value) for key, value in item.items() if key != "embedding"}
        return public_item

    def _embed(self, text: str) -> List[float]:
        text = text or ""
        cached = self.embedding_cache.get(text)
        if cached is not None:
            return cached
        recovery_attempt = 0
        retryable_status_codes = {408, 409, 429, 500, 502, 503, 504}
        retryable_error_types = {
            "APIConnectionError",
            "APITimeoutError",
            "ConnectError",
            "ConnectTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "TimeoutException",
        }
        while True:
            try:
                response = self.embedding_client.embeddings.create(model=self.embedding_model, input=text)
                break
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                error_type = type(exc).__name__
                cause_type = type(getattr(exc, "__cause__", None)).__name__
                is_retryable_connection_error = (
                    error_type in retryable_error_types
                    or cause_type in retryable_error_types
                )
                can_recover = (
                    (status_code == 431 or status_code in retryable_status_codes or is_retryable_connection_error)
                    and recovery_attempt < self.embedding_recovery_attempts
                )
                if not can_recover:
                    raise
                recovery_attempt += 1
                self.embedding_recovery_log.append(
                    {
                        "status_code": status_code,
                        "attempt": recovery_attempt,
                        "error_type": error_type,
                        "cause_type": cause_type,
                        "recovered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                )
                print(
                    "Embedding API call failed; resetting the HTTP client "
                    f"and retrying ({recovery_attempt}/{self.embedding_recovery_attempts})."
                )
                if self.embedding_client_factory is not None:
                    old_client = self.embedding_client
                    self.embedding_client = self.embedding_client_factory()
                    try:
                        old_client.close()
                    except Exception:
                        pass
                time.sleep(min(2 ** (recovery_attempt - 1), 8))
        embedding = response.data[0].embedding
        self.embedding_cache[text] = embedding
        return embedding

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator == 0.0:
            return 0.0
        return float(np.dot(a, b) / denominator)
