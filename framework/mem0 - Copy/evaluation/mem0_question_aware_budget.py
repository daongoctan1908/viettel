"""
Question-aware selective evidence retrieval for Mem0 local QA.

This module is retrieval-only. It does not modify ingestion, memory updates,
timestamps, summaries, or the evaluator. It takes a question, plans a retrieval
budget per speaker, and selects a compact evidence pack from retrieved memory
candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class QuestionType(str, Enum):
    SINGLE_HOP = "single_hop"
    TEMPORAL = "temporal"
    MULTI_HOP = "multi_hop"
    FALLBACK_AMBIGUOUS = "fallback_ambiguous"


@dataclass
class QASEConfig:
    # Candidate pool requested from the base semantic retriever.
    candidate_single_hop_target: int = 16
    candidate_single_hop_other: int = 3
    candidate_temporal_target: int = 22
    candidate_temporal_other: int = 8
    candidate_multi_hop_target: int = 20
    candidate_multi_hop_other: int = 20
    candidate_fallback_each: int = 18

    # Final cap after evidence selection. These are intentionally below the
    # current Mem0 baseline upper bound of 30 total memories (15 per speaker).
    final_single_hop_target: int = 9
    final_single_hop_other: int = 1
    final_temporal_target: int = 12
    final_temporal_other: int = 3
    final_multi_hop_target: int = 10
    final_multi_hop_other: int = 10
    final_fallback_each: int = 8

    max_total_final_memories: int = 22
    min_single_hop: int = 6
    min_temporal: int = 8
    min_multi_hop: int = 9
    min_fallback_ambiguous: int = 3
    semantic_anchor_simple: int = 2
    semantic_anchor_complex: int = 3
    candidate_pool_scale: float = 1.0
    final_k_scale: float = 1.0
    budget_allocation_mode: str = "speaker_aware"

    adaptive_cutoff_enabled: bool = True
    adaptive_drop_simple: float = 0.18
    adaptive_drop_complex: float = 0.28
    complex_min_fraction_of_cap: float = 0.60

    diversity_lambda_simple: float = 0.08
    diversity_lambda_complex: float = 0.18
    score_mode: str = "similarity"
    temporal_bonus_weight: float = 0.12
    lexical_weight: float = 0.30

    planner_model_root: Optional[str] = None
    question_type_model_dir: Optional[str] = None
    target_speaker_model_dir: Optional[str] = None
    planner_model_device: int = -1

    confidence_fallback_enabled: bool = True
    confidence_fallback_min_fraction_simple: float = 0.70
    confidence_fallback_min_fraction_complex: float = 0.80
    confidence_fallback_low_score_threshold: float = 0.62
    semantic_anchor_enabled: bool = True
    adaptive_cutoff_mode: str = "best_delta"


@dataclass
class RetrievalPlan:
    question: str
    question_type: QuestionType
    target_speakers: List[str]
    all_speakers: List[str]
    candidate_top_k_by_speaker: Dict[str, int]
    final_top_k_cap_by_speaker: Dict[str, int]
    max_total_final_memories: int
    needs_time: bool
    needs_cross_speaker: bool
    needs_diversity: bool
    complexity_score: float
    reason: str
    matched_signals: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["question_type"] = self.question_type.value
        return data


@dataclass
class SelectionLog:
    question_type: str
    total_input_candidates: int
    total_selected: int
    selected_ids: List[str]
    dropped_ids: List[str]
    per_speaker: Dict[str, Dict[str, Any]]
    max_total_final_memories: int
    latency_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "the",
    "their",
    "they",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "with",
    "would",
}

_SIGNAL_PATTERNS: Dict[str, List[str]] = {
    "comparison": [
        r"\bboth\b",
        r"\btogether\b",
        r"\bin common\b",
        r"\bsimilar\b",
        r"\bsame\b",
        r"\bdifference\b",
        r"\bdifferent\b",
        r"\bcompare\b",
        r"\bbetween\b",
        r"\beach other\b",
        r"\bshared\b",
        r"\balike\b",
    ],
    "temporal_change": [
        r"\bwhat changes?\b",
        r"\bhow .* changed?\b",
        r"\bchanged? .* (over time|after|since|from|during)\b",
        r"\bpreviously\b",
        r"\bused to\b",
        r"\bno longer\b",
        r"\bformerly\b",
        r"\btransition journey\b",
    ],
    "temporal_fact": [
        r"\bwhen\b",
        r"\bwhat year\b",
        r"\bwhat month\b",
        r"\bwhat date\b",
        r"\bwhat day\b",
        r"\bhow long\b",
        r"\bhow often\b",
        r"\bduring\b",
        r"\bbefore\b",
        r"\bafter\b",
        r"\bsince\b",
        r"\brecently\b",
        r"\blast year\b",
        r"\blast summer\b",
        r"\bin \d{4}\b",
        r"\b\d{4}\b",
        *(rf"\b{month}\b" for month in _MONTHS),
    ],
    "list_open": [
        r"\bwhat are some\b",
        r"\bsome changes\b",
        r"\bwhat activities\b",
        r"\bwhat events\b",
        r"\bin what ways\b",
        r"\bwhich (two|three|several|multiple) .*novels?\b",
        r"\bwhich (two|three|several|multiple) .*books?\b",
        r"\blist\b",
        r"\ball\b",
    ],
    "bounded_list": [
        r"\bhow many\b",
        r"^where\s+(?:has|have|did|does|is|are|was|were|can|could|would)\b",
    ],
    "preference_inference": [
        r"\bwould\b",
        r"\blikely\b",
        r"\bprobably\b",
        r"\bmight\b",
        r"\bcould\b",
        r"\bif\b",
        r"\bprefer\b",
        r"\benjoy\b",
        r"\binterested in\b",
        r"\bconsidered\b",
    ],
    "multi_hop": [
        r"\bwhy\b",
        r"\bbecause\b",
        r"\breason\b",
        r"\brelationship\b",
        r"\brelated\b",
        r"\bconnection\b",
        r"\bled to\b",
        r"\binfluence\b",
        r"\bimpact\b",
        r"\bcaused?\b",
        r"\binspired\b",
        r"\btake away\b",
    ],
    "profile": [
        r"\blike[s]?\b",
        r"\benjoy[s]?\b",
        r"\bprefer[s]?\b",
        r"\bfavorite\b",
        r"\bhobby\b",
        r"\bhobbies\b",
        r"\binterest[s]?\b",
        r"\busually\b",
        r"\boften\b",
        r"\bjob\b",
        r"\bwork\b",
        r"\bstudy\b",
        r"\bfamily\b",
        r"\bidentity\b",
        r"\bcareer\b",
        r"\bbackground\b",
        r"\bmovie[s]?\b",
        r"\btrilogy\b",
        r"\bvideo game[s]?\b",
        r"\bcomposer[s]?\b",
        r"\bcharacter[s]?\b",
        r"\bnovel[s]?\b",
        r"\btheme\b",
        r"\boutdoor activities\b",
    ],
}

_TIME_TEXT_SIGNAL_PATTERNS = [
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}\b",
    r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:"
    + "|".join(_MONTHS)
    + r")\b",
    r"\b(?:"
    + "|".join(_MONTHS)
    + r")\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?\b",
    r"\b(?:"
    + "|".join(_MONTHS)
    + r")\b",
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(?:today|yesterday|tomorrow|tonight|currently|now)\b",
    r"\b(?:last|next|this|previous|following)\s+"
    r"(?:day|week|month|year|summer|spring|winter|fall|autumn|semester|weekend)\b",
    r"\b(?:recently|lately|previously|formerly|earlier|later|afterwards|subsequently)\b",
    r"\b(?:before|after|since|during|until|by)\b",
    r"\b(?:used to|no longer)\b",
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|few|several)\s+"
    r"(?:days?|weeks?|months?|years?)\s+(?:ago|later|after|before)\b",
]

_SPEAKER_ALIASES: Dict[str, set[str]] = {
    "melanie": {"mel"},
}

def _normalize_text(text: Any) -> str:
    normalized = str(text or "").replace("`", "'")
    normalized = re.sub(r"\s+", " ", normalized.strip())
    return normalized


def _tokens(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
        if token and token not in _STOPWORDS and len(token) > 1
    ]


def _token_set(text: str) -> set[str]:
    return set(_tokens(text))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _memory_text(memory: Dict[str, Any]) -> str:
    return str(memory.get("memory") or memory.get("text") or memory.get("content") or "")


def _memory_id(memory: Dict[str, Any], fallback: str) -> str:
    return str(memory.get("id") or memory.get("memory_id") or memory.get("uuid") or fallback)


def _memory_metadata(memory: Dict[str, Any]) -> Dict[str, Any]:
    metadata = memory.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _memory_temporal_metadata(memory: Dict[str, Any]) -> Tuple[Any, Any]:
    metadata = _memory_metadata(memory)
    event_at = memory.get("event_at") or metadata.get("event_at")
    event_time_text = memory.get("event_time_text") or metadata.get("event_time_text")
    return event_at, event_time_text


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def _memory_has_time_text(memory: Dict[str, Any]) -> bool:
    text = _memory_text(memory).lower()
    return any(re.search(pattern, text) for pattern in _TIME_TEXT_SIGNAL_PATTERNS)


def _memory_has_time_metadata(memory: Dict[str, Any]) -> bool:
    event_at, event_time_text = _memory_temporal_metadata(memory)
    return _has_value(event_at) or _has_value(event_time_text)


def _memory_has_time(memory: Dict[str, Any]) -> bool:
    return _memory_has_time_text(memory) or _memory_has_time_metadata(memory)


def _compile_matches(question: str, patterns: Sequence[str]) -> List[str]:
    q = question.lower()
    return [pattern for pattern in patterns if re.search(pattern, q)]


class QuestionAwareSelectiveEvidence:
    def __init__(self, config: Optional[QASEConfig] = None):
        self.config = config or QASEConfig()
        self._question_type_classifier = None
        self._target_speaker_classifier = None
        self._question_type_id2label: Dict[str, str] = {}
        self._target_speaker_id2label: Dict[str, str] = {}

    def plan(self, question: str, all_speakers: Sequence[str]) -> RetrievalPlan:
        question = _normalize_text(question)
        all_speakers_list = [str(speaker) for speaker in all_speakers]
        if not all_speakers_list:
            raise ValueError("all_speakers must not be empty")

        matched = {name: _compile_matches(question, patterns) for name, patterns in _SIGNAL_PATTERNS.items()}
        targets, target_reason = self._target_speakers_for_plan(question, all_speakers_list)
        cross_speaker_relation = len(targets) >= 2 and self._is_directional_two_speaker_question(question)
        if cross_speaker_relation:
            matched["cross_speaker_relation"] = ["directional speaker relation"]
        qtype, reason = self._classify_for_plan(question, targets, matched, cross_speaker_relation)
        if target_reason:
            reason = f"{reason}; {target_reason}"
        candidate_budget, final_budget = self._budget_for_type(qtype, targets, all_speakers_list)
        final_budget = self._cap_final_budget(final_budget, qtype, targets, all_speakers_list)
        needs_time = qtype == QuestionType.TEMPORAL
        needs_cross = len(targets) >= 2
        needs_diversity = qtype in {QuestionType.MULTI_HOP, QuestionType.FALLBACK_AMBIGUOUS}
        return RetrievalPlan(
            question=question,
            question_type=qtype,
            target_speakers=targets,
            all_speakers=all_speakers_list,
            candidate_top_k_by_speaker=candidate_budget,
            final_top_k_cap_by_speaker=final_budget,
            max_total_final_memories=self.config.max_total_final_memories,
            needs_time=needs_time,
            needs_cross_speaker=needs_cross,
            needs_diversity=needs_diversity,
            complexity_score=self._complexity_score(question, qtype, targets, matched),
            reason=reason,
            matched_signals=matched,
        )

    def _target_speakers_for_plan(self, question: str, all_speakers: Sequence[str]) -> Tuple[List[str], str]:
        label = self._predict_target_speaker_label(question)
        if not label:
            return self._target_speakers(question, all_speakers), ""
        if label == "speaker_a":
            return ([all_speakers[0]] if len(all_speakers) >= 1 else []), "target speaker from model: speaker_a"
        if label == "speaker_b":
            return ([all_speakers[1]] if len(all_speakers) >= 2 else []), "target speaker from model: speaker_b"
        if label == "both":
            return list(all_speakers), "target speaker from model: both"
        if label == "unknown":
            return [], "target speaker from model: unknown"
        return self._target_speakers(question, all_speakers), f"target speaker model label ignored: {label}"

    def _classify_for_plan(
        self,
        question: str,
        target_speakers: Sequence[str],
        matched: Dict[str, List[str]],
        cross_speaker_relation: bool = False,
    ) -> Tuple[QuestionType, str]:
        label = self._predict_question_type_label(question)
        if label:
            try:
                return QuestionType(label), f"question type from model: {label}"
            except ValueError:
                return QuestionType.FALLBACK_AMBIGUOUS, f"unknown question type model label: {label}"
        return self._classify(question, target_speakers, matched, cross_speaker_relation)

    def _predict_question_type_label(self, question: str) -> Optional[str]:
        model_dir = self._resolve_planner_model_dir(
            self.config.question_type_model_dir,
            "qase_question_type_distilroberta",
        )
        if model_dir is None:
            return None
        if self._question_type_classifier is None:
            self._question_type_id2label = self._load_id2label(model_dir)
            self._question_type_classifier = self._load_text_classifier(model_dir)
        return self._predict_label(self._question_type_classifier, self._question_type_id2label, question)

    def _predict_target_speaker_label(self, question: str) -> Optional[str]:
        model_dir = self._resolve_planner_model_dir(
            self.config.target_speaker_model_dir,
            "qase_target_speaker_distilroberta",
        )
        if model_dir is None:
            return None
        if self._target_speaker_classifier is None:
            self._target_speaker_id2label = self._load_id2label(model_dir)
            self._target_speaker_classifier = self._load_text_classifier(model_dir)
        return self._predict_label(self._target_speaker_classifier, self._target_speaker_id2label, question)

    def _resolve_planner_model_dir(self, explicit_dir: Optional[str], default_leaf: str) -> Optional[Path]:
        if explicit_dir:
            path = Path(explicit_dir)
        elif self.config.planner_model_root:
            path = Path(self.config.planner_model_root) / default_leaf
        else:
            return None
        return path if path.exists() else None

    def _load_id2label(self, model_dir: Path) -> Dict[str, str]:
        label_map_path = model_dir / "label_map.json"
        if not label_map_path.exists():
            return {}
        try:
            data = json.loads(label_map_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {str(key): str(value) for key, value in (data.get("id2label") or {}).items()}

    def _load_text_classifier(self, model_dir: Path):
        try:
            import torch  # noqa: F401
            from transformers import pipeline
        except Exception as exc:
            raise RuntimeError(
                "QASE planner model requires PyTorch and transformers. "
                "Install torch in the active environment before using planner_model_root."
            ) from exc
        return pipeline(
            "text-classification",
            model=str(model_dir),
            tokenizer=str(model_dir),
            device=int(self.config.planner_model_device),
        )

    def _predict_label(self, classifier: Any, id2label: Dict[str, str], question: str) -> Optional[str]:
        output = classifier(question)
        if isinstance(output, list):
            if not output:
                return None
            output = output[0]
        label = str((output or {}).get("label") or "").strip()
        if not label:
            return None
        match = re.fullmatch(r"LABEL_(\d+)", label)
        if match and id2label:
            return id2label.get(match.group(1))
        return label

    def select(
        self,
        candidates_by_speaker: Dict[str, Sequence[Dict[str, Any]]],
        plan: RetrievalPlan,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], SelectionLog]:
        start = time.time()
        question_tokens = _token_set(plan.question)
        selected_by_speaker: Dict[str, List[Dict[str, Any]]] = {}
        per_speaker_log: Dict[str, Dict[str, Any]] = {}
        all_input_ids: List[str] = []

        for speaker in plan.all_speakers:
            candidates = list(candidates_by_speaker.get(speaker) or [])
            cap = max(0, int(plan.final_top_k_cap_by_speaker.get(speaker) or 0))
            all_input_ids.extend(_memory_id(item, f"{speaker}:{idx}") for idx, item in enumerate(candidates))
            if cap <= 0 or not candidates:
                selected_by_speaker[speaker] = []
                per_speaker_log[speaker] = {
                    "input_count": len(candidates),
                    "selected_count": 0,
                    "cap": cap,
                    "reason": "empty candidate list or zero cap",
                }
                continue

            scored = self._score_candidates(candidates, speaker, plan, question_tokens)
            cutoff_k, cutoff_reason = self._adaptive_cutoff_k(scored, cap, plan.question_type)
            selected = self._select_diverse(scored, cutoff_k, plan.needs_diversity)
            selected, semantic_anchor_enforced, semantic_anchor_reason = self._enforce_semantic_anchors(
                selected,
                scored,
                cap,
                speaker,
                plan,
            )
            selected, type_min_enforced = self._enforce_type_minimum(selected, scored, cap, speaker, plan)
            selected, fallback_enforced, fallback_reason = self._enforce_confidence_fallback(
                selected,
                scored,
                cap,
                speaker,
                plan,
            )
            selected = sorted(
                selected,
                key=lambda item: (-float(item.get("qase_score") or 0.0), int(item.get("source_rank") or 0)),
            )
            selected_by_speaker[speaker] = [item["memory"] for item in selected]
            per_speaker_log[speaker] = {
                "input_count": len(candidates),
                "selected_count": len(selected),
                "cap": cap,
                "cutoff_k": cutoff_k,
                "reason": cutoff_reason,
                "semantic_anchor_enforced": semantic_anchor_enforced,
                "semantic_anchor_reason": semantic_anchor_reason,
                "min_type_enforced": type_min_enforced,
                "confidence_fallback_enforced": fallback_enforced,
                "confidence_fallback_reason": fallback_reason,
                "selected_ids": [item["id"] for item in selected],
                "score_preview": [round(float(item["qase_score"]), 4) for item in scored[: min(8, len(scored))]],
            }

        selected_ids = [
            str(memory.get("qase_candidate_id") or _memory_id(memory, f"{speaker}:selected:{idx}"))
            for speaker in plan.all_speakers
            for idx, memory in enumerate(selected_by_speaker.get(speaker) or [])
        ]
        selected_set = set(selected_ids)
        dropped_ids = [memory_id for memory_id in all_input_ids if memory_id not in selected_set]

        total_selected = sum(len(items) for items in selected_by_speaker.values())
        return selected_by_speaker, SelectionLog(
            question_type=plan.question_type.value,
            total_input_candidates=len(all_input_ids),
            total_selected=total_selected,
            selected_ids=selected_ids,
            dropped_ids=dropped_ids,
            per_speaker=per_speaker_log,
            max_total_final_memories=plan.max_total_final_memories,
            latency_sec=time.time() - start,
        )

    def select_global(
        self,
        candidates_by_speaker: Dict[str, Sequence[Dict[str, Any]]],
        plan: RetrievalPlan,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], SelectionLog]:
        start = time.time()
        question_tokens = _token_set(plan.question)
        selected_by_speaker: Dict[str, List[Dict[str, Any]]] = {speaker: [] for speaker in plan.all_speakers}
        per_speaker_log: Dict[str, Dict[str, Any]] = {}
        all_input_ids: List[str] = []
        all_scored: List[Dict[str, Any]] = []

        for speaker in plan.all_speakers:
            candidates = list(candidates_by_speaker.get(speaker) or [])
            all_input_ids.extend(_memory_id(item, f"{speaker}:{idx}") for idx, item in enumerate(candidates))
            scored = self._score_candidates(candidates, speaker, plan, question_tokens)
            all_scored.extend(scored)
            per_speaker_log[speaker] = {
                "input_count": len(candidates),
                "selected_count": 0,
                "cap": int(plan.final_top_k_cap_by_speaker.get(speaker) or 0),
                "selection_scope": "global",
            }

        total_cap = min(
            max(1, int(plan.max_total_final_memories)),
            sum(max(0, int(plan.final_top_k_cap_by_speaker.get(speaker) or 0)) for speaker in plan.all_speakers),
        )
        if total_cap <= 0 or not all_scored:
            return selected_by_speaker, SelectionLog(
                question_type=plan.question_type.value,
                total_input_candidates=len(all_input_ids),
                total_selected=0,
                selected_ids=[],
                dropped_ids=all_input_ids,
                per_speaker=per_speaker_log,
                max_total_final_memories=total_cap,
                latency_sec=time.time() - start,
            )

        all_scored.sort(
            key=lambda item: (
                -float(item.get("qase_score") or 0.0),
                int(item.get("source_rank") or 0),
                str(item.get("speaker") or ""),
            )
        )
        cutoff_k, cutoff_reason = self._adaptive_cutoff_k(all_scored, total_cap, plan.question_type)
        selected = self._select_diverse(all_scored, cutoff_k, plan.needs_diversity)
        selected, type_min_enforced = self._enforce_global_type_minimum(selected, all_scored, total_cap, plan)
        selected = sorted(
            selected,
            key=lambda item: (-float(item.get("qase_score") or 0.0), int(item.get("source_rank") or 0)),
        )

        for item in selected:
            speaker = str(item.get("speaker") or "")
            selected_by_speaker.setdefault(speaker, []).append(item["memory"])

        for speaker in plan.all_speakers:
            per_speaker_log[speaker].update(
                {
                    "selected_count": len(selected_by_speaker.get(speaker) or []),
                    "global_cap": total_cap,
                    "cutoff_k": cutoff_k,
                    "reason": cutoff_reason,
                    "min_type_enforced": type_min_enforced,
                    "score_preview": [
                        round(float(item["qase_score"]), 4)
                        for item in all_scored[: min(8, len(all_scored))]
                    ],
                }
            )

        selected_ids = [
            str(item["id"])
            for item in selected
        ]
        selected_set = set(selected_ids)
        dropped_ids = [memory_id for memory_id in all_input_ids if memory_id not in selected_set]

        return selected_by_speaker, SelectionLog(
            question_type=plan.question_type.value,
            total_input_candidates=len(all_input_ids),
            total_selected=len(selected),
            selected_ids=selected_ids,
            dropped_ids=dropped_ids,
            per_speaker=per_speaker_log,
            max_total_final_memories=total_cap,
            latency_sec=time.time() - start,
        )

    def _target_speakers(self, question: str, all_speakers: Sequence[str]) -> List[str]:
        q = question.lower()
        targets: List[str] = []
        for speaker in all_speakers:
            full = speaker.lower().strip()
            first = full.split()[0] if full else full
            aliases = {full, first}
            aliases.update(_SPEAKER_ALIASES.get(full, set()))
            aliases.update(_SPEAKER_ALIASES.get(first, set()))
            for alias in aliases:
                if not alias:
                    continue
                matched_alias = False
                for match in re.finditer(rf"\b{re.escape(alias)}\b", q):
                    if self._is_external_name_mention(question, match.end(), alias, first, full):
                        continue
                    matched_alias = True
                    break
                if matched_alias:
                    targets.append(speaker)
                    break
        out: List[str] = []
        for speaker in targets:
            if speaker not in out:
                out.append(speaker)
        return out

    def _is_external_name_mention(
        self,
        question: str,
        match_end: int,
        alias: str,
        first: str,
        full: str,
    ) -> bool:
        # If a one-token speaker alias is immediately followed by another
        # capitalized token, it is often an external entity/person name rather
        # than the LOCOMO speaker.
        if alias != first or full != first:
            return False
        suffix = question[match_end:]
        match = re.match(r"\s+([A-Z][A-Za-z]+)", suffix)
        if not match:
            return False
        return match.group(1).lower() not in {"and", "or"}

    def _classify(
        self,
        question: str,
        target_speakers: Sequence[str],
        matched: Dict[str, List[str]],
        cross_speaker_relation: bool = False,
    ) -> Tuple[QuestionType, str]:
        q = question.lower()
        has_multi_signal = (
            cross_speaker_relation
            or bool(matched["comparison"])
            or bool(matched["multi_hop"])
            or bool(matched["bounded_list"])
            or bool(matched["list_open"])
            or (
                bool(matched["preference_inference"])
                and self._is_counterfactual_or_likelihood_question(q)
            )
            or (len(target_speakers) >= 2 and not self._is_directional_two_speaker_question(q))
        )
        if has_multi_signal:
            return QuestionType.MULTI_HOP, "complex evidence signal"

        has_temporal_signal = (
            bool(matched["temporal_change"])
            or (bool(matched["temporal_fact"]) and self._is_temporal_fact_question(q))
        )
        if has_temporal_signal:
            return QuestionType.TEMPORAL, "time-related signal"

        if matched["profile"] or matched["preference_inference"]:
            return QuestionType.SINGLE_HOP, "profile/preference/background signal"
        if len(target_speakers) == 1:
            return QuestionType.SINGLE_HOP, "single target speaker and no complex signal"
        return QuestionType.FALLBACK_AMBIGUOUS, "no target speaker or strong signal; safe medium budget"

    def _is_temporal_fact_question(self, question: str) -> bool:
        q = question.lower().strip()
        if re.search(r"^(when|what year|what month|what date|what day|how long|for how long|how often)\b", q):
            return True
        if re.search(r"\b(in|on|during|before|after|since|by) \d{4}\b", q):
            return True
        if re.search(r"\b(before|after|since|during)\b", q):
            return True
        if re.search(r"\b(last|next|this) (year|month|week|summer|spring|winter|fall|autumn)\b", q):
            return True
        if re.search(r"\brecently\b", q):
            return True
        return any(re.search(rf"\b{month}\b", q) for month in _MONTHS)

    def _is_counterfactual_or_likelihood_question(self, question: str) -> bool:
        return bool(
            re.search(
                r"\b(would|likely|probably|maybe|might|could|should|if|considered|consider)\b",
                question.lower(),
            )
        )

    def _is_directional_two_speaker_question(self, question: str) -> bool:
        q = question.lower()
        if re.search(r"\b(both|together|in common|similar|same|different|difference|compare|alike)\b", q):
            return False
        if re.search(r"\bshared\b", q) and not re.search(r"\bshare(?:d)?\b.{0,40}\bwith\b", q):
            return False
        return bool(
            re.search(
                r"\b(recommend(?:ed)?|suggest(?:ed)?|tell|told|ask(?:ed)?|give|gave|send|sent|show(?:ed)?|share(?:d)?|"
                r"say|said|mention(?:ed)?|advise(?:d)?|invite(?:d)?|help(?:ed)?|support(?:ed)?)\b",
                q,
            )
        )

    def _is_bounded_list_question(self, question: str) -> bool:
        q = question.lower().strip()
        if re.search(r"^how many\b", q):
            return True
        if re.search(r"^where\s+(?:has|have|did|does|is|are|was|were|can|could|would)\b", q):
            return True

        def has_plural_head(head_text: str) -> bool:
            head = re.sub(r"\b[a-z]+'s\b", " ", head_text)
            head = re.split(r"\b(?:for|about|with|from|to|in|on|during|after|before)\b", head, maxsplit=1)[0]
            head = re.sub(r"\b(?:the|a|an|any|some|all|of|his|her|their|its|this|that|these|those)\b", " ", head)
            head = re.sub(r"\s+", " ", head).strip()
            if not head:
                return False
            head_tokens = [token for token in re.findall(r"[a-z]+", head) if token not in _STOPWORDS]
            if not head_tokens:
                return False
            last = head_tokens[-1]
            if last.endswith("ss") or last in {"is", "was", "does"}:
                return False
            return last.endswith("s")

        post_aux_match = re.match(r"^what\s+(?:are|were)\s+(?!some\b)(?P<head>.+)$", q)
        if post_aux_match and has_plural_head(post_aux_match.group("head")):
            return True

        match = re.match(
            r"^(what|which)\s+(?!kind\b|type\b)(?P<head>.+?)\s+"
            r"(?:has|have|had|did|does|do|is|are|was|were|can|could|would|should)\b",
            q,
        )
        if not match:
            return False
        return has_plural_head(match.group("head"))

    def _budget_for_type(
        self,
        question_type: QuestionType,
        target_speakers: Sequence[str],
        all_speakers: Sequence[str],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        cfg = self.config
        candidate = {speaker: 0 for speaker in all_speakers}
        final = {speaker: 0 for speaker in all_speakers}

        def scaled(value: int, scale: float) -> int:
            return max(0, int(math.ceil(value * scale)))

        def maybe_balance(
            candidate_budget: Dict[str, int],
            final_budget: Dict[str, int],
        ) -> Tuple[Dict[str, int], Dict[str, int]]:
            if self.config.budget_allocation_mode != "balanced_per_speaker":
                return candidate_budget, final_budget
            return (
                self._balanced_budget(candidate_budget, all_speakers),
                self._balanced_budget(final_budget, all_speakers),
            )

        if question_type == QuestionType.FALLBACK_AMBIGUOUS or not target_speakers:
            for speaker in all_speakers:
                candidate[speaker] = scaled(cfg.candidate_fallback_each, cfg.candidate_pool_scale)
                final[speaker] = scaled(cfg.final_fallback_each, cfg.final_k_scale)
            return maybe_balance(candidate, final)

        mapping = {
            QuestionType.SINGLE_HOP: (
                cfg.candidate_single_hop_target,
                cfg.candidate_single_hop_other,
                cfg.final_single_hop_target,
                cfg.final_single_hop_other,
            ),
            QuestionType.TEMPORAL: (
                cfg.candidate_temporal_target,
                cfg.candidate_temporal_other,
                cfg.final_temporal_target,
                cfg.final_temporal_other,
            ),
            QuestionType.MULTI_HOP: (
                cfg.candidate_multi_hop_target,
                cfg.candidate_multi_hop_other,
                cfg.final_multi_hop_target,
                cfg.final_multi_hop_other,
            ),
        }
        target_candidate, other_candidate, target_final, other_final = mapping.get(
            question_type,
            (
                cfg.candidate_fallback_each,
                cfg.candidate_fallback_each,
                cfg.final_fallback_each,
                cfg.final_fallback_each,
            ),
        )
        targets = set(target_speakers)
        for speaker in all_speakers:
            if speaker in targets:
                candidate[speaker] = scaled(target_candidate, cfg.candidate_pool_scale)
                final[speaker] = scaled(target_final, cfg.final_k_scale)
            else:
                candidate[speaker] = scaled(other_candidate, cfg.candidate_pool_scale)
                final[speaker] = scaled(other_final, cfg.final_k_scale)
        return maybe_balance(candidate, final)

    def _balanced_budget(
        self,
        budget_by_speaker: Dict[str, int],
        all_speakers: Sequence[str],
    ) -> Dict[str, int]:
        speakers = list(all_speakers)
        if not speakers:
            return {}
        total = sum(max(0, int(budget_by_speaker.get(speaker) or 0)) for speaker in speakers)
        per_speaker = int(math.ceil(total / len(speakers)))
        return {speaker: per_speaker for speaker in speakers}

    def _cap_final_budget(
        self,
        final_budget: Dict[str, int],
        question_type: QuestionType,
        target_speakers: Sequence[str],
        all_speakers: Sequence[str],
    ) -> Dict[str, int]:
        capped = {speaker: max(0, int(final_budget.get(speaker) or 0)) for speaker in all_speakers}
        max_total = max(1, int(self.config.max_total_final_memories))
        if sum(capped.values()) <= max_total:
            return capped

        if question_type == QuestionType.FALLBACK_AMBIGUOUS:
            order = list(all_speakers)
        else:
            targets = [speaker for speaker in target_speakers if speaker in all_speakers]
            order = targets + [speaker for speaker in all_speakers if speaker not in targets]
            if not order:
                order = list(all_speakers)

        out = {speaker: 0 for speaker in all_speakers}
        while sum(out.values()) < max_total:
            progressed = False
            for speaker in order:
                if out[speaker] >= capped.get(speaker, 0):
                    continue
                out[speaker] += 1
                progressed = True
                if sum(out.values()) >= max_total:
                    break
            if not progressed:
                break
        return out

    def _complexity_score(
        self,
        question: str,
        question_type: QuestionType,
        target_speakers: Sequence[str],
        matched: Dict[str, List[str]],
    ) -> float:
        base = {
            QuestionType.SINGLE_HOP: 0.20,
            QuestionType.TEMPORAL: 0.45,
            QuestionType.MULTI_HOP: 0.78,
            QuestionType.FALLBACK_AMBIGUOUS: 0.50,
        }[question_type]
        signal_bonus = min(sum(len(items) for items in matched.values()) * 0.025, 0.12)
        speaker_bonus = 0.08 if len(target_speakers) >= 2 else 0.0
        length_bonus = min(len(question.split()) / 100.0, 0.10)
        return min(1.0, base + signal_bonus + speaker_bonus + length_bonus)

    def _score_candidates(
        self,
        candidates: Sequence[Dict[str, Any]],
        speaker: str,
        plan: RetrievalPlan,
        question_tokens: set[str],
    ) -> List[Dict[str, Any]]:
        scored = []
        target_set = set(plan.target_speakers)
        for rank, memory in enumerate(candidates):
            text = _memory_text(memory)
            memory_tokens = _token_set(text)
            semantic = _safe_float(memory.get("score"))
            if semantic is None:
                semantic_component = max(0.0, 1.0 - rank * 0.03)
            elif self.config.score_mode == "distance":
                semantic_component = 1.0 / (1.0 + max(semantic, 0.0))
            else:
                semantic_component = semantic
            lexical_score = _safe_float(memory.get("qase_bm25_score"))
            if lexical_score is None:
                lexical_score = _jaccard(question_tokens, memory_tokens)
            lexical_score = max(0.0, min(1.0, lexical_score))
            speaker_bonus = 0.0
            if target_set:
                if speaker in target_set:
                    speaker_bonus = 0.10
            has_time_text_signal = _memory_has_time_text(memory)
            has_time_metadata_signal = _memory_has_time_metadata(memory)
            has_time_signal = has_time_text_signal or has_time_metadata_signal
            temporal_bonus = (
                self.config.temporal_bonus_weight
                if plan.needs_time and has_time_signal
                else 0.0
            )
            qase_score = (
                semantic_component
                + lexical_score * self.config.lexical_weight
                + speaker_bonus
                + temporal_bonus
            )
            candidate_id = _memory_id(memory, f"{speaker}:{rank}")
            copied = dict(memory)
            copied["qase_candidate_id"] = candidate_id
            copied["qase_score"] = qase_score
            copied["qase_semantic_score"] = semantic_component
            copied["qase_lexical_score"] = lexical_score
            copied["qase_overlap_score"] = lexical_score
            copied["qase_speaker_bonus"] = speaker_bonus
            copied["qase_temporal_bonus"] = temporal_bonus
            copied["qase_has_time_signal"] = has_time_signal
            copied["qase_has_time_text_signal"] = has_time_text_signal
            copied["qase_has_time_metadata_signal"] = has_time_metadata_signal
            copied["qase_source_rank"] = rank
            copied["qase_question_type"] = plan.question_type.value
            copied["qase_expanded"] = False
            copied["qase_score_details"] = {
                "semantic": semantic_component,
                "lexical": lexical_score,
                "lexical_weight": self.config.lexical_weight,
                "speaker_bonus": speaker_bonus,
                "time_bonus": temporal_bonus,
                "final_score": qase_score,
                "expanded": False,
            }
            scored.append(
                {
                    "id": candidate_id,
                    "memory": copied,
                    "speaker": speaker,
                    "text": text,
                    "tokens": memory_tokens,
                    "qase_score": qase_score,
                    "source_rank": rank,
                }
            )
        scored.sort(key=lambda item: (-float(item["qase_score"]), int(item["source_rank"])))
        return scored

    def _adaptive_cutoff_k(
        self,
        scored: Sequence[Dict[str, Any]],
        cap: int,
        question_type: QuestionType,
    ) -> Tuple[int, str]:
        if not scored or cap <= 0:
            return 0, "empty scored candidates or zero cap"
        n = min(len(scored), cap)
        if not self.config.adaptive_cutoff_enabled:
            return n, "adaptive cutoff disabled"
        scores = [float(item["qase_score"]) for item in scored[:n]]
        if len(scores) < 3:
            return n, "too few candidates for adaptive cutoff"
        is_complex = question_type in {QuestionType.MULTI_HOP, QuestionType.FALLBACK_AMBIGUOUS}
        drop = self.config.adaptive_drop_complex if is_complex else self.config.adaptive_drop_simple
        use_temporal_largest_gap = (
            self.config.adaptive_cutoff_mode == "temporal_largest_gap"
            and question_type == QuestionType.TEMPORAL
        )
        use_largest_gap = self.config.adaptive_cutoff_mode == "largest_gap"
        if use_temporal_largest_gap:
            gaps = [scores[idx] - scores[idx + 1] for idx in range(len(scores) - 1)]
            largest_gap = max(gaps, default=0.0)
            threshold_k = gaps.index(largest_gap) + 1 if gaps else n
            reason = f"temporal largest gap {largest_gap:.2f}"
        elif use_largest_gap:
            gaps = [scores[idx] - scores[idx + 1] for idx in range(len(scores) - 1)]
            largest_gap = max(gaps, default=0.0)
            if largest_gap >= drop:
                threshold_k = gaps.index(largest_gap) + 1
                reason = f"largest gap {largest_gap:.2f} >= {drop:.2f}"
            else:
                threshold_k = n
                reason = f"no gap >= {drop:.2f}"
        else:
            best = scores[0]
            threshold_k = 0
            for score in scores:
                if score >= best - drop:
                    threshold_k += 1
                else:
                    break
            reason = f"score cutoff best-{drop:.2f}"
        threshold_k = max(1, threshold_k)
        if is_complex:
            min_k = max(1, int(math.ceil(cap * self.config.complex_min_fraction_of_cap)))
            chosen = max(threshold_k, min_k)
        else:
            chosen = threshold_k
        return min(chosen, n), reason

    def _select_diverse(
        self,
        scored: Sequence[Dict[str, Any]],
        k: int,
        needs_diversity: bool,
    ) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        if not needs_diversity or len(scored) <= k:
            return list(scored[:k])
        diversity_lambda = self.config.diversity_lambda_complex
        selected: List[Dict[str, Any]] = []
        remaining = list(scored)
        while remaining and len(selected) < k:
            best_index = 0
            best_score = -float("inf")
            for index, item in enumerate(remaining):
                max_similarity = max((_jaccard(item["tokens"], selected_item["tokens"]) for selected_item in selected), default=0.0)
                adjusted = float(item["qase_score"]) - diversity_lambda * max_similarity
                if adjusted > best_score:
                    best_score = adjusted
                    best_index = index
            selected.append(remaining.pop(best_index))
        return selected

    def _insert_or_replace(
        self,
        selected_list: List[Dict[str, Any]],
        candidate: Dict[str, Any],
        cap: int,
        replace_filter: Optional[Any] = None,
    ) -> bool:
        candidate_id = str(candidate["id"])
        if any(str(item["id"]) == candidate_id for item in selected_list):
            return False
        if len(selected_list) < cap:
            selected_list.append(candidate)
            return True

        replaceable_indices = [
            index
            for index, item in enumerate(selected_list)
            if replace_filter is None or replace_filter(item)
        ]
        if not replaceable_indices:
            return False

        worst_index = min(
            replaceable_indices,
            key=lambda index: (
                float(selected_list[index].get("qase_score") or 0.0),
                -int(selected_list[index].get("source_rank") or 0),
            ),
        )
        selected_list[worst_index] = candidate
        return True

    def _enforce_semantic_anchors(
        self,
        selected: Sequence[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        cap: int,
        speaker: str,
        plan: RetrievalPlan,
    ) -> Tuple[List[Dict[str, Any]], bool, str]:
        selected_list = list(selected)
        if not self.config.semantic_anchor_enabled:
            return selected_list, False, "disabled"
        if cap <= 0 or not candidates:
            return selected_list, False, "empty candidate list or zero cap"

        target_set = set(plan.target_speakers)
        if target_set and speaker not in target_set and not plan.needs_cross_speaker:
            return selected_list, False, "non-target speaker"

        if plan.question_type in {QuestionType.SINGLE_HOP, QuestionType.TEMPORAL}:
            anchor_count = self.config.semantic_anchor_simple
        else:
            anchor_count = self.config.semantic_anchor_complex
        anchor_count = min(cap, len(candidates), max(0, int(anchor_count)))
        if anchor_count <= 0:
            return selected_list, False, "zero anchors"

        anchors = sorted(candidates, key=lambda item: int(item.get("source_rank") or 0))[:anchor_count]
        before_ids = {str(item["id"]) for item in selected_list}
        for anchor in anchors:
            self._insert_or_replace(selected_list, anchor, cap)
        changed = before_ids != {str(item["id"]) for item in selected_list}
        return selected_list, changed, f"top_{anchor_count}_semantic_anchor"

    def _enforce_type_minimum(
        self,
        selected: Sequence[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        cap: int,
        speaker: str,
        plan: RetrievalPlan,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        selected_list = list(selected)
        if cap <= 0 or not candidates:
            return selected_list, False

        target_set = set(plan.target_speakers)
        is_target = not target_set or speaker in target_set
        if not is_target:
            return selected_list, False

        min_by_type = {
            QuestionType.SINGLE_HOP: self.config.min_single_hop,
            QuestionType.TEMPORAL: self.config.min_temporal,
            QuestionType.MULTI_HOP: self.config.min_multi_hop,
            QuestionType.FALLBACK_AMBIGUOUS: self.config.min_fallback_ambiguous,
        }
        min_required = min_by_type.get(plan.question_type)
        if min_required is None:
            return selected_list, False

        min_required = min(cap, max(0, int(min_required)), len(candidates))
        if len(selected_list) >= min_required:
            return selected_list, False

        selected_ids = {str(item["id"]) for item in selected_list}
        for item in candidates:
            item_id = str(item["id"])
            if item_id in selected_ids:
                continue
            selected_list.append(item)
            selected_ids.add(item_id)
            if len(selected_list) >= min_required:
                break
        return selected_list, len(selected_list) > len(selected)

    def _enforce_global_type_minimum(
        self,
        selected: Sequence[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        cap: int,
        plan: RetrievalPlan,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        selected_list = list(selected)
        if cap <= 0 or not candidates:
            return selected_list, False

        min_by_type = {
            QuestionType.SINGLE_HOP: self.config.min_single_hop,
            QuestionType.TEMPORAL: self.config.min_temporal,
            QuestionType.MULTI_HOP: self.config.min_multi_hop,
            QuestionType.FALLBACK_AMBIGUOUS: self.config.min_fallback_ambiguous,
        }
        min_required = min_by_type.get(plan.question_type)
        if min_required is None:
            return selected_list, False

        min_required = min(cap, max(0, int(min_required)), len(candidates))
        if len(selected_list) >= min_required:
            return selected_list, False

        selected_ids = {str(item["id"]) for item in selected_list}
        for item in candidates:
            item_id = str(item["id"])
            if item_id in selected_ids:
                continue
            selected_list.append(item)
            selected_ids.add(item_id)
            if len(selected_list) >= min_required:
                break
        return selected_list, len(selected_list) > len(selected)

    def _enforce_confidence_fallback(
        self,
        selected: Sequence[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        cap: int,
        speaker: str,
        plan: RetrievalPlan,
    ) -> Tuple[List[Dict[str, Any]], bool, str]:
        selected_list = list(selected)
        if not self.config.confidence_fallback_enabled:
            return selected_list, False, "disabled"
        if cap <= 0 or not candidates or len(selected_list) >= cap:
            return selected_list, False, "not needed"

        high_risk_types = {QuestionType.MULTI_HOP, QuestionType.FALLBACK_AMBIGUOUS}
        if plan.question_type not in high_risk_types:
            return selected_list, False, "simple question type"

        target_set = set(plan.target_speakers)
        is_target_or_cross = not target_set or speaker in target_set or plan.needs_cross_speaker
        if not is_target_or_cross:
            return selected_list, False, "non-target speaker"

        fraction = (
            self.config.confidence_fallback_min_fraction_complex
            if plan.question_type in high_risk_types
            else self.config.confidence_fallback_min_fraction_simple
        )
        desired = int(math.ceil(cap * max(0.0, min(fraction, 1.0))))
        selected_scores = [float(item.get("qase_score") or 0.0) for item in selected_list]
        top_score = max(selected_scores, default=0.0)
        reasons: List[str] = []
        if len(selected_list) < desired:
            reasons.append(f"selected<{desired}")
        if top_score < self.config.confidence_fallback_low_score_threshold:
            desired = cap
            reasons.append(f"top_score<{self.config.confidence_fallback_low_score_threshold:.2f}")
        if not reasons:
            return selected_list, False, "confidence sufficient"

        desired = min(cap, len(candidates), max(desired, len(selected_list)))
        selected_ids = {str(item["id"]) for item in selected_list}
        before_count = len(selected_list)
        for item in candidates:
            item_id = str(item["id"])
            if item_id in selected_ids:
                continue
            selected_list.append(item)
            selected_ids.add(item_id)
            if len(selected_list) >= desired:
                break
        return selected_list, len(selected_list) > before_count, ", ".join(reasons)

def build_qase_trace(plan: RetrievalPlan, selection_log: SelectionLog) -> Dict[str, Any]:
    return {
        "mode": "question_aware_selective",
        "plan": plan.to_dict(),
        "selection_log": selection_log.to_dict(),
    }
