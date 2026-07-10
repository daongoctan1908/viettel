import argparse
import inspect
import json
import math
import os
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent
for path in (PROJECT_ROOT, EVALUATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dotenv import load_dotenv
from jinja2 import Template
from mem0 import Memory
from openai import OpenAI
import tiktoken

try:
    from prompts import ANSWER_PROMPT
    from paper_update_memory import (
        LocalPaperMemoryStore,
        decide_memory_operation,
        extract_candidate_memories,
        verify_update_information_content,
    )
    from mem0_question_aware_budget import (
        QASEConfig,
        QuestionAwareSelectiveEvidence,
        build_qase_trace,
    )
except ImportError:
    from evaluation.prompts import ANSWER_PROMPT
    from evaluation.paper_update_memory import (
        LocalPaperMemoryStore,
        decide_memory_operation,
        extract_candidate_memories,
        verify_update_information_content,
    )
    from evaluation.mem0_question_aware_budget import (
        QASEConfig,
        QuestionAwareSelectiveEvidence,
        build_qase_trace,
    )

try:
    from mragent_reconstruction import (
        Mem0ActiveMemoryIndex,
        active_reconstruct_memories,
        add_active_usage_totals,
        empty_active_usage_totals,
        evidence_gap_retrieve_memories,
        hybrid_adaptive_retrieve_memories,
    )
except ImportError:
    try:
        from evaluation.mragent_reconstruction import (
            Mem0ActiveMemoryIndex,
            active_reconstruct_memories,
            add_active_usage_totals,
            empty_active_usage_totals,
            evidence_gap_retrieve_memories,
            hybrid_adaptive_retrieve_memories,
        )
    except ImportError:
        Mem0ActiveMemoryIndex = None
        active_reconstruct_memories = None
        evidence_gap_retrieve_memories = None
        hybrid_adaptive_retrieve_memories = None

        def empty_active_usage_totals() -> Dict[str, Any]:
            prefixes = (
                "active_router",
                "evidence_rerank",
                "sufficiency",
                "egc_contract",
                "egc_coverage",
                "egc_proof_pack",
            )
            totals: Dict[str, Any] = {}
            for prefix in prefixes:
                for field in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
                    totals[f"{prefix}_{field}"] = 0
                totals[f"{prefix}_latency_sec"] = 0.0
                totals[f"{prefix}_calls"] = 0
                totals[f"{prefix}_parse_errors"] = 0
            return totals

        def add_active_usage_totals(totals: Dict[str, Any], log: Dict[str, Any]) -> None:
            for key, default_value in empty_active_usage_totals().items():
                totals.setdefault(key, default_value)
            for key, value in (log or {}).items():
                if key in totals:
                    totals[key] += float(value or 0.0) if key.endswith("_latency_sec") else int(value or 0)

TARGET_ONLY_CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Exact event dates when the messages establish them
   - Locations, family and relationship facts, lists, preferences, identity/persona facts, and temporal facts

2. If messages contain [TARGET MESSAGE - EXTRACT MEMORY FROM THIS MESSAGE], extract memories only from those target messages.
   Use [CONTEXT ONLY - DO NOT EXTRACT MEMORY FROM THIS MESSAGE] messages only for disambiguation, speaker grounding, and resolving relative time references.

3. If target/context markers are absent, extract memories only from user messages in the provided target batch, not from assistant responses.

4. Resolve relative time references using the session timestamp in the message content whenever possible.
   Treat the session timestamp as the observation time, not automatically as the event time. Include an exact event
   date only when the message establishes it directly or through an unambiguous relative expression.

5. Do not invent dates, names, locations, or relationships. Preserve uncertainty instead of fabricating a detail.

6. Capture all salient facts, including meaningful one-off events, while skipping greetings, filler, duplicates, and
   details with no future informational value. Make memories specific and self-contained rather than verbose.
""".strip()


GENERAL_CUSTOM_INSTRUCTIONS = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Exact event dates when the messages establish them
   - Locations, family and relationship facts, lists, preferences, identity/persona facts, and temporal facts

2. Extract memories only from user messages in the provided target batch, not from assistant responses.

3. Resolve relative time references using the session timestamp in the message content whenever possible.
   Treat the session timestamp as the observation time, not automatically as the event time. Include an exact event
   date only when the message establishes it directly or through an unambiguous relative expression.

4. Do not invent dates, names, locations, or relationships. Preserve uncertainty instead of fabricating a detail.

5. Capture all salient facts, including meaningful one-off events, while skipping greetings, filler, duplicates, and
   details with no future informational value. Make memories specific and self-contained rather than verbose.
""".strip()


LOCOMO_DATETIME_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M %p on %d %b, %Y",
)
LOCOMO_TIMEZONE_ASSUMPTION = "UTC"
CHECKPOINT_SCHEMA_VERSION = 1
RESUME_CONFIG_IGNORED_FIELDS = {
    "api_max_retries",
    "api_timeout_sec",
    "checkpoint_path",
    "output",
    "resume_checkpoint",
    "reuse_memory_checkpoint",
    "clear_reuse_embedding_cache",
}
HYBRID_ADAPTIVE_CONFIG_FIELDS = {
    "hybrid_router_mode",
    "protected_semantic_top_k",
    "active_extra_k_per_speaker",
    "har_active_seed_top_k",
    "har_evidence_rerank_mode",
    "har_evidence_rerank_trigger",
    "har_evidence_rerank_max_candidates",
    "har_evidence_rerank_max_tokens",
    "har_sufficiency_mode",
    "har_sufficiency_max_rounds",
    "har_sufficiency_max_candidates",
    "har_sufficiency_max_tokens",
}
HYBRID_ADAPTIVE_CONFIG_DEFAULTS = {
    "har_sufficiency_mode": "diagnostic",
    "har_sufficiency_max_rounds": 0,
    "har_sufficiency_max_candidates": 40,
    "har_sufficiency_max_tokens": 800,
}
EVIDENCE_GAP_CONFIG_FIELDS = {
    "egc_contract_mode",
    "egc_coverage_mode",
    "egc_max_gap_rounds",
    "egc_max_slots",
    "egc_min_slot_coverage",
    "egc_gap_top_k_per_slot",
    "egc_max_gap_evidence_per_speaker",
    "egc_proof_pack_rerank",
    "egc_keep_protected_baseline",
    "egc_contract_max_tokens",
    "egc_coverage_max_candidates",
    "egc_coverage_max_tokens",
    "egc_proof_pack_max_candidates",
    "egc_proof_pack_max_tokens",
    "egc_max_tool_calls_per_round",
}
QUESTION_AWARE_SELECTIVE_CONFIG_FIELDS = {
    "qase_max_total_final_memories",
    "qase_candidate_pool_scale",
    "qase_final_k_scale",
    "qase_budget_profile",
    "qase_budget_allocation_mode",
    "qase_selection_scope",
    "qase_adaptive_cutoff_mode",
    "qase_bm25_candidate_rerank",
    "qase_bm25_k1",
    "qase_bm25_b",
    "qase_lexical_weight",
    "qase_hybrid_semantic_weight",
    "qase_hybrid_bm25_weight",
    "qase_hybrid_entity_weight",
    "qase_hybrid_temporal_weight",
    "qase_disable_adaptive_cutoff",
    "qase_disable_diversity",
    "qase_diversity_lambda",
    "qase_disable_confidence_fallback",
    "qase_confidence_fallback_low_score_threshold",
    "qase_planner_model_root",
    "qase_question_type_model_dir",
    "qase_target_speaker_model_dir",
    "qase_planner_model_device",
}
QUESTION_AWARE_SELECTIVE_CONFIG_DEFAULTS = {
    "qase_disable_confidence_fallback": False,
    "qase_confidence_fallback_low_score_threshold": 0.62,
    "qase_hybrid_semantic_weight": 0.60,
    "qase_hybrid_bm25_weight": 0.20,
    "qase_hybrid_entity_weight": 0.15,
    "qase_hybrid_temporal_weight": 0.05,
    "qase_budget_profile": "stable_v2",
    "qase_budget_allocation_mode": "speaker_aware",
    "qase_selection_scope": "per_speaker",
    "qase_adaptive_cutoff_mode": None,
    "qase_bm25_candidate_rerank": False,
    "qase_bm25_k1": 1.5,
    "qase_bm25_b": 0.75,
    "qase_lexical_weight": 0.30,
    "qase_planner_model_root": None,
    "qase_question_type_model_dir": None,
    "qase_target_speaker_model_dir": None,
    "qase_planner_model_device": -1,
}
QUESTION_AWARE_RETRIEVAL_MODES = {
    "question_aware_selective",
    "question_aware_semantic",
    "question_aware_semantic_adaptive",
    "question_aware_hybrid_adaptive",
}

ACTIVE_RECONSTRUCTION_ONLY_CONFIG_FIELDS = {
    "active_reconstruction_seed_top_k",
}
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run local OSS Mem0 on LOCOMO with a paper-style protocol: per-speaker memories, "
            "small message batches, timestamp metadata, dual-speaker search, and LOCOMO answer prompt."
        )
    )
    parser.add_argument("--dataset", default="evaluation/dataset/locomo10.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--method", default="mem0_local_paper")

    parser.add_argument("--llm_model", default=None)
    parser.add_argument("--embedding_model", default=None)
    parser.add_argument(
        "--memory_update_mode",
        choices=["sdk_additive", "paper_update_wrapper"],
        default="sdk_additive",
    )
    parser.add_argument("--update_similar_top_k", type=int, default=10)
    parser.add_argument("--candidate_extraction_max_tokens", type=int, default=None)
    parser.add_argument("--update_decision_max_tokens", type=int, default=None)
    parser.add_argument(
        "--memory_wrapper_store",
        choices=["mem0_sdk", "qdrant_direct", "json_local"],
        default="json_local",
    )
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--search_threshold", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument("--retrieved_memory_token_encoding", default="cl100k_base")
    parser.add_argument("--retrieval_token_budget", type=int, default=None)
    parser.add_argument(
        "--retrieval_budget_strategy",
        choices=["interleave_rank", "balanced_per_speaker", "global_score"],
        default="interleave_rank",
    )
    parser.add_argument(
        "--qa_retrieval_mode",
        choices=[
            "semantic_topk",
            "active_reconstruction",
            "hybrid_adaptive",
            "evidence_gap_hybrid",
            "question_aware_selective",
            "question_aware_semantic",
            "question_aware_semantic_adaptive",
            "question_aware_hybrid_adaptive",
        ],
        default="semantic_topk",
        help=(
            "semantic_topk keeps the local Mem0 baseline. active_reconstruction adds a "
            "MRAgent-inspired Cue/Tag/Topic/Time retrieval loop over the final Mem0 store. "
            "hybrid_adaptive keeps semantic top-k protected and only adds adaptive evidence for complex queries. "
            "evidence_gap_hybrid builds evidence slots, checks baseline coverage, then retrieves missing slots. "
            "question_aware_selective adjusts retrieval budget by question type and selects a compact evidence pack. "
            "question_aware_semantic uses the same question-aware budget but keeps pure semantic top-k without "
            "evidence rescoring or selection. question_aware_semantic_adaptive retrieves a larger semantic "
            "candidate pool, then applies a semantic-score cutoff under the question-aware final budget. "
            "question_aware_hybrid_adaptive rescores the same candidate pool with semantic+BM25+entity+time "
            "fusion before adaptive-k cutoff."
        ),
    )
    parser.add_argument("--active_reconstruction_max_steps", type=int, default=5)
    parser.add_argument("--active_reconstruction_max_tool_calls_per_step", type=int, default=10)
    parser.add_argument("--active_reconstruction_seed_top_k", type=int, default=5)
    parser.add_argument("--active_reconstruction_use_llm_router", action="store_true", default=False)
    parser.add_argument("--active_reconstruction_router_max_tokens", type=int, default=800)
    parser.add_argument("--hybrid_router_mode", choices=["heuristic", "hybrid_llm"], default="heuristic")
    parser.add_argument("--protected_semantic_top_k", type=int, default=15)
    parser.add_argument("--active_extra_k_per_speaker", type=int, default=5)
    parser.add_argument("--har_active_seed_top_k", type=int, default=5)
    parser.add_argument("--har_evidence_rerank_mode", choices=["off", "llm"], default="off")
    parser.add_argument("--har_evidence_rerank_trigger", choices=["adaptive_only", "all"], default="adaptive_only")
    parser.add_argument("--har_evidence_rerank_max_candidates", type=int, default=40)
    parser.add_argument("--har_evidence_rerank_max_tokens", type=int, default=1000)
    parser.add_argument("--har_sufficiency_mode", choices=["diagnostic", "heuristic", "llm"], default="diagnostic")
    parser.add_argument("--har_sufficiency_max_rounds", type=int, default=0)
    parser.add_argument("--har_sufficiency_max_candidates", type=int, default=40)
    parser.add_argument("--har_sufficiency_max_tokens", type=int, default=800)
    parser.add_argument("--egc_contract_mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--egc_coverage_mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--egc_max_gap_rounds", type=int, default=2)
    parser.add_argument("--egc_max_slots", type=int, default=4)
    parser.add_argument("--egc_min_slot_coverage", type=float, default=0.65)
    parser.add_argument("--egc_gap_top_k_per_slot", type=int, default=4)
    parser.add_argument("--egc_max_gap_evidence_per_speaker", type=int, default=5)
    parser.add_argument("--egc_proof_pack_rerank", choices=["off", "llm"], default="off")
    parser.add_argument("--egc_keep_protected_baseline", action="store_true", default=True)
    parser.add_argument("--no_egc_keep_protected_baseline", dest="egc_keep_protected_baseline", action="store_false")
    parser.add_argument("--egc_contract_max_tokens", type=int, default=900)
    parser.add_argument("--egc_coverage_max_candidates", type=int, default=40)
    parser.add_argument("--egc_coverage_max_tokens", type=int, default=900)
    parser.add_argument("--egc_proof_pack_max_candidates", type=int, default=40)
    parser.add_argument("--egc_proof_pack_max_tokens", type=int, default=1000)
    parser.add_argument("--egc_max_tool_calls_per_round", type=int, default=12)
    parser.add_argument("--qase_max_total_final_memories", type=int, default=22)
    parser.add_argument("--qase_candidate_pool_scale", type=float, default=1.0)
    parser.add_argument("--qase_final_k_scale", type=float, default=1.0)
    parser.add_argument(
        "--qase_budget_profile",
        choices=["stable_v2", "simple_report"],
        default="stable_v2",
        help=(
            "stable_v2 keeps the original subtype-specific QASE budgets. simple_report collapses nearby "
            "subtypes into shared budgets for cleaner ablations/reporting."
        ),
    )
    parser.add_argument(
        "--qase_budget_allocation_mode",
        choices=["speaker_aware", "balanced_per_speaker"],
        default="speaker_aware",
        help=(
            "speaker_aware keeps target/other speaker budgets. balanced_per_speaker converts each "
            "question-type budget into an equal per-speaker cap as an ablation."
        ),
    )
    parser.add_argument(
        "--qase_selection_scope",
        choices=["per_speaker", "global"],
        default="per_speaker",
        help=(
            "per_speaker runs Evidence Selector independently for each speaker namespace. global merges "
            "all speaker candidates, then runs one Evidence Selector over the combined pool."
        ),
    )
    parser.add_argument(
        "--qase_adaptive_cutoff_mode",
        choices=["best_delta", "largest_gap", "temporal_largest_gap"],
        default=None,
        help=(
            "Overrides the QASE evidence cutoff rule. best_delta keeps candidates within a score drop "
            "from the top item; largest_gap cuts after the largest adjacent score gap; "
            "temporal_largest_gap always cuts temporal questions after their largest adjacent score gap "
            "and uses best_delta otherwise."
        ),
    )
    parser.add_argument(
        "--qase_bm25_candidate_rerank",
        action="store_true",
        help=(
            "Build a per-speaker BM25 index over the existing local memory store during QA and use BM25 "
            "only as a lexical reranking feature for semantic QASE candidates. This does not re-ingest "
            "memories and does not add BM25-only candidates."
        ),
    )
    parser.add_argument("--qase_bm25_k1", type=float, default=1.5)
    parser.add_argument("--qase_bm25_b", type=float, default=0.75)
    parser.add_argument(
        "--qase_lexical_weight",
        type=float,
        default=0.30,
        help="Weight applied to QASE lexical score S_lex in the rule-based selector.",
    )
    parser.add_argument(
        "--qase_planner_model_root",
        default=None,
        help=(
            "Optional root folder containing qase_question_type_distilroberta and "
            "qase_target_speaker_distilroberta. When provided, QASE uses these classifiers "
            "for question type and target-speaker planning instead of heuristic planner labels."
        ),
    )
    parser.add_argument(
        "--qase_question_type_model_dir",
        default=None,
        help="Optional direct path to the fine-tuned question-type classifier folder.",
    )
    parser.add_argument(
        "--qase_target_speaker_model_dir",
        default=None,
        help="Optional direct path to the fine-tuned target-speaker classifier folder.",
    )
    parser.add_argument(
        "--qase_planner_model_device",
        type=int,
        default=-1,
        help="Transformers pipeline device for QASE planner classifiers: -1 for CPU, 0 for first GPU.",
    )
    parser.add_argument(
        "--qase_adaptive_budget_profile",
        choices=["balanced", "adaptive_k_only"],
        default="balanced",
        help=(
            "For question_aware_semantic_adaptive and question_aware_hybrid_adaptive: balanced applies "
            "type-specific floor/cap guardrails; adaptive_k_only lets Adaptive-k choose final k from the "
            "retrieved candidate pool."
        ),
    )
    parser.add_argument("--qase_hybrid_semantic_weight", type=float, default=0.60)
    parser.add_argument("--qase_hybrid_bm25_weight", type=float, default=0.20)
    parser.add_argument("--qase_hybrid_entity_weight", type=float, default=0.15)
    parser.add_argument("--qase_hybrid_temporal_weight", type=float, default=0.05)
    parser.add_argument("--qase_disable_adaptive_cutoff", action="store_true", default=False)
    parser.add_argument("--qase_disable_diversity", action="store_true", default=False)
    parser.add_argument("--qase_diversity_lambda", type=float, default=0.18)
    parser.add_argument("--qase_disable_confidence_fallback", action="store_true", default=False)
    parser.add_argument("--qase_confidence_fallback_low_score_threshold", type=float, default=0.62)
    parser.add_argument("--include_context_in_add_prompt", action="store_true", default=False)
    parser.add_argument("--target_only_custom_instructions", action="store_true", default=True)
    parser.add_argument("--no_target_only_custom_instructions", dest="target_only_custom_instructions", action="store_false")
    parser.add_argument(
        "--paper_extraction_mode",
        choices=["target_only", "include_m_context", "include_summary_and_m_context"],
        default="target_only",
    )
    parser.add_argument("--summary_mode", choices=["none", "rolling_llm", "dataset_if_safe"], default="none")
    parser.add_argument("--summary_model", default=None)
    parser.add_argument("--summary_max_tokens", type=int, default=None)
    parser.add_argument(
        "--summary_update_scope",
        choices=["after_each_batch", "after_each_session"],
        default="after_each_session",
    )
    parser.add_argument("--final_store_top_k", type=int, default=10000)

    parser.add_argument("--max_conversations", type=int, default=None)
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--max_qa", type=int, default=None)
    parser.add_argument("--skip_category_5", dest="skip_category_5", action="store_true", default=True)
    parser.add_argument("--include_category_5", dest="skip_category_5", action="store_false")

    parser.add_argument("--collection_prefix", default="mem0_locomo")
    parser.add_argument("--qdrant_path", default="local_qdrant_locomo")
    parser.add_argument("--answer_max_tokens", type=int, default=None)
    parser.add_argument("--memory_max_tokens", type=int, default=1000)
    parser.add_argument("--no_custom_instructions", action="store_true")
    parser.add_argument("--api_max_retries", type=int, default=10)
    parser.add_argument("--api_timeout_sec", type=float, default=120.0)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument(
        "--reuse_memory_checkpoint",
        default=None,
        help=(
            "Load an already-ingested json_local wrapper store from another run checkpoint. "
            "Use with --qa_only_reuse_memory to benchmark a new retrieval mode without re-adding memories."
        ),
    )
    parser.add_argument(
        "--qa_only_reuse_memory",
        action="store_true",
        default=False,
        help="Skip ingestion and answer QA using memories loaded from --reuse_memory_checkpoint.",
    )
    parser.add_argument(
        "--clear_reuse_embedding_cache",
        action="store_true",
        default=False,
        help=(
            "When loading --reuse_memory_checkpoint, clear the carried query/text embedding cache while "
            "keeping stored memory vectors. Use this for fair latency comparisons across retrieval modes."
        ),
    )

    return parser.parse_args()


def load_root_env() -> Path:
    cwd = Path.cwd()
    env_path = cwd / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        return cwd

    parent_env = Path(__file__).resolve().parents[1] / ".env"
    if parent_env.exists():
        load_dotenv(parent_env, override=True)
        return Path(__file__).resolve().parents[1]

    load_dotenv(override=True)
    return cwd


def get_env_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    value = value.strip() if value else ""
    return value or None


def is_deepseek_model(model_name: Optional[str]) -> bool:
    return "deepseek" in str(model_name or "").lower()


def apply_model_runtime_defaults(args: argparse.Namespace) -> Dict[str, Any]:
    if is_deepseek_model(args.llm_model):
        profile = "deepseek_reasoning_safe"
        defaults = {
            "candidate_extraction_max_tokens": 4000,
            "update_decision_max_tokens": 2000,
            "answer_max_tokens": 1000,
            "summary_max_tokens": 1000,
        }
    else:
        profile = "standard"
        defaults = {
            "candidate_extraction_max_tokens": 500,
            "update_decision_max_tokens": 500,
            "answer_max_tokens": 80,
            "summary_max_tokens": 300,
        }

    applied = {}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
            applied[name] = value

    return {
        "profile": profile,
        "applied_defaults": applied,
        "deepseek_model_detected": is_deepseek_model(args.llm_model),
    }


def create_chat_client(api_max_retries: int = 10, api_timeout_sec: float = 120.0) -> Tuple[OpenAI, Dict[str, Any]]:
    beeknoee_api_key = get_env_value("BEEKNOEE_API_KEY")
    beeknoee_base_url = get_env_value("BEEKNOEE_BASE_URL")
    if not beeknoee_api_key:
        raise RuntimeError("BEEKNOEE_API_KEY is required for chat completions; OPENAI_API_KEY is used only for embeddings.")
    if not beeknoee_base_url:
        raise RuntimeError("BEEKNOEE_BASE_URL is required for chat completions.")
    return OpenAI(
        api_key=beeknoee_api_key,
        base_url=beeknoee_base_url,
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


def create_embedding_client(
    api_max_retries: int = 10,
    api_timeout_sec: float = 120.0,
) -> Tuple[OpenAI, Dict[str, Any]]:
    openai_api_key = get_env_value("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
    return OpenAI(
        api_key=openai_api_key,
        max_retries=api_max_retries,
        timeout=api_timeout_sec,
    ), {
        "provider": "openai",
        "uses_openai_api_key": True,
        "base_url_configured": False,
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
    temporary_path = path.with_name(f"{path.name}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, ensure_ascii=False, indent=2)
        file_handle.flush()
        os.fsync(file_handle.fileno())
    last_error = None
    for attempt in range(1, 11):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(0.25 * attempt, 2.0))
    raise last_error


def build_resume_config(args: argparse.Namespace, dataset_path: Path) -> Dict[str, Any]:
    config = {
        name: value
        for name, value in sorted(vars(args).items())
        if name not in RESUME_CONFIG_IGNORED_FIELDS
    }
    if getattr(args, "qa_retrieval_mode", None) != "hybrid_adaptive":
        for name in HYBRID_ADAPTIVE_CONFIG_FIELDS:
            config.pop(name, None)
    if getattr(args, "qa_retrieval_mode", None) != "evidence_gap_hybrid":
        for name in EVIDENCE_GAP_CONFIG_FIELDS:
            config.pop(name, None)
    if getattr(args, "qa_retrieval_mode", None) not in QUESTION_AWARE_RETRIEVAL_MODES:
        for name in QUESTION_AWARE_SELECTIVE_CONFIG_FIELDS:
            config.pop(name, None)
    if getattr(args, "qa_retrieval_mode", None) != "active_reconstruction":
        for name in ACTIVE_RECONSTRUCTION_ONLY_CONFIG_FIELDS:
            config.pop(name, None)
    config["dataset_path"] = str(dataset_path.resolve())
    return config


def load_resume_checkpoint(path: Path, expected_config: Dict[str, Any]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_handle:
        checkpoint = json.load(file_handle)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema version: {checkpoint.get('schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )
    saved_config = dict(checkpoint.get("resume_config") or {})
    if saved_config.get("qa_retrieval_mode") == "hybrid_adaptive":
        for name, default_value in HYBRID_ADAPTIVE_CONFIG_DEFAULTS.items():
            saved_config.setdefault(name, default_value)
    if saved_config.get("qa_retrieval_mode") in QUESTION_AWARE_RETRIEVAL_MODES:
        for name, default_value in QUESTION_AWARE_SELECTIVE_CONFIG_DEFAULTS.items():
            saved_config.setdefault(name, default_value)
    if saved_config.get("qa_retrieval_mode") != "hybrid_adaptive":
        for name in HYBRID_ADAPTIVE_CONFIG_FIELDS:
            saved_config.pop(name, None)
    if saved_config.get("qa_retrieval_mode") != "evidence_gap_hybrid":
        for name in EVIDENCE_GAP_CONFIG_FIELDS:
            saved_config.pop(name, None)
    if saved_config.get("qa_retrieval_mode") not in QUESTION_AWARE_RETRIEVAL_MODES:
        for name in QUESTION_AWARE_SELECTIVE_CONFIG_FIELDS:
            saved_config.pop(name, None)
    if saved_config.get("qa_retrieval_mode") != "active_reconstruction":
        for name in ACTIVE_RECONSTRUCTION_ONLY_CONFIG_FIELDS:
            saved_config.pop(name, None)
    mismatches = {
        key: {"checkpoint": saved_config.get(key), "current": expected_config.get(key)}
        for key in sorted(set(saved_config) | set(expected_config))
        if saved_config.get(key) != expected_config.get(key)
    }
    if mismatches:
        mismatch_text = json.dumps(mismatches, ensure_ascii=False, indent=2)
        raise ValueError(f"Checkpoint configuration does not match this run:\n{mismatch_text}")
    return checkpoint


def load_reuse_memory_checkpoint(path: Path, dataset_path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file_handle:
        checkpoint = json.load(file_handle)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported reuse checkpoint schema version: {checkpoint.get('schema_version')}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}."
        )

    saved_config = checkpoint.get("resume_config") or {}
    saved_dataset_path = saved_config.get("dataset_path")
    if saved_dataset_path:
        try:
            saved_dataset_resolved = Path(saved_dataset_path).resolve()
            current_dataset_resolved = dataset_path.resolve()
        except OSError:
            saved_dataset_resolved = Path(saved_dataset_path)
            current_dataset_resolved = dataset_path
        if saved_dataset_resolved != current_dataset_resolved:
            raise ValueError(
                "--reuse_memory_checkpoint was created for a different dataset:\n"
                f"checkpoint: {saved_dataset_resolved}\n"
                f"current:    {current_dataset_resolved}"
            )

    state = checkpoint.get("state") or {}
    wrapper_store_state = state.get("wrapper_store_state") or {}
    memories_by_user = wrapper_store_state.get("memories_by_user") or {}
    if not memories_by_user:
        raise ValueError(
            "--reuse_memory_checkpoint does not contain json_local wrapper_store_state memories. "
            "Use a checkpoint produced by --memory_update_mode paper_update_wrapper "
            "--memory_wrapper_store json_local after ingestion has run."
        )

    ingested_sample_ids = set(checkpoint.get("completed_sample_ids") or [])
    partial_sample_state = state.get("partial_sample_state") or {}
    if partial_sample_state.get("ingestion_complete") and partial_sample_state.get("sample_id"):
        ingested_sample_ids.add(str(partial_sample_state["sample_id"]))
    if not ingested_sample_ids:
        raise ValueError(
            "--reuse_memory_checkpoint has memory rows, but no sample is marked ingestion-complete. "
            "Resume or finish ingestion first before using QA-only reuse."
        )

    return {
        "run_id": checkpoint.get("run_id"),
        "path": str(path.resolve()),
        "state": state,
        "wrapper_store_state": wrapper_store_state,
        "ingested_sample_ids": sorted(ingested_sample_ids),
    }


def find_reused_speaker_user_id(
    wrapper_store_state: Dict[str, Any],
    sample_id: str,
    conversation_index: int,
    speaker: str,
) -> Optional[str]:
    memories_by_user = wrapper_store_state.get("memories_by_user") or {}
    prefix = f"locomo_{slugify(sample_id)}_{conversation_index}_{slugify(speaker)}_"
    candidates = [user_id for user_id in memories_by_user if user_id.startswith(prefix)]
    if not candidates:
        return None
    candidates.sort(key=lambda user_id: len(memories_by_user.get(user_id) or []), reverse=True)
    return candidates[0]


def write_run_checkpoint(
    checkpoint_path: Path,
    run_id: str,
    output_path: Path,
    resume_config: Dict[str, Any],
    completed_sample_ids: Iterable[str],
    state: Dict[str, Any],
    completed: bool = False,
) -> None:
    atomic_write_json(
        checkpoint_path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": run_id,
            "output_path": str(output_path.resolve()),
            "resume_config": resume_config,
            "completed_sample_ids": sorted(completed_sample_ids),
            "completed": completed,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "state": state,
        },
    )


def select_custom_instructions(args) -> Optional[str]:
    if args.no_custom_instructions:
        return None
    if args.target_only_custom_instructions:
        return TARGET_ONLY_CUSTOM_INSTRUCTIONS
    return GENERAL_CUSTOM_INSTRUCTIONS


def inspect_local_sdk_capabilities() -> Dict[str, Any]:
    capabilities = {
        "local_sdk_memory_add_extraction_prompt": "unknown",
        "local_sdk_memory_add_prompt_builder": "unknown",
        "local_sdk_add_existing_memory_retrieval_detected": None,
        "local_sdk_add_existing_memory_retrieval_top_k": None,
        "local_sdk_internal_last_k_messages": None,
        "local_sdk_update_phase_detected": None,
        "local_sdk_update_similar_top_k": None,
        "local_sdk_update_top_k_configurable": False,
        "local_sdk_final_store_fetch_supported": hasattr(Memory, "get_all"),
        "local_sdk_direct_methods": {
            "get": hasattr(Memory, "get"),
            "get_all": hasattr(Memory, "get_all"),
            "search": hasattr(Memory, "search"),
            "update": hasattr(Memory, "update"),
            "delete": hasattr(Memory, "delete"),
            "delete_all": hasattr(Memory, "delete_all"),
            "history": hasattr(Memory, "history"),
            "_create_memory": hasattr(Memory, "_create_memory"),
        },
        "local_sdk_graph_enabled_in_runner": False,
        "local_sdk_notes": [],
    }

    try:
        source = inspect.getsource(Memory._add_to_vector_store)
    except (OSError, TypeError):
        capabilities["local_sdk_notes"].append("Could not inspect Memory._add_to_vector_store source at runtime.")
        return capabilities

    if "ADDITIVE_EXTRACTION_PROMPT" in source:
        capabilities["local_sdk_memory_add_extraction_prompt"] = "ADDITIVE_EXTRACTION_PROMPT"
    if "generate_additive_extraction_prompt" in source:
        capabilities["local_sdk_memory_add_prompt_builder"] = "generate_additive_extraction_prompt"

    last_k_match = re.search(r"get_last_messages\(.*?limit\s*=\s*(\d+)", source, flags=re.DOTALL)
    if last_k_match:
        capabilities["local_sdk_internal_last_k_messages"] = int(last_k_match.group(1))

    retrieval_match = re.search(r"vector_store\.search\(.*?top_k\s*=\s*(\d+)", source, flags=re.DOTALL)
    if retrieval_match:
        capabilities["local_sdk_add_existing_memory_retrieval_detected"] = True
        capabilities["local_sdk_add_existing_memory_retrieval_top_k"] = int(retrieval_match.group(1))
    else:
        capabilities["local_sdk_add_existing_memory_retrieval_detected"] = False

    update_events = re.findall(r'"event"\s*:\s*"(UPDATE|DELETE|NOOP)"', source)
    calls_memory_update = "_update_memory(" in source or "_delete_memory(" in source
    capabilities["local_sdk_update_phase_detected"] = bool(update_events or calls_memory_update)
    if capabilities["local_sdk_update_phase_detected"]:
        capabilities["local_sdk_update_similar_top_k"] = capabilities["local_sdk_add_existing_memory_retrieval_top_k"]
    else:
        capabilities["local_sdk_update_similar_top_k"] = None
        capabilities["local_sdk_notes"].append(
            "Inspected local Memory.add path appears additive: it returns ADD records, skips duplicates by hash, "
            "and does not expose ADD/UPDATE/DELETE/NOOP decision events in _add_to_vector_store."
        )

    capabilities["local_sdk_notes"].append(
        "Existing-memory retrieval during add is used as extraction/linking context, not verified as the paper update phase."
    )
    return capabilities


def resolve_effective_modes(args) -> Tuple[str, str, List[str]]:
    warnings = []
    paper_mode = args.paper_extraction_mode

    if args.include_context_in_add_prompt and paper_mode == "target_only":
        paper_mode = "include_m_context"
        warnings.append(
            "--include_context_in_add_prompt was provided with target_only; effective paper_extraction_mode is include_m_context."
        )

    summary_mode = args.summary_mode
    if summary_mode == "dataset_if_safe":
        summary_mode = "none"
        warnings.append(
            "summary_mode=dataset_if_safe was not used because LOCOMO dataset summaries are not proven prefix-only/non-leaking here."
        )

    if paper_mode == "include_summary_and_m_context" and summary_mode == "none":
        raise ValueError(
            "paper_extraction_mode=include_summary_and_m_context requires summary_mode=rolling_llm. "
            "dataset_if_safe is disabled unless prefix-safety is proven."
        )

    return paper_mode, summary_mode, warnings


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


def build_speaker_views(
    session: List[Dict[str, Any]],
    speaker_a: str,
    speaker_b: str,
    session_index: int,
    session_date: Optional[str],
) -> Dict[str, List[Dict[str, str]]]:
    messages = {speaker_a: [], speaker_b: []}

    for turn in session:
        speaker = turn.get("speaker")
        content = dialogue_to_content(turn, session_index, session_date)

        if speaker == speaker_a:
            messages[speaker_a].append({"role": "user", "content": content})
            messages[speaker_b].append({"role": "assistant", "content": content})
        elif speaker == speaker_b:
            messages[speaker_a].append({"role": "assistant", "content": content})
            messages[speaker_b].append({"role": "user", "content": content})
        else:
            raise ValueError(f"Unknown speaker: {speaker}")

    return messages


def chunked(items: List[Dict[str, str]], size: int) -> Iterable[Tuple[int, List[Dict[str, str]]]]:
    if size <= 0:
        raise ValueError("--batch_size must be greater than 0")

    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def mark_message(message: Dict[str, str], marker: str) -> Dict[str, str]:
    return {
        "role": message["role"],
        "content": f"{marker} {message['content']}",
    }


def build_add_messages(
    *,
    context_messages: List[Dict[str, str]],
    target_messages: List[Dict[str, str]],
    paper_extraction_mode: str,
    summary_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    if paper_extraction_mode == "target_only":
        return target_messages

    summary_messages = []
    if paper_extraction_mode == "include_summary_and_m_context":
        summary = summary_text.strip() if summary_text else "(no prior conversation summary yet)"
        summary_messages.append(
            {
                "role": "system",
                "content": (
                    "[CONVERSATION SUMMARY - CONTEXT ONLY - DO NOT EXTRACT MEMORY DIRECTLY FROM THIS SUMMARY]\n"
                    f"{summary}"
                ),
            }
        )

    return [
        *summary_messages,
        *[
            mark_message(message, "[CONTEXT ONLY - DO NOT EXTRACT MEMORY FROM THIS MESSAGE]")
            for message in context_messages
        ],
        *[
            mark_message(message, "[TARGET MESSAGE - EXTRACT MEMORY FROM THIS MESSAGE]")
            for message in target_messages
        ],
    ]


def build_add_metadata(
    *,
    sample_id: str,
    run_id: str,
    session_key: str,
    session_index: int,
    session_date: Optional[str],
    observed_at: Optional[str],
    perspective_speaker: str,
    speaker_a: str,
    speaker_b: str,
    batch_start: int,
    batch_size: int,
    context_window: int,
    context_start: int,
    num_context_messages_available: int,
    num_context_messages: int,
    num_context_messages_sent: int,
    num_summary_messages_sent: int,
    num_target_messages: int,
    num_messages_added: int,
) -> Dict[str, Any]:
    metadata = {
        "source": "locomo",
        "sample_id": sample_id,
        "benchmark_run_id": run_id,
        "session": session_key,
        "session_index": session_index,
        "session_timestamp_raw": session_date,
        "observed_at": observed_at,
        "perspective_speaker": perspective_speaker,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "batch_start": batch_start,
        "batch_size": batch_size,
        "context_window": context_window,
        "context_start": context_start,
        "num_context_messages_available": num_context_messages_available,
        "num_context_messages": num_context_messages,
        "num_context_messages_in_window": num_context_messages,
        "num_context_messages_sent": num_context_messages_sent,
        "num_summary_messages_sent": num_summary_messages_sent,
        "num_target_messages": num_target_messages,
        "num_messages_added": num_messages_added,
    }

    return metadata


def build_dia_lookup(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    dia_lookup = {}

    for session_index in get_session_indices(conversation):
        session_key = f"session_{session_index}"
        date_key = f"session_{session_index}_date_time"
        session_date = conversation.get(date_key)
        session = conversation.get(session_key) or []

        for turn in session:
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
    for eid in evidence_ids or []:
        item = dia_lookup.get(eid)
        if item:
            evidence_texts.append(item)
        else:
            evidence_texts.append({"dia_id": eid, "missing": True})
    return evidence_texts


def is_likely_benchmark_timestamp(value: Optional[str]) -> bool:
    if not value:
        return False
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def locomo_created_at(item: Dict[str, Any]) -> Optional[str]:
    metadata = item.get("metadata") or {}
    metadata_created_at = metadata.get("created_at")
    if metadata_created_at and not is_likely_benchmark_timestamp(metadata_created_at):
        return metadata_created_at

    created_at = item.get("created_at")
    if created_at and not is_likely_benchmark_timestamp(created_at):
        return created_at

    return None


def memory_observed_at(item: Dict[str, Any]) -> Optional[str]:
    metadata = item.get("metadata") or {}
    return (
        item.get("observed_at")
        or metadata.get("observed_at")
        or locomo_created_at(item)
    )


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
        reconstruction_context = item.get("mragent_reconstruction_context")
        if reconstruction_context:
            temporal_context += f"; reconstruction context: {reconstruction_context}"
        formatted.append(f"{temporal_context}: {memory_text}")
    return formatted


def count_text_tokens(encoding: tiktoken.Encoding, texts: List[str]) -> int:
    return sum(len(encoding.encode(text)) for text in texts)


def count_memory_tokens(encoding: tiktoken.Encoding, memories: List[Dict[str, Any]]) -> int:
    return count_text_tokens(encoding, format_memories_for_prompt(memories))


def count_memory_text_tokens(encoding: tiktoken.Encoding, memories: List[Dict[str, Any]]) -> int:
    return count_text_tokens(encoding, [item.get("memory", "") for item in memories])


def empty_event_counts() -> Dict[str, int]:
    return {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0, "UNKNOWN": 0}


def update_event_counts(counts: Dict[str, int], memories: List[Dict[str, Any]]) -> None:
    for item in memories:
        event = str(item.get("event", "UNKNOWN")).upper() if isinstance(item, dict) else "UNKNOWN"
        if event not in counts:
            event = "UNKNOWN"
        counts[event] += 1


def event_counts_for(memories: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = empty_event_counts()
    update_event_counts(counts, memories)
    return counts


def messages_to_plain_text(messages: List[Dict[str, str]]) -> str:
    return "\n".join(f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in messages)


def is_retryable_api_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in RETRYABLE_API_STATUS_CODES:
        return True
    error_type = type(exc).__name__
    cause_type = type(getattr(exc, "__cause__", None)).__name__
    return error_type in RETRYABLE_API_ERROR_TYPES or cause_type in RETRYABLE_API_ERROR_TYPES


def chat_completion_with_retries(chat_client: OpenAI, *, max_attempts: int = 10, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return chat_client.chat.completions.create(**kwargs)
        except Exception as exc:
            if attempt >= max_attempts or not is_retryable_api_error(exc):
                raise
            delay = min(2 ** (attempt - 1), 30)
            print(
                "Chat API call failed; retrying "
                f"({attempt}/{max_attempts}) after {delay}s: {type(exc).__name__}"
            )
            time.sleep(delay)


def update_rolling_summary(
    chat_client: OpenAI,
    *,
    summary_model: str,
    summary_max_tokens: int,
    previous_summary: str,
    new_messages: List[Dict[str, str]],
    perspective_speaker: str,
) -> Tuple[str, Dict[str, Any]]:
    prompt = f"""
You maintain a concise rolling conversation summary for future memory extraction.

Rules:
- Use only the existing summary and the newly ingested messages below.
- Do not use or infer future information.
- Preserve explicit speaker attribution using real names. Do not transfer one speaker's facts to the other speaker.
- Preserve salient names, relationships, dates, locations, plans, preferences, meaningful events, and temporal facts.
- Treat session timestamps as observation times. Preserve relative time expressions together with their observation
  date, and resolve them only when unambiguous.
- Preserve uncertainty. Do not invent an exact event date, relationship, cause, or fact that the messages do not establish.
- Keep the summary compact and useful as context for future memory extraction.
- Return only the updated summary text, no bullets unless useful.

Perspective speaker:
{perspective_speaker}

Existing summary:
{previous_summary or "(empty)"}

Newly ingested messages:
{messages_to_plain_text(new_messages)}

Updated summary:
""".strip()

    start = time.time()
    response = chat_completion_with_retries(
        chat_client,
        model=summary_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You update evidence-bound rolling conversation summaries. Preserve speaker attribution, "
                    "temporal uncertainty, and observation-versus-event time distinctions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=summary_max_tokens,
    )
    latency = time.time() - start
    usage = response.usage.model_dump() if response.usage else {}
    summary = response.choices[0].message.content.strip()
    log = {
        "summary_prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "summary_completion_tokens": int(usage.get("completion_tokens") or 0),
        "summary_total_tokens": int(usage.get("total_tokens") or 0),
        "summary_latency_sec": latency,
    }
    return summary, log


def fetch_final_memories(memory: Memory, user_id: str, top_k: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        result = memory.get_all(filters={"user_id": user_id}, top_k=top_k)
        return result.get("results", []) if isinstance(result, dict) else [], None
    except Exception as exc:
        return [], str(exc)


def trim_memories_to_token_budget(
    encoding: tiktoken.Encoding,
    speaker_1_memories: List[Dict[str, Any]],
    speaker_2_memories: List[Dict[str, Any]],
    token_budget: Optional[int],
    strategy: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    pre_trim_speaker_1_tokens = count_memory_tokens(encoding, speaker_1_memories)
    pre_trim_speaker_2_tokens = count_memory_tokens(encoding, speaker_2_memories)
    pre_trim_total_tokens = pre_trim_speaker_1_tokens + pre_trim_speaker_2_tokens

    info = {
        "retrieval_token_budget": token_budget,
        "retrieval_budget_strategy": strategy,
        "retrieval_token_budget_applied": False,
        "speaker_1_memory_tokens_pre_trim": pre_trim_speaker_1_tokens,
        "speaker_2_memory_tokens_pre_trim": pre_trim_speaker_2_tokens,
        "retrieved_memory_tokens_pre_trim": pre_trim_total_tokens,
        "num_speaker_1_memories_pre_trim": len(speaker_1_memories),
        "num_speaker_2_memories_pre_trim": len(speaker_2_memories),
        "num_retrieved_memories_pre_trim": len(speaker_1_memories) + len(speaker_2_memories),
        "num_retrieved_memories_dropped_by_budget": 0,
    }

    if token_budget is None:
        return speaker_1_memories, speaker_2_memories, info

    if token_budget <= 0:
        raise ValueError("--retrieval_token_budget must be greater than 0 when provided")

    if pre_trim_total_tokens <= token_budget:
        info["retrieval_token_budget_applied"] = True
        return speaker_1_memories, speaker_2_memories, info

    def ranked_items(memories: List[Dict[str, Any]], speaker_index: int) -> List[Dict[str, Any]]:
        return [
            {
                "speaker_index": speaker_index,
                "rank": rank,
                "memory": memory_item,
                "tokens": count_memory_tokens(encoding, [memory_item]),
                "score": memory_item.get("score") if isinstance(memory_item, dict) else None,
            }
            for rank, memory_item in enumerate(memories)
        ]

    combined = []
    for speaker_index, memories in ((1, speaker_1_memories), (2, speaker_2_memories)):
        combined.extend(ranked_items(memories, speaker_index))

    def keep_items(items: List[Dict[str, Any]], budget: int) -> List[Dict[str, Any]]:
        kept = []
        used_tokens = 0
        for item in items:
            if used_tokens + item["tokens"] > budget:
                continue
            used_tokens += item["tokens"]
            kept.append(item)
        if not kept and items:
            kept.append(items[0])
        return kept

    if strategy == "interleave_rank":
        ordered = sorted(combined, key=lambda item: (item["rank"], item["speaker_index"]))
        kept_items = keep_items(ordered, token_budget)
    elif strategy == "balanced_per_speaker":
        budget_1 = token_budget // 2
        budget_2 = token_budget - budget_1
        speaker_1_items = sorted([item for item in combined if item["speaker_index"] == 1], key=lambda item: item["rank"])
        speaker_2_items = sorted([item for item in combined if item["speaker_index"] == 2], key=lambda item: item["rank"])
        kept_items = [
            *keep_items(speaker_1_items, budget_1),
            *keep_items(speaker_2_items, budget_2),
        ]
    elif strategy == "global_score":
        ordered = sorted(
            combined,
            key=lambda item: (-(item["score"] if item["score"] is not None else -1.0), item["rank"], item["speaker_index"]),
        )
        kept_items = keep_items(ordered, token_budget)
    else:
        raise ValueError(f"Unsupported retrieval budget strategy: {strategy}")

    kept_items.sort(key=lambda item: (item["speaker_index"], item["rank"]))
    kept_speaker_1 = [item["memory"] for item in kept_items if item["speaker_index"] == 1]
    kept_speaker_2 = [item["memory"] for item in kept_items if item["speaker_index"] == 2]

    info["retrieval_token_budget_applied"] = True
    info["num_retrieved_memories_dropped_by_budget"] = (
        len(speaker_1_memories) + len(speaker_2_memories) - len(kept_speaker_1) - len(kept_speaker_2)
    )

    return kept_speaker_1, kept_speaker_2, info


def annotate_memories(memories: List[Dict[str, Any]], source_user_id: str, source_speaker: str) -> List[Dict[str, Any]]:
    annotated = []
    for item in memories:
        copied = dict(item)
        copied["source_user_id"] = source_user_id
        copied["source_speaker"] = source_speaker
        copied["observed_at"] = memory_observed_at(item)
        copied["event_at"] = memory_event_at(item)
        copied["event_time_text"] = memory_event_time_text(item)
        annotated.append(copied)
    return annotated


def search_memories(
    memory: Memory,
    *,
    question: str,
    user_id: str,
    top_k: int,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], float]:
    start = time.time()
    retrieved = memory.search(query=question, filters={"user_id": user_id}, top_k=top_k, threshold=threshold)
    latency = time.time() - start
    return retrieved.get("results", []) if isinstance(retrieved, dict) else [], latency


def search_wrapper_memories(
    store: LocalPaperMemoryStore,
    *,
    question: str,
    user_id: str,
    top_k: int,
    threshold: float,
) -> Tuple[List[Dict[str, Any]], float]:
    start = time.time()
    retrieved = store.search(user_id=user_id, query=question, top_k=top_k)
    if threshold > 0:
        retrieved = [item for item in retrieved if float(item.get("score") or 0.0) >= threshold]
    latency = time.time() - start
    return retrieved, latency


BM25_STOPWORDS = {
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
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "that",
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
    "whom",
    "why",
    "with",
}


def bm25_tokenize(text: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", (text or "").lower())
        if token not in BM25_STOPWORDS and len(token) > 1
    ]


def memory_text_for_retrieval(memory: Dict[str, Any]) -> str:
    return str(memory.get("memory") or memory.get("text") or memory.get("content") or "")


def score_bm25_semantic_candidates(
    store: LocalPaperMemoryStore,
    *,
    question: str,
    user_id: str,
    speaker: str,
    candidates: Sequence[Dict[str, Any]],
    k1: float,
    b: float,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    start = time.time()
    annotated = [dict(memory) for memory in candidates]
    query_terms = bm25_tokenize(question)

    for memory in annotated:
        memory["qase_bm25_score_raw"] = 0.0
        memory["qase_bm25_score"] = 0.0
        memory["qase_bm25_rank"] = None
        memory["qase_candidate_source"] = memory.get("qase_candidate_source") or "semantic"

    docs = store.get_all(user_id)
    if not annotated or not docs or not query_terms:
        latency = time.time() - start
        return annotated, latency, {
            "speaker": speaker,
            "retriever": "bm25_candidate_rerank",
            "store_size": len(docs),
            "candidate_count": len(annotated),
            "query_terms": query_terms,
            "positive_candidate_count": 0,
            "latency_sec": latency,
        }

    tokenized_docs: List[Counter] = []
    doc_freq: Counter = Counter()
    total_doc_len = 0
    for item in docs:
        tokens = bm25_tokenize(memory_text_for_retrieval(item))
        term_counts = Counter(tokens)
        tokenized_docs.append(term_counts)
        total_doc_len += len(tokens)
        doc_freq.update(term_counts.keys())

    avg_doc_len = total_doc_len / max(len(tokenized_docs), 1)
    query_unique_terms = list(dict.fromkeys(query_terms))
    num_docs = len(tokenized_docs)

    raw_scores: List[Tuple[float, int]] = []
    for index, memory in enumerate(annotated):
        term_counts = Counter(bm25_tokenize(memory_text_for_retrieval(memory)))
        doc_len = sum(term_counts.values())
        score = 0.0
        if doc_len > 0:
            for term in query_unique_terms:
                term_freq = term_counts.get(term, 0)
                if term_freq <= 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
                norm = k1 * (1.0 - b + b * doc_len / max(avg_doc_len, 1e-9))
                score += idf * (term_freq * (k1 + 1.0)) / (term_freq + norm)
        if score > 0:
            raw_scores.append((score, index))
        memory["qase_bm25_score_raw"] = score

    raw_scores.sort(key=lambda row: (-row[0], row[1]))
    max_score = raw_scores[0][0] if raw_scores else 0.0
    for rank, (raw_score, index) in enumerate(raw_scores):
        annotated[index]["qase_bm25_score"] = raw_score / max_score if max_score > 0 else 0.0
        annotated[index]["qase_bm25_rank"] = rank
        annotated[index]["qase_candidate_source"] = "semantic+bm25_candidate_rerank"

    latency = time.time() - start
    return annotated, latency, {
        "speaker": speaker,
        "retriever": "bm25_candidate_rerank",
        "store_size": len(docs),
        "candidate_count": len(annotated),
        "query_terms": query_terms,
        "positive_candidate_count": len(raw_scores),
        "max_raw_score": max_score,
        "latency_sec": latency,
    }


def search_qase_candidate_memories(
    *,
    search_call: Callable[[str, int], Tuple[List[Dict[str, Any]], float]],
    question: str,
    speaker: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    if top_k <= 0:
        return [], 0.0, {
            "speaker": speaker,
            "queries": [],
            "per_query_top_k": [],
            "raw_result_counts": [],
            "result_count": 0,
        }

    memories, latency = search_call(question, top_k)
    return memories, latency, {
        "speaker": speaker,
        "queries": [question],
        "per_query_top_k": [top_k],
        "raw_result_counts": [len(memories)],
        "result_count": len(memories),
    }


def _memory_candidate_id(memory: Dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "memory_id", "uuid"):
        value = memory.get(key)
        if value is not None:
            return str(value)
    text = memory.get("memory") or memory.get("text") or memory.get("content")
    return f"fallback:{fallback_index}:{text}"


def merge_memory_candidates(
    primary: Sequence[Dict[str, Any]],
    extra: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for index, memory in enumerate(list(primary) + list(extra)):
        key = _memory_candidate_id(memory, index)
        if key in seen:
            continue
        seen.add(key)
        merged.append(memory)
    added_count = max(0, len(merged) - len(primary))
    return merged, added_count


ADAPTIVE_COMPLEX_QUESTION_TYPES = {
    "comparison",
    "multi_hop",
    "temporal_change",
    "bounded_list",
    "list_open",
    "preference_inference",
    "ambiguous",
}

ADAPTIVE_CUTOFF_PROFILE_NAME = "adaptive_v6_balanced_budget_adaptive_k"

ADAPTIVE_MIN_FRACTION_BY_QUESTION_TYPE = {
    "single_fact": 0.0,
    "single_profile": 0.0,
    "temporal_fact": 0.0,
    "temporal_change": 0.0,
    "comparison": 0.0,
    "multi_hop": 0.0,
    "bounded_list": 0.0,
    "list_open": 0.0,
    "preference_inference": 0.0,
    "ambiguous": 0.0,
}

ADAPTIVE_TOTAL_BUDGET_BY_QUESTION_TYPE = {
    "single_fact": {"floor": 10, "cap": 13},
    "single_profile": {"floor": 8, "cap": 10},
    "temporal_fact": {"floor": 13, "cap": 15},
    "temporal_change": {"floor": 18, "cap": 21},
    "comparison": {"floor": 20, "cap": 22},
    "multi_hop": {"floor": 16, "cap": 20},
    "bounded_list": {"floor": 11, "cap": 13},
    "list_open": {"floor": 18, "cap": 21},
    # This group regressed under adaptive cutoff despite high memory count, so
    # keep a semantic top-k pack instead of applying score/elbow pruning.
    "preference_inference": {"floor": 21, "cap": 24, "fixed_semantic": True},
    "ambiguous": {"floor": 10, "cap": 12},
}

HYBRID_FUSION_PROFILE_NAME = "hybrid_fusion_v8_semantic_bm25_entity_time"

HYBRID_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
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
    "hers",
    "him",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "why",
    "with",
    "you",
    "your",
}

HYBRID_QUESTION_ENTITY_STOPWORDS = {
    "after",
    "before",
    "did",
    "does",
    "during",
    "how",
    "in",
    "latest",
    "recent",
    "recently",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

HYBRID_MONTH_ALIASES = {
    "january": "01",
    "jan": "01",
    "february": "02",
    "feb": "02",
    "march": "03",
    "mar": "03",
    "april": "04",
    "apr": "04",
    "may": "05",
    "june": "06",
    "jun": "06",
    "july": "07",
    "jul": "07",
    "august": "08",
    "aug": "08",
    "september": "09",
    "sep": "09",
    "sept": "09",
    "october": "10",
    "oct": "10",
    "november": "11",
    "nov": "11",
    "december": "12",
    "dec": "12",
}


def _hybrid_memory_text(memory: Dict[str, Any]) -> str:
    parts = [
        memory.get("memory"),
        memory.get("text"),
        memory.get("content"),
    ]
    return " ".join(str(part) for part in parts if part)


def _hybrid_memory_temporal_text(memory: Dict[str, Any]) -> str:
    metadata = memory.get("metadata") or {}
    parts = [
        memory.get("memory"),
        memory.get("text"),
        memory.get("content"),
        memory.get("event_at"),
        memory.get("event_time_text"),
        metadata.get("event_at"),
        metadata.get("event_time_text"),
    ]
    return " ".join(str(part) for part in parts if part)


def _hybrid_memory_observed_text(memory: Dict[str, Any]) -> str:
    metadata = memory.get("metadata") or {}
    parts = [
        memory.get("observed_at"),
        metadata.get("observed_at"),
    ]
    return " ".join(str(part) for part in parts if part)


def _hybrid_tokens(text: Any) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", str(text or "").lower())
        if len(token) > 1 and token not in HYBRID_STOPWORDS
    ]


def _hybrid_unique(tokens: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _hybrid_extract_time_keys(text: Any) -> List[str]:
    lowered = str(text or "").lower()
    keys = set()

    for match in re.finditer(r"\b((?:19|20)\d{2})[-/](\d{1,2})(?:[-/](\d{1,2}))?\b", lowered):
        year = match.group(1)
        month = f"{int(match.group(2)):02d}"
        keys.add(f"year:{year}")
        keys.add(f"month:{month}")
        keys.add(f"month_year:{year}-{month}")
        if match.group(3):
            day = f"{int(match.group(3)):02d}"
            keys.add(f"date:{year}-{month}-{day}")

    for month_name, month in HYBRID_MONTH_ALIASES.items():
        pattern = rf"\b{re.escape(month_name)}\b(?:\s+(\d{{1,2}})(?:st|nd|rd|th)?)?(?:,)?\s+((?:19|20)\d{{2}})"
        for match in re.finditer(pattern, lowered):
            day = match.group(1)
            year = match.group(2)
            keys.add(f"year:{year}")
            keys.add(f"month:{month}")
            keys.add(f"month_year:{year}-{month}")
            if day:
                keys.add(f"date:{year}-{month}-{int(day):02d}")
        if re.search(rf"\b{re.escape(month_name)}\b", lowered):
            keys.add(f"month:{month}")

    for match in re.finditer(r"\b((?:19|20)\d{2})\b", lowered):
        keys.add(f"year:{match.group(1)}")

    if re.search(r"\b(latest|recent|recently|current|currently|now|today|newest)\b", lowered):
        keys.add("relative:latest")
    if re.search(r"\b(before|earlier|previously|prior)\b", lowered):
        keys.add("relative:before")
    if re.search(r"\b(after|later|since|subsequently)\b", lowered):
        keys.add("relative:after")
    return sorted(keys)


def _hybrid_extract_entities(text: Any, speaker_names: Sequence[str] = ()) -> List[str]:
    raw = str(text or "")
    lowered = raw.lower()
    entities = set()

    for speaker in speaker_names:
        speaker_text = str(speaker or "").strip()
        if not speaker_text:
            continue
        speaker_lower = speaker_text.lower()
        first_lower = speaker_lower.split()[0]
        if re.search(rf"\b{re.escape(speaker_lower)}\b", lowered) or re.search(
            rf"\b{re.escape(first_lower)}\b",
            lowered,
        ):
            entities.add(first_lower)
            entities.add(speaker_lower)

    for match in re.finditer(r"\"([^\"]{2,80})\"|'([^']{2,80})'", raw):
        phrase = (match.group(1) or match.group(2) or "").strip().lower()
        if phrase and phrase not in HYBRID_QUESTION_ENTITY_STOPWORDS:
            entities.add(phrase)

    proper_pattern = r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|\d{4}))*\b"
    for match in re.finditer(proper_pattern, raw):
        phrase = re.sub(r"\s+", " ", match.group(0).strip()).lower()
        if not phrase:
            continue
        if phrase in HYBRID_QUESTION_ENTITY_STOPWORDS:
            continue
        phrase_parts = phrase.split()
        if any(part in HYBRID_MONTH_ALIASES for part in phrase_parts):
            continue
        if any(part.isdigit() and len(part) == 4 for part in phrase_parts):
            continue
        entities.add(phrase)
        if " " in phrase:
            entities.add(phrase.split()[0])

    return sorted(entities)


def _hybrid_bm25_scores(query_tokens: Sequence[str], document_tokens: Sequence[Sequence[str]]) -> List[float]:
    if not document_tokens:
        return []
    if not query_tokens:
        return [0.0 for _ in document_tokens]

    query_terms = _hybrid_unique(query_tokens)
    document_count = len(document_tokens)
    avg_doc_len = sum(len(tokens) for tokens in document_tokens) / max(1, document_count)
    document_token_sets = [set(tokens) for tokens in document_tokens]
    document_frequencies: Dict[str, int] = {}
    for term in query_terms:
        document_frequencies[term] = sum(1 for token_set in document_token_sets if term in token_set)

    raw_scores = []
    k1 = 1.2
    b = 0.75
    for tokens in document_tokens:
        term_counts: Dict[str, int] = {}
        for token in tokens:
            term_counts[token] = term_counts.get(token, 0) + 1
        doc_len = len(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_counts.get(term, 0)
            if frequency <= 0:
                continue
            df = document_frequencies.get(term, 0)
            idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (1.0 - b + b * doc_len / max(avg_doc_len, 1e-9))
            score += idf * (frequency * (k1 + 1.0)) / max(denominator, 1e-9)
        raw_scores.append(score)

    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score <= 0:
        return [0.0 for _ in raw_scores]
    return [score / max_score for score in raw_scores]


def _hybrid_semantic_norms(memories: Sequence[Dict[str, Any]]) -> List[float]:
    if not memories:
        return []
    raw_scores = [_semantic_score_for_cutoff(memory, rank) for rank, memory in enumerate(memories)]
    if all(0.0 <= score <= 1.0 for score in raw_scores):
        return raw_scores
    low = min(raw_scores)
    high = max(raw_scores)
    if high > low:
        return [(score - low) / (high - low) for score in raw_scores]
    if len(memories) == 1:
        return [1.0]
    return [max(0.0, 1.0 - rank / max(1, len(memories) - 1)) for rank, _ in enumerate(memories)]


def _hybrid_time_score(
    query_time_keys: Sequence[str],
    memory_time_keys: Sequence[str],
    *,
    needs_time: bool,
) -> Tuple[float, List[str]]:
    query_set = set(query_time_keys)
    memory_set = set(memory_time_keys)
    query_has_specific_date = any(key.startswith("date:") or key.startswith("month_year:") for key in query_set)
    query_has_month = any(key.startswith("month:") for key in query_set)
    exact = sorted(query_set & memory_set)
    if exact:
        if any(key.startswith("date:") or key.startswith("month_year:") for key in exact):
            return 1.0, exact
        if any(key.startswith("month:") for key in exact):
            return 0.45, exact
        if any(key.startswith("year:") for key in exact):
            return (0.30 if query_has_specific_date or query_has_month else 0.75), exact
        if any(key.startswith("relative:") for key in exact):
            return 0.35, exact

    query_years = {key for key in query_set if key.startswith("year:")}
    memory_years = {key for key in memory_set if key.startswith("year:")}
    year_overlap = sorted(query_years & memory_years)
    if year_overlap:
        return (0.25 if query_has_specific_date or query_has_month else 0.65), year_overlap

    query_months = {key for key in query_set if key.startswith("month:")}
    memory_months = {key for key in memory_set if key.startswith("month:")}
    month_overlap = sorted(query_months & memory_months)
    if month_overlap:
        return 0.35, month_overlap

    if any(key.startswith("relative:") for key in query_set) and memory_set:
        return 0.25, sorted(key for key in query_set if key.startswith("relative:"))

    if needs_time and memory_set:
        return 0.15, []

    return 0.0, []


def _hybrid_weights(args: argparse.Namespace) -> Dict[str, float]:
    weights = {
        "semantic": max(0.0, float(args.qase_hybrid_semantic_weight)),
        "bm25": max(0.0, float(args.qase_hybrid_bm25_weight)),
        "entity": max(0.0, float(args.qase_hybrid_entity_weight)),
        "temporal": max(0.0, float(args.qase_hybrid_temporal_weight)),
    }
    total = sum(weights.values())
    if total <= 0:
        return {"semantic": 1.0, "bm25": 0.0, "entity": 0.0, "temporal": 0.0}
    return {key: value / total for key, value in weights.items()}


def hybrid_fuse_candidate_memories(
    memories: Sequence[Dict[str, Any]],
    *,
    question: str,
    speaker: str,
    speaker_names: Sequence[str],
    needs_time: bool,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    start = time.time()
    if not memories:
        return [], {
            "enabled": True,
            "profile": HYBRID_FUSION_PROFILE_NAME,
            "speaker": speaker,
            "input_count": 0,
            "output_count": 0,
            "latency_sec": 0.0,
            "reason": "empty candidates",
        }

    weights = _hybrid_weights(args)
    question_tokens = _hybrid_tokens(question)
    question_entities = _hybrid_extract_entities(question, speaker_names)
    question_entity_set = set(question_entities)
    question_time_keys = _hybrid_extract_time_keys(question)
    document_texts = [_hybrid_memory_text(memory) for memory in memories]
    document_tokens = [_hybrid_tokens(text) for text in document_texts]
    bm25_scores = _hybrid_bm25_scores(question_tokens, document_tokens)
    semantic_scores = _hybrid_semantic_norms(memories)

    rescored: List[Dict[str, Any]] = []
    preview = []
    for rank, memory in enumerate(memories):
        text = document_texts[rank]
        memory_entities = _hybrid_extract_entities(text, speaker_names)
        matched_entities = sorted(question_entity_set & set(memory_entities))
        entity_score = (
            min(1.0, len(matched_entities) / max(1, len(question_entity_set)))
            if question_entity_set
            else 0.0
        )
        memory_time_keys = _hybrid_extract_time_keys(_hybrid_memory_temporal_text(memory))
        temporal_score, matched_time = _hybrid_time_score(
            question_time_keys,
            memory_time_keys,
            needs_time=needs_time,
        )
        temporal_source = "content_or_event"
        if temporal_score <= 0.0:
            observed_time_keys = _hybrid_extract_time_keys(_hybrid_memory_observed_text(memory))
            observed_temporal_score, observed_matched_time = _hybrid_time_score(
                question_time_keys,
                observed_time_keys,
                needs_time=needs_time,
            )
            if observed_temporal_score > 0:
                temporal_score = min(0.25, observed_temporal_score)
                matched_time = observed_matched_time
                temporal_source = "observed_at_fallback"
        semantic_score = semantic_scores[rank] if rank < len(semantic_scores) else 0.0
        bm25_score = bm25_scores[rank] if rank < len(bm25_scores) else 0.0
        fused_score = (
            weights["semantic"] * semantic_score
            + weights["bm25"] * bm25_score
            + weights["entity"] * entity_score
            + weights["temporal"] * temporal_score
        )
        copied = dict(memory)
        copied["hybrid_source_rank"] = rank
        copied["hybrid_semantic_score"] = semantic_score
        copied["hybrid_bm25_score"] = bm25_score
        copied["hybrid_entity_score"] = entity_score
        copied["hybrid_temporal_score"] = temporal_score
        copied["hybrid_fused_score"] = fused_score
        copied["qase_fused_score"] = fused_score
        copied["hybrid_matched_entities"] = matched_entities[:12]
        copied["hybrid_matched_time_keys"] = matched_time[:12]
        copied["hybrid_temporal_source"] = temporal_source
        copied["hybrid_question_entities"] = question_entities[:20]
        copied["hybrid_question_time_keys"] = question_time_keys[:20]
        copied["hybrid_fusion_profile"] = HYBRID_FUSION_PROFILE_NAME
        rescored.append(copied)
        if len(preview) < 8:
            preview.append(
                {
                    "source_rank": rank,
                    "semantic": round(semantic_score, 4),
                    "bm25": round(bm25_score, 4),
                    "entity": round(entity_score, 4),
                    "temporal": round(temporal_score, 4),
                    "fused": round(fused_score, 4),
                    "matched_entities": matched_entities[:6],
                    "matched_time_keys": matched_time[:6],
                    "temporal_source": temporal_source,
                }
            )

    rescored.sort(
        key=lambda memory: (
            -float(memory.get("hybrid_fused_score") or 0.0),
            int(memory.get("hybrid_source_rank") or 0),
        )
    )
    selected_preview = []
    for rank, memory in enumerate(rescored[:8]):
        selected_preview.append(
            {
                "rank": rank,
                "source_rank": int(memory.get("hybrid_source_rank") or 0),
                "fused": round(float(memory.get("hybrid_fused_score") or 0.0), 4),
                "semantic": round(float(memory.get("hybrid_semantic_score") or 0.0), 4),
                "bm25": round(float(memory.get("hybrid_bm25_score") or 0.0), 4),
                "entity": round(float(memory.get("hybrid_entity_score") or 0.0), 4),
                "temporal": round(float(memory.get("hybrid_temporal_score") or 0.0), 4),
                "matched_entities": list(memory.get("hybrid_matched_entities") or [])[:6],
                "matched_time_keys": list(memory.get("hybrid_matched_time_keys") or [])[:6],
                "temporal_source": memory.get("hybrid_temporal_source"),
            }
        )

    return rescored, {
        "enabled": True,
        "profile": HYBRID_FUSION_PROFILE_NAME,
        "speaker": speaker,
        "input_count": len(memories),
        "output_count": len(rescored),
        "weights": weights,
        "question_entities": question_entities[:20],
        "question_time_keys": question_time_keys[:20],
        "needs_time": needs_time,
        "pre_sort_preview": preview,
        "top_preview": selected_preview,
        "latency_sec": time.time() - start,
    }


def _semantic_score_for_cutoff(memory: Dict[str, Any], rank: int) -> float:
    for key in ("qase_fused_score", "hybrid_fused_score", "score"):
        try:
            score = float(memory.get(key))
            if math.isnan(score) or math.isinf(score):
                raise ValueError
            return score
        except Exception:
            continue
    return max(0.0, 1.0 - rank * 0.03)


def _adaptive_min_fraction(question_type: str, config: QASEConfig) -> float:
    configured_default = (
        config.complex_min_fraction_of_cap
        if question_type in ADAPTIVE_COMPLEX_QUESTION_TYPES
        else 0.70
    )
    return ADAPTIVE_MIN_FRACTION_BY_QUESTION_TYPE.get(question_type, configured_default)


def _adaptive_total_budget_profile(
    question_type: str,
    *,
    max_total_final_memories: int,
) -> Dict[str, Any]:
    configured = ADAPTIVE_TOTAL_BUDGET_BY_QUESTION_TYPE.get(question_type) or {}
    raw_global_cap = max(0, int(max_total_final_memories or 0))
    requested_cap = int(configured.get("cap") or raw_global_cap)
    requested_floor = int(configured.get("floor") or 0)
    if raw_global_cap <= 0:
        total_cap = max(0, requested_cap)
    elif requested_cap <= 0:
        total_cap = raw_global_cap
    else:
        total_cap = min(raw_global_cap, requested_cap)
    total_floor = min(total_cap, max(0, requested_floor))
    return {
        "question_type": question_type,
        "total_floor": total_floor,
        "total_cap": total_cap,
        "requested_floor": requested_floor,
        "requested_cap": requested_cap,
        "runner_max_total_final_memories": raw_global_cap,
        "fixed_semantic": bool(configured.get("fixed_semantic")),
        "profile_name": ADAPTIVE_CUTOFF_PROFILE_NAME,
    }


def _cap_speaker_budgets_to_total(
    caps_by_speaker: Dict[str, int],
    *,
    total_cap: int,
    preferred_speakers: Sequence[str] = (),
) -> Dict[str, int]:
    caps = {speaker: max(0, int(cap or 0)) for speaker, cap in caps_by_speaker.items()}
    if total_cap <= 0:
        return {speaker: 0 for speaker in caps}
    if sum(caps.values()) <= total_cap:
        return caps

    order: List[str] = []
    for speaker in preferred_speakers:
        if speaker in caps and speaker not in order:
            order.append(speaker)
    order.extend(speaker for speaker in caps if speaker not in order)
    if not order:
        order = list(caps)

    out = {speaker: 0 for speaker in caps}
    while sum(out.values()) < total_cap:
        progressed = False
        for speaker in order:
            if out[speaker] >= caps.get(speaker, 0):
                continue
            out[speaker] += 1
            progressed = True
            if sum(out.values()) >= total_cap:
                break
        if not progressed:
            break
    return out


def _adaptive_elbow_cutoff_k(
    scores: List[float],
    *,
    question_type: str,
) -> Tuple[int, Dict[str, Any]]:
    if len(scores) < 4:
        return len(scores), {
            "used": False,
            "cutoff_k": len(scores),
            "reason": "too few scores for elbow cutoff",
        }

    drops = [max(0.0, scores[index] - scores[index + 1]) for index in range(len(scores) - 1)]
    if not drops:
        return len(scores), {
            "used": False,
            "cutoff_k": len(scores),
            "reason": "no adjacent score drops",
        }

    max_drop = max(drops)
    max_drop_index = drops.index(max_drop)
    avg_drop = sum(drops) / max(1, len(drops))
    is_complex = question_type in ADAPTIVE_COMPLEX_QUESTION_TYPES
    min_absolute_drop = 0.045 if is_complex else 0.055
    mean_multiplier = 1.75 if is_complex else 2.00
    required_drop = max(min_absolute_drop, avg_drop * mean_multiplier)
    used = max_drop >= required_drop
    cutoff_k = max_drop_index + 1 if used else len(scores)
    return cutoff_k, {
        "used": used,
        "cutoff_k": cutoff_k,
        "max_drop": round(max_drop, 4),
        "max_drop_after_rank": max_drop_index + 1,
        "avg_drop": round(avg_drop, 4),
        "required_drop": round(required_drop, 4),
        "reason": "largest adjacent semantic score drop" if used else "no clear elbow",
    }


def _adaptive_k_window_indices(
    length: int,
    *,
    ignore_extreme: float = 0.0,
    ignore_extreme_tail: float = 0.0,
    ignore_below_median: bool = False,
) -> Tuple[int, int]:
    if length <= 0:
        return 0, 0
    if ignore_below_median:
        ignore_extreme_tail = 0.5
    start = int((length - 1) * ignore_extreme) if isinstance(ignore_extreme, float) else int(ignore_extreme)
    tail = (
        int((length - 1) * ignore_extreme_tail)
        if isinstance(ignore_extreme_tail, float)
        else int(ignore_extreme_tail)
    )
    start = max(0, min(start, length - 1))
    end = length - max(0, tail)
    end = max(start + 1, min(end, length))
    return start, end


def _adaptive_k_largest_gap_threshold(
    scores: List[float],
    *,
    ignore_extreme: float = 0.0,
    ignore_extreme_tail: float = 0.0,
    ignore_below_median: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    # Ported from megagonlabs/adaptive-k-retrieval Retriever.find_threshold_largest_gap,
    # rewritten without torch and returning a 0-based threshold index.
    if len(scores) < 2:
        return max(0, len(scores) - 1), {
            "strategy": "largest_gap",
            "reason": "too few scores",
        }
    start, end = _adaptive_k_window_indices(
        len(scores),
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
        ignore_below_median=ignore_below_median,
    )
    last_gap_index = end - 1
    if start >= last_gap_index:
        return end - 1, {
            "strategy": "largest_gap",
            "reason": "empty gap window",
            "window": [start, end],
        }
    gap_items = [
        (idx, max(0.0, scores[idx] - scores[idx + 1]))
        for idx in range(start, last_gap_index)
    ]
    threshold_index, max_gap = max(gap_items, key=lambda item: (item[1], -item[0]))
    return threshold_index, {
        "strategy": "largest_gap",
        "threshold_index": threshold_index,
        "cutoff_k": threshold_index + 1,
        "max_gap": round(max_gap, 4),
        "window": [start, end],
        "reason": "largest adjacent similarity gap",
    }


def _adaptive_k_2diff_spike_threshold(
    scores: List[float],
    *,
    ignore_extreme: float = 0.0,
    ignore_extreme_tail: float = 0.0,
    ignore_below_median: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    # Ported from megagonlabs/adaptive-k-retrieval Retriever.find_threshold_2diff_spike.
    # Adaptive-k returns a threshold index; retrieval keeps items with i <= threshold.
    if len(scores) < 4:
        threshold, fallback_log = _adaptive_k_largest_gap_threshold(
            scores,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
            ignore_below_median=ignore_below_median,
        )
        return threshold, {
            "strategy": "2diff_spike",
            "used": False,
            "fallback": fallback_log,
            "reason": "too few scores for second-difference spike",
        }

    start, end = _adaptive_k_window_indices(
        len(scores),
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
        ignore_below_median=ignore_below_median,
    )
    window_scores = scores[start:end]
    if len(window_scores) < 4:
        threshold, fallback_log = _adaptive_k_largest_gap_threshold(
            scores,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
            ignore_below_median=ignore_below_median,
        )
        return threshold, {
            "strategy": "2diff_spike",
            "used": False,
            "fallback": fallback_log,
            "window": [start, end],
            "reason": "window too small for second-difference spike",
        }

    first_diff = [window_scores[idx + 1] - window_scores[idx] for idx in range(len(window_scores) - 1)]
    second_diff = [first_diff[idx + 1] - first_diff[idx] for idx in range(len(first_diff) - 1)]
    if len(second_diff) < 2:
        threshold, fallback_log = _adaptive_k_largest_gap_threshold(
            scores,
            ignore_extreme=ignore_extreme,
            ignore_extreme_tail=ignore_extreme_tail,
            ignore_below_median=ignore_below_median,
        )
        return threshold, {
            "strategy": "2diff_spike",
            "used": False,
            "fallback": fallback_log,
            "window": [start, end],
            "reason": "not enough second-difference values",
        }

    cumulative_min: List[float] = []
    current_min = second_diff[0]
    for value in second_diff:
        current_min = min(current_min, value)
        cumulative_min.append(current_min)

    for local_idx, value in enumerate(second_diff[1:]):
        if value > 0 and cumulative_min[local_idx] < 0:
            threshold_index = start + local_idx + 2
            threshold_index = min(max(0, threshold_index), len(scores) - 1)
            return threshold_index, {
                "strategy": "2diff_spike",
                "used": True,
                "threshold_index": threshold_index,
                "cutoff_k": threshold_index + 1,
                "window": [start, end],
                "first_diff_preview": [round(value, 4) for value in first_diff[: min(6, len(first_diff))]],
                "second_diff_preview": [round(value, 4) for value in second_diff[: min(6, len(second_diff))]],
                "reason": "first second-difference spike after negative curvature",
            }

    threshold, fallback_log = _adaptive_k_largest_gap_threshold(
        scores,
        ignore_extreme=ignore_extreme,
        ignore_extreme_tail=ignore_extreme_tail,
        ignore_below_median=ignore_below_median,
    )
    return threshold, {
        "strategy": "2diff_spike",
        "used": False,
        "fallback": fallback_log,
        "window": [start, end],
        "reason": "no second-difference spike; fallback to largest_gap",
    }


def _adaptive_k_cutoff_k(scores: List[float]) -> Tuple[int, Dict[str, Any]]:
    if not scores:
        return 0, {
            "enabled": False,
            "reason": "empty score list",
        }
    threshold_index, log = _adaptive_k_2diff_spike_threshold(scores)
    cutoff_k = min(len(scores), max(1, int(threshold_index) + 1))
    return cutoff_k, {
        "enabled": True,
        "source": "megagonlabs/adaptive-k-retrieval",
        "profile": ADAPTIVE_CUTOFF_PROFILE_NAME,
        "cutoff_k": cutoff_k,
        "threshold_index": int(threshold_index),
        "threshold": log,
    }


def _memory_identity(memory: Dict[str, Any], fallback_index: int) -> str:
    for key in ("id", "memory_id"):
        value = memory.get(key)
        if value is not None:
            return str(value)
    memory_text = memory.get("memory")
    if memory_text is not None:
        return str(memory_text)
    try:
        return json.dumps(memory, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return f"memory@{fallback_index}"


def adaptive_semantic_cutoff(
    memories: List[Dict[str, Any]],
    *,
    cap: int,
    question_type: str,
    config: QASEConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if cap <= 0 or not memories:
        return [], {
            "input_count": len(memories),
            "selected_count": 0,
            "cap": cap,
            "cutoff_k": 0,
            "reason": "empty memory list or zero cap",
        }

    capped = list(memories[: min(cap, len(memories))])
    if not config.adaptive_cutoff_enabled or len(capped) < 3:
        return capped, {
            "input_count": len(memories),
            "selected_count": len(capped),
            "cap": cap,
            "cutoff_k": len(capped),
            "reason": "adaptive cutoff disabled or too few candidates",
            "score_preview": [
                round(_semantic_score_for_cutoff(item, idx), 4)
                for idx, item in enumerate(capped[: min(8, len(capped))])
            ],
        }

    is_complex = question_type in ADAPTIVE_COMPLEX_QUESTION_TYPES
    drop = config.adaptive_drop_complex if is_complex else config.adaptive_drop_simple
    min_fraction = _adaptive_min_fraction(question_type, config)
    scores = [_semantic_score_for_cutoff(item, idx) for idx, item in enumerate(capped)]
    best = scores[0]
    score_threshold_k = 0
    for score in scores:
        if score >= best - drop:
            score_threshold_k += 1
        else:
            break
    adaptive_k, adaptive_k_log = _adaptive_k_cutoff_k(scores)
    threshold_k = adaptive_k
    min_k = max(1, int(math.ceil(len(capped) * min_fraction)))
    cutoff_k = min(len(capped), max(threshold_k, min_k))
    selected = capped[:cutoff_k]
    return selected, {
        "input_count": len(memories),
        "selected_count": len(selected),
        "cap": cap,
        "cutoff_k": cutoff_k,
        "score_threshold_k": score_threshold_k,
        "adaptive_k": adaptive_k,
        "floor_k": min_k,
        "min_fraction": min_fraction,
        "adaptive_k_log": adaptive_k_log,
        "legacy_drop_threshold": f"best-{drop:.2f}",
        "reason": f"adaptive-k cutoff with min_fraction={min_fraction:.2f}",
        "score_preview": [round(score, 4) for score in scores[: min(8, len(scores))]],
    }


def rebalance_adaptive_semantic_memories(
    *,
    candidates_by_speaker: Dict[str, List[Dict[str, Any]]],
    selected_by_speaker: Dict[str, List[Dict[str, Any]]],
    caps_by_speaker: Dict[str, int],
    question_type: str,
    config: QASEConfig,
    max_total_final_memories: int,
    total_floor: Optional[int] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    updated = {speaker: list(selected_by_speaker.get(speaker) or []) for speaker in candidates_by_speaker}
    initial_total = sum(len(items) for items in updated.values())
    planned_total_cap = sum(max(0, int(value or 0)) for value in caps_by_speaker.values())
    global_cap = min(max(0, int(max_total_final_memories or 0)), planned_total_cap)
    if global_cap <= 0:
        return updated, {
            "enabled": False,
            "initial_total_selected": initial_total,
            "final_total_selected": initial_total,
            "reason": "zero global cap",
        }

    min_fraction = _adaptive_min_fraction(question_type, config)
    if total_floor is None:
        global_floor = min(global_cap, max(1, int(math.ceil(global_cap * min_fraction))))
    else:
        global_floor = min(global_cap, max(0, int(total_floor)))

    selected_keys_by_speaker: Dict[str, set] = {}
    rank_by_speaker_key: Dict[Tuple[str, str], int] = {}
    for speaker, candidates in candidates_by_speaker.items():
        rank_map: Dict[str, int] = {}
        for rank, memory in enumerate(candidates):
            identity = _memory_identity(memory, rank)
            rank_map.setdefault(identity, rank)
            rank_by_speaker_key[(speaker, identity)] = rank_map[identity]
        selected_keys_by_speaker[speaker] = {
            _memory_identity(memory, rank)
            for rank, memory in enumerate(updated.get(speaker) or [])
        }

    remaining_candidates: List[Tuple[float, int, str, str, Dict[str, Any]]] = []
    for speaker, candidates in candidates_by_speaker.items():
        seen_keys = selected_keys_by_speaker.setdefault(speaker, set())
        for rank, memory in enumerate(candidates):
            identity = _memory_identity(memory, rank)
            if identity in seen_keys:
                continue
            score = _semantic_score_for_cutoff(memory, rank)
            remaining_candidates.append((score, rank, speaker, identity, memory))

    remaining_candidates.sort(key=lambda item: (-item[0], item[1]))
    if initial_total >= global_floor:
        return updated, {
            "enabled": False,
            "initial_total_selected": initial_total,
            "final_total_selected": initial_total,
            "global_cap": global_cap,
            "global_floor": global_floor,
            "min_fraction": min_fraction,
            "floor_only": True,
            "reason": "already meets global floor",
        }

    top_up_by_speaker = {speaker: 0 for speaker in candidates_by_speaker}
    top_up_preview = []
    top_up_due_to_floor = 0
    total_selected = initial_total
    for score, rank, speaker, identity, memory in remaining_candidates:
        if total_selected >= global_floor or total_selected >= global_cap:
            break
        updated.setdefault(speaker, []).append(memory)
        selected_keys_by_speaker.setdefault(speaker, set()).add(identity)
        top_up_by_speaker[speaker] = top_up_by_speaker.get(speaker, 0) + 1
        top_up_due_to_floor += 1
        if len(top_up_preview) < 8:
            top_up_preview.append(
                {
                    "speaker": speaker,
                    "rank": rank,
                    "score": round(score, 4),
                    "id": identity,
                }
            )
        total_selected += 1

    for speaker, selected in updated.items():
        selected.sort(
            key=lambda memory: rank_by_speaker_key.get(
                (speaker, _memory_identity(memory, 0)),
                10**9,
            )
        )

    return updated, {
        "enabled": True,
        "initial_total_selected": initial_total,
        "final_total_selected": sum(len(items) for items in updated.values()),
        "global_cap": global_cap,
        "global_floor": global_floor,
        "min_fraction": min_fraction,
        "floor_only": True,
        "top_up_total": sum(top_up_by_speaker.values()),
        "top_up_due_to_floor": top_up_due_to_floor,
        "top_up_due_to_score": 0,
        "top_up_by_speaker": top_up_by_speaker,
        "top_up_preview": top_up_preview,
        "reason": "global top-up only until floor is met",
    }


def serialize_qase_selection_log(selection_log: Any) -> Optional[Dict[str, Any]]:
    if selection_log is None:
        return None
    if isinstance(selection_log, dict):
        return selection_log
    if hasattr(selection_log, "to_dict"):
        return selection_log.to_dict()
    return dict(selection_log)


def update_usage_totals(totals: Dict[str, Any], log: Dict[str, Any], prefix: str) -> None:
    for field in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        totals[f"{prefix}_{field}"] += int(log.get(f"{prefix}_{field}") or 0)
    totals[f"{prefix}_latency_sec"] += float(log.get(f"{prefix}_latency_sec") or 0.0)


def make_wrapper_latency_summary(
    totals: Dict[str, Any],
    *,
    num_candidate_extractions: int,
    num_candidates: int,
    num_update_decisions: int,
    num_information_content_guards: int,
) -> Dict[str, Any]:
    total_latency = (
        float(totals.get("candidate_extraction_latency_sec") or 0.0)
        + float(totals.get("update_decision_latency_sec") or 0.0)
        + float(totals.get("information_content_guard_latency_sec") or 0.0)
        + float(totals.get("store_operation_latency_sec") or 0.0)
    )
    return {
        **totals,
        "total_wrapper_latency_sec": total_latency,
        "num_candidate_extractions": num_candidate_extractions,
        "num_candidates": num_candidates,
        "num_update_decisions": num_update_decisions,
        "num_information_content_guards": num_information_content_guards,
        "avg_candidate_extraction_latency_sec": (
            totals["candidate_extraction_latency_sec"] / num_candidate_extractions
            if num_candidate_extractions
            else 0.0
        ),
        "avg_update_decision_latency_sec": (
            totals["update_decision_latency_sec"] / num_update_decisions if num_update_decisions else 0.0
        ),
        "avg_information_content_guard_latency_sec": (
            totals["information_content_guard_latency_sec"] / num_information_content_guards
            if num_information_content_guards
            else 0.0
        ),
        "avg_total_wrapper_latency_per_candidate_sec": total_latency / num_candidates if num_candidates else 0.0,
    }


def answer_question(
    chat_client: OpenAI,
    *,
    llm_model: str,
    question: str,
    speaker_1_name: str,
    speaker_2_name: str,
    speaker_1_memories: List[Dict[str, Any]],
    speaker_2_memories: List[Dict[str, Any]],
    answer_max_tokens: int,
) -> Tuple[str, Dict[str, Any], str]:
    template = Template(ANSWER_PROMPT)
    prompt = template.render(
        speaker_1_user_id=speaker_1_name,
        speaker_2_user_id=speaker_2_name,
        speaker_1_memories=json.dumps(format_memories_for_prompt(speaker_1_memories), indent=4, ensure_ascii=False),
        speaker_2_memories=json.dumps(format_memories_for_prompt(speaker_2_memories), indent=4, ensure_ascii=False),
        question=question,
    )

    response = chat_completion_with_retries(
        chat_client,
        model=llm_model,
        messages=[{"role": "system", "content": prompt}],
        temperature=0,
        max_tokens=answer_max_tokens,
    )

    answer = (response.choices[0].message.content or "").strip()
    usage = response.usage.model_dump() if response.usage else {}

    if not answer:
        retry_prompt = (
            "The previous response was empty. Return the answer requested by the system prompt. "
            "Do not return an empty response. "
            "If the memories do not contain enough evidence, answer exactly: No information available."
        )
        retry_response = chat_completion_with_retries(
            chat_client,
            model=llm_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0,
            max_tokens=answer_max_tokens,
        )
        retry_answer = (retry_response.choices[0].message.content or "").strip()
        retry_usage = retry_response.usage.model_dump() if retry_response.usage else {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[key] = int(usage.get(key) or 0) + int(retry_usage.get(key) or 0)
        usage["empty_answer_retry_count"] = 1
        answer = retry_answer or "No information available."
    else:
        usage["empty_answer_retry_count"] = 0
    return answer, usage, prompt


def run():
    args = parse_args()
    if args.context_window < 0:
        raise ValueError("--context_window must be >= 0")
    if args.final_store_top_k <= 0:
        raise ValueError("--final_store_top_k must be greater than 0")
    if args.update_similar_top_k <= 0:
        raise ValueError("--update_similar_top_k must be greater than 0")
    if args.api_max_retries < 0:
        raise ValueError("--api_max_retries must be >= 0")
    if args.api_timeout_sec <= 0:
        raise ValueError("--api_timeout_sec must be greater than 0")
    if args.active_reconstruction_max_steps <= 0:
        raise ValueError("--active_reconstruction_max_steps must be greater than 0")
    if args.active_reconstruction_max_tool_calls_per_step <= 0:
        raise ValueError("--active_reconstruction_max_tool_calls_per_step must be greater than 0")
    if (
        args.qa_retrieval_mode in {"active_reconstruction", "hybrid_adaptive", "evidence_gap_hybrid"}
        and (
            Mem0ActiveMemoryIndex is None
            or active_reconstruct_memories is None
            or hybrid_adaptive_retrieve_memories is None
            or evidence_gap_retrieve_memories is None
        )
    ):
        raise ValueError(
            "--qa_retrieval_mode active_reconstruction/hybrid_adaptive/evidence_gap_hybrid "
            "requires the optional legacy mragent_reconstruction module, which is not present in this cleaned setup."
        )
    if args.qa_retrieval_mode == "active_reconstruction" and args.active_reconstruction_seed_top_k <= 0:
        raise ValueError("--active_reconstruction_seed_top_k must be greater than 0")
    if args.active_reconstruction_router_max_tokens <= 0:
        raise ValueError("--active_reconstruction_router_max_tokens must be greater than 0")
    if args.protected_semantic_top_k <= 0:
        raise ValueError("--protected_semantic_top_k must be greater than 0")
    if args.active_extra_k_per_speaker < 0:
        raise ValueError("--active_extra_k_per_speaker must be >= 0")
    if args.har_active_seed_top_k <= 0:
        raise ValueError("--har_active_seed_top_k must be greater than 0")
    if args.har_active_seed_top_k > args.top_k:
        raise ValueError("--har_active_seed_top_k must be less than or equal to --top_k")
    if args.har_evidence_rerank_max_candidates <= 0:
        raise ValueError("--har_evidence_rerank_max_candidates must be greater than 0")
    if args.har_evidence_rerank_max_tokens <= 0:
        raise ValueError("--har_evidence_rerank_max_tokens must be greater than 0")
    if args.har_sufficiency_max_rounds < 0:
        raise ValueError("--har_sufficiency_max_rounds must be greater than or equal to 0")
    if args.har_sufficiency_max_candidates <= 0:
        raise ValueError("--har_sufficiency_max_candidates must be greater than 0")
    if args.har_sufficiency_max_tokens <= 0:
        raise ValueError("--har_sufficiency_max_tokens must be greater than 0")
    if args.egc_max_gap_rounds < 0:
        raise ValueError("--egc_max_gap_rounds must be greater than or equal to 0")
    if args.egc_max_slots <= 0:
        raise ValueError("--egc_max_slots must be greater than 0")
    if not 0.0 <= args.egc_min_slot_coverage <= 1.0:
        raise ValueError("--egc_min_slot_coverage must be between 0 and 1")
    if args.egc_gap_top_k_per_slot <= 0:
        raise ValueError("--egc_gap_top_k_per_slot must be greater than 0")
    if args.egc_max_gap_evidence_per_speaker < 0:
        raise ValueError("--egc_max_gap_evidence_per_speaker must be greater than or equal to 0")
    if args.egc_contract_max_tokens <= 0:
        raise ValueError("--egc_contract_max_tokens must be greater than 0")
    if args.egc_coverage_max_candidates <= 0:
        raise ValueError("--egc_coverage_max_candidates must be greater than 0")
    if args.egc_coverage_max_tokens <= 0:
        raise ValueError("--egc_coverage_max_tokens must be greater than 0")
    if args.egc_proof_pack_max_candidates <= 0:
        raise ValueError("--egc_proof_pack_max_candidates must be greater than 0")
    if args.egc_proof_pack_max_tokens <= 0:
        raise ValueError("--egc_proof_pack_max_tokens must be greater than 0")
    if args.egc_max_tool_calls_per_round <= 0:
        raise ValueError("--egc_max_tool_calls_per_round must be greater than 0")
    if args.qase_max_total_final_memories <= 0:
        raise ValueError("--qase_max_total_final_memories must be greater than 0")
    if args.qase_candidate_pool_scale <= 0:
        raise ValueError("--qase_candidate_pool_scale must be greater than 0")
    if args.qase_final_k_scale <= 0:
        raise ValueError("--qase_final_k_scale must be greater than 0")
    if args.qase_diversity_lambda < 0:
        raise ValueError("--qase_diversity_lambda must be >= 0")
    if not 0.0 <= args.qase_confidence_fallback_low_score_threshold <= 2.0:
        raise ValueError("--qase_confidence_fallback_low_score_threshold must be between 0 and 2")
    qase_hybrid_weights = (
        args.qase_hybrid_semantic_weight,
        args.qase_hybrid_bm25_weight,
        args.qase_hybrid_entity_weight,
        args.qase_hybrid_temporal_weight,
    )
    if any(weight < 0 for weight in qase_hybrid_weights):
        raise ValueError("QASE hybrid fusion weights must be >= 0")
    if sum(qase_hybrid_weights) <= 0:
        raise ValueError("At least one QASE hybrid fusion weight must be > 0")
    if args.qase_bm25_k1 <= 0:
        raise ValueError("--qase_bm25_k1 must be > 0")
    if not 0.0 <= args.qase_bm25_b <= 1.0:
        raise ValueError("--qase_bm25_b must be between 0 and 1")
    if args.qase_lexical_weight < 0:
        raise ValueError("--qase_lexical_weight must be >= 0")
    if args.qase_bm25_candidate_rerank and args.memory_update_mode != "paper_update_wrapper":
        raise ValueError(
            "QASE BM25 modes currently require --memory_update_mode paper_update_wrapper "
            "because BM25 reranking reads the local memory store."
        )
    if args.memory_update_mode == "paper_update_wrapper" and args.memory_wrapper_store != "json_local":
        raise ValueError(
            "paper_update_wrapper currently implements only --memory_wrapper_store json_local. "
            "mem0_sdk/qdrant_direct are reserved for future prototypes."
        )
    if args.qa_only_reuse_memory and args.memory_update_mode != "paper_update_wrapper":
        raise ValueError("--qa_only_reuse_memory currently supports only --memory_update_mode paper_update_wrapper.")
    if args.qa_only_reuse_memory and not args.reuse_memory_checkpoint and not args.resume_checkpoint:
        raise ValueError("--qa_only_reuse_memory requires --reuse_memory_checkpoint for a new run.")

    root_dir = load_root_env()
    dataset_path = normalize_dataset_path(args.dataset, root_dir)
    args.llm_model = args.llm_model or get_env_value("MODEL") or "gpt-4o-mini"
    args.embedding_model = args.embedding_model or get_env_value("EMBEDDING_MODEL") or "text-embedding-3-small"
    args.summary_model = args.summary_model or args.llm_model
    model_runtime_defaults = apply_model_runtime_defaults(args)
    for token_arg in (
        "candidate_extraction_max_tokens",
        "update_decision_max_tokens",
        "answer_max_tokens",
        "summary_max_tokens",
    ):
        if getattr(args, token_arg) <= 0:
            raise ValueError(f"--{token_arg} must be greater than 0")
    effective_paper_extraction_mode, effective_summary_mode, configuration_warnings = resolve_effective_modes(args)
    local_sdk_capabilities = inspect_local_sdk_capabilities()
    resume_config = build_resume_config(args, dataset_path)
    reuse_memory_source = None
    if args.reuse_memory_checkpoint:
        reuse_memory_source = load_reuse_memory_checkpoint(Path(args.reuse_memory_checkpoint), dataset_path)

    resume_checkpoint = None
    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        resume_checkpoint = load_resume_checkpoint(checkpoint_path, resume_config)
        run_id = str(resume_checkpoint["run_id"])
        output_path = Path(args.output) if args.output else Path(resume_checkpoint["output_path"])
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (
            Path(args.output)
            if args.output
            else root_dir / "evaluation" / "results" / f"{args.method}_{run_id}.json"
        )
        checkpoint_path = (
            Path(args.checkpoint_path)
            if args.checkpoint_path
            else output_path.with_name(f"{output_path.name}.checkpoint.json")
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chat_client, chat_client_config = create_chat_client(args.api_max_retries, args.api_timeout_sec)
    embedding_client, embedding_client_config = create_embedding_client(
        args.api_max_retries,
        args.api_timeout_sec,
    )
    chat_api_key = get_env_value("BEEKNOEE_API_KEY")
    chat_base_url = get_env_value("BEEKNOEE_BASE_URL")
    embedding_api_key = get_env_value("OPENAI_API_KEY")

    print("Root dir:", root_dir)
    print("Dataset:", dataset_path)
    print("Output:", output_path)
    print("Checkpoint:", checkpoint_path)
    print("Resuming checkpoint:", bool(resume_checkpoint))
    print("reuse_memory_checkpoint:", reuse_memory_source["path"] if reuse_memory_source else None)
    print("qa_only_reuse_memory:", args.qa_only_reuse_memory)
    print("clear_reuse_embedding_cache:", args.clear_reuse_embedding_cache)
    if reuse_memory_source:
        print("reuse_memory_ingested_samples:", ", ".join(reuse_memory_source["ingested_sample_ids"]))
    print("Using LLM model:", args.llm_model)
    print("runtime_default_profile:", model_runtime_defaults["profile"])
    print("Chat API provider:", chat_client_config["provider"])
    print("Chat BEEKNOEE_BASE_URL configured:", chat_client_config["base_url_configured"])
    print("Using embedding model:", args.embedding_model)
    print("Embedding API provider:", embedding_client_config["provider"])
    print("memory_update_mode:", args.memory_update_mode)
    print("memory_wrapper_store:", args.memory_wrapper_store)
    print("update_similar_top_k:", args.update_similar_top_k)
    print("top_k:", args.top_k)
    print("batch_size:", args.batch_size)
    print("context_window:", args.context_window)
    print("candidate_extraction_max_tokens:", args.candidate_extraction_max_tokens)
    print("update_decision_max_tokens:", args.update_decision_max_tokens)
    print("answer_max_tokens:", args.answer_max_tokens)
    print("summary_max_tokens:", args.summary_max_tokens)
    print("include_context_in_add_prompt:", args.include_context_in_add_prompt)
    print("paper_extraction_mode:", args.paper_extraction_mode)
    print("effective_paper_extraction_mode:", effective_paper_extraction_mode)
    print("summary_mode:", args.summary_mode)
    print("effective_summary_mode:", effective_summary_mode)
    print("retrieval_token_budget:", args.retrieval_token_budget)
    print("retrieval_budget_strategy:", args.retrieval_budget_strategy)
    print("qa_retrieval_mode:", args.qa_retrieval_mode)
    print("active_reconstruction_max_steps:", args.active_reconstruction_max_steps)
    print("active_reconstruction_max_tool_calls_per_step:", args.active_reconstruction_max_tool_calls_per_step)
    if args.qa_retrieval_mode == "hybrid_adaptive":
        print(
            "active_reconstruction_seed_top_k:",
            (
                f"{args.active_reconstruction_seed_top_k} "
                f"(ignored by hybrid_adaptive; HAR seeds with har_active_seed_top_k={args.har_active_seed_top_k})"
            ),
        )
    else:
        print("active_reconstruction_seed_top_k:", args.active_reconstruction_seed_top_k)
    print("active_reconstruction_use_llm_router:", args.active_reconstruction_use_llm_router)
    print("active_reconstruction_router_max_tokens:", args.active_reconstruction_router_max_tokens)
    print("hybrid_router_mode:", args.hybrid_router_mode)
    print("protected_semantic_top_k:", args.protected_semantic_top_k)
    print("active_extra_k_per_speaker:", args.active_extra_k_per_speaker)
    print("har_active_seed_top_k:", args.har_active_seed_top_k)
    print("har_evidence_rerank_mode:", args.har_evidence_rerank_mode)
    print("har_evidence_rerank_trigger:", args.har_evidence_rerank_trigger)
    print("har_evidence_rerank_max_candidates:", args.har_evidence_rerank_max_candidates)
    print("har_evidence_rerank_max_tokens:", args.har_evidence_rerank_max_tokens)
    print("har_sufficiency_mode:", args.har_sufficiency_mode)
    print("har_sufficiency_max_rounds:", args.har_sufficiency_max_rounds)
    print("har_sufficiency_max_candidates:", args.har_sufficiency_max_candidates)
    print("har_sufficiency_max_tokens:", args.har_sufficiency_max_tokens)
    print("egc_contract_mode:", args.egc_contract_mode)
    print("egc_coverage_mode:", args.egc_coverage_mode)
    print("egc_max_gap_rounds:", args.egc_max_gap_rounds)
    print("egc_max_slots:", args.egc_max_slots)
    print("egc_min_slot_coverage:", args.egc_min_slot_coverage)
    print("egc_gap_top_k_per_slot:", args.egc_gap_top_k_per_slot)
    print("egc_max_gap_evidence_per_speaker:", args.egc_max_gap_evidence_per_speaker)
    print("egc_proof_pack_rerank:", args.egc_proof_pack_rerank)
    print("egc_keep_protected_baseline:", args.egc_keep_protected_baseline)
    print("qase_max_total_final_memories:", args.qase_max_total_final_memories)
    print("qase_candidate_pool_scale:", args.qase_candidate_pool_scale)
    print("qase_final_k_scale:", args.qase_final_k_scale)
    print("qase_budget_profile:", args.qase_budget_profile)
    print("qase_budget_allocation_mode:", args.qase_budget_allocation_mode)
    print("qase_selection_scope:", args.qase_selection_scope)
    print("qase_adaptive_cutoff_mode:", args.qase_adaptive_cutoff_mode or "profile_default")
    print("qase_bm25_candidate_rerank:", args.qase_bm25_candidate_rerank)
    print("qase_bm25_params:", {"k1": args.qase_bm25_k1, "b": args.qase_bm25_b})
    print("qase_lexical_weight:", args.qase_lexical_weight)
    print(
        "qase_hybrid_fusion_weights:",
        {
            "semantic": args.qase_hybrid_semantic_weight,
            "bm25": args.qase_hybrid_bm25_weight,
            "entity": args.qase_hybrid_entity_weight,
            "temporal": args.qase_hybrid_temporal_weight,
        },
    )
    print("qase_disable_adaptive_cutoff:", args.qase_disable_adaptive_cutoff)
    print("qase_disable_diversity:", args.qase_disable_diversity)
    print("qase_diversity_lambda:", args.qase_diversity_lambda)
    print("qase_disable_confidence_fallback:", args.qase_disable_confidence_fallback)
    print("qase_confidence_fallback_low_score_threshold:", args.qase_confidence_fallback_low_score_threshold)
    print("qase_planner_model_root:", args.qase_planner_model_root)
    print("qase_question_type_model_dir:", args.qase_question_type_model_dir)
    print("qase_target_speaker_model_dir:", args.qase_target_speaker_model_dir)
    print("qase_planner_model_device:", args.qase_planner_model_device)
    print("search_threshold:", args.search_threshold)
    print("api_max_retries:", args.api_max_retries)
    print("api_timeout_sec:", args.api_timeout_sec)
    if configuration_warnings:
        print("Configuration warnings:")
        for warning in configuration_warnings:
            print("  -", warning)

    token_encoding = tiktoken.get_encoding(args.retrieved_memory_token_encoding)
    custom_instructions = select_custom_instructions(args)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.max_conversations is not None:
        dataset = dataset[: args.max_conversations]

    config = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": args.llm_model,
                "api_key": chat_api_key,
                **({"openai_base_url": chat_base_url} if chat_base_url else {}),
                "temperature": 0,
                "max_tokens": args.memory_max_tokens,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": args.embedding_model,
                "api_key": embedding_api_key,
                "openai_base_url": "https://api.openai.com/v1",
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": f"{args.collection_prefix}_{run_id}",
                "path": str(root_dir / args.qdrant_path),
            },
        },
        "custom_instructions": custom_instructions,
    }

    memory = Memory.from_config(config) if args.memory_update_mode == "sdk_additive" else None

    def embedding_client_factory() -> OpenAI:
        return create_embedding_client(args.api_max_retries, args.api_timeout_sec)[0]

    wrapper_store = (
        LocalPaperMemoryStore(
            embedding_model=args.embedding_model,
            embedding_client=embedding_client,
            embedding_client_factory=embedding_client_factory,
            embedding_recovery_attempts=max(args.api_max_retries, 1),
        )
        if args.memory_update_mode == "paper_update_wrapper"
        else None
    )
    qase_config_kwargs = {
        "max_total_final_memories": args.qase_max_total_final_memories,
        "candidate_pool_scale": args.qase_candidate_pool_scale,
        "final_k_scale": args.qase_final_k_scale,
        "budget_allocation_mode": args.qase_budget_allocation_mode,
        "adaptive_cutoff_enabled": not args.qase_disable_adaptive_cutoff,
        "diversity_lambda_complex": 0.0 if args.qase_disable_diversity else args.qase_diversity_lambda,
        "diversity_lambda_simple": 0.0 if args.qase_disable_diversity else min(args.qase_diversity_lambda, 0.08),
        "confidence_fallback_enabled": not args.qase_disable_confidence_fallback,
        "confidence_fallback_low_score_threshold": args.qase_confidence_fallback_low_score_threshold,
        "lexical_weight": args.qase_lexical_weight,
        "planner_model_root": args.qase_planner_model_root,
        "question_type_model_dir": args.qase_question_type_model_dir,
        "target_speaker_model_dir": args.qase_target_speaker_model_dir,
        "planner_model_device": args.qase_planner_model_device,
    }
    if args.qase_budget_profile == "simple_report":
        qase_config_kwargs.update(
            {
                # The report profile uses four planner groups:
                # single-hop, temporal, multi-hop, and fallback/ambiguous.
                "candidate_multi_hop_target": 20,
                "candidate_multi_hop_other": 20,
                "final_multi_hop_target": 10,
                "final_multi_hop_other": 10,
                # Keep the report-friendly selector small: scoring + adaptive cutoff + type minimum.
                "semantic_anchor_enabled": False,
                "confidence_fallback_enabled": False,
                "diversity_lambda_simple": 0.0,
                "diversity_lambda_complex": 0.0,
                "complex_min_fraction_of_cap": 0.0,
                "temporal_bonus_weight": 0.12,
                "adaptive_cutoff_mode": "largest_gap",
                "min_single_hop": 2,
                "min_temporal": 3,
                "min_multi_hop": 4,
                "min_fallback_ambiguous": 3,
            }
        )
    if args.qase_adaptive_cutoff_mode:
        qase_config_kwargs["adaptive_cutoff_mode"] = args.qase_adaptive_cutoff_mode
    qase_controller = QuestionAwareSelectiveEvidence(QASEConfig(**qase_config_kwargs))
    print("qase_effective_adaptive_cutoff_mode:", qase_controller.config.adaptive_cutoff_mode)
    if resume_checkpoint and args.memory_update_mode != "paper_update_wrapper":
        raise ValueError(
            "--resume_checkpoint currently supports only --memory_update_mode paper_update_wrapper; "
            "an interrupted sdk_additive sample may have partial persistent Qdrant writes."
        )

    add_logs = []
    results = []
    all_extracted_memories_by_sample = {}
    memory_add_output_tokens_by_sample = {}
    final_memory_store_tokens_by_sample = {}
    final_num_memories_by_sample = {}
    final_store_fetch_warnings = []
    add_event_counts = empty_event_counts()
    add_event_counts_by_sample = {}
    add_event_counts_by_speaker = {}
    add_event_counts_by_session = {}
    add_event_counts_by_batch = []
    summary_snapshots = []
    summary_usage_totals = {
        "summary_prompt_tokens": 0,
        "summary_completion_tokens": 0,
        "summary_total_tokens": 0,
        "summary_latency_sec": 0.0,
    }
    paper_update_logs = []
    candidate_extraction_usage_totals = {
        "candidate_extraction_prompt_tokens": 0,
        "candidate_extraction_completion_tokens": 0,
        "candidate_extraction_total_tokens": 0,
        "candidate_extraction_reasoning_tokens": 0,
        "candidate_extraction_latency_sec": 0.0,
    }
    update_decision_usage_totals = {
        "update_decision_prompt_tokens": 0,
        "update_decision_completion_tokens": 0,
        "update_decision_total_tokens": 0,
        "update_decision_reasoning_tokens": 0,
        "update_decision_latency_sec": 0.0,
    }
    information_content_guard_usage_totals = {
        "information_content_guard_prompt_tokens": 0,
        "information_content_guard_completion_tokens": 0,
        "information_content_guard_total_tokens": 0,
        "information_content_guard_reasoning_tokens": 0,
        "information_content_guard_latency_sec": 0.0,
    }
    wrapper_latency_totals = {
        "candidate_extraction_latency_sec": 0.0,
        "update_decision_latency_sec": 0.0,
        "information_content_guard_latency_sec": 0.0,
        "store_operation_latency_sec": 0.0,
    }
    active_reconstruction_usage_totals = empty_active_usage_totals()
    active_reconstruction_latency_totals = {
        "active_reconstruction_latency_sec": 0.0,
        "active_reconstruction_num_questions": 0,
    }
    num_candidate_extractions = 0
    num_wrapper_candidates = 0
    num_update_decisions = 0
    num_information_content_guards = 0
    num_updates_rejected_by_information_content_guard = 0

    total_qa_seen = 0
    total_qa_answered = 0
    completed_sample_ids = set()
    partial_sample_state = None
    reused_memory_source_info = None
    reuse_ingested_sample_ids = set()

    if reuse_memory_source and not resume_checkpoint:
        if not wrapper_store:
            raise ValueError("--reuse_memory_checkpoint requires --memory_update_mode paper_update_wrapper.")
        wrapper_store.load_state(reuse_memory_source["wrapper_store_state"])
        reuse_embedding_cache_entries = len(getattr(wrapper_store, "embedding_cache", {}) or {})
        if args.clear_reuse_embedding_cache:
            wrapper_store.embedding_cache = {}
        reused_memory_source_info = {
            "path": reuse_memory_source["path"],
            "run_id": reuse_memory_source.get("run_id"),
            "embedding_cache_entries_loaded": reuse_embedding_cache_entries,
            "embedding_cache_cleared": bool(args.clear_reuse_embedding_cache),
        }
        reuse_ingested_sample_ids = set(reuse_memory_source["ingested_sample_ids"])
        print(
            "Loaded reusable ingested memory:",
            len((reuse_memory_source["wrapper_store_state"].get("memories_by_user") or {})),
            "speaker namespace(s)",
        )
        print(
            "Reusable embedding cache:",
            reuse_embedding_cache_entries,
            "entry(s)",
            "- cleared" if args.clear_reuse_embedding_cache else "- kept",
        )

    if resume_checkpoint:
        checkpoint_state = resume_checkpoint.get("state") or {}
        add_logs = checkpoint_state.get("add_logs", add_logs)
        results = checkpoint_state.get("results", results)
        all_extracted_memories_by_sample = checkpoint_state.get(
            "all_extracted_memories_by_sample",
            all_extracted_memories_by_sample,
        )
        memory_add_output_tokens_by_sample = checkpoint_state.get(
            "memory_add_output_tokens_by_sample",
            memory_add_output_tokens_by_sample,
        )
        final_memory_store_tokens_by_sample = checkpoint_state.get(
            "final_memory_store_tokens_by_sample",
            final_memory_store_tokens_by_sample,
        )
        final_num_memories_by_sample = checkpoint_state.get(
            "final_num_memories_by_sample",
            final_num_memories_by_sample,
        )
        final_store_fetch_warnings = checkpoint_state.get(
            "final_store_fetch_warnings",
            final_store_fetch_warnings,
        )
        add_event_counts = checkpoint_state.get("add_event_counts", add_event_counts)
        add_event_counts_by_sample = checkpoint_state.get(
            "add_event_counts_by_sample",
            add_event_counts_by_sample,
        )
        add_event_counts_by_speaker = checkpoint_state.get(
            "add_event_counts_by_speaker",
            add_event_counts_by_speaker,
        )
        add_event_counts_by_session = checkpoint_state.get(
            "add_event_counts_by_session",
            add_event_counts_by_session,
        )
        add_event_counts_by_batch = checkpoint_state.get(
            "add_event_counts_by_batch",
            add_event_counts_by_batch,
        )
        summary_snapshots = checkpoint_state.get("summary_snapshots", summary_snapshots)
        summary_usage_totals = checkpoint_state.get("summary_usage_totals", summary_usage_totals)
        paper_update_logs = checkpoint_state.get("paper_update_logs", paper_update_logs)
        candidate_extraction_usage_totals = checkpoint_state.get(
            "candidate_extraction_usage_totals",
            candidate_extraction_usage_totals,
        )
        update_decision_usage_totals = checkpoint_state.get(
            "update_decision_usage_totals",
            update_decision_usage_totals,
        )
        information_content_guard_usage_totals = checkpoint_state.get(
            "information_content_guard_usage_totals",
            information_content_guard_usage_totals,
        )
        wrapper_latency_totals = checkpoint_state.get(
            "wrapper_latency_totals",
            wrapper_latency_totals,
        )
        active_reconstruction_usage_totals = checkpoint_state.get(
            "active_reconstruction_usage_totals",
            active_reconstruction_usage_totals,
        )
        active_reconstruction_latency_totals = checkpoint_state.get(
            "active_reconstruction_latency_totals",
            active_reconstruction_latency_totals,
        )
        num_candidate_extractions = int(
            checkpoint_state.get("num_candidate_extractions", num_candidate_extractions)
        )
        num_wrapper_candidates = int(checkpoint_state.get("num_wrapper_candidates", num_wrapper_candidates))
        num_update_decisions = int(checkpoint_state.get("num_update_decisions", num_update_decisions))
        num_information_content_guards = int(
            checkpoint_state.get("num_information_content_guards", num_information_content_guards)
        )
        num_updates_rejected_by_information_content_guard = int(
            checkpoint_state.get(
                "num_updates_rejected_by_information_content_guard",
                num_updates_rejected_by_information_content_guard,
            )
        )
        total_qa_seen = int(checkpoint_state.get("total_qa_seen", total_qa_seen))
        total_qa_answered = int(checkpoint_state.get("total_qa_answered", total_qa_answered))
        completed_sample_ids = set(resume_checkpoint.get("completed_sample_ids") or [])
        partial_sample_state = checkpoint_state.get("partial_sample_state")
        reused_memory_source_info = checkpoint_state.get("reused_memory_source_info")
        reuse_ingested_sample_ids = set(checkpoint_state.get("reuse_ingested_sample_ids") or [])
        wrapper_store_state = checkpoint_state.get("wrapper_store_state")
        if wrapper_store_state:
            wrapper_store.load_state(wrapper_store_state)
        elif completed_sample_ids:
            raise ValueError(
                "Checkpoint does not contain LocalPaperMemoryStore state. "
                "Restart this run from scratch with the patched runner so future checkpoints can resume safely."
            )
        wrapper_store.embedding_recovery_log = checkpoint_state.get(
            "embedding_recovery_log",
            wrapper_store.embedding_recovery_log,
        )
        print(f"Checkpoint contains {len(completed_sample_ids)} completed conversation(s).")

    def current_checkpoint_state() -> Dict[str, Any]:
        return {
            "add_logs": add_logs,
            "results": results,
            "all_extracted_memories_by_sample": all_extracted_memories_by_sample,
            "memory_add_output_tokens_by_sample": memory_add_output_tokens_by_sample,
            "final_memory_store_tokens_by_sample": final_memory_store_tokens_by_sample,
            "final_num_memories_by_sample": final_num_memories_by_sample,
            "final_store_fetch_warnings": final_store_fetch_warnings,
            "add_event_counts": add_event_counts,
            "add_event_counts_by_sample": add_event_counts_by_sample,
            "add_event_counts_by_speaker": add_event_counts_by_speaker,
            "add_event_counts_by_session": add_event_counts_by_session,
            "add_event_counts_by_batch": add_event_counts_by_batch,
            "summary_snapshots": summary_snapshots,
            "summary_usage_totals": summary_usage_totals,
            "paper_update_logs": paper_update_logs,
            "candidate_extraction_usage_totals": candidate_extraction_usage_totals,
            "update_decision_usage_totals": update_decision_usage_totals,
            "information_content_guard_usage_totals": information_content_guard_usage_totals,
            "wrapper_latency_totals": wrapper_latency_totals,
            "active_reconstruction_usage_totals": active_reconstruction_usage_totals,
            "active_reconstruction_latency_totals": active_reconstruction_latency_totals,
            "num_candidate_extractions": num_candidate_extractions,
            "num_wrapper_candidates": num_wrapper_candidates,
            "num_update_decisions": num_update_decisions,
            "num_information_content_guards": num_information_content_guards,
            "num_updates_rejected_by_information_content_guard": (
                num_updates_rejected_by_information_content_guard
            ),
            "total_qa_seen": total_qa_seen,
            "total_qa_answered": total_qa_answered,
            "partial_sample_state": partial_sample_state,
            "reused_memory_source_info": reused_memory_source_info,
            "reuse_ingested_sample_ids": sorted(reuse_ingested_sample_ids),
            "embedding_recovery_log": wrapper_store.embedding_recovery_log if wrapper_store else [],
            "wrapper_store_state": wrapper_store.export_state() if wrapper_store else None,
        }

    for conv_idx, sample in enumerate(dataset):
        display_idx = conv_idx + 1
        sample_id = sample.get("sample_id", f"sample_{display_idx}")
        if sample_id in completed_sample_ids:
            print(f"[{display_idx}/{len(dataset)}] Skipping completed sample: {sample_id}")
            continue
        conversation = sample["conversation"]
        qa_list = sample.get("qa", [])
        dia_lookup = build_dia_lookup(conversation)

        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")
        if not speaker_a or not speaker_b:
            raise ValueError(f"Missing speaker_a/speaker_b for sample {sample_id}")

        partial_for_sample = (
            partial_sample_state
            if partial_sample_state and partial_sample_state.get("sample_id") == sample_id
            else None
        )
        speaker_user_ids = {
            speaker_a: make_user_id(sample_id, conv_idx, speaker_a, run_id),
            speaker_b: make_user_id(sample_id, conv_idx, speaker_b, run_id),
        }
        reused_ingestion_for_sample = False
        partial_speaker_user_ids = partial_for_sample.get("speaker_user_ids") if partial_for_sample else None
        if partial_speaker_user_ids:
            speaker_user_ids = {
                speaker_a: partial_speaker_user_ids.get(speaker_a, speaker_user_ids[speaker_a]),
                speaker_b: partial_speaker_user_ids.get(speaker_b, speaker_user_ids[speaker_b]),
            }
        elif args.qa_only_reuse_memory:
            if sample_id not in reuse_ingested_sample_ids:
                raise ValueError(
                    f"--qa_only_reuse_memory cannot answer {sample_id}: "
                    "the reuse checkpoint does not mark this sample as ingestion-complete."
                )
            current_store_state = wrapper_store.export_state() if wrapper_store else {}
            speaker_a_reuse_user_id = find_reused_speaker_user_id(
                current_store_state,
                sample_id,
                conv_idx,
                speaker_a,
            )
            speaker_b_reuse_user_id = find_reused_speaker_user_id(
                current_store_state,
                sample_id,
                conv_idx,
                speaker_b,
            )
            if not speaker_a_reuse_user_id or not speaker_b_reuse_user_id:
                raise ValueError(
                    f"--qa_only_reuse_memory cannot map reused speaker memories for {sample_id}. "
                    f"Missing: {speaker_a if not speaker_a_reuse_user_id else ''} "
                    f"{speaker_b if not speaker_b_reuse_user_id else ''}".strip()
                )
            speaker_user_ids = {
                speaker_a: speaker_a_reuse_user_id,
                speaker_b: speaker_b_reuse_user_id,
            }
            reused_ingestion_for_sample = True

        print("\n" + "=" * 100)
        print(f"[{display_idx}/{len(dataset)}] Sample: {sample_id}")
        print(f"Speakers: {speaker_a} -> {speaker_user_ids[speaker_a]}, {speaker_b} -> {speaker_user_ids[speaker_b]}")
        if reused_ingestion_for_sample:
            print(f"Reusing ingested memory for {sample_id}; ingestion will be skipped.")

        if partial_for_sample:
            print(f"Resuming partial sample: {sample_id} | phase={partial_for_sample.get('phase')}")
            sample_all_memories = partial_for_sample.get("sample_all_memories") or {
                speaker_a: [],
                speaker_b: [],
            }
            speaker_history_messages = partial_for_sample.get("speaker_history_messages") or {
                speaker_a: [],
                speaker_b: [],
            }
            rolling_summaries = partial_for_sample.get("rolling_summaries") or {
                speaker_a: "",
                speaker_b: "",
            }
            sample_event_counts = partial_for_sample.get("sample_event_counts") or {
                "total": empty_event_counts(),
                speaker_a: empty_event_counts(),
                speaker_b: empty_event_counts(),
            }
            completed_ingest_batches = set(partial_for_sample.get("completed_ingest_batches") or [])
            completed_summary_updates = set(partial_for_sample.get("completed_summary_updates") or [])
            completed_qa_keys = set(partial_for_sample.get("completed_qa_keys") or [])
            partial_ingestion_complete = bool(partial_for_sample.get("ingestion_complete"))
        else:
            sample_all_memories = {speaker_a: [], speaker_b: []}
            speaker_history_messages = {speaker_a: [], speaker_b: []}
            rolling_summaries = {speaker_a: "", speaker_b: ""}
            sample_event_counts = {
                "total": empty_event_counts(),
                speaker_a: empty_event_counts(),
                speaker_b: empty_event_counts(),
            }
            completed_ingest_batches = set()
            completed_summary_updates = set()
            completed_qa_keys = set()
            partial_ingestion_complete = False

        sample_all_memories.setdefault(speaker_a, [])
        sample_all_memories.setdefault(speaker_b, [])
        speaker_history_messages.setdefault(speaker_a, [])
        speaker_history_messages.setdefault(speaker_b, [])
        rolling_summaries.setdefault(speaker_a, "")
        rolling_summaries.setdefault(speaker_b, "")
        sample_event_counts.setdefault("total", empty_event_counts())
        sample_event_counts.setdefault(speaker_a, empty_event_counts())
        sample_event_counts.setdefault(speaker_b, empty_event_counts())
        if reused_ingestion_for_sample and not partial_for_sample:
            partial_ingestion_complete = True

        def update_partial_sample_state(phase: str, *, ingestion_complete: Optional[bool] = None) -> None:
            nonlocal partial_sample_state
            partial_sample_state = {
                "sample_id": sample_id,
                "display_idx": display_idx,
                "conv_idx": conv_idx,
                "phase": phase,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "speaker_user_ids": speaker_user_ids,
                "sample_all_memories": sample_all_memories,
                "speaker_history_messages": speaker_history_messages,
                "rolling_summaries": rolling_summaries,
                "sample_event_counts": sample_event_counts,
                "completed_ingest_batches": sorted(completed_ingest_batches),
                "completed_summary_updates": sorted(completed_summary_updates),
                "completed_qa_keys": sorted(completed_qa_keys),
                "ingestion_complete": (
                    partial_ingestion_complete if ingestion_complete is None else ingestion_complete
                ),
                "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }

        def save_progress_checkpoint(label: str) -> None:
            write_run_checkpoint(
                checkpoint_path=checkpoint_path,
                run_id=run_id,
                output_path=output_path,
                resume_config=resume_config,
                completed_sample_ids=completed_sample_ids,
                state=current_checkpoint_state(),
            )
            print(f"Checkpoint saved after {label}: {checkpoint_path}")

        session_indices = get_session_indices(conversation)
        if args.max_sessions is not None:
            session_indices = session_indices[: args.max_sessions]

        for session_index in session_indices:
            if partial_ingestion_complete:
                continue
            session_key = f"session_{session_index}"
            date_key = f"session_{session_index}_date_time"
            session_date = conversation.get(date_key)
            session_time = observation_time_fields(session_date)
            session = conversation.get(session_key) or []

            if not session:
                continue

            speaker_views = build_speaker_views(session, speaker_a, speaker_b, session_index, session_date)
            print(f"Adding {session_key}: {len(session)} turns | timestamp={session_date}")

            for perspective_speaker in (speaker_a, speaker_b):
                user_id = speaker_user_ids[perspective_speaker]
                messages = speaker_views[perspective_speaker]
                speaker_memory_count = 0
                speaker_latency = 0.0
                num_batches = 0

                for batch_start, batch_messages in chunked(messages, args.batch_size):
                    batch_key = f"{sample_id}:{session_key}:{perspective_speaker}:batch:{batch_start}"
                    batch_summary_key = f"{sample_id}:{session_key}:{perspective_speaker}:batch_summary:{batch_start}"
                    if batch_key in completed_ingest_batches:
                        num_batches += 1
                        if (
                            effective_summary_mode == "rolling_llm"
                            and args.summary_update_scope == "after_each_batch"
                            and batch_summary_key not in completed_summary_updates
                        ):
                            previous_summary = rolling_summaries[perspective_speaker]
                            updated_summary, summary_log = update_rolling_summary(
                                chat_client,
                                summary_model=args.summary_model,
                                summary_max_tokens=args.summary_max_tokens,
                                previous_summary=previous_summary,
                                new_messages=batch_messages,
                                perspective_speaker=perspective_speaker,
                            )
                            rolling_summaries[perspective_speaker] = updated_summary
                            for key in summary_usage_totals:
                                summary_usage_totals[key] += summary_log[key]
                            summary_snapshots.append(
                                {
                                    "sample_id": sample_id,
                                    "perspective_speaker": perspective_speaker,
                                    "session": session_key,
                                    "batch_start": batch_start,
                                    "update_scope": "after_each_batch",
                                    "summary": updated_summary,
                                    **summary_log,
                                }
                            )
                            completed_summary_updates.add(batch_summary_key)
                            update_partial_sample_state("ingesting", ingestion_complete=False)
                            save_progress_checkpoint(
                                f"{sample_id}/{session_key}/{perspective_speaker}/batch_{batch_start}_summary"
                            )
                        continue

                    num_batches += 1
                    history = speaker_history_messages[perspective_speaker]
                    num_context_messages_available = len(history)
                    context_messages = history[-args.context_window :] if args.context_window else []
                    context_start = num_context_messages_available - len(context_messages)
                    add_messages = build_add_messages(
                        context_messages=context_messages,
                        target_messages=batch_messages,
                        paper_extraction_mode=effective_paper_extraction_mode,
                        summary_text=rolling_summaries[perspective_speaker],
                    )
                    num_context_messages_sent = (
                        len(context_messages)
                        if effective_paper_extraction_mode in {"include_m_context", "include_summary_and_m_context"}
                        else 0
                    )
                    num_summary_messages_sent = (
                        1 if effective_paper_extraction_mode == "include_summary_and_m_context" else 0
                    )
                    metadata = build_add_metadata(
                        sample_id=sample_id,
                        run_id=run_id,
                        session_key=session_key,
                        session_index=session_index,
                        session_date=session_date,
                        observed_at=session_time["observed_at"],
                        perspective_speaker=perspective_speaker,
                        speaker_a=speaker_a,
                        speaker_b=speaker_b,
                        batch_start=batch_start,
                        batch_size=args.batch_size,
                        context_window=args.context_window,
                        context_start=context_start,
                        num_context_messages_available=num_context_messages_available,
                        num_context_messages=len(context_messages),
                        num_context_messages_sent=num_context_messages_sent,
                        num_summary_messages_sent=num_summary_messages_sent,
                        num_target_messages=len(batch_messages),
                        num_messages_added=len(add_messages),
                    )

                    if args.memory_update_mode == "sdk_additive":
                        start = time.time()
                        add_result = memory.add(add_messages, user_id=user_id, metadata=metadata)
                        add_latency = time.time() - start
                        added_items = add_result.get("results", []) if isinstance(add_result, dict) else []
                    else:
                        wrapper_summary_text = (
                            rolling_summaries[perspective_speaker]
                            if effective_paper_extraction_mode == "include_summary_and_m_context"
                            else None
                        )
                        wrapper_context_messages = (
                            context_messages
                            if effective_paper_extraction_mode in {"include_m_context", "include_summary_and_m_context"}
                            else []
                        )
                        batch_start_time = time.time()
                        candidates, extraction_log = extract_candidate_memories(
                            chat_client,
                            args.llm_model,
                            wrapper_summary_text,
                            wrapper_context_messages,
                            batch_messages,
                            session_date,
                            session_time["observed_at"],
                            perspective_speaker,
                            args.candidate_extraction_max_tokens,
                        )
                        num_candidate_extractions += 1
                        update_usage_totals(
                            candidate_extraction_usage_totals,
                            extraction_log,
                            "candidate_extraction",
                        )
                        wrapper_latency_totals["candidate_extraction_latency_sec"] += float(
                            extraction_log.get("candidate_extraction_latency_sec") or 0.0
                        )

                        added_items = []
                        wrapper_operations = []
                        for candidate_index, candidate in enumerate(candidates):
                            num_wrapper_candidates += 1
                            existing_memories = wrapper_store.search(
                                user_id=user_id,
                                query=candidate.get("memory", ""),
                                top_k=args.update_similar_top_k,
                            )
                            decision, decision_log = decide_memory_operation(
                                chat_client,
                                args.llm_model,
                                candidate,
                                existing_memories,
                                args.update_decision_max_tokens,
                            )
                            num_update_decisions += 1
                            update_usage_totals(update_decision_usage_totals, decision_log, "update_decision")
                            wrapper_latency_totals["update_decision_latency_sec"] += float(
                                decision_log.get("update_decision_latency_sec") or 0.0
                            )

                            classified_operation = decision["operation"]
                            classified_target_memory_id = decision.get("target_memory_id")
                            information_content_guard = None
                            information_content_guard_log = {}
                            if classified_operation == "UPDATE":
                                target_memory = next(
                                    (
                                        item
                                        for item in existing_memories
                                        if str(item.get("id")) == str(classified_target_memory_id)
                                    ),
                                    None,
                                )
                                if target_memory is None:
                                    information_content_guard = {
                                        "candidate_is_strictly_richer": False,
                                        "reason": "UPDATE rejected because its target memory was not retrieved.",
                                        "fallback": True,
                                    }
                                else:
                                    information_content_guard, information_content_guard_log = (
                                        verify_update_information_content(
                                            chat_client,
                                            args.llm_model,
                                            candidate,
                                            target_memory,
                                            args.update_decision_max_tokens,
                                        )
                                    )
                                    num_information_content_guards += 1
                                    update_usage_totals(
                                        information_content_guard_usage_totals,
                                        information_content_guard_log,
                                        "information_content_guard",
                                    )
                                    wrapper_latency_totals["information_content_guard_latency_sec"] += float(
                                        information_content_guard_log.get(
                                            "information_content_guard_latency_sec"
                                        )
                                        or 0.0
                                    )

                                if not information_content_guard["candidate_is_strictly_richer"]:
                                    num_updates_rejected_by_information_content_guard += 1
                                    decision = {
                                        **decision,
                                        "operation": "NOOP",
                                        "target_memory_id": None,
                                        "new_memory": None,
                                        "reason": information_content_guard["reason"],
                                        "fallback": bool(
                                            decision.get("fallback")
                                            or information_content_guard.get("fallback")
                                        ),
                                        "update_rejected_by_information_content_guard": True,
                                    }

                            operation = decision["operation"]
                            operation_start = time.time()
                            target_memory_id = decision.get("target_memory_id")
                            memory_text = candidate.get("memory", "")
                            new_memory = decision.get("new_memory") or memory_text
                            applied = True
                            output_memory_id = target_memory_id

                            operation_metadata = {
                                **metadata,
                                "candidate_memory_type": candidate.get("type"),
                                "candidate_memory_importance": candidate.get("importance"),
                                "event_at": candidate.get("event_at"),
                                "event_time_text": candidate.get("event_time_text"),
                                "memory_update_mode": args.memory_update_mode,
                                "memory_wrapper_store": args.memory_wrapper_store,
                            }

                            if operation == "ADD":
                                output_memory_id = wrapper_store.add_memory(
                                    user_id=user_id,
                                    memory_text=new_memory,
                                    metadata=operation_metadata,
                                )
                            elif operation == "UPDATE":
                                applied = wrapper_store.update_memory(
                                    user_id=user_id,
                                    memory_id=target_memory_id,
                                    new_memory_text=new_memory,
                                    metadata_update=operation_metadata,
                                )
                            elif operation == "DELETE":
                                applied = wrapper_store.delete_memory(user_id=user_id, memory_id=target_memory_id)
                            elif operation == "NOOP":
                                applied = True
                            else:
                                operation = "ADD"
                                output_memory_id = wrapper_store.add_memory(
                                    user_id=user_id,
                                    memory_text=memory_text,
                                    metadata=operation_metadata,
                                )
                                applied = True

                            store_operation_latency = time.time() - operation_start
                            wrapper_latency_totals["store_operation_latency_sec"] += store_operation_latency
                            added_item = {
                                "id": output_memory_id,
                                "memory": new_memory if operation != "DELETE" else memory_text,
                                "event": operation,
                                "metadata": operation_metadata,
                                "candidate_memory": candidate,
                                "target_memory_id": target_memory_id,
                                "new_memory": new_memory if operation in {"ADD", "UPDATE"} else None,
                                "reason": decision.get("reason"),
                                "applied": applied,
                                "fallback": decision.get("fallback", False),
                            }
                            added_items.append(added_item)

                            operation_log = {
                                "sample_id": sample_id,
                                "session": session_key,
                                "session_index": session_index,
                                "session_date": session_date,
                                "session_timestamp_raw": session_time["session_timestamp_raw"],
                                "observed_at": session_time["observed_at"],
                                "perspective_speaker": perspective_speaker,
                                "user_id": user_id,
                                "batch_start": batch_start,
                                "candidate_index": candidate_index,
                                "candidate_memory": candidate,
                                "num_existing_memories": len(existing_memories),
                                "update_similar_top_k": args.update_similar_top_k,
                                "existing_memories": existing_memories,
                                "operation": operation,
                                "classified_operation": classified_operation,
                                "classified_target_memory_id": classified_target_memory_id,
                                "target_memory_id": target_memory_id,
                                "new_memory": added_item["new_memory"],
                                "reason": decision.get("reason"),
                                "applied": applied,
                                "fallback": decision.get("fallback", False),
                                "information_content_guard": information_content_guard,
                                "store_operation_latency_sec": store_operation_latency,
                                **decision_log,
                                **information_content_guard_log,
                            }
                            wrapper_operations.append(operation_log)
                            paper_update_logs.append(operation_log)

                        add_latency = time.time() - batch_start_time
                        add_result = {
                            "results": added_items,
                            "memory_update_mode": args.memory_update_mode,
                            "memory_wrapper_store": args.memory_wrapper_store,
                            "candidate_extraction": extraction_log,
                            "num_candidate_memories": len(candidates),
                            "paper_update_operations": wrapper_operations,
                            "note": "Mem0-inspired local prototype; not official Mem0 SDK implementation.",
                        }
                    batch_event_counts = event_counts_for(added_items)
                    sample_all_memories[perspective_speaker].extend(added_items)
                    update_event_counts(add_event_counts, added_items)
                    update_event_counts(sample_event_counts["total"], added_items)
                    update_event_counts(sample_event_counts[perspective_speaker], added_items)
                    speaker_event_key = f"{sample_id}:{perspective_speaker}"
                    session_event_key = f"{sample_id}:{session_key}"
                    add_event_counts_by_speaker.setdefault(speaker_event_key, empty_event_counts())
                    add_event_counts_by_session.setdefault(session_event_key, empty_event_counts())
                    update_event_counts(add_event_counts_by_speaker[speaker_event_key], added_items)
                    update_event_counts(add_event_counts_by_session[session_event_key], added_items)
                    speaker_memory_count += len(added_items)
                    speaker_latency += add_latency
                    speaker_history_messages[perspective_speaker].extend(batch_messages)

                    add_logs.append(
                        {
                            "sample_id": sample_id,
                            "user_id": user_id,
                            "perspective_speaker": perspective_speaker,
                            "session": session_key,
                            "session_index": session_index,
                            "session_date": session_date,
                            "session_timestamp_raw": session_time["session_timestamp_raw"],
                            "observed_at": session_time["observed_at"],
                            "batch_start": batch_start,
                            "batch_size": args.batch_size,
                            "actual_batch_size": len(batch_messages),
                            "context_window": args.context_window,
                            "context_start": context_start,
                            "num_context_messages_available": num_context_messages_available,
                            "num_context_messages": len(context_messages),
                            "num_context_messages_in_window": len(context_messages),
                            "num_context_messages_sent": num_context_messages_sent,
                            "num_summary_messages_sent": num_summary_messages_sent,
                            "num_target_messages": len(batch_messages),
                            "num_target_messages_sent": len(batch_messages),
                            "num_messages_added": len(add_messages),
                            "include_context_in_add_prompt": num_context_messages_sent > 0,
                            "paper_extraction_mode": args.paper_extraction_mode,
                            "effective_paper_extraction_mode": effective_paper_extraction_mode,
                            "summary_mode": args.summary_mode,
                            "effective_summary_mode": effective_summary_mode,
                            "memory_update_mode": args.memory_update_mode,
                            "memory_wrapper_store": args.memory_wrapper_store,
                            "update_similar_top_k": args.update_similar_top_k,
                            "num_turns": len(batch_messages),
                            "num_memories": len(added_items),
                            "event_counts": batch_event_counts,
                            "latency_sec": add_latency,
                            "add_result": add_result,
                        }
                    )
                    add_event_counts_by_batch.append(
                        {
                            "sample_id": sample_id,
                            "user_id": user_id,
                            "perspective_speaker": perspective_speaker,
                            "session": session_key,
                            "batch_start": batch_start,
                            "event_counts": batch_event_counts,
                        }
                    )
                    completed_ingest_batches.add(batch_key)
                    update_partial_sample_state("ingesting", ingestion_complete=False)
                    save_progress_checkpoint(f"{sample_id}/{session_key}/{perspective_speaker}/batch_{batch_start}")

                    if (
                        effective_summary_mode == "rolling_llm"
                        and args.summary_update_scope == "after_each_batch"
                        and batch_summary_key not in completed_summary_updates
                    ):
                        previous_summary = rolling_summaries[perspective_speaker]
                        updated_summary, summary_log = update_rolling_summary(
                            chat_client,
                            summary_model=args.summary_model,
                            summary_max_tokens=args.summary_max_tokens,
                            previous_summary=previous_summary,
                            new_messages=batch_messages,
                            perspective_speaker=perspective_speaker,
                        )
                        rolling_summaries[perspective_speaker] = updated_summary
                        for key in summary_usage_totals:
                            summary_usage_totals[key] += summary_log[key]
                        summary_snapshots.append(
                            {
                                "sample_id": sample_id,
                                "perspective_speaker": perspective_speaker,
                                "session": session_key,
                                "batch_start": batch_start,
                                "update_scope": "after_each_batch",
                                "summary": updated_summary,
                                **summary_log,
                            }
                        )
                        completed_summary_updates.add(batch_summary_key)
                        update_partial_sample_state("ingesting", ingestion_complete=False)
                        save_progress_checkpoint(
                            f"{sample_id}/{session_key}/{perspective_speaker}/batch_{batch_start}_summary"
                        )

                print(
                    f"  {perspective_speaker}: batches={num_batches} | "
                    f"memories={speaker_memory_count} | add_latency={speaker_latency:.2f}s"
                )

                session_summary_key = f"{sample_id}:{session_key}:{perspective_speaker}:session_summary"
                if (
                    effective_summary_mode == "rolling_llm"
                    and args.summary_update_scope == "after_each_session"
                    and session_summary_key not in completed_summary_updates
                ):
                    previous_summary = rolling_summaries[perspective_speaker]
                    updated_summary, summary_log = update_rolling_summary(
                        chat_client,
                        summary_model=args.summary_model,
                        summary_max_tokens=args.summary_max_tokens,
                        previous_summary=previous_summary,
                        new_messages=messages,
                        perspective_speaker=perspective_speaker,
                    )
                    rolling_summaries[perspective_speaker] = updated_summary
                    for key in summary_usage_totals:
                        summary_usage_totals[key] += summary_log[key]
                    summary_snapshots.append(
                        {
                            "sample_id": sample_id,
                            "perspective_speaker": perspective_speaker,
                            "session": session_key,
                            "update_scope": "after_each_session",
                            "summary": updated_summary,
                            **summary_log,
                        }
                    )
                    completed_summary_updates.add(session_summary_key)
                    update_partial_sample_state("ingesting", ingestion_complete=False)
                    save_progress_checkpoint(f"{sample_id}/{session_key}/{perspective_speaker}/session_summary")

        partial_ingestion_complete = True
        update_partial_sample_state("qa", ingestion_complete=True)
        save_progress_checkpoint(f"{sample_id}/ingestion_complete")

        all_extracted_memories_by_sample[sample_id] = sample_all_memories
        add_event_counts_by_sample[sample_id] = sample_event_counts
        speaker_a_store_tokens = count_memory_text_tokens(token_encoding, sample_all_memories[speaker_a])
        speaker_b_store_tokens = count_memory_text_tokens(token_encoding, sample_all_memories[speaker_b])
        memory_add_output_tokens_by_sample[sample_id] = {
            speaker_a: speaker_a_store_tokens,
            speaker_b: speaker_b_store_tokens,
            "total": speaker_a_store_tokens + speaker_b_store_tokens,
            "num_memories": {
                speaker_a: len(sample_all_memories[speaker_a]),
                speaker_b: len(sample_all_memories[speaker_b]),
                "total": len(sample_all_memories[speaker_a]) + len(sample_all_memories[speaker_b]),
            },
            "token_encoding": args.retrieved_memory_token_encoding,
            "note": (
                "Counts memory operation outputs returned by memory.add or paper_update_wrapper, not necessarily the final "
                "persisted memory store if UPDATE/DELETE/NOOP operations occurred. It is also separate from retrieved prompt tokens."
            ),
        }
        final_memory_store_tokens_by_sample[sample_id] = {}
        final_num_memories_by_sample[sample_id] = {}
        final_total_tokens = 0
        final_total_memories = 0
        active_index_memories_by_speaker = {}
        if args.memory_update_mode == "paper_update_wrapper":
            for speaker in (speaker_a, speaker_b):
                final_memories = wrapper_store.get_all(user_id=speaker_user_ids[speaker])
                active_index_memories_by_speaker[speaker] = final_memories
                final_tokens = count_memory_text_tokens(token_encoding, final_memories)
                final_count = len(final_memories)
                final_total_tokens += final_tokens
                final_total_memories += final_count

                final_memory_store_tokens_by_sample[sample_id][speaker] = final_tokens
                final_num_memories_by_sample[sample_id][speaker] = final_count

            final_memory_store_tokens_by_sample[sample_id]["total"] = final_total_tokens
            final_memory_store_tokens_by_sample[sample_id]["token_encoding"] = args.retrieved_memory_token_encoding
            final_memory_store_tokens_by_sample[sample_id]["note"] = (
                "Counts final persisted memories in LocalPaperMemoryStore after sample ingestion."
            )
            final_num_memories_by_sample[sample_id]["total"] = final_total_memories
        elif local_sdk_capabilities["local_sdk_final_store_fetch_supported"]:
            for speaker in (speaker_a, speaker_b):
                final_memories, fetch_error = fetch_final_memories(
                    memory,
                    user_id=speaker_user_ids[speaker],
                    top_k=args.final_store_top_k,
                )
                if fetch_error:
                    final_store_fetch_warnings.append(
                        {
                            "sample_id": sample_id,
                            "speaker": speaker,
                            "user_id": speaker_user_ids[speaker],
                            "error": fetch_error,
                        }
                    )
                    final_tokens = None
                    final_count = None
                else:
                    active_index_memories_by_speaker[speaker] = final_memories
                    final_tokens = count_memory_text_tokens(token_encoding, final_memories)
                    final_count = len(final_memories)
                    final_total_tokens += final_tokens
                    final_total_memories += final_count

                final_memory_store_tokens_by_sample[sample_id][speaker] = final_tokens
                final_num_memories_by_sample[sample_id][speaker] = final_count

            final_memory_store_tokens_by_sample[sample_id]["total"] = final_total_tokens
            final_memory_store_tokens_by_sample[sample_id]["token_encoding"] = args.retrieved_memory_token_encoding
            final_memory_store_tokens_by_sample[sample_id]["note"] = (
                "Counts final persisted memories fetched with Memory.get_all after sample ingestion."
            )
            final_num_memories_by_sample[sample_id]["total"] = final_total_memories
        else:
            final_store_fetch_warnings.append(
                {
                    "sample_id": sample_id,
                    "error": "Memory.get_all is not supported by this local SDK.",
                }
            )

        active_memory_index = None
        if args.qa_retrieval_mode in {"active_reconstruction", "hybrid_adaptive", "evidence_gap_hybrid"}:
            missing_index_speakers = [
                speaker
                for speaker in (speaker_a, speaker_b)
                if speaker not in active_index_memories_by_speaker
            ]
            if missing_index_speakers:
                raise ValueError(
                    "--qa_retrieval_mode active_reconstruction/hybrid_adaptive/evidence_gap_hybrid "
                    "requires final memories for every speaker. "
                    f"Missing: {missing_index_speakers}. Use --memory_update_mode paper_update_wrapper "
                    "or a local SDK build with Memory.get_all support."
                )
            active_memory_index = Mem0ActiveMemoryIndex(
                active_index_memories_by_speaker,
                speaker_user_ids,
            )
            print(
                "Active reconstruction index:",
                sum(len(items) for items in active_index_memories_by_speaker.values()),
                "memories",
            )

        qa_counter_for_sample = 0

        for qa_idx, qa in enumerate(qa_list, start=1):
            category = str(qa.get("category", "unknown"))
            if args.skip_category_5 and category == "5":
                continue

            qa_counter_for_sample += 1
            if args.max_qa is not None and qa_counter_for_sample > args.max_qa:
                break

            qa_key = f"{sample_id}:qa:{qa_idx}"
            if qa_key in completed_qa_keys:
                print(f"QA {qa_counter_for_sample}: skipping completed qa_index={qa_idx}")
                continue

            total_qa_seen += 1

            question = qa.get("question", "")
            ground_truth = str(qa.get("answer", ""))
            evidence = qa.get("evidence", []) or []
            evidence_texts = get_evidence_texts(evidence, dia_lookup)

            print(f"QA {qa_counter_for_sample}: {question}")

            speaker_1_user_id = speaker_user_ids[speaker_a]
            speaker_2_user_id = speaker_user_ids[speaker_b]
            qase_plan = None
            qase_selection_log = None
            qase_query_log = None
            qase_search_top_k = {speaker_a: args.top_k, speaker_b: args.top_k}
            qase_planner_latency = 0.0
            qase_selector_latency = 0.0
            qase_bm25_latency = 0.0
            semantic_search_latency = 0.0
            if args.qa_retrieval_mode in QUESTION_AWARE_RETRIEVAL_MODES:
                qase_planner_start = time.time()
                qase_plan = qase_controller.plan(question, (speaker_a, speaker_b))
                qase_planner_latency = time.time() - qase_planner_start
                if args.qa_retrieval_mode in {
                    "question_aware_selective",
                    "question_aware_semantic_adaptive",
                    "question_aware_hybrid_adaptive",
                }:
                    qase_search_top_k = {
                        speaker_a: int(qase_plan.candidate_top_k_by_speaker.get(speaker_a) or 0),
                        speaker_b: int(qase_plan.candidate_top_k_by_speaker.get(speaker_b) or 0),
                    }
                else:
                    qase_search_top_k = {
                        speaker_a: int(qase_plan.final_top_k_cap_by_speaker.get(speaker_a) or 0),
                        speaker_b: int(qase_plan.final_top_k_cap_by_speaker.get(speaker_b) or 0),
                    }

            if args.memory_update_mode == "paper_update_wrapper":
                speaker_1_search_call = lambda query, top_k: search_wrapper_memories(
                        wrapper_store,
                        question=query,
                        user_id=speaker_1_user_id,
                        top_k=top_k,
                        threshold=args.search_threshold,
                    )
                speaker_2_search_call = lambda query, top_k: search_wrapper_memories(
                        wrapper_store,
                        question=query,
                        user_id=speaker_2_user_id,
                        top_k=top_k,
                        threshold=args.search_threshold,
                    )
            else:
                speaker_1_search_call = lambda query, top_k: search_memories(
                        memory,
                        question=query,
                        user_id=speaker_1_user_id,
                        top_k=top_k,
                        threshold=args.search_threshold,
                    )
                speaker_2_search_call = lambda query, top_k: search_memories(
                        memory,
                        question=query,
                        user_id=speaker_2_user_id,
                        top_k=top_k,
                        threshold=args.search_threshold,
                    )
            speaker_1_memories, speaker_1_memory_time, speaker_1_query_log = search_qase_candidate_memories(
                search_call=speaker_1_search_call,
                question=question,
                speaker=speaker_a,
                top_k=qase_search_top_k[speaker_a],
            )
            speaker_2_memories, speaker_2_memory_time, speaker_2_query_log = search_qase_candidate_memories(
                search_call=speaker_2_search_call,
                question=question,
                speaker=speaker_b,
                top_k=qase_search_top_k[speaker_b],
            )
            if args.qa_retrieval_mode in QUESTION_AWARE_RETRIEVAL_MODES:
                qase_query_log = {
                    speaker_a: speaker_1_query_log,
                    speaker_b: speaker_2_query_log,
                }
            semantic_seed_speaker_1_memories = speaker_1_memories
            semantic_seed_speaker_2_memories = speaker_2_memories
            semantic_seed_speaker_1_memory_time = speaker_1_memory_time
            semantic_seed_speaker_2_memory_time = speaker_2_memory_time
            semantic_search_latency = speaker_1_memory_time + speaker_2_memory_time
            if args.qase_bm25_candidate_rerank and args.qa_retrieval_mode == "question_aware_selective":
                if wrapper_store is None:
                    raise RuntimeError("BM25 candidate reranking requires paper_update_wrapper local store.")
                (
                    semantic_seed_speaker_1_memories,
                    bm25_speaker_1_time,
                    bm25_speaker_1_log,
                ) = score_bm25_semantic_candidates(
                    wrapper_store,
                    question=question,
                    user_id=speaker_1_user_id,
                    speaker=speaker_a,
                    candidates=semantic_seed_speaker_1_memories,
                    k1=args.qase_bm25_k1,
                    b=args.qase_bm25_b,
                )
                (
                    semantic_seed_speaker_2_memories,
                    bm25_speaker_2_time,
                    bm25_speaker_2_log,
                ) = score_bm25_semantic_candidates(
                    wrapper_store,
                    question=question,
                    user_id=speaker_2_user_id,
                    speaker=speaker_b,
                    candidates=semantic_seed_speaker_2_memories,
                    k1=args.qase_bm25_k1,
                    b=args.qase_bm25_b,
                )
                semantic_seed_speaker_1_memory_time += bm25_speaker_1_time
                semantic_seed_speaker_2_memory_time += bm25_speaker_2_time
                speaker_1_memory_time += bm25_speaker_1_time
                speaker_2_memory_time += bm25_speaker_2_time
                qase_bm25_latency += bm25_speaker_1_time + bm25_speaker_2_time
                qase_query_log[speaker_a]["bm25_candidate_rerank"] = bm25_speaker_1_log
                qase_query_log[speaker_b]["bm25_candidate_rerank"] = bm25_speaker_2_log
                qase_query_log[speaker_a]["result_count_after_bm25_rerank"] = len(semantic_seed_speaker_1_memories)
                qase_query_log[speaker_b]["result_count_after_bm25_rerank"] = len(semantic_seed_speaker_2_memories)
            active_reconstruction_info = None
            active_reconstruction_latency = 0.0
            max_total_retrieved_memories = args.top_k * 2
            if args.qa_retrieval_mode == "question_aware_selective":
                if qase_plan is None:
                    raise RuntimeError("QASE plan was not initialized.")
                qase_candidate_pool = {
                    speaker_a: semantic_seed_speaker_1_memories,
                    speaker_b: semantic_seed_speaker_2_memories,
                }
                if args.qase_selection_scope == "global":
                    selected_by_speaker, qase_selection_log = qase_controller.select_global(
                        qase_candidate_pool,
                        qase_plan,
                    )
                else:
                    selected_by_speaker, qase_selection_log = qase_controller.select(
                        qase_candidate_pool,
                        qase_plan,
                    )
                speaker_1_memories = selected_by_speaker.get(speaker_a, [])
                speaker_2_memories = selected_by_speaker.get(speaker_b, [])
                qase_selector_latency = float(qase_selection_log.latency_sec or 0.0)
                active_reconstruction_latency = qase_selector_latency
                active_reconstruction_latency_totals["active_reconstruction_latency_sec"] += (
                    active_reconstruction_latency
                )
                active_reconstruction_latency_totals["active_reconstruction_num_questions"] += 1
                max_total_retrieved_memories = qase_plan.max_total_final_memories
                active_reconstruction_info = {
                    **build_qase_trace(qase_plan, qase_selection_log),
                    "selection_scope": args.qase_selection_scope,
                    "latency_sec": active_reconstruction_latency,
                    "planner_latency_sec": qase_planner_latency,
                    "selector_latency_sec": qase_selector_latency,
                    "bm25_latency_sec": qase_bm25_latency,
                    "semantic_search_latency_sec": semantic_search_latency,
                    "query_log": qase_query_log,
                    "semantic_seed_num_speaker_1_memories": len(semantic_seed_speaker_1_memories),
                    "semantic_seed_num_speaker_2_memories": len(semantic_seed_speaker_2_memories),
                    "semantic_seed_speaker_1_memory_time": semantic_seed_speaker_1_memory_time,
                    "semantic_seed_speaker_2_memory_time": semantic_seed_speaker_2_memory_time,
                    "final_num_speaker_1_memories": len(speaker_1_memories),
                    "final_num_speaker_2_memories": len(speaker_2_memories),
                }
            elif args.qa_retrieval_mode == "question_aware_semantic":
                if qase_plan is None:
                    raise RuntimeError("Question-aware semantic plan was not initialized.")
                max_total_retrieved_memories = qase_plan.max_total_final_memories
                active_reconstruction_info = {
                    "mode": "question_aware_semantic",
                    "plan": qase_plan.to_dict(),
                    "selection_log": None,
                    "latency_sec": 0.0,
                    "query_log": qase_query_log,
                    "semantic_only": True,
                    "semantic_seed_num_speaker_1_memories": len(semantic_seed_speaker_1_memories),
                    "semantic_seed_num_speaker_2_memories": len(semantic_seed_speaker_2_memories),
                    "semantic_seed_speaker_1_memory_time": semantic_seed_speaker_1_memory_time,
                    "semantic_seed_speaker_2_memory_time": semantic_seed_speaker_2_memory_time,
                    "final_num_speaker_1_memories": len(speaker_1_memories),
                    "final_num_speaker_2_memories": len(speaker_2_memories),
                    "note": "Question-aware semantic control: uses QASE planner budget, then keeps raw semantic top-k.",
                }
            elif args.qa_retrieval_mode in {"question_aware_semantic_adaptive", "question_aware_hybrid_adaptive"}:
                if qase_plan is None:
                    raise RuntimeError("Question-aware semantic adaptive plan was not initialized.")
                qase_selector_start = time.time()
                question_type_value = qase_plan.question_type.value
                adaptive_seed_speaker_1_memories = semantic_seed_speaker_1_memories
                adaptive_seed_speaker_2_memories = semantic_seed_speaker_2_memories
                hybrid_fusion_log = None
                hybrid_fusion_latency = 0.0
                if args.qa_retrieval_mode == "question_aware_hybrid_adaptive":
                    adaptive_seed_speaker_1_memories, speaker_1_hybrid_log = hybrid_fuse_candidate_memories(
                        semantic_seed_speaker_1_memories,
                        question=question,
                        speaker=speaker_a,
                        speaker_names=(speaker_a, speaker_b),
                        needs_time=bool(getattr(qase_plan, "needs_time", False)),
                        args=args,
                    )
                    adaptive_seed_speaker_2_memories, speaker_2_hybrid_log = hybrid_fuse_candidate_memories(
                        semantic_seed_speaker_2_memories,
                        question=question,
                        speaker=speaker_b,
                        speaker_names=(speaker_a, speaker_b),
                        needs_time=bool(getattr(qase_plan, "needs_time", False)),
                        args=args,
                    )
                    hybrid_fusion_log = {
                        speaker_a: speaker_1_hybrid_log,
                        speaker_b: speaker_2_hybrid_log,
                    }
                    hybrid_fusion_latency = float(speaker_1_hybrid_log.get("latency_sec") or 0.0) + float(
                        speaker_2_hybrid_log.get("latency_sec") or 0.0
                    )
                    active_reconstruction_latency += hybrid_fusion_latency
                original_caps_by_speaker = {
                    speaker_a: int(qase_plan.final_top_k_cap_by_speaker.get(speaker_a) or 0),
                    speaker_b: int(qase_plan.final_top_k_cap_by_speaker.get(speaker_b) or 0),
                }
                if args.qase_adaptive_budget_profile == "adaptive_k_only":
                    adaptive_caps_by_speaker = {
                        speaker_a: len(adaptive_seed_speaker_1_memories),
                        speaker_b: len(adaptive_seed_speaker_2_memories),
                    }
                    adaptive_budget_profile = {
                        "question_type": question_type_value,
                        "total_floor": 0,
                        "total_cap": sum(adaptive_caps_by_speaker.values()),
                        "requested_floor": 0,
                        "requested_cap": None,
                        "runner_max_total_final_memories": qase_plan.max_total_final_memories,
                        "fixed_semantic": False,
                        "profile_name": "adaptive_v7_adaptive_k_only",
                        "candidate_pool_only": True,
                        "note": (
                            "No type-specific final floor/cap is applied; Adaptive-k chooses final k from "
                            "the retrieved candidate pool."
                        ),
                    }
                else:
                    adaptive_budget_profile = _adaptive_total_budget_profile(
                        question_type_value,
                        max_total_final_memories=qase_plan.max_total_final_memories,
                    )
                    adaptive_caps_by_speaker = _cap_speaker_budgets_to_total(
                        original_caps_by_speaker,
                        total_cap=int(adaptive_budget_profile["total_cap"]),
                        preferred_speakers=getattr(qase_plan, "target_speakers", ()) or (),
                    )
                speaker_1_cap = int(adaptive_caps_by_speaker.get(speaker_a) or 0)
                speaker_2_cap = int(adaptive_caps_by_speaker.get(speaker_b) or 0)

                if adaptive_budget_profile.get("fixed_semantic"):
                    speaker_1_memories = adaptive_seed_speaker_1_memories[:speaker_1_cap]
                    speaker_2_memories = adaptive_seed_speaker_2_memories[:speaker_2_cap]
                    speaker_1_adaptive_log = {
                        "input_count": len(adaptive_seed_speaker_1_memories),
                        "selected_count": len(speaker_1_memories),
                        "cap": speaker_1_cap,
                        "cutoff_k": len(speaker_1_memories),
                        "reason": "fixed top-k for protected question type",
                        "adaptive_cutoff_disabled": True,
                    }
                    speaker_2_adaptive_log = {
                        "input_count": len(adaptive_seed_speaker_2_memories),
                        "selected_count": len(speaker_2_memories),
                        "cap": speaker_2_cap,
                        "cutoff_k": len(speaker_2_memories),
                        "reason": "fixed top-k for protected question type",
                        "adaptive_cutoff_disabled": True,
                    }
                    adaptive_rebalance_log = {
                        "enabled": False,
                        "initial_total_selected": len(speaker_1_memories) + len(speaker_2_memories),
                        "final_total_selected": len(speaker_1_memories) + len(speaker_2_memories),
                        "global_cap": adaptive_budget_profile["total_cap"],
                        "global_floor": adaptive_budget_profile["total_floor"],
                        "reason": "fixed semantic profile does not rebalance",
                    }
                else:
                    speaker_1_memories, speaker_1_adaptive_log = adaptive_semantic_cutoff(
                        adaptive_seed_speaker_1_memories,
                        cap=speaker_1_cap,
                        question_type=question_type_value,
                        config=qase_controller.config,
                    )
                    speaker_2_memories, speaker_2_adaptive_log = adaptive_semantic_cutoff(
                        adaptive_seed_speaker_2_memories,
                        cap=speaker_2_cap,
                        question_type=question_type_value,
                        config=qase_controller.config,
                    )
                    rebalanced_by_speaker, adaptive_rebalance_log = rebalance_adaptive_semantic_memories(
                        candidates_by_speaker={
                            speaker_a: adaptive_seed_speaker_1_memories,
                            speaker_b: adaptive_seed_speaker_2_memories,
                        },
                        selected_by_speaker={
                            speaker_a: speaker_1_memories,
                            speaker_b: speaker_2_memories,
                        },
                        caps_by_speaker={
                            speaker_a: speaker_1_cap,
                            speaker_b: speaker_2_cap,
                        },
                        question_type=question_type_value,
                        config=qase_controller.config,
                        max_total_final_memories=int(adaptive_budget_profile["total_cap"]),
                        total_floor=int(adaptive_budget_profile["total_floor"]),
                    )
                    speaker_1_memories = rebalanced_by_speaker.get(speaker_a, speaker_1_memories)
                    speaker_2_memories = rebalanced_by_speaker.get(speaker_b, speaker_2_memories)
                qase_selector_latency = time.time() - qase_selector_start
                active_reconstruction_latency = qase_selector_latency
                max_total_retrieved_memories = int(adaptive_budget_profile["total_cap"])
                qase_selection_log = {
                    "question_type": question_type_value,
                    "total_input_candidates": len(adaptive_seed_speaker_1_memories)
                    + len(adaptive_seed_speaker_2_memories),
                    "total_selected": len(speaker_1_memories) + len(speaker_2_memories),
                    "per_speaker": {
                        speaker_a: speaker_1_adaptive_log,
                        speaker_b: speaker_2_adaptive_log,
                    },
                    "hybrid_fusion": hybrid_fusion_log,
                    "rebalance": adaptive_rebalance_log,
                    "max_total_final_memories": adaptive_budget_profile["total_cap"],
                    "original_final_top_k_cap_by_speaker": original_caps_by_speaker,
                    "adaptive_final_top_k_cap_by_speaker": adaptive_caps_by_speaker,
                    "adaptive_budget_profile": adaptive_budget_profile,
                    "mode": (
                        "hybrid_fusion_adaptive_cutoff"
                        if args.qa_retrieval_mode == "question_aware_hybrid_adaptive"
                        else "semantic_adaptive_cutoff"
                    ),
                }
                active_reconstruction_info = {
                    "mode": args.qa_retrieval_mode,
                    "plan": qase_plan.to_dict(),
                    "selection_log": qase_selection_log,
                    "latency_sec": qase_selector_latency,
                    "query_log": qase_query_log,
                    "semantic_only": args.qa_retrieval_mode == "question_aware_semantic_adaptive",
                    "hybrid_fusion_enabled": args.qa_retrieval_mode == "question_aware_hybrid_adaptive",
                    "hybrid_fusion_profile": (
                        HYBRID_FUSION_PROFILE_NAME
                        if args.qa_retrieval_mode == "question_aware_hybrid_adaptive"
                        else None
                    ),
                    "hybrid_fusion_latency_sec": hybrid_fusion_latency,
                    "planner_latency_sec": qase_planner_latency,
                    "selector_latency_sec": qase_selector_latency,
                    "bm25_latency_sec": qase_bm25_latency,
                    "semantic_search_latency_sec": semantic_search_latency,
                    "adaptive_cutoff": True,
                    "adaptive_semantic_cutoff": True,
                    "semantic_seed_num_speaker_1_memories": len(semantic_seed_speaker_1_memories),
                    "semantic_seed_num_speaker_2_memories": len(semantic_seed_speaker_2_memories),
                    "adaptive_seed_num_speaker_1_memories": len(adaptive_seed_speaker_1_memories),
                    "adaptive_seed_num_speaker_2_memories": len(adaptive_seed_speaker_2_memories),
                    "semantic_seed_speaker_1_memory_time": semantic_seed_speaker_1_memory_time,
                    "semantic_seed_speaker_2_memory_time": semantic_seed_speaker_2_memory_time,
                    "final_num_speaker_1_memories": len(speaker_1_memories),
                    "final_num_speaker_2_memories": len(speaker_2_memories),
                    "note": (
                        "Question-aware adaptive retrieval: retrieves the planner candidate pool, optionally "
                        "rescoring it with hybrid semantic+BM25+entity+time fusion, then applies adaptive-k "
                        "thresholding. Budget profile: "
                        f"{args.qase_adaptive_budget_profile}."
                    ),
                }
            elif args.qa_retrieval_mode == "active_reconstruction":
                if active_memory_index is None:
                    raise RuntimeError("Active reconstruction index was not initialized.")
                active_result = active_reconstruct_memories(
                    index=active_memory_index,
                    question=question,
                    speaker_names=(speaker_a, speaker_b),
                    top_k_per_speaker=args.top_k,
                    initial_memories_by_speaker={
                        speaker_a: semantic_seed_speaker_1_memories,
                        speaker_b: semantic_seed_speaker_2_memories,
                    },
                    max_steps=args.active_reconstruction_max_steps,
                    max_tool_calls_per_step=args.active_reconstruction_max_tool_calls_per_step,
                    seed_top_k_per_speaker=args.active_reconstruction_seed_top_k,
                    use_llm_router=args.active_reconstruction_use_llm_router,
                    chat_client=chat_client,
                    llm_model=args.llm_model,
                    router_max_tokens=args.active_reconstruction_router_max_tokens,
                )
                speaker_1_memories = active_result["memories_by_speaker"].get(speaker_a, [])
                speaker_2_memories = active_result["memories_by_speaker"].get(speaker_b, [])
                active_reconstruction_latency = float(active_result.get("latency_sec") or 0.0)
                active_reconstruction_latency_totals["active_reconstruction_latency_sec"] += (
                    active_reconstruction_latency
                )
                active_reconstruction_latency_totals["active_reconstruction_num_questions"] += 1
                add_active_usage_totals(
                    active_reconstruction_usage_totals,
                    active_result.get("usage_totals") or {},
                )
                active_reconstruction_info = {
                    "router": active_result.get("router"),
                    "trace": active_result.get("trace"),
                    "usage_totals": active_result.get("usage_totals"),
                    "latency_sec": active_reconstruction_latency,
                    "seed_top_k_per_speaker": active_result.get("seed_top_k_per_speaker"),
                    "max_steps": active_result.get("max_steps"),
                    "max_tool_calls_per_step": active_result.get("max_tool_calls_per_step"),
                    "use_llm_router": active_result.get("use_llm_router"),
                    "num_evidence_memories": active_result.get("num_evidence_memories"),
                    "num_final_memories": active_result.get("num_final_memories"),
                    "semantic_seed_num_speaker_1_memories": len(semantic_seed_speaker_1_memories),
                    "semantic_seed_num_speaker_2_memories": len(semantic_seed_speaker_2_memories),
                    "semantic_seed_speaker_1_memory_time": semantic_seed_speaker_1_memory_time,
                    "semantic_seed_speaker_2_memory_time": semantic_seed_speaker_2_memory_time,
                }
            elif args.qa_retrieval_mode == "hybrid_adaptive":
                if active_memory_index is None:
                    raise RuntimeError("Hybrid adaptive retrieval index was not initialized.")
                active_result = hybrid_adaptive_retrieve_memories(
                    index=active_memory_index,
                    question=question,
                    speaker_names=(speaker_a, speaker_b),
                    top_k_per_speaker=args.top_k,
                    initial_memories_by_speaker={
                        speaker_a: semantic_seed_speaker_1_memories,
                        speaker_b: semantic_seed_speaker_2_memories,
                    },
                    protected_semantic_top_k=args.protected_semantic_top_k,
                    active_extra_k_per_speaker=args.active_extra_k_per_speaker,
                    active_seed_top_k_per_speaker=args.har_active_seed_top_k,
                    router_mode=args.hybrid_router_mode,
                    max_steps=args.active_reconstruction_max_steps,
                    max_tool_calls_per_step=args.active_reconstruction_max_tool_calls_per_step,
                    use_llm_router=args.active_reconstruction_use_llm_router,
                    chat_client=chat_client,
                    llm_model=args.llm_model,
                    router_max_tokens=args.active_reconstruction_router_max_tokens,
                    evidence_rerank_mode=args.har_evidence_rerank_mode,
                    evidence_rerank_trigger=args.har_evidence_rerank_trigger,
                    evidence_rerank_max_candidates=args.har_evidence_rerank_max_candidates,
                    evidence_rerank_max_tokens=args.har_evidence_rerank_max_tokens,
                    sufficiency_mode=args.har_sufficiency_mode,
                    sufficiency_max_rounds=args.har_sufficiency_max_rounds,
                    sufficiency_max_candidates=args.har_sufficiency_max_candidates,
                    sufficiency_max_tokens=args.har_sufficiency_max_tokens,
                )
                speaker_1_memories = active_result["memories_by_speaker"].get(speaker_a, [])
                speaker_2_memories = active_result["memories_by_speaker"].get(speaker_b, [])
                active_reconstruction_latency = float(active_result.get("latency_sec") or 0.0)
                active_reconstruction_latency_totals["active_reconstruction_latency_sec"] += (
                    active_reconstruction_latency
                )
                active_reconstruction_latency_totals["active_reconstruction_num_questions"] += 1
                add_active_usage_totals(
                    active_reconstruction_usage_totals,
                    active_result.get("usage_totals") or {},
                )
                max_total_retrieved_memories = (
                    min(args.top_k, args.protected_semantic_top_k) + args.active_extra_k_per_speaker
                ) * 2
                active_reconstruction_info = {
                    "mode": "hybrid_adaptive",
                    "router": active_result.get("router"),
                    "trace": active_result.get("trace"),
                    "usage_totals": active_result.get("usage_totals"),
                    "latency_sec": active_reconstruction_latency,
                    "used_adaptive_retrieval": active_result.get("used_adaptive_retrieval"),
                    "protected_semantic_top_k": active_result.get("protected_semantic_top_k"),
                    "active_extra_k_per_speaker": active_result.get("active_extra_k_per_speaker"),
                    "active_seed_top_k_per_speaker": active_result.get("active_seed_top_k_per_speaker"),
                    "max_steps": active_result.get("max_steps"),
                    "max_tool_calls_per_step": active_result.get("max_tool_calls_per_step"),
                    "use_llm_router": active_result.get("use_llm_router"),
                    "evidence_rerank": active_result.get("evidence_rerank"),
                    "merge_log": active_result.get("merge_log"),
                    "sufficiency_check": active_result.get("sufficiency_check"),
                    "sufficiency_rounds": active_result.get("sufficiency_rounds"),
                    "sufficiency_guided_runs": active_result.get("sufficiency_guided_runs"),
                    "sufficiency_mode": active_result.get("sufficiency_mode"),
                    "sufficiency_max_rounds": active_result.get("sufficiency_max_rounds"),
                    "active_result": active_result.get("active_result"),
                    "num_final_memories": active_result.get("num_final_memories"),
                    "semantic_seed_num_speaker_1_memories": len(semantic_seed_speaker_1_memories),
                    "semantic_seed_num_speaker_2_memories": len(semantic_seed_speaker_2_memories),
                    "semantic_seed_speaker_1_memory_time": semantic_seed_speaker_1_memory_time,
                    "semantic_seed_speaker_2_memory_time": semantic_seed_speaker_2_memory_time,
                }
            elif args.qa_retrieval_mode == "evidence_gap_hybrid":
                if active_memory_index is None:
                    raise RuntimeError("Evidence-gap retrieval index was not initialized.")
                active_result = evidence_gap_retrieve_memories(
                    index=active_memory_index,
                    question=question,
                    speaker_names=(speaker_a, speaker_b),
                    top_k_per_speaker=args.top_k,
                    initial_memories_by_speaker={
                        speaker_a: semantic_seed_speaker_1_memories,
                        speaker_b: semantic_seed_speaker_2_memories,
                    },
                    protected_semantic_top_k=args.protected_semantic_top_k,
                    max_gap_rounds=args.egc_max_gap_rounds,
                    max_slots=args.egc_max_slots,
                    min_slot_coverage=args.egc_min_slot_coverage,
                    gap_top_k_per_slot=args.egc_gap_top_k_per_slot,
                    max_gap_evidence_per_speaker=args.egc_max_gap_evidence_per_speaker,
                    contract_mode=args.egc_contract_mode,
                    coverage_mode=args.egc_coverage_mode,
                    proof_pack_rerank_mode=args.egc_proof_pack_rerank,
                    keep_protected_baseline=args.egc_keep_protected_baseline,
                    chat_client=chat_client,
                    llm_model=args.llm_model,
                    contract_max_tokens=args.egc_contract_max_tokens,
                    coverage_max_candidates=args.egc_coverage_max_candidates,
                    coverage_max_tokens=args.egc_coverage_max_tokens,
                    proof_pack_max_candidates=args.egc_proof_pack_max_candidates,
                    proof_pack_max_tokens=args.egc_proof_pack_max_tokens,
                    max_tool_calls_per_round=args.egc_max_tool_calls_per_round,
                )
                speaker_1_memories = active_result["memories_by_speaker"].get(speaker_a, [])
                speaker_2_memories = active_result["memories_by_speaker"].get(speaker_b, [])
                active_reconstruction_latency = float(active_result.get("latency_sec") or 0.0)
                active_reconstruction_latency_totals["active_reconstruction_latency_sec"] += (
                    active_reconstruction_latency
                )
                active_reconstruction_latency_totals["active_reconstruction_num_questions"] += 1
                add_active_usage_totals(
                    active_reconstruction_usage_totals,
                    active_result.get("usage_totals") or {},
                )
                max_total_retrieved_memories = (
                    min(args.top_k, args.protected_semantic_top_k) + args.egc_max_gap_evidence_per_speaker
                ) * 2
                active_reconstruction_info = {
                    "mode": "evidence_gap_hybrid",
                    "evidence_contract": active_result.get("evidence_contract"),
                    "initial_slot_coverage": active_result.get("initial_slot_coverage"),
                    "slot_coverage": active_result.get("slot_coverage"),
                    "gap_rounds": active_result.get("gap_rounds"),
                    "proof_pack_rerank": active_result.get("proof_pack_rerank"),
                    "merge_log": active_result.get("merge_log"),
                    "usage_totals": active_result.get("usage_totals"),
                    "latency_sec": active_reconstruction_latency,
                    "used_gap_retrieval": active_result.get("used_gap_retrieval"),
                    "gap_round_count": active_result.get("gap_round_count"),
                    "protected_semantic_top_k": active_result.get("protected_semantic_top_k"),
                    "max_gap_rounds": active_result.get("max_gap_rounds"),
                    "max_slots": active_result.get("max_slots"),
                    "min_slot_coverage": active_result.get("min_slot_coverage"),
                    "gap_top_k_per_slot": active_result.get("gap_top_k_per_slot"),
                    "max_gap_evidence_per_speaker": active_result.get("max_gap_evidence_per_speaker"),
                    "num_final_memories": active_result.get("num_final_memories"),
                    "semantic_seed_num_speaker_1_memories": len(semantic_seed_speaker_1_memories),
                    "semantic_seed_num_speaker_2_memories": len(semantic_seed_speaker_2_memories),
                    "semantic_seed_speaker_1_memory_time": semantic_seed_speaker_1_memory_time,
                    "semantic_seed_speaker_2_memory_time": semantic_seed_speaker_2_memory_time,
                }
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
            qase_token_log = None
            if qase_plan is not None:
                qase_candidate_speaker_1_tokens = count_memory_tokens(
                    token_encoding,
                    semantic_seed_speaker_1_memories,
                )
                qase_candidate_speaker_2_tokens = count_memory_tokens(
                    token_encoding,
                    semantic_seed_speaker_2_memories,
                )
                qase_token_log = {
                    "candidate_tokens_by_speaker": {
                        speaker_a: qase_candidate_speaker_1_tokens,
                        speaker_b: qase_candidate_speaker_2_tokens,
                    },
                    "candidate_tokens_total": qase_candidate_speaker_1_tokens
                    + qase_candidate_speaker_2_tokens,
                    "selected_tokens_by_speaker": {
                        speaker_a: speaker_1_memory_tokens,
                        speaker_b: speaker_2_memory_tokens,
                    },
                    "selected_tokens_total": retrieved_memory_tokens,
                    "candidate_count_by_speaker": {
                        speaker_a: len(semantic_seed_speaker_1_memories),
                        speaker_b: len(semantic_seed_speaker_2_memories),
                    },
                    "selected_count_by_speaker": {
                        speaker_a: len(speaker_1_memories),
                        speaker_b: len(speaker_2_memories),
                    },
                }
                if active_reconstruction_info is not None:
                    active_reconstruction_info["token_log"] = qase_token_log

            gen_start = time.time()
            predicted_answer, usage, answer_prompt = answer_question(
                chat_client,
                llm_model=args.llm_model,
                question=question,
                speaker_1_name=speaker_a,
                speaker_2_name=speaker_b,
                speaker_1_memories=speaker_1_memories,
                speaker_2_memories=speaker_2_memories,
                answer_max_tokens=args.answer_max_tokens,
            )
            generation_latency = time.time() - gen_start

            total_qa_answered += 1

            retrieved_memories = [
                *annotate_memories(speaker_1_memories, speaker_1_user_id, speaker_a),
                *annotate_memories(speaker_2_memories, speaker_2_user_id, speaker_b),
            ]
            if args.qa_retrieval_mode in QUESTION_AWARE_RETRIEVAL_MODES:
                search_latency = (
                    semantic_search_latency
                    + qase_bm25_latency
                    + qase_planner_latency
                    + qase_selector_latency
                )
            else:
                search_latency = speaker_1_memory_time + speaker_2_memory_time + active_reconstruction_latency

            results.append(
                {
                    "sample_id": sample_id,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "speaker_1_user_id": speaker_1_user_id,
                    "speaker_2_user_id": speaker_2_user_id,
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
                    "memory_update_mode": args.memory_update_mode,
                    "memory_wrapper_store": args.memory_wrapper_store,
                    "qa_retrieval_mode": args.qa_retrieval_mode,
                    "active_reconstruction": active_reconstruction_info,
                    "qase_question_type": qase_plan.question_type.value if qase_plan else None,
                    "qase_target_speakers": qase_plan.target_speakers if qase_plan else None,
                    "qase_candidate_top_k_by_speaker": (
                        qase_plan.candidate_top_k_by_speaker if qase_plan else None
                    ),
                    "qase_final_top_k_cap_by_speaker": (
                        qase_plan.final_top_k_cap_by_speaker if qase_plan else None
                    ),
                    "qase_search_top_k_by_speaker": qase_search_top_k if qase_plan else None,
                    "qase_query_log": qase_query_log,
                    "qase_selection_log": serialize_qase_selection_log(qase_selection_log),
                    "qase_token_log": qase_token_log,
                    "num_speaker_1_memories": len(speaker_1_memories),
                    "num_speaker_2_memories": len(speaker_2_memories),
                    "top_k_per_speaker": args.top_k,
                    "max_total_retrieved_memories": max_total_retrieved_memories,
                    "retrieval_token_budget": args.retrieval_token_budget,
                    **retrieval_budget_info,
                    "speaker_1_memory_tokens": speaker_1_memory_tokens,
                    "speaker_2_memory_tokens": speaker_2_memory_tokens,
                    "retrieved_memory_tokens": retrieved_memory_tokens,
                    "retrieved_memory_token_encoding": args.retrieved_memory_token_encoding,
                    "speaker_1_memory_time": speaker_1_memory_time,
                    "speaker_2_memory_time": speaker_2_memory_time,
                    "semantic_seed_speaker_1_memory_time": semantic_seed_speaker_1_memory_time,
                    "semantic_seed_speaker_2_memory_time": semantic_seed_speaker_2_memory_time,
                    "semantic_search_latency_sec": semantic_search_latency,
                    "qase_bm25_latency_sec": qase_bm25_latency,
                    "qase_planner_latency_sec": qase_planner_latency,
                    "qase_selector_latency_sec": qase_selector_latency,
                    "active_reconstruction_latency_sec": active_reconstruction_latency,
                    "search_latency_sec": search_latency,
                    "generation_latency_sec": generation_latency,
                    "total_latency_sec": search_latency + generation_latency,
                    "usage": usage,
                    "answer_prompt": answer_prompt,
                }
            )

            print(f"  pred={predicted_answer}")
            print(
                f"  search={search_latency:.2f}s "
                f"({speaker_a}={speaker_1_memory_time:.2f}s, {speaker_b}={speaker_2_memory_time:.2f}s, "
                f"planner={qase_planner_latency:.2f}s, selector={qase_selector_latency:.2f}s, "
                f"active={active_reconstruction_latency:.2f}s) | "
                f"gen={generation_latency:.2f}s | answer_tokens={usage.get('total_tokens')} | "
                f"memory_tokens={retrieved_memory_tokens}"
            )
            completed_qa_keys.add(qa_key)
            update_partial_sample_state("qa", ingestion_complete=True)
            save_progress_checkpoint(f"{sample_id}/qa_{qa_counter_for_sample}")

        completed_sample_ids.add(sample_id)
        partial_sample_state = None
        write_run_checkpoint(
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            output_path=output_path,
            resume_config=resume_config,
            completed_sample_ids=completed_sample_ids,
            state=current_checkpoint_state(),
        )
        print(f"Checkpoint saved after {sample_id}: {checkpoint_path}")

    non_add_events = add_event_counts["UPDATE"] + add_event_counts["DELETE"] + add_event_counts["NOOP"] + add_event_counts["UNKNOWN"]
    add_event_warning = None
    if add_event_counts["ADD"] > 0 and non_add_events == 0:
        add_event_warning = (
            "All memory operations were ADD in this run; local SDK/wrapper behavior may differ from the paper update phase "
            "or this data/configuration did not trigger updates/deletes/noops."
        )

    wrapper_latency_summary = make_wrapper_latency_summary(
        wrapper_latency_totals,
        num_candidate_extractions=num_candidate_extractions,
        num_candidates=num_wrapper_candidates,
        num_update_decisions=num_update_decisions,
        num_information_content_guards=num_information_content_guards,
    )

    output = {
        "run_id": run_id,
        "method": args.method,
        "benchmark_protocol": "locomo_paper_style_local_oss",
        "checkpoint_path": str(checkpoint_path.resolve()),
        "resumed_from_checkpoint": str(Path(args.resume_checkpoint).resolve()) if args.resume_checkpoint else None,
        "qa_only_reuse_memory": args.qa_only_reuse_memory,
        "reused_memory_source_info": reused_memory_source_info,
        "reuse_ingested_sample_ids": sorted(reuse_ingested_sample_ids),
        "completed_sample_ids": sorted(completed_sample_ids),
        "local_sdk_capabilities": local_sdk_capabilities,
        "local_sdk_update_phase_detected": local_sdk_capabilities["local_sdk_update_phase_detected"],
        "local_sdk_update_similar_top_k": local_sdk_capabilities["local_sdk_update_similar_top_k"],
        "local_sdk_update_top_k_configurable": local_sdk_capabilities["local_sdk_update_top_k_configurable"],
        "local_sdk_final_store_fetch_supported": local_sdk_capabilities["local_sdk_final_store_fetch_supported"],
        "final_store_fetch_supported": (
            args.memory_update_mode == "paper_update_wrapper"
            or local_sdk_capabilities["local_sdk_final_store_fetch_supported"]
        ),
        "dataset": "locomo10",
        "dataset_path": str(dataset_path),
        "timestamp_schema": {
            "session_timestamp_raw": "Original LOCOMO session timestamp string.",
            "observed_at": "Canonical UTC ISO-8601 time when the conversation was observed.",
            "event_at": "Canonical event time inferred from memory content; null when unknown.",
            "event_time_text": "Original relative or absolute event-time expression from the dialogue.",
            "created_at": "Store write time; never used as the conversation observation time.",
            "updated_at": "Store update time; never used as the memory event time.",
        },
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "model_runtime_defaults": model_runtime_defaults,
        "model_env_var": "MODEL",
        "embedding_model_env_var": "EMBEDDING_MODEL",
        "chat_client": chat_client_config,
        "embedding_client": embedding_client_config,
        "embedding_recovery_log": wrapper_store.embedding_recovery_log if wrapper_store else [],
        "memory_update_mode": args.memory_update_mode,
        "update_similar_top_k": args.update_similar_top_k,
        "candidate_extraction_max_tokens": args.candidate_extraction_max_tokens,
        "update_decision_max_tokens": args.update_decision_max_tokens,
        "memory_wrapper_store": args.memory_wrapper_store,
        "paper_update_wrapper_note": (
            "Mem0-inspired local prototype; not official Mem0 SDK implementation and not an exact paper reproduction."
        ),
        "top_k": args.top_k,
        "top_k_per_speaker": args.top_k,
        "max_total_retrieved_memories": (
            (min(args.top_k, args.protected_semantic_top_k) + args.active_extra_k_per_speaker) * 2
            if args.qa_retrieval_mode == "hybrid_adaptive"
            else (
                (min(args.top_k, args.protected_semantic_top_k) + args.egc_max_gap_evidence_per_speaker) * 2
                if args.qa_retrieval_mode == "evidence_gap_hybrid"
                else (
                    args.qase_max_total_final_memories
                    if args.qa_retrieval_mode in QUESTION_AWARE_RETRIEVAL_MODES
                    else args.top_k * 2
                )
            )
        ),
        "search_threshold": args.search_threshold,
        "batch_size": args.batch_size,
        "context_window": args.context_window,
        "include_context_in_add_prompt": args.include_context_in_add_prompt,
        "paper_extraction_mode": args.paper_extraction_mode,
        "effective_paper_extraction_mode": effective_paper_extraction_mode,
        "local_sdk_context_isolation_not_guaranteed": effective_paper_extraction_mode != "target_only",
        "summary_mode": args.summary_mode,
        "effective_summary_mode": effective_summary_mode,
        "summary_model": args.summary_model,
        "summary_max_tokens": args.summary_max_tokens,
        "summary_update_scope": args.summary_update_scope,
        "summary_usage_totals": summary_usage_totals,
        "summary_snapshots": summary_snapshots,
        "target_only_custom_instructions": args.target_only_custom_instructions,
        "custom_instructions_mode": (
            "disabled"
            if args.no_custom_instructions
            else ("target_only" if args.target_only_custom_instructions else "general")
        ),
        "retrieval_token_budget": args.retrieval_token_budget,
        "retrieval_budget_strategy": args.retrieval_budget_strategy,
        "retrieved_memory_token_encoding": args.retrieved_memory_token_encoding,
        "qa_retrieval_mode": args.qa_retrieval_mode,
        "active_reconstruction_config": {
            "inspired_by": "MRAgent active memory reconstruction / Cue-Tag-Content graph",
            "scope": (
                "Retrieval-layer improvement over the local Mem0 final memory store; "
                "Mem0 candidate extraction and ADD/UPDATE/DELETE/NOOP update logic are kept."
            ),
            "max_steps": args.active_reconstruction_max_steps,
            "max_tool_calls_per_step": args.active_reconstruction_max_tool_calls_per_step,
            "seed_top_k_per_speaker": (
                args.active_reconstruction_seed_top_k if args.qa_retrieval_mode == "active_reconstruction" else None
            ),
            "hybrid_effective_seed_top_k_per_speaker": (
                args.har_active_seed_top_k if args.qa_retrieval_mode == "hybrid_adaptive" else None
            ),
            "use_llm_router": args.active_reconstruction_use_llm_router,
            "router_max_tokens": args.active_reconstruction_router_max_tokens,
            "hybrid_router_mode": args.hybrid_router_mode,
            "protected_semantic_top_k": args.protected_semantic_top_k,
            "active_extra_k_per_speaker": args.active_extra_k_per_speaker,
            "har_active_seed_top_k": args.har_active_seed_top_k,
            "har_evidence_rerank_mode": args.har_evidence_rerank_mode,
            "har_evidence_rerank_trigger": args.har_evidence_rerank_trigger,
            "har_evidence_rerank_max_candidates": args.har_evidence_rerank_max_candidates,
            "har_evidence_rerank_max_tokens": args.har_evidence_rerank_max_tokens,
            "har_sufficiency_mode": args.har_sufficiency_mode,
            "har_sufficiency_max_rounds": args.har_sufficiency_max_rounds,
            "har_sufficiency_max_candidates": args.har_sufficiency_max_candidates,
            "har_sufficiency_max_tokens": args.har_sufficiency_max_tokens,
            "hybrid_note": (
                "Only used by --qa_retrieval_mode hybrid_adaptive. The semantic baseline is protected first; "
                "adaptive retrieval can add extra evidence for complex queries instead of replacing baseline top-k. "
                "If HAR sufficiency is enabled, insufficient evidence can trigger bounded follow-up retrieval rounds."
            ),
            "egc_contract_mode": args.egc_contract_mode,
            "egc_coverage_mode": args.egc_coverage_mode,
            "egc_max_gap_rounds": args.egc_max_gap_rounds,
            "egc_max_slots": args.egc_max_slots,
            "egc_min_slot_coverage": args.egc_min_slot_coverage,
            "egc_gap_top_k_per_slot": args.egc_gap_top_k_per_slot,
            "egc_max_gap_evidence_per_speaker": args.egc_max_gap_evidence_per_speaker,
            "egc_proof_pack_rerank": args.egc_proof_pack_rerank,
            "egc_keep_protected_baseline": args.egc_keep_protected_baseline,
            "egc_contract_max_tokens": args.egc_contract_max_tokens,
            "egc_coverage_max_candidates": args.egc_coverage_max_candidates,
            "egc_coverage_max_tokens": args.egc_coverage_max_tokens,
            "egc_proof_pack_max_candidates": args.egc_proof_pack_max_candidates,
            "egc_proof_pack_max_tokens": args.egc_proof_pack_max_tokens,
            "egc_max_tool_calls_per_round": args.egc_max_tool_calls_per_round,
            "egc_note": (
                "Only used by --qa_retrieval_mode evidence_gap_hybrid. EGC builds evidence slots, checks "
                "baseline coverage, retrieves only missing slots, and can optionally rerank proof packs."
            ),
            "qase_strategy_version": "stable_v2_slim_question_aware_budget",
            "qase_adaptive_budget_profile_name": ADAPTIVE_CUTOFF_PROFILE_NAME,
            "qase_adaptive_budget_profile": args.qase_adaptive_budget_profile,
            "qase_adaptive_total_budget_by_question_type": ADAPTIVE_TOTAL_BUDGET_BY_QUESTION_TYPE,
            "qase_hybrid_fusion_profile_name": HYBRID_FUSION_PROFILE_NAME,
            "qase_hybrid_fusion_weights": {
                "semantic": args.qase_hybrid_semantic_weight,
                "bm25": args.qase_hybrid_bm25_weight,
                "entity": args.qase_hybrid_entity_weight,
                "temporal": args.qase_hybrid_temporal_weight,
            },
            "qase_max_total_final_memories": args.qase_max_total_final_memories,
            "qase_candidate_pool_scale": args.qase_candidate_pool_scale,
            "qase_final_k_scale": args.qase_final_k_scale,
            "qase_budget_profile": args.qase_budget_profile,
            "qase_budget_allocation_mode": args.qase_budget_allocation_mode,
            "qase_selection_scope": args.qase_selection_scope,
            "qase_adaptive_cutoff_enabled": not args.qase_disable_adaptive_cutoff,
            "qase_adaptive_cutoff_mode": qase_controller.config.adaptive_cutoff_mode,
            "qase_bm25_candidate_rerank": args.qase_bm25_candidate_rerank,
            "qase_bm25_k1": args.qase_bm25_k1,
            "qase_bm25_b": args.qase_bm25_b,
            "qase_lexical_weight": qase_controller.config.lexical_weight,
            "qase_temporal_bonus_weight": qase_controller.config.temporal_bonus_weight,
            "qase_planner_model_root": qase_controller.config.planner_model_root,
            "qase_question_type_model_dir": qase_controller.config.question_type_model_dir,
            "qase_target_speaker_model_dir": qase_controller.config.target_speaker_model_dir,
            "qase_planner_model_device": qase_controller.config.planner_model_device,
            "qase_diversity_enabled": (
                qase_controller.config.diversity_lambda_simple > 0
                or qase_controller.config.diversity_lambda_complex > 0
            ),
            "qase_diversity_lambda": {
                "simple": qase_controller.config.diversity_lambda_simple,
                "complex": qase_controller.config.diversity_lambda_complex,
            },
            "qase_confidence_fallback_enabled": qase_controller.config.confidence_fallback_enabled,
            "qase_confidence_fallback_low_score_threshold": args.qase_confidence_fallback_low_score_threshold,
            "qase_semantic_anchor_enabled": qase_controller.config.semantic_anchor_enabled,
            "qase_semantic_anchor_simple": qase_controller.config.semantic_anchor_simple,
            "qase_semantic_anchor_complex": qase_controller.config.semantic_anchor_complex,
            "qase_multi_hop_candidate_target": qase_controller.config.candidate_multi_hop_target,
            "qase_multi_hop_final_target": qase_controller.config.final_multi_hop_target,
            "qase_min_by_type": {
                "single_hop": qase_controller.config.min_single_hop,
                "temporal": qase_controller.config.min_temporal,
                "multi_hop": qase_controller.config.min_multi_hop,
                "fallback_ambiguous": qase_controller.config.min_fallback_ambiguous,
            },
            "qase_note": (
                "Used by question-aware QA modes. question_aware_selective classifies the question, retrieves "
                "a larger candidate pool when needed, then selects a compact deterministic evidence pack. "
                "With --qase_bm25_candidate_rerank, it also builds a per-speaker BM25 index over the existing "
                "local memory store during QA and uses BM25 only as a lexical reranking feature for semantic "
                "QASE candidates; it does not add BM25-only memories to the selector pool. "
                "question_aware_semantic is the control variant: it uses the same planner/final budget but keeps "
                "pure semantic top-k without evidence scoring, adaptive cutoff, diversity, anchors, or fallback. "
                "question_aware_semantic_adaptive retrieves the planner candidate pool and applies a type-specific "
                "balanced memory budget with adaptive-k similarity thresholding by default. With "
                "--qase_adaptive_budget_profile adaptive_k_only, it removes type-specific final floor/cap guardrails "
                "and lets Adaptive-k choose final k from the candidate pool. question_aware_hybrid_adaptive uses "
                "the same adaptive selection path, but first re-scores the candidate pool with semantic, BM25, "
                "entity-match, and temporal-match fusion."
            ),
        },
        "active_reconstruction_usage_totals": active_reconstruction_usage_totals,
        "active_reconstruction_latency_totals": active_reconstruction_latency_totals,
        "custom_instructions_enabled": not args.no_custom_instructions,
        "max_conversations": args.max_conversations,
        "max_sessions": args.max_sessions,
        "max_qa": args.max_qa,
        "skip_category_5": args.skip_category_5,
        "total_qa_seen": total_qa_seen,
        "total_qa_answered": total_qa_answered,
        "configuration_warnings": configuration_warnings,
        "add_logs": add_logs,
        "add_event_counts": add_event_counts,
        "add_event_counts_by_sample": add_event_counts_by_sample,
        "add_event_counts_by_speaker": add_event_counts_by_speaker,
        "add_event_counts_by_session": add_event_counts_by_session,
        "add_event_counts_by_batch": add_event_counts_by_batch,
        "add_event_warning": add_event_warning,
        "paper_update_logs": paper_update_logs,
        "candidate_extraction_usage_totals": candidate_extraction_usage_totals,
        "candidate_extraction_tokens_total": candidate_extraction_usage_totals["candidate_extraction_total_tokens"],
        "update_decision_usage_totals": update_decision_usage_totals,
        "update_decision_tokens_total": update_decision_usage_totals["update_decision_total_tokens"],
        "information_content_guard_usage_totals": information_content_guard_usage_totals,
        "information_content_guard_tokens_total": information_content_guard_usage_totals[
            "information_content_guard_total_tokens"
        ],
        "num_information_content_guards": num_information_content_guards,
        "num_updates_rejected_by_information_content_guard": (
            num_updates_rejected_by_information_content_guard
        ),
        "wrapper_add_update_delete_noop_counts": (
            add_event_counts if args.memory_update_mode == "paper_update_wrapper" else None
        ),
        "wrapper_latency_summary": wrapper_latency_summary if args.memory_update_mode == "paper_update_wrapper" else None,
        "all_extracted_memories_by_sample": all_extracted_memories_by_sample,
        "memory_add_output_tokens_by_sample": memory_add_output_tokens_by_sample,
        "memory_store_tokens_by_sample": memory_add_output_tokens_by_sample,
        "final_memory_store_tokens_by_sample": final_memory_store_tokens_by_sample,
        "final_num_memories_by_sample": final_num_memories_by_sample,
        "final_store_fetch_warnings": final_store_fetch_warnings,
        "final_store_top_k": args.final_store_top_k,
        "results": results,
        "limitations_note": (
            "Local OSS Memory.from_config is not guaranteed to reproduce the hosted Mem0Client/paper benchmark exactly. "
            "The paper describes conversation summary S plus m=10 recent messages and s=10 similar memories for the update phase; "
            "this runner exposes explicit summary/m-context modes, but the local SDK does not provide a stable public "
            "runner-level API for configuring the update-phase similar-memory count. The local SDK may internally retrieve "
            "similar existing memories during add, but the exact s=10 paper parameter is not exposed as a stable runner-level configuration. "
            "paper_update_wrapper is a Mem0-inspired local prototype that implements candidate extraction plus ADD/UPDATE/DELETE/NOOP "
            "decisioning in a json_local in-memory store; it is not the official Mem0 SDK update implementation and should be reported as such. "
            "By default paper_extraction_mode is target_only because the local SDK cannot guarantee isolation of context-only messages from "
            "target extraction. If m-context or summary modes are selected, context/target markers are added, but the local SDK/wrapper LLM may still "
            "extract from context despite instructions. retrieval_token_budget is an ablation option, not a paper hyperparameter."
        ),
        "paper_alignment_note": {
            "matched": [
                "LOCOMO-style per-speaker memory namespaces and perspective views",
                "target batch default batch_size=2 approximates message-pair ingestion",
                "GPT-4o-mini default LLM and text-embedding-3-small dense embeddings",
                "adversarial/category 5 skipped by default",
                "answer prompt uses separate timestamped memories for both speakers",
                "retrieved_memory_tokens are counted with cl100k_base by default",
                "optional retrieval_token_budget can trim retrieved memories toward a target memory-token budget",
            ],
            "partially_matched": [
                (
                    "m=10 recent messages is explicit only when effective_paper_extraction_mode includes m_context; "
                    "the inspected local SDK also has its own internal last_k_messages context."
                ),
                "summary S is included only when effective_summary_mode=rolling_llm and effective_paper_extraction_mode=include_summary_and_m_context",
                "s=10 update phase is not controlled by this runner unless local_sdk_update_phase_detected and configurable are true",
                "paper_update_wrapper controls top s through update_similar_top_k, but it is a local prototype rather than hosted Mem0",
                "paper_update_wrapper separately verifies Algorithm 1's InformationContent condition before applying UPDATE",
                "memory store tokens are final for paper_update_wrapper/json_local; for sdk_additive they are final only when local_sdk_final_store_fetch_supported is true and get_all succeeds",
            ],
            "not_matched_or_local_only": [
                "Hosted Mem0Client/version=v2 internals are not reproduced by this local OSS runner",
                "Exact hosted/paper update phase is not reproduced when local_sdk_update_phase_detected is false",
                "top_k_per_speaker is a local/repo-style configurable QA retrieval setting, not a paper PDF parameter",
                "Mem0g/graph memory is not enabled in this runner",
                "Exact paper infrastructure latency is not reproduced in local Qdrant/OpenAI calls",
            ],
            "retrieval_budget_note": (
                "Default retrieval_budget_strategy is interleave_rank: memories are interleaved by original rank across the two "
                "speakers, then kept until the token budget is reached. It is not balanced_per_speaker or global_score unless "
                "that strategy is explicitly selected."
            ),
        },
    }

    atomic_write_json(output_path, output)
    write_run_checkpoint(
        checkpoint_path=checkpoint_path,
        run_id=run_id,
        output_path=output_path,
        resume_config=resume_config,
        completed_sample_ids=completed_sample_ids,
        state=current_checkpoint_state(),
        completed=True,
    )

    print("\nSaved result JSON:", output_path)
    print("Total QA answered:", total_qa_answered)
    print("Total add operations:", len(add_logs))


if __name__ == "__main__":
    run()
