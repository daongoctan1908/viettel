import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import string
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


PAPER_BINARY_JUDGE_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {ground_truth}
Generated answer: {prediction}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Just return the label CORRECT or WRONG in a json format with the key as "label".
""".strip()


def normalize_answer(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    common = 0
    for token in gold_tokens:
        if pred_counts.get(token, 0) > 0:
            common += 1
            pred_counts[token] -= 1

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_counts = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1

    clipped_matches = 0
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            clipped_matches += 1
            gold_counts[token] -= 1

    precision = clipped_matches / len(pred_tokens)

    if len(pred_tokens) > len(gold_tokens):
        bp = 1.0
    else:
        bp = math.exp(1 - len(gold_tokens) / max(len(pred_tokens), 1))

    return bp * precision


def safe_mean(values: List[float]) -> float:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else 0.0


def percentile(values: List[float], p: float) -> float:
    values = sorted([v for v in values if v is not None])
    if not values:
        return 0.0
    k = (len(values) - 1) * (p / 100)
    lower = int(k)
    upper = min(lower + 1, len(values) - 1)
    weight = k - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def trimmed_mean_without_max(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    values = sorted(values)
    return safe_mean(values[:-1])


def max_latency_question(rows: List[Dict[str, Any]], latency_key: str = "total_latency_sec") -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    row = max(rows, key=lambda item: float(item.get(latency_key) or 0.0))
    return {
        "sample_id": row.get("sample_id"),
        "qa_index": row.get("qa_index"),
        "category": row.get("category"),
        "question": row.get("question"),
        "search_latency_sec": row.get("search_latency_sec"),
        "generation_latency_sec": row.get("generation_latency_sec"),
        "total_latency_sec": row.get("total_latency_sec"),
    }


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, input_price_per_1m: float, output_price_per_1m: float) -> float:
    return prompt_tokens / 1_000_000 * input_price_per_1m + completion_tokens / 1_000_000 * output_price_per_1m


def optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_json_loads(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def stable_judge_key(item: Dict[str, Any], question: str, ground_truth: str, prediction: str) -> str:
    payload = {
        "sample_id": item.get("sample_id"),
        "qa_index": item.get("qa_index"),
        "question": question,
        "ground_truth": ground_truth,
        "predicted_answer": prediction,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_judge_cache(cache_file: Optional[str]) -> Dict[str, Any]:
    if not cache_file:
        return {}
    path = Path(cache_file)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_judge_cache(cache_file: Optional[str], cache: Dict[str, Any]) -> None:
    if not cache_file:
        return
    path = Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary_path, path)


def get_env_value(name: str) -> Optional[str]:
    value = os.getenv(name)
    value = value.strip() if value else ""
    return value or None


def create_chat_client(api_max_retries: int, api_timeout_sec: float) -> tuple[OpenAI, Dict[str, Any]]:
    beeknoee_api_key = get_env_value("BEEKNOEE_API_KEY")
    beeknoee_base_url = get_env_value("BEEKNOEE_BASE_URL")
    if not beeknoee_api_key:
        raise RuntimeError("BEEKNOEE_API_KEY is required for judge chat completions; OPENAI_API_KEY is used only for embeddings.")
    if not beeknoee_base_url:
        raise RuntimeError("BEEKNOEE_BASE_URL is required for judge chat completions.")
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


def judge_answer(
    chat_client: OpenAI,
    judge_model: str,
    question: str,
    ground_truth: str,
    prediction: str,
    *,
    mode: str,
    temperature: float,
) -> Dict[str, Any]:
    if mode == "paper_binary":
        prompt = PAPER_BINARY_JUDGE_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            prediction=prediction,
        )
        messages = [{"role": "user", "content": prompt}]
    else:
        prompt = f"""
You are evaluating a predicted answer for a long-term memory QA benchmark.

Question:
{question}

Ground truth answer:
{ground_truth}

Predicted answer:
{prediction}

Score:
- 1.0 = correct and sufficiently complete
- 0.5 = partially correct, incomplete, or too vague
- 0.0 = incorrect, unsupported, or contradicts the ground truth

Accept paraphrases. Penalize missing dates, names, entities, or temporal details when required.

Return only valid JSON:
{{
  "score": 1.0,
  "reason": "short explanation"
}}
""".strip()
        messages = [
            {"role": "system", "content": "You are a strict but fair evaluator. Return JSON only."},
            {"role": "user", "content": prompt},
        ]

    response = chat_client.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    usage = response.usage.model_dump() if response.usage else {}

    try:
        parsed = safe_json_loads(raw)
        if mode == "paper_binary":
            label = str(parsed.get("label", "")).upper().strip()
            if label not in {"CORRECT", "WRONG"}:
                return {
                    "score": 0.0,
                    "label": "INVALID_LABEL",
                    "reason": parsed.get("reason", f"Invalid label: {label}"),
                    "usage": usage,
                    "raw": raw,
                }
            score = 1.0 if label == "CORRECT" else 0.0
            return {
                "score": score,
                "label": label,
                "reason": parsed.get("reason", ""),
                "usage": usage,
                "raw": raw,
            }

        score = float(parsed.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "label": parsed.get("label"),
            "reason": parsed.get("reason", ""),
            "usage": usage,
            "raw": raw,
        }
    except Exception:
        return {
            "score": 0.0,
            "label": "PARSE_ERROR",
            "reason": f"Failed to parse judge response: {raw}",
            "usage": usage,
            "raw": raw,
        }


def get_usage(item: Dict[str, Any]) -> Dict[str, int]:
    usage = item.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def run_paper_judge_rounds(
    *,
    results: List[Dict[str, Any]],
    chat_client: OpenAI,
    judge_model: str,
    judge_mode: str,
    judge_runs: int,
    judge_temperature: float,
    judge_limit: Optional[int],
    judge_cache_file: Optional[str],
    judge_cache: Dict[str, Any],
    evaluation_run_id: str,
) -> Dict[str, int]:
    judged_results = results[:judge_limit] if judge_limit is not None else results
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    for run_index in range(judge_runs):
        print(f"\n=== JUDGE RUN {run_index + 1}/{judge_runs} ===")
        for item_index, item in enumerate(judged_results):
            question = str(item.get("question", ""))
            ground_truth = str(item.get("ground_truth", item.get("answer", "")))
            prediction = str(item.get("predicted_answer", item.get("response", "")))
            judge_cache_key = stable_judge_key(item, question, ground_truth, prediction)
            cache_entry = judge_cache.setdefault(
                judge_cache_key,
                {
                    "sample_id": item.get("sample_id"),
                    "qa_index": item.get("qa_index"),
                    "question": question,
                    "ground_truth": ground_truth,
                    "predicted_answer": prediction,
                    "judge_results": [],
                },
            )
            cached_results = cache_entry.setdefault("judge_results", [])

            if len(cached_results) > run_index:
                cached_result = cached_results[run_index]
                cached_result.setdefault("run_index", run_index + 1)
                label = cached_result.get("label") or "CACHED"
                print(
                    f"Judge run {run_index + 1}/{judge_runs} | "
                    f"{item_index + 1}/{len(judged_results)}: {label} (cached)"
                )
                continue

            judge = judge_answer(
                chat_client,
                judge_model,
                question,
                ground_truth,
                prediction,
                mode=judge_mode,
                temperature=judge_temperature,
            )
            judge_record = {
                "score": judge["score"],
                "label": judge.get("label"),
                "reason": judge.get("reason"),
                "raw": judge.get("raw"),
                "usage": judge.get("usage") or {},
                "judge_model": judge_model,
                "judge_mode": judge_mode,
                "run_index": run_index + 1,
                "run_id": evaluation_run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            cached_results.append(judge_record)
            save_judge_cache(judge_cache_file, judge_cache)

            judge_usage = judge_record["usage"]
            usage_totals["prompt_tokens"] += int(judge_usage.get("prompt_tokens") or 0)
            usage_totals["completion_tokens"] += int(judge_usage.get("completion_tokens") or 0)
            usage_totals["total_tokens"] += int(judge_usage.get("total_tokens") or 0)
            print(
                f"Judge run {run_index + 1}/{judge_runs} | "
                f"{item_index + 1}/{len(judged_results)}: {judge_record['label']}"
            )

    return usage_totals


def main():
    parser = argparse.ArgumentParser(description="Unified evaluator for memory benchmark outputs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=None)

    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--judge_limit", type=int, default=None)
    parser.add_argument("--judge_mode", choices=["paper_binary", "graded"], default="paper_binary")
    parser.add_argument("--judge_runs", type=int, default=1)
    parser.add_argument("--judge_temperature", type=float, default=0.0)
    parser.add_argument("--judge_cache_file", default=None)
    parser.add_argument("--append_judge_cache", action="store_true")
    parser.add_argument("--api_max_retries", type=int, default=10)
    parser.add_argument("--api_timeout_sec", type=float, default=600.0)

    parser.add_argument("--input_price_per_1m", type=float, default=0.15)
    parser.add_argument("--output_price_per_1m", type=float, default=0.60)

    args = parser.parse_args()
    if args.judge_runs <= 0:
        raise ValueError("--judge_runs must be greater than 0")
    if args.api_max_retries < 0:
        raise ValueError("--api_max_retries must be >= 0")
    if args.api_timeout_sec <= 0:
        raise ValueError("--api_timeout_sec must be greater than 0")

    root_dir = Path(__file__).resolve().parents[1]
    load_dotenv(root_dir / ".env", override=True)
    args.judge_model = args.judge_model or get_env_value("JUDGE_MODEL") or get_env_value("MODEL") or "gpt-4o-mini"

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    method = run_data.get("method", run_data.get("mode", "unknown_method"))
    run_id = run_data.get("run_id", input_path.stem)
    llm_model = run_data.get("llm_model", "unknown")
    embedding_model = run_data.get("embedding_model", "unknown")
    memory_update_mode = run_data.get("memory_update_mode", "unknown")
    memory_wrapper_store = run_data.get("memory_wrapper_store")
    results = run_data.get("results", [])

    if not results:
        raise ValueError("No results found. Expected top-level key: results")

    chat_client, chat_client_config = (
        create_chat_client(args.api_max_retries, args.api_timeout_sec) if args.judge else (None, None)
    )
    judge_cache = load_judge_cache(args.judge_cache_file) if args.judge else {}
    evaluation_run_id = datetime.now(timezone.utc).isoformat()

    rows = []
    judge_prompt_tokens = 0
    judge_completion_tokens = 0
    judge_total_tokens = 0

    if args.judge:
        judge_usage_totals = run_paper_judge_rounds(
            results=results,
            chat_client=chat_client,
            judge_model=args.judge_model,
            judge_mode=args.judge_mode,
            judge_runs=args.judge_runs,
            judge_temperature=args.judge_temperature,
            judge_limit=args.judge_limit,
            judge_cache_file=args.judge_cache_file,
            judge_cache=judge_cache,
            evaluation_run_id=evaluation_run_id,
        )
        judge_prompt_tokens = judge_usage_totals["prompt_tokens"]
        judge_completion_tokens = judge_usage_totals["completion_tokens"]
        judge_total_tokens = judge_usage_totals["total_tokens"]

    for idx, item in enumerate(results):
        question = str(item.get("question", ""))
        ground_truth = str(item.get("ground_truth", item.get("answer", "")))
        prediction = str(item.get("predicted_answer", item.get("response", "")))
        category = str(item.get("category", "unknown"))

        em = exact_match(prediction, ground_truth)
        f1 = token_f1(prediction, ground_truth)
        b1 = bleu1(prediction, ground_truth)

        search_latency = float(item.get("search_latency_sec") or 0.0)
        generation_latency = float(item.get("generation_latency_sec") or 0.0)
        total_latency = float(item.get("total_latency_sec") or (search_latency + generation_latency))
        semantic_search_latency = float(item.get("semantic_search_latency_sec") or 0.0)
        qase_bm25_latency = float(item.get("qase_bm25_latency_sec") or 0.0)
        qase_planner_latency = float(item.get("qase_planner_latency_sec") or 0.0)
        qase_selector_latency = float(item.get("qase_selector_latency_sec") or 0.0)

        usage = get_usage(item)

        judge_score = None
        judge_reason = None
        judge_scores = []
        judge_labels = []
        judge_reasons = []
        judge_cache_key = stable_judge_key(item, question, ground_truth, prediction)

        if args.judge and (args.judge_limit is None or idx < args.judge_limit):
            cache_entry = judge_cache.get(judge_cache_key) or {}
            cached_results = cache_entry.get("judge_results") or []
            effective_results = cached_results[: args.judge_runs]
            judge_scores = [float(result.get("score") or 0.0) for result in effective_results]
            judge_labels = [result.get("label") for result in effective_results]
            judge_reasons = [result.get("reason") for result in effective_results if result.get("reason")]

            judge_score = safe_mean(judge_scores)
            judge_reason = " | ".join(judge_reasons)

        retrieved_memories = item.get("retrieved_memories") or []
        top1_memory_score = None
        if retrieved_memories and isinstance(retrieved_memories[0], dict):
            top1_memory_score = retrieved_memories[0].get("score")

        retrieved_memory_tokens = optional_int(item.get("retrieved_memory_tokens"))
        speaker_1_memory_tokens = optional_int(item.get("speaker_1_memory_tokens"))
        speaker_2_memory_tokens = optional_int(item.get("speaker_2_memory_tokens"))
        num_speaker_1_memories = optional_int(item.get("num_speaker_1_memories"))
        num_speaker_2_memories = optional_int(item.get("num_speaker_2_memories"))
        top_k_per_speaker = optional_int(item.get("top_k_per_speaker"))
        max_total_retrieved_memories = optional_int(item.get("max_total_retrieved_memories"))
        retrieval_token_budget = optional_int(item.get("retrieval_token_budget"))
        retrieval_budget_strategy = item.get("retrieval_budget_strategy")
        retrieved_memory_tokens_pre_trim = optional_int(item.get("retrieved_memory_tokens_pre_trim"))
        speaker_1_memory_tokens_pre_trim = optional_int(item.get("speaker_1_memory_tokens_pre_trim"))
        speaker_2_memory_tokens_pre_trim = optional_int(item.get("speaker_2_memory_tokens_pre_trim"))
        num_retrieved_memories_pre_trim = optional_int(item.get("num_retrieved_memories_pre_trim"))
        num_retrieved_memories_dropped_by_budget = optional_int(item.get("num_retrieved_memories_dropped_by_budget"))

        rows.append({
            "method": method,
            "run_id": run_id,
            "sample_id": item.get("sample_id"),
            "qa_index": item.get("qa_index"),
            "category": category,
            "memory_update_mode": item.get("memory_update_mode", memory_update_mode),
            "memory_wrapper_store": item.get("memory_wrapper_store", memory_wrapper_store),
            "judge_mode": args.judge_mode if args.judge else None,
            "judge_runs": args.judge_runs if args.judge else 0,

            "question": question,
            "ground_truth": ground_truth,
            "predicted_answer": prediction,

            "exact_match": em,
            "f1": f1,
            "bleu1": b1,
            "judge_score": judge_score,
            "judge_reason": judge_reason,
            "judge_scores": judge_scores,
            "judge_labels": judge_labels,
            "judge_cache_key": judge_cache_key if args.judge else None,

            "search_latency_sec": search_latency,
            "semantic_search_latency_sec": semantic_search_latency,
            "qase_bm25_latency_sec": qase_bm25_latency,
            "qase_planner_latency_sec": qase_planner_latency,
            "qase_selector_latency_sec": qase_selector_latency,
            "generation_latency_sec": generation_latency,
            "total_latency_sec": total_latency,

            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],

            "num_retrieved_memories": len(retrieved_memories),
            "num_speaker_1_memories": num_speaker_1_memories,
            "num_speaker_2_memories": num_speaker_2_memories,
            "top_k_per_speaker": top_k_per_speaker,
            "max_total_retrieved_memories": max_total_retrieved_memories,
            "retrieval_token_budget": retrieval_token_budget,
            "retrieval_budget_strategy": retrieval_budget_strategy,
            "retrieval_token_budget_applied": item.get("retrieval_token_budget_applied"),
            "speaker_1_memory_tokens": speaker_1_memory_tokens,
            "speaker_2_memory_tokens": speaker_2_memory_tokens,
            "retrieved_memory_tokens": retrieved_memory_tokens,
            "speaker_1_memory_tokens_pre_trim": speaker_1_memory_tokens_pre_trim,
            "speaker_2_memory_tokens_pre_trim": speaker_2_memory_tokens_pre_trim,
            "retrieved_memory_tokens_pre_trim": retrieved_memory_tokens_pre_trim,
            "num_retrieved_memories_pre_trim": num_retrieved_memories_pre_trim,
            "num_retrieved_memories_dropped_by_budget": num_retrieved_memories_dropped_by_budget,
            "top1_memory_score": top1_memory_score,
        })

    if args.judge and args.judge_cache_file:
        save_judge_cache(args.judge_cache_file, judge_cache)

    prompt_tokens = sum(r["prompt_tokens"] for r in rows)
    completion_tokens = sum(r["completion_tokens"] for r in rows)
    total_tokens = sum(r["total_tokens"] for r in rows)

    answer_cost = estimate_cost_usd(prompt_tokens, completion_tokens, args.input_price_per_1m, args.output_price_per_1m)
    judge_cost = estimate_cost_usd(judge_prompt_tokens, judge_completion_tokens, args.input_price_per_1m, args.output_price_per_1m)

    judge_values = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    judge_run_means = []
    max_judge_runs = max((len(r["judge_scores"]) for r in rows), default=0)
    for run_index in range(max_judge_runs):
        run_values = [r["judge_scores"][run_index] for r in rows if len(r["judge_scores"]) > run_index]
        if run_values:
            judge_run_means.append(safe_mean(run_values))
    judge_run_std = statistics.stdev(judge_run_means) if len(judge_run_means) > 1 else 0.0
    judge_run_counts = [len(r["judge_scores"]) for r in rows if r.get("judge_scores")]
    retrieved_memory_token_values = [
        r["retrieved_memory_tokens"] for r in rows if r.get("retrieved_memory_tokens") is not None
    ]
    retrieved_memory_token_pre_trim_values = [
        r["retrieved_memory_tokens_pre_trim"] for r in rows if r.get("retrieved_memory_tokens_pre_trim") is not None
    ]

    summary = {
        "method": method,
        "run_id": run_id,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "memory_update_mode": memory_update_mode,
        "memory_wrapper_store": memory_wrapper_store,
        "update_similar_top_k": run_data.get("update_similar_top_k"),
        "input_file": str(input_path),
        "num_questions": len(rows),
        "judge_mode": args.judge_mode if args.judge else None,
        "judge_model": args.judge_model if args.judge else None,
        "judge_model_env_var": "JUDGE_MODEL",
        "chat_client": chat_client_config if args.judge else None,
        "judge_runs": args.judge_runs if args.judge else 0,
        "judge_runs_requested": args.judge_runs if args.judge else 0,
        "judge_runs_effective_avg": safe_mean(judge_run_counts) if judge_run_counts else 0,
        "judge_cache_file": args.judge_cache_file,
        "judge_cache_num_items": len(judge_cache) if args.judge_cache_file else 0,

        "exact_match_avg": safe_mean([r["exact_match"] for r in rows]),
        "f1_avg": safe_mean([r["f1"] for r in rows]),
        "bleu1_avg": safe_mean([r["bleu1"] for r in rows]),
        "judge_score_avg": safe_mean(judge_values) if judge_values else None,
        "judge_score_run_means": judge_run_means,
        "judge_score_mean_across_runs": safe_mean(judge_run_means) if judge_run_means else None,
        "judge_score_std_across_runs": judge_run_std if judge_run_means else None,
        "num_judged": len(judge_values),

        "search_latency_avg": safe_mean([r["search_latency_sec"] for r in rows]),
        "search_latency_p50": percentile([r["search_latency_sec"] for r in rows], 50),
        "search_latency_p95": percentile([r["search_latency_sec"] for r in rows], 95),
        "search_latency_trimmed_avg_without_max": trimmed_mean_without_max([r["search_latency_sec"] for r in rows]),

        "generation_latency_avg": safe_mean([r["generation_latency_sec"] for r in rows]),
        "generation_latency_p50": percentile([r["generation_latency_sec"] for r in rows], 50),
        "generation_latency_p95": percentile([r["generation_latency_sec"] for r in rows], 95),

        "total_latency_avg": safe_mean([r["total_latency_sec"] for r in rows]),
        "total_latency_p50": percentile([r["total_latency_sec"] for r in rows], 50),
        "total_latency_p95": percentile([r["total_latency_sec"] for r in rows], 95),
        "total_latency_trimmed_avg_without_max": trimmed_mean_without_max([r["total_latency_sec"] for r in rows]),
        "max_latency_question": max_latency_question(rows),

        "retrieved_memory_tokens_avg": (
            safe_mean(retrieved_memory_token_values) if retrieved_memory_token_values else None
        ),
        "retrieved_memory_tokens_p50": (
            percentile(retrieved_memory_token_values, 50) if retrieved_memory_token_values else None
        ),
        "retrieved_memory_tokens_p95": (
            percentile(retrieved_memory_token_values, 95) if retrieved_memory_token_values else None
        ),
        "retrieved_memory_tokens_total": (
            sum(retrieved_memory_token_values) if retrieved_memory_token_values else None
        ),
        "retrieved_memory_tokens_pre_trim_avg": (
            safe_mean(retrieved_memory_token_pre_trim_values) if retrieved_memory_token_pre_trim_values else None
        ),
        "retrieved_memory_tokens_pre_trim_p50": (
            percentile(retrieved_memory_token_pre_trim_values, 50) if retrieved_memory_token_pre_trim_values else None
        ),
        "retrieved_memory_tokens_pre_trim_p95": (
            percentile(retrieved_memory_token_pre_trim_values, 95) if retrieved_memory_token_pre_trim_values else None
        ),
        "retrieved_memory_tokens_pre_trim_total": (
            sum(retrieved_memory_token_pre_trim_values) if retrieved_memory_token_pre_trim_values else None
        ),

        "answer_prompt_tokens": prompt_tokens,
        "answer_completion_tokens": completion_tokens,
        "answer_total_tokens": total_tokens,

        "judge_prompt_tokens": judge_prompt_tokens,
        "judge_completion_tokens": judge_completion_tokens,
        "judge_total_tokens": judge_total_tokens,

        "estimated_answer_cost_usd": answer_cost,
        "estimated_judge_cost_usd": judge_cost,
        "estimated_total_logged_cost_usd": answer_cost + judge_cost,

        "note": (
            "Cost estimate includes only logged answer-generation and judge tokens. "
            "It does not include internal framework calls during memory.add/search unless the runner logs them."
        ),
    }

    category_summary = {}
    for cat in sorted(set(r["category"] for r in rows)):
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_judge = [r["judge_score"] for r in cat_rows if r["judge_score"] is not None]
        cat_run_means = []
        cat_max_judge_runs = max((len(r["judge_scores"]) for r in cat_rows), default=0)
        for run_index in range(cat_max_judge_runs):
            run_values = [r["judge_scores"][run_index] for r in cat_rows if len(r["judge_scores"]) > run_index]
            if run_values:
                cat_run_means.append(safe_mean(run_values))
        cat_retrieved_memory_token_values = [
            r["retrieved_memory_tokens"] for r in cat_rows if r.get("retrieved_memory_tokens") is not None
        ]
        cat_retrieved_memory_token_pre_trim_values = [
            r["retrieved_memory_tokens_pre_trim"] for r in cat_rows if r.get("retrieved_memory_tokens_pre_trim") is not None
        ]

        category_summary[cat] = {
            "num_questions": len(cat_rows),
            "exact_match_avg": safe_mean([r["exact_match"] for r in cat_rows]),
            "f1_avg": safe_mean([r["f1"] for r in cat_rows]),
            "bleu1_avg": safe_mean([r["bleu1"] for r in cat_rows]),
            "judge_score_avg": safe_mean(cat_judge) if cat_judge else None,
            "judge_score_run_means": cat_run_means,
            "judge_score_std_across_runs": (
                statistics.stdev(cat_run_means) if len(cat_run_means) > 1 else (0.0 if cat_run_means else None)
            ),
            "search_latency_avg": safe_mean([r["search_latency_sec"] for r in cat_rows]),
            "search_latency_p50": percentile([r["search_latency_sec"] for r in cat_rows], 50),
            "search_latency_p95": percentile([r["search_latency_sec"] for r in cat_rows], 95),
            "search_latency_trimmed_avg_without_max": trimmed_mean_without_max([r["search_latency_sec"] for r in cat_rows]),
            "generation_latency_avg": safe_mean([r["generation_latency_sec"] for r in cat_rows]),
            "total_latency_avg": safe_mean([r["total_latency_sec"] for r in cat_rows]),
            "total_latency_p50": percentile([r["total_latency_sec"] for r in cat_rows], 50),
            "total_latency_p95": percentile([r["total_latency_sec"] for r in cat_rows], 95),
            "total_latency_trimmed_avg_without_max": trimmed_mean_without_max([r["total_latency_sec"] for r in cat_rows]),
            "max_latency_question": max_latency_question(cat_rows),
            "retrieved_memory_tokens_avg": (
                safe_mean(cat_retrieved_memory_token_values) if cat_retrieved_memory_token_values else None
            ),
            "retrieved_memory_tokens_p50": (
                percentile(cat_retrieved_memory_token_values, 50) if cat_retrieved_memory_token_values else None
            ),
            "retrieved_memory_tokens_p95": (
                percentile(cat_retrieved_memory_token_values, 95) if cat_retrieved_memory_token_values else None
            ),
            "retrieved_memory_tokens_total": (
                sum(cat_retrieved_memory_token_values) if cat_retrieved_memory_token_values else None
            ),
            "retrieved_memory_tokens_pre_trim_avg": (
                safe_mean(cat_retrieved_memory_token_pre_trim_values)
                if cat_retrieved_memory_token_pre_trim_values
                else None
            ),
            "retrieved_memory_tokens_pre_trim_p50": (
                percentile(cat_retrieved_memory_token_pre_trim_values, 50)
                if cat_retrieved_memory_token_pre_trim_values
                else None
            ),
            "retrieved_memory_tokens_pre_trim_p95": (
                percentile(cat_retrieved_memory_token_pre_trim_values, 95)
                if cat_retrieved_memory_token_pre_trim_values
                else None
            ),
            "retrieved_memory_tokens_pre_trim_total": (
                sum(cat_retrieved_memory_token_pre_trim_values)
                if cat_retrieved_memory_token_pre_trim_values
                else None
            ),
            "total_tokens": sum(r["total_tokens"] for r in cat_rows),
        }

    add_logs = run_data.get("add_logs") or []
    add_latencies = [float(x.get("latency_sec") or 0.0) for x in add_logs]

    def infer_context_sent(log_item: Dict[str, Any]) -> int:
        if "num_context_messages_sent" in log_item:
            return int(log_item.get("num_context_messages_sent") or 0)
        if log_item.get("include_context_in_add_prompt"):
            return int(log_item.get("num_context_messages") or 0)
        return 0

    def empty_event_counts() -> Dict[str, int]:
        return {"ADD": 0, "UPDATE": 0, "DELETE": 0, "NOOP": 0, "UNKNOWN": 0}

    def infer_add_event_counts() -> Dict[str, int]:
        counts = empty_event_counts()
        provided = run_data.get("add_event_counts")
        if isinstance(provided, dict):
            for key in counts:
                counts[key] = int(provided.get(key) or 0)
            return counts

        for log_item in add_logs:
            add_result = log_item.get("add_result") or {}
            for result_item in add_result.get("results", []) if isinstance(add_result, dict) else []:
                event = str(result_item.get("event", "UNKNOWN")).upper() if isinstance(result_item, dict) else "UNKNOWN"
                if event not in counts:
                    event = "UNKNOWN"
                counts[event] += 1
        return counts

    add_summary = {
        "num_add_operations": len(add_logs),
        "total_add_latency_sec": sum(add_latencies),
        "avg_add_latency_sec": safe_mean(add_latencies),
        "p50_add_latency_sec": percentile(add_latencies, 50),
        "p95_add_latency_sec": percentile(add_latencies, 95),
        "total_num_message_instances_added": sum(int(x.get("num_messages_added") or x.get("num_turns") or 0) for x in add_logs),
        "total_num_target_messages": sum(int(x.get("num_target_messages") or x.get("num_turns") or 0) for x in add_logs),
        "total_num_context_messages_available": sum(int(x.get("num_context_messages_available") or 0) for x in add_logs),
        "total_num_context_messages_in_window": sum(
            int(x.get("num_context_messages_in_window") if x.get("num_context_messages_in_window") is not None else x.get("num_context_messages") or 0)
            for x in add_logs
        ),
        "total_num_context_messages_sent": sum(infer_context_sent(x) for x in add_logs),
        "total_num_memories": sum(int(x.get("num_memories") or 0) for x in add_logs),
        "add_event_counts": infer_add_event_counts(),
        "add_event_counts_by_sample": run_data.get("add_event_counts_by_sample"),
        "note": (
            "Message counts are perspective/batch instances sent to memory.add or paper_update_wrapper, not raw LOCOMO dialogue turns. "
            "For the paper-style local runner each raw dialogue can appear in both speaker perspectives. "
            "Context messages available/in-window are logged separately from context messages actually sent to the memory update path."
        ),
    }

    memory_add_output_tokens_by_sample = (
        run_data.get("memory_add_output_tokens_by_sample") or run_data.get("memory_store_tokens_by_sample") or {}
    )
    memory_add_output_total_values = [
        optional_int(item.get("total"))
        for item in memory_add_output_tokens_by_sample.values()
        if isinstance(item, dict)
    ]
    memory_add_output_total_values = [value for value in memory_add_output_total_values if value is not None]
    memory_add_output_summary = {
        "num_samples": len(memory_add_output_tokens_by_sample),
        "memory_add_output_tokens_total": (
            sum(memory_add_output_total_values) if memory_add_output_total_values else None
        ),
        "memory_add_output_tokens_avg_per_sample": (
            safe_mean(memory_add_output_total_values) if memory_add_output_total_values else None
        ),
        "memory_add_output_tokens_p50": (
            percentile(memory_add_output_total_values, 50) if memory_add_output_total_values else None
        ),
        "memory_add_output_tokens_p95": (
            percentile(memory_add_output_total_values, 95) if memory_add_output_total_values else None
        ),
        "by_sample": memory_add_output_tokens_by_sample,
        "note": (
            "Counts memory outputs returned by memory.add or paper_update_wrapper, not necessarily the final persisted memory store "
            "if UPDATE/DELETE/NOOP operations occurred. It is separate from retrieved_memory_tokens per QA."
        ),
    }

    final_memory_store_tokens_by_sample = run_data.get("final_memory_store_tokens_by_sample") or {}
    final_memory_store_total_values = [
        optional_int(item.get("total"))
        for item in final_memory_store_tokens_by_sample.values()
        if isinstance(item, dict)
    ]
    final_memory_store_total_values = [value for value in final_memory_store_total_values if value is not None]
    final_memory_store_summary = {
        "fetch_supported": run_data.get(
            "final_store_fetch_supported",
            run_data.get("local_sdk_final_store_fetch_supported"),
        ),
        "fetch_warnings": run_data.get("final_store_fetch_warnings") or [],
        "num_samples": len(final_memory_store_tokens_by_sample),
        "final_memory_store_tokens_total": (
            sum(final_memory_store_total_values) if final_memory_store_total_values else None
        ),
        "final_memory_store_tokens_avg_per_sample": (
            safe_mean(final_memory_store_total_values) if final_memory_store_total_values else None
        ),
        "final_memory_store_tokens_p50": (
            percentile(final_memory_store_total_values, 50) if final_memory_store_total_values else None
        ),
        "final_memory_store_tokens_p95": (
            percentile(final_memory_store_total_values, 95) if final_memory_store_total_values else None
        ),
        "tokens_by_sample": final_memory_store_tokens_by_sample,
        "num_memories_by_sample": run_data.get("final_num_memories_by_sample") or {},
        "note": (
            "Counts final persisted memories fetched with Memory.get_all for sdk_additive or from LocalPaperMemoryStore "
            "for paper_update_wrapper when available."
        ),
    }

    wrapper_summary = {
        "memory_update_mode": memory_update_mode,
        "paper_update_wrapper_note": run_data.get("paper_update_wrapper_note"),
        "update_similar_top_k": run_data.get("update_similar_top_k"),
        "candidate_extraction_tokens_total": run_data.get("candidate_extraction_tokens_total"),
        "candidate_extraction_usage_totals": run_data.get("candidate_extraction_usage_totals"),
        "update_decision_tokens_total": run_data.get("update_decision_tokens_total"),
        "update_decision_usage_totals": run_data.get("update_decision_usage_totals"),
        "wrapper_add_update_delete_noop_counts": run_data.get("wrapper_add_update_delete_noop_counts"),
        "wrapper_latency_summary": run_data.get("wrapper_latency_summary"),
        "num_paper_update_logs": len(run_data.get("paper_update_logs") or []),
    }

    output_base = input_path.stem
    csv_path = output_dir / f"{output_base}_unified_metrics.csv"
    summary_path = output_dir / f"{output_base}_unified_summary.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "category_summary": category_summary,
                "add_summary": add_summary,
                "memory_add_output_summary": memory_add_output_summary,
                "memory_store_summary": memory_add_output_summary,
                "final_memory_store_summary": final_memory_store_summary,
                "wrapper_summary": wrapper_summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved metrics CSV:", csv_path)
    print("Saved summary JSON:", summary_path)

    print("\n=== OVERALL SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n=== CATEGORY SUMMARY ===")
    print(json.dumps(category_summary, ensure_ascii=False, indent=2))

    print("\n=== ADD MEMORY SUMMARY ===")
    print(json.dumps(add_summary, ensure_ascii=False, indent=2))

    print("\n=== MEMORY ADD OUTPUT SUMMARY ===")
    print(json.dumps(memory_add_output_summary, ensure_ascii=False, indent=2))

    print("\n=== FINAL MEMORY STORE SUMMARY ===")
    print(json.dumps(final_memory_store_summary, ensure_ascii=False, indent=2))

    print("\n=== WRAPPER SUMMARY ===")
    print(json.dumps(wrapper_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
