import argparse
import csv
import json
import re
from pathlib import Path


QUESTION_TYPE_BY_CATEGORY = {
    1: "single_hop",
    2: "temporal",
    3: "multi_hop",
    4: "fallback_ambiguous",
}


def _contains_name(question: str, name: str) -> bool:
    if not name:
        return False
    pattern = r"(?<![A-Za-z])" + re.escape(name.lower()) + r"(?:'s)?(?![A-Za-z])"
    return re.search(pattern, question.lower()) is not None


def _target_speaker(question: str, speaker_a: str, speaker_b: str) -> str:
    has_a = _contains_name(question, speaker_a)
    has_b = _contains_name(question, speaker_b)
    if has_a and has_b:
        return "both"
    if has_a:
        return "speaker_a"
    if has_b:
        return "speaker_b"
    return "unknown"


def _records(samples, split_name: str):
    rows = []
    for sample_index, sample in enumerate(samples):
        conv = sample.get("conversation", {})
        speaker_a = conv.get("speaker_a", "")
        speaker_b = conv.get("speaker_b", "")
        sample_id = sample.get("sample_id", f"sample_{sample_index}")
        for qa_index, qa in enumerate(sample.get("qa", []), start=1):
            category = qa.get("category")
            if category not in QUESTION_TYPE_BY_CATEGORY:
                continue
            question = str(qa.get("question", "")).strip()
            rows.append(
                {
                    "id": f"{sample_id}_qa_{qa_index}",
                    "split": split_name,
                    "sample_id": sample_id,
                    "qa_index": qa_index,
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                    "question": question,
                    "text": f"Speaker A: {speaker_a}\nSpeaker B: {speaker_b}\nQuestion: {question}",
                    "locomo_category": category,
                    "question_type": QUESTION_TYPE_BY_CATEGORY[category],
                    "target_speaker": _target_speaker(question, speaker_a, speaker_b),
                }
            )
    return rows


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "dataset" / "locomo10.json"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "dataset" / "qase_planner_finetune"),
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    train_rows = _records(samples[5:], "train")
    test_rows = _records(samples[:5], "test")
    all_rows = train_rows + test_rows

    _write_jsonl(output_dir / "qase_planner_train_last5.jsonl", train_rows)
    _write_jsonl(output_dir / "qase_planner_test_first5.jsonl", test_rows)
    _write_jsonl(output_dir / "qase_planner_all_10conv_no_cat5.jsonl", all_rows)
    _write_csv(output_dir / "qase_planner_train_last5.csv", train_rows)
    _write_csv(output_dir / "qase_planner_test_first5.csv", test_rows)

    label_map = {
        "question_type": sorted(set(QUESTION_TYPE_BY_CATEGORY.values())),
        "target_speaker": ["speaker_a", "speaker_b", "both", "unknown"],
        "category_mapping": QUESTION_TYPE_BY_CATEGORY,
        "split": {
            "train": [sample.get("sample_id") for sample in samples[5:]],
            "test": [sample.get("sample_id") for sample in samples[:5]],
        },
        "note": "Category 5 adversarial questions are excluded. Target speaker labels are weak labels derived from speaker-name matches in the question.",
    }
    (output_dir / "label_maps.json").write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {len(train_rows)} train rows and {len(test_rows)} test rows to {output_dir}")


if __name__ == "__main__":
    main()
