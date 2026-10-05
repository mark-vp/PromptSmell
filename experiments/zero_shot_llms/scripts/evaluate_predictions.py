"""Recompute four-class metrics from the released predictions and frozen labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verify_protocol() -> tuple[dict, dict[str, int]]:
    protocol = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    for relative, expected in protocol["sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen file hash mismatch: {relative}")
    for relative, expected in protocol["prediction_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Released prediction hash mismatch: {relative}")

    inputs = read_jsonl(ROOT / "data/test_matched_decoder.jsonl")
    gold_rows = read_jsonl(ROOT / "data/gold_test.jsonl")
    inputs_by_id = {row["sample_id"]: row for row in inputs}
    gold = {row["sample_id"]: row["label"] for row in gold_rows}
    expected_count = protocol["test_samples"]
    if (len(inputs) != expected_count or len(gold) != expected_count
            or len(inputs_by_id) != expected_count or set(inputs_by_id) != set(gold)):
        raise ValueError("Test input and gold IDs do not form the frozen test set")
    for sample_id, row in inputs_by_id.items():
        actual = hashlib.sha256(row["code"].encode("utf-8")).hexdigest()
        if row["code_sha256"] != actual or type(gold[sample_id]) is not int or gold[sample_id] not in range(4):
            raise ValueError(f"Invalid frozen sample: {sample_id}")
    if {str(k): v for k, v in sorted(Counter(gold.values()).items())} != protocol["label_counts"]:
        raise ValueError("Frozen class counts do not match")
    return protocol, gold


def read_predictions(path: Path, gold: dict[str, int]) -> dict[str, int]:
    predictions = {}
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in predictions or sample_id not in gold:
                raise ValueError(f"Duplicate or unknown sample ID: {sample_id}")
            if int(row["gold"]) != gold[sample_id]:
                raise ValueError(f"Released gold column mismatch: {sample_id}")
            valid = row["status"] == "valid"
            prediction = int(row["prediction"]) if valid else -1
            if valid and prediction not in range(4):
                raise ValueError(f"Invalid class: {sample_id}")
            predictions[sample_id] = prediction
    elif path.suffix == ".jsonl":
        for row in read_jsonl(path):
            sample_id = row["sample_id"]
            if sample_id not in gold:
                raise ValueError(f"Unknown sample ID: {sample_id}")
            prediction = row.get("prediction")
            if row.get("status") == "valid" and type(prediction) is int and prediction in range(4):
                predictions[sample_id] = prediction
            else:
                predictions[sample_id] = -1
    else:
        raise ValueError("Prediction file must be CSV or JSONL")
    if set(predictions) != set(gold):
        raise ValueError(f"Incomplete test set: {len(predictions)} of {len(gold)}")
    return predictions


def metrics(gold: dict[str, int], predictions: dict[str, int]) -> dict:
    n = len(gold)
    supports = Counter(gold.values())
    predicted = Counter(predictions.values())
    true_positives = Counter(
        truth for sample_id, truth in gold.items() if predictions[sample_id] == truth
    )
    recalls = {}
    weighted_precision = weighted_f1 = 0.0
    for label in range(4):
        support = supports[label]
        tp = true_positives[label]
        precision = tp / predicted[label] if predicted[label] else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls[str(label)] = recall
        weighted_precision += support * precision / n
        weighted_f1 += support * f1 / n
    accuracy = sum(true_positives.values()) / n
    return {
        "samples": n,
        "accuracy": accuracy,
        "precision_w": weighted_precision,
        "recall_w": accuracy,
        "f1_w": weighted_f1,
        "class_recall": recalls,
        "invalid_outputs": predicted[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-file", type=Path, action="append",
                        help="CSV or JSONL; default: all released CSV files")
    args = parser.parse_args()
    _, gold = verify_protocol()
    files = args.prediction_file or sorted((ROOT / "predictions").glob("*.csv"))
    for path in files:
        result = metrics(gold, read_predictions(path, gold))
        print(json.dumps({"model_file": path.name, **result}, indent=2))


if __name__ == "__main__":
    main()
