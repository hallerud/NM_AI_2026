#!/usr/bin/env python3
"""Compare YOLO training runs by parsing results.csv files."""
import csv
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"

def best_epoch(csv_path):
    """Return dict with best-epoch stats (by mAP50)."""
    best = None
    for row in csv.DictReader(open(csv_path)):
        row = {k.strip(): v.strip() for k, v in row.items()}
        try:
            mAP50 = float(row["metrics/mAP50(B)"])
        except (ValueError, KeyError):
            continue
        if best is None or mAP50 > best["mAP50"]:
            best = {
                "epoch": int(row["epoch"]),
                "mAP50": mAP50,
                "mAP50-95": float(row["metrics/mAP50-95(B)"]),
                "precision": float(row["metrics/precision(B)"]),
                "recall": float(row["metrics/recall(B)"]),
            }
    return best

def main():
    results = []
    for csv_path in sorted(RUNS_DIR.glob("*/results.csv")):
        name = csv_path.parent.name
        best = best_epoch(csv_path)
        if best:
            total_epochs = sum(1 for _ in open(csv_path)) - 1
            results.append((name, total_epochs, best))

    if not results:
        print("No results.csv files found in", RUNS_DIR)
        return

    # Sort by mAP50 descending
    results.sort(key=lambda x: x[2]["mAP50"], reverse=True)

    # Print table
    hdr = f"{'Run':<25} {'Epochs':>6} {'Best':>5} {'Prec':>6} {'Recall':>6} {'mAP50':>7} {'mAP50-95':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, total, b in results:
        print(f"{name:<25} {total:>6} {b['epoch']:>5} {b['precision']:>6.3f} {b['recall']:>6.3f} {b['mAP50']:>7.4f} {b['mAP50-95']:>8.4f}")

if __name__ == "__main__":
    main()
