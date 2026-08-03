from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_kt_paper_suite import (
    DATASET_LABELS,
    STRATEGY_SPECS,
    SUMMARY_COLUMNS,
    dedupe_method_rows,
    read_results_table,
    write_csv,
    write_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge KT paper suite shard summaries.")
    parser.add_argument("--root", required=True, help="Root directory containing shard_* subdirectories.")
    parser.add_argument("--output-name", default="merged_full_suite")
    parser.add_argument("--datasets", nargs="*", default=["assist2009", "assistments15", "assistments17"])
    return parser.parse_args()


def _load_rows_from_results(results_csv: Path) -> list[dict]:
    run_dir = results_csv.parent
    try:
        dataset_name, model_name, strategy_name = run_dir.name.split("__")
    except ValueError:
        return []

    split_summary = {}
    split_summary_path = run_dir / "split_artifacts" / "split_summary.json"
    if split_summary_path.exists():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    result_rows = dedupe_method_rows(read_results_table(results_csv))
    flat_rows: list[dict] = []
    for rec in result_rows:
        row = {
            "dataset": dataset_name,
            "dataset_label": DATASET_LABELS.get(dataset_name, dataset_name),
            "model": model_name.upper(),
            "strategy": strategy_name,
            "strategy_backend": STRATEGY_SPECS.get(strategy_name, {}).get("backend", ""),
            "method": rec.get("unlearner"),
            "return_code": 0,
            "results_csv": str(results_csv),
            "run_dir": str(run_dir),
            "input_csv": "",
            "epochs": "",
            "batch_size": "",
            "emb_size": "",
            "dropout": "",
            "forget_user_count": split_summary.get("forget_user_count"),
            "forget_mean_correctness": split_summary.get("forget_mean_correctness"),
            "retain_mean_correctness": split_summary.get("other_mean_correctness"),
            "test_mean_correctness": split_summary.get("test_mean_correctness"),
        }
        for key in SUMMARY_COLUMNS:
            if key not in row:
                row[key] = rec.get(key, "")
        flat_rows.append(row)
    return flat_rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    shard_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard_")])
    rows: list[dict] = []
    for shard_dir in shard_dirs:
        for results_csv in sorted(shard_dir.glob("**/results.csv")):
            dataset_name = results_csv.parent.name.split("__")[0]
            if args.datasets and dataset_name not in set(args.datasets):
                continue
            rows.extend(_load_rows_from_results(results_csv))

    output_dir = root / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "paper_suite_summary.csv"
    summary_md = output_dir / "paper_suite_summary.md"
    write_csv(rows, summary_csv)
    write_markdown(rows, summary_md)
    print(
        {
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "row_count": len(rows),
            "expected_columns": len(SUMMARY_COLUMNS),
        }
    )


if __name__ == "__main__":
    main()
