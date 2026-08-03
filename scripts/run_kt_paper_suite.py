from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import run_batch


APP_ROOT = Path(__file__).resolve().parent
FRAMEWORK_ROOT = APP_ROOT.parent

DATASET_LABELS = {
    "assist2009": "ASSIST2009",
    "assistments12": "ASSIST2012",
    "assistments15": "ASSIST2015",
    "assistments17": "ASSIST2017",
}

DATASET_SOURCES = {
    "assist2009": FRAMEWORK_ROOT / "data" / "processed_datasets" / "ASSIST2009" / "skill_builder_data.csv",
    "assistments12": None,
    "assistments15": None,
    "assistments17": None,
}

STRATEGY_SPECS = {
    "class1": {"backend": "high_performance", "percent": 0.2},
    "class2": {"backend": "low_performance", "percent": 0.2},
    "random20": {"backend": "random_percent", "percent": 0.2},
}

METHODS = [
    "GoldModel",
    "Finetuning",
    "FisherForgetting",
    "SelectiveSynapticDampening",
    "QEFU-KT",
]

SUMMARY_COLUMNS = [
    "dataset",
    "dataset_label",
    "model",
    "strategy",
    "strategy_backend",
    "method",
    "return_code",
    "results_csv",
    "run_dir",
    "input_csv",
    "epochs",
    "batch_size",
    "emb_size",
    "dropout",
    "forget_user_count",
    "forget_mean_correctness",
    "retain_mean_correctness",
    "test_mean_correctness",
    "RunTime",
    "AUS",
    "AIN",
    "UMIA",
    "UMIA_AUC",
    "sklearn.metrics.accuracy_score.forget.original",
    "sklearn.metrics.accuracy_score.forget.unlearned",
    "sklearn.metrics.accuracy_score.retain.original",
    "sklearn.metrics.accuracy_score.retain.unlearned",
    "sklearn.metrics.accuracy_score.test.original",
    "sklearn.metrics.accuracy_score.test.unlearned",
    "kt.acc.forget.original",
    "kt.acc.forget.unlearned",
    "kt.auc.forget.original",
    "kt.auc.forget.unlearned",
    "kt.acc.retain.original",
    "kt.acc.retain.unlearned",
    "kt.auc.retain.original",
    "kt.auc.retain.unlearned",
    "kt.acc.test.original",
    "kt.acc.test.unlearned",
    "kt.auc.test.original",
    "kt.auc.test.unlearned",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the KT paper suite with baselines and QEFU-KT.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["assist2009", "assistments12", "assistments15", "assistments17"],
        choices=list(DATASET_LABELS.keys()),
    )
    parser.add_argument("--models", nargs="+", default=["DKT", "SAKT", "DKVMN"], choices=["DKT", "SAKT", "DKVMN"])
    parser.add_argument("--strategies", nargs="+", default=["class1", "class2", "random20"], choices=list(STRATEGY_SPECS.keys()))
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=str(FRAMEWORK_ROOT / "output" / "runs" / "kt" / "paper_suite"))
    parser.add_argument("--benchmark-name", default="assist2009_2012_2015_2017_dkt_sakt_dkvmn")
    parser.add_argument("--full-eval", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _full_eval_enabled(args: argparse.Namespace) -> bool:
    return bool(args.full_eval)


def _prepared_input(dataset_name: str, output_root: Path) -> Path:
    if dataset_name == "assist2009":
        return DATASET_SOURCES["assist2009"]
    return run_batch.build_fold_csv(dataset_name, output_root)


def combo_params(dataset_name: str, model_name: str) -> dict[str, Any]:
    dataset_preset = run_batch.FORMAL_DATASET_PRESETS.get(
        "assist2009_raw" if dataset_name == "assist2009" else dataset_name, {"epochs": 4, "batch_size": 64}
    )
    model_preset = run_batch.FORMAL_MODEL_PRESETS.get(model_name, {"emb_size": 64, "batch_size_cap": 64})
    epochs = int(dataset_preset["epochs"])
    batch_size = min(int(dataset_preset["batch_size"]), int(model_preset["batch_size_cap"]), 64)
    emb_size = int(model_preset["emb_size"])
    return {
        "epochs": epochs,
        "batch_size": batch_size,
        "emb_size": emb_size,
        "dropout": 0.1,
    }


def read_results_table(results_csv: Path) -> list[dict[str, Any]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dedupe_method_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        method = str(row.get("unlearner", ""))
        if not method:
            continue
        if method not in deduped:
            order.append(method)
        deduped[method] = row
    return [deduped[method] for method in order]


def run_combo(
    *,
    dataset_name: str,
    input_csv: Path,
    model_name: str,
    strategy_name: str,
    methods: list[str],
    seed: int,
    output_root: Path,
    full_eval: bool,
    overwrite: bool,
) -> dict[str, Any]:
    strategy = STRATEGY_SPECS[strategy_name]
    run_dir = output_root / f"{dataset_name}__{model_name.lower()}__{strategy_name}"
    if overwrite and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"
    log_path = run_dir / "run.log"
    if not overwrite and results_csv.exists():
        existing_rows = read_results_table(results_csv)
        existing_methods = {row.get("unlearner") for row in existing_rows}
        if set(methods).issubset(existing_methods):
            return {
                "return_code": 0,
                "stdout": "",
                "stderr": "",
                "results_csv": str(results_csv),
                "run_dir": str(run_dir),
                "dataset_name": dataset_name,
                "model_name": model_name,
                "strategy_name": strategy_name,
                "input_csv": str(input_csv),
                "params": combo_params(dataset_name, model_name),
                "summary_rows": dedupe_method_rows(existing_rows),
                "log_path": str(log_path),
                "reused": True,
            }

    params = combo_params(dataset_name, model_name)
    cmd = [
        sys.executable,
        str(APP_ROOT / "kt_backend.py"),
        "--input-csv",
        str(input_csv),
        "--run-dir",
        str(run_dir),
        "--strategy",
        strategy["backend"],
        "--model-name",
        model_name,
        "--methods",
        *methods,
        "--seed",
        str(seed),
        "--epochs",
        str(params["epochs"]),
        "--batch-size",
        str(params["batch_size"]),
        "--emb-size",
        str(params["emb_size"]),
        "--dropout",
        str(params["dropout"]),
        "--percent",
        str(strategy["percent"]),
    ]
    if full_eval:
        cmd.append("--full-eval")

    env = dict(**os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(FRAMEWORK_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    return {
        "return_code": proc.returncode,
        "stdout": "",
        "stderr": "",
        "results_csv": str(results_csv),
        "run_dir": str(run_dir),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "strategy_name": strategy_name,
        "input_csv": str(input_csv),
        "params": params,
        "summary_rows": dedupe_method_rows(read_results_table(results_csv)) if proc.returncode == 0 else [],
        "log_path": str(log_path),
        "reused": False,
    }


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def flatten_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results_csv = Path(payload["results_csv"])
    split_summary_path = Path(payload["run_dir"]) / "split_artifacts" / "split_summary.json"
    split_summary = {}
    if split_summary_path.exists():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    for rec in payload.get("summary_rows", []):
        row: dict[str, Any] = {
            "dataset": payload["dataset_name"],
            "dataset_label": DATASET_LABELS[payload["dataset_name"]],
            "model": payload["model_name"],
            "strategy": payload["strategy_name"],
            "strategy_backend": STRATEGY_SPECS[payload["strategy_name"]]["backend"],
            "method": rec.get("unlearner"),
            "return_code": payload["return_code"],
            "results_csv": str(results_csv),
            "run_dir": payload["run_dir"],
            "input_csv": payload["input_csv"],
            "epochs": payload["params"]["epochs"],
            "batch_size": payload["params"]["batch_size"],
            "emb_size": payload["params"]["emb_size"],
            "dropout": payload["params"]["dropout"],
            "forget_user_count": split_summary.get("forget_user_count"),
            "forget_mean_correctness": split_summary.get("forget_mean_correctness"),
            "retain_mean_correctness": split_summary.get("other_mean_correctness"),
            "test_mean_correctness": split_summary.get("test_mean_correctness"),
        }
        for key in SUMMARY_COLUMNS:
            if key in row:
                continue
            row[key] = rec.get(key)
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def _fmt_metric(orig: Any, unl: Any) -> str:
    o = _to_float(orig)
    u = _to_float(unl)
    if o is None and u is None:
        return "-"
    if o is None:
        return f"{u:.4f}"
    if u is None:
        return f"{o:.4f}"
    return f"{o:.4f}->{u:.4f}"


def write_markdown(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# KT Paper Suite Summary",
        "",
        "Columns use `original->unlearned` for acc/auc, and plain values for unlearning metrics.",
        "",
    ]
    if not rows:
        lines.append("No completed runs.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "model", "strategy", "method"])
    for (dataset_label, model, strategy), group in df.groupby(["dataset_label", "model", "strategy"], sort=False):
        lines.extend(
            [
                f"## {dataset_label} / {model} / {strategy}",
                "",
                "| Method | Forget acc | Forget auc | Retain acc | Retain auc | Test acc | Test auc | UMIA | UMIA_AUC | AIN | RunTime |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for _, row in group.iterrows():
            lines.append(
                "| {method} | {forget_acc} | {forget_auc} | {retain_acc} | {retain_auc} | {test_acc} | {test_auc} | {umia} | {umia_auc} | {ain} | {runtime} |".format(
                    method=row["method"],
                    forget_acc=_fmt_metric(row["kt.acc.forget.original"], row["kt.acc.forget.unlearned"]),
                    forget_auc=_fmt_metric(row["kt.auc.forget.original"], row["kt.auc.forget.unlearned"]),
                    retain_acc=_fmt_metric(row["kt.acc.retain.original"], row["kt.acc.retain.unlearned"]),
                    retain_auc=_fmt_metric(row["kt.auc.retain.original"], row["kt.auc.retain.unlearned"]),
                    test_acc=_fmt_metric(row["kt.acc.test.original"], row["kt.acc.test.unlearned"]),
                    test_auc=_fmt_metric(row["kt.auc.test.original"], row["kt.auc.test.unlearned"]),
                    umia=f"{_to_float(row['UMIA']):.4f}" if _to_float(row["UMIA"]) is not None else "-",
                    umia_auc=f"{_to_float(row['UMIA_AUC']):.4f}" if _to_float(row["UMIA_AUC"]) is not None else "-",
                    ain=f"{_to_float(row['AIN']):.4f}" if _to_float(row["AIN"]) is not None else "-",
                    runtime=f"{_to_float(row['RunTime']):.2f}s" if _to_float(row["RunTime"]) is not None else "-",
                )
            )
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root) / args.benchmark_name
    output_root.mkdir(parents=True, exist_ok=True)

    prepared_inputs: dict[str, Path] = {}
    for dataset_name in args.datasets:
        prepared_inputs[dataset_name] = _prepared_input(dataset_name, output_root)

    payloads: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        input_csv = prepared_inputs[dataset_name]
        for model_name in args.models:
            for strategy_name in args.strategies:
                payload = run_combo(
                    dataset_name=dataset_name,
                    input_csv=input_csv,
                    model_name=model_name,
                    strategy_name=strategy_name,
                    methods=list(args.methods),
                    seed=args.seed,
                    output_root=output_root,
                    full_eval=_full_eval_enabled(args),
                    overwrite=args.overwrite,
                )
                payloads.append(payload)
                rows.extend(flatten_payload(payload))
                status = "reused" if payload.get("reused") else "done"
                print(
                    f"[{status}] dataset={dataset_name} model={model_name} strategy={strategy_name} "
                    f"code={payload['return_code']} rows={len(payload.get('summary_rows', []))}"
                )
                if payload["return_code"] != 0 and args.stop_on_error:
                    raise SystemExit(payload["return_code"])

    summary_csv = output_root / "paper_suite_summary.csv"
    payloads_json = output_root / "paper_suite_payloads.json"
    summary_md = output_root / "paper_suite_summary.md"
    write_csv(rows, summary_csv)
    payloads_json.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(rows, summary_md)

    print(json.dumps(
        {
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "payloads_json": str(payloads_json),
            "output_root": str(output_root),
            "row_count": len(rows),
            "combo_count": len(payloads),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
