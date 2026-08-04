from __future__ import annotations

import argparse
import csv
import copy
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


# =========================
# Edit These Defaults First
# =========================
DEFAULT_DATASETS = [
    "assist2009",
    "assistments15",
    "assistments17",
    "assist2012",
    "statics2011",
    "ednet-kt1",
]

DEFAULT_MODELS = ["DKT", "SAKT", "AKT", "DKVMN"]

DEFAULT_STRATEGIES = ["class1", "class2", "random20"]

CORE_METHODS = [
    "QEFU-KT",
    "GoldModel",
    "FisherForgetting",
    "SelectiveSynapticDampening",
    "Finetuning",
    "NegGrad",
    "AdvancedNegGrad",
]

OPTIONAL_METHODS = [
    "Scrub",
    "BadTeaching",
]

INCLUDE_OPTIONAL_METHODS_BY_DEFAULT = False

DEFAULT_OUTPUT_ROOT = FRAMEWORK_ROOT / "output" / "runs" / "kt" / "qefu_full_batch_suite"
DEFAULT_BENCHMARK_NAME = "six_datasets_four_models_core_methods"

DATASET_PRESETS = {
    "assist2009": {"epochs": 4, "batch_size": 64},
    "assistments15": {"epochs": 3, "batch_size": 128},
    "assistments17": {"epochs": 5, "batch_size": 64},
    "assist2012": {"epochs": 3, "batch_size": 32},
    "statics2011": {"epochs": 4, "batch_size": 16},
    "ednet-kt1": {"epochs": 2, "batch_size": 16},
}


DATASET_LABELS = {
    "assist2009": "ASSIST2009",
    "assistments15": "ASSIST2015",
    "assistments17": "ASSIST2017",
    "assist2012": "ASSIST2012",
    "statics2011": "STATICS2011",
    "ednet-kt1": "EdNet-KT1",
}

DATASET_SPECS = {
    "assist2009": {
        "input_mode": "direct",
        "path": FRAMEWORK_ROOT / "output" / "runs" / "kt" / "pyedmine_converted" / "assist2009_sequences.csv",
    },
    "assistments15": {
        "input_mode": "direct",
        "path": FRAMEWORK_ROOT / "output" / "runs" / "kt" / "pyedmine_converted" / "assistments15_sequences.csv",
    },
    "assistments17": {
        "input_mode": "direct",
        "path": FRAMEWORK_ROOT / "output" / "runs" / "kt" / "pyedmine_converted" / "assistments17_sequences.csv",
    },
    "assist2012": {
        "input_mode": "direct",
        "path": FRAMEWORK_ROOT / "output" / "runs" / "kt" / "pyedmine_converted" / "assist2012_sequences.csv",
    },
    "statics2011": {
        "input_mode": "direct",
        "path": FRAMEWORK_ROOT / "output" / "runs" / "kt" / "pyedmine_converted" / "statics2011_sequences.csv",
    },
    "ednet-kt1": {
        "input_mode": "direct",
        "path": FRAMEWORK_ROOT / "output" / "runs" / "kt" / "pyedmine_converted" / "ednet-kt1_sequences.csv",
    },
}

STRATEGY_SPECS = {
    "class1": {"backend": "high_performance", "percent": 0.2},
    "class2": {"backend": "low_performance", "percent": 0.2},
    "random20": {"backend": "random_percent", "percent": 0.2},
}

MODEL_PRESETS = {
    "DKT": {"emb_size": 64, "batch_size_cap": 128},
    "SAKT": {"emb_size": 64, "batch_size_cap": 64},
    "AKT": {"emb_size": 64, "batch_size_cap": 64},
    "DKVMN": {"emb_size": 64, "batch_size_cap": 64},
}

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
    parser = argparse.ArgumentParser(
        description="Run the six-dataset / four-model QEFU-KT comparison suite with editable defaults."
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=list(DATASET_SPECS.keys()))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES, choices=list(STRATEGY_SPECS.keys()))
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--include-optional-methods", action="store_true", default=INCLUDE_OPTIONAL_METHODS_BY_DEFAULT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--benchmark-name", default=DEFAULT_BENCHMARK_NAME)
    parser.add_argument("--full-eval", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def resolve_methods(args: argparse.Namespace) -> list[str]:
    if args.methods:
        return list(args.methods)
    methods = list(CORE_METHODS)
    if args.include_optional_methods:
        methods.extend(OPTIONAL_METHODS)
    return methods


def prepared_input(dataset_name: str, output_root: Path) -> Path:
    spec = DATASET_SPECS[dataset_name]
    if spec["input_mode"] == "build_fold":
        return run_batch.build_fold_csv(spec["dataset_key"], output_root)
    input_path = Path(spec["path"])
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found for {dataset_name}: {input_path}")
    return input_path


def combo_params(dataset_name: str, model_name: str) -> dict[str, Any]:
    dataset_preset = DATASET_PRESETS.get(dataset_name, {"epochs": 4, "batch_size": 64})
    model_preset = MODEL_PRESETS.get(model_name, {"emb_size": 64, "batch_size_cap": 64})
    return {
        "epochs": int(dataset_preset["epochs"]),
        "batch_size": min(int(dataset_preset["batch_size"]), int(model_preset["batch_size_cap"])),
        "emb_size": int(model_preset["emb_size"]),
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


def safe_dir_name(name: str) -> str:
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def materialize_method_views(payload: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = Path(payload["run_dir"])
    rows = payload.get("summary_rows", [])
    split_summary_path = run_dir / "split_artifacts" / "split_summary.json"
    split_summary = {}
    if split_summary_path.exists():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    method_views: list[dict[str, Any]] = []
    for rec in rows:
        method_name = str(rec.get("unlearner", "")).strip()
        if not method_name:
            continue
        method_dir = run_dir.parent / safe_dir_name(method_name)
        method_dir.mkdir(parents=True, exist_ok=True)

        method_csv = method_dir / "results.csv"
        with method_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rec.keys()))
            writer.writeheader()
            writer.writerow(rec)

        result_json = {
            "dataset": payload["dataset_name"],
            "dataset_label": DATASET_LABELS[payload["dataset_name"]],
            "model": payload["model_name"],
            "strategy": payload["strategy_name"],
            "strategy_backend": STRATEGY_SPECS[payload["strategy_name"]]["backend"],
            "method": method_name,
            "params": copy.deepcopy(payload["params"]),
            "input_csv": payload["input_csv"],
            "bundle_dir": str(run_dir),
            "bundle_results_csv": payload["results_csv"],
            "split_summary": split_summary,
            "metrics": rec,
        }
        (method_dir / "result.json").write_text(
            json.dumps(result_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        method_views.append(
            {
                "method": method_name,
                "method_dir": str(method_dir),
                "method_results_csv": str(method_csv),
            }
        )
    return method_views


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
    strategy_root = output_root / dataset_name / model_name / strategy_name
    run_dir = strategy_root / "_bundle"
    if overwrite and run_dir.exists():
        shutil.rmtree(strategy_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"
    log_path = run_dir / "run.log"
    if not overwrite and results_csv.exists():
        existing_rows = read_results_table(results_csv)
        existing_methods = {row.get("unlearner") for row in existing_rows}
        if set(methods).issubset(existing_methods):
            payload = {
                "return_code": 0,
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
            payload["method_views"] = materialize_method_views(payload)
            return payload

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

    payload = {
        "return_code": proc.returncode,
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
    if proc.returncode == 0:
        payload["method_views"] = materialize_method_views(payload)
    else:
        payload["method_views"] = []
    return payload


def flatten_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results_csv = Path(payload["results_csv"])
    split_summary_path = Path(payload["run_dir"]) / "split_artifacts" / "split_summary.json"
    split_summary = {}
    if split_summary_path.exists():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))

    method_view_map = {view["method"]: view for view in payload.get("method_views", [])}
    for rec in payload.get("summary_rows", []):
        method_name = rec.get("unlearner")
        method_view = method_view_map.get(method_name, {})
        row: dict[str, Any] = {
            "dataset": payload["dataset_name"],
            "dataset_label": DATASET_LABELS[payload["dataset_name"]],
            "model": payload["model_name"],
            "strategy": payload["strategy_name"],
            "strategy_backend": STRATEGY_SPECS[payload["strategy_name"]]["backend"],
            "method": method_name,
            "return_code": payload["return_code"],
            "results_csv": method_view.get("method_results_csv", str(results_csv)),
            "run_dir": method_view.get("method_dir", payload["run_dir"]),
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


def _to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


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
        "# QEFU-KT Full Batch Suite Summary",
        "",
        "Columns use `original->unlearned` for ACC/AUC, and plain values for unlearning metrics.",
        "",
    ]
    if not rows:
        lines.append("No completed runs.")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    df = pd.DataFrame(rows).sort_values(["dataset", "model", "strategy", "method"])
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
    methods = resolve_methods(args)
    output_root = Path(args.output_root) / args.benchmark_name
    output_root.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "datasets": args.datasets,
        "models": args.models,
        "strategies": args.strategies,
        "methods": methods,
        "seed": args.seed,
        "full_eval": bool(args.full_eval),
        "include_optional_methods": bool(args.include_optional_methods),
    }
    (output_root / "suite_config.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prepared_inputs: dict[str, Path] = {}
    for dataset_name in args.datasets:
        prepared_inputs[dataset_name] = prepared_input(dataset_name, output_root)

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
                    methods=list(methods),
                    seed=args.seed,
                    output_root=output_root,
                    full_eval=bool(args.full_eval),
                    overwrite=bool(args.overwrite),
                )
                payloads.append(payload)
                rows.extend(flatten_payload(payload))
                status = "reused" if payload.get("reused") else "done"
                print(
                    f"[{status}] dataset={dataset_name} model={model_name} "
                    f"strategy={strategy_name} code={payload['return_code']} "
                    f"rows={len(payload.get('summary_rows', []))}"
                )
                if payload["return_code"] != 0 and args.stop_on_error:
                    raise SystemExit(payload["return_code"])

    summary_csv = output_root / "full_suite_summary.csv"
    summary_md = output_root / "full_suite_summary.md"
    payloads_json = output_root / "full_suite_payloads.json"
    write_csv(rows, summary_csv)
    write_markdown(rows, summary_md)
    payloads_json.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "summary_csv": str(summary_csv),
                "summary_md": str(summary_md),
                "payloads_json": str(payloads_json),
                "output_root": str(output_root),
                "row_count": len(rows),
                "combo_count": len(payloads),
                "methods": methods,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
