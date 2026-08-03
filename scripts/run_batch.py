from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
FRAMEWORK_ROOT = APP_ROOT.parent
SRC_ROOT = FRAMEWORK_ROOT / "src"
ERASURE_ROOT = FRAMEWORK_ROOT.parent / "ERASURE-main"
for import_root in (SRC_ROOT, ERASURE_ROOT):
    if import_root.exists() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

PYEDMINE_DATA_ROOT = FRAMEWORK_ROOT / "data" / "processed_datasets"
DEFAULT_OUTPUT_ROOT = FRAMEWORK_ROOT / "output" / "runs" / "kt" / "benchmark"

DATASET_SOURCES = {
    "assistments12": {
        "train": PYEDMINE_DATA_ROOT / "assistments12" / "preprocessed_data_train.csv",
        "test": PYEDMINE_DATA_ROOT / "assistments12" / "preprocessed_data_test.csv",
    },
    "assistments15": {
        "train": PYEDMINE_DATA_ROOT / "assistments15" / "preprocessed_data_train.csv",
        "test": PYEDMINE_DATA_ROOT / "assistments15" / "preprocessed_data_test.csv",
    },
    "assistments17": {
        "train": PYEDMINE_DATA_ROOT / "assistments17" / "preprocessed_data_train.csv",
        "test": PYEDMINE_DATA_ROOT / "assistments17" / "preprocessed_data_test.csv",
    },
    "algebra05": {
        "train": PYEDMINE_DATA_ROOT / "algebra05" / "preprocessed_data_train.csv",
        "test": PYEDMINE_DATA_ROOT / "algebra05" / "preprocessed_data_test.csv",
    },
    "bridge_algebra06": {
        "train": PYEDMINE_DATA_ROOT / "bridge_algebra06" / "preprocessed_data_train.csv",
        "test": PYEDMINE_DATA_ROOT / "bridge_algebra06" / "preprocessed_data_test.csv",
    },
    "spanish": {
        "train": PYEDMINE_DATA_ROOT / "spanish" / "preprocessed_data_train.csv",
        "test": PYEDMINE_DATA_ROOT / "spanish" / "preprocessed_data_test.csv",
    },
    "statics": {
        "train": PYEDMINE_DATA_ROOT / "statics" / "preprocessed_data_train.csv",
        "test": PYEDMINE_DATA_ROOT / "statics" / "preprocessed_data_test.csv",
    },
    "assist2009_raw": {
        "full": PYEDMINE_DATA_ROOT / "ASSIST2009" / "skill_builder_data.csv",
    },
}

SUMMARY_COLUMNS = [
    "dataset",
    "model",
    "method",
    "profile",
    "epochs",
    "batch_size",
    "emb_size",
    "dropout",
    "return_code",
    "input_format",
    "user_count",
    "sequence_count",
    "question_count",
    "forget_user_count",
    "forget_mean_correctness",
    "AUS",
    "RelearnTime",
    "AIN",
    "UMIA",
    "UMIA_AUC",
    "kt.loss.test.original",
    "kt.acc.test.original",
    "kt.auc.test.original",
    "kt.loss.test.unlearned",
    "kt.acc.test.unlearned",
    "kt.auc.test.unlearned",
    "kt.loss.forget.original",
    "kt.acc.forget.original",
    "kt.auc.forget.original",
    "kt.loss.forget.unlearned",
    "kt.acc.forget.unlearned",
    "kt.auc.forget.unlearned",
    "kt.loss.retain.original",
    "kt.acc.retain.original",
    "kt.auc.retain.original",
    "kt.loss.retain.unlearned",
    "kt.acc.retain.unlearned",
    "kt.auc.retain.unlearned",
    "RunTime",
    "results_csv",
    "run_dir",
]

FORMAL_DATASET_PRESETS = {
    "assistments12": {"epochs": 3, "batch_size": 128},
    "assistments15": {"epochs": 3, "batch_size": 128},
    "assistments17": {"epochs": 5, "batch_size": 64},
    "algebra05": {"epochs": 5, "batch_size": 64},
    "bridge_algebra06": {"epochs": 5, "batch_size": 64},
    "spanish": {"epochs": 5, "batch_size": 64},
    "statics": {"epochs": 6, "batch_size": 32},
    "assist2009_raw": {"epochs": 4, "batch_size": 64},
}

FORMAL_MODEL_PRESETS = {
    "DKT": {"emb_size": 64, "batch_size_cap": 128},
    "AKT": {"emb_size": 64, "batch_size_cap": 64},
    "SimpleKT": {"emb_size": 64, "batch_size_cap": 64},
    "DTransformer": {"emb_size": 64, "batch_size_cap": 32},
    "DKT+": {"emb_size": 64, "batch_size_cap": 64},
    "SAKT": {"emb_size": 64, "batch_size_cap": 64},
    "DKVMN": {"emb_size": 64, "batch_size_cap": 64},
}

QUICK_MODEL_PRESETS = {
    "DKT": {"emb_size": 32, "batch_size_cap": 64},
    "AKT": {"emb_size": 32, "batch_size_cap": 32},
    "SimpleKT": {"emb_size": 32, "batch_size_cap": 32},
    "DTransformer": {"emb_size": 32, "batch_size_cap": 32},
    "DKT+": {"emb_size": 32, "batch_size_cap": 32},
    "SAKT": {"emb_size": 32, "batch_size_cap": 32},
    "DKVMN": {"emb_size": 32, "batch_size_cap": 32},
}


def cli_quote(value: Any) -> str:
    text = str(value)
    if not text:
        return '""'
    if any(ch.isspace() for ch in text) or '"' in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def build_rerun_command() -> str:
    parts = [sys.executable, APP_ROOT / "run_batch.py", *sys.argv[1:]]
    return " ".join(cli_quote(part) for part in parts)


def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


def normalize_row_table(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    if "user_id" not in df.columns and "uid" in df.columns:
        rename_map["uid"] = "user_id"
    if "item_id" not in df.columns:
        if "problem_id" in df.columns:
            rename_map["problem_id"] = "item_id"
        elif "question_id" in df.columns:
            rename_map["question_id"] = "item_id"
    if "correct" not in df.columns:
        for candidate in ["response", "is_correct"]:
            if candidate in df.columns:
                rename_map[candidate] = "correct"
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    required = ["user_id", "item_id", "correct"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in row table")
    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))
    if "skill_id" not in df.columns:
        df["skill_id"] = -1
    return df


def build_fold_csv(dataset_name: str, run_root: Path) -> Path:
    spec = DATASET_SOURCES[dataset_name]
    input_dir = run_root / "prepared_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    merged_path = input_dir / f"{dataset_name}_fold.csv"

    if "full" in spec:
        df = normalize_row_table(read_table(spec["full"]))
        df["fold"] = 0
        df.to_csv(merged_path, index=False)
        return merged_path

    train_df = normalize_row_table(read_table(spec["train"]))
    test_df = normalize_row_table(read_table(spec["test"]))
    train_df["fold"] = 0
    test_df["fold"] = 1
    merged = pd.concat([train_df, test_df], ignore_index=True)
    merged.to_csv(merged_path, index=False)
    return merged_path


def experiment_params(
    dataset_name: str,
    model_name: str,
    method_name: str,
    profile: str,
    epochs_override: int | None,
    batch_size_override: int | None,
    emb_size_override: int | None,
    dropout_override: float | None,
) -> dict[str, Any]:
    if profile == "formal":
        dataset_preset = FORMAL_DATASET_PRESETS.get(dataset_name, {"epochs": 4, "batch_size": 64})
        model_preset = FORMAL_MODEL_PRESETS.get(model_name, {"emb_size": 64, "batch_size_cap": 64})
    else:
        dataset_preset = {"epochs": 1, "batch_size": 32}
        model_preset = QUICK_MODEL_PRESETS.get(model_name, {"emb_size": 32, "batch_size_cap": 32})

    epochs = epochs_override if epochs_override is not None else int(dataset_preset["epochs"])
    batch_size = batch_size_override if batch_size_override is not None else min(
        int(dataset_preset["batch_size"]), int(model_preset["batch_size_cap"])
    )
    emb_size = emb_size_override if emb_size_override is not None else int(model_preset["emb_size"])
    dropout = dropout_override if dropout_override is not None else 0.1

    if method_name == "FisherForgetting":
        batch_size = min(batch_size, 64)
    if method_name == "SelectiveSynapticDampening":
        batch_size = min(batch_size, 64)

    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "emb_size": int(emb_size),
        "dropout": float(dropout),
    }


def run_backend(
    dataset_name: str,
    dataset_csv: Path,
    model_name: str,
    method_name: str,
    output_root: Path,
    strategy: str,
    seed: int,
    params: dict[str, Any],
    percent: float,
    full_eval: bool,
    profile: str,
) -> dict[str, Any]:
    run_name = f"{dataset_name}__{model_name.lower()}__{method_name.lower()}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(APP_ROOT / "kt_backend.py"),
        "--input-csv",
        str(dataset_csv),
        "--run-dir",
        str(run_dir),
        "--strategy",
        strategy,
        "--model-name",
        model_name,
        "--methods",
        method_name,
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
        str(percent),
    ]
    if full_eval:
        cmd.append("--full-eval")

    proc = subprocess.run(cmd, cwd=str(FRAMEWORK_ROOT), capture_output=True, text=True)
    stdout = proc.stdout.strip()
    payload = None
    if stdout:
        lines = [line for line in stdout.splitlines() if line.strip()]
        for idx in range(len(lines)):
            candidate = "\n".join(lines[idx:])
            try:
                payload = json.loads(candidate)
                break
            except Exception:
                continue
    if payload is None:
        payload = {
            "return_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "summary_rows": [],
            "dataset_meta": {},
            "results_csv": str(run_dir / "results.csv"),
        }
    payload["return_code"] = proc.returncode
    payload["run_dir"] = str(run_dir)
    payload["dataset_name"] = dataset_name
    payload["model_name"] = model_name
    payload["method_name"] = method_name
    payload["profile"] = profile
    payload["params"] = params
    return payload


def summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "dataset": payload.get("dataset_name"),
        "model": payload.get("model_name"),
        "method": payload.get("method_name"),
        "profile": payload.get("profile"),
        "return_code": payload.get("return_code"),
        "results_csv": payload.get("results_csv"),
        "run_dir": payload.get("run_dir"),
    }
    params = payload.get("params", {})
    row["epochs"] = params.get("epochs")
    row["batch_size"] = params.get("batch_size")
    row["emb_size"] = params.get("emb_size")
    row["dropout"] = params.get("dropout")
    row.update(payload.get("dataset_meta", {}))
    summary_rows = payload.get("summary_rows", [])
    if summary_rows:
        metrics = summary_rows[0]
        row["forget_user_count"] = metrics.get("forget_user_count")
        row["forget_mean_correctness"] = metrics.get("forget_mean_correctness")
        for key in SUMMARY_COLUMNS:
            if key in metrics:
                row[key] = metrics.get(key)
    return row


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(SUMMARY_COLUMNS)
    extras = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames and key not in extras:
                extras.append(key)
    fieldnames.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_benchmark_manifest(
    *,
    output_root: Path,
    args: argparse.Namespace,
    prepared_inputs: dict[str, Path],
    payloads: list[dict[str, Any]],
    summary_path: Path,
    payloads_path: Path,
) -> Path:
    manifest_path = output_root / "benchmark_manifest.json"
    runs = []
    for payload in payloads:
        runs.append(
            {
                "dataset": payload.get("dataset_name"),
                "model": payload.get("model_name"),
                "method": payload.get("method_name"),
                "profile": payload.get("profile"),
                "return_code": payload.get("return_code"),
                "config_path": payload.get("config_path"),
                "results_csv": payload.get("results_csv"),
                "run_dir": payload.get("run_dir"),
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "framework_root": str(FRAMEWORK_ROOT),
        "benchmark_name": args.benchmark_name or output_root.name,
        "entrypoints": {
            "batch": str(APP_ROOT / "run_batch.py"),
            "single": str(FRAMEWORK_ROOT / "main.py"),
        },
        "selection": {
            "datasets": list(args.datasets),
            "models": list(args.models),
            "methods": list(args.methods),
            "strategy": args.strategy,
            "profile": args.profile,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "emb_size": args.emb_size,
            "dropout": args.dropout,
            "percent": args.percent,
            "full_eval": args.full_eval,
        },
        "prepared_inputs": {name: str(path) for name, path in prepared_inputs.items()},
        "artifacts": {
            "summary_csv": str(summary_path),
            "payloads_json": str(payloads_path),
        },
        "rerun_command": build_rerun_command(),
        "runs": runs,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def write_benchmark_readme(
    *,
    output_root: Path,
    args: argparse.Namespace,
    manifest_path: Path,
    summary_path: Path,
    payloads_path: Path,
) -> Path:
    readme_path = output_root / "README.md"
    lines = [
        "# Batch Benchmark Output",
        "",
        "This directory contains a reproducible KT batch benchmark run.",
        "",
        "## Selection",
        "",
        f"- datasets: {', '.join(args.datasets)}",
        f"- models: {', '.join(args.models)}",
        f"- methods: {', '.join(args.methods)}",
        f"- strategy: {args.strategy}",
        f"- profile: {args.profile}",
        f"- seed: {args.seed}",
        f"- full_eval: {args.full_eval}",
        "",
        "## Key Artifacts",
        "",
        f"- summary: `{summary_path.name}`",
        f"- payloads: `{payloads_path.name}`",
        f"- manifest: `{manifest_path.name}`",
        "- per-run configs: `<run_dir>/config.jsonc`",
        "- per-run metrics: `<run_dir>/results.csv`",
        "",
        "## Re-run",
        "",
        "```bash",
        build_rerun_command(),
        "```",
        "",
        "## Single Config Entry",
        "",
        "```bash",
        f"{cli_quote(sys.executable)} {cli_quote(FRAMEWORK_ROOT / 'main.py')} <config.jsonc>",
        "```",
        "",
        "This README is generated by `scripts/run_batch.py`.",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible KT benchmark batches.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["assistments12", "assistments17", "algebra05", "statics"],
        choices=sorted(DATASET_SOURCES.keys()),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DKT", "AKT", "SimpleKT"],
        choices=["DKT", "DKT+", "SAKT", "DKVMN", "AKT", "SimpleKT", "DTransformer"],
    )
    parser.add_argument("--methods", nargs="+", default=["GoldModel"])
    parser.add_argument("--strategy", default="high_performance", choices=["high_performance", "low_performance"])
    parser.add_argument("--profile", default="formal", choices=["quick", "formal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--emb-size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--percent", type=float, default=0.2)
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--benchmark-name", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_output_root = Path(args.output_root)
    output_root = base_output_root / args.benchmark_name if args.benchmark_name else base_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    prepared_inputs: dict[str, Path] = {}
    for dataset_name in args.datasets:
        prepared_inputs[dataset_name] = build_fold_csv(dataset_name, output_root)

    all_payloads = []
    summary_rows = []
    for dataset_name in args.datasets:
        dataset_csv = prepared_inputs[dataset_name]
        for model_name in args.models:
            for method_name in args.methods:
                payload = run_backend(
                    dataset_name=dataset_name,
                    dataset_csv=dataset_csv,
                    model_name=model_name,
                    method_name=method_name,
                    output_root=output_root,
                    strategy=args.strategy,
                    seed=args.seed,
                    params=experiment_params(
                        dataset_name=dataset_name,
                        model_name=model_name,
                        method_name=method_name,
                        profile=args.profile,
                        epochs_override=args.epochs,
                        batch_size_override=args.batch_size,
                        emb_size_override=args.emb_size,
                        dropout_override=args.dropout,
                    ),
                    percent=args.percent,
                    full_eval=args.full_eval,
                    profile=args.profile,
                )
                all_payloads.append(payload)
                summary_rows.append(summarize_payload(payload))
                print(
                    f"[done] dataset={dataset_name} model={model_name} method={method_name} "
                    f"profile={args.profile} epochs={payload['params']['epochs']} "
                    f"batch={payload['params']['batch_size']} emb={payload['params']['emb_size']} "
                    f"code={payload.get('return_code')}"
                )

    summary_path = output_root / "batch_summary.csv"
    write_summary(summary_rows, summary_path)
    payloads_path = output_root / "batch_payloads.json"
    payloads_path.write_text(json.dumps(all_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = write_benchmark_manifest(
        output_root=output_root,
        args=args,
        prepared_inputs=prepared_inputs,
        payloads=all_payloads,
        summary_path=summary_path,
        payloads_path=payloads_path,
    )
    readme_path = write_benchmark_readme(
        output_root=output_root,
        args=args,
        manifest_path=manifest_path,
        summary_path=summary_path,
        payloads_path=payloads_path,
    )
    print(f"[summary] {summary_path}")
    print(f"[payloads] {payloads_path}")
    print(f"[manifest] {manifest_path}")
    print(f"[readme] {readme_path}")


if __name__ == "__main__":
    main()
