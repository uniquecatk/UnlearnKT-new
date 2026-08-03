from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import os
import subprocess
import sys
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

METRIC_NAMES = [
    "unlearner",
    "dataset",
    "RunTime",
    "TorchFlops",
    "PeakRSS_MB",
    "CudaPeak_MB",
    "AUS",
    "RelearnTime",
    "RelearnInteractions",
    "AIN",
    "AIN_unlearned_interactions",
    "AIN_gold_interactions",
    "UMIA",
    "UMIA_AUC",
    "UMIA_threshold",
    "UMIA_threshold_AUC",
    "NoMUS",
    "sklearn.metrics.accuracy_score.test.original",
    "sklearn.metrics.accuracy_score.test.unlearned",
    "sklearn.metrics.accuracy_score.forget.original",
    "sklearn.metrics.accuracy_score.forget.unlearned",
    "sklearn.metrics.accuracy_score.retain.original",
    "sklearn.metrics.accuracy_score.retain.unlearned",
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
]

STANDARD_RESULT_COLUMNS = list(METRIC_NAMES)


def _set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_set_max_csv_field_size()


def _detect_sequence_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    uid_candidates = ["uid", "user_id"]
    q_candidates = ["questions", "question_seq", "item_seq"]
    r_candidates = ["responses", "correct_seq", "response_seq"]
    uid_col = next((c for c in uid_candidates if c in df.columns), None)
    q_col = next((c for c in q_candidates if c in df.columns), None)
    r_col = next((c for c in r_candidates if c in df.columns), None)
    return uid_col, q_col, r_col


def prepare_sequence_csv(input_csv: Path, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    try:
        df = pd.read_csv(input_csv, sep=None, engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, sep=None, engine="python", encoding="latin-1")
    uid_col, q_col, r_col = _detect_sequence_columns(df)

    if uid_col and q_col and r_col:
        out = pd.DataFrame()
        out["uid"] = df[uid_col].astype(str)
        out["questions"] = df[q_col]
        out["responses"] = df[r_col]
        if "concepts" in df.columns:
            out["concepts"] = df["concepts"]
        out["fold"] = df["fold"] if "fold" in df.columns else 0
        seq_path = run_dir / "input" / "kt_sequences.csv"
        seq_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(seq_path, index=False)
        return seq_path, {
            "input_format": "sequence",
            "user_count": int(out["uid"].nunique()),
            "sequence_count": int(len(out)),
            "has_fold": "fold" in df.columns,
        }

    user_candidates = ["user_id", "uid"]
    item_candidates = ["problem_id", "question_id", "item_id"]
    correct_candidates = ["correct", "response", "is_correct"]
    order_candidates = ["order_id", "timestamp", "event_time", "seq_idx"]
    concept_candidates = ["skill_id", "concept_id"]

    user_col = next((c for c in user_candidates if c in df.columns), None)
    item_col = next((c for c in item_candidates if c in df.columns), None)
    correct_col = next((c for c in correct_candidates if c in df.columns), None)
    order_col = next((c for c in order_candidates if c in df.columns), None)
    concept_col = next((c for c in concept_candidates if c in df.columns), None)
    if not (user_col and item_col and correct_col):
        raise ValueError(
            "CSV must be sequence format (uid/questions/responses) or row format (user_id/problem_id/question_id/item_id + correct)."
        )

    work = df.copy()
    work[user_col] = work[user_col].astype(str)
    work[correct_col] = pd.to_numeric(work[correct_col], errors="coerce")
    work = work.dropna(subset=[user_col, item_col, correct_col]).copy()
    work[correct_col] = work[correct_col].astype(int)
    if order_col is not None:
        work = work.sort_values([user_col, order_col])
    else:
        work = work.reset_index(drop=True)

    item_values = sorted({str(v) for v in work[item_col].tolist()})
    item_codes = {v: i for i, v in enumerate(item_values)}
    concept_codes = (
        {v: i for i, v in enumerate(sorted({str(v) for v in work[concept_col].fillna("-1").tolist()}))}
        if concept_col is not None
        else None
    )

    rows = []
    for uid, group in work.groupby(user_col, sort=False):
        q = [item_codes[str(x)] for x in group[item_col].tolist()]
        r = [int(x) for x in group[correct_col].tolist()]
        if len(q) < 2:
            continue
        row = {"uid": str(uid), "questions": ",".join(map(str, q)), "responses": ",".join(map(str, r)), "fold": 0}
        if concept_codes is not None:
            c = [concept_codes["-1" if pd.isna(x) else str(x)] for x in group[concept_col].tolist()]
            row["concepts"] = ",".join(map(str, c))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No valid KT sequences could be formed from uploaded CSV.")
    seq_path = run_dir / "input" / "kt_sequences.csv"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(seq_path, index=False)
    return seq_path, {
        "input_format": "row",
        "user_count": int(out["uid"].nunique()),
        "sequence_count": int(len(out)),
        "question_count": int(len(item_codes)),
    }


def unlearner_config(method_name: str) -> dict[str, Any]:
    mapping = {
        "GoldModel": {
            "class": "kt_unlearn.unlearners.GoldModel.GoldModel",
            "parameters": {"training_set": "retain", "cached": False},
        },
        "Finetuning": {
            "class": "kt_unlearn.unlearners.Finetuning.Finetuning",
            "parameters": {
                "epochs": 2,
                "ref_data": "retain",
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0005}},
            },
        },
        "NegGrad": {
            "class": "kt_unlearn.unlearners.NegGrad.NegGrad",
            "parameters": {
                "epochs": 1,
                "ref_data": "forget",
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0001}},
            },
        },
        "AdvancedNegGrad": {
            "class": "kt_unlearn.unlearners.AdvancedNegGrad.AdvancedNegGrad",
            "parameters": {
                "epochs": 1,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0001}},
            },
        },
        "SelectiveSynapticDampening": {
            "class": "kt_unlearn.unlearners.SelectiveSynapticDampening.SelectiveSynapticDampening",
            "parameters": {
                "ref_data_train": "train_source",
                "ref_data_forget": "forget",
                "dampening_constant": 0.1,
                "selection_weighting": 20,
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.001}},
            },
        },
        "FisherForgetting": {
            "class": "kt_unlearn.unlearners.FisherForgetting.FisherForgetting",
            "parameters": {"ref_data": "retain", "alpha": 1e-6, "ff_epochs": 64, "task": "kt"},
        },
        "Scrub": {
            "class": "kt_unlearn.unlearners.Scrub.Scrub",
            "parameters": {
                "epochs": 1,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "T": 2.0,
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0001}},
            },
        },
        "QEFU-KT": {
            "class": "kt_unlearn.unlearners.QEFUKT.QEFUKT",
            "parameters": {
                "ref_data": "retain",
                "ref_data_forget": "forget",
                "alpha": 1e-6,
                "ff_epochs": 64,
                "task": "kt",
                "target_strength": 1.5,
                "background_strength": 0.2,
                "exposure_power": 1.5,
                "retain_smoothing": 10.0,
                "min_question_weight": 0.05,
                "top_question_fraction": 0.1,
                "ascent_steps": 2,
                "ascent_lr": 0.02,
                "ascent_batches": 8,
                "max_param_shift": 0.5,
                "use_sign_updates": True,
            },
        },
        "KTFisherFocused": {
            "class": "kt_unlearn.unlearners.QEFUKT.QEFUKT",
            "parameters": {
                "ref_data": "retain",
                "ref_data_forget": "forget",
                "alpha": 1e-6,
                "ff_epochs": 64,
                "task": "kt",
                "target_strength": 8.0,
                "background_strength": 0.05,
                "exposure_power": 2.0,
                "retain_smoothing": 1.0,
                "min_question_weight": 0.0,
                "top_question_fraction": 0.2,
                "ascent_steps": 3,
                "ascent_lr": 0.05,
                "ascent_batches": 12,
                "max_param_shift": 1.0,
                "use_sign_updates": True,
            },
        },
        "KTFisherBalanced": {
            "class": "kt_unlearn.unlearners.QEFUKT.QEFUKT",
            "parameters": {
                "ref_data": "retain",
                "ref_data_forget": "forget",
                "alpha": 1e-6,
                "ff_epochs": 64,
                "task": "kt",
                "target_strength": 3.0,
                "background_strength": 0.1,
                "exposure_power": 1.8,
                "retain_smoothing": 2.0,
                "min_question_weight": 0.0,
                "top_question_fraction": 0.08,
                "ascent_steps": 1,
                "ascent_lr": 0.015,
                "ascent_batches": 6,
                "max_param_shift": 0.35,
                "use_sign_updates": True,
            },
        },
        "FisherCalHead": {
            "class": "kt_unlearn.unlearners.composite.Cascade",
            "parameters": {
                "sub_unlearner": [
                    {
                        "class": "kt_unlearn.unlearners.FisherForgetting.FisherForgetting",
                        "parameters": {"ref_data": "retain", "alpha": 8e-7, "ff_epochs": 64, "task": "kt"},
                    },
                    {
                        "class": "kt_unlearn.unlearners.Finetuning.Finetuning",
                        "parameters": {
                            "epochs": 1,
                            "ref_data": "retain",
                            "last_trainable_layers": 1,
                            "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.00015}},
                        },
                    },
                ]
            },
        },
        "FisherCalFull": {
            "class": "kt_unlearn.unlearners.composite.Cascade",
            "parameters": {
                "sub_unlearner": [
                    {
                        "class": "kt_unlearn.unlearners.FisherForgetting.FisherForgetting",
                        "parameters": {"ref_data": "retain", "alpha": 8e-7, "ff_epochs": 64, "task": "kt"},
                    },
                    {
                        "class": "kt_unlearn.unlearners.Finetuning.Finetuning",
                        "parameters": {
                            "epochs": 1,
                            "ref_data": "retain",
                            "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0001}},
                        },
                    },
                ]
            },
        },
        "FisherCalLite": {
            "class": "kt_unlearn.unlearners.FisherCalibrated.FisherCalibrated",
            "parameters": {
                "ref_data": "retain",
                "alpha": 1e-6,
                "ff_epochs": 64,
                "task": "kt",
                "calibration_ref_data": "retain",
                "calibration_steps": 16,
                "anchor_lambda": 0.001,
                "last_trainable_layers": 1,
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.00015}},
            },
        },
        "FisherCalLite32": {
            "class": "kt_unlearn.unlearners.FisherCalibrated.FisherCalibrated",
            "parameters": {
                "ref_data": "retain",
                "alpha": 8e-7,
                "ff_epochs": 64,
                "task": "kt",
                "calibration_ref_data": "retain",
                "calibration_steps": 32,
                "anchor_lambda": 0.002,
                "last_trainable_layers": 1,
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0001}},
            },
        },
        "FisherCalShield": {
            "class": "kt_unlearn.unlearners.FisherCalibrated.FisherCalibrated",
            "parameters": {
                "ref_data": "retain",
                "alpha": 1.5e-6,
                "ff_epochs": 64,
                "task": "kt",
                "calibration_ref_data": "retain",
                "calibration_steps": 8,
                "anchor_lambda": 0.002,
                "confidence_lambda": 0.02,
                "last_trainable_layers": 1,
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0001}},
            },
        },
        "STIRKT": {
            "class": "kt_unlearn.unlearners.STIRKT.STIRKT",
            "parameters": {
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "alpha": 0.0025,
                "mask_ratio": 0.025,
                "forget_steps": 3,
                "repair_epochs": 0,
                "step_repair_epochs": 3,
                "interleave_repair": True,
                "repair_weight": 1.0,
                "anchor_weight": 0.0015,
                "importance_power": 1.0,
                "grad_clip": 5.0,
                "max_update_norm": 0.02,
                "keep_head": True,
                "forget_optimizer": {"class": "torch.optim.SGD", "parameters": {"lr": 0.0025}},
                "repair_optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.00008}},
            },
        },
        "BadTeaching": {
            "class": "kt_unlearn.unlearners.BadTeaching.BadTeaching",
            "parameters": {
                "epochs": 1,
                "ref_data_retain": "retain",
                "ref_data_forget": "forget",
                "transform": None,
                "KL_temperature": 1.0,
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.0001}},
            },
        },
    }
    if method_name not in mapping:
        raise ValueError(f"Unsupported method: {method_name}")
    config = copy.deepcopy(mapping[method_name])
    config["alias"] = method_name
    config.setdefault("parameters", {})
    config["parameters"].setdefault("alias", method_name)
    return config


def strategy_splitter(strategy: str, params: dict[str, Any], split_dir: Path) -> dict[str, Any]:
    base = {
        "source_partition_name": "train_source",
        "forget_partition_name": "forget",
        "other_partition_name": "other_users",
        "parts_names": ["forget", "other_users"],
        "user_id_field": "uid",
        "correctness_field": "responses",
        "artifact_dir": str(split_dir),
    }
    if strategy == "high_performance":
        return {
            "class": "kt_unlearn.data.KTDataSplitterByPerformance",
            "parameters": {**base, "mode": "top_percent", "percent": params["percent"], "tie_break": "stable_uid", "seed": params["seed"]},
        }
    if strategy == "low_performance":
        return {
            "class": "kt_unlearn.data.KTDataSplitterByPerformance",
            "parameters": {**base, "mode": "bottom_percent", "percent": params["percent"], "tie_break": "stable_uid", "seed": params["seed"]},
        }
    if strategy == "random_percent":
        return {
            "class": "kt_unlearn.data.KTDataSplitterByRandomUsers",
            "parameters": {**base, "percent": params["percent"], "seed": params["seed"]},
        }
    if strategy == "low_participation":
        return {
            "class": "kt_unlearn.data.KTDataSplitterByParticipation",
            "parameters": {**base, "mode": params["participation_mode"], "threshold": params["threshold"]},
        }
    if strategy == "uid_list":
        return {
            "class": "kt_unlearn.data.KTDataSplitterByUserList",
            "parameters": {
                **base,
                "uid_list_file": str(params["uid_list_file"]),
                "allow_missing_uids": True,
                "sort_uids_before_use": True,
            },
        }
    raise ValueError(f"Unsupported strategy: {strategy}")


def predictor_model_config(model_name: str, emb_size: int, dropout: float) -> dict[str, Any]:
    normalized = model_name.strip().lower()
    if normalized == "dkt":
        return {
            "class": "kt_unlearn.model.KTModel.DKTSequenceModel",
            "parameters": {"emb_size": emb_size, "dropout": dropout},
        }
    if normalized in {"dkt+", "dktplus"}:
        return {
            "class": "kt_unlearn.model.KTModel.DKTPlusSequenceModel",
            "parameters": {"emb_size": emb_size, "hidden_size": emb_size, "dropout": dropout},
        }
    if normalized == "sakt":
        return {
            "class": "kt_unlearn.model.KTModel.SAKTSequenceModel",
            "parameters": {"emb_size": emb_size, "max_seq_len": 100, "num_attn_heads": 4, "num_blocks": 2, "dropout": dropout},
        }
    if normalized == "dkvmn":
        return {
            "class": "kt_unlearn.model.KTModel.DKVMNSequenceModel",
            "parameters": {"emb_size": emb_size, "memory_size": 50, "dropout": dropout},
        }
    if normalized == "akt":
        return {
            "class": "kt_unlearn.model.KTModel.AKTSequenceModel",
            "parameters": {"emb_size": emb_size, "num_attn_heads": 4, "num_blocks": 2, "dropout": dropout},
        }
    if normalized == "simplekt":
        return {
            "class": "kt_unlearn.model.KTModel.SimpleKTSequenceModel",
            "parameters": {"emb_size": emb_size, "num_attn_heads": 4, "num_blocks": 2, "dropout": dropout},
        }
    if normalized == "dtransformer":
        return {
            "class": "kt_unlearn.model.KTModel.DTransformerSequenceModel",
            "parameters": {"emb_size": emb_size, "num_attn_heads": 4, "num_blocks": 2, "num_prototypes": 16, "dropout": dropout},
        }
    raise ValueError(f"Unsupported KT model: {model_name}")


def build_config(
    dataset_path: Path,
    run_dir: Path,
    model_name: str,
    methods: list[str],
    strategy: str,
    strategy_params: dict[str, Any],
    seed: int,
    epochs: int,
    batch_size: int,
    emb_size: int,
    dropout: float,
    full_eval: bool,
) -> Path:
    results_csv = run_dir / "results.csv"
    split_dir = run_dir / "split_artifacts"
    retain_test_dir = run_dir / "retain_test_split"

    measures: list[dict[str, Any]] = [
        {
            "class": "erasure.evaluations.running.ChainOfRunners",
            "parameters": {
                "runners": [
                    "erasure.evaluations.running.TorchFlops",
                    "erasure.evaluations.running.RunTime",
                ]
            },
        }
    ]
    if full_eval:
        measures.extend(
            [
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTAUS", "parameters": {"forget_part": "forget", "test_part": "test"}},
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTRelearnTime", "parameters": {"forget_part": "forget", "max_epochs": 8}},
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTAIN", "parameters": {"alpha": 0.05, "forget_part": "forget", "max_epochs": 8}},
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTUMIA", "parameters": {"member_part": "forget", "nonmember_part": "test"}},
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTNoMUS", "parameters": {"l": 0.5, "acc_split": "test", "member_part": "forget", "nonmember_part": "test"}},
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTPartitionInfo", "parameters": {"partition": "forget"}},
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTPartitionInfo", "parameters": {"partition": "retain"}},
                {"class": "kt_unlearn.evaluations.erasure_bridge.KTPartitionInfo", "parameters": {"partition": "test"}},
            ]
        )
    measures.extend(
        [
            {"class": "kt_unlearn.evaluations.erasure_bridge.KTTorchSKLearn", "parameters": {"partition": "test", "target": "original"}},
            {"class": "kt_unlearn.evaluations.erasure_bridge.KTTorchSKLearn", "parameters": {"partition": "test", "target": "unlearned"}},
            {"class": "kt_unlearn.evaluations.erasure_bridge.KTTorchSKLearn", "parameters": {"partition": "forget", "target": "original"}},
            {"class": "kt_unlearn.evaluations.erasure_bridge.KTTorchSKLearn", "parameters": {"partition": "forget", "target": "unlearned"}},
            {"class": "kt_unlearn.evaluations.erasure_bridge.KTTorchSKLearn", "parameters": {"partition": "retain", "target": "original"}},
            {"class": "kt_unlearn.evaluations.erasure_bridge.KTTorchSKLearn", "parameters": {"partition": "retain", "target": "unlearned"}},
            {
                "class": "kt_unlearn.evaluations.erasure_bridge.KTSaveValues",
                "parameters": {
                    "path": str(results_csv),
                    "output_format": "csv",
                    "exclude_prefixes": ["parameters"],
                    "column_order": STANDARD_RESULT_COLUMNS,
                },
            },
        ]
    )

    config = {
        "data": {
            "class": "kt_unlearn.data.datasets.DatasetManager.DatasetManager",
            "parameters": {
                "DataSource": {
                    "class": "kt_unlearn.data.data_sources.KTFileDataSource.KTFileDataSource",
                    "parameters": {"path": str(dataset_path), "max_seq_len": 100, "pad_val": -1},
                },
                "partitions": [
                    {"class": "kt_unlearn.data.datasets.KTDataSplitter.KTDataSplitterByFold", "parameters": {"parts_names": ["train_source", "not_train_source"], "fold_values": [0]}},
                    strategy_splitter(strategy, {**strategy_params, "seed": seed}, split_dir=split_dir),
                    {
                        "class": "kt_unlearn.data.KTDataSplitterTrainRetainTest",
                        "parameters": {
                            "parts_names": ["retain", "test"],
                            "source_partition_name": "other_users",
                            "mode": "ratio",
                            "retain_ratio": 0.8,
                            "user_id_field": "uid",
                            "correctness_field": "responses",
                            "seed": seed,
                            "user_level_split": True,
                            "artifact_dir": str(retain_test_dir),
                        },
                    },
                ],
                "batch_size": batch_size,
            },
        },
        "predictor": {
            "class": "kt_unlearn.model.KTModel.KTModel",
            "parameters": {
                "epochs": epochs,
                "optimizer": {"class": "torch.optim.Adam", "parameters": {"lr": 0.001}},
                "loss_fn": {"class": "kt_unlearn.model.KTModel.KTLoss", "parameters": {}},
                "model": predictor_model_config(model_name, emb_size=emb_size, dropout=dropout),
                "training_set": "train_source",
                "batch_size": batch_size,
            },
        },
        "unlearners": [unlearner_config(name) for name in methods],
        "evaluator": {"class": "erasure.evaluations.manager.Evaluator", "parameters": {"measures": measures}},
        "globals": {"cached": "false", "seed": seed},
    }
    config_path = run_dir / "config.jsonc"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def run_experiment(config_path: Path) -> tuple[int, str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, str(FRAMEWORK_ROOT / "main.py"), str(config_path)],
        cwd=str(FRAMEWORK_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _coerce_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _coerce_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_coerce_value(val) for val in value]
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return ""
        if text.lower() in {"nan", "none", "null"}:
            return None
        try:
            num = float(text)
        except Exception:
            return value
        if math.isnan(num) or math.isinf(num):
            return None
        return num
    return value


def parse_results(results_csv: Path, split_summary_json: Path | None) -> list[dict[str, Any]]:
    split_summary = {}
    if split_summary_json and split_summary_json.exists():
        split_summary = _coerce_value(json.loads(split_summary_json.read_text(encoding="utf-8")))
    rows = []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            row = {
                "unlearner": rec.get("unlearner"),
                "forget_user_count": split_summary.get("forget_user_count"),
                "forget_mean_correctness": split_summary.get("forget_mean_correctness"),
                "forget_mean_interactions_per_user": split_summary.get("forget_mean_interactions_per_user"),
            }
            for key in METRIC_NAMES:
                if key in rec:
                    row[key] = _coerce_value(rec.get(key))
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--strategy", required=True, choices=["high_performance", "low_performance", "random_percent", "low_participation", "uid_list"])
    parser.add_argument("--model-name", default="DKT")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--emb-size", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--full-eval", action="store_true")
    parser.add_argument("--percent", type=float, default=0.2)
    parser.add_argument("--participation-mode", default="less_than")
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--uid-list-file")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_path, dataset_meta = prepare_sequence_csv(Path(args.input_csv), run_dir)
    strategy_params: dict[str, Any] = {}
    if args.strategy in {"high_performance", "low_performance", "random_percent"}:
        strategy_params["percent"] = args.percent
    elif args.strategy == "low_participation":
        strategy_params["participation_mode"] = args.participation_mode
        strategy_params["threshold"] = args.threshold
    else:
        if not args.uid_list_file:
            raise ValueError("--uid-list-file is required for strategy=uid_list")
        strategy_params["uid_list_file"] = Path(args.uid_list_file)

    config_path = build_config(
        dataset_path=dataset_path,
        run_dir=run_dir,
        model_name=args.model_name,
        methods=args.methods,
        strategy=args.strategy,
        strategy_params=strategy_params,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_size=args.emb_size,
        dropout=args.dropout,
        full_eval=args.full_eval,
    )
    code, stdout, stderr = run_experiment(config_path)
    results_csv = run_dir / "results.csv"
    split_summary_json = run_dir / "split_artifacts" / "split_summary.json"
    split_summary = _coerce_value(json.loads(split_summary_json.read_text(encoding="utf-8"))) if split_summary_json.exists() else {}
    payload = {
        "return_code": code,
        "stdout": stdout,
        "stderr": stderr,
        "dataset_meta": dataset_meta,
        "config_path": str(config_path),
        "results_csv": str(results_csv),
        "split_summary_json": str(split_summary_json),
        "summary_rows": parse_results(results_csv, split_summary_json) if code == 0 and results_csv.exists() else [],
        "split_summary": split_summary,
    }
    print(json.dumps(_coerce_value(payload), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
