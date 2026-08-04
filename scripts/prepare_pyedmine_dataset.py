from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
FRAMEWORK_ROOT = APP_ROOT.parent
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kt_lib.data.FileManager import FileManager
from kt_lib.data.KTDataProcessor import KTDataProcessor
from kt_lib.utils.data_io import write_json, write_kt_file


DATASET_ALIASES = {
    "assistments12": "assist2012",
    "assistments15": "assist2015",
    "assistments17": "assist2017",
}

DATASET_CHOICES = sorted(set(FileManager.data_preprocessed_dir.keys()) | set(DATASET_ALIASES.keys()))


def _coerce(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _coerce(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_coerce(item) for item in obj]
    return obj


def resolve_dataset_name(dataset_name: str) -> str:
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def dataset_status(dataset_name: str) -> dict[str, Any]:
    file_manager = FileManager(str(FRAMEWORK_ROOT), init_dirs=True)
    canonical_name = resolve_dataset_name(dataset_name)
    try:
        raw_path = Path(file_manager.get_dataset_raw_path(canonical_name))
    except KeyError:
        raw_path = FRAMEWORK_ROOT / "data" / "raw_datasets" / canonical_name
    try:
        processed_dir = Path(file_manager.get_preprocessed_dir(canonical_name))
    except KeyError:
        processed_dir = FRAMEWORK_ROOT / "data" / "processed_datasets" / dataset_name
    return {
        "requested_dataset_name": dataset_name,
        "dataset_name": canonical_name,
        "raw_path": raw_path,
        "raw_exists": raw_path.exists(),
        "processed_dir": processed_dir,
        "processed_dir_exists": processed_dir.exists(),
        "data_txt_path": processed_dir / "data.txt",
        "q_table_path": processed_dir / "Q_table.npy",
        "user_map_path": processed_dir / "user_id_map.csv",
        "question_map_path": processed_dir / "question_id_map.csv",
        "concept_map_path": processed_dir / "concept_id_map.csv",
    }


def prepare_dataset(dataset_name: str, raw_path: str | None = None, run_bridge: bool = False) -> dict[str, Any]:
    file_manager = FileManager(str(FRAMEWORK_ROOT), init_dirs=True)
    canonical_name = resolve_dataset_name(dataset_name)
    effective_raw_path = Path(raw_path).resolve() if raw_path else Path(file_manager.get_dataset_raw_path(canonical_name))
    if not effective_raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset for {dataset_name} not found. Expected: {effective_raw_path}"
        )

    params = {
        "dataset_name": canonical_name,
        "data_path": str(effective_raw_path),
    }
    processor = KTDataProcessor(params, file_manager)
    data_uniformed = processor.preprocess_data()

    processed_dir = Path(file_manager.get_preprocessed_dir(canonical_name))
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_txt_path = Path(file_manager.get_preprocessed_path(canonical_name))
    write_kt_file(data_uniformed, str(data_txt_path))
    if processor.statics_raw is not None:
        file_manager.save_data_statics_raw(processor.statics_raw, canonical_name)
    if processor.statics_preprocessed is not None:
        file_manager.save_data_statics_processed(processor.statics_preprocessed, canonical_name)
    if processor.Q_table is not None:
        file_manager.save_q_table(processor.Q_table, canonical_name)
    file_manager.save_data_id_map(processor.get_all_id_maps(), canonical_name)

    result = {
        "action": "preprocess",
        "requested_dataset_name": dataset_name,
        "dataset_name": canonical_name,
        "raw_path": effective_raw_path,
        "processed_dir": processed_dir,
        "data_txt_path": data_txt_path,
        "sequence_count": len(data_uniformed),
        "status": dataset_status(dataset_name),
    }
    if run_bridge:
        from kt_dataset_bridge import convert_dataset

        result["bridge"] = convert_dataset(dataset_name)
    return _coerce(result)


def _read_row_table(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="latin-1")
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _normalize_row_table(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    if "user_id" not in df.columns and "uid" in df.columns:
        rename_map["uid"] = "user_id"
    if "item_id" not in df.columns:
        if "problem_id" in df.columns:
            rename_map["problem_id"] = "item_id"
        elif "question_id" in df.columns:
            rename_map["question_id"] = "item_id"
    if "correct" not in df.columns:
        for candidate in ["correctness", "response", "is_correct"]:
            if candidate in df.columns:
                rename_map[candidate] = "correct"
                break
    if "timestamp" not in df.columns:
        for candidate in ["order_id", "log_id"]:
            if candidate in df.columns:
                rename_map[candidate] = "timestamp"
                break
    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["user_id", "item_id", "correct"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in row-table csv")

    df = df.copy()
    df["user_id"] = df["user_id"]
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce")
    if "skill_id" in df.columns:
        df["skill_id"] = pd.to_numeric(df["skill_id"], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "correct"]).copy()
    df["item_id"] = df["item_id"].astype(int)
    df["correct"] = df["correct"].astype(int)
    if "skill_id" in df.columns:
        df["skill_id"] = df["skill_id"].fillna(-1).astype(int)
    if "timestamp" not in df.columns:
        df["timestamp"] = range(len(df))
    else:
        df["timestamp"] = df["timestamp"].ffill().bfill().fillna(0).astype(int)
    return df


def import_row_csv(dataset_name: str, csv_path: str, run_bridge: bool = False) -> dict[str, Any]:
    canonical_name = resolve_dataset_name(dataset_name)
    source_path = Path(csv_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Row-table csv not found: {source_path}")

    file_manager = FileManager(str(FRAMEWORK_ROOT), init_dirs=True)
    try:
        processed_dir = Path(file_manager.get_preprocessed_dir(canonical_name))
    except KeyError:
        processed_dir = FRAMEWORK_ROOT / "data" / "processed_datasets" / dataset_name
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = _normalize_row_table(_read_row_table(source_path))
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    user_values = list(pd.unique(df["user_id"]))
    question_values = list(pd.unique(df["item_id"]))
    if "skill_id" in df.columns:
        concept_values = list(pd.unique(df["skill_id"]))
    else:
        concept_values = question_values
        df["skill_id"] = df["item_id"]

    user_map = {value: idx for idx, value in enumerate(user_values)}
    question_map = {value: idx for idx, value in enumerate(question_values)}
    concept_map = {value: idx for idx, value in enumerate(concept_values)}

    df["user_id_mapped"] = df["user_id"].map(user_map)
    df["item_id_mapped"] = df["item_id"].map(question_map)
    df["skill_id_mapped"] = df["skill_id"].map(concept_map)

    q_table = np.zeros((len(question_values), len(concept_values)), dtype=int)
    for _, row in df[["item_id_mapped", "skill_id_mapped"]].drop_duplicates().iterrows():
        q_table[int(row["item_id_mapped"]), int(row["skill_id_mapped"])] = 1

    data_uniformed = []
    for user_id, group in df.groupby("user_id_mapped", sort=False):
        group = group.sort_values(["timestamp", "item_id_mapped"])
        question_seq = [int(v) for v in group["item_id_mapped"].tolist()]
        correctness_seq = [int(v) for v in group["correct"].tolist()]
        concept_seq = [int(v) for v in group["skill_id_mapped"].tolist()]
        time_seq = [int(v) for v in group["timestamp"].tolist()]
        if len(question_seq) < 2:
            continue
        data_uniformed.append(
            {
                "user_id": int(user_id),
                "seq_len": len(question_seq),
                "question_seq": question_seq,
                "correctness_seq": correctness_seq,
                "concept_seq": concept_seq,
                "time_seq": time_seq,
            }
        )

    if not data_uniformed:
        raise ValueError(f"No valid sequences could be built from {source_path}")

    data_txt_path = processed_dir / "data.txt"
    write_kt_file(data_uniformed, str(data_txt_path))
    np.save(processed_dir / "Q_table.npy", q_table)
    pd.DataFrame({"original_id": question_values, "mapped_id": range(len(question_values))}).to_csv(
        processed_dir / "question_id_map.csv", index=False
    )
    pd.DataFrame({"original_id": concept_values, "mapped_id": range(len(concept_values))}).to_csv(
        processed_dir / "concept_id_map.csv", index=False
    )
    pd.DataFrame({"original_id": user_values, "mapped_id": range(len(user_values))}).to_csv(
        processed_dir / "user_id_map.csv", index=False
    )

    statics = {
        "num_user": len(user_values),
        "num_question": len(question_values),
        "num_concept": len(concept_values),
        "num_interaction": int(len(df)),
        "average_seq_len": float(sum(item["seq_len"] for item in data_uniformed) / len(data_uniformed)),
    }
    write_json(statics, processed_dir / "statics_raw.json")
    write_json(statics, processed_dir / "statics_preprocessed.json")

    result = {
        "action": "import_row_csv",
        "requested_dataset_name": dataset_name,
        "dataset_name": canonical_name,
        "csv_path": source_path,
        "processed_dir": processed_dir,
        "data_txt_path": data_txt_path,
        "sequence_count": len(data_uniformed),
        "user_count": len(user_values),
        "question_count": len(question_values),
        "concept_count": len(concept_values),
    }
    if run_bridge:
        from kt_dataset_bridge import convert_dataset

        result["bridge"] = convert_dataset(dataset_name)
    return _coerce(result)


def prepare_ednet_raw(
    dataset_src_dir: str,
    contents_dir: str,
    chunk_size: int = 5000,
    max_users: int = 5000,
) -> dict[str, Any]:
    file_manager = FileManager(str(FRAMEWORK_ROOT), init_dirs=True)
    dataset_src_root = Path(dataset_src_dir).resolve()
    contents_root = Path(contents_dir).resolve()
    questions_path = contents_root / "questions.csv"
    if not dataset_src_root.exists():
        raise FileNotFoundError(f"EdNet KT1 source directory not found: {dataset_src_root}")
    if not questions_path.exists():
        raise FileNotFoundError(f"EdNet contents file not found: {questions_path}")

    save_dir = Path(file_manager.get_dataset_raw_path("ednet-kt1"))
    save_dir.mkdir(parents=True, exist_ok=True)

    question_content = pd.read_csv(questions_path, usecols=["question_id", "correct_answer", "tags"])
    question_content["tags"] = question_content["tags"].astype(str).map(lambda x: x.replace(";", "_"))
    question_content = question_content[question_content["tags"] != "-1"]

    user_lengths: list[tuple[int, int]] = []
    for user_path in dataset_src_root.glob("u*.csv"):
        try:
            uid = int(user_path.stem[1:])
        except ValueError:
            continue
        try:
            df = pd.read_csv(user_path, usecols=["question_id"])
        except Exception:
            df = pd.read_csv(user_path)
        user_lengths.append((uid, len(df)))

    if not user_lengths:
        raise ValueError(f"No user csv files were found under {dataset_src_root}")

    random.shuffle(user_lengths)
    user_lengths.sort(key=lambda item: item[1], reverse=True)
    selected_users = user_lengths[:max_users] if max_users > 0 else user_lengths

    files: list[pd.DataFrame] = []
    written_files: list[Path] = []
    for user_idx, (uid, _) in enumerate(selected_users, start=1):
        user_path = dataset_src_root / f"u{uid}.csv"
        df = pd.read_csv(user_path)
        df["user_id"] = uid
        files.append(df)
        flush_now = (user_idx % chunk_size == 0) or (user_idx == len(selected_users))
        if not flush_now:
            continue

        chunk_df = pd.concat(files, axis=0, ignore_index=True)
        chunk_df = pd.merge(chunk_df, question_content, how="left", on="question_id")
        chunk_df = chunk_df.dropna(subset=["user_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
        chunk_df["correct"] = (chunk_df["correct_answer"] == chunk_df["user_answer"]).astype(int)
        chunk_df = chunk_df[["timestamp", "question_id", "elapsed_time", "user_id", "tags", "correct"]]

        chunk_index = len(written_files)
        save_path = save_dir / f"users_{chunk_index}.csv"
        chunk_df.to_csv(save_path, index=False)
        written_files.append(save_path)
        files = []

    return _coerce(
        {
            "action": "prepare_ednet_raw",
            "dataset_name": "ednet-kt1",
            "dataset_src_dir": dataset_src_root,
            "contents_dir": contents_root,
            "save_dir": save_dir,
            "selected_user_count": len(selected_users),
            "chunk_size": chunk_size,
            "written_files": written_files,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare PyEdmine-style KT datasets under the current UnlearnKT data directories."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--dataset-name", required=True, choices=DATASET_CHOICES)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--dataset-name", required=True, choices=DATASET_CHOICES)
    preprocess_parser.add_argument("--raw-path")
    preprocess_parser.add_argument("--run-bridge", action="store_true")

    row_csv_parser = subparsers.add_parser("import-row-csv")
    row_csv_parser.add_argument("--dataset-name", required=True)
    row_csv_parser.add_argument("--csv-path", required=True)
    row_csv_parser.add_argument("--run-bridge", action="store_true")

    ednet_parser = subparsers.add_parser("prepare-ednet-raw")
    ednet_parser.add_argument("--dataset-src-dir", required=True)
    ednet_parser.add_argument("--contents-dir", required=True)
    ednet_parser.add_argument("--chunk-size", type=int, default=5000)
    ednet_parser.add_argument("--max-users", type=int, default=5000)

    args = parser.parse_args()
    if args.command == "list":
        payload = {"datasets": [_coerce(dataset_status(name)) for name in DATASET_CHOICES]}
    elif args.command == "status":
        payload = {"dataset": _coerce(dataset_status(args.dataset_name))}
    elif args.command == "preprocess":
        payload = prepare_dataset(args.dataset_name, raw_path=args.raw_path, run_bridge=args.run_bridge)
    elif args.command == "import-row-csv":
        payload = import_row_csv(args.dataset_name, csv_path=args.csv_path, run_bridge=args.run_bridge)
    elif args.command == "prepare-ednet-raw":
        payload = prepare_ednet_raw(
            dataset_src_dir=args.dataset_src_dir,
            contents_dir=args.contents_dir,
            chunk_size=args.chunk_size,
            max_users=args.max_users,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(_coerce(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
