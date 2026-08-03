from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
FRAMEWORK_ROOT = APP_ROOT.parent
SRC_ROOT = FRAMEWORK_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kt_lib.utils.data_io import read_kt_file

DATA_ROOT = FRAMEWORK_ROOT / "data" / "processed_datasets"
CONVERTED_ROOT = FRAMEWORK_ROOT / "output" / "runs" / "kt" / "pyedmine_converted"

DATASET_DESCRIPTIONS = {
    "assist2009": "ASSISTments 2009 常用 KT 基准数据集",
    "assist2009-full": "ASSISTments 2009 全量版本",
    "assist2012": "ASSISTments 2012 常用基准",
    "assist2015": "ASSISTments 2015 常用基准",
    "assist2017": "ASSISTments 2017 常用基准",
    "statics2011": "Statics2011 物理题目序列数据",
    "junyi2015": "Junyi Academy 2015 学习行为数据",
    "algebra2005": "KDD Cup 2010 Algebra 2005",
    "algebra2006": "KDD Cup 2010 Algebra 2006",
    "algebra2008": "KDD Cup 2010 Algebra 2008",
    "bridge2algebra2006": "Bridge to Algebra 2006",
    "bridge2algebra2008": "Bridge to Algebra 2008",
    "ednet-kt1": "EdNet KT1 大规模交互数据",
    "slepemapy-anatomy": "Slepemapy Anatomy KT 数据",
    "SLP-bio": "SLP 生物学科数据",
    "SLP-chi": "SLP 语文学科数据",
    "SLP-eng": "SLP 英语学科数据",
    "SLP-geo": "SLP 地理学科数据",
    "SLP-his": "SLP 历史学科数据",
    "SLP-mat": "SLP 数学学科数据",
    "SLP-phy": "SLP 物理学科数据",
    "poj": "Programming Online Judge 数据",
    "xes3g5m": "希沃题目序列数据",
    "DBE-KT22": "DBE-KT22 数据集",
    "edi2020-task1": "EDI2020 Task1",
    "edi2020-task34": "EDI2020 Task3/4",
}


def _dataset_names() -> list[str]:
    if not DATA_ROOT.exists():
        return []
    return sorted([path.name for path in DATA_ROOT.iterdir() if path.is_dir()])


def _converted_path(dataset_name: str) -> Path:
    return CONVERTED_ROOT / f"{dataset_name}_sequences.csv"


def _dataset_dir(dataset_name: str) -> Path:
    direct = DATA_ROOT / dataset_name
    if direct.exists():
        return direct
    lowered = dataset_name.lower()
    for path in DATA_ROOT.iterdir():
        if path.is_dir() and path.name.lower() == lowered:
            return path
    return direct


def _detect_existing_files(dataset_name: str) -> dict[str, Path]:
    dataset_dir = _dataset_dir(dataset_name)
    return {
        "dataset_dir": dataset_dir,
        "data_txt": dataset_dir / "data.txt",
        "preprocessed": dataset_dir / "preprocessed_data.csv",
        "train": dataset_dir / "preprocessed_data_train.csv",
        "test": dataset_dir / "preprocessed_data_test.csv",
        "q_mat": dataset_dir / "q_mat.npz",
        "assist2009_full": dataset_dir / "skill_builder_data.csv",
    }


def _coerce(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {key: _coerce(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_coerce(item) for item in obj]
    return obj


def _path_has_payload(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    try:
        next(path.iterdir())
        return True
    except StopIteration:
        return False


def get_dataset_status(dataset_name: str) -> dict[str, Any]:
    files = _detect_existing_files(dataset_name)
    dataset_dir = files["dataset_dir"]
    raw_path = files["assist2009_full"]
    preprocessed_path = files["preprocessed"]
    train_path = files["train"]
    test_path = files["test"]
    q_table_path = files["q_mat"]
    converted_path = _converted_path(dataset_name)
    return {
        "dataset_name": dataset_name,
        "description": DATASET_DESCRIPTIONS.get(dataset_name, ""),
        "dataset_dir": dataset_dir,
        "raw_path": raw_path,
        "preprocessed_path": preprocessed_path,
        "train_path": train_path,
        "test_path": test_path,
        "q_table_path": q_table_path,
        "converted_path": converted_path,
        "raw_exists": raw_path.exists(),
        "dataset_exists": dataset_dir.exists(),
        "preprocessed_exists": preprocessed_path.exists(),
        "train_exists": train_path.exists(),
        "test_exists": test_path.exists(),
        "q_table_exists": q_table_path.exists(),
        "converted_exists": converted_path.exists(),
    }


def list_dataset_statuses() -> list[dict[str, Any]]:
    statuses = [get_dataset_status(name) for name in _dataset_names()]
    return [_coerce(item) for item in statuses]


def preprocess_dataset(dataset_name: str) -> dict[str, Any]:
    return {
        "action": "preprocess",
        "dataset_name": dataset_name,
        "message": "fin-1 打包的是已处理 KT 数据集。原始 pyedmine 预处理源码已保留在 src/kt_lib，但这里默认不再直接重跑原始全量预处理。",
        "status": get_dataset_status(dataset_name),
    }


def convert_dataset(dataset_name: str) -> dict[str, Any]:
    files = _detect_existing_files(dataset_name)
    question_ids = set()
    concept_ids = set()
    rows = []
    source_path = files["preprocessed"] if files["preprocessed"].exists() else files["train"]
    if source_path.exists():
        df = pd.read_csv(source_path)
        required = {"user_id", "item_id", "correct"}
        if not required.issubset(df.columns):
            raise ValueError(f"Expected columns {sorted(required)} in {source_path}")

        order_cols = [col for col in ["timestamp", "order_id"] if col in df.columns]
        if order_cols:
            df = df.sort_values(["user_id", *order_cols])

        for uid, group in df.groupby("user_id", sort=False):
            questions = [int(v) for v in group["item_id"].tolist()]
            responses = [int(v) for v in group["correct"].tolist()]
            if len(questions) < 2:
                continue
            row = {
                "uid": str(uid),
                "questions": ",".join(map(str, questions)),
                "responses": ",".join(map(str, responses)),
                "fold": 0,
            }
            if "skill_id" in group.columns:
                concepts = [int(v) for v in group["skill_id"].tolist()]
                row["concepts"] = ",".join(map(str, concepts))
                concept_ids.update(concepts)
            rows.append(row)
            question_ids.update(questions)
    elif files["data_txt"].exists():
        data = read_kt_file(str(files["data_txt"]))
        for item in data:
            questions = [int(v) for v in item.get("question_seq", [])]
            responses = [int(v) for v in item.get("correctness_seq", [])]
            if len(questions) != len(responses) or len(questions) < 2:
                continue
            row = {
                "uid": str(item.get("user_id")),
                "questions": ",".join(map(str, questions)),
                "responses": ",".join(map(str, responses)),
                "fold": 0,
            }
            concept_seq = item.get("concept_seq")
            if isinstance(concept_seq, list) and len(concept_seq) == len(questions):
                row["concepts"] = ",".join(map(str, map(int, concept_seq)))
                concept_ids.update(int(v) for v in concept_seq)
            rows.append(row)
            question_ids.update(questions)
    else:
        raise FileNotFoundError(
            f"No packaged processed KT source found for {dataset_name}: "
            f"{files['preprocessed']} / {files['train']} / {files['data_txt']}"
        )

    if not rows:
        raise ValueError(f"No valid KT sequences were converted from packaged dataset {dataset_name}")

    CONVERTED_ROOT.mkdir(parents=True, exist_ok=True)
    converted_path = _converted_path(dataset_name)
    pd.DataFrame(rows).to_csv(converted_path, index=False)
    meta = {
        "action": "convert",
        "dataset_name": dataset_name,
        "converted_path": converted_path,
        "sequence_count": len(rows),
        "user_count": len({row["uid"] for row in rows}),
        "question_count": len(question_ids),
        "concept_count": len(concept_ids),
        "status": get_dataset_status(dataset_name),
    }
    return _coerce(meta)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list")

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--dataset-name", required=True)

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("--dataset-name", required=True)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--dataset-name", required=True)

    args = parser.parse_args()
    if args.command == "list":
        payload = {"datasets": list_dataset_statuses()}
    elif args.command == "status":
        payload = {"dataset": _coerce(get_dataset_status(args.dataset_name))}
    elif args.command == "preprocess":
        payload = preprocess_dataset(args.dataset_name)
    elif args.command == "convert":
        payload = convert_dataset(args.dataset_name)
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(_coerce(payload), ensure_ascii=False))


if __name__ == "__main__":
    main()
