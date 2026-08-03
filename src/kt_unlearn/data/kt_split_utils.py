from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_sequence_field(value: Any) -> list[int]:
    if value is None:
        return []
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [int(x) for x in s.split(",") if x != ""]


def ensure_partition_exists(partitions: dict[str, Any], partition_name: str) -> None:
    if partition_name not in partitions:
        raise ValueError(f"Source partition '{partition_name}' does not exist. Available: {sorted(partitions.keys())}")


def resolve_source_indices(partitions: dict[str, Any], partition_name: str) -> list[int]:
    ensure_partition_exists(partitions, partition_name)
    if partition_name == "all":
        return list(range(len(partitions["all"].data)))
    return list(partitions[partition_name])


def load_uid_list(uid_list: list[Any] | None, uid_list_file: str | None, sort_before_use: bool) -> list[str]:
    if uid_list is None and uid_list_file is None:
        raise ValueError("Provide either 'uid_list' or 'uid_list_file'.")
    if uid_list is not None and uid_list_file is not None:
        raise ValueError("Provide only one of 'uid_list' or 'uid_list_file'.")

    if uid_list is not None:
        result = [str(x) for x in uid_list]
    else:
        file_path = Path(str(uid_list_file))
        if not file_path.exists():
            raise FileNotFoundError(f"uid_list_file not found: {file_path}")
        if file_path.suffix.lower() == ".json":
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list):
                raise ValueError("uid_list_file JSON must contain a list of user ids.")
            result = [str(x) for x in payload]
        else:
            result = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return sorted(result) if sort_before_use else result


def get_aligned_rows(source) -> list[dict[str, Any]]:
    rows = getattr(source, "aligned_rows", None)
    if not rows:
        raise ValueError("KT splitters require source.aligned_rows. Recreate data through KTFileDataSource.")
    return list(rows)


def build_user_stats(
    source,
    source_indices: list[int],
    user_id_field: str,
    correctness_field: str,
    min_interactions_per_user: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    rows = get_aligned_rows(source)
    by_user: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}

    for idx in source_indices:
        row = rows[idx]
        if user_id_field not in row:
            raise ValueError(f"user_id_field '{user_id_field}' not found in KT row.")
        if correctness_field not in row:
            raise ValueError(f"correctness_field '{correctness_field}' not found in KT row.")

        uid = str(row[user_id_field])
        correctness_values = parse_sequence_field(row[correctness_field])
        n_interactions = len(correctness_values)
        if n_interactions == 0:
            scalar = row[correctness_field]
            try:
                correctness_values = [int(scalar)]
                n_interactions = 1
            except Exception:
                correctness_values = []
                n_interactions = 0

        entry = by_user.setdefault(
            uid,
            {"user_id": uid, "indices": [], "n_interactions": 0, "correct_sum": 0.0, "n_values": 0},
        )
        entry["indices"].append(idx)
        entry["n_interactions"] += n_interactions
        entry["correct_sum"] += float(sum(correctness_values))
        entry["n_values"] += len(correctness_values)
        by_index[idx] = row

    for uid in list(by_user.keys()):
        entry = by_user[uid]
        entry["avg_correct"] = (
            entry["correct_sum"] / entry["n_values"] if entry["n_values"] > 0 else float("nan")
        )
        if min_interactions_per_user is not None and entry["n_interactions"] < int(min_interactions_per_user):
            del by_user[uid]

    return by_user, by_index


def select_top_bottom_users(
    user_stats: dict[str, dict[str, Any]],
    mode: str,
    percent: float,
    tie_break: str,
    seed: int,
) -> list[str]:
    if not (0.0 < float(percent) <= 1.0):
        raise ValueError("percent must be in (0, 1].")
    if mode not in {"top_percent", "bottom_percent"}:
        raise ValueError("mode must be one of ['top_percent', 'bottom_percent'].")
    if tie_break not in {"stable_uid", "random_with_seed"}:
        raise ValueError("tie_break must be one of ['stable_uid', 'random_with_seed'].")

    users = list(user_stats.values())
    if not users:
        raise ValueError("No users available after filtering; forget set cannot be created.")

    rng = random.Random(seed)
    if tie_break == "random_with_seed":
        rng.shuffle(users)
    else:
        users = sorted(users, key=lambda x: str(x["user_id"]))

    reverse = mode == "top_percent"
    users = sorted(users, key=lambda x: (x["avg_correct"], x["n_interactions"]), reverse=reverse)

    k = max(1, int(len(users) * float(percent)))
    return [str(x["user_id"]) for x in users[:k]]


def select_random_users(user_stats: dict[str, dict[str, Any]], percent: float, seed: int) -> list[str]:
    if not (0.0 < float(percent) <= 1.0):
        raise ValueError("percent must be in (0, 1].")

    uids = sorted(user_stats.keys())
    if not uids:
        raise ValueError("No users available after filtering; forget set cannot be created.")

    rng = random.Random(seed)
    rng.shuffle(uids)
    k = max(1, int(len(uids) * float(percent)))
    return sorted(uids[:k])


def select_users_by_participation(
    user_stats: dict[str, dict[str, Any]], mode: str, threshold: int
) -> list[str]:
    if threshold < 0:
        raise ValueError("threshold must be a non-negative integer.")
    if mode not in {"less_than", "less_equal", "greater_than", "greater_equal", "exact"}:
        raise ValueError("Invalid participation mode.")

    out = []
    for uid, entry in user_stats.items():
        count = int(entry["n_interactions"])
        keep = (
            (mode == "less_than" and count < threshold)
            or (mode == "less_equal" and count <= threshold)
            or (mode == "greater_than" and count > threshold)
            or (mode == "greater_equal" and count >= threshold)
            or (mode == "exact" and count == threshold)
        )
        if keep:
            out.append(uid)
    return sorted(out)


def split_indices_by_users(source_indices: list[int], user_stats: dict[str, dict[str, Any]], forget_uids: set[str]) -> tuple[list[int], list[int]]:
    forget_indices: list[int] = []
    other_indices: list[int] = []
    for uid, entry in user_stats.items():
        if uid in forget_uids:
            forget_indices.extend(entry["indices"])
        else:
            other_indices.extend(entry["indices"])

    # preserve source partition order for reproducibility
    source_position = {idx: pos for pos, idx in enumerate(source_indices)}
    forget_indices.sort(key=lambda x: source_position[x])
    other_indices.sort(key=lambda x: source_position[x])
    return forget_indices, other_indices


def user_level_ratio_split(
    user_stats: dict[str, dict[str, Any]], retain_ratio: float, seed: int
) -> tuple[list[str], list[str]]:
    if not (0.0 < retain_ratio <= 1.0):
        raise ValueError("retain_ratio must be in (0, 1].")
    uids = sorted(user_stats.keys())
    rng = random.Random(seed)
    rng.shuffle(uids)
    split_point = max(1, int(len(uids) * retain_ratio))
    split_point = min(split_point, len(uids) - 1) if len(uids) > 1 else len(uids)
    retain_uids = uids[:split_point]
    test_uids = uids[split_point:]
    return sorted(retain_uids), sorted(test_uids)


def partition_summary_for_users(
    user_stats: dict[str, dict[str, Any]], selected_uids: list[str]
) -> dict[str, Any]:
    subset = [user_stats[uid] for uid in selected_uids if uid in user_stats]
    total_users = len(subset)
    total_interactions = sum(int(x["n_interactions"]) for x in subset)
    mean_correctness = (
        float(sum(float(x["correct_sum"]) for x in subset) / max(1, sum(int(x["n_values"]) for x in subset)))
        if subset
        else float("nan")
    )
    mean_interactions = float(total_interactions / total_users) if total_users > 0 else float("nan")
    return {
        "user_count": total_users,
        "interaction_count": total_interactions,
        "mean_correctness": mean_correctness,
        "mean_interactions_per_user": mean_interactions,
    }


def save_split_artifacts(
    output_dir: str | Path,
    forget_uids: list[str],
    retain_uids: list[str],
    test_uids: list[str],
    summary: dict[str, Any],
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "forget_user_ids.json").write_text(json.dumps(forget_uids, indent=2), encoding="utf-8")
    (out_dir / "retain_user_ids.json").write_text(json.dumps(retain_uids, indent=2), encoding="utf-8")
    (out_dir / "test_user_ids.json").write_text(json.dumps(test_uids, indent=2), encoding="utf-8")
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = out_dir / "split_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
