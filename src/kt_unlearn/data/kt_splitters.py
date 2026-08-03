from __future__ import annotations

from pathlib import Path

from kt_unlearn.data.datasets.DataSplitter import DataSplitter

from .kt_split_utils import (
    build_user_stats,
    load_uid_list,
    partition_summary_for_users,
    resolve_source_indices,
    save_split_artifacts,
    select_random_users,
    select_top_bottom_users,
    select_users_by_participation,
    split_indices_by_users,
    user_level_ratio_split,
)


class _KTBaseUserSplitter(DataSplitter):
    def __init__(
        self,
        source_partition_name: str | None = None,
        forget_partition_name: str | None = None,
        other_partition_name: str | None = None,
        user_id_field: str = "uid",
        parts_names: list[str] | None = None,
        ref_data: str | None = None,
        correctness_field: str = "responses",
        artifact_dir: str | None = None,
    ):
        source_partition_name = source_partition_name or ref_data
        if source_partition_name is None:
            raise ValueError("source_partition_name (or ref_data) is required.")
        if parts_names is not None:
            if len(parts_names) != 2:
                raise ValueError("parts_names must contain exactly two names.")
            forget_partition_name = parts_names[0]
            other_partition_name = parts_names[1]
        if forget_partition_name is None or other_partition_name is None:
            raise ValueError("forget_partition_name/other_partition_name (or parts_names) are required.")
        super().__init__(source_partition_name, [forget_partition_name, other_partition_name])
        self.source_partition_name = source_partition_name
        self.forget_partition_name = forget_partition_name
        self.other_partition_name = other_partition_name
        self.user_id_field = user_id_field
        self.correctness_field = correctness_field
        self.artifact_dir = artifact_dir

        self.last_forget_user_ids: list[str] = []
        self.last_other_user_ids: list[str] = []
        self.last_summary: dict = {}

    def _artifact_dir(self) -> Path:
        if self.artifact_dir is not None:
            return Path(self.artifact_dir)
        dataset_name = getattr(self.source, "get_name", lambda: "kt_dataset")()
        return Path("resources") / "kt_splits" / dataset_name / self.__class__.__name__

    def _save_artifacts(
        self,
        user_stats: dict[str, dict],
        forget_user_ids: list[str],
        other_user_ids: list[str],
        test_user_ids: list[str] | None = None,
        extra_summary: dict | None = None,
    ) -> None:
        forget_summary = partition_summary_for_users(user_stats, forget_user_ids)
        other_summary = partition_summary_for_users(user_stats, other_user_ids)
        test_summary = partition_summary_for_users(user_stats, test_user_ids or [])
        summary = {
            "source_partition_name": self.source_partition_name,
            "forget_partition_name": self.forget_partition_name,
            "other_partition_name": self.other_partition_name,
            "total_users": len(user_stats),
            "forget_user_count": forget_summary["user_count"],
            "other_user_count": other_summary["user_count"],
            "test_user_count": test_summary["user_count"],
            "forget_interactions": forget_summary["interaction_count"],
            "other_interactions": other_summary["interaction_count"],
            "test_interactions": test_summary["interaction_count"],
            "forget_mean_correctness": forget_summary["mean_correctness"],
            "other_mean_correctness": other_summary["mean_correctness"],
            "test_mean_correctness": test_summary["mean_correctness"],
            "forget_mean_interactions_per_user": forget_summary["mean_interactions_per_user"],
            "other_mean_interactions_per_user": other_summary["mean_interactions_per_user"],
            "test_mean_interactions_per_user": test_summary["mean_interactions_per_user"],
        }
        if extra_summary:
            summary.update(extra_summary)

        save_split_artifacts(
            output_dir=self._artifact_dir(),
            forget_uids=forget_user_ids,
            retain_uids=other_user_ids,
            test_uids=test_user_ids or [],
            summary=summary,
        )
        self.last_forget_user_ids = list(forget_user_ids)
        self.last_other_user_ids = list(other_user_ids)
        self.last_summary = summary


class KTDataSplitterByPerformance(_KTBaseUserSplitter):
    def __init__(
        self,
        source_partition_name: str | None = None,
        forget_partition_name: str | None = None,
        other_partition_name: str | None = None,
        mode: str = "top_percent",
        percent: float = 0.2,
        user_id_field: str = "uid",
        correctness_field: str = "responses",
        parts_names: list[str] | None = None,
        ref_data: str | None = None,
        tie_break: str = "stable_uid",
        min_interactions_per_user: int | None = None,
        seed: int | None = None,
        artifact_dir: str | None = None,
    ):
        super().__init__(
            source_partition_name=source_partition_name,
            forget_partition_name=forget_partition_name,
            other_partition_name=other_partition_name,
            user_id_field=user_id_field,
            parts_names=parts_names,
            ref_data=ref_data,
            correctness_field=correctness_field,
            artifact_dir=artifact_dir,
        )
        self.mode = mode
        self.percent = float(percent)
        self.tie_break = tie_break
        self.min_interactions_per_user = min_interactions_per_user
        self.seed = 0 if seed is None else int(seed)

    def split_data(self, partitions):
        source_indices = resolve_source_indices(partitions, self.source_partition_name)
        user_stats, _ = build_user_stats(
            self.source,
            source_indices=source_indices,
            user_id_field=self.user_id_field,
            correctness_field=self.correctness_field,
            min_interactions_per_user=self.min_interactions_per_user,
        )
        forget_uids = select_top_bottom_users(
            user_stats=user_stats,
            mode=self.mode,
            percent=self.percent,
            tie_break=self.tie_break,
            seed=self.seed,
        )
        forget_indices, other_indices = split_indices_by_users(source_indices, user_stats, set(forget_uids))
        if not forget_indices:
            raise ValueError("Forget set is empty after performance split.")
        if not other_indices:
            raise ValueError("Other set is empty after performance split.")

        partitions[self.forget_partition_name] = forget_indices
        partitions[self.other_partition_name] = other_indices
        self._save_artifacts(
            user_stats=user_stats,
            forget_user_ids=forget_uids,
            other_user_ids=sorted([uid for uid in user_stats.keys() if uid not in set(forget_uids)]),
            extra_summary={
                "strategy": "performance",
                "mode": self.mode,
                "percent": self.percent,
                "tie_break": self.tie_break,
                "seed": self.seed,
            },
        )
        return partitions


class KTDataSplitterByParticipation(_KTBaseUserSplitter):
    def __init__(
        self,
        source_partition_name: str | None = None,
        forget_partition_name: str | None = None,
        other_partition_name: str | None = None,
        mode: str = "less_than",
        threshold: int = 5,
        user_id_field: str = "uid",
        parts_names: list[str] | None = None,
        ref_data: str | None = None,
        correctness_field: str = "responses",
        artifact_dir: str | None = None,
    ):
        super().__init__(
            source_partition_name=source_partition_name,
            forget_partition_name=forget_partition_name,
            other_partition_name=other_partition_name,
            user_id_field=user_id_field,
            parts_names=parts_names,
            ref_data=ref_data,
            correctness_field=correctness_field,
            artifact_dir=artifact_dir,
        )
        self.mode = mode
        self.threshold = int(threshold)

    def split_data(self, partitions):
        source_indices = resolve_source_indices(partitions, self.source_partition_name)
        user_stats, _ = build_user_stats(
            self.source,
            source_indices=source_indices,
            user_id_field=self.user_id_field,
            correctness_field=self.correctness_field,
        )
        forget_uids = select_users_by_participation(user_stats, mode=self.mode, threshold=self.threshold)
        forget_indices, other_indices = split_indices_by_users(source_indices, user_stats, set(forget_uids))
        if not forget_indices:
            raise ValueError("Forget set is empty after participation split.")
        if not other_indices:
            raise ValueError("Other set is empty after participation split.")

        partitions[self.forget_partition_name] = forget_indices
        partitions[self.other_partition_name] = other_indices
        self._save_artifacts(
            user_stats=user_stats,
            forget_user_ids=forget_uids,
            other_user_ids=sorted([uid for uid in user_stats.keys() if uid not in set(forget_uids)]),
            extra_summary={"strategy": "participation", "mode": self.mode, "threshold": self.threshold},
        )
        return partitions


class KTDataSplitterByRandomUsers(_KTBaseUserSplitter):
    def __init__(
        self,
        source_partition_name: str | None = None,
        forget_partition_name: str | None = None,
        other_partition_name: str | None = None,
        percent: float = 0.2,
        user_id_field: str = "uid",
        correctness_field: str = "responses",
        parts_names: list[str] | None = None,
        ref_data: str | None = None,
        seed: int | None = None,
        artifact_dir: str | None = None,
    ):
        super().__init__(
            source_partition_name=source_partition_name,
            forget_partition_name=forget_partition_name,
            other_partition_name=other_partition_name,
            user_id_field=user_id_field,
            parts_names=parts_names,
            ref_data=ref_data,
            correctness_field=correctness_field,
            artifact_dir=artifact_dir,
        )
        self.percent = float(percent)
        self.seed = 0 if seed is None else int(seed)

    def split_data(self, partitions):
        source_indices = resolve_source_indices(partitions, self.source_partition_name)
        user_stats, _ = build_user_stats(
            self.source,
            source_indices=source_indices,
            user_id_field=self.user_id_field,
            correctness_field=self.correctness_field,
        )
        forget_uids = select_random_users(user_stats=user_stats, percent=self.percent, seed=self.seed)
        forget_indices, other_indices = split_indices_by_users(source_indices, user_stats, set(forget_uids))
        if not forget_indices:
            raise ValueError("Forget set is empty after random split.")
        if not other_indices:
            raise ValueError("Other set is empty after random split.")

        partitions[self.forget_partition_name] = forget_indices
        partitions[self.other_partition_name] = other_indices
        self._save_artifacts(
            user_stats=user_stats,
            forget_user_ids=forget_uids,
            other_user_ids=sorted([uid for uid in user_stats.keys() if uid not in set(forget_uids)]),
            extra_summary={"strategy": "random_users", "percent": self.percent, "seed": self.seed},
        )
        return partitions


class KTDataSplitterByUserList(_KTBaseUserSplitter):
    def __init__(
        self,
        source_partition_name: str | None = None,
        forget_partition_name: str | None = None,
        other_partition_name: str | None = None,
        user_id_field: str = "uid",
        parts_names: list[str] | None = None,
        ref_data: str | None = None,
        uid_list: list | None = None,
        uid_list_file: str | None = None,
        allow_missing_uids: bool = False,
        sort_uids_before_use: bool = True,
        correctness_field: str = "responses",
        artifact_dir: str | None = None,
    ):
        super().__init__(
            source_partition_name=source_partition_name,
            forget_partition_name=forget_partition_name,
            other_partition_name=other_partition_name,
            user_id_field=user_id_field,
            parts_names=parts_names,
            ref_data=ref_data,
            correctness_field=correctness_field,
            artifact_dir=artifact_dir,
        )
        self.uid_list = uid_list
        self.uid_list_file = uid_list_file
        self.allow_missing_uids = bool(allow_missing_uids)
        self.sort_uids_before_use = bool(sort_uids_before_use)

    def split_data(self, partitions):
        source_indices = resolve_source_indices(partitions, self.source_partition_name)
        user_stats, _ = build_user_stats(
            self.source,
            source_indices=source_indices,
            user_id_field=self.user_id_field,
            correctness_field=self.correctness_field,
        )
        requested_uids = load_uid_list(
            uid_list=self.uid_list,
            uid_list_file=self.uid_list_file,
            sort_before_use=self.sort_uids_before_use,
        )
        missing = [uid for uid in requested_uids if uid not in user_stats]
        if missing and not self.allow_missing_uids:
            raise ValueError(f"Some requested forget users are missing from source partition: {missing[:10]}")
        forget_uids = [uid for uid in requested_uids if uid in user_stats]
        forget_indices, other_indices = split_indices_by_users(source_indices, user_stats, set(forget_uids))
        if not forget_indices:
            raise ValueError("Forget set is empty after user-list split.")
        if not other_indices:
            raise ValueError("Other set is empty after user-list split.")

        partitions[self.forget_partition_name] = forget_indices
        partitions[self.other_partition_name] = other_indices
        self._save_artifacts(
            user_stats=user_stats,
            forget_user_ids=forget_uids,
            other_user_ids=sorted([uid for uid in user_stats.keys() if uid not in set(forget_uids)]),
            extra_summary={
                "strategy": "user_list",
                "allow_missing_uids": self.allow_missing_uids,
                "requested_uid_count": len(requested_uids),
                "missing_uid_count": len(missing),
            },
        )
        return partitions


class KTDataSplitterTrainRetainTest(DataSplitter):
    def __init__(
        self,
        source_partition_name: str | None = None,
        retain_partition_name: str | None = None,
        test_partition_name: str | None = None,
        mode: str = "ratio",
        user_id_field: str = "uid",
        parts_names: list[str] | None = None,
        ref_data: str | None = None,
        correctness_field: str = "responses",
        retain_ratio: float | None = None,
        fold_id_field: str | None = None,
        seed: int | None = None,
        user_level_split: bool = True,
        test_fold_ids: list[int] | None = None,
        artifact_dir: str | None = None,
    ):
        source_partition_name = source_partition_name or ref_data
        if source_partition_name is None:
            raise ValueError("source_partition_name (or ref_data) is required.")
        if parts_names is not None:
            if len(parts_names) != 2:
                raise ValueError("parts_names must contain exactly two names.")
            retain_partition_name = parts_names[0]
            test_partition_name = parts_names[1]
        if retain_partition_name is None or test_partition_name is None:
            raise ValueError("retain_partition_name/test_partition_name (or parts_names) are required.")
        super().__init__(source_partition_name, [retain_partition_name, test_partition_name])
        self.source_partition_name = source_partition_name
        self.retain_partition_name = retain_partition_name
        self.test_partition_name = test_partition_name
        self.mode = mode
        self.user_id_field = user_id_field
        self.correctness_field = correctness_field
        self.retain_ratio = retain_ratio
        self.fold_id_field = fold_id_field
        self.seed = 0 if seed is None else int(seed)
        self.user_level_split = bool(user_level_split)
        self.test_fold_ids = [int(x) for x in (test_fold_ids or [])]
        self.artifact_dir = artifact_dir

    def _artifact_dir(self) -> Path:
        if self.artifact_dir is not None:
            return Path(self.artifact_dir)
        dataset_name = getattr(self.source, "get_name", lambda: "kt_dataset")()
        return Path("resources") / "kt_splits" / dataset_name / self.__class__.__name__

    def split_data(self, partitions):
        source_indices = resolve_source_indices(partitions, self.source_partition_name)
        user_stats, _ = build_user_stats(
            self.source,
            source_indices=source_indices,
            user_id_field=self.user_id_field,
            correctness_field=self.correctness_field,
        )

        if self.mode == "ratio":
            if not self.user_level_split:
                raise ValueError("KTDataSplitterTrainRetainTest currently requires user_level_split=True for mode='ratio'.")
            retain_uids, test_uids = user_level_ratio_split(user_stats, retain_ratio=float(self.retain_ratio), seed=self.seed)
        elif self.mode == "fold_id":
            if not self.fold_id_field:
                raise ValueError("fold_id_field is required when mode='fold_id'.")
            if not self.test_fold_ids:
                raise ValueError("test_fold_ids is required when mode='fold_id'.")
            rows = self.source.aligned_rows
            retain_uids = []
            test_uids = []
            for uid, entry in user_stats.items():
                row_fold_values = {int(rows[idx][self.fold_id_field]) for idx in entry["indices"]}
                if row_fold_values & set(self.test_fold_ids):
                    test_uids.append(uid)
                else:
                    retain_uids.append(uid)
            retain_uids = sorted(retain_uids)
            test_uids = sorted(test_uids)
        else:
            raise ValueError("mode must be one of ['ratio', 'fold_id'].")

        test_indices, retain_indices = split_indices_by_users(source_indices, user_stats, set(test_uids))
        if not retain_indices:
            raise ValueError("Retain set is empty after KT retain/test split.")
        if not test_indices:
            raise ValueError("Test set is empty after KT retain/test split.")

        partitions[self.retain_partition_name] = retain_indices
        partitions[self.test_partition_name] = test_indices
        summary = {
            "source_partition_name": self.source_partition_name,
            "strategy": "train_retain_test",
            "mode": self.mode,
            "total_users": len(user_stats),
            "forget_user_count": 0,
            "retain_user_count": len(retain_uids),
            "test_user_count": len(test_uids),
            "forget_interactions": 0,
            "retain_interactions": partition_summary_for_users(user_stats, retain_uids)["interaction_count"],
            "test_interactions": partition_summary_for_users(user_stats, test_uids)["interaction_count"],
            "forget_mean_correctness": float("nan"),
            "retain_mean_correctness": partition_summary_for_users(user_stats, retain_uids)["mean_correctness"],
            "test_mean_correctness": partition_summary_for_users(user_stats, test_uids)["mean_correctness"],
            "forget_mean_interactions_per_user": float("nan"),
            "retain_mean_interactions_per_user": partition_summary_for_users(user_stats, retain_uids)["mean_interactions_per_user"],
            "test_mean_interactions_per_user": partition_summary_for_users(user_stats, test_uids)["mean_interactions_per_user"],
        }
        save_split_artifacts(
            output_dir=self._artifact_dir(),
            forget_uids=[],
            retain_uids=retain_uids,
            test_uids=test_uids,
            summary=summary,
        )
        return partitions
