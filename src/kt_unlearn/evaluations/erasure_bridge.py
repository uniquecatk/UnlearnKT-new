from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from erasure.core.factory_base import get_function
from erasure.core.measure import Measure
from erasure.evaluations.evaluation import Evaluation
from kt_unlearn.core.factory_base import get_instance_kvargs
from kt_unlearn.utils.config.local_ctx import Local


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.size == 0 or y_score.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _safe_std(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=np.float64)
    return values.std(axis=1).astype(np.float64)


def _threshold_mia(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> tuple[float, float]:
    if member_scores.size == 0 or nonmember_scores.size == 0:
        return float("nan"), float("nan")

    y_true = np.concatenate(
        [np.ones(member_scores.shape[0], dtype=np.int64), np.zeros(nonmember_scores.shape[0], dtype=np.int64)]
    )
    scores = np.concatenate([member_scores, nonmember_scores]).astype(np.float64)

    best_acc = float("-inf")
    for threshold in np.unique(scores):
        pred = (scores <= threshold).astype(np.int64)
        best_acc = max(best_acc, float((pred == y_true).mean()))

    auc = _safe_auc(y_true.astype(np.float64), -scores)
    return best_acc, auc


def _resolve_metric_inputs(metric_func, metric_name: str, stats: "KTBatchStats") -> tuple[np.ndarray, np.ndarray, dict]:
    params = dict(stats.metric_params)
    func_name = getattr(metric_func, "__name__", "")
    metric_ref = f"{metric_name}.{func_name}".lower()

    if "roc_auc" in metric_ref:
        return stats.y_true, stats.y_prob, params

    if "log_loss" in metric_ref:
        params.setdefault("labels", [0, 1])
        return stats.y_true, np.clip(stats.y_prob, 1e-7, 1.0 - 1e-7), params

    return stats.y_true, stats.y_pred, params


@dataclass
class KTBatchStats:
    y_true: np.ndarray
    y_prob: np.ndarray
    y_pred: np.ndarray
    per_seq_loss: np.ndarray
    per_seq_acc: np.ndarray
    per_seq_confidence: np.ndarray
    per_seq_entropy: np.ndarray
    seq_lengths: np.ndarray
    loss: float
    acc: float
    auc: float
    valid_interactions: int
    sequence_count: int
    metric_params: dict

    @classmethod
    def collect(cls, erasure_model, loader) -> "KTBatchStats":
        erasure_model.model.eval()

        losses: list[float] = []
        y_true_all: list[float] = []
        y_prob_all: list[float] = []
        per_seq_loss_all: list[float] = []
        per_seq_acc_all: list[float] = []
        per_seq_conf_all: list[float] = []
        per_seq_entropy_all: list[float] = []
        seq_lengths_all: list[int] = []

        with torch.no_grad():
            for x, labels in loader:
                x = x.to(erasure_model.model.device)
                labels = labels.to(erasure_model.model.device)
                _, pred = erasure_model.model(x)

                loss = erasure_model.loss_fn(pred, labels)
                losses.append(float(loss.detach().cpu().item()))

                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                if labels.dim() == 2:
                    labels = labels.unsqueeze(0)

                target = labels[:, 0, :]
                mask = labels[:, 1, :].bool()
                if mask.numel() == 0 or not mask.any():
                    continue

                prob = torch.sigmoid(pred)
                bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
                mask_f = mask.to(bce.dtype)
                valid_per_seq = mask.sum(dim=1).clamp_min(1)
                valid_per_seq_f = valid_per_seq.to(bce.dtype)

                masked_bce = bce * mask_f
                per_seq_loss = masked_bce.sum(dim=1) / valid_per_seq_f

                pred_label = (prob >= 0.5).to(target.dtype)
                per_seq_acc = ((pred_label == target).to(mask_f.dtype) * mask_f).sum(dim=1) / valid_per_seq_f

                true_confidence = torch.where(target > 0.5, prob, 1.0 - prob)
                per_seq_confidence = (true_confidence * mask_f).sum(dim=1) / valid_per_seq_f

                entropy = -(prob.clamp(1e-7, 1.0 - 1e-7) * torch.log(prob.clamp(1e-7, 1.0 - 1e-7))
                    + (1.0 - prob).clamp(1e-7, 1.0 - 1e-7) * torch.log((1.0 - prob).clamp(1e-7, 1.0 - 1e-7)))
                per_seq_entropy = (entropy * mask_f).sum(dim=1) / valid_per_seq_f

                y_true_all.extend(target[mask].detach().cpu().numpy().reshape(-1).tolist())
                y_prob_all.extend(prob[mask].detach().cpu().numpy().reshape(-1).tolist())
                per_seq_loss_all.extend(per_seq_loss.detach().cpu().numpy().reshape(-1).tolist())
                per_seq_acc_all.extend(per_seq_acc.detach().cpu().numpy().reshape(-1).tolist())
                per_seq_conf_all.extend(per_seq_confidence.detach().cpu().numpy().reshape(-1).tolist())
                per_seq_entropy_all.extend(per_seq_entropy.detach().cpu().numpy().reshape(-1).tolist())
                seq_lengths_all.extend(valid_per_seq.detach().cpu().numpy().reshape(-1).astype(int).tolist())

        y_true = np.asarray(y_true_all, dtype=np.float64)
        y_prob = np.asarray(y_prob_all, dtype=np.float64)
        y_pred = (y_prob >= 0.5).astype(np.float64) if y_prob.size else np.asarray([], dtype=np.float64)

        return cls(
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
            per_seq_loss=np.asarray(per_seq_loss_all, dtype=np.float64),
            per_seq_acc=np.asarray(per_seq_acc_all, dtype=np.float64),
            per_seq_confidence=np.asarray(per_seq_conf_all, dtype=np.float64),
            per_seq_entropy=np.asarray(per_seq_entropy_all, dtype=np.float64),
            seq_lengths=np.asarray(seq_lengths_all, dtype=np.float64),
            loss=_safe_mean(losses),
            acc=float((y_true == y_pred).mean()) if y_true.size else float("nan"),
            auc=_safe_auc(y_true, y_prob),
            valid_interactions=int(y_true.size),
            sequence_count=int(len(per_seq_loss_all)),
            metric_params={},
        )

    def attack_features(self) -> np.ndarray:
        if self.per_seq_loss.size == 0:
            return np.empty((0, 5), dtype=np.float64)
        return np.column_stack(
            [
                self.per_seq_loss,
                1.0 - self.per_seq_acc,
                1.0 - self.per_seq_confidence,
                self.per_seq_entropy,
                np.log1p(self.seq_lengths),
            ]
        ).astype(np.float64)


def _get_stats(e: Evaluation, partition: str, target: str) -> KTBatchStats:
    erasure_model = e.predictor if target == "original" else e.unlearned_model
    loader, _ = e.unlearner.dataset.get_loader_for(partition, drop_last=False)
    return KTBatchStats.collect(erasure_model, loader)


def _make_optimizer(erasure_model):
    opt_cfg = erasure_model.local_config["parameters"]["optimizer"]
    return get_instance_kvargs(
        opt_cfg["class"],
        {"params": erasure_model.model.parameters(), **opt_cfg["parameters"]},
    )


def _relearn_time(erasure_model, loader, target_acc: float, max_epochs: int) -> tuple[int, int]:
    model = copy.deepcopy(erasure_model)
    model.model.to(model.model.device)
    model.optimizer = _make_optimizer(model)

    current_stats = KTBatchStats.collect(model, loader)
    if current_stats.acc == current_stats.acc and target_acc == target_acc and current_stats.acc >= target_acc:
        return 0, 0

    interactions_seen = 0
    for epoch in range(1, max_epochs + 1):
        model.model.train()
        for x, labels in loader:
            x = x.to(model.model.device)
            labels = labels.to(model.model.device)
            interactions_seen += int(labels[:, 1, :].sum().detach().cpu().item())

            model.optimizer.zero_grad()
            _, pred = model.model(x)
            loss = model.loss_fn(pred, labels)
            loss.backward()
            model.optimizer.step()

        current_stats = KTBatchStats.collect(model, loader)
        if current_stats.acc == current_stats.acc and target_acc == target_acc and current_stats.acc >= target_acc:
            return epoch, interactions_seen

    return max_epochs, interactions_seen


class KTTorchSKLearn(Measure):
    def init(self):
        super().init()
        self.partition_name = self.params["partition"]
        self.target = self.params["target"]
        self.metric_name = self.params["name"]
        self.metric_params = self.params["function"]["parameters"]
        self.metric_func = get_function(self.params["function"]["class"])

    def check_configuration(self):
        super().check_configuration()
        self.params["function"] = self.params.get(
            "function", {"class": "sklearn.metrics.accuracy_score", "parameters": {}}
        )
        self.params["partition"] = self.params.get("partition", "test")
        self.params["target"] = self.params.get("target", "unlearned")
        self.params["name"] = self.params.get("name", self.params["function"]["class"])

    def process(self, e: Evaluation):
        stats = _get_stats(e, self.partition_name, self.target)
        stats.metric_params = dict(self.metric_params)

        inputs_true, inputs_pred, metric_kwargs = _resolve_metric_inputs(self.metric_func, self.metric_name, stats)
        if inputs_true.size == 0:
            value = float("nan")
        else:
            value = float(self.metric_func(inputs_true, inputs_pred, **metric_kwargs))

        self.info(
            f"{self.metric_name} of \"{self.partition_name}\" on {self.target}: {value}"
        )
        e.add_value(f"{self.metric_name}.{self.partition_name}.{self.target}", value)

        # Keep KT-native diagnostics alongside ERASURE-style keys for downstream analysis.
        e.add_value(f"kt.loss.{self.partition_name}.{self.target}", stats.loss)
        e.add_value(f"kt.acc.{self.partition_name}.{self.target}", stats.acc)
        e.add_value(f"kt.auc.{self.partition_name}.{self.target}", stats.auc)

        return e


class KTAUS(Measure):
    def init(self):
        super().init()
        self.forget_part = self.params["forget_part"]
        self.test_part = self.params["test_part"]

    def check_configuration(self):
        super().check_configuration()
        self.params["forget_part"] = self.params.get("forget_part", "forget")
        self.params["test_part"] = self.params.get("test_part", "test")

    def process(self, e: Evaluation):
        original_test_acc = _get_stats(e, self.test_part, "original").acc
        unlearned_test_acc = _get_stats(e, self.test_part, "unlearned").acc
        unlearned_forget_acc = _get_stats(e, self.forget_part, "unlearned").acc

        aus = (1.0 - (original_test_acc - unlearned_test_acc)) / (
            1.0 + abs(unlearned_test_acc - unlearned_forget_acc)
        )
        e.add_value("AUS", float(aus))
        return e


class KTRelearnTime(Measure):
    def init(self):
        super().init()
        self.forget_part = self.params["forget_part"]
        self.max_epochs = int(self.params["max_epochs"])

    def check_configuration(self):
        super().check_configuration()
        self.params["forget_part"] = self.params.get("forget_part", "forget")
        self.params["max_epochs"] = int(self.params.get("max_epochs", 10))

    def process(self, e: Evaluation):
        forget_loader, _ = e.unlearner.dataset.get_loader_for(self.forget_part, drop_last=False)
        original_acc = KTBatchStats.collect(e.predictor, forget_loader).acc
        relearn_time, relearn_interactions = _relearn_time(
            e.unlearned_model,
            forget_loader,
            target_acc=original_acc,
            max_epochs=self.max_epochs,
        )
        e.add_value("RelearnTime", int(relearn_time))
        e.add_value("RelearnInteractions", int(relearn_interactions))
        return e


class KTAIN(Measure):
    def init(self):
        super().init()
        self.alpha = float(self.params["alpha"])
        self.forget_part = self.params["forget_part"]
        self.max_epochs = int(self.params["max_epochs"])
        self.gold_model_cfg = self.params["gold_model"]

    def check_configuration(self):
        super().check_configuration()
        self.params["alpha"] = float(self.params.get("alpha", 0.05))
        self.params["forget_part"] = self.params.get("forget_part", "forget")
        self.params["max_epochs"] = int(self.params.get("max_epochs", 10))
        self.params["gold_model"] = self.params.get(
            "gold_model",
            {
                "class": "kt_unlearn.unlearners.GoldModel.GoldModel",
                "parameters": {"training_set": "retain", "cached": False},
            },
        )

    def process(self, e: Evaluation):
        forget_loader, _ = e.unlearner.dataset.get_loader_for(self.forget_part, drop_last=False)
        original_forget_acc = KTBatchStats.collect(e.predictor, forget_loader).acc
        max_accuracy = (1.0 - self.alpha) * original_forget_acc

        rt_unlearned, rt_unlearned_interactions = _relearn_time(
            e.unlearned_model, forget_loader, target_acc=max_accuracy, max_epochs=self.max_epochs
        )

        current = Local(self.gold_model_cfg)
        gold_unlearner = self.global_ctx.factory.get_object(current)
        gold_model = gold_unlearner.unlearn()
        rt_gold, rt_gold_interactions = _relearn_time(
            gold_model, forget_loader, target_acc=max_accuracy, max_epochs=self.max_epochs
        )

        epsilon = 0.01
        ain = (rt_unlearned + epsilon) / (rt_gold + epsilon)
        e.add_value("AIN", float(ain))
        e.add_value("AIN_unlearned_interactions", int(rt_unlearned_interactions))
        e.add_value("AIN_gold_interactions", int(rt_gold_interactions))
        return e


class KTPartitionInfo(Measure):
    def init(self):
        super().init()
        self.partition_name = self.params["partition"]

    def check_configuration(self):
        super().check_configuration()
        self.params["partition"] = self.params.get("partition", "forget")

    def process(self, e: Evaluation):
        partition = e.unlearner.dataset.partitions[self.partition_name]
        rows = getattr(e.unlearner.dataset.datasource, "aligned_rows", [])
        unique_users = set()
        for idx in partition:
            if idx < len(rows):
                uid = rows[idx].get("uid")
                if uid is not None:
                    unique_users.add(str(uid))

        stats = _get_stats(e, self.partition_name, "unlearned")
        info = {
            "name": self.partition_name,
            "size": int(len(partition)),
            "sequence_count": int(stats.sequence_count),
            "user_count": int(len(unique_users) if unique_users else len(partition)),
            "interaction_count": int(stats.valid_interactions),
            "mean_correctness": float(stats.y_true.mean()) if stats.y_true.size else float("nan"),
            "mean_interactions_per_sequence": (
                float(stats.valid_interactions / max(stats.sequence_count, 1))
                if stats.sequence_count
                else float("nan")
            ),
        }
        e.add_value(f"part_info.{self.partition_name}", info)
        return e


class KTUMIA(Measure):
    def init(self):
        super().init()
        self.member_part = self.params["member_part"]
        self.nonmember_part = self.params["nonmember_part"]

    def check_configuration(self):
        super().check_configuration()
        self.params["member_part"] = self.params.get("member_part", "forget")
        self.params["nonmember_part"] = self.params.get("nonmember_part", "test")

    def process(self, e: Evaluation):
        member_stats = _get_stats(e, self.member_part, "unlearned")
        nonmember_stats = _get_stats(e, self.nonmember_part, "unlearned")

        member_features = member_stats.attack_features()
        nonmember_features = nonmember_stats.attack_features()
        sample_size = min(member_features.shape[0], nonmember_features.shape[0])
        if sample_size < 2:
            e.add_value("UMIA", float("nan"))
            e.add_value("UMIA_AUC", float("nan"))
            e.add_value("UMIA_threshold", float("nan"))
            e.add_value("UMIA_threshold_AUC", float("nan"))
            return e

        x = np.vstack([member_features[:sample_size], nonmember_features[:sample_size]])
        y = np.concatenate(
            [np.ones(sample_size, dtype=np.int64), np.zeros(sample_size, dtype=np.int64)]
        )

        n_splits = min(5, sample_size)
        if n_splits < 2:
            umia = float("nan")
            umia_auc = float("nan")
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.global_ctx.config.globals["seed"])
            attack_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, random_state=self.global_ctx.config.globals["seed"]),
            )
            prob_member = cross_val_predict(attack_model, x, y, cv=cv, method="predict_proba")[:, 1]
            pred_member = (prob_member >= 0.5).astype(np.int64)
            umia = float(accuracy_score(y, pred_member))
            umia_auc = _safe_auc(y.astype(np.float64), prob_member.astype(np.float64))

        threshold_acc, threshold_auc = _threshold_mia(
            member_stats.per_seq_loss[:sample_size], nonmember_stats.per_seq_loss[:sample_size]
        )

        e.add_value("UMIA", umia)
        e.add_value("UMIA_AUC", umia_auc)
        e.add_value("UMIA_threshold", threshold_acc)
        e.add_value("UMIA_threshold_AUC", threshold_auc)
        return e


class KTNoMUS(Measure):
    def init(self):
        super().init()
        self.l = float(self.params["l"])
        self.acc_split = self.params["acc_split"]
        self.member_part = self.params["member_part"]
        self.nonmember_part = self.params["nonmember_part"]

    def check_configuration(self):
        super().check_configuration()
        self.params["l"] = float(self.params.get("l", 0.5))
        self.params["acc_split"] = self.params.get("acc_split", "test")
        self.params["member_part"] = self.params.get("member_part", "forget")
        self.params["nonmember_part"] = self.params.get("nonmember_part", "test")

    def process(self, e: Evaluation):
        acc_metric = f"sklearn.metrics.accuracy_score.{self.acc_split}.unlearned"
        if acc_metric not in e.data_info:
            current = self.global_ctx.factory.get_object(
                Local(
                    {
                        "class": "kt_unlearn.evaluations.erasure_bridge.KTTorchSKLearn",
                        "parameters": {"partition": self.acc_split, "target": "unlearned"},
                    }
                )
            )
            e = current.process(e)

        if "UMIA" not in e.data_info:
            current = self.global_ctx.factory.get_object(
                Local(
                    {
                        "class": "kt_unlearn.evaluations.erasure_bridge.KTUMIA",
                        "parameters": {
                            "member_part": self.member_part,
                            "nonmember_part": self.nonmember_part,
                        },
                    }
                )
            )
            e = current.process(e)

        acc = float(e.data_info[acc_metric])
        umia = float(e.data_info["UMIA"])
        forget_score = abs(umia - 0.5)
        nomus = self.l * acc + (1.0 - self.l) * (1.0 - forget_score * 2.0)
        e.add_value("NoMUS", float(nomus))
        return e


class KTSaveValues(Measure):
    def init(self):
        super().init()
        self.path = self.params["path"]
        self.output_format = self.local_config["parameters"].get("output_format", self.path.split(".")[-1])
        self.include_keys = self.local_config["parameters"].get("include_keys")
        self.exclude_prefixes = self.local_config["parameters"].get("exclude_prefixes", [])
        self.column_order = self.local_config["parameters"].get("column_order", [])

        valid_extensions = {"json": ".json", "csv": ".csv", "yaml": ".yaml", "xlsx": ".xlsx"}
        if self.output_format not in valid_extensions:
            self.info(f"Unsupported output format: {self.output_format}, defaulting to json")
            self.output_format = "json"
        if not self.path.endswith(valid_extensions[self.output_format]):
            self.info(
                f"Path '{self.path}' does not match output format '{self.output_format}', defaulting to json"
            )
            self.output_format = "json"
            self.path = "".join(self.path.split(".")[:-1]) + ".json"

    def process(self, e: Evaluation):
        if self.output_format == "json":
            self._process_json(e)
        elif self.output_format == "csv":
            self._process_csv(e)
        elif self.output_format == "yaml":
            self._process_yaml(e)
        elif self.output_format == "xlsx":
            self._process_excel(e)
        return e

    def _process_json(self, e: Evaluation):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as json_file:
            json.dump(self._prepare_data(e.data_info), json_file, indent=2)
            json_file.write(",")

    def _process_csv(self, e: Evaluation):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df = pd.DataFrame.from_dict([self._prepare_data(e.data_info)])
        if self.column_order:
            for col in self.column_order:
                if col not in df.columns:
                    df[col] = np.nan
            remaining = [col for col in df.columns if col not in self.column_order]
            df = df[self.column_order + remaining]
        if not pd.io.common.file_exists(self.path):
            df.to_csv(self.path, mode="w", index=False)
        else:
            df.to_csv(self.path, mode="a", index=False, header=False)

    def _process_excel(self, e: Evaluation):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        df = pd.DataFrame.from_dict([self._prepare_data(e.data_info)])
        if self.column_order:
            for col in self.column_order:
                if col not in df.columns:
                    df[col] = np.nan
            remaining = [col for col in df.columns if col not in self.column_order]
            df = df[self.column_order + remaining]
        if not os.path.exists(self.path):
            df.to_excel(self.path, index=False, engine="openpyxl")
        else:
            with pd.ExcelWriter(self.path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                sheet_name = "Sheet1"
                startrow = writer.sheets[sheet_name].max_row
                df.to_excel(writer, index=False, header=False, startrow=startrow)

    def _process_yaml(self, e: Evaluation):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        flat_data = self._prepare_data(e.data_info)
        with open(self.path, "a", encoding="utf-8") as yaml_file:
            yaml.dump(flat_data, yaml_file, default_flow_style=False, allow_unicode=False)

    def _prepare_data(self, data: dict) -> dict:
        flat = self._flatten_dict(data)
        if self.include_keys is not None:
            flat = {key: value for key, value in flat.items() if key in self.include_keys}
        if self.exclude_prefixes:
            flat = {
                key: value
                for key, value in flat.items()
                if not any(key.startswith(prefix) for prefix in self.exclude_prefixes)
            }
        return flat

    def _flatten_dict(self, data: dict, parent_key: str = "", sep: str = ".") -> dict:
        items = []
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, sep=sep).items())
            else:
                items.append((new_key, value))
        return dict(items)
