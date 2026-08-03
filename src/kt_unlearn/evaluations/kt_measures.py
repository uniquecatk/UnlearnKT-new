from __future__ import annotations

import copy
import time

import numpy as np
import torch

from kt_unlearn.core.measure import Measure
from kt_unlearn.core.factory_base import get_instance_kvargs
from kt_unlearn.evaluations.evaluation import Evaluation
from kt_unlearn.utils.config.local_ctx import Local


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return float("nan")
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _make_optimizer(erasure_model):
    opt_cfg = erasure_model.local_config["parameters"]["optimizer"]
    return get_instance_kvargs(
        opt_cfg["class"],
        {"params": erasure_model.model.parameters(), **opt_cfg["parameters"]},
    )


def _collect_partition_stats(erasure_model, loader):
    erasure_model.model.eval()

    losses = []
    y_true_all = []
    y_prob_all = []
    per_seq_loss_all = []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(erasure_model.model.device)
            labels = labels.to(erasure_model.model.device)
            _, pred = erasure_model.model(x)

            loss = erasure_model.loss_fn(pred, labels)
            losses.append(float(loss.detach().cpu().item()))

            target = labels[:, 0, :]
            mask = labels[:, 1, :].bool()
            if mask.numel() == 0 or not mask.any():
                continue

            prob = torch.sigmoid(pred)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
            per_seq_loss = (bce * mask.to(bce.dtype)).sum(dim=1) / (mask.sum(dim=1).to(bce.dtype) + 1e-12)
            per_seq_loss_all.extend(per_seq_loss.detach().cpu().numpy().reshape(-1).tolist())

            y_true_all.extend(target[mask].detach().cpu().numpy().reshape(-1).tolist())
            y_prob_all.extend(prob[mask].detach().cpu().numpy().reshape(-1).tolist())

    y_true = np.asarray(y_true_all, dtype=np.float64)
    y_prob = np.asarray(y_prob_all, dtype=np.float64)
    y_pred = (y_prob >= 0.5).astype(np.float64) if y_prob.size else np.asarray([], dtype=np.float64)

    return {
        "loss": _safe_mean(losses),
        "acc": float((y_true == y_pred).mean()) if y_true.size else float("nan"),
        "auc": _safe_auc(y_true, y_prob),
        "y_true": y_true,
        "y_prob": y_prob,
        "per_seq_loss": np.asarray(per_seq_loss_all, dtype=np.float64),
    }


def _relearn_time(erasure_model, loader, target_acc: float, max_epochs: int) -> int:
    model = copy.deepcopy(erasure_model)
    model.model.to(model.model.device)
    model.optimizer = _make_optimizer(model)

    current_acc = _collect_partition_stats(model, loader)["acc"]
    if current_acc == current_acc and target_acc == target_acc and current_acc >= target_acc:
        return 0

    for epoch in range(1, max_epochs + 1):
        model.model.train()
        for x, labels in loader:
            x = x.to(model.model.device)
            labels = labels.to(model.model.device)
            model.optimizer.zero_grad()
            _, pred = model.model(x)
            loss = model.loss_fn(pred, labels)
            loss.backward()
            model.optimizer.step()

        current_acc = _collect_partition_stats(model, loader)["acc"]
        if current_acc == current_acc and target_acc == target_acc and current_acc >= target_acc:
            return epoch

    return max_epochs


def _threshold_mia(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> tuple[float, float]:
    if member_scores.size == 0 or nonmember_scores.size == 0:
        return float("nan"), float("nan")

    y_true = np.concatenate([np.ones_like(member_scores), np.zeros_like(nonmember_scores)])
    scores = np.concatenate([member_scores, nonmember_scores])
    order = np.argsort(scores)
    scores_sorted = scores[order]

    best_acc = -1.0
    for th in scores_sorted:
        pred = (scores <= th).astype(np.float64)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc

    auc = _safe_auc(y_true, -scores)
    return best_acc, auc


class KTMetrics(Measure):
    def init(self):
        super().init()
        self.partition_name = self.local.config["parameters"]["partition"]
        self.target = self.local.config["parameters"]["target"]

    def check_configuration(self):
        super().check_configuration()
        self.local.config["parameters"]["partition"] = self.local.config["parameters"].get("partition", "test")
        self.local.config["parameters"]["target"] = self.local.config["parameters"].get("target", "unlearned")

    def process(self, e: Evaluation):
        erasure_model = e.predictor if self.target == "original" else e.unlearned_model
        erasure_model.model.eval()

        loader, _ = e.unlearner.dataset.get_loader_for(self.partition_name, drop_last=False)
        losses = []
        y_true_all = []
        y_prob_all = []

        with torch.no_grad():
            for x, labels in loader:
                x = x.to(erasure_model.model.device)
                labels = labels.to(erasure_model.model.device)
                _, pred = erasure_model.model(x)
                loss = erasure_model.loss_fn(pred, labels)
                losses.append(float(loss.detach().cpu().item()))

                target = labels[:, 0, :]
                mask = labels[:, 1, :].bool()
                if mask.numel() == 0 or not mask.any():
                    continue
                prob = torch.sigmoid(pred)
                y_true_all.extend(target[mask].detach().cpu().numpy().reshape(-1).tolist())
                y_prob_all.extend(prob[mask].detach().cpu().numpy().reshape(-1).tolist())

        y_true = np.asarray(y_true_all, dtype=np.float64)
        y_prob = np.asarray(y_prob_all, dtype=np.float64)
        y_pred = (y_prob >= 0.5).astype(np.float64) if y_prob.size else np.asarray([], dtype=np.float64)

        acc = float((y_true == y_pred).mean()) if y_true.size else float("nan")
        auc = _safe_auc(y_true, y_prob)
        loss = float(np.mean(losses)) if losses else float("nan")

        self.info(
            f"KT metrics partition={self.partition_name} target={self.target} loss={loss:.4f} acc={acc:.4f} auc={auc:.4f}"
        )
        e.add_value(f"kt.loss.{self.partition_name}.{self.target}", loss)
        e.add_value(f"kt.acc.{self.partition_name}.{self.target}", acc)
        e.add_value(f"kt.auc.{self.partition_name}.{self.target}", auc)
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
        original = e.predictor
        unlearned = e.unlearned_model

        test_loader, _ = e.unlearner.dataset.get_loader_for(self.test_part, drop_last=False)
        forget_loader, _ = e.unlearner.dataset.get_loader_for(self.forget_part, drop_last=False)

        original_test_acc = _collect_partition_stats(original, test_loader)["acc"]
        unlearned_test_acc = _collect_partition_stats(unlearned, test_loader)["acc"]
        unlearned_forget_acc = _collect_partition_stats(unlearned, forget_loader)["acc"]

        aus = (1.0 - (original_test_acc - unlearned_test_acc)) / (
            1.0 + abs(unlearned_test_acc - unlearned_forget_acc)
        )
        self.info(f"KT AUS: {aus}")
        e.add_value("AUS", aus)
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
        original_acc = _collect_partition_stats(e.predictor, forget_loader)["acc"]
        relearn_time = _relearn_time(e.unlearned_model, forget_loader, target_acc=original_acc, max_epochs=self.max_epochs)
        self.info(f"KT RelearnTime: {relearn_time}")
        e.add_value("RelearnTime", relearn_time)
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
        original_forget_acc = _collect_partition_stats(e.predictor, forget_loader)["acc"]
        max_accuracy = (1.0 - self.alpha) * original_forget_acc

        rt_unlearned = _relearn_time(
            e.unlearned_model, forget_loader, target_acc=max_accuracy, max_epochs=self.max_epochs
        )

        current = Local(self.gold_model_cfg)
        gold_unlearner = self.global_ctx.factory.get_object(current)
        gold_model = gold_unlearner.unlearn()
        rt_gold = _relearn_time(gold_model, forget_loader, target_acc=max_accuracy, max_epochs=self.max_epochs)

        epsilon = 0.01
        ain = (rt_unlearned + epsilon) / (rt_gold + epsilon)
        self.info(f"KT AIN: {ain}")
        e.add_value("AIN", ain)
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
        member_loader, _ = e.unlearner.dataset.get_loader_for(self.member_part, drop_last=False)
        nonmember_loader, _ = e.unlearner.dataset.get_loader_for(self.nonmember_part, drop_last=False)

        member_scores = _collect_partition_stats(e.unlearned_model, member_loader)["per_seq_loss"]
        nonmember_scores = _collect_partition_stats(e.unlearned_model, nonmember_loader)["per_seq_loss"]
        umia, umia_auc = _threshold_mia(member_scores, nonmember_scores)

        self.info(f"KT UMIA: {umia} (auc={umia_auc})")
        e.add_value("UMIA", umia)
        e.add_value("UMIA_AUC", umia_auc)
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
        acc_metric = f"kt.acc.{self.acc_split}.unlearned"

        if acc_metric not in e.data_info:
            self.info(f"Accuracy metric {acc_metric} not found in data_info. Calculating it now.")
            measure = {
                "class": "kt_unlearn.evaluations.kt_measures.KTMetrics",
                "parameters": {"partition": self.acc_split, "target": "unlearned"},
            }
            current = self.global_ctx.factory.get_object(Local(measure))
            e = current.process(e)

        if "UMIA" not in e.data_info:
            self.info("UMIA metric not found in data_info. Calculating it now.")
            measure = {
                "class": "kt_unlearn.evaluations.kt_measures.KTUMIA",
                "parameters": {
                    "member_part": self.member_part,
                    "nonmember_part": self.nonmember_part,
                },
            }
            current = self.global_ctx.factory.get_object(Local(measure))
            e = current.process(e)

        acc = e.data_info[acc_metric]
        umia = e.data_info["UMIA"]
        forget_score = abs(umia - 0.5)
        nomus = self.l * acc + (1.0 - self.l) * (1.0 - forget_score * 2.0)

        self.info(f"KT NoMUS: {nomus}")
        e.add_value("NoMUS", nomus)
        return e
