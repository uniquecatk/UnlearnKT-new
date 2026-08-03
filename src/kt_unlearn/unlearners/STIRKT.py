import copy
from fractions import Fraction

import torch

from kt_unlearn.core.factory_base import get_instance_kvargs
from kt_unlearn.unlearners.torchunlearner import TorchUnlearner


class STIRKT(TorchUnlearner):
    """
    Selective Temporal Influence Repair for Knowledge Tracing.

    The method performs:
    1. KT-aware Fisher scoring on forget/retain partitions
    2. masked gradient-ascent updates on the forget partition
    3. repair on the retain partition with a light anchor to the original model
    """

    def init(self):
        super().init()

        params = self.local.config["parameters"]
        self.ref_data_retain = params["ref_data_retain"]
        self.ref_data_forget = params["ref_data_forget"]
        self.mask_ratio = float(params["mask_ratio"])
        self.alpha = float(params["alpha"])
        self.forget_steps = int(params["forget_steps"])
        self.repair_epochs = int(params["repair_epochs"])
        self.repair_weight = float(params["repair_weight"])
        self.anchor_weight = float(params["anchor_weight"])
        self.importance_power = float(params["importance_power"])
        self.grad_clip = float(params["grad_clip"])
        self.min_score = float(params["min_score"])
        self.keep_head = bool(params["keep_head"])
        self.interleave_repair = bool(params["interleave_repair"])
        self.step_repair_epochs = int(params["step_repair_epochs"])
        self.max_update_norm = float(params["max_update_norm"])

        self.reference_state = {
            name: param.detach().clone().cpu()
            for name, param in self.predictor.model.named_parameters()
            if param.requires_grad
        }

        self.forget_optimizer_cfg = copy.deepcopy(params["forget_optimizer"])
        self.repair_optimizer_cfg = copy.deepcopy(params["repair_optimizer"])

    def _make_optimizer(self, optimizer_cfg):
        return get_instance_kvargs(
            optimizer_cfg["class"],
            {"params": self.predictor.model.parameters(), **optimizer_cfg["parameters"]},
        )

    def _flatten_if_supported(self):
        flatten_fn = getattr(self.predictor.model, "flatten_parameters", None)
        if callable(flatten_fn):
            flatten_fn()

    def _zero_like_params(self):
        return {
            name: torch.zeros_like(param, device=param.device)
            for name, param in self.predictor.model.named_parameters()
            if param.requires_grad
        }

    def _compute_importance(self, loader):
        scores = self._zero_like_params()
        self.predictor.model.train()

        batches = 0
        for x, labels in loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            self._flatten_if_supported()
            self.predictor.model.zero_grad(set_to_none=True)
            _, pred = self.predictor.model(x)
            loss = self.predictor.loss_fn(pred, labels)
            loss.backward()

            for name, param in self.predictor.model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                scores[name] += param.grad.detach().pow(2)

            batches += 1

        if batches == 0:
            return scores

        for name in scores:
            scores[name] /= float(batches)
        return scores

    def _build_masks(self, fisher_retain, fisher_forget):
        raw_scores = {}
        merged = []

        for name, param in self.predictor.model.named_parameters():
            if not param.requires_grad:
                continue
            retain_score = fisher_retain[name].detach()
            forget_score = fisher_forget[name].detach()
            score = forget_score / (retain_score + 1e-8)
            score = score.clamp(min=self.min_score).pow(self.importance_power)
            raw_scores[name] = score
            merged.append(score.reshape(-1))

        if not merged:
            return {}, {}

        merged_scores = torch.cat(merged)
        keep_count = max(1, int(merged_scores.numel() * self.mask_ratio))
        threshold = torch.topk(merged_scores, k=keep_count).values[-1]

        masks = {}
        normalized = {}
        for name, score in raw_scores.items():
            mask = (score >= threshold).to(score.dtype)
            if self.keep_head and ("out" in name or "readout" in name):
                mask = torch.ones_like(mask)
            masks[name] = mask
            normalized[name] = score / (score.mean() + 1e-8)
        return masks, normalized

    def _restore_unmasked(self, masks):
        with torch.no_grad():
            for name, param in self.predictor.model.named_parameters():
                if name not in masks:
                    continue
                ref = self.reference_state[name].to(param.device)
                mask = masks[name].to(param.device)
                param.data.copy_(mask * param.data + (1.0 - mask) * ref)

    def _anchor_penalty(self, masks):
        penalty = None
        for name, param in self.predictor.model.named_parameters():
            if name not in masks:
                continue
            ref = self.reference_state[name].to(param.device)
            mask = masks[name].to(param.device)
            term = ((param - ref) * mask).pow(2).mean()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return torch.zeros((), device=self.device)
        return penalty

    def _masked_forget_step(self, forget_loader, masks, scores, fisher_retain):
        self.predictor.model.train()
        grad_sums = self._zero_like_params()
        batches = 0

        for x, labels in forget_loader:
            x = x.to(self.device)
            labels = labels.to(self.device)
            self._flatten_if_supported()
            self.predictor.model.zero_grad(set_to_none=True)
            _, pred = self.predictor.model(x)
            loss = self.predictor.loss_fn(pred, labels)
            loss.backward()

            for name, param in self.predictor.model.named_parameters():
                if name not in grad_sums or param.grad is None:
                    continue
                grad_sums[name] += param.grad.detach()
            batches += 1

        if batches == 0:
            return

        with torch.no_grad():
            for name, param in self.predictor.model.named_parameters():
                if name not in masks:
                    continue
                avg_grad = grad_sums[name] / float(batches)
                scale = fisher_retain[name].to(param.device).sqrt().clamp(min=1e-6)
                mask = masks[name].to(param.device)
                score = scores[name].to(param.device)
                update = self.alpha * mask * score * (avg_grad / scale)
                if self.max_update_norm > 0:
                    update = update.clamp(min=-self.max_update_norm, max=self.max_update_norm)
                param.add_(update)

    def _repair_pass(self, retain_loader, masks, epochs=None):
        optimizer = self._make_optimizer(self.repair_optimizer_cfg)
        self.predictor.model.train()

        total_epochs = self.repair_epochs if epochs is None else int(epochs)
        for _ in range(total_epochs):
            for x, labels in retain_loader:
                x = x.to(self.device)
                labels = labels.to(self.device)
                self._flatten_if_supported()
                optimizer.zero_grad(set_to_none=True)
                _, pred = self.predictor.model(x)
                retain_loss = self.predictor.loss_fn(pred, labels)
                anchor = self._anchor_penalty(masks)
                loss = self.repair_weight * retain_loss + self.anchor_weight * anchor
                loss.backward()

                for name, param in self.predictor.model.named_parameters():
                    if name in masks and param.grad is not None:
                        param.grad.mul_(masks[name].to(param.device))

                torch.nn.utils.clip_grad_norm_(self.predictor.model.parameters(), self.grad_clip)
                optimizer.step()

    def __unlearn__(self):
        self.info("Starting STIR-KT")

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction("0"))
        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction("0"))

        fisher_retain = self._compute_importance(retain_loader)
        fisher_forget = self._compute_importance(forget_loader)
        masks, scores = self._build_masks(fisher_retain, fisher_forget)

        for _ in range(self.forget_steps):
            self._masked_forget_step(forget_loader, masks, scores, fisher_retain)
            if self.interleave_repair and self.step_repair_epochs > 0:
                self._repair_pass(retain_loader, masks, epochs=self.step_repair_epochs)

        if not self.interleave_repair or self.repair_epochs > 0:
            final_epochs = 0 if self.interleave_repair else self.repair_epochs
            if final_epochs > 0:
                self._repair_pass(retain_loader, masks, epochs=final_epochs)
        self._restore_unmasked(masks)

        self.info("STIR-KT completed")
        return self.predictor

    def check_configuration(self):
        super().check_configuration()
        params = self.local.config["parameters"]
        params["ref_data_retain"] = params.get("ref_data_retain", "retain")
        params["ref_data_forget"] = params.get("ref_data_forget", "forget")
        params["mask_ratio"] = float(params.get("mask_ratio", 0.02))
        params["alpha"] = float(params.get("alpha", 0.001))
        params["forget_steps"] = int(params.get("forget_steps", 2))
        params["repair_epochs"] = int(params.get("repair_epochs", 2))
        params["repair_weight"] = float(params.get("repair_weight", 1.0))
        params["anchor_weight"] = float(params.get("anchor_weight", 1e-4))
        params["importance_power"] = float(params.get("importance_power", 1.0))
        params["grad_clip"] = float(params.get("grad_clip", 5.0))
        params["min_score"] = float(params.get("min_score", 0.0))
        params["keep_head"] = bool(params.get("keep_head", True))
        params["interleave_repair"] = bool(params.get("interleave_repair", True))
        params["step_repair_epochs"] = int(params.get("step_repair_epochs", 1))
        params["max_update_norm"] = float(params.get("max_update_norm", 0.1))
        params["forget_optimizer"] = params.get(
            "forget_optimizer",
            {"class": "torch.optim.SGD", "parameters": {"lr": params["alpha"]}},
        )
        params["repair_optimizer"] = params.get(
            "repair_optimizer",
            {"class": "torch.optim.Adam", "parameters": {"lr": 1e-4}},
        )
