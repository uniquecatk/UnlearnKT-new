from fractions import Fraction
import copy

import torch

from kt_unlearn.unlearners.FisherForgetting import FisherForgetting


class QEFUKT(FisherForgetting):
    """
    KT-specific Fisher Forgetting.

    The key idea is to keep the Fisher-noise backbone, but bias the injected noise
    toward question representations that are highly exposed in the forget set.
    For DKT-like KT models this mainly means:
    - rows in interaction embeddings corresponding to forget-heavy questions
    - rows/bias entries in the question prediction head
    """

    def init(self):
        super().init()
        params = self.local.config["parameters"]
        self.ref_data_forget = params["ref_data_forget"]
        self.target_strength = float(params["target_strength"])
        self.background_strength = float(params["background_strength"])
        self.exposure_power = float(params["exposure_power"])
        self.retain_smoothing = float(params["retain_smoothing"])
        self.min_question_weight = float(params["min_question_weight"])
        self.top_question_fraction = float(params["top_question_fraction"])
        self.ascent_steps = int(params["ascent_steps"])
        self.ascent_lr = float(params["ascent_lr"])
        self.ascent_batches = int(params["ascent_batches"])
        self.max_param_shift = float(params["max_param_shift"])
        self.use_sign_updates = bool(params["use_sign_updates"])
        self.question_weights = None
        self.question_mask = None

    def _flatten_if_supported(self):
        flatten_fn = getattr(self.predictor.model, "flatten_parameters", None)
        if callable(flatten_fn):
            flatten_fn()

    def _safe_question_count(self):
        num_questions = getattr(self.predictor.model, "num_questions", None)
        if num_questions is None:
            out_layer = getattr(self.predictor.model, "out", None)
            if out_layer is not None and hasattr(out_layer, "out_features"):
                num_questions = int(out_layer.out_features)
        if num_questions is None:
            raise ValueError("KT-specific Fisher requires a model with num_questions or an output head.")
        return int(num_questions)

    def _collect_question_counts(self, loader):
        num_questions = self._safe_question_count()
        counts = torch.zeros(num_questions, dtype=torch.float64)

        for x, labels in loader:
            q_in = x[:, 0, :].long()
            q_next = x[:, 2, :].long()
            mask = labels[:, 1, :].bool()

            valid_q_in = q_in[mask]
            valid_q_next = q_next[mask]

            for questions in (valid_q_in, valid_q_next):
                if questions.numel() == 0:
                    continue
                questions = questions[(questions >= 0) & (questions < num_questions)]
                if questions.numel() == 0:
                    continue
                counts += torch.bincount(questions.cpu(), minlength=num_questions).to(dtype=torch.float64)

        return counts

    def _build_question_weights(self):
        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction("0"), drop_last=False)
        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction("0"), drop_last=False)

        forget_counts = self._collect_question_counts(forget_loader)
        retain_counts = self._collect_question_counts(retain_loader)

        exposure = (forget_counts + 1.0) / (retain_counts + self.retain_smoothing)
        if forget_counts.max().item() > 0:
            exposure = exposure * (forget_counts / forget_counts.max().clamp_min(1.0))
        exposure = exposure.pow(self.exposure_power)

        if exposure.max().item() > 0:
            exposure = exposure / exposure.max()
        exposure = exposure.clamp(min=self.min_question_weight, max=1.0)

        self.question_weights = exposure.to(device=self.device, dtype=torch.float32)
        self.info(
            "KT question-aware Fisher weights prepared: "
            f"questions={len(self.question_weights)}, "
            f"mean={self.question_weights.mean().item():.4f}, "
            f"max={self.question_weights.max().item():.4f}"
        )

    def _parameter_scale(self, name, param):
        scale = torch.full_like(param.data0, float(self.background_strength))

        if self.question_weights is None:
            return scale

        question_weights = self.question_weights.to(param.device, dtype=param.data0.dtype)

        if name.endswith("interaction_emb.weight") and param.data0.ndim == 2:
            num_questions = question_weights.numel()
            interaction_scale = torch.full_like(param.data0, float(self.background_strength))
            interaction_scale[:num_questions] = self.background_strength + self.target_strength * question_weights.unsqueeze(1)
            interaction_scale[num_questions : num_questions * 2] = (
                self.background_strength + self.target_strength * question_weights.unsqueeze(1)
            )
            return interaction_scale

        if name.endswith("out.weight") and param.data0.ndim == 2 and param.data0.size(0) == question_weights.numel():
            return self.background_strength + self.target_strength * question_weights.unsqueeze(1)

        if name.endswith("out.bias") and param.data0.ndim == 1 and param.data0.numel() == question_weights.numel():
            return self.background_strength + self.target_strength * question_weights

        return scale

    def _build_question_mask(self):
        if self.question_weights is None:
            raise ValueError("Question weights must be built before question mask.")

        q_weights = self.question_weights.detach()
        num_questions = q_weights.numel()
        positive_idx = torch.nonzero(q_weights > self.min_question_weight, as_tuple=False).flatten()
        if positive_idx.numel() == 0:
            keep_count = max(1, int(num_questions * self.top_question_fraction))
            threshold = torch.topk(q_weights, k=keep_count).values[-1]
            mask = (q_weights >= threshold)
        else:
            keep_count = max(1, int(positive_idx.numel() * self.top_question_fraction))
            positive_weights = q_weights[positive_idx]
            threshold = torch.topk(positive_weights, k=keep_count).values[-1]
            mask = torch.zeros_like(q_weights, dtype=torch.bool)
            chosen_positive = positive_idx[positive_weights >= threshold]
            mask[chosen_positive] = True

        self.question_mask = mask.to(dtype=torch.float32, device=self.device)
        self.info(
            "KT targeted question mask prepared: "
            f"selected={int(self.question_mask.sum().item())}/{num_questions}, "
            f"threshold={float(threshold):.6f}"
        )

    def _question_parameter_mask(self, name, param):
        if self.question_mask is None:
            return None

        question_mask = self.question_mask.to(param.device, dtype=param.dtype)
        num_questions = question_mask.numel()

        if name.endswith("interaction_emb.weight") and param.ndim == 2:
            mask = torch.zeros_like(param)
            mask[:num_questions] = question_mask.unsqueeze(1)
            mask[num_questions : num_questions * 2] = question_mask.unsqueeze(1)
            return mask

        if name.endswith("out.weight") and param.ndim == 2 and param.size(0) == num_questions:
            return question_mask.unsqueeze(1).expand_as(param)

        if name.endswith("out.bias") and param.ndim == 1 and param.numel() == num_questions:
            return question_mask

        return None

    def _targeted_forget_ascent(self):
        if self.ascent_steps <= 0 or self.question_mask is None:
            return

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction("0"), drop_last=False)
        reference_state = {
            name: param.detach().clone()
            for name, param in self.predictor.model.named_parameters()
            if param.requires_grad
        }

        self.predictor.model.train()
        for _ in range(self.ascent_steps):
            batch_count = 0
            grad_sums = {}
            for x, labels in forget_loader:
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
                    param_mask = self._question_parameter_mask(name, param)
                    if param_mask is None:
                        continue
                    if name not in grad_sums:
                        grad_sums[name] = torch.zeros_like(param)
                    grad_sums[name] += param.grad.detach() * param_mask

                batch_count += 1
                if batch_count >= self.ascent_batches:
                    break

            if batch_count == 0:
                continue

            with torch.no_grad():
                for name, param in self.predictor.model.named_parameters():
                    if name not in grad_sums:
                        continue
                    avg_grad = grad_sums[name] / float(batch_count)
                    update = avg_grad.sign() if self.use_sign_updates else avg_grad
                    update = self.ascent_lr * update
                    param.add_(update)

                    ref = reference_state[name].to(param.device)
                    delta = (param - ref).clamp(min=-self.max_param_shift, max=self.max_param_shift)
                    param.copy_(ref + delta)

    def apply_fisher_noise(self):
        """
        Apply Fisher noise with KT-aware question scaling.
        """
        for name, p in self.predictor.model.named_parameters():
            if not isinstance(p, torch.Tensor):
                continue

            if not hasattr(p, "data0"):
                p.data0 = copy.deepcopy(p.data.clone())

            if not hasattr(p, "grad2_acc") or not isinstance(p.grad2_acc, torch.Tensor):
                continue

            mu, var = self.get_mean_var(p)
            scale = self._parameter_scale(name, p)
            sampled_noise = var.sqrt() * torch.empty_like(p.data0).normal_()
            p.data = mu + scale * sampled_noise

    def __unlearn__(self):
        self.info("Starting KT-specific Fisher Forgetting")

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction("0"))

        self._build_question_weights()
        self._build_question_mask()

        self.info("Computing Fisher Information Matrix")
        if self.task == "auto":
            self.compute_fisher_information(retain_loader.dataset)
        elif self.task == "multilabel":
            self.compute_fisher_information_multilabel(retain_loader.dataset)
        elif self.task == "kt":
            self.compute_fisher_information_kt(retain_loader.dataset)

        self.info("Applying KT-aware Fisher noise")
        self.apply_fisher_noise()
        self.info("Running targeted forget ascent on KT question rows")
        self._targeted_forget_ascent()
        return self.predictor

    def check_configuration(self):
        super().check_configuration()
        params = self.local.config["parameters"]
        params["ref_data_forget"] = params.get("ref_data_forget", "forget")
        params["target_strength"] = float(params.get("target_strength", 1.5))
        params["background_strength"] = float(params.get("background_strength", 0.2))
        params["exposure_power"] = float(params.get("exposure_power", 1.5))
        params["retain_smoothing"] = float(params.get("retain_smoothing", 10.0))
        params["min_question_weight"] = float(params.get("min_question_weight", 0.05))
        params["top_question_fraction"] = float(params.get("top_question_fraction", 0.1))
        params["ascent_steps"] = int(params.get("ascent_steps", 2))
        params["ascent_lr"] = float(params.get("ascent_lr", 0.02))
        params["ascent_batches"] = int(params.get("ascent_batches", 8))
        params["max_param_shift"] = float(params.get("max_param_shift", 0.5))
        params["use_sign_updates"] = bool(params.get("use_sign_updates", True))
