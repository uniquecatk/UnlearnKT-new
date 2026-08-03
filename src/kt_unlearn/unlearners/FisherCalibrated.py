from fractions import Fraction

import torch

from kt_unlearn.core.factory_base import get_instance_kvargs
from kt_unlearn.unlearners.FisherForgetting import FisherForgetting


class FisherCalibrated(FisherForgetting):
    def init(self):
        super().init()
        params = self.local.config["parameters"]
        self.calibration_ref_data = params["calibration_ref_data"]
        self.calibration_steps = int(params["calibration_steps"])
        self.anchor_lambda = float(params["anchor_lambda"])
        self.confidence_lambda = float(params["confidence_lambda"])
        self.optimizer_cfg = params["optimizer"]

    def _run_lightweight_calibration(self):
        if self.calibration_steps <= 0:
            return

        retain_loader, _ = self.dataset.get_loader_for(self.calibration_ref_data, Fraction("0"))
        trainable_params = [p for p in self.predictor.model.parameters() if p.requires_grad]
        if not trainable_params:
            self.info("Skipping calibration because there are no trainable parameters")
            return

        optimizer = get_instance_kvargs(
            self.optimizer_cfg["class"],
            {"params": trainable_params, **self.optimizer_cfg["parameters"]},
        )
        anchors = {
            name: param.detach().clone()
            for name, param in self.predictor.model.named_parameters()
            if param.requires_grad
        }

        self.predictor.model.train()
        losses = []
        data_iter = iter(retain_loader)

        for step in range(self.calibration_steps):
            try:
                data, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(retain_loader)
                data, labels = next(data_iter)

            data, labels = data.to(self.device), labels.to(self.device)
            if hasattr(self.predictor.model, "lstm"):
                self.predictor.model.lstm.flatten_parameters()

            optimizer.zero_grad(set_to_none=True)
            _, outputs = self.predictor.model(data)
            task_loss = self.predictor.loss_fn(outputs, labels)

            target = labels[:, 0, :]
            mask = labels[:, 1, :].bool()
            if mask.any():
                prob = torch.sigmoid(outputs[mask])
                confidence_penalty = (prob - 0.5).pow(2).mean()
            else:
                confidence_penalty = torch.zeros((), device=self.device)

            anchor_penalty = torch.zeros((), device=self.device)
            for name, param in self.predictor.model.named_parameters():
                if param.requires_grad:
                    anchor_penalty = anchor_penalty + (param - anchors[name]).pow(2).mean()

            loss = (
                task_loss
                + self.anchor_lambda * anchor_penalty
                + self.confidence_lambda * confidence_penalty
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        self.info(
            "Calibration finished: "
            f"steps={self.calibration_steps}, mean_loss={sum(losses) / max(len(losses), 1):.4f}"
        )

    def __unlearn__(self):
        self.info("Starting Fisher Forgetting with Lightweight Calibration")

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction("0"))

        self.info("Computing Fisher Information Matrix")
        if self.task == "auto":
            self.compute_fisher_information(retain_loader.dataset)
        elif self.task == "multilabel":
            self.compute_fisher_information_multilabel(retain_loader.dataset)
        elif self.task == "kt":
            self.compute_fisher_information_kt(retain_loader.dataset)

        self.info("Applying Fisher noise for selective forgetting")
        self.apply_fisher_noise()

        self.info("Running lightweight retain calibration")
        self._run_lightweight_calibration()
        return self.predictor

    def check_configuration(self):
        super().check_configuration()
        params = self.local.config["parameters"]
        params["calibration_ref_data"] = params.get("calibration_ref_data", "retain")
        params["calibration_steps"] = int(params.get("calibration_steps", 16))
        params["anchor_lambda"] = float(params.get("anchor_lambda", 1e-3))
        params["confidence_lambda"] = float(params.get("confidence_lambda", 0.0))
        params["optimizer"] = params.get(
            "optimizer",
            {"class": "torch.optim.Adam", "parameters": {"lr": 1e-4}},
        )
