from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from kt_unlearn.core.factory_base import get_instance_kvargs
from kt_unlearn.core.trainable_base import Trainable
from kt_unlearn.utils.cfg_utils import init_dflts_to_of


def _unpack_kt_batch(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_in = x[:, 0, :].long()
    r_in = x[:, 1, :].long().clamp(min=0, max=1)
    q_next = x[:, 2, :].long()
    return q_in, r_in, q_next


def _safe_questions(q: torch.Tensor) -> torch.Tensor:
    return torch.clamp(q, min=0)


def _causal_mask(seq_len: int, device: torch.device | str) -> torch.Tensor:
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def _num_heads_for(emb_size: int, requested: int) -> int:
    for candidate in [requested, 4, 2, 1]:
        if candidate > 0 and emb_size % candidate == 0:
            return candidate
    return 1


class _TransformerBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size * 4, emb_size),
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor):
        attn_out, _ = self.attn(query, key, value, attn_mask=attn_mask)
        x = self.norm1(query + self.dropout(attn_out))
        ff = self.ffn(x)
        return self.norm2(x + self.dropout(ff))


class _CausalSelfAttentionStack(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, num_blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlock(emb_size, _num_heads_for(emb_size, num_heads), dropout) for _ in range(int(num_blocks))]
        )

    def forward(self, x: torch.Tensor):
        mask = _causal_mask(x.size(1), x.device)
        for block in self.blocks:
            x = block(x, x, x, mask)
        return x


class _CausalCrossAttentionStack(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, num_blocks: int, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TransformerBlock(emb_size, _num_heads_for(emb_size, num_heads), dropout) for _ in range(int(num_blocks))]
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor):
        mask = _causal_mask(query.size(1), query.device)
        for block in self.blocks:
            query = block(query, memory, memory, mask)
        return query


class DKTSequenceModel(nn.Module):
    def __init__(self, num_questions: int, emb_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_questions = int(num_questions)
        self.emb_size = int(emb_size)
        self.interaction_emb = nn.Embedding(self.num_questions * 2, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.emb_size, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Linear(self.emb_size, self.num_questions)
        self.device = "cpu"

    def forward(self, x: torch.Tensor):
        q_in, r_in, q_next = _unpack_kt_batch(x)
        q_in_safe = _safe_questions(q_in)
        q_next_safe = _safe_questions(q_next)
        interaction = q_in_safe + self.num_questions * r_in
        emb = self.interaction_emb(interaction)
        h, _ = self.lstm(emb)
        h = self.dropout(h)
        logits_all = self.out(h)
        logits_next = torch.gather(logits_all, dim=-1, index=q_next_safe.unsqueeze(-1)).squeeze(-1)
        return h, logits_next

    def flatten_parameters(self):
        self.lstm.flatten_parameters()


class DKTPlusSequenceModel(nn.Module):
    def __init__(self, num_questions: int, emb_size: int = 64, hidden_size: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.num_questions = int(num_questions)
        self.emb_size = int(emb_size)
        self.hidden_size = int(hidden_size or emb_size)
        self.interaction_emb = nn.Embedding(self.num_questions * 2, self.emb_size)
        self.lstm = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.hidden_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.num_questions)
        self.device = "cpu"

    def forward(self, x: torch.Tensor):
        q_in, r_in, q_next = _unpack_kt_batch(x)
        q_in_safe = _safe_questions(q_in)
        q_next_safe = _safe_questions(q_next)
        interaction = q_in_safe + self.num_questions * r_in
        emb = self.interaction_emb(interaction)
        h, _ = self.lstm(emb)
        h = self.dropout(torch.relu(self.hidden_proj(h)))
        logits_all = self.out(h)
        logits_next = torch.gather(logits_all, dim=-1, index=q_next_safe.unsqueeze(-1)).squeeze(-1)
        return h, logits_next

    def flatten_parameters(self):
        self.lstm.flatten_parameters()


class SAKTSequenceModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        emb_size: int = 64,
        max_seq_len: int = 100,
        num_attn_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_questions = int(num_questions)
        self.emb_size = int(emb_size)
        self.max_seq_len = int(max_seq_len)
        self.interaction_emb = nn.Embedding(self.num_questions * 2, self.emb_size)
        self.exercise_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.position_emb = nn.Embedding(self.max_seq_len, self.emb_size)
        self.blocks = nn.ModuleList(
            [_TransformerBlock(self.emb_size, int(num_attn_heads), float(dropout)) for _ in range(int(num_blocks))]
        )
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Linear(self.emb_size, 1)
        self.device = "cpu"

    def forward(self, x: torch.Tensor):
        q_in, r_in, q_next = _unpack_kt_batch(x)
        q_in_safe = _safe_questions(q_in)
        q_next_safe = _safe_questions(q_next)
        seq_len = q_in_safe.size(1)

        interaction = q_in_safe + self.num_questions * r_in
        xemb = self.interaction_emb(interaction)
        qemb = self.exercise_emb(q_next_safe)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        xemb = xemb + self.position_emb(positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)

        h = qemb
        for block in self.blocks:
            h = block(h, xemb, xemb, mask)
        logits_next = self.out(self.dropout(h)).squeeze(-1)
        return h, logits_next


class DKVMNSequenceModel(nn.Module):
    def __init__(self, num_questions: int, emb_size: int = 64, memory_size: int = 50, dropout: float = 0.2):
        super().__init__()
        self.num_questions = int(num_questions)
        self.emb_size = int(emb_size)
        self.memory_size = int(memory_size)
        self.k_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.v_emb = nn.Embedding(self.num_questions * 2, self.emb_size)
        self.memory_key = nn.Parameter(torch.randn(self.memory_size, self.emb_size) * 0.1)
        self.memory_value = nn.Parameter(torch.randn(self.memory_size, self.emb_size) * 0.1)
        self.erase_layer = nn.Linear(self.emb_size, self.emb_size)
        self.add_layer = nn.Linear(self.emb_size, self.emb_size)
        self.read_layer = nn.Linear(self.emb_size * 2, self.emb_size)
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Linear(self.emb_size, 1)
        self.device = "cpu"

    def forward(self, x: torch.Tensor):
        q_in, r_in, q_next = _unpack_kt_batch(x)
        q_in_safe = _safe_questions(q_in)
        q_next_safe = _safe_questions(q_next)
        batch_size, seq_len = q_in_safe.shape

        write_key = self.k_emb(q_in_safe)
        write_value = self.v_emb(q_in_safe + self.num_questions * r_in)
        query_key = self.k_emb(q_next_safe)
        memory = self.memory_value.unsqueeze(0).expand(batch_size, -1, -1).clone()
        hidden_states = []

        for t in range(seq_len):
            w_read = torch.softmax(torch.matmul(query_key[:, t, :], self.memory_key.T), dim=-1)
            read_content = torch.sum(w_read.unsqueeze(-1) * memory, dim=1)
            hidden = torch.tanh(torch.cat([read_content, query_key[:, t, :]], dim=-1))
            hidden = self.read_layer(hidden)
            hidden_states.append(hidden)

            w_write = torch.softmax(torch.matmul(write_key[:, t, :], self.memory_key.T), dim=-1)
            erase = torch.sigmoid(self.erase_layer(write_value[:, t, :]))
            add = torch.tanh(self.add_layer(write_value[:, t, :]))
            memory = memory * (1 - w_write.unsqueeze(-1) * erase.unsqueeze(1))
            memory = memory + w_write.unsqueeze(-1) * add.unsqueeze(1)

        h = torch.stack(hidden_states, dim=1)
        logits_next = self.out(self.dropout(h)).squeeze(-1)
        return h, logits_next


class AKTSequenceModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        emb_size: int = 64,
        num_attn_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_questions = int(num_questions)
        self.emb_size = int(emb_size)
        self.question_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.correctness_emb = nn.Embedding(2, self.emb_size)
        self.question_variation_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.correctness_variation_emb = nn.Embedding(2, self.emb_size)
        self.question_difficulty_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.knowledge_encoder = _CausalSelfAttentionStack(self.emb_size, num_attn_heads, num_blocks, float(dropout))
        self.question_encoder = _CausalCrossAttentionStack(self.emb_size, num_attn_heads, num_blocks, float(dropout))
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.emb_size, 1),
        )
        self.device = "cpu"

    def forward(self, x: torch.Tensor):
        q_in, r_in, q_next = _unpack_kt_batch(x)
        q_in_safe = _safe_questions(q_in)
        q_next_safe = _safe_questions(q_next)

        hist_q = self.question_emb(q_in_safe)
        hist_diff = self.question_difficulty_emb(q_in_safe)
        hist_q = hist_q + hist_diff * self.question_variation_emb(q_in_safe)
        hist_interaction = hist_q + self.correctness_emb(r_in) + hist_diff * self.correctness_variation_emb(r_in)

        target_q = self.question_emb(q_next_safe)
        target_diff = self.question_difficulty_emb(q_next_safe)
        target_q = target_q + target_diff * self.question_variation_emb(q_next_safe)

        memory = self.knowledge_encoder(hist_interaction)
        latent = self.question_encoder(target_q, memory)
        logits_next = self.out(self.dropout(torch.cat([latent, target_q], dim=-1))).squeeze(-1)
        return latent, logits_next


class SimpleKTSequenceModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        emb_size: int = 64,
        num_attn_heads: int = 4,
        num_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_questions = int(num_questions)
        self.emb_size = int(emb_size)
        self.question_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.correctness_emb = nn.Embedding(2, self.emb_size)
        self.interaction_encoder = _CausalSelfAttentionStack(self.emb_size, num_attn_heads, num_blocks, float(dropout))
        self.question_encoder = _CausalCrossAttentionStack(self.emb_size, num_attn_heads, num_blocks, float(dropout))
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.emb_size, 1),
        )
        self.device = "cpu"

    def forward(self, x: torch.Tensor):
        q_in, r_in, q_next = _unpack_kt_batch(x)
        q_in_safe = _safe_questions(q_in)
        q_next_safe = _safe_questions(q_next)
        interaction = self.question_emb(q_in_safe) + self.correctness_emb(r_in)
        target = self.question_emb(q_next_safe)
        memory = self.interaction_encoder(interaction)
        latent = self.question_encoder(target, memory)
        logits_next = self.out(self.dropout(torch.cat([latent, target], dim=-1))).squeeze(-1)
        return latent, logits_next


class DTransformerSequenceModel(nn.Module):
    def __init__(
        self,
        num_questions: int,
        emb_size: int = 64,
        num_attn_heads: int = 4,
        num_blocks: int = 2,
        num_prototypes: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_questions = int(num_questions)
        self.emb_size = int(emb_size)
        self.num_prototypes = int(num_prototypes)
        self.question_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.correctness_emb = nn.Embedding(2, self.emb_size)
        self.question_variation_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.correctness_variation_emb = nn.Embedding(2, self.emb_size)
        self.question_difficulty_emb = nn.Embedding(self.num_questions, self.emb_size)
        self.question_encoder = _CausalSelfAttentionStack(self.emb_size, num_attn_heads, num_blocks, float(dropout))
        self.knowledge_encoder = _CausalSelfAttentionStack(self.emb_size, num_attn_heads, num_blocks, float(dropout))
        self.knowledge_retriever = _CausalCrossAttentionStack(self.emb_size, num_attn_heads, num_blocks, float(dropout))
        self.prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.emb_size) * 0.1)
        self.dropout = nn.Dropout(float(dropout))
        self.out = nn.Sequential(
            nn.Linear(self.emb_size * 3, self.emb_size * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.emb_size, 1),
        )
        self.device = "cpu"

    def forward(self, x: torch.Tensor):
        q_in, r_in, q_next = _unpack_kt_batch(x)
        q_in_safe = _safe_questions(q_in)
        q_next_safe = _safe_questions(q_next)

        hist_diff = self.question_difficulty_emb(q_in_safe)
        hist_q = self.question_emb(q_in_safe) + hist_diff * self.question_variation_emb(q_in_safe)
        interaction = hist_q + self.correctness_emb(r_in) + hist_diff * self.correctness_variation_emb(r_in)

        target_diff = self.question_difficulty_emb(q_next_safe)
        target_q = self.question_emb(q_next_safe) + target_diff * self.question_variation_emb(q_next_safe)

        question_ctx = self.question_encoder(target_q)
        interaction_ctx = self.knowledge_encoder(interaction)
        latent = self.knowledge_retriever(question_ctx, interaction_ctx)

        proto_scores = torch.matmul(question_ctx, self.prototypes.T)
        proto_weights = torch.softmax(proto_scores, dim=-1)
        proto_ctx = torch.matmul(proto_weights, self.prototypes)

        logits_next = self.out(self.dropout(torch.cat([question_ctx, latent, proto_ctx], dim=-1))).squeeze(-1)
        return latent, logits_next


class KTLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, labels: torch.Tensor):
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        if labels.dim() == 2:
            labels = labels.unsqueeze(0)

        target = labels[:, 0, :]
        mask = labels[:, 1, :].bool()
        if mask.numel() == 0 or not mask.any():
            if self.reduction == "none":
                return torch.zeros(pred.size(0), device=pred.device, dtype=pred.dtype)
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        bce = torch.nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        masked = bce * mask.to(bce.dtype)
        valid_per_seq = mask.sum(dim=1).clamp_min(1).to(bce.dtype)
        per_seq = masked.sum(dim=1) / valid_per_seq

        if self.reduction == "none":
            return per_seq
        if self.reduction == "sum":
            return masked.sum()
        if self.reduction != "mean":
            raise ValueError(f"Unsupported reduction: {self.reduction}")
        return masked.sum() / mask.sum().to(masked.dtype).clamp_min(1.0)


class KTModel(Trainable):
    def init(self):
        self.epochs = int(self.local_config["parameters"]["epochs"])
        self.model = get_instance_kvargs(
            self.local_config["parameters"]["model"]["class"],
            self.local_config["parameters"]["model"]["parameters"],
        )
        self.optimizer = get_instance_kvargs(
            self.local_config["parameters"]["optimizer"]["class"],
            {"params": self.model.parameters(), **self.local_config["parameters"]["optimizer"]["parameters"]},
        )
        self.loss_fn = get_instance_kvargs(
            self.local_config["parameters"]["loss_fn"]["class"],
            self.local_config["parameters"]["loss_fn"]["parameters"],
        )
        self.early_stopping_threshold = self.local_config["parameters"]["early_stopping_threshold"]
        self.lr_scheduler = lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.5, total_iters=max(self.epochs, 1)
        )
        self.training_set = self.local_config["parameters"].get("training_set", "train")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model.to(self.device)
        self.model.device = self.device
        self.patience = 0
        self.fit()

    def real_fit(self):
        train_loader, val_loader = self.dataset.get_loader_for(self.training_set, fold_fraction=None, drop_last=False)
        best_val_loss = None

        for epoch in range(self.epochs):
            losses = []
            self.model.train()
            self._flatten_if_supported()
            for x, labels in train_loader:
                x = x.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                _, pred = self.model(x)
                loss = self.loss_fn(pred, labels)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach().cpu().item()))

            self.global_ctx.logger.info(
                f"epoch = {epoch} ---> loss = {np.mean(losses) if losses else float('nan'):.4f}"
            )
            self.lr_scheduler.step()

            if self.early_stopping_threshold is not None and val_loader is not None and len(val_loader) > 0:
                val_loss = self._eval_loss(val_loader)
                self.global_ctx.logger.info(f"epoch = {epoch} ---> var_loss = {val_loss:.4f}")
                if best_val_loss is not None and abs(best_val_loss - val_loss) < self.early_stopping_threshold:
                    self.patience += 1
                    if self.patience >= 3:
                        self.global_ctx.logger.info(f"Early stopped training at epoch {epoch}")
                        break
                else:
                    self.patience = 0
                best_val_loss = val_loss

    def _eval_loss(self, loader):
        self.model.eval()
        self._flatten_if_supported()
        losses = []
        with torch.no_grad():
            for x, labels in loader:
                x = x.to(self.device)
                labels = labels.to(self.device)
                _, pred = self.model(x)
                loss = self.loss_fn(pred, labels)
                losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses)) if losses else float("nan")

    def _flatten_if_supported(self):
        flatten_fn = getattr(self.model, "flatten_parameters", None)
        if callable(flatten_fn):
            flatten_fn()

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        local_config["parameters"]["epochs"] = local_config["parameters"].get("epochs", 5)
        local_config["parameters"]["batch_size"] = local_config["parameters"].get("batch_size", 16)
        local_config["parameters"]["early_stopping_threshold"] = local_config["parameters"].get(
            "early_stopping_threshold", None
        )
        init_dflts_to_of(local_config, "optimizer", "torch.optim.Adam", lr=0.001)
        init_dflts_to_of(local_config, "loss_fn", "kt_unlearn.model.KTModel.KTLoss")
        model_params = local_config["parameters"]["model"]["parameters"]
        model_params["num_questions"] = model_params.get("num_questions", self.dataset.datasource.num_questions)
        local_config["parameters"]["alias"] = local_config["parameters"].get(
            "alias", local_config["parameters"]["model"]["class"]
        )
        local_config["parameters"]["training_set"] = local_config["parameters"].get("training_set", "train")
