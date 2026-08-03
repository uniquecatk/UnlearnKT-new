from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from kt_unlearn.data.data_sources.datasource import DataSource
from kt_unlearn.data.datasets.Dataset import DatasetWrapper


def _parse_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return []
    return [int(x) for x in s.split(",") if x != ""]


class KTSequenceTensorDataset(Dataset):
    def __init__(self, samples: list[tuple[torch.Tensor, torch.Tensor]]):
        self.samples = samples
        self.classes = [0, 1]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


class KTTensorDatasetWrapper(DatasetWrapper):
    def get_n_classes(self):
        return 2


class KTFileDataSource(DataSource):
    """Read KT sequence csv and pack samples into tensor pairs (X, y)."""

    def __init__(self, global_ctx, local_ctx):
        super().__init__(global_ctx, local_ctx)
        self.path = Path(self.params["path"])
        self.max_seq_len = int(self.params.get("max_seq_len", 100))
        self.pad_val = int(self.params.get("pad_val", -1))
        self.aligned_rows: list[dict] = []

    def get_name(self):
        return self.path.stem

    def get_simple_wrapper(self, data):
        return KTTensorDatasetWrapper(data, self.preprocess)

    def create_data(self) -> DatasetWrapper:
        if not self.path.exists():
            raise FileNotFoundError(f"KT csv not found: {self.path}")

        df = pd.read_csv(self.path)
        required = ["uid", "questions", "responses", "fold"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required KT column '{col}' in {self.path}")

        samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        aligned_rows: list[dict] = []
        q_max = -1
        for _, row in df.iterrows():
            q = _parse_int_list(row["questions"])
            r = _parse_int_list(row["responses"])
            if len(q) != len(r) or len(q) < 2:
                continue
            q_max = max(q_max, max(q))

            q = q[-self.max_seq_len :]
            r = r[-self.max_seq_len :]
            full_len = len(q)
            pad_needed = self.max_seq_len - full_len
            if pad_needed > 0:
                q = q + [self.pad_val] * pad_needed
                r = r + [self.pad_val] * pad_needed

            q_tensor = torch.tensor(q, dtype=torch.long)
            r_tensor = torch.tensor(r, dtype=torch.float32)
            q_in = q_tensor[:-1]
            q_next = q_tensor[1:]
            r_in = r_tensor[:-1]
            r_next = r_tensor[1:]
            mask = ((q_in != self.pad_val) & (q_next != self.pad_val)).to(torch.float32)

            x = torch.stack([q_in.to(torch.float32), r_in, q_next.to(torch.float32)], dim=0)
            y = torch.stack([r_next, mask], dim=0)
            samples.append((x, y))
            aligned_rows.append(row.to_dict())

        self.num_questions = q_max + 1
        self.aligned_rows = aligned_rows
        return KTTensorDatasetWrapper(KTSequenceTensorDataset(samples), self.preprocess)

