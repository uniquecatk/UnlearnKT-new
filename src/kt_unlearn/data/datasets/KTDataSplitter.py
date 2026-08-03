from __future__ import annotations

import pandas as pd

from kt_unlearn.data.datasets.DataSplitter import DataSplitter


class KTDataSplitterByFold(DataSplitter):
    """Split KT samples by the `fold` column from the original csv."""

    def __init__(self, fold_values, parts_names, ref_data="all"):
        super().__init__(ref_data, parts_names)
        self.fold_values = list(fold_values)

    def split_data(self, partitions):
        if self.ref_data != "all":
            raise ValueError("KTDataSplitterByFold currently supports ref_data='all' only.")

        dataset = partitions["all"].data
        csv_path = getattr(self.source, "path", None)
        if csv_path is None:
            raise ValueError("KTDataSplitterByFold requires source.path.")

        df = pd.read_csv(csv_path, usecols=["fold"])
        if len(df) != len(dataset):
            valid_rows = []
            full_df = pd.read_csv(csv_path)
            for _, row in full_df.iterrows():
                q = str(row["questions"]).split(",") if "questions" in full_df.columns else []
                r = str(row["responses"]).split(",") if "responses" in full_df.columns else []
                if len(q) == len(r) and len(q) >= 2:
                    valid_rows.append(int(row["fold"]))
            folds = valid_rows
        else:
            folds = [int(x) for x in df["fold"].tolist()]

        selected = [idx for idx, fold in enumerate(folds) if fold in self.fold_values]
        remainder = [idx for idx, fold in enumerate(folds) if fold not in self.fold_values]
        partitions[self.parts_names[0]] = selected
        if len(self.parts_names) > 1:
            partitions[self.parts_names[1]] = remainder
        return partitions

