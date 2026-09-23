"""Dataset loaders reading straight from the preprocessed split manifests."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLIT_COLUMNS = ["dataset", "label", "path", "md5", "size_bytes", "ahash",
                 "width", "height", "cluster", "split"]


class CSVImageDataset(Dataset):
    """Image classification dataset from a preprocessed split CSV.

    The CSV must contain at least the columns path, label, split.
    """

    def __init__(self, split_csv: Path, split: str,
                 transform=None, cache: Optional[dict] = None):
        df = pd.read_csv(split_csv)
        missing = [c for c in ("path", "label", "split") if c not in df.columns]
        if missing:
            raise ValueError(f"{split_csv} missing columns: {missing}")
        self.df = df[df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split {split!r} in {split_csv}")
        self.labels = sorted(self.df["label"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.labels)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = self.class_to_idx[row["label"]]
        return img, label


def make_class_weights(split_csv: Path, split: str = "train") -> torch.Tensor:
    df = pd.read_csv(split_csv)
    counts = df[df["split"] == split]["label"].value_counts().sort_index()
    total = int(counts.sum())
    n = int(len(counts))
    weights = torch.tensor([total / (n * counts[i]) for i in sorted(counts.index)])
    return weights / weights.sum()