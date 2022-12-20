from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchdata.datapipes as dp
from torch.utils.data import DataLoader

from . import utils as _utils


class DeepCPI(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: str,
        seq_length: int = 12,
        train_split_ratio: float = 0.8,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_path = dataset_path
        self.seq_length = seq_length
        self.train_split_ratio = train_split_ratio
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def _load_data_and_split(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        df = pd.read_csv(self.dataset_path)
        data = df.values[:, 1:].astype(np.float32)

        total_size = data.shape[0]
        train_size = int(total_size * self.train_split_ratio)
        val_size = total_size - train_size

        train_data, val_data = data[:train_size, :], data[train_size:, :]

        def generate_samples(data: np.ndarray) -> list[np.ndarray]:
            length = data.shape[0]
            return [
                (data[end - self.seq_length:end, 1:], data[end, 1])
                for end in range(self.seq_length, length - 1)
            ]

        train_samples = generate_samples(train_data)
        val_samples = generate_samples(val_data)

        return train_samples, val_samples

    def _build_datapipe(
        self,
        samples: list[np.ndarray],
        shuffle: bool = False,
    ):
        pipe = dp.iter.IterableWrapper(samples)

        # pipe = pipe.distributed_sharding_filter()
        if shuffle:
            pipe = pipe.shuffle()

        return pipe

    def setup(self, stage: Optional[str] = None):
        train_samples, val_samples = self._load_data_and_split()

        if stage in (None, "fit", "validate"):
            self.train_data_pipe = self._build_datapipe(
                train_samples, shuffle=True
            )
            self.val_data_pipe = self._build_datapipe(
                val_samples, shuffle=False
            )

        if stage in (None, "test"):
            self.test_data_pipe = self._build_datapipe(
                val_samples, shuffle=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data_pipe,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data_pipe,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data_pipe,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
