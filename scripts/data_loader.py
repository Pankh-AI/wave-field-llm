"""
Memory-mapped sharded data loader for pre-tokenized binary data.

Reads np.uint16 shard files produced by pretokenize.py with zero RAM overhead.
Each DataLoader worker gets disjoint shards. Shard order shuffled per epoch.

Usage:
    from scripts.data_loader import create_dataloader

    train_loader = create_dataloader("/data/tokenized", batch_size=16, seq_len=512)
    for input_ids, labels in train_loader:
        ...
"""

import json
import math
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader


class ShardedDataset(IterableDataset):
    """
    Memory-mapped iterable dataset over pre-tokenized binary shards.

    Each shard is a flat np.uint16 file. Yields (input_ids, labels) chunks
    of seq_len tokens from memmapped data — zero RAM overhead.
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 512,
        shard_indices: Optional[list] = None,
        seed: int = 42,
        epoch: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.seed = seed
        self.epoch = epoch

        manifest_path = os.path.join(data_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        num_shards = self.manifest["num_shards"]
        if shard_indices is not None:
            self.shard_indices = shard_indices
        else:
            self.shard_indices = list(range(num_shards))

    def _get_shard_path(self, shard_idx: int) -> str:
        return os.path.join(self.data_dir, f"shard_{shard_idx:04d}.bin")

    def _iter_shard(self, shard_idx: int):
        """Yield (input_ids, labels) chunks from a single shard via memmap."""
        path = self._get_shard_path(shard_idx)
        if not os.path.exists(path):
            return

        num_tokens = os.path.getsize(path) // 2  # uint16 = 2 bytes
        if num_tokens < self.seq_len + 1:
            return

        data = np.memmap(path, dtype=np.uint16, mode="r", shape=(num_tokens,))

        # Yield contiguous chunks of seq_len+1 tokens (input + target)
        num_chunks = (num_tokens - 1) // self.seq_len
        for i in range(num_chunks):
            start = i * self.seq_len
            end = start + self.seq_len + 1
            if end > num_tokens:
                break
            chunk = np.array(data[start:end], dtype=np.int64)  # copy from memmap
            input_ids = torch.from_numpy(chunk[:-1])
            labels = torch.from_numpy(chunk[1:])
            yield input_ids, labels

    def __iter__(self):
        # Get worker info for multi-worker splitting
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            # Disjoint shard assignment
            worker_shards = [
                s for i, s in enumerate(self.shard_indices) if i % num_workers == worker_id
            ]
        else:
            worker_shards = self.shard_indices

        # Shuffle shards per epoch
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(worker_shards)

        for shard_idx in worker_shards:
            yield from self._iter_shard(shard_idx)

    def set_epoch(self, epoch: int):
        """Set epoch for shard shuffle reproducibility."""
        self.epoch = epoch


def create_dataloader(
    data_dir: str,
    batch_size: int = 16,
    seq_len: int = 512,
    num_workers: int = 2,
    seed: int = 42,
    epoch: int = 0,
    val_split: float = 0.05,
    split: str = "train",
) -> DataLoader:
    """
    Create a DataLoader from pre-tokenized shards.

    Args:
        data_dir: Directory containing shard files and manifest.json
        batch_size: Batch size
        seq_len: Sequence length
        num_workers: DataLoader workers
        seed: Random seed
        epoch: Current epoch (for shard shuffle)
        val_split: Fraction of shards for validation
        split: "train" or "val"

    Returns:
        DataLoader yielding (input_ids, labels) batches
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    num_shards = manifest["num_shards"]
    all_indices = list(range(num_shards))

    # Deterministic train/val split by shard index
    val_count = max(1, int(num_shards * val_split))
    val_indices = all_indices[-val_count:]
    train_indices = all_indices[:-val_count]

    if split == "val":
        shard_indices = val_indices
    else:
        shard_indices = train_indices

    dataset = ShardedDataset(
        data_dir=data_dir,
        seq_len=seq_len,
        shard_indices=shard_indices,
        seed=seed,
        epoch=epoch,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    return loader
