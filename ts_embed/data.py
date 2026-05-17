"""Dataset, masking augmentation, and collation.

Storage assumption
------------------
With 3M accounts x 24 months x 100 features (float32), the raw tensor is ~28 GB.
That does not fit in most GPU hosts' RAM comfortably, so the dataset reads from
numpy memory-mapped files:

    numeric.npy      shape (N, T, F_num)   float32   (imputed values)
    missing.npy      shape (N, T, F_num)   uint8     (1 = originally missing)
    categorical.npy  shape (N, T, F_cat)   int8      (0/1)

Build these once during data prep (e.g. from your parquet/tensor frame) and the
training loop streams batches with multiple worker processes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


@dataclass
class DatasetPaths:
    numeric: str | Path
    missing: str | Path
    categorical: str | Path | None = None


class TimeSeriesDataset(Dataset):
    def __init__(self, paths: DatasetPaths, seq_len: int = 24, n_numeric: int = 98,
                 n_categorical: int = 2):
        self.numeric = np.load(paths.numeric, mmap_mode="r")
        self.missing = np.load(paths.missing, mmap_mode="r")
        self.categorical = (
            np.load(paths.categorical, mmap_mode="r") if paths.categorical else None
        )

        n, t, f = self.numeric.shape
        assert t == seq_len, f"seq_len mismatch: file has {t}, expected {seq_len}"
        assert f == n_numeric, f"n_numeric mismatch: file has {f}, expected {n_numeric}"
        assert self.missing.shape == self.numeric.shape, "missing/numeric shape mismatch"
        if self.categorical is not None:
            assert self.categorical.shape[:2] == (n, t)
            assert self.categorical.shape[2] == n_categorical

        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Copy out of the memmap into a regular array so workers can pin memory.
        numeric = torch.from_numpy(np.asarray(self.numeric[idx], dtype=np.float32))
        missing = torch.from_numpy(np.asarray(self.missing[idx], dtype=np.float32))
        item = {"numeric": numeric, "missing": missing}
        if self.categorical is not None:
            item["categorical"] = torch.from_numpy(
                np.asarray(self.categorical[idx], dtype=np.int64)
            )
        return item


class ChunkedIterableDataset(IterableDataset):
    """Memory-bounded streaming dataset for >3M accounts on CPU-constrained hosts.

    Why not the random-access ``TimeSeriesDataset``? Random indexing into a
    ~28 GB memmap pulls scattered 4 KB pages into the OS page cache; under many
    DataLoader workers the resident set grows until the box thrashes. This
    dataset instead reads **contiguous chunks**, yields the samples in that
    chunk, then ``del``s the chunk so its RAM is reclaimed before the next one.
    Peak host memory per worker is bounded by ``chunk_size`` (not by N).

    Sharding for GPU clusters
    -------------------------
    The global index range is split into ``world_size`` contiguous rank shards,
    then each rank's chunks are round-robin'd across its DataLoader workers.
    Every (rank, worker) pair therefore sees a disjoint slice with no
    DistributedSampler needed. Call ``set_epoch`` each epoch so the chunk /
    in-chunk shuffle reseeds.
    """

    def __init__(
        self,
        paths: DatasetPaths,
        chunk_size: int = 4096,
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        seq_len: int = 24,
        n_numeric: int = 98,
    ):
        super().__init__()
        self.paths = paths
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        # Read the header only (no data) to learn N; don't keep the mmap open
        # across a fork — workers reopen it in __iter__.
        head = np.load(paths.numeric, mmap_mode="r")
        n, t, f = head.shape
        assert t == seq_len and f == n_numeric, "shape mismatch vs config"
        self.n = n
        del head

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def steps_per_epoch(self, global_batch_size: int) -> int:
        return self.n // global_batch_size

    def _rank_range(self) -> tuple[int, int]:
        per = self.n // self.world_size
        start = self.rank * per
        end = self.n if self.rank == self.world_size - 1 else start + per
        return start, end

    def __iter__(self):
        # Open memmaps inside the worker process (post-fork) so pages aren't
        # shared/retained from the parent.
        numeric = np.load(self.paths.numeric, mmap_mode="r")
        missing = np.load(self.paths.missing, mmap_mode="r")
        categorical = (
            np.load(self.paths.categorical, mmap_mode="r")
            if self.paths.categorical else None
        )

        info = get_worker_info()
        wid, nworkers = (0, 1) if info is None else (info.id, info.num_workers)

        r_start, r_end = self._rank_range()
        starts = list(range(r_start, r_end, self.chunk_size))
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(starts)
        starts = starts[wid::nworkers]  # this worker's chunks

        for s in starts:
            e = min(s + self.chunk_size, r_end)
            b_num = np.asarray(numeric[s:e], dtype=np.float32)
            b_mis = np.asarray(missing[s:e], dtype=np.float32)
            b_cat = (
                np.asarray(categorical[s:e], dtype=np.int64)
                if categorical is not None else None
            )
            order = np.arange(e - s)
            if self.shuffle:
                rng.shuffle(order)
            for i in order:
                item = {
                    "numeric": torch.from_numpy(b_num[i].copy()),
                    "missing": torch.from_numpy(b_mis[i].copy()),
                }
                if b_cat is not None:
                    item["categorical"] = torch.from_numpy(b_cat[i].copy())
                yield item
            # Reclaim the chunk's RAM before reading the next one.
            del b_num, b_mis, b_cat


class TimeFeatureMasker:
    """Generates a masked "view" of an account for contrastive training.

    Two independent masks:
    - time_keep_mask : drops whole months. We pass it to the encoder as a
      key_padding_mask so attention truly ignores those positions.
    - feature_mask : zeroes individual numeric cells AND sets their missing
      indicator to 1, so the encoder treats them like real missing values
      (re-using the missing-handling pathway).

    Categorical features are kept intact in the masked view (only 2 binary
    cols, masking them out is too destructive).
    """

    def __init__(
        self,
        time_mask_prob: float = 0.25,
        feature_mask_prob: float = 0.30,
        min_kept_steps: int = 6,
    ):
        self.time_mask_prob = time_mask_prob
        self.feature_mask_prob = feature_mask_prob
        self.min_kept_steps = min_kept_steps

    def __call__(self, numeric: torch.Tensor, missing: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        t, f = numeric.shape

        # Feature-level mask
        if self.feature_mask_prob > 0:
            feat_drop = torch.rand(t, f) < self.feature_mask_prob
            numeric = numeric.clone()
            missing = missing.clone()
            numeric[feat_drop] = 0.0
            missing[feat_drop] = 1.0

        # Time-level mask, with a floor so we don't lose the whole sequence.
        if self.time_mask_prob > 0:
            time_drop = torch.rand(t) < self.time_mask_prob
            # Ensure at least `min_kept_steps` survive.
            kept = (~time_drop).sum().item()
            if kept < self.min_kept_steps:
                idx = torch.randperm(t)[: self.min_kept_steps]
                time_drop = torch.ones(t, dtype=torch.bool)
                time_drop[idx] = False
            time_keep = (~time_drop).float()
        else:
            time_keep = torch.ones(t)

        return numeric, missing, time_keep


class ContrastiveCollator:
    """Builds two augmented views per sample.

    View A: no input masking (the "anchor" the user wants the masked view to
            match).
    View B: time + feature masking applied by TimeFeatureMasker.

    Returns one dict with stacked tensors; both views are batched together so
    we only run the encoder once per view per step.
    """

    def __init__(self, masker: TimeFeatureMasker):
        self.masker = masker

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        numeric = torch.stack([b["numeric"] for b in batch])
        missing = torch.stack([b["missing"] for b in batch])
        categorical = (
            torch.stack([b["categorical"] for b in batch]) if "categorical" in batch[0] else None
        )

        b_num, b_miss, b_keep = [], [], []
        for i in range(numeric.size(0)):
            n2, m2, k2 = self.masker(numeric[i], missing[i])
            b_num.append(n2)
            b_miss.append(m2)
            b_keep.append(k2)
        numeric_b = torch.stack(b_num)
        missing_b = torch.stack(b_miss)
        time_keep_b = torch.stack(b_keep)
        time_keep_a = torch.ones(numeric.size(0), numeric.size(1))

        out = {
            "numeric_a": numeric,
            "missing_a": missing,
            "time_keep_a": time_keep_a,
            "numeric_b": numeric_b,
            "missing_b": missing_b,
            "time_keep_b": time_keep_b,
        }
        if categorical is not None:
            out["categorical_a"] = categorical
            out["categorical_b"] = categorical
        return out
