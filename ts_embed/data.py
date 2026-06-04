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
    """Generates a time-series-aware masked "view" of an account for contrastive
    training.

    Three augmentations, all reusing the missing-data pathway (a masked numeric
    cell is zeroed and its missing indicator set to 1, so the encoder treats it
    exactly like a real missing value):

    - time-span masking : hides whole CONTIGUOUS blocks of months. Scattered
      single-month dropout is a weak signal for a sequence model -- it can copy
      an adjacent month. Contiguous spans force reliance on longer-range
      temporal context. Dropped months are removed from attention via the
      encoder's key_padding_mask (the sequence is not cropped, so positional
      encodings still reflect true month indices).
    - feature-span masking : for a random subset of feature CHANNELS, hides a
      contiguous time window of that channel (mimics a feature going dark for a
      stretch of months), rather than scattered isolated cells.
    - jitter (optional, off by default) : small Gaussian noise on the numeric
      values; a standard, cheap time-series contrastive augmentation.

    Categorical features are kept intact in the masked view (only 2 binary
    cols, masking them out is too destructive).
    """

    def __init__(
        self,
        time_mask_prob: float = 0.25,
        feature_mask_prob: float = 0.30,
        n_time_spans: int = 2,
        feat_span_frac: float = 0.5,
        jitter_std: float = 0.0,
        min_kept_steps: int = 6,
    ):
        self.time_mask_prob = time_mask_prob
        self.feature_mask_prob = feature_mask_prob
        self.n_time_spans = n_time_spans
        self.feat_span_frac = feat_span_frac
        self.jitter_std = jitter_std
        self.min_kept_steps = min_kept_steps

    @staticmethod
    def _span_lengths(total: int, n: int) -> list[int]:
        """Split `total` masked months into `n` roughly-equal contiguous spans."""
        n = max(1, min(n, total))
        base, rem = divmod(total, n)
        return [base + (1 if i < rem else 0) for i in range(n)]

    def __call__(self, numeric: torch.Tensor, missing: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        t, f = numeric.shape
        numeric = numeric.clone()
        missing = missing.clone()

        # Optional jitter on the numeric values.
        if self.jitter_std > 0:
            numeric = numeric + self.jitter_std * torch.randn_like(numeric)

        # Feature-span masking: a contiguous time window per selected channel.
        if self.feature_mask_prob > 0:
            chan_drop = torch.nonzero(
                torch.rand(f) < self.feature_mask_prob, as_tuple=False
            ).flatten()
            span = max(1, min(t, round(self.feat_span_frac * t)))
            for c in chan_drop.tolist():
                start = int(torch.randint(0, t - span + 1, (1,)))
                numeric[start:start + span, c] = 0.0
                missing[start:start + span, c] = 1.0

        # Time-span masking: contiguous blocks of months, with a floor so we
        # never lose the whole sequence.
        time_keep = torch.ones(t)
        if self.time_mask_prob > 0:
            total = min(round(self.time_mask_prob * t), t - self.min_kept_steps)
            if total > 0:
                drop = torch.zeros(t, dtype=torch.bool)
                for length in self._span_lengths(total, self.n_time_spans):
                    if length <= 0:
                        continue
                    start = int(torch.randint(0, t - length + 1, (1,)))
                    drop[start:start + length] = True
                if int((~drop).sum()) < self.min_kept_steps:  # safety floor
                    keep_idx = torch.randperm(t)[: self.min_kept_steps]
                    drop = torch.ones(t, dtype=torch.bool)
                    drop[keep_idx] = False
                time_keep = (~drop).float()

        return numeric, missing, time_keep


class ContrastiveCollator:
    """Builds two augmented views per sample.

    View A: no input masking (the "anchor" the user wants the masked view to
            match).
    View B: time-span + feature-span masking applied by TimeFeatureMasker.

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


def aspect_preserving_view(
    numeric: torch.Tensor,
    missing: torch.Tensor,
    categorical: torch.Tensor | None,
    feature_group: torch.Tensor,
    aspect_id: int,
    mode: str = "shuffle",
    noise_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Build a view that PRESERVES one semantic aspect's features and perturbs
    the rest. Used for aspect-specific augmentation contrastive learning
    (`AspectAugContrastiveLoss`).

    The chunk that corresponds to `aspect_id` should be invariant between the
    returned (perturbed) view and the original anchor view; the contrastive
    loss enforces that geometry.

    Args
    ----
    numeric        : (B, T, F) float -- imputed numeric features.
    missing        : (B, T, F) float -- 1 where originally missing, 0 elsewhere.
    categorical    : (B, T, F_cat) long or None. Kept INTACT in every mode --
                     categorical features carry shared signal across aspects.
    feature_group  : (F,) long -- `feature_group[i]` is the aspect id of
                     numeric feature i.
    aspect_id      : int -- the aspect to PRESERVE. All numeric features whose
                     group != aspect_id are perturbed.
    mode           : 'mask' | 'shuffle' | 'noise'
        - 'mask'    : zero non-aspect numeric values and set their missing
                      indicator to 1 (re-uses the encoder's missing pathway).
        - 'shuffle' : permute non-aspect feature values across the batch
                      dimension (each customer gets another customer's values
                      for those features). Preserves marginal distributions.
        - 'noise'   : add Gaussian noise scaled to each feature's std.

    Returns (numeric_aug, missing_aug, categorical_aug).
    """
    other = feature_group != aspect_id   # (F,) bool, True where feature is OUTSIDE the preserved aspect
    numeric = numeric.clone()
    missing = missing.clone()

    if mode == "mask":
        numeric[..., other] = 0.0
        missing[..., other] = 1.0
    elif mode == "shuffle":
        b = numeric.size(0)
        perm = torch.randperm(b, device=numeric.device)
        numeric[:, :, other] = numeric[perm][:, :, other]
        missing[:, :, other] = missing[perm][:, :, other]
    elif mode == "noise":
        feat_std = numeric[..., other].std(dim=(0, 1), keepdim=True) + 1e-6
        numeric[..., other] = (
            numeric[..., other]
            + noise_std * feat_std * torch.randn_like(numeric[..., other])
        )
    else:
        raise ValueError(
            f"unknown mode: {mode!r}; pick from 'mask', 'shuffle', 'noise'"
        )

    return numeric, missing, categorical
