"""Multi-GPU / multi-node DDP training.

Launch on a single node with 8 GPUs:
    torchrun --standalone --nproc_per_node=8 scripts/train_ddp.py \
        --numeric data/numeric.npy --missing data/missing.npy \
        --categorical data/categorical.npy \
        --out runs/ddp --batch-size 512 --epochs 30

Launch on a cluster (Slurm-style), per node:
    torchrun \
        --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_ON_NODE \
        --node_rank=$SLURM_NODEID \
        --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        scripts/train_ddp.py ...

Notes
-----
* `--batch-size` is per-GPU; effective batch is `batch_size * world_size`.
* DistributedSampler shards accounts across ranks each epoch.
* VICReg's variance/covariance terms gather projections across ranks so the
  decorrelation statistics use the full effective batch.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ts_embed.data import (
    ChunkedIterableDataset,
    ContrastiveCollator,
    DatasetPaths,
    TimeFeatureMasker,
    TimeSeriesDataset,
)
from ts_embed.loss import VICRegConfig, VICRegLoss
from ts_embed.model import TSEmbeddingModel, TSEncoderConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--numeric", required=True)
    p.add_argument("--missing", required=True)
    p.add_argument("--categorical", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=512, help="per-GPU batch size")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--time-mask-prob", type=float, default=0.25)
    p.add_argument("--feature-mask-prob", type=float, default=0.30)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-scale-batches", action="store_true", default=True,
                   help="linear-scale base LR by world_size")
    p.add_argument("--streaming", action="store_true",
                   help="use ChunkedIterableDataset (memory-bounded contiguous "
                        "reads) instead of random-access memmap indexing")
    p.add_argument("--chunk-size", type=int, default=4096,
                   help="rows held in RAM per worker when --streaming")
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="DataLoader prefetch batches per worker (lower = less RAM)")
    p.add_argument("--empty-cache-every", type=int, default=0,
                   help="call torch.cuda.empty_cache() every N steps (0 = never)")
    return p.parse_args()


def cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def setup_dist() -> tuple[int, int, int]:
    """Initialize torch.distributed using torchrun-provided env vars."""
    if "RANK" not in os.environ:
        raise RuntimeError("Launch with torchrun so RANK/WORLD_SIZE/LOCAL_RANK are set")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = setup_dist()
    # Make seeds rank-dependent for augmentation diversity but reproducible.
    torch.manual_seed(args.seed + rank)

    device = torch.device(f"cuda:{local_rank}")
    out_dir = Path(args.out)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    paths = DatasetPaths(numeric=args.numeric, missing=args.missing, categorical=args.categorical)
    masker = TimeFeatureMasker(
        time_mask_prob=args.time_mask_prob,
        feature_mask_prob=args.feature_mask_prob,
    )
    collator = ContrastiveCollator(masker)

    global_batch = args.batch_size * world_size
    common_loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )
    if args.num_workers > 0:
        common_loader_kw["prefetch_factor"] = args.prefetch_factor

    if args.streaming:
        # IterableDataset shards internally by (rank, worker) and frees each
        # contiguous chunk after it is consumed -> peak host RAM bounded by
        # chunk_size, not by N. No DistributedSampler.
        dataset = ChunkedIterableDataset(
            paths, chunk_size=args.chunk_size, shuffle=True, seed=args.seed,
            rank=rank, world_size=world_size,
        )
        sampler = None
        steps_per_epoch = dataset.steps_per_epoch(global_batch)
        loader = DataLoader(dataset, **common_loader_kw)
    else:
        dataset = TimeSeriesDataset(paths)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True, seed=args.seed)
        loader = DataLoader(dataset, sampler=sampler, **common_loader_kw)
        steps_per_epoch = len(loader)

    cfg = TSEncoderConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        proj_dim=args.proj_dim,
    )
    model = TSEmbeddingModel(cfg).to(device)
    # SyncBatchNorm so the projection head's BN works correctly across ranks.
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    loss_fn = VICRegLoss(VICRegConfig(gather_distributed=True))

    base_lr = args.lr * (world_size if args.lr_scale_batches else 1)
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: cosine_warmup(s, args.warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        else:
            dataset.set_epoch(epoch)  # reseed streaming chunk/sample shuffle
        model.train()
        t0 = time.time()

        for batch in loader:
            num_a = batch["numeric_a"].to(device, non_blocking=True)
            mis_a = batch["missing_a"].to(device, non_blocking=True)
            keep_a = batch["time_keep_a"].to(device, non_blocking=True)
            num_b = batch["numeric_b"].to(device, non_blocking=True)
            mis_b = batch["missing_b"].to(device, non_blocking=True)
            keep_b = batch["time_keep_b"].to(device, non_blocking=True)
            cat_a = batch.get("categorical_a")
            cat_b = batch.get("categorical_b")
            if cat_a is not None:
                cat_a = cat_a.to(device, non_blocking=True)
                cat_b = cat_b.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp, dtype=torch.bfloat16):
                _, z_a = model(num_a, mis_a, cat_a, keep_a)
                _, z_b = model(num_b, mis_b, cat_b, keep_b)
                losses = loss_fn(z_a, z_b)

            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            if is_main(rank) and step % args.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"epoch {epoch} step {step} "
                    f"loss {losses['loss'].item():.4f} "
                    f"sim {losses['sim'].item():.4f} "
                    f"var {losses['var'].item():.4f} "
                    f"cov {losses['cov'].item():.4f} "
                    f"lr {lr:.2e}",
                    flush=True,
                )
            # Drop references so Python frees the host (pinned) staging buffers
            # and CUDA frees the activation tensors before the next batch.
            del num_a, mis_a, keep_a, num_b, mis_b, keep_b, cat_a, cat_b
            del z_a, z_b, losses, batch
            if args.empty_cache_every and step % args.empty_cache_every == 0:
                torch.cuda.empty_cache()
            step += 1

        if is_main(rank) and (epoch + 1) % args.ckpt_every == 0:
            ckpt = {
                "model": model.module.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "step": step,
            }
            torch.save(ckpt, out_dir / f"ckpt_epoch{epoch:03d}.pt")
        if is_main(rank):
            print(f"epoch {epoch} took {time.time() - t0:.1f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
