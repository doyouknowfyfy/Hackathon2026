"""Single-GPU training for the time-series embedding transformer.

Usage:
    python scripts/train_single_gpu.py \
        --numeric data/numeric.npy --missing data/missing.npy \
        --categorical data/categorical.npy \
        --out runs/single_gpu --batch-size 1024 --epochs 30
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ts_embed.data import (
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
    p.add_argument("--batch-size", type=int, default=1024)
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
    return p.parse_args()


def cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device required; got CPU only")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths(numeric=args.numeric, missing=args.missing, categorical=args.categorical)
    dataset = TimeSeriesDataset(paths)

    masker = TimeFeatureMasker(
        time_mask_prob=args.time_mask_prob,
        feature_mask_prob=args.feature_mask_prob,
    )
    collator = ContrastiveCollator(masker)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
    )

    cfg = TSEncoderConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        proj_dim=args.proj_dim,
    )
    model = TSEmbeddingModel(cfg).to(device)
    model = torch.compile(model, mode="default")  # remove if torch.compile is unavailable

    loss_fn = VICRegLoss(VICRegConfig(gather_distributed=False))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda s: cosine_warmup(s, args.warmup_steps, total_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    step = 0
    for epoch in range(args.epochs):
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

            if step % args.log_every == 0:
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
            step += 1

        if (epoch + 1) % args.ckpt_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "step": step,
            }
            torch.save(ckpt, out_dir / f"ckpt_epoch{epoch:03d}.pt")
        print(f"epoch {epoch} took {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
