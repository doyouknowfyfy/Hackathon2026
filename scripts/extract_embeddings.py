"""Run the trained encoder over the full dataset and dump per-account embeddings.

Usage:
    python scripts/extract_embeddings.py \
        --ckpt runs/ddp/ckpt_epoch029.pt \
        --numeric data/numeric.npy --missing data/missing.npy \
        --categorical data/categorical.npy \
        --out embeddings.npy --batch-size 4096
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ts_embed.data import DatasetPaths, TimeSeriesDataset
from ts_embed.model import TSEmbeddingModel, TSEncoderConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--numeric", required=True)
    p.add_argument("--missing", required=True)
    p.add_argument("--categorical", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=8)
    return p.parse_args()


def collate(batch):
    out = {
        "numeric": torch.stack([b["numeric"] for b in batch]),
        "missing": torch.stack([b["missing"] for b in batch]),
    }
    if "categorical" in batch[0]:
        out["categorical"] = torch.stack([b["categorical"] for b in batch])
    return out


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = TSEncoderConfig(**ckpt["cfg"])
    model = TSEmbeddingModel(cfg).to(device)
    state = ckpt["model"]
    # Strip torch.compile / DDP prefixes if present.
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    paths = DatasetPaths(numeric=args.numeric, missing=args.missing, categorical=args.categorical)
    dataset = TimeSeriesDataset(paths)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    emb = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(len(dataset), cfg.d_model)
    )

    cursor = 0
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            num = batch["numeric"].to(device, non_blocking=True)
            mis = batch["missing"].to(device, non_blocking=True)
            cat = batch.get("categorical")
            if cat is not None:
                cat = cat.to(device, non_blocking=True)
            z = model.encode(num, mis, cat).float().cpu().numpy()
            emb[cursor:cursor + z.shape[0]] = z
            cursor += z.shape[0]

    emb.flush()
    print(f"wrote {cursor} embeddings to {out_path}")


if __name__ == "__main__":
    main()
