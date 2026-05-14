"""VICReg loss for contrastive embedding learning without negative pairs.

Reference: Bardes, Ponce, LeCun (2022) "VICReg: Variance-Invariance-Covariance
Regularization for Self-Supervised Learning".

Three terms:
- Invariance: MSE between paired projections (the "minimize distance" objective).
- Variance:   hinge that pushes each projection dimension's stddev >= 1, which
              directly prevents the encoder from collapsing to a constant.
- Covariance: pushes off-diagonal feature covariance to 0, which prevents
              "informational collapse" (different dims encoding the same thing).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VICRegConfig:
    sim_coef: float = 25.0
    var_coef: float = 25.0
    cov_coef: float = 1.0
    eps: float = 1e-4
    var_gamma: float = 1.0  # target std per dim
    gather_distributed: bool = False


def _maybe_all_gather(x: torch.Tensor) -> torch.Tensor:
    """Gather a batch across DDP ranks so variance/covariance statistics use the
    full effective batch. Gradients only flow through the local shard."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    if world == 1:
        return x
    gathered = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(gathered, x.contiguous())
    gathered[dist.get_rank()] = x
    return torch.cat(gathered, dim=0)


class VICRegLoss(nn.Module):
    def __init__(self, cfg: VICRegConfig | None = None):
        super().__init__()
        self.cfg = cfg or VICRegConfig()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        sim = F.mse_loss(z_a, z_b)

        if cfg.gather_distributed:
            z_a_full = _maybe_all_gather(z_a)
            z_b_full = _maybe_all_gather(z_b)
        else:
            z_a_full, z_b_full = z_a, z_b

        za = z_a_full - z_a_full.mean(dim=0)
        zb = z_b_full - z_b_full.mean(dim=0)

        std_a = torch.sqrt(za.var(dim=0) + cfg.eps)
        std_b = torch.sqrt(zb.var(dim=0) + cfg.eps)
        var = torch.mean(F.relu(cfg.var_gamma - std_a)) + torch.mean(F.relu(cfg.var_gamma - std_b))

        n, d = za.shape
        cov_a = (za.T @ za) / max(n - 1, 1)
        cov_b = (zb.T @ zb) / max(n - 1, 1)
        off_diag = lambda c: c.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        cov = off_diag(cov_a).pow(2).sum() / d + off_diag(cov_b).pow(2).sum() / d

        total = cfg.sim_coef * sim + cfg.var_coef * var + cfg.cov_coef * cov
        return {"loss": total, "sim": sim.detach(), "var": var.detach(), "cov": cov.detach()}
