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


# ---------------------------------------------------------------------------
# Aspect-wise contrastive heads
# ---------------------------------------------------------------------------
"""Each aspect owns a disjoint slice of the encoder embedding and a private
projection head + supervised-contrastive loss. Because an aspect's loss only
reads its own slice (and its own head only takes that slice as input), the
gradient of loss_A w.r.t. any dimension outside slice_A is exactly zero -- so
each chunk is optimized *only* by its own objective and specializes. The shared
transformer backbone still receives the summed gradient (expected: the trunk
learns features useful to every aspect, the chunks carve out the specialization).

Positives are defined per aspect by a similarity signal the caller computes from
domain descriptors:
  * spending     -> MCC-distribution / ticket-size / velocity cluster
  * utilization  -> utilization-curve / paydown / revolve-vs-transact cluster
Pass either an integer label per aspect (same label == positive, e.g. a KMeans
cluster id over the aspect descriptors) or an explicit (B, B) boolean positive
mask (e.g. from a descriptor-distance kNN graph). The two augmented views of the
same customer are always mutual positives, so augmentation-invariance and
aspect-similarity grouping are learned by the same SupCon term.
"""


@dataclass
class AspectSpec:
    name: str
    start: int          # inclusive index into the encoder embedding
    end: int            # exclusive
    proj_dim: int = 64
    proj_hidden: int = 128
    temperature: float = 0.1
    weight: float = 1.0


def supcon_loss(
    feats: torch.Tensor,
    pos_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020).

    feats    : (M, d) L2-normalized.
    pos_mask : (M, M) bool; True where j is a positive for anchor i. The
               diagonal is ignored. Anchors with no positive are skipped.
    """
    m = feats.size(0)
    sim = (feats @ feats.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # stability

    self_mask = torch.eye(m, dtype=torch.bool, device=feats.device)
    pos_mask = pos_mask & ~self_mask

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return feats.sum() * 0.0  # keeps graph connected, contributes nothing
    per_anchor = -(log_prob * pos_mask).sum(dim=1)[valid] / pos_count[valid]
    return per_anchor.mean()


class AspectContrastiveLoss(nn.Module):
    def __init__(self, aspects: list[AspectSpec]):
        super().__init__()
        self.specs = {a.name: a for a in aspects}
        self.heads = nn.ModuleDict()
        for a in aspects:
            chunk_dim = a.end - a.start
            self.heads[a.name] = nn.Sequential(
                nn.Linear(chunk_dim, a.proj_hidden),
                nn.BatchNorm1d(a.proj_hidden),
                nn.GELU(),
                nn.Linear(a.proj_hidden, a.proj_dim),
            )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: dict[str, torch.Tensor] | None = None,
        pos_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """emb_a / emb_b : (B, d_model) encoder embeddings of the two views.

        For each aspect, slice both views, project with the aspect head,
        L2-normalize, stack the 2B features, and run SupCon. Provide per-aspect
        `labels` (B,) longs OR `pos_masks` (B, B) bool; the (B,B) mask is
        expanded to the stacked (2B, 2B) layout and the cross-view diagonal is
        forced positive so the two views of a customer attract.
        """
        labels = labels or {}
        pos_masks = pos_masks or {}
        out: dict[str, torch.Tensor] = {}
        total = emb_a.new_zeros(())

        b = emb_a.size(0)
        for name, spec in self.specs.items():
            za = emb_a[:, spec.start:spec.end]
            zb = emb_b[:, spec.start:spec.end]
            head = self.heads[name]
            fa = F.normalize(head(za), dim=-1)
            fb = F.normalize(head(zb), dim=-1)
            feats = torch.cat([fa, fb], dim=0)  # (2B, p)

            if name in pos_masks:
                base = pos_masks[name].bool()
            elif name in labels:
                lab = labels[name]
                base = lab.unsqueeze(0) == lab.unsqueeze(1)  # (B, B)
            else:
                raise KeyError(f"aspect '{name}' needs a label or pos_mask")

            # Tile (B,B) into the stacked (2B,2B) space; the two views share the
            # same aspect identity, so all four quadrants use `base`.
            pm = torch.cat(
                [torch.cat([base, base], dim=1),
                 torch.cat([base, base], dim=1)], dim=0
            )
            # Force same-customer cross-view pairs to be positives.
            idx = torch.arange(b, device=feats.device)
            pm[idx, idx + b] = True
            pm[idx + b, idx] = True

            l = supcon_loss(feats, pm, spec.temperature)
            out[name] = l.detach()
            total = total + spec.weight * l

        out["loss"] = total
        return out
