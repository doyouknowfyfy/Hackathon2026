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


# ---------------------------------------------------------------------------
# Structured contrastive learning (distinct semantic subspaces)
# ---------------------------------------------------------------------------
"""Goal: make each embedding chunk capture a *distinct* semantic, where here the
four semantics are ``balance``, ``payment``, ``delinquency``, ``other``.

Two forces, combined:

1. Per-semantic SupCon (the "specialize" force). Each chunk has a private head
   and a supervised-contrastive loss with semantic-specific positives, so the
   chunk learns to encode that semantic. Because the loss only touches its own
   slice, gradients don't leak to other chunks.

2. Cross-chunk decorrelation (the "be distinct" force). Specialization alone
   does not stop two chunks from redundantly encoding the *same* signal. We
   z-score each chunk across the batch and drive the cross-correlation between
   every pair of *different* chunks toward zero (Barlow-Twins-style, but on the
   off-block between chunks). This pushes the subspaces to be statistically
   independent, so they carry non-redundant information.

A small per-chunk variance hinge keeps every dimension active, so the
decorrelation term can't be trivially satisfied by collapsing dimensions.
"""


@dataclass
class SemanticSpec:
    name: str
    start: int                 # inclusive index into the encoder embedding
    end: int                   # exclusive
    proj_dim: int = 64
    proj_hidden: int = 128
    temperature: float = 0.1
    contrastive_weight: float = 1.0


def _stacked_pos_mask(base: torch.Tensor, b: int) -> torch.Tensor:
    """Tile a (B,B) positive mask into the stacked (2B,2B) two-view layout and
    force same-customer cross-view pairs positive."""
    pm = torch.cat(
        [torch.cat([base, base], dim=1), torch.cat([base, base], dim=1)], dim=0
    )
    idx = torch.arange(b, device=base.device)
    pm[idx, idx + b] = True
    pm[idx + b, idx] = True
    return pm


class StructuredContrastiveLoss(nn.Module):
    def __init__(
        self,
        semantics: list[SemanticSpec],
        decorr_weight: float = 1.0,
        var_weight: float = 1.0,
        var_gamma: float = 1.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.specs = {s.name: s for s in semantics}
        self.decorr_weight = decorr_weight
        self.var_weight = var_weight
        self.var_gamma = var_gamma
        self.var_eps = var_eps
        self.heads = nn.ModuleDict()
        for s in semantics:
            cdim = s.end - s.start
            self.heads[s.name] = nn.Sequential(
                nn.Linear(cdim, s.proj_hidden),
                nn.BatchNorm1d(s.proj_hidden),
                nn.GELU(),
                nn.Linear(s.proj_hidden, s.proj_dim),
            )

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float) -> torch.Tensor:
        return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: dict[str, torch.Tensor] | None = None,
        pos_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        labels = labels or {}
        pos_masks = pos_masks or {}
        b = emb_a.size(0)
        out: dict[str, torch.Tensor] = {}

        # 1. Per-semantic SupCon -------------------------------------------------
        contrastive = emb_a.new_zeros(())
        for name, s in self.specs.items():
            za, zb = emb_a[:, s.start:s.end], emb_b[:, s.start:s.end]
            head = self.heads[name]
            feats = torch.cat(
                [F.normalize(head(za), dim=-1), F.normalize(head(zb), dim=-1)], 0
            )
            if name in pos_masks:
                base = pos_masks[name].bool()
            elif name in labels:
                lab = labels[name]
                base = lab.unsqueeze(0) == lab.unsqueeze(1)
            else:
                # Self-supervised fallback: only the two views of the same
                # customer are positives.
                base = torch.zeros(b, b, dtype=torch.bool, device=emb_a.device)
            pm = _stacked_pos_mask(base, b)
            l = supcon_loss(feats, pm, s.temperature)
            out[f"c_{name}"] = l.detach()
            contrastive = contrastive + s.contrastive_weight * l

        # 2. Cross-chunk decorrelation (use both views) -------------------------
        names = list(self.specs)
        decorr = emb_a.new_zeros(())
        n_pairs = 0
        for view in (emb_a, emb_b):
            zs = {
                nm: self._zscore(view[:, self.specs[nm].start:self.specs[nm].end],
                                 self.var_eps)
                for nm in names
            }
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    zi, zj = zs[names[i]], zs[names[j]]
                    cross = (zi.t() @ zj) / max(b - 1, 1)  # (d_i, d_j) correlations
                    decorr = decorr + cross.pow(2).mean()
                    n_pairs += 1
        decorr = decorr / max(n_pairs, 1)
        out["decorr"] = decorr.detach()

        # 3. Per-chunk variance hinge (anti-collapse) ---------------------------
        var = emb_a.new_zeros(())
        for view in (emb_a, emb_b):
            for nm in names:
                s = self.specs[nm]
                std = torch.sqrt(view[:, s.start:s.end].var(0) + self.var_eps)
                var = var + F.relu(self.var_gamma - std).mean()
        var = var / (2 * len(names))
        out["var"] = var.detach()

        out["loss"] = (
            contrastive
            + self.decorr_weight * decorr
            + self.var_weight * var
        )
        return out


# ---------------------------------------------------------------------------
# Target-supervised contrastive loss
# ---------------------------------------------------------------------------
class SupConLoss(nn.Module):
    """Supervised contrastive loss over the full embedding, guided by the
    downstream target label.

    Samples with the SAME target label are positives (pulled together);
    DIFFERENT labels are negatives (pushed apart). The two augmented
    (masked / unmasked) views of a sample are also forced to be mutual
    positives, so the encoder learns augmentation-invariance and
    target-discrimination from a single objective. Built on `supcon_loss`
    (Khosla et al. 2020).

    Unlike VICReg, SupCon does not need a separate anti-collapse term: the
    push-apart between different-label samples keeps the embedding spread out.
    """

    def __init__(self, in_dim: int, proj_dim: int = 128, proj_hidden: int = 256,
                 temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        # BatchNorm in the projector follows standard SimCLR/SupCon practice and
        # matters for contrastive performance.
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, proj_dim),
        )

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """emb_a / emb_b : (B, in_dim) encoder embeddings of the two views.
        labels : (B,) integer target labels."""
        b = emb_a.size(0)
        fa = F.normalize(self.proj(emb_a), dim=-1)
        fb = F.normalize(self.proj(emb_b), dim=-1)
        feats = torch.cat([fa, fb], dim=0)  # (2B, proj_dim)

        # Same-label pairs are positives. A batch with a single label gives an
        # all-positive mask (a weak, no-op step but not an error); use a
        # class-balanced sampler so that is vanishingly rare.
        base = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        pm = _stacked_pos_mask(base, b)
        return {"loss": supcon_loss(feats, pm, self.temperature)}


# ---------------------------------------------------------------------------
# Aspect-specific augmentation contrastive loss
# ---------------------------------------------------------------------------
"""Fixed-partitioned semantic embeddings via aspect-specific augmentations.

Idea: each chunk should be invariant to perturbations of features that belong
to OTHER aspects, while remaining sensitive to perturbations of its OWN
aspect's features. The result is a fixed partitioning where each chunk is
*structurally* tied to its semantic -- no clustering labels needed.

How it works. For each aspect g, build a "g-preserving" view of the input
where features outside g have been perturbed (masked / shuffled / noised) and
features inside g are intact. Run the encoder on:
  - the anchor view (all features intact), and
  - one g-preserving view per aspect g.

For chunk g, apply an InfoNCE / NT-Xent loss between the (anchor, g-preserving)
pair: positives = same customer across the two views; negatives = other
customers in the batch.

Why this pins chunk g to aspect g's features:
- Positives differ ONLY in non-g features -> the contrastive force makes chunk
  g invariant to those.
- Negatives differ in g's own features -> chunk g must be DIFFERENT for them,
  i.e. discriminative over g-aspect content.
Both forces together select for "chunk g encodes g's features specifically".

Optional cross-chunk decorrelation and per-chunk variance hinge keep the chunks
distinct and prevent dimensional collapse.

The augmentation lives in `ts_embed.data.aspect_preserving_view`; this loss
operates on the resulting encoder embeddings only.
"""


class AspectAugContrastiveLoss(nn.Module):
    def __init__(
        self,
        aspects: list[AspectSpec],
        decorr_weight: float = 0.5,
        var_weight: float = 0.0,
        var_gamma: float = 1.0,
        var_eps: float = 1e-4,
    ):
        super().__init__()
        self.specs = {a.name: a for a in aspects}
        self.decorr_weight = decorr_weight
        self.var_weight = var_weight
        self.var_gamma = var_gamma
        self.var_eps = var_eps
        self.heads = nn.ModuleDict()
        for s in aspects:
            cdim = s.end - s.start
            self.heads[s.name] = nn.Sequential(
                nn.Linear(cdim, s.proj_hidden),
                nn.BatchNorm1d(s.proj_hidden),
                nn.GELU(),
                nn.Linear(s.proj_hidden, s.proj_dim),
            )

    @staticmethod
    def _zscore(x: torch.Tensor, eps: float) -> torch.Tensor:
        return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)

    def forward(
        self,
        emb_anchor: torch.Tensor,
        emb_per_aspect: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        emb_anchor      : (B, d_model) -- encoder embedding of the anchor view.
        emb_per_aspect  : dict[name -> (B, d_model)] -- encoder embedding of
                          each aspect's *preserving* view (features outside
                          that aspect have been perturbed).
        """
        b = emb_anchor.size(0)
        device = emb_anchor.device
        out: dict[str, torch.Tensor] = {}
        contrastive = emb_anchor.new_zeros(())

        # Single-positive (NT-Xent) mask: only same-customer cross-view pairs.
        pos_mask = torch.zeros(2 * b, 2 * b, dtype=torch.bool, device=device)
        idx = torch.arange(b, device=device)
        pos_mask[idx, idx + b] = True
        pos_mask[idx + b, idx] = True

        for name, s in self.specs.items():
            if name not in emb_per_aspect:
                raise KeyError(f"missing aspect '{name}' in emb_per_aspect")
            anchor_g = emb_anchor[:, s.start:s.end]
            aug_g = emb_per_aspect[name][:, s.start:s.end]
            head = self.heads[name]
            fa = F.normalize(head(anchor_g), dim=-1)
            fb = F.normalize(head(aug_g), dim=-1)
            feats = torch.cat([fa, fb], dim=0)            # (2B, proj_dim)
            l = supcon_loss(feats, pos_mask, s.temperature)
            out[f"c_{name}"] = l.detach()
            contrastive = contrastive + s.weight * l

        # Cross-chunk decorrelation on the anchor embedding -- pushes the
        # fixed chunks to encode distinct (non-redundant) information.
        names = list(self.specs)
        decorr = emb_anchor.new_zeros(())
        n_pairs = 0
        zs = {
            nm: self._zscore(
                emb_anchor[:, self.specs[nm].start:self.specs[nm].end],
                self.var_eps,
            )
            for nm in names
        }
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                zi, zj = zs[names[i]], zs[names[j]]
                cross = (zi.t() @ zj) / max(b - 1, 1)
                decorr = decorr + cross.pow(2).mean()
                n_pairs += 1
        decorr = decorr / max(n_pairs, 1)
        out["decorr"] = decorr.detach()

        # Per-chunk variance hinge (optional anti-collapse insurance).
        var = emb_anchor.new_zeros(())
        for nm in names:
            s = self.specs[nm]
            std = torch.sqrt(emb_anchor[:, s.start:s.end].var(0) + self.var_eps)
            var = var + F.relu(self.var_gamma - std).mean()
        var = var / max(len(names), 1)
        out["var"] = var.detach()

        out["loss"] = (
            contrastive
            + self.decorr_weight * decorr
            + self.var_weight * var
        )
        return out
