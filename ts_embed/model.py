"""Transformer encoder for 24-month x 100-feature account time series.

Layout assumptions
------------------
- numeric_x:     (B, T, F_num)   float32 (imputed values)
- missing_mask:  (B, T, F_num)   float32, 1.0 where the value was originally
                                 missing (and later imputed), 0.0 elsewhere
- cat_x:         (B, T, F_cat)   long, binary categorical features (0/1)
- B = batch, T = 24 months, F_num = 98 numeric, F_cat = 2 categorical
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class TSEncoderConfig:
    n_numeric: int = 98
    cat_cardinalities: tuple[int, ...] = (2, 2)
    seq_len: int = 24
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    cat_embed_dim: int = 8
    proj_dim: int = 256
    pool: str = "mean"  # mean | cls


class FeatureEmbedding(nn.Module):
    """Combine numeric features (with missing indicator) and categorical embeddings."""

    def __init__(self, cfg: TSEncoderConfig):
        super().__init__()
        self.cfg = cfg
        # Numeric path: project [value, is_missing] per feature into a shared trunk.
        # Concatenating the missing indicator lets the model learn an offset
        # for imputed cells without relying on the imputed magnitude.
        self.numeric_proj = nn.Linear(cfg.n_numeric * 2, cfg.d_model)
        # Learned "missing" bias that gets added when the entire row is missing-heavy.
        self.missing_bias = nn.Parameter(torch.zeros(cfg.d_model))

        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(card, cfg.cat_embed_dim) for card in cfg.cat_cardinalities]
        )
        cat_total = cfg.cat_embed_dim * len(cfg.cat_cardinalities)
        self.cat_proj = nn.Linear(cat_total, cfg.d_model) if cat_total > 0 else None

        self.layer_norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        numeric_x: torch.Tensor,
        missing_mask: torch.Tensor,
        cat_x: torch.Tensor | None,
    ) -> torch.Tensor:
        # Zero-out imputed values so the model can't shortcut through imputation noise,
        # then feed [value, is_missing] pairs.
        value = numeric_x * (1.0 - missing_mask)
        pair = torch.stack([value, missing_mask], dim=-1)  # (B,T,F,2)
        pair = pair.flatten(-2)  # (B,T,F*2)
        h = self.numeric_proj(pair)

        # Soft "row is mostly missing" cue.
        row_missing = missing_mask.mean(dim=-1, keepdim=True)  # (B,T,1)
        h = h + row_missing * self.missing_bias

        if self.cat_proj is not None and cat_x is not None:
            embs = [emb(cat_x[..., i]) for i, emb in enumerate(self.cat_embeds)]
            cat = torch.cat(embs, dim=-1)
            h = h + self.cat_proj(cat)

        return self.layer_norm(h)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TSEncoder(nn.Module):
    """Transformer encoder that returns a single embedding per account."""

    def __init__(self, cfg: TSEncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg or TSEncoderConfig()
        self.embed = FeatureEmbedding(self.cfg)
        self.pos = SinusoidalPositionalEncoding(self.cfg.d_model, max_len=self.cfg.seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.d_model,
            nhead=self.cfg.n_heads,
            dim_feedforward=self.cfg.d_model * self.cfg.ff_mult,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.cfg.n_layers)
        self.out_norm = nn.LayerNorm(self.cfg.d_model)

        if self.cfg.pool == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.cfg.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

    def forward(
        self,
        numeric_x: torch.Tensor,
        missing_mask: torch.Tensor,
        cat_x: torch.Tensor | None = None,
        time_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (B, d_model) account embeddings.

        time_keep_mask : (B, T) bool/float with 1 where the timestep is kept.
            Used by the contrastive augmenter to drop whole months. We pass it
            through Transformer's key_padding_mask so attention ignores them.
        """
        h = self.embed(numeric_x, missing_mask, cat_x)
        if self.cls_token is not None:
            cls = self.cls_token.expand(h.size(0), -1, -1)
            h = torch.cat([cls, h], dim=1)

        h = self.pos(h)

        key_padding_mask = None
        if time_keep_mask is not None:
            kpm = (time_keep_mask < 0.5)  # True == ignore
            if self.cls_token is not None:
                cls_keep = torch.zeros(kpm.size(0), 1, dtype=torch.bool, device=kpm.device)
                kpm = torch.cat([cls_keep, kpm], dim=1)
            key_padding_mask = kpm

        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h = self.out_norm(h)

        if self.cls_token is not None:
            return h[:, 0]
        if time_keep_mask is not None:
            w = time_keep_mask.unsqueeze(-1)
            return (h * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
        return h.mean(dim=1)


class ProjectionHead(nn.Module):
    """MLP projector — VICReg / Barlow Twins style. Embeddings used downstream
    come from the encoder, not from the projector head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TSEmbeddingModel(nn.Module):
    def __init__(self, cfg: TSEncoderConfig | None = None, projector_hidden: int = 512):
        super().__init__()
        self.cfg = cfg or TSEncoderConfig()
        self.encoder = TSEncoder(self.cfg)
        self.projector = ProjectionHead(
            in_dim=self.cfg.d_model,
            hidden_dim=projector_hidden,
            out_dim=self.cfg.proj_dim,
        )

    def encode(self, numeric_x, missing_mask, cat_x=None, time_keep_mask=None):
        return self.encoder(numeric_x, missing_mask, cat_x, time_keep_mask)

    def forward(self, numeric_x, missing_mask, cat_x=None, time_keep_mask=None):
        z = self.encode(numeric_x, missing_mask, cat_x, time_keep_mask)
        return z, self.projector(z)


# ---------------------------------------------------------------------------
# Downstream fine-tuning: encoder + classification head
# ---------------------------------------------------------------------------
def _strip_prefixes(state: dict) -> dict:
    """Drop DDP / torch.compile key prefixes from a state_dict."""
    return {
        k.removeprefix("_orig_mod.").removeprefix("module."): v
        for k, v in state.items()
    }


class ClassificationHead(nn.Module):
    """MLP classification head.

    Uses LayerNorm (not BatchNorm): it behaves identically in train/eval mode
    and is robust to the small / uneven batch sizes common when fine-tuning, so
    there is no train/eval statistics shift.
    """

    def __init__(self, in_dim: int, n_classes: int = 1, hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TSClassifier(nn.Module):
    """TSEncoder + ClassificationHead for supervised fine-tuning on a target.

    n_classes == 1 -> binary; forward returns (B,) logits for BCEWithLogitsLoss.
    n_classes  > 1 -> multiclass; forward returns (B, n_classes) for CrossEntropy.

    Typical use: pretrain a TSEmbeddingModel contrastively, then
    ``load_pretrained_encoder`` here and fine-tune end-to-end.
    """

    def __init__(self, encoder_cfg: TSEncoderConfig, n_classes: int = 1,
                 head_hidden: int = 256, head_dropout: float = 0.2):
        super().__init__()
        self.cfg = encoder_cfg
        self.n_classes = n_classes
        self.encoder = TSEncoder(encoder_cfg)
        self.head = ClassificationHead(encoder_cfg.d_model, n_classes,
                                       head_hidden, head_dropout)

    def encode(self, numeric_x, missing_mask, cat_x=None, time_keep_mask=None):
        return self.encoder(numeric_x, missing_mask, cat_x, time_keep_mask)

    def forward(self, numeric_x, missing_mask, cat_x=None, time_keep_mask=None):
        z = self.encoder(numeric_x, missing_mask, cat_x, time_keep_mask)
        logits = self.head(z)
        return logits.squeeze(-1) if self.n_classes == 1 else logits

    def load_pretrained_encoder(self, state_dict: dict, strict: bool = True):
        """Load encoder weights from a pretrained checkpoint.

        Accepts a full ``TSEmbeddingModel`` state_dict (keys prefixed
        ``encoder.`` / ``projector.``) or a bare ``TSEncoder`` state_dict.
        """
        state = _strip_prefixes(state_dict)
        enc = {k[len("encoder."):]: v for k, v in state.items()
               if k.startswith("encoder.")}
        if not enc:  # already a bare encoder state_dict
            enc = {k: v for k, v in state.items()
                   if not k.startswith("projector.")}
        return self.encoder.load_state_dict(enc, strict=strict)

    def set_encoder_trainable(self, flag: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = flag

    def param_groups(self, encoder_lr: float, head_lr: float,
                     weight_decay: float = 1e-4) -> list[dict]:
        """Discriminative learning rates: a small LR for the pretrained encoder,
        a larger LR for the freshly-initialised head."""
        return [
            {"params": self.encoder.parameters(), "lr": encoder_lr,
             "weight_decay": weight_decay},
            {"params": self.head.parameters(), "lr": head_lr,
             "weight_decay": weight_decay},
        ]
