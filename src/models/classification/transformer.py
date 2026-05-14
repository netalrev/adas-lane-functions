"""
src/models/classification/transformer.py
==========================================
MFTransformer — multi-task Transformer for ADAS lane-function classification.

Architecture
------------
    Input [B, T, D]
        │
        ├─ Linear(D, d_model)                  # input projection
        │
    [CLS] prepend → [B, T+1, d_model]          # learnable classification token
        │
    PositionalEncoding                         # sinusoidal, over T+1 positions
        │
    TransformerEncoder (n_layers × Pre-LN)     # shared temporal encoder
        │
    [CLS] token output [B, d_model]
        │
        ├─ CIPVHead    → [B, 1]   logits
        ├─ LaneAssignHead → [B, 5] logits
        └─ CutInHead   → [B, 1]   logits

Design choices
--------------
- Pre-LN (norm_first=True): more stable gradient flow without LR warm-up tuning.
- Learnable [CLS] token initialised with trunc_normal(std=0.02): standard ViT
  practice, avoids zero-initialisation deadlocks in very small models.
- Sinusoidal positional encoding (not learned): generalises better when the
  window length (T) is changed at inference time.
- Shared encoder, separate heads: adding a new ADAS signal requires only a new
  head, no model surgery.  This is the Strategy pattern applied to task heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.models.classification.heads import CIPVHead, LaneAssignHead, CutInHead


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al., 2017).

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int
        Maximum sequence length supported.  Defaults to 64 (generous for T+1).
    """

    def __init__(self, d_model: int, max_len: int = 64) -> None:
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Register as buffer so it moves with .to(device) but is not a parameter.
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq, d_model]
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# MFTransformer
# ---------------------------------------------------------------------------

class MFTransformer(nn.Module):
    """
    Multi-task Transformer for CIPV, Lane Assignment, and Cut-In.

    Parameters
    ----------
    cfg : DictConfig
        Hydra training config.  Reads from cfg.training.model.*:
            d_model, n_heads, n_layers, d_ff, dropout, input_dim, seq_len.

    Returns (from forward)
    ----------------------
    cipv_logits   : Tensor [B, 1]
    lane_logits   : Tensor [B, 5]
    cut_in_logits : Tensor [B, 1]
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        m         = cfg.training.model
        d_model   = int(m.d_model)
        n_heads   = int(m.n_heads)
        n_layers  = int(m.n_layers)
        d_ff      = int(m.d_ff)
        dropout   = float(m.dropout)
        input_dim = int(m.input_dim)

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Linear(input_dim, d_model)

        # ── [CLS] token ───────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding (T+1 positions) ───────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=int(m.seq_len) + 2)

        # ── Transformer encoder ───────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,   # [B, seq, d_model] convention
            norm_first=True,    # Pre-LN: more stable than Post-LN for small models
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Task heads ────────────────────────────────────────────────────
        self.cipv_head    = CIPVHead(d_model, dropout)
        self.lane_head    = LaneAssignHead(d_model, dropout)
        self.cut_in_head  = CutInHead(d_model, dropout)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  [B, T, D]
            Stacked MF windows.

        Returns
        -------
        cipv_logits   : Tensor [B, 1]
        lane_logits   : Tensor [B, 5]
        cut_in_logits : Tensor [B, 1]
        """
        B = x.size(0)

        # Project D → d_model
        x = self.input_proj(x)                          # [B, T, d_model]

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, d_model]
        x   = torch.cat([cls, x], dim=1)                # [B, T+1, d_model]

        # Add positional encoding
        x = self.pos_enc(x)                             # [B, T+1, d_model]

        # Encode
        x = self.encoder(x)                             # [B, T+1, d_model]

        # Pool [CLS] position
        cls_emb = x[:, 0, :]                            # [B, d_model]

        return (
            self.cipv_head(cls_emb),
            self.lane_head(cls_emb),
            self.cut_in_head(cls_emb),
        )

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
