"""
src/models/classification/heads.py
=====================================
Task-specific classification heads for the MFTransformer.

Each head takes the [CLS] embedding from the shared encoder ([B, d_model])
and produces task-specific raw logits.  Losses are applied externally so the
caller controls the loss function (BCEWithLogitsLoss, CrossEntropyLoss, etc.)
without needing the head to know about them.

Heads
-----
CIPVHead         Binary — CIPV / not-CIPV.             Output: [B, 1]
LaneAssignHead   5-class — {−2, −1, 0, +1, +2}.        Output: [B, 5]
CutInHead        Binary — cut-in / no cut-in.           Output: [B, 1]
"""

from __future__ import annotations

import torch.nn as nn


class CIPVHead(nn.Module):
    """
    Binary CIPV classification head.

    Architecture: LayerNorm → Dropout → Linear(d_model, 1).
    Pre-normalisation improves training stability and avoids
    a separate batch-norm layer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, cls_emb):
        """
        Parameters
        ----------
        cls_emb : Tensor [B, d_model]

        Returns
        -------
        Tensor [B, 1]  — raw logits (no sigmoid).
        """
        return self.net(cls_emb)


class LaneAssignHead(nn.Module):
    """
    5-class lane assignment head for classes {−2, −1, 0, +1, +2}.

    Class index mapping (consistent with MFDataset.LANE_CLASSES):
        0 → −2  (two lanes left)
        1 → −1  (one lane left)
        2 →  0  (ego lane)
        3 → +1  (one lane right)
        4 → +2  (two lanes right)
    """

    NUM_CLASSES = 5

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.NUM_CLASSES),
        )

    def forward(self, cls_emb):
        """
        Parameters
        ----------
        cls_emb : Tensor [B, d_model]

        Returns
        -------
        Tensor [B, 5]  — raw logits (no softmax).
        """
        return self.net(cls_emb)


class CutInHead(nn.Module):
    """
    Binary cut-in detection head.

    Cut-in is defined as a lane-assignment transition from a non-ego lane
    into the ego lane (lane_assignment ≠ 0 → 0) within the last N frames.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, cls_emb):
        """
        Parameters
        ----------
        cls_emb : Tensor [B, d_model]

        Returns
        -------
        Tensor [B, 1]  — raw logits (no sigmoid).
        """
        return self.net(cls_emb)
