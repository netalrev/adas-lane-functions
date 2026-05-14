"""
src/training/trainer.py
=========================
Trainer class for the MFTransformer multi-task model.

Responsibilities
----------------
- Multi-task loss computation (CIPV BCE + Lane CE + Cut-In BCE).
- AdamW optimisation with cosine LR schedule + linear warm-up.
- Gradient clipping for stable training on small models.
- Best-checkpoint saving monitored by a configurable validation metric.
- Per-epoch console logging and metrics history.

The Trainer is intentionally framework-agnostic: it does not depend on
Lightning, Ignite, or any other training framework.  This keeps the codebase
auditable and easy to extend for deployment engineers who may not be familiar
with those frameworks.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.evaluation.metrics import binary_metrics, multiclass_metrics
from src.evaluation.report import ReportWriter


class Trainer:
    """
    Manages the full training loop for MFTransformer.

    Parameters
    ----------
    model : nn.Module
        The MFTransformer instance.
    train_loader : DataLoader
    val_loader   : DataLoader
    cfg : DictConfig
        Hydra training config node (conf/training/default.yaml).
    device : torch.device
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          DictConfig,
        device:       torch.device,
    ) -> None:
        self._model        = model.to(device)
        self._train_loader = train_loader
        self._val_loader   = val_loader
        self._cfg          = cfg
        self._device       = device

        tcfg = cfg.training.training

        # ── Loss functions ────────────────────────────────────────────────
        cipv_pw   = torch.tensor([float(tcfg.cipv_pos_weight)]).to(device)
        cut_in_pw = torch.tensor([float(tcfg.cut_in_pos_weight)]).to(device)

        self._cipv_loss    = nn.BCEWithLogitsLoss(pos_weight=cipv_pw)
        self._lane_loss    = nn.CrossEntropyLoss(
            label_smoothing=float(tcfg.lane_label_smoothing)
        )
        self._cut_in_loss  = nn.BCEWithLogitsLoss(pos_weight=cut_in_pw)

        self._w_cipv   = float(tcfg.loss_weight_cipv)
        self._w_lane   = float(tcfg.loss_weight_lane)
        self._w_cut_in = float(tcfg.loss_weight_cut_in)
        self._grad_clip = float(tcfg.grad_clip)

        # ── Optimiser ─────────────────────────────────────────────────────
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=float(tcfg.learning_rate),
            weight_decay=float(tcfg.weight_decay),
        )

        # ── LR scheduler ─────────────────────────────────────────────────
        n_epochs = int(tcfg.max_epochs)
        warmup   = int(tcfg.warmup_epochs)
        schedule = str(tcfg.lr_schedule)

        self._scheduler = _build_scheduler(
            self._optimizer, schedule, n_epochs, warmup
        )

        # ── Checkpoint / reporting ────────────────────────────────────────
        ckpt_dir  = Path(cfg.training.output.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path   = ckpt_dir / "best_model.pt"
        self._best_metric = -float("inf")
        self._monitor     = str(cfg.training.output.save_best_metric)

        self._report_writer = ReportWriter(cfg.training.output.report_dir)
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> list[dict]:
        """
        Run the full training loop.

        Returns
        -------
        list[dict]
            One entry per epoch containing train and val metrics.
        """
        n_epochs = int(self._cfg.training.training.max_epochs)
        print(
            f"\n[Trainer] Starting training  "
            f"device={self._device}  "
            f"epochs={n_epochs}  "
            f"monitor={self._monitor}\n"
        )

        for epoch in range(1, n_epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            val_metrics   = self._evaluate(self._val_loader, split="val",
                                            epoch=epoch)

            row = {"epoch": epoch, **train_metrics, **val_metrics}
            self._history.append(row)

            # Checkpoint
            score = val_metrics.get(self._monitor, val_metrics.get("cipv_f1", 0.0))
            if score > self._best_metric:
                self._best_metric = score
                self._save_checkpoint(epoch, val_metrics)

            self._scheduler.step()
            self._log_epoch(epoch, n_epochs, row)

        print(f"\n[Trainer] Training complete.  "
              f"Best {self._monitor} = {self._best_metric:.4f}  "
              f"checkpoint → {self._ckpt_path}")
        return self._history

    def evaluate_split(self, loader: DataLoader, split: str) -> dict:
        """Evaluate on any DataLoader and write a report."""
        return self._evaluate(loader, split=split, epoch=None, write_report=True)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> dict:
        self._model.train()
        losses = []

        for mf, cipv_gt, lane_gt, cut_in_gt in self._train_loader:
            mf         = mf.to(self._device)
            cipv_gt    = cipv_gt.to(self._device).unsqueeze(1)   # [B, 1]
            lane_gt    = lane_gt.to(self._device)                 # [B]
            cut_in_gt  = cut_in_gt.to(self._device).unsqueeze(1)  # [B, 1]

            cipv_logits, lane_logits, cut_in_logits = self._model(mf)

            loss_cipv   = self._cipv_loss(cipv_logits,   cipv_gt)
            loss_lane   = self._lane_loss(lane_logits,   lane_gt)
            loss_cut_in = self._cut_in_loss(cut_in_logits, cut_in_gt)
            loss        = (self._w_cipv   * loss_cipv
                         + self._w_lane   * loss_lane
                         + self._w_cut_in * loss_cut_in)

            self._optimizer.zero_grad()
            loss.backward()
            if self._grad_clip > 0.0:
                nn.utils.clip_grad_norm_(self._model.parameters(), self._grad_clip)
            self._optimizer.step()

            losses.append(loss.item())

        return {"train_loss": round(float(np.mean(losses)), 5)}

    @torch.no_grad()
    def _evaluate(
        self,
        loader:       DataLoader,
        split:        str,
        epoch:        int | None,
        write_report: bool = False,
    ) -> dict:
        self._model.eval()

        cipv_probs_all:   list[np.ndarray] = []
        lane_preds_all:   list[np.ndarray] = []
        cut_in_probs_all: list[np.ndarray] = []
        cipv_gt_all:      list[np.ndarray] = []
        lane_gt_all:      list[np.ndarray] = []
        cut_in_gt_all:    list[np.ndarray] = []
        losses: list[float] = []

        for mf, cipv_gt, lane_gt, cut_in_gt in loader:
            mf         = mf.to(self._device)
            cipv_gt_d  = cipv_gt.to(self._device).unsqueeze(1)
            lane_gt_d  = lane_gt.to(self._device)
            cut_in_d   = cut_in_gt.to(self._device).unsqueeze(1)

            cipv_logits, lane_logits, cut_in_logits = self._model(mf)

            loss = (self._w_cipv   * self._cipv_loss(cipv_logits, cipv_gt_d)
                  + self._w_lane   * self._lane_loss(lane_logits, lane_gt_d)
                  + self._w_cut_in * self._cut_in_loss(cut_in_logits, cut_in_d))
            losses.append(loss.item())

            cipv_probs   = torch.sigmoid(cipv_logits).squeeze(1).cpu().numpy()
            lane_preds   = torch.argmax(lane_logits, dim=1).cpu().numpy()
            cut_in_probs = torch.sigmoid(cut_in_logits).squeeze(1).cpu().numpy()

            cipv_probs_all.append(cipv_probs)
            lane_preds_all.append(lane_preds)
            cut_in_probs_all.append(cut_in_probs)
            cipv_gt_all.append(cipv_gt.numpy())
            lane_gt_all.append(lane_gt.numpy())
            cut_in_gt_all.append(cut_in_gt.numpy())

            if write_report:
                self._report_writer.add_batch(
                    cipv_probs, lane_preds, cut_in_probs,
                    cipv_gt.numpy(), lane_gt.numpy(), cut_in_gt.numpy(),
                )

        cipv_prob_arr   = np.concatenate(cipv_probs_all)
        lane_pred_arr   = np.concatenate(lane_preds_all)
        cut_in_prob_arr = np.concatenate(cut_in_probs_all)
        cipv_gt_arr     = np.concatenate(cipv_gt_all)
        lane_gt_arr     = np.concatenate(lane_gt_all)
        cut_in_gt_arr   = np.concatenate(cut_in_gt_all)

        cipv_m   = binary_metrics(cipv_gt_arr,   cipv_prob_arr,  label="cipv")
        cut_in_m = binary_metrics(cut_in_gt_arr, cut_in_prob_arr, label="cut_in")
        lane_m   = multiclass_metrics(lane_gt_arr, lane_pred_arr, n_classes=5)

        metrics = {
            f"{split}_loss":         round(float(np.mean(losses)), 5),
            f"{split}_cipv_f1":      cipv_m["cipv_f1"],
            f"{split}_cipv_recall":  cipv_m["cipv_recall"],
            f"{split}_lane_macro_f1": lane_m["macro_f1"],
            f"{split}_lane_acc":     lane_m["accuracy"],
            f"{split}_cut_in_f1":    cut_in_m["cut_in_f1"],
        }

        # Convenience aliases used by checkpoint monitor (strip split prefix)
        metrics["cipv_f1"]       = cipv_m["cipv_f1"]
        metrics["lane_macro_f1"] = lane_m["macro_f1"]
        metrics["cut_in_f1"]     = cut_in_m["cut_in_f1"]
        metrics["total_loss"]    = metrics[f"{split}_loss"]

        if write_report:
            run_name = f"{split}" + (f"_epoch{epoch:03d}" if epoch else "")
            self._report_writer.write(run_name)

        return metrics

    def _save_checkpoint(self, epoch: int, val_metrics: dict) -> None:
        torch.save(
            {
                "epoch":       epoch,
                "model_state": self._model.state_dict(),
                "optim_state": self._optimizer.state_dict(),
                "val_metrics": val_metrics,
                "cfg":         dict(self._cfg.training),
            },
            self._ckpt_path,
        )

    @staticmethod
    def _log_epoch(epoch: int, n_epochs: int, row: dict) -> None:
        print(
            f"[{epoch:3d}/{n_epochs}]  "
            f"train={row.get('train_loss', 0):.4f}  "
            f"val={row.get('val_loss', 0):.4f}  "
            f"CIPV_F1={row.get('cipv_f1', 0):.4f}  "
            f"Lane_F1={row.get('lane_macro_f1', 0):.4f}  "
            f"CutIn_F1={row.get('cut_in_f1', 0):.4f}"
        )


# ---------------------------------------------------------------------------
# LR scheduler factory
# ---------------------------------------------------------------------------

def _build_scheduler(
    optimizer:   torch.optim.Optimizer,
    schedule:    str,
    n_epochs:    int,
    warmup:      int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a learning-rate scheduler, optionally with a linear warm-up phase
    implemented via LambdaLR.
    """
    if schedule == "cosine":
        base = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(n_epochs - warmup, 1), eta_min=1e-6
        )
    elif schedule == "step":
        base = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(n_epochs // 3, 1), gamma=0.3
        )
    else:  # constant
        base = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if warmup <= 0:
        return base

    def warmup_fn(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        return 1.0

    warmup_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, base], milestones=[warmup]
    )
