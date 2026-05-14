"""
src/evaluation/report.py
==========================
Structured evaluation report writer.

Produces two artefacts per evaluation run:
  1. A human-readable text report saved to <report_dir>/<run_name>.txt
  2. NumPy arrays for each confusion matrix saved as <run_name>_cm_*.npy

The report covers all three tasks: CIPV, Lane Assignment, and Cut-In.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from src.evaluation.metrics import (
    binary_metrics,
    multiclass_metrics,
    confusion_matrix_counts,
)

# Lane class names for readable confusion matrix headers.
_LANE_CLASS_NAMES = ["L2(-2)", "L1(-1)", "EGO(0)", "R1(+1)", "R2(+2)"]


class ReportWriter:
    """
    Collects per-batch predictions, computes final metrics, and writes reports.

    Usage
    -----
        rw = ReportWriter(report_dir="outputs/reports")
        # For each evaluation batch:
        rw.add_batch(cipv_probs, lane_probs, cut_in_probs, cipv_gt, lane_gt, cut_in_gt)
        # After all batches:
        metrics = rw.write(run_name="val_epoch_05")

    Parameters
    ----------
    report_dir : str | Path
        Directory where text reports and numpy arrays are saved.
    """

    def __init__(self, report_dir: str | Path) -> None:
        self._dir = Path(report_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._reset()

    def _reset(self) -> None:
        self._cipv_prob:    list[np.ndarray] = []
        self._cipv_gt:      list[np.ndarray] = []
        self._lane_pred:    list[np.ndarray] = []
        self._lane_gt:      list[np.ndarray] = []
        self._cut_in_prob:  list[np.ndarray] = []
        self._cut_in_gt:    list[np.ndarray] = []

    def add_batch(
        self,
        cipv_prob:    np.ndarray,
        lane_pred:    np.ndarray,
        cut_in_prob:  np.ndarray,
        cipv_gt:      np.ndarray,
        lane_gt:      np.ndarray,
        cut_in_gt:    np.ndarray,
    ) -> None:
        """
        Accumulate one evaluation batch.

        Parameters
        ----------
        cipv_prob   : [B] float — CIPV predicted probabilities (after sigmoid).
        lane_pred   : [B] int   — Lane assignment predicted class indices (0..4).
        cut_in_prob : [B] float — Cut-in predicted probabilities (after sigmoid).
        cipv_gt     : [B] int   — CIPV ground truth (0 or 1).
        lane_gt     : [B] int   — Lane assignment ground truth class indices (0..4).
        cut_in_gt   : [B] int   — Cut-In ground truth (0 or 1).
        """
        self._cipv_prob.append(np.asarray(cipv_prob).ravel())
        self._cipv_gt.append(np.asarray(cipv_gt).ravel())
        self._lane_pred.append(np.asarray(lane_pred).ravel())
        self._lane_gt.append(np.asarray(lane_gt).ravel())
        self._cut_in_prob.append(np.asarray(cut_in_prob).ravel())
        self._cut_in_gt.append(np.asarray(cut_in_gt).ravel())

    def write(self, run_name: str = "") -> dict:
        """
        Compute all metrics, write the text report and confusion matrices,
        and reset the internal buffers.

        Parameters
        ----------
        run_name : str
            Label for this evaluation run used in file names and report headers.
            Defaults to a timestamp.

        Returns
        -------
        dict
            Flat metrics dict (values are Python scalars) — suitable for
            logging to Comet ML or stdout.
        """
        if not run_name:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        cipv_prob   = np.concatenate(self._cipv_prob)
        cipv_gt     = np.concatenate(self._cipv_gt)
        lane_pred   = np.concatenate(self._lane_pred)
        lane_gt     = np.concatenate(self._lane_gt)
        cut_in_prob = np.concatenate(self._cut_in_prob)
        cut_in_gt   = np.concatenate(self._cut_in_gt)

        # ── Per-task metrics ──────────────────────────────────────────────
        cipv_m    = binary_metrics(cipv_gt,   cipv_prob,  label="cipv")
        cut_in_m  = binary_metrics(cut_in_gt, cut_in_prob, label="cut_in")
        lane_m    = multiclass_metrics(
            lane_gt, lane_pred,
            n_classes=5,
            class_names=_LANE_CLASS_NAMES,
        )

        # ── Confusion matrices ────────────────────────────────────────────
        cipv_cm   = confusion_matrix_counts(cipv_gt.astype(int),
                                            (cipv_prob >= 0.5).astype(int), 2)
        lane_cm   = confusion_matrix_counts(lane_gt.astype(int),
                                            lane_pred.astype(int), 5)
        cut_in_cm = confusion_matrix_counts(cut_in_gt.astype(int),
                                            (cut_in_prob >= 0.5).astype(int), 2)

        # Save confusion matrices
        np.save(self._dir / f"{run_name}_cm_cipv.npy",   cipv_cm)
        np.save(self._dir / f"{run_name}_cm_lane.npy",   lane_cm)
        np.save(self._dir / f"{run_name}_cm_cut_in.npy", cut_in_cm)

        # ── Flat metrics dict ─────────────────────────────────────────────
        flat = {**cipv_m, **cut_in_m}
        flat["lane_macro_f1"]       = lane_m["macro_f1"]
        flat["lane_macro_precision"]= lane_m["macro_precision"]
        flat["lane_macro_recall"]   = lane_m["macro_recall"]
        flat["lane_accuracy"]       = lane_m["accuracy"]
        for pc in lane_m["per_class"]:
            safe_name = pc["class"].replace("(", "").replace(")", "").replace("-", "m").replace("+", "p")
            flat[f"lane_f1_{safe_name}"] = pc["f1"]

        # ── Text report ───────────────────────────────────────────────────
        txt_path = self._dir / f"{run_name}.txt"
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(self._build_text_report(
                run_name, cipv_m, lane_m, cut_in_m,
                cipv_cm, lane_cm, cut_in_cm,
            ))

        # Also save flat metrics as JSON for programmatic access
        json_path = self._dir / f"{run_name}_metrics.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(flat, fh, indent=2)

        print(f"[ReportWriter] Report saved → {txt_path}")
        self._reset()
        return flat

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_text_report(
        run_name:   str,
        cipv_m:     dict,
        lane_m:     dict,
        cut_in_m:   dict,
        cipv_cm:    np.ndarray,
        lane_cm:    np.ndarray,
        cut_in_cm:  np.ndarray,
    ) -> str:
        lines = []
        sep   = "=" * 60

        lines.append(sep)
        lines.append(f"  Evaluation Report: {run_name}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(sep)

        # CIPV
        lines.append("\n── CIPV (Binary) ──────────────────────────────────")
        lines.append(f"  Precision : {cipv_m['cipv_precision']:.4f}")
        lines.append(f"  Recall    : {cipv_m['cipv_recall']:.4f}")
        lines.append(f"  F1        : {cipv_m['cipv_f1']:.4f}")
        lines.append(f"  Accuracy  : {cipv_m['cipv_accuracy']:.4f}")
        lines.append(f"  Positives : {cipv_m['cipv_n_positive']} / "
                     f"{cipv_m['cipv_n_positive'] + cipv_m['cipv_n_negative']}")
        lines.append(_format_binary_cm(cipv_cm, "CIPV"))

        # Lane Assignment
        lines.append("\n── Lane Assignment (5-class) ──────────────────────")
        lines.append(f"  Macro F1        : {lane_m['macro_f1']:.4f}")
        lines.append(f"  Macro Precision : {lane_m['macro_precision']:.4f}")
        lines.append(f"  Macro Recall    : {lane_m['macro_recall']:.4f}")
        lines.append(f"  Accuracy        : {lane_m['accuracy']:.4f}")
        lines.append("\n  Per-class breakdown:")
        lines.append(f"  {'Class':<12} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>10}")
        lines.append("  " + "-" * 50)
        for pc in lane_m["per_class"]:
            lines.append(
                f"  {pc['class']:<12} {pc['precision']:>8.4f} "
                f"{pc['recall']:>8.4f} {pc['f1']:>8.4f} {pc['support']:>10d}"
            )
        lines.append(_format_multiclass_cm(lane_cm, _LANE_CLASS_NAMES, "Lane Assignment"))

        # Cut-In
        lines.append("\n── Cut-In (Binary) ─────────────────────────────────")
        lines.append(f"  Precision : {cut_in_m['cut_in_precision']:.4f}")
        lines.append(f"  Recall    : {cut_in_m['cut_in_recall']:.4f}")
        lines.append(f"  F1        : {cut_in_m['cut_in_f1']:.4f}")
        lines.append(f"  Accuracy  : {cut_in_m['cut_in_accuracy']:.4f}")
        lines.append(f"  Positives : {cut_in_m['cut_in_n_positive']} / "
                     f"{cut_in_m['cut_in_n_positive'] + cut_in_m['cut_in_n_negative']}")
        lines.append(_format_binary_cm(cut_in_cm, "Cut-In"))

        lines.append("\n" + sep)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_binary_cm(cm: np.ndarray, label: str) -> str:
    lines = [f"\n  Confusion Matrix ({label}):"]
    lines.append("              Pred-0   Pred-1")
    lines.append(f"  True-0  {cm[0,0]:>8d} {cm[0,1]:>8d}")
    lines.append(f"  True-1  {cm[1,0]:>8d} {cm[1,1]:>8d}")
    return "\n".join(lines)


def _format_multiclass_cm(cm: np.ndarray, names: list[str], label: str) -> str:
    col_w = 9
    lines = [f"\n  Confusion Matrix ({label}):"]
    header = "  " + " " * 10 + "".join(f"{n[:col_w]:>{col_w}}" for n in names)
    lines.append(header)
    for i, row_name in enumerate(names):
        row = "  " + f"{row_name:<10}" + "".join(f"{cm[i,j]:>{col_w}d}" for j in range(len(names)))
        lines.append(row)
    return "\n".join(lines)
