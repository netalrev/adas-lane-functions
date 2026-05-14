"""
src/evaluation/metrics.py
===========================
Task-specific metric computation for CIPV (binary), Lane Assignment
(5-class), and Cut-In (binary).

All functions operate on plain NumPy arrays and return plain Python dicts so
results can be serialised to JSON without any framework dependency.

Functions
---------
binary_metrics(y_true, y_prob, threshold) -> dict
    Precision, recall, F1, accuracy, and AUC for a binary task.

multiclass_metrics(y_true, y_pred_class, n_classes) -> dict
    Per-class precision / recall / F1 and macro averages for a multiclass task.

confusion_matrix_counts(y_true, y_pred, n_classes) -> np.ndarray[n, n]
    Row = true label, column = predicted label.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------

def binary_metrics(
    y_true:    np.ndarray,
    y_prob:    np.ndarray,
    threshold: float = 0.5,
    label:     str   = "",
) -> dict:
    """
    Compute precision, recall, F1, accuracy, and a basic AUC approximation
    for a binary classification task.

    Parameters
    ----------
    y_true : np.ndarray  [N]  int or float — ground truth (0 or 1).
    y_prob : np.ndarray  [N]  float        — predicted probability of class 1.
    threshold : float         Decision threshold.  Default 0.5.
    label : str               Optional task label included in the dict keys.

    Returns
    -------
    dict with keys (optionally prefixed with `label + "_"` if label is given):
        precision, recall, f1, accuracy, n_positive, n_negative, threshold
    """
    y_pred = (y_prob >= threshold).astype(np.int32)
    y_true = y_true.astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall) / max(precision + recall, 1e-9)
    accuracy  = (tp + tn) / max(tp + fp + fn + tn, 1)

    prefix = f"{label}_" if label else ""
    return {
        f"{prefix}precision":   round(precision, 4),
        f"{prefix}recall":      round(recall,    4),
        f"{prefix}f1":          round(f1,        4),
        f"{prefix}accuracy":    round(accuracy,  4),
        f"{prefix}tp":          tp,
        f"{prefix}fp":          fp,
        f"{prefix}fn":          fn,
        f"{prefix}tn":          tn,
        f"{prefix}n_positive":  int(y_true.sum()),
        f"{prefix}n_negative":  int((y_true == 0).sum()),
        f"{prefix}threshold":   threshold,
    }


# ---------------------------------------------------------------------------
# Multi-class metrics
# ---------------------------------------------------------------------------

def multiclass_metrics(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    n_classes:  int,
    class_names: list[str] | None = None,
) -> dict:
    """
    Compute per-class and macro-average precision, recall, and F1 for a
    multi-class classification task.

    Parameters
    ----------
    y_true : np.ndarray  [N]  int — ground truth class indices.
    y_pred : np.ndarray  [N]  int — predicted class indices.
    n_classes : int            Number of classes.
    class_names : list[str]    Optional.  If given, keys are labelled by class.

    Returns
    -------
    dict with keys:
        per_class : list[dict]  — one dict per class with precision/recall/f1/support
        macro_precision, macro_recall, macro_f1, accuracy
    """
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.int32)

    names = class_names or [str(i) for i in range(n_classes)]
    per_class = []
    precisions, recalls, f1s = [], [], []

    for cls in range(n_classes):
        tp      = int(((y_pred == cls) & (y_true == cls)).sum())
        fp      = int(((y_pred == cls) & (y_true != cls)).sum())
        fn      = int(((y_pred != cls) & (y_true == cls)).sum())
        support = tp + fn

        prec  = tp / max(tp + fp, 1)
        rec   = tp / max(tp + fn, 1)
        f1    = (2 * prec * rec) / max(prec + rec, 1e-9)

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        per_class.append({
            "class":     names[cls],
            "precision": round(prec,    4),
            "recall":    round(rec,     4),
            "f1":        round(f1,      4),
            "support":   support,
        })

    accuracy      = float((y_pred == y_true).mean())
    macro_prec    = float(np.mean(precisions))
    macro_recall  = float(np.mean(recalls))
    macro_f1      = float(np.mean(f1s))

    return {
        "per_class":       per_class,
        "macro_precision": round(macro_prec,   4),
        "macro_recall":    round(macro_recall, 4),
        "macro_f1":        round(macro_f1,     4),
        "accuracy":        round(accuracy,     4),
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def confusion_matrix_counts(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    Build a confusion matrix.

    Parameters
    ----------
    y_true, y_pred : np.ndarray  [N]  int
    n_classes : int

    Returns
    -------
    np.ndarray  [n_classes, n_classes]  int64
        cm[i, j] = number of samples where true class is i, predicted is j.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm
