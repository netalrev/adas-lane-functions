"""
src/training/dataset.py
=========================
PyTorch Dataset for loading MF windows and labels from an HDF5 file.

Segment-based split strategy
------------------------------
All samples from the same TFRecord segment are assigned to the same split
(train / val / test).  This prevents temporal leakage, where frames from the
same driving sequence appear in both training and evaluation.

Split procedure:
  1. Collect all unique segment names from the HDF5 file.
  2. Shuffle them with a fixed seed for reproducibility.
  3. Assign contiguous slices: first 80 % → train, next 10 % → val, last 10 % → test.
  4. For each split, keep only the HDF5 row indices whose segment name falls in
     the corresponding slice.

Memory strategy
---------------
All data for the requested split is loaded into NumPy arrays during __init__
and held in RAM.  This avoids per-item HDF5 file opens, which are expensive on
NFS/WSL file systems.  For the expected dataset sizes (< 1 M samples × [10, 18]
float32 = < 720 MB) this is tractable on a developer laptop.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig


# Lane assignment class index mapping.
# Must stay in sync with GTBuilder._lane_assignment() and heads.LaneAssignHead.
_LANE_CLASSES = [-2, -1, 0, 1, 2]
_LANE_CLASS_TO_IDX = {v: i for i, v in enumerate(_LANE_CLASSES)}


class MFDataset(Dataset):
    """
    Dataset wrapper around the HDF5 file produced by DatasetWriter.

    Parameters
    ----------
    h5_path : str
        Path to the HDF5 dataset file.
    split : str
        One of "train", "val", "test".
    cfg : DictConfig
        Hydra training config node.  Reads cfg.data.{train_fraction,
        val_fraction, seed}.
    """

    # Expose for use in loss-weight computation (class frequencies).
    LANE_CLASSES    = _LANE_CLASSES
    N_LANE_CLASSES  = len(_LANE_CLASSES)

    def __init__(self, h5_path: str, split: str, cfg: DictConfig) -> None:
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}. Must be train/val/test.")

        self._split = split

        with h5py.File(h5_path, "r") as f:
            # Decode segment names (h5py may return bytes or str)
            raw_names   = f["segment_names"][:]
            all_seg_names = np.array([
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in raw_names
            ])
            seg_ids = f["segment_ids"][:]   # index into all_seg_names

        # --- Segment-level split -------------------------------------------
        unique_segs = sorted(set(all_seg_names.tolist()))
        n_segs      = len(unique_segs)
        rng         = np.random.default_rng(int(cfg.data.seed))
        perm        = rng.permutation(n_segs)

        n_train = int(n_segs * float(cfg.data.train_fraction))
        n_val   = int(n_segs * float(cfg.data.val_fraction))

        if split == "train":
            chosen_indices = set(perm[:n_train].tolist())
        elif split == "val":
            chosen_indices = set(perm[n_train: n_train + n_val].tolist())
        else:  # test
            chosen_indices = set(perm[n_train + n_val:].tolist())

        chosen_names = {unique_segs[i] for i in chosen_indices}

        # Build boolean mask: sample belongs to this split?
        sample_seg_names = all_seg_names[seg_ids]
        mask = np.isin(sample_seg_names, list(chosen_names))
        indices = np.where(mask)[0]

        if len(indices) == 0:
            raise RuntimeError(
                f"Split '{split}' produced zero samples from {h5_path}. "
                "The dataset may be too small for the configured split fractions."
            )

        # --- Preload split data into RAM -----------------------------------
        with h5py.File(h5_path, "r") as f:
            self._mf     = f["mf_sequences"][indices].astype(np.float32)
            self._cipv   = f["cipv_labels"][indices].astype(np.int8)
            self._lane   = f["lane_labels"][indices].astype(np.int8)
            self._cut_in = f["cut_in_labels"][indices].astype(np.int8)

        n = len(indices)
        mb = self._mf.nbytes / 1024 / 1024
        print(
            f"[MFDataset] split={split:5s}  "
            f"segments={len(chosen_names):4d}  "
            f"samples={n:7,d}  "
            f"RAM={mb:.1f} MB"
        )

    def __len__(self) -> int:
        return len(self._mf)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns
        -------
        mf        : Tensor [T, D]  float32
        cipv      : Tensor scalar  float32   (0 or 1)
        lane_idx  : Tensor scalar  int64     (class index 0..4)
        cut_in    : Tensor scalar  float32   (0 or 1)
        """
        lane_raw = int(self._lane[idx])
        lane_idx = _LANE_CLASS_TO_IDX.get(lane_raw, 2)  # default to ego lane

        return (
            torch.from_numpy(self._mf[idx]),                        # [T, D]
            torch.tensor(int(self._cipv[idx]),   dtype=torch.float32),
            torch.tensor(lane_idx,               dtype=torch.long),
            torch.tensor(int(self._cut_in[idx]), dtype=torch.float32),
        )

    def class_weights(self) -> dict:
        """
        Compute positive-class weights for imbalanced binary tasks and
        per-class counts for the lane assignment task.

        Returns
        -------
        dict with keys:
            "cipv_pos_weight"   : float — (#negatives / #positives) for CIPV
            "cut_in_pos_weight" : float — (#negatives / #positives) for cut-in
            "lane_counts"       : np.ndarray[5] — sample count per lane class
        """
        cipv_pos   = float(self._cipv.sum())
        cipv_neg   = float(len(self._cipv) - cipv_pos)
        cipv_w     = cipv_neg / max(cipv_pos, 1.0)

        cut_in_pos = float(self._cut_in.sum())
        cut_in_neg = float(len(self._cut_in) - cut_in_pos)
        cut_in_w   = cut_in_neg / max(cut_in_pos, 1.0)

        lane_counts = np.zeros(self.N_LANE_CLASSES, dtype=np.int64)
        for raw, idx in _LANE_CLASS_TO_IDX.items():
            lane_counts[idx] = int((self._lane == raw).sum())

        return {
            "cipv_pos_weight":   cipv_w,
            "cut_in_pos_weight": cut_in_w,
            "lane_counts":       lane_counts,
        }
