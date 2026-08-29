"""
src/pipeline/segment_resolver.py
==================================
Resolves the ordered list of .tfrecord segment paths to process from the
Hydra dataset config (list file, directory scan, or a single legacy path).
"""
from __future__ import annotations

import os
import random

from omegaconf import DictConfig


def resolve_segments(cfg: DictConfig) -> list[str]:
    """
    Return an ordered list of .tfrecord paths to process.

    Priority (first non-empty wins):
      1. dataset.segment_list  — explicit .txt file (one path per line, # ok)
      2. dataset.segment_dir   — auto-discover every *.tfrecord in a directory
      3. dataset.tfrecord_path — single legacy path

    Applies max_segments cap and optional shuffle.
    """
    ds = cfg.dataset
    paths: list[str] = []

    seg_list = getattr(ds, "segment_list", "")
    seg_dir  = getattr(ds, "segment_dir",  "")

    if seg_list:
        # Read explicit list file; skip blank lines and comments
        with open(seg_list) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    paths.append(line)
        print(f"[batch] Segment list file: {seg_list} → {len(paths)} entries")

    elif seg_dir:
        # Auto-discover all .tfrecord files recursively
        for root, _, files in os.walk(seg_dir):
            for fname in sorted(files):
                if fname.endswith(".tfrecord"):
                    paths.append(os.path.join(root, fname))
        print(f"[batch] Discovered {len(paths)} tfrecords under: {seg_dir}")

    else:
        single = getattr(ds, "tfrecord_path", "")
        if single:
            paths = [single]
        else:
            raise ValueError(
                "No segment source configured. Set one of: "
                "dataset.tfrecord_path, dataset.segment_dir, dataset.segment_list"
            )

    if not paths:
        raise ValueError("Segment source resolved to zero files.")

    # Filter out paths that don't exist on disk yet (e.g. not downloaded yet).
    # This allows a segment_list to contain future segments without crashing.
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[batch] WARNING: file not found, skipping — {p}")
        paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(
            "All resolved segments are missing from disk.\n"
            "Download them first — see 'dataset.gcs_prefix' in conf/config.yaml "
            "and run:\n"
            "  gsutil -m cp '<gcs_prefix>/segment-*.tfrecord' <local_dir>/"
        )

    # Optional shuffle before capping (for reproducible random sampling)
    shuffle = getattr(ds, "shuffle_segments", False)
    if shuffle:
        random.shuffle(paths)

    # Cap to max_segments
    max_seg = getattr(ds, "max_segments", 1)
    if max_seg is not None and len(paths) > max_seg:
        paths = paths[:max_seg]

    return paths
