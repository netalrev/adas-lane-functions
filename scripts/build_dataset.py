"""
scripts/build_dataset.py
=========================
Offline dataset builder for ADAS lane-function classification.

Reads a list of pre-processed segment JSON files, replays each frame,
assembles the Measurement Feature (MF) vector per track, derives GT labels
from Waymo 3D boxes, and writes the dataset to an HDF5 file.

Usage
-----
    python scripts/build_dataset.py \\
        --segments_list segments_to_run.txt \\
        --json_dir      src/data \\
        --output        outputs/dataset.h5 \\
        --cfg           conf/features/mf.yaml

The pipeline JSON files are expected to follow the naming convention:
    {json_dir}/{segment_name}.json

Each JSON file must contain the frame-level keys written by pipeline_input.py:
    timestamp, tracks, boxes_3d, host_lane, drivable_path

GT labels are computed per track per frame using GTBuilder (algorithmic,
no neural network needed at dataset-building time).

Progress is logged to stdout every 50 segments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Ensure the project root is on sys.path when running as a standalone script
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.mf_assembler   import MFAssembler
from src.features.gt_builder      import GTBuilder
from src.features.dataset_writer  import DatasetWriter


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline dataset builder: JSON segments → HDF5 training set"
    )
    parser.add_argument(
        "--segments_list",
        default="segments_to_run.txt",
        help="Path to text file with one segment name per line.",
    )
    parser.add_argument(
        "--json_dir",
        default="src/data",
        help="Directory that contains the per-segment JSON files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/dataset.h5",
        help="Path for the output HDF5 file.",
    )
    parser.add_argument(
        "--cfg",
        default="conf/features/mf.yaml",
        help="Path to the Hydra MF feature config YAML.",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=1,
        help="Flush the HDF5 writer every N segments (default: every segment).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# JSON reconstruction helpers
# ---------------------------------------------------------------------------

def _load_segment_json(json_path: Path) -> list[dict] | None:
    """
    Load and return the frame list from a segment JSON file.
    Returns None on parse errors so the caller can skip the segment.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and "frames" in data:
            return data["frames"]
        if isinstance(data, list):
            return data
        print(f"[WARN] Unexpected JSON structure in {json_path}, skipping.")
        return None
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARN] Failed to load {json_path}: {exc}")
        return None


def _extract_lane_results(frame_data: dict) -> dict:
    """
    Reconstruct the lane_results dict expected by MFAssembler from the
    keys stored in the pipeline JSON.
    """
    return {
        "host_lane": frame_data.get("host_lane", {}),
        "drivable_path": frame_data.get("drivable_path", {}),
    }


# ---------------------------------------------------------------------------
# Per-segment processing
# ---------------------------------------------------------------------------

def _process_segment(
    segment_name: str,
    json_path:    Path,
    assembler:    MFAssembler,
    gt_builder:   GTBuilder,
    writer:       DatasetWriter,
) -> int:
    """
    Process one segment and return the number of samples staged.

    Parameters
    ----------
    segment_name : str
        Segment identifier (used as the /segment_names key in HDF5).
    json_path : Path
        Path to the segment JSON file.
    assembler : MFAssembler
        Feature assembler (will be reset after this segment).
    gt_builder : GTBuilder
        GT label builder (will be reset after this segment).
    writer : DatasetWriter
        HDF5 dataset writer (samples are staged, not yet flushed).

    Returns
    -------
    int
        Number of samples staged during this segment.
    """
    frames = _load_segment_json(json_path)
    if frames is None:
        return 0

    assembler.reset()
    gt_builder.reset()

    staged = 0

    for frame_idx, frame_data in enumerate(frames):
        tracks   = frame_data.get("tracks",   [])
        boxes_3d = frame_data.get("boxes_3d", [])

        if not tracks:
            continue

        lane_results = _extract_lane_results(frame_data)

        # image shape: try to read from JSON, fall back to Waymo FRONT camera default
        img_h = int(frame_data.get("image_height", 886))
        img_w = int(frame_data.get("image_width",  1920))
        img_shape = (img_h, img_w)

        # --- Feature assembly ---
        completed_windows = assembler.update(tracks, lane_results, img_shape)
        if not completed_windows:
            continue

        # --- GT label derivation ---
        labels = gt_builder.compute_labels(boxes_3d, tracks)
        if not labels:
            continue

        # --- Stage samples where both MF window and GT label exist ---
        for track_id, mf_window in completed_windows.items():
            if track_id not in labels:
                continue
            lbl = labels[track_id]
            writer.add_sample(
                mf              = mf_window,
                cipv            = lbl["cipv"],
                lane_assignment = lbl["lane_assignment"],
                cut_in          = lbl["cut_in"],
                segment_name    = segment_name,
                track_id        = track_id,
                frame_idx       = frame_idx,
            )
            staged += 1

    return staged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Load feature config
    cfg_path = Path(args.cfg)
    if not cfg_path.exists():
        print(f"[ERROR] Feature config not found: {cfg_path}")
        sys.exit(1)
    cfg = OmegaConf.load(cfg_path)

    # Read segment list
    seg_list_path = Path(args.segments_list)
    if not seg_list_path.exists():
        print(f"[ERROR] Segment list not found: {seg_list_path}")
        sys.exit(1)

    with open(seg_list_path, "r", encoding="utf-8") as fh:
        segment_names = [ln.strip() for ln in fh if ln.strip()]

    if not segment_names:
        print("[ERROR] Segment list is empty.")
        sys.exit(1)

    print(f"[INFO] Building dataset from {len(segment_names)} segment(s).")
    print(f"[INFO] JSON dir : {args.json_dir}")
    print(f"[INFO] Output   : {args.output}")

    assembler  = MFAssembler(cfg)
    gt_builder = GTBuilder(cfg)
    writer     = DatasetWriter(args.output)

    t0           = time.time()
    total_staged = 0
    total_flushed = 0
    skipped      = 0

    for seg_idx, segment_name in enumerate(segment_names, start=1):
        json_path = Path(args.json_dir) / f"{segment_name}.json"
        if not json_path.exists():
            print(f"[WARN] JSON not found, skipping: {json_path}")
            skipped += 1
            continue

        staged = _process_segment(
            segment_name, json_path, assembler, gt_builder, writer
        )
        total_staged += staged

        # Flush to disk every N segments
        if seg_idx % args.flush_every == 0:
            total_flushed = writer.flush()

        if seg_idx % 50 == 0 or seg_idx == len(segment_names):
            elapsed = time.time() - t0
            print(
                f"[INFO] {seg_idx}/{len(segment_names)} segments processed | "
                f"staged={total_staged} | flushed={total_flushed} | "
                f"elapsed={elapsed:.0f}s"
            )

    # Final flush for remaining staged samples
    total_flushed = writer.flush()

    elapsed = time.time() - t0
    print(
        f"\n[DONE] Dataset build complete.\n"
        f"  Segments processed : {len(segment_names) - skipped}/{len(segment_names)}\n"
        f"  Samples in dataset : {total_flushed}\n"
        f"  Output file        : {args.output}\n"
        f"  Elapsed time       : {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
