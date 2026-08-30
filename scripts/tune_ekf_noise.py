#!/usr/bin/env python3
"""
scripts/tune_ekf_noise.py
===========================
Grid-search VehicleEKF process/measurement noise (q_pos, q_vel, r_pos) by
REPLAYING already-computed detections from existing segment JSON files
through a fresh VehicleTrackManager -- no YOLO/lane-model re-run needed, so
a full grid sweep takes seconds instead of minutes.

Camera calibration is extracted for real from frame 0 of the matching
.tfrecord (same as pipeline_input.py) -- an earlier version of this script
approximated it with CameraCalibration.default_front(), and that mismatch
was enough to flip the ranking relative to a true end-to-end run.  Pass
--tfrecord paths aligned 1:1 with --json to get the real geometry; without
them this falls back to the approximation (fast, but re-validate any
winner against a real pipeline run before trusting it).

Usage
-----
    python scripts/tune_ekf_noise.py \\
        --json src/data/segment-A.json --tfrecord src/data/segment-A.tfrecord
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from omegaconf import OmegaConf

from src.detectors.vehicle.detector import Detection
from src.evaluation.perception_report import evaluate_state_estimation
from src.measurements.vehicle_track_manager import VehicleTrackManager
from src.visualization.visualizer import CameraCalibration

# Defaults from conf/model/vehicle_ekf.yaml -- overridden per-candidate below.
_BASE_CFG = dict(
    max_age_tentative=3, max_age_confirmed=20, min_hits=2,
    iou_threshold=0.25, reacquire_dist_px=100, default_dt=0.1,
    fx=2000.0, fy=2000.0,
    process_noise_pos=0.10, process_noise_vel=1.00, process_noise_heading=0.05,
    process_noise_size=0.05, process_noise_length=0.10,
    measurement_noise_pos=0.50, measurement_noise_aspect=0.05,
)


def _real_calibration(tfrecord_path: str) -> CameraCalibration:
    """Extract the true camera calibration from frame 0, same as pipeline_input.py."""
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    for data in dataset.take(1):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(data.numpy()))
        for cam_cal in frame.context.camera_calibrations:
            if cam_cal.name == open_dataset.CameraName.FRONT:
                return CameraCalibration.from_waymo_camera(cam_cal)
    raise RuntimeError(f"No FRONT camera calibration found in {tfrecord_path}")


def _replay_segment(frames: list[dict], cfg_overrides: dict, calib: CameraCalibration) -> list[dict]:
    """Re-run tracking on one segment's stored detections with the given noise cfg."""
    cfg = OmegaConf.create({**_BASE_CFG, **cfg_overrides})
    mgr = VehicleTrackManager(cfg)
    mgr.set_camera_params(calib.K, calib.R_vc, calib.t_vc)

    out_frames = []
    prev_ts = None
    for frame in frames:
        dets = [
            Detection(bbox_xyxy=np.array(d["bbox_xyxy"], dtype=np.float64),
                      confidence=d["confidence"], class_id=d["class_id"],
                      class_name=d.get("class_name", ""))
            for d in frame.get("detections", [])
        ]
        ts = float(frame.get("timestamp", 0.0))
        dt = (ts - prev_ts) if prev_ts is not None else None
        prev_ts = ts

        tracks = mgr.update(dets, dt=dt)
        out_frames.append({
            "timestamp":          frame.get("timestamp"),
            "boxes_3d":           frame.get("boxes_3d", []),
            "vehicle_ekf_tracks": [t.to_dict() for t in tracks],
        })
    return out_frames


def _weighted_mean(pairs: list[tuple]) -> float | None:
    vals = [(v, w) for v, w in pairs if v is not None and w]
    return round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 3) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", nargs="*", help="Segment JSON files to replay.")
    parser.add_argument("--tfrecord", nargs="*", default=None,
                         help="Matching .tfrecord paths (same order as --json) for real calibration.")
    parser.add_argument("--segment-dir", default="src/data")
    parser.add_argument("--top", type=int, default=10, help="How many candidates to print.")
    args = parser.parse_args()

    json_paths = args.json or sorted(glob.glob(os.path.join(args.segment_dir, "*.json")))
    all_frames = []
    for p in json_paths:
        with open(p) as f:
            all_frames.append(json.load(f))

    if args.tfrecord:
        assert len(args.tfrecord) == len(json_paths), "--tfrecord must align 1:1 with --json"
        print("[tune_ekf_noise] Extracting real per-segment calibration...")
        calibs = [_real_calibration(p) for p in args.tfrecord]
    else:
        print("[tune_ekf_noise] WARNING: no --tfrecord given, approximating with "
              "CameraCalibration.default_front() -- re-validate any winner end-to-end.")
        calibs = [CameraCalibration.default_front()] * len(json_paths)

    print(f"[tune_ekf_noise] Loaded {len(json_paths)} segment(s), "
          f"{sum(len(f) for f in all_frames)} frames total.")

    grid = {
        "process_noise_pos":     [0.05, 0.10, 0.20],
        "process_noise_vel":     [0.5, 1.0, 2.0],
        "measurement_noise_pos": [0.3, 0.5, 1.0],
    }
    keys = list(grid.keys())
    n_candidates = int(np.prod([len(v) for v in grid.values()]))
    print(f"[tune_ekf_noise] Replaying {n_candidates} candidate(s)...")

    results = []
    for combo in itertools.product(*grid.values()):
        overrides = dict(zip(keys, combo))
        pos_pairs, vel_pairs = [], []
        switches = matched = 0
        for frames, calib in zip(all_frames, calibs):
            replayed = _replay_segment(frames, overrides, calib)
            m = evaluate_state_estimation(replayed)
            pos_pairs.append((m["pos_rmse_m"], m["n_matched"]))
            vel_pairs.append((m["vel_rmse_mps"], m["n_matched"]))
            switches += m["id_switch_count"]
            matched  += m["n_matched"]

        results.append({
            **overrides,
            "pos_rmse_m":      _weighted_mean(pos_pairs),
            "vel_rmse_mps":    _weighted_mean(vel_pairs),
            "id_switch_count": switches,
            "n_matched":       matched,
        })

    results.sort(key=lambda r: (r["pos_rmse_m"] is None, r["pos_rmse_m"] or 1e9))

    print(f"\n{'q_pos':>7}{'q_vel':>7}{'r_pos':>7}{'pos_rmse_m':>12}{'vel_rmse_mps':>14}{'id_sw':>8}{'matched':>9}")
    for r in results[:args.top]:
        print(f"{r['process_noise_pos']:>7}{r['process_noise_vel']:>7}{r['measurement_noise_pos']:>7}"
              f"{str(r['pos_rmse_m']):>12}{str(r['vel_rmse_mps']):>14}{r['id_switch_count']:>8}{r['n_matched']:>9}")

    baseline = next(
        r for r in results
        if r["process_noise_pos"] == 0.10 and r["process_noise_vel"] == 1.00 and r["measurement_noise_pos"] == 0.50
    )
    print(f"\ncurrent config (0.10, 1.00, 0.50): pos_rmse_m={baseline['pos_rmse_m']}  "
          f"vel_rmse_mps={baseline['vel_rmse_mps']}  id_switches={baseline['id_switch_count']}")


if __name__ == "__main__":
    main()
