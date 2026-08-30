#!/usr/bin/env python3
"""
scripts/evaluate_perception.py
================================
CLI for the perception-quality evaluation harness (src/evaluation/perception_report.py).

Usage
-----
    # Evaluate every segment JSON already produced under src/data/
    python scripts/evaluate_perception.py

    # Evaluate specific files, and save the full report
    python scripts/evaluate_perception.py --json src/data/segment-A.json src/data/segment-B.json \\
        --output docs/baseline_metrics.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.perception_report import evaluate_segments  # noqa: E402


def _print_report(report: dict) -> None:
    s = report["summary"]
    print(f"\n{'='*70}\nPerception quality -- {s['n_segments']} segment(s), {s['n_frames']} frames\n{'='*70}")

    print("\nLane quality vs HD map:")
    print(f"  {'path':<16}{'valid_rate':>12}{'mean_abs_err_px':>18}")
    for path_type, m in s["lane_quality"].items():
        err = "n/a" if m["mean_abs_error_px"] is None else f"{m['mean_abs_error_px']:.1f}"
        print(f"  {path_type:<16}{m['valid_rate']:>12.3f}{err:>18}")

    se = s["state_estimation"]
    print("\nState estimation (EKF vs Waymo GT 3D boxes):")
    print(f"  n_matched={se['n_matched']}  pos_rmse_m={se['pos_rmse_m']}  "
          f"vel_rmse_mps={se['vel_rmse_mps']}  id_switches={se['id_switch_count']}")

    print("\nPer-segment breakdown:")
    for seg in report["per_segment"]:
        se = seg["state_estimation"]
        print(f"  [{seg['segment_name']}] frames={seg['n_frames']}  "
              f"pos_rmse_m={se['pos_rmse_m']}  n_matched={se['n_matched']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", nargs="*", help="Explicit list of segment JSON files to evaluate.")
    parser.add_argument("--segment-dir", default="src/data", help="Directory to auto-discover *.json in.")
    parser.add_argument("--output", default=None, help="Optional path to save the full report as JSON.")
    args = parser.parse_args()

    json_paths = args.json or sorted(glob.glob(os.path.join(args.segment_dir, "*.json")))
    if not json_paths:
        raise SystemExit(f"No segment JSON files found under {args.segment_dir!r}. Pass --json explicitly.")

    print(f"[evaluate_perception] Evaluating {len(json_paths)} segment JSON file(s)...")
    report = evaluate_segments(json_paths)
    _print_report(report)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved -> {args.output}")


if __name__ == "__main__":
    main()
