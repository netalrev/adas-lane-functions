"""
src/evaluation/perception_report.py
=====================================
Perception-quality evaluation -- measures how good the lane detectors, the
vehicle EKF, and the lane-relation measurements are, using data already
present in pipeline_input.py's per-segment JSON output.

Distinct from src/evaluation/metrics.py, which scores the *downstream*
CIPV / Lane-Assignment / Cut-in classifier. This module scores the
*perception layer feeding that classifier* -- the thing that needs to be
trustworthy before training on top of it.

Public API
----------
    evaluate_segment(json_path) -> dict
        Full metrics for one segment's JSON output.

    evaluate_segments(json_paths) -> dict
        {"summary": <frame-count-weighted aggregate>, "per_segment": [...]}

Metrics
-------
lane_quality[path_type]   (kinematic, drivable_path, host_lane)
    valid_rate          -- fraction of frames with a valid centerline
    comparable_to_hdmap -- False for "kinematic": its serialized "center" is
                            in Vehicle-Frame metres, not pixels like the other
                            three path types, so a pixel-space diff against
                            hdmap would silently compare the wrong units.
                            Comparing it properly needs the per-segment camera
                            calibration, which isn't stored in the JSON output
                            (follow-up: convert hdmap px -> VF metres instead).
    n_compared_frames   -- frames where both this path and hdmap are valid
    mean_abs_error_px   -- mean |x| deviation from the hdmap centerline,
                            matched by image row (v); None if not comparable
                            or no overlapping frames

hdmap_quality
    valid_rate, confidence_mean  -- sanity check on the "ground truth" itself

state_estimation
    n_matched        -- EKF-track-to-GT-box matches within match_dist_m
    pos_rmse_m       -- RMSE of matched position error (m) -- the most
                        reliable of these metrics
    vel_rmse_mps     -- RMSE of EKF velocity vs GT finite-difference velocity
    id_switch_count  -- times a GT id's matched track_id changed

    LIMITATION: matching is greedy nearest-neighbor per frame with no
    persistence, so when two GT vehicles are close together the match can
    flicker between them. Treat vel_rmse_mps / id_switch_count as noisy
    upper bounds, not precise measurements -- pos_rmse_m is unaffected since
    it only uses the winning (nearest) match. A persistent/Hungarian
    association (or reusing GTBuilder's matcher) is the natural v2 fix.

lane_relations[path_type]  (kinematic, drivable_path, host_lane, hdmap)
    valid_rate, inside_rate  -- descriptive stats from the already-computed
                                 per-frame LaneRelationMeasurer output
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

_GT_VEHICLE_TYPE = 1
_PATH_TYPES = ["kinematic", "drivable_path", "host_lane"]

# "kinematic" is serialized in Vehicle-Frame metres; the rest are pixel-space
# (see the module docstring's lane_quality section for why this matters).
_PIXEL_SPACE_PATH_TYPES = ["drivable_path", "host_lane"]


# ---------------------------------------------------------------------------
# Lane quality vs HD map
# ---------------------------------------------------------------------------

def _lane_agreement_px(center_a: list, center_b: list) -> float | None:
    """Mean absolute pixel-x deviation between two centerlines, matched by row (v)."""
    a = np.asarray(center_a, dtype=np.float64)
    b = np.asarray(center_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or len(a) < 2 or len(b) < 2:
        return None
    a = a[np.argsort(a[:, 1])]
    b = b[np.argsort(b[:, 1])]
    v_lo, v_hi = max(a[0, 1], b[0, 1]), min(a[-1, 1], b[-1, 1])
    if v_hi <= v_lo:
        return None
    v_c = np.linspace(v_lo, v_hi, 20)
    xa = np.interp(v_c, a[:, 1], a[:, 0])
    xb = np.interp(v_c, b[:, 1], b[:, 0])
    return float(np.mean(np.abs(xa - xb)))


def evaluate_lane_quality(frames: list[dict]) -> dict:
    result = {}
    for path_type in _PATH_TYPES:
        comparable = path_type in _PIXEL_SPACE_PATH_TYPES
        n_valid = n_compared = 0
        errors: list[float] = []
        for frame in frames:
            entry = frame.get(path_type, {})
            if entry.get("valid_center"):
                n_valid += 1
                hdmap = frame.get("hdmap", {})
                if comparable and hdmap.get("valid_center"):
                    err = _lane_agreement_px(entry.get("center", []), hdmap.get("center", []))
                    if err is not None:
                        n_compared += 1
                        errors.append(err)
        result[path_type] = {
            "valid_rate":          round(n_valid / max(len(frames), 1), 3),
            "comparable_to_hdmap": comparable,
            "n_compared_frames":   n_compared,
            "mean_abs_error_px":   round(float(np.mean(errors)), 2) if errors else None,
        }
    return result


def evaluate_hdmap_quality(frames: list[dict]) -> dict:
    n_valid = sum(1 for f in frames if f.get("hdmap", {}).get("valid_center"))
    confs = [f["hdmap"]["confidence_center"] for f in frames if f.get("hdmap", {}).get("valid_center")]
    return {
        "valid_rate":      round(n_valid / max(len(frames), 1), 3),
        "confidence_mean": round(float(np.mean(confs)), 3) if confs else None,
    }


# ---------------------------------------------------------------------------
# State estimation quality (EKF vs Waymo GT 3D boxes)
# ---------------------------------------------------------------------------

def evaluate_state_estimation(frames: list[dict], match_dist_m: float = 3.0) -> dict:
    # Pass 1: collect every GT vehicle-box observation per persistent id.
    gt_history: dict[str, list[tuple[int, float, float, float]]] = {}
    for f_idx, frame in enumerate(frames):
        ts = float(frame.get("timestamp", f_idx))
        for b in frame.get("boxes_3d", []):
            if b.get("type") != _GT_VEHICLE_TYPE:
                continue
            gt_history.setdefault(b["id"], []).append((f_idx, ts, b["center_x"], b["center_y"]))

    # Pass 2: match EKF tracks to the nearest GT box per frame; accumulate errors.
    pos_errors: list[float] = []
    vel_errors: list[float] = []
    last_matched_track_for_gt: dict[str, int] = {}
    id_switch_count = 0
    n_matched = 0

    for f_idx, frame in enumerate(frames):
        boxes = [b for b in frame.get("boxes_3d", []) if b.get("type") == _GT_VEHICLE_TYPE]
        if not boxes:
            continue
        for trk in frame.get("vehicle_ekf_tracks", []):
            dists = [math.hypot(trk["x_veh"] - b["center_x"], trk["y_veh"] - b["center_y"]) for b in boxes]
            j = int(np.argmin(dists))
            if dists[j] > match_dist_m:
                continue

            gt_id = boxes[j]["id"]
            n_matched += 1
            pos_errors.append(dists[j])

            prev_track_id = last_matched_track_for_gt.get(gt_id)
            if prev_track_id is not None and prev_track_id != trk["track_id"]:
                id_switch_count += 1
            last_matched_track_for_gt[gt_id] = trk["track_id"]

            # Velocity: finite-difference this GT id's two most recent
            # observations up to and including this frame.
            hist = [h for h in gt_history.get(gt_id, []) if h[0] <= f_idx]
            if len(hist) >= 2:
                (_, t0, x0, y0), (_, t1, x1, y1) = hist[-2], hist[-1]
                dt = t1 - t0
                if dt > 1e-6:
                    gt_vx, gt_vy = (x1 - x0) / dt, (y1 - y0) / dt
                    vel_errors.append(math.hypot(
                        trk.get("vx_veh", 0.0) - gt_vx,
                        trk.get("vy_veh", 0.0) - gt_vy,
                    ))

    def _rmse(vals: list[float]) -> float | None:
        return round(float(np.sqrt(np.mean(np.square(vals)))), 3) if vals else None

    return {
        "n_matched":       n_matched,
        "pos_rmse_m":      _rmse(pos_errors),
        "vel_rmse_mps":    _rmse(vel_errors),
        "id_switch_count": id_switch_count,
    }


# ---------------------------------------------------------------------------
# Lane-relation descriptive stats (from the already-computed per-frame field)
# ---------------------------------------------------------------------------

def evaluate_lane_relations(frames: list[dict]) -> dict:
    result = {}
    for path_type in _PATH_TYPES + ["hdmap"]:
        n_valid = n_inside = n_total = 0
        for frame in frames:
            for entry in frame.get("lane_relations", []):
                rel = entry.get("relations", {}).get(path_type)
                if rel is None:
                    continue
                n_total += 1
                if rel.get("valid"):
                    n_valid += 1
                    if rel.get("inside_bounds"):
                        n_inside += 1
        result[path_type] = {
            "valid_rate":     round(n_valid / max(n_total, 1), 3),
            "inside_rate":    round(n_inside / n_valid, 3) if n_valid else None,
            "n_observations": n_total,
        }
    return result


# ---------------------------------------------------------------------------
# Per-segment / multi-segment entry points
# ---------------------------------------------------------------------------

def evaluate_segment(json_path: str) -> dict:
    """Compute every perception-quality metric for one segment's JSON output."""
    with open(json_path) as f:
        frames = json.load(f)
    return {
        "segment_name":     Path(json_path).stem,
        "n_frames":         len(frames),
        "lane_quality":     evaluate_lane_quality(frames),
        "hdmap_quality":    evaluate_hdmap_quality(frames),
        "state_estimation": evaluate_state_estimation(frames),
        "lane_relations":   evaluate_lane_relations(frames),
    }


def _weighted_mean(values_and_weights: list[tuple]) -> float | None:
    vals = [(v, w) for v, w in values_and_weights if v is not None and w]
    if not vals:
        return None
    return round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 3)


def evaluate_segments(json_paths: list[str]) -> dict:
    """Evaluate multiple segments and aggregate a frame-count-weighted summary."""
    per_segment = [evaluate_segment(p) for p in json_paths]
    total_frames = sum(s["n_frames"] for s in per_segment)

    summary: dict = {"n_segments": len(per_segment), "n_frames": total_frames}

    summary["lane_quality"] = {
        path_type: {
            "valid_rate": _weighted_mean(
                [(s["lane_quality"][path_type]["valid_rate"], s["n_frames"]) for s in per_segment]
            ),
            "mean_abs_error_px": _weighted_mean(
                [(s["lane_quality"][path_type]["mean_abs_error_px"],
                  s["lane_quality"][path_type]["n_compared_frames"]) for s in per_segment]
            ),
        }
        for path_type in _PATH_TYPES
    }

    summary["state_estimation"] = {
        "n_matched": sum(s["state_estimation"]["n_matched"] for s in per_segment),
        "pos_rmse_m": _weighted_mean(
            [(s["state_estimation"]["pos_rmse_m"], s["state_estimation"]["n_matched"]) for s in per_segment]
        ),
        "vel_rmse_mps": _weighted_mean(
            [(s["state_estimation"]["vel_rmse_mps"], s["state_estimation"]["n_matched"]) for s in per_segment]
        ),
        "id_switch_count": sum(s["state_estimation"]["id_switch_count"] for s in per_segment),
    }

    return {"summary": summary, "per_segment": per_segment}
