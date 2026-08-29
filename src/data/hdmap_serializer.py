"""
src/data/hdmap_serializer.py
==============================
Serializes HD-map (Path 2) left/right boundary polylines from
project_hdmap_lanes() into the same PathData-shaped dict used by the other
three lane-path strategies (kinematic, drivable_path, host_lane).

The center line is derived (not detected): it is the pointwise midpoint of
the left/right boundaries, resampled onto a shared y-grid so the two
polylines can be averaged even when they have different point counts.
"""
from __future__ import annotations

import numpy as np


def _as_point_list(arr) -> list:
    if arr is None:
        return []
    pts = arr.tolist() if hasattr(arr, "tolist") else arr
    return pts if len(pts) >= 2 else []


def serialize_hdmap(hdmap_data: dict | None) -> dict:
    """Build the gt_data["hdmap"] PathData dict from project_hdmap_lanes() output."""
    if hdmap_data is not None:
        left  = _as_point_list(hdmap_data["left_lane"])
        right = _as_point_list(hdmap_data["right_lane"])
    else:
        left, right = [], []

    center: list = []
    if len(left) >= 2 and len(right) >= 2:
        left_arr  = np.array(left,  dtype=np.float64)
        right_arr = np.array(right, dtype=np.float64)
        y_lo = float(max(left_arr[:, 1].min(), right_arr[:, 1].min()))
        y_hi = float(min(left_arr[:, 1].max(), right_arr[:, 1].max()))
        if y_hi > y_lo:
            left_sorted  = left_arr[np.argsort(left_arr[:, 1])]
            right_sorted = right_arr[np.argsort(right_arr[:, 1])]
            y_c = np.linspace(y_lo, y_hi, 30)
            xl  = np.interp(y_c, left_sorted[:, 1],  left_sorted[:, 0])
            xr  = np.interp(y_c, right_sorted[:, 1], right_sorted[:, 0])
            center = np.column_stack(
                [((xl + xr) / 2).astype(np.int32), y_c.astype(np.int32)]
            ).tolist()

    return {
        "center":            center,
        "left":              left,
        "right":             right,
        "valid_center":      len(center) >= 2,
        "valid_left":        len(left) >= 2,
        "valid_right":       len(right) >= 2,
        "confidence_center": 1.0 if len(center) >= 2 else 0.0,
        "confidence_left":   1.0 if len(left) >= 2 else 0.0,
        "confidence_right":  1.0 if len(right) >= 2 else 0.0,
        "timestamps_s":      [],
        "source":            "hdmap",
        "is_gt":             True,
    }
