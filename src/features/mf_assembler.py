"""
src/features/mf_assembler.py
==============================
Rolling-window Measurement Feature (MF) vector assembler.

Public API
----------
    MFAssembler(cfg)
        Instantiate once.  Call update() per frame.  Call reset() between
        TFRecord segments.

    MFAssembler.update(tracks, lane_results, img_shape) -> dict[int, np.ndarray]
        Update rolling buffers for all active tracks and return completed
        windows of shape [T, D] for tracks that have T frames of history.

    MFAssembler.reset()
        Clear all buffers.

Feature vector layout (D = 18 per frame)
-----------------------------------------
Index  Name                        Computation
----- --------------------------- -------------------------------------------
  0   range_norm                  x_veh / range_clip_m,  clipped to [-1, 1]
  1   lateral_norm                y_veh / lateral_clip_m, clipped
  2   range_rate_norm             vx_veh / speed_clip_mps, clipped
  3   lateral_rate_norm           vy_veh / speed_clip_mps, clipped
  4   heading_delta_norm          atan2(vy, |vx|) / π   ∈ (-1, 1)
  5   dist_host_left_norm         (bbox_cx − left_lane_x@bottom) / lane_w_px
  6   dist_host_right_norm        (right_lane_x@bottom − bbox_cx) / lane_w_px
  7   dist_dp_center_norm         (bbox_cx − dp_center_x@bottom) / img_w
  8   in_drivable                 1.0 if bbox_cx inside drivable bounds else 0.0
  9   ttc_norm                    clip(range / closing_rate, 0, ttc_clip) / ttc_clip
 10   euclidean_dist_norm         sqrt(x²+y²) / range_clip_m, clipped to [0, 1]
 11   bbox_width_norm             (x2−x1) / img_w
 12   bbox_height_norm            (y2−y1) / img_h
 13   bbox_aspect_norm            (w/h) / 3.0, clipped to [0, 1]
 14   is_vehicle                  class_id == 0
 15   is_pedestrian               class_id == 1
 16   is_cyclist                  class_id == 2
 17   is_other                    class_id == 3
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Polyline interpolation helper
# ---------------------------------------------------------------------------

def _polyline_x_at_y(polyline: list, y_target: float) -> float | None:
    """
    Interpolate the x coordinate of a pixel-space polyline at a given y.

    Parameters
    ----------
    polyline : list
        List of [x, y] pixel coordinate pairs.  Must have at least 2 points.
    y_target : float
        The y pixel coordinate at which to evaluate.

    Returns
    -------
    float | None
        Interpolated x, or None when y_target is outside the polyline's range.
    """
    if not polyline or len(polyline) < 2:
        return None
    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None
    order  = np.argsort(pts[:, 1])
    pts    = pts[order]
    y_min, y_max = float(pts[0, 1]), float(pts[-1, 1])
    if y_target < y_min or y_target > y_max:
        return None
    return float(np.interp(y_target, pts[:, 1], pts[:, 0]))


# ---------------------------------------------------------------------------
# Feature vector computation (single track, single frame)
# ---------------------------------------------------------------------------

def _compute_feature_vector(
    track: dict,
    lane_results: dict,
    img_w: int,
    img_h: int,
    cfg: DictConfig,
) -> np.ndarray:
    """
    Compute the D=18 MF vector for one track in one frame.

    Parameters
    ----------
    track : dict
        TrackState.to_dict() — must contain bbox_xyxy, x_veh, y_veh,
        vx_veh, vy_veh, class_id.
    lane_results : dict
        Dict with "host_lane" and "drivable_path" sub-dicts from the
        pipeline JSON output.
    img_w, img_h : int
        Original frame dimensions in pixels.
    cfg : DictConfig
        Hydra features config node (conf/features/mf.yaml).
    """
    feat = np.zeros(18, dtype=np.float32)

    x_veh  = float(track["x_veh"])
    y_veh  = float(track["y_veh"])
    vx_veh = float(track["vx_veh"])
    vy_veh = float(track["vy_veh"])
    bbox   = track["bbox_xyxy"]

    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
        float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    )
    bbox_cx     = (bbox_x1 + bbox_x2) / 2.0
    bbox_bottom = bbox_y2
    bbox_w      = max(bbox_x2 - bbox_x1, 1.0)
    bbox_h      = max(bbox_y2 - bbox_y1, 1.0)

    # -- Features 0-4: Kalman state ------------------------------------------
    feat[0] = float(np.clip(x_veh  / cfg.range_clip_m,   -1.0, 1.0))
    feat[1] = float(np.clip(y_veh  / cfg.lateral_clip_m, -1.0, 1.0))
    feat[2] = float(np.clip(vx_veh / cfg.speed_clip_mps, -1.0, 1.0))
    feat[3] = float(np.clip(vy_veh / cfg.speed_clip_mps, -1.0, 1.0))
    feat[4] = math.atan2(vy_veh, max(abs(vx_veh), 1e-3)) / math.pi

    # -- Features 5-8: Pixel lane distances -----------------------------------
    host     = lane_results.get("host_lane", {})
    dp       = lane_results.get("drivable_path", {})

    ll_x  = _polyline_x_at_y(host.get("left",   []), bbox_bottom)
    rl_x  = _polyline_x_at_y(host.get("right",  []), bbox_bottom)
    dp_cx = _polyline_x_at_y(dp.get("center",   []), bbox_bottom)
    dp_lx = _polyline_x_at_y(dp.get("left",     []), bbox_bottom)
    dp_rx = _polyline_x_at_y(dp.get("right",    []), bbox_bottom)

    if ll_x is not None and rl_x is not None:
        lane_w_px = max(abs(rl_x - ll_x), 1.0)
    else:
        lane_w_px = img_w / 4.0  # fallback: assume lane ≈ quarter of image width

    feat[5] = float(np.clip((bbox_cx - ll_x) / lane_w_px, -2.0, 2.0)) \
              if ll_x is not None else 0.0
    feat[6] = float(np.clip((rl_x - bbox_cx) / lane_w_px, -2.0, 2.0)) \
              if rl_x is not None else 0.0
    feat[7] = float(np.clip((bbox_cx - dp_cx) / img_w, -1.0, 1.0)) \
              if dp_cx is not None else 0.0

    if dp_lx is not None and dp_rx is not None:
        feat[8] = 1.0 if min(dp_lx, dp_rx) <= bbox_cx <= max(dp_lx, dp_rx) else 0.0
    else:
        feat[8] = 0.0

    # -- Features 9-10: Derived kinematic -------------------------------------
    closing_rate = -min(vx_veh, 0.0)    # positive when target is approaching
    if x_veh > 0.0 and closing_rate > 1e-3:
        ttc         = x_veh / closing_rate
        feat[9]     = float(np.clip(ttc / cfg.ttc_clip_s, 0.0, 1.0))
    else:
        feat[9]     = 0.0

    feat[10] = float(np.clip(
        math.sqrt(x_veh ** 2 + y_veh ** 2) / cfg.range_clip_m, 0.0, 1.0
    ))

    # -- Features 11-13: Bounding box shape -----------------------------------
    feat[11] = float(np.clip(bbox_w / img_w, 0.0, 1.0))
    feat[12] = float(np.clip(bbox_h / img_h, 0.0, 1.0))
    feat[13] = float(np.clip((bbox_w / bbox_h) / 3.0, 0.0, 1.0))

    # -- Features 14-17: Class one-hot ----------------------------------------
    cls = int(track.get("class_id", 3))
    if 0 <= cls <= 3:
        feat[14 + cls] = 1.0

    return feat


# ---------------------------------------------------------------------------
# MFAssembler
# ---------------------------------------------------------------------------

class MFAssembler:
    """
    Maintains a rolling T-frame feature buffer per track ID and returns
    completed [T, D] windows.

    Parameters
    ----------
    cfg : DictConfig
        Hydra features config node (conf/features/mf.yaml).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._T   = int(cfg.window_size)
        self._D   = int(cfg.feature_dim)
        self._cfg = cfg
        self._buffers: dict[int, deque] = {}

    def update(
        self,
        tracks: list[dict],
        lane_results: dict,
        img_shape: tuple[int, int],
    ) -> dict[int, np.ndarray]:
        """
        Process one frame: compute feature vectors, update rolling buffers,
        and return completed windows.

        Parameters
        ----------
        tracks : list[dict]
            List of TrackState.to_dict() for all active confirmed tracks
            in this frame.
        lane_results : dict
            Dict with at least "host_lane" and "drivable_path" keys
            (as stored in the pipeline JSON output).
        img_shape : tuple[int, int]
            (height, width) of the original frame image.

        Returns
        -------
        dict[int, np.ndarray]
            {track_id: np.ndarray[T, D]} for every track whose buffer has
            exactly T frames.  Tracks with less history are excluded.
        """
        img_h, img_w = img_shape
        active_ids:  set[int] = set()
        completed:   dict[int, np.ndarray] = {}

        for trk in tracks:
            tid = int(trk["track_id"])
            active_ids.add(tid)

            feat = _compute_feature_vector(trk, lane_results, img_w, img_h, self._cfg)

            if tid not in self._buffers:
                self._buffers[tid] = deque(maxlen=self._T)
            self._buffers[tid].append(feat)

            if len(self._buffers[tid]) == self._T:
                completed[tid] = np.array(list(self._buffers[tid]),
                                          dtype=np.float32)

        # Remove buffers for tracks no longer active (prevents stale memory growth)
        for tid in list(self._buffers.keys()):
            if tid not in active_ids:
                del self._buffers[tid]

        return completed

    def reset(self) -> None:
        """Clear all buffers.  Call between TFRecord segments."""
        self._buffers.clear()
