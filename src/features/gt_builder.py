"""
src/features/gt_builder.py
============================
Algorithmic ground-truth label derivation from Waymo GT 3D boxes.

Public API
----------
    GTBuilder(cfg)
        Instantiate once.  Call compute_labels() per frame.
        Call reset() between TFRecord segments.

    GTBuilder.compute_labels(boxes_3d, tracks) -> dict[int, dict]
        Returns per-track labels for the current frame.

Label definitions
-----------------
CIPV (Closest In-Path Vehicle)
    The single nearest vehicle-class GT box that satisfies:
      • center_x >= cipv_min_range_m  (ahead of ego)
      • |center_y| <= ego_lane_half_width_m  (within ego lane corridor)
    At most one object per frame is CIPV=1; all others are CIPV=0.

Lane Assignment
    Integer lane index relative to ego:
      lane_assignment = round(center_y / lane_width_m)
      clipped to [-lane_max_idx, +lane_max_idx]
      Positive = left of ego, negative = right (Y_veh convention: left=+).

Cut-In
    True when lane_assignment transitions from any non-zero value → 0
    within the last `cut_in_window_n` frames for the same GT object ID.

GT derivation uses Waymo GT 3D boxes (perfect labels, no tracker noise).
Tracker tracks are then matched to GT boxes by minimum Euclidean distance
in vehicle frame so each confirmed track inherits its GT label.

Matching strategy
-----------------
For each confirmed tracker track: find the Waymo GT 3D box whose
vehicle-frame center is closest (Euclidean distance in the X-Y plane).
If the closest box is within `gt_match_dist_m`, the track receives that
box's label.  Unmatched tracks are excluded from the training set.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from omegaconf import DictConfig


# Waymo GT object type constants (same as open_dataset.Label.Type)
_TYPE_VEHICLE    = 1
_TYPE_PEDESTRIAN = 2
_TYPE_SIGN       = 3
_TYPE_CYCLIST    = 4


class GTBuilder:
    """
    Derives CIPV, Lane Assignment, and Cut-In labels from Waymo GT 3D boxes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra features config node (conf/features/mf.yaml).
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._lane_width_m          = float(cfg.lane_width_m)
        self._ego_lane_half_width_m = float(cfg.ego_lane_half_width_m)
        self._cipv_min_range_m      = float(cfg.cipv_min_range_m)
        self._gt_match_dist_m       = float(cfg.gt_match_dist_m)
        self._cut_in_window_n       = int(cfg.cut_in_window_n)
        self._min_lidar_points      = int(cfg.min_lidar_points)
        self._lane_max_idx          = int(cfg.lane_max_idx)

        # Per-GT-object lane assignment history for cut-in detection.
        # Key = Waymo object id string.  Value = deque of past lane assignments.
        self._lane_history: dict[str, deque] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_labels(
        self,
        boxes_3d: list[dict],
        tracks: list[dict],
    ) -> dict[int, dict]:
        """
        Compute CIPV, Lane Assignment, and Cut-In labels for each tracker
        track by matching to Waymo GT 3D boxes.

        Parameters
        ----------
        boxes_3d : list[dict]
            Waymo GT 3D boxes for this frame (from extract_gt_3d_boxes).
        tracks : list[dict]
            Active confirmed tracker tracks (TrackState.to_dict()).

        Returns
        -------
        dict[int, dict]
            {track_id: {"cipv": int, "lane_assignment": int, "cut_in": int}}
            Only contains entries for tracks successfully matched to a GT box.
        """
        if not boxes_3d or not tracks:
            return {}

        # Quality filter: discard boxes with too few lidar points
        valid_boxes = [
            b for b in boxes_3d
            if int(b.get("num_lidar_points", 0)) >= self._min_lidar_points
        ]
        if not valid_boxes:
            return {}

        # Step 1: compute GT labels for every valid GT box
        cipv_gt_id  = self._find_cipv_gt_id(valid_boxes)
        box_labels: dict[str, dict] = {}

        for box in valid_boxes:
            gt_id     = box["id"]
            lane_asgn = self._lane_assignment(float(box["center_y"]))
            cut_in    = self._detect_cut_in(gt_id, lane_asgn)

            # Update history AFTER reading it (so this frame's value is not
            # included in the cut-in check for the same frame)
            if gt_id not in self._lane_history:
                self._lane_history[gt_id] = deque(maxlen=self._cut_in_window_n)
            self._lane_history[gt_id].append(lane_asgn)

            box_labels[gt_id] = {
                "cipv":            int(gt_id == cipv_gt_id),
                "lane_assignment": lane_asgn,
                "cut_in":          int(cut_in),
            }

        # Step 2: match tracker tracks to GT boxes
        return self._match_tracks_to_gt(tracks, valid_boxes, box_labels)

    def reset(self) -> None:
        """
        Clear per-object lane history.  Call between TFRecord segments to
        prevent cut-in detections from spanning segment boundaries.
        """
        self._lane_history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_cipv_gt_id(self, boxes: list[dict]) -> str | None:
        """
        Identify the CIPV: the closest vehicle-class GT box in the ego
        lane corridor.  Returns its GT object ID, or None if no candidate.
        """
        candidates = [
            b for b in boxes
            if int(b["type"]) == _TYPE_VEHICLE
            and float(b["center_x"]) >= self._cipv_min_range_m
            and abs(float(b["center_y"])) <= self._ego_lane_half_width_m
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda b: float(b["center_x"]))["id"]

    def _lane_assignment(self, center_y: float) -> int:
        """
        Map vehicle-frame lateral position to integer lane index.
        Y > 0 (left) → positive index.  Y < 0 (right) → negative index.
        """
        raw = center_y / self._lane_width_m
        return int(np.clip(round(raw), -self._lane_max_idx, self._lane_max_idx))

    def _detect_cut_in(self, gt_id: str, current_lane: int) -> bool:
        """
        Cut-In rule: the object is currently in the ego lane (lane == 0)
        AND was in an adjacent lane (lane != 0) in any of the last N frames.
        """
        if current_lane != 0:
            return False
        history = self._lane_history.get(gt_id)
        if not history:
            return False
        return any(past != 0 for past in history)

    def _match_tracks_to_gt(
        self,
        tracks: list[dict],
        boxes: list[dict],
        box_labels: dict[str, dict],
    ) -> dict[int, dict]:
        """
        For each tracker track find the nearest GT box (by Euclidean distance
        in the vehicle-frame X-Y plane).  Assign the GT label if the match
        distance is within the configured threshold.
        """
        if not tracks or not boxes:
            return {}

        gt_positions = [
            (b["id"], float(b["center_x"]), float(b["center_y"]))
            for b in boxes
        ]
        result: dict[int, dict] = {}

        for trk in tracks:
            tid = int(trk["track_id"])
            tx  = float(trk["x_veh"])
            ty  = float(trk["y_veh"])

            best_dist  = float("inf")
            best_gt_id = None
            for gt_id, gx, gy in gt_positions:
                dist = math.sqrt((tx - gx) ** 2 + (ty - gy) ** 2)
                if dist < best_dist:
                    best_dist, best_gt_id = dist, gt_id

            if best_dist <= self._gt_match_dist_m and best_gt_id in box_labels:
                result[tid] = box_labels[best_gt_id]

        return result
