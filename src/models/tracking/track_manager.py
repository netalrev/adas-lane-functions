"""
src/models/tracking/track_manager.py
======================================
Multi-object track lifecycle manager using IoU-based Hungarian assignment
and per-track Constant Velocity Kalman filters.

Public API
----------
    TrackState
        Dataclass holding the full state of one confirmed track.

    TrackManager(cfg)
        Instantiate once per pipeline run.  Call update() each frame.
        Call reset() between TFRecord segments.

Update lifecycle (per frame)
-----------------------------
1. Predict   — advance all existing Kalman filters by dt.
2. Assign    — compute IoU cost matrix (tracks × detections); run
               Hungarian assignment.
3. Update    — matched tracks: Kalman update + bbox refresh.
4. Unmatched detections   → birth new tentative tracks.
5. Unmatched tracks       → increment miss counter; kill if > max_age.
6. Confirm   — tracks with hits >= min_hits become confirmed.
7. Return    — list of confirmed TrackState objects.

Assignment algorithm
--------------------
Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm) when scipy
is available; falls back to greedy IoU matching otherwise.  For typical
ADAS scenes (< 50 objects), both produce near-identical results.

IoU cost between a track and a detection is computed in image space using
the last known bounding box of the track as its predicted position.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from src.models.detection.detector import Detection
from .kalman_tracker import KalmanTracker

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional scipy import
# ---------------------------------------------------------------------------

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    log.warning(
        "scipy not found — using greedy IoU matching instead of Hungarian "
        "assignment.  Install scipy for optimal assignment: pip install scipy"
    )


# ---------------------------------------------------------------------------
# TrackState
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """
    Full state of one confirmed track after the current frame's update.

    Attributes
    ----------
    track_id : int
        Unique integer identifier, monotonically increasing.
    class_id : int
        Internal class: 0=vehicle, 1=pedestrian, 2=cyclist, 3=other.
    class_name : str
        Human-readable class label.
    bbox_xyxy : np.ndarray, shape (4,)
        Last associated detection bounding box [x1, y1, x2, y2].
    x_veh : float
        Vehicle-frame range (m), positive = forward.
    y_veh : float
        Vehicle-frame lateral offset (m), positive = left.
    vx_veh : float
        Range rate (m/s), positive = moving away from ego.
    vy_veh : float
        Lateral rate (m/s), positive = moving left.
    age : int
        Total frames since track creation.
    hits : int
        Number of frames the track has been matched to a detection.
    """
    track_id:   int
    class_id:   int
    class_name: str
    bbox_xyxy:  np.ndarray
    x_veh:      float
    y_veh:      float
    vx_veh:     float
    vy_veh:     float
    age:        int
    hits:       int

    def to_dict(self) -> dict:
        return {
            "track_id":   self.track_id,
            "class_id":   self.class_id,
            "class_name": self.class_name,
            "bbox_xyxy":  self.bbox_xyxy.tolist(),
            "x_veh":      self.x_veh,
            "y_veh":      self.y_veh,
            "vx_veh":     self.vx_veh,
            "vy_veh":     self.vy_veh,
            "age":        self.age,
            "hits":       self.hits,
        }


# ---------------------------------------------------------------------------
# Internal track record (not exposed outside this module)
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    track_id: int
    class_id: int
    class_name: str
    kalman: KalmanTracker
    bbox_xyxy: np.ndarray    # last known image bbox (used for IoU matching)
    age:    int = 1
    hits:   int = 1
    misses: int = 0


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1)
    ih  = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------

def _assign(
    tracks: list[_Track],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Assign detections to tracks by maximising IoU.

    Returns
    -------
    matched   : list of (track_idx, detection_idx) pairs
    unmatched_tracks : list of track indices without a match
    unmatched_dets   : list of detection indices without a match
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    n_t = len(tracks)
    n_d = len(detections)

    # Build IoU cost matrix (negate for minimisation)
    cost = np.zeros((n_t, n_d), dtype=np.float64)
    for ti, trk in enumerate(tracks):
        for di, det in enumerate(detections):
            cost[ti, di] = _iou(trk.bbox_xyxy, det.bbox_xyxy)

    if _HAS_SCIPY:
        row_ind, col_ind = _hungarian(-cost)   # maximise IoU
        pairs_all = list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        pairs_all = _greedy_assign(cost)

    matched, unmatched_tracks, unmatched_dets = [], [], []
    matched_t, matched_d = set(), set()

    for ti, di in pairs_all:
        if cost[ti, di] >= iou_threshold:
            matched.append((ti, di))
            matched_t.add(ti)
            matched_d.add(di)

    unmatched_tracks = [i for i in range(n_t) if i not in matched_t]
    unmatched_dets   = [i for i in range(n_d) if i not in matched_d]
    return matched, unmatched_tracks, unmatched_dets


def _greedy_assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """
    Greedy assignment: repeatedly pick the maximum IoU pair.
    O(n²) — sufficient for scenes with < 100 objects.
    """
    pairs = []
    used_r, used_c = set(), set()
    # Iterate in descending IoU order
    flat_order = np.argsort(-cost.ravel())
    for idx in flat_order:
        r, c = divmod(int(idx), cost.shape[1])
        if r not in used_r and c not in used_c:
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    return pairs


# ---------------------------------------------------------------------------
# TrackManager
# ---------------------------------------------------------------------------

class TrackManager:
    """
    Manages the full lifecycle of all tracks across a segment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config node at ``cfg.perception.tracker``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._max_age       = int(cfg.max_age)
        self._min_hits      = int(cfg.min_hits)
        self._iou_threshold = float(cfg.iou_threshold)
        self._default_dt    = float(cfg.default_dt)
        self._q_pos         = float(cfg.process_noise_pos)
        self._q_vel         = float(cfg.process_noise_vel)
        self._r_pos         = float(cfg.measurement_noise)

        self._tracks: list[_Track] = []
        self._next_id: int = 0

        log.info(
            "TrackManager ready — max_age=%d  min_hits=%d  iou_thr=%.2f",
            self._max_age, self._min_hits, self._iou_threshold,
        )

    def reset(self) -> None:
        """
        Clear all tracks.  Call between TFRecord segments to prevent
        stale tracklets from one segment leaking into the next.
        """
        self._tracks.clear()
        self._next_id = 0

    def update(
        self,
        detections: list[Detection],
        rw_positions: list[tuple[float, float] | None],
        dt: float | None = None,
    ) -> list[TrackState]:
        """
        Run one full predict-assign-update cycle.

        Parameters
        ----------
        detections : list[Detection]
            Filtered detections from TargetDetector for this frame.
        rw_positions : list[tuple[float, float] | None]
            Parallel list of (range_m, lateral_m) ground-plane projections.
            An entry is None when the projection was geometrically invalid
            (e.g. box too close to horizon).  Detections with None positions
            are used for IoU matching but do not update the Kalman state.
        dt : float | None
            Inter-frame time in seconds.  Defaults to cfg.default_dt.

        Returns
        -------
        list[TrackState]
            All confirmed tracks (hits >= min_hits) after this frame's update.
        """
        effective_dt = dt if dt is not None else self._default_dt

        # 1. Predict all existing tracks
        for trk in self._tracks:
            trk.kalman.update_dt(effective_dt)
            trk.kalman.predict()
            trk.age += 1
            trk.misses += 1   # will be reset to 0 on match

        # 2. Assign detections to tracks
        matched, unmatched_tracks, unmatched_dets = _assign(
            self._tracks, detections, self._iou_threshold
        )

        # 3. Update matched tracks
        for ti, di in matched:
            trk     = self._tracks[ti]
            det     = detections[di]
            rw_pos  = rw_positions[di]

            trk.bbox_xyxy  = det.bbox_xyxy
            trk.class_id   = det.class_id
            trk.class_name = det.class_name
            trk.hits  += 1
            trk.misses = 0

            if rw_pos is not None:
                trk.kalman.update(np.array(rw_pos, dtype=np.float64))

        # 4. Birth new tracks for unmatched detections
        for di in unmatched_dets:
            det    = detections[di]
            rw_pos = rw_positions[di]
            if rw_pos is None:
                continue   # cannot initialise tracker without a valid position
            trk = _Track(
                track_id   = self._next_id,
                class_id   = det.class_id,
                class_name = det.class_name,
                kalman     = KalmanTracker(
                    initial_xy = np.array(rw_pos, dtype=np.float64),
                    dt         = effective_dt,
                    q_pos      = self._q_pos,
                    q_vel      = self._q_vel,
                    r_pos      = self._r_pos,
                ),
                bbox_xyxy = det.bbox_xyxy,
            )
            self._tracks.append(trk)
            self._next_id += 1

        # 5. Kill stale tracks
        self._tracks = [t for t in self._tracks if t.misses <= self._max_age]

        # 6. Return confirmed track states
        output: list[TrackState] = []
        for trk in self._tracks:
            if trk.hits < self._min_hits:
                continue
            state = trk.kalman.state   # [x, y, vx, vy]
            output.append(TrackState(
                track_id   = trk.track_id,
                class_id   = trk.class_id,
                class_name = trk.class_name,
                bbox_xyxy  = trk.bbox_xyxy,
                x_veh      = float(state[0]),
                y_veh      = float(state[1]),
                vx_veh     = float(state[2]),
                vy_veh     = float(state[3]),
                age        = trk.age,
                hits       = trk.hits,
            ))

        return output
