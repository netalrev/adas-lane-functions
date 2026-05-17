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
2. Assign    — two-stage matching:
     Stage 1: IoU cost matrix (tracks × detections) → Hungarian assignment.
     Stage 2: unmatched *confirmed* tracks → bounding-box centre-distance
              matching against still-unmatched detections.
              Prevents ID jumps when IoU drops due to partial occlusion or
              bbox drift during coasting frames.
3. Update    — matched tracks: Kalman update + bbox refresh.
4. Unmatched detections   → birth new tentative tracks.
5. Unmatched tracks       → increment miss counter; kill when over budget:
     • tentative tracks  (hits < min_hits): killed after max_age_tentative misses.
     • confirmed tracks  (hits >= min_hits): killed after max_age_confirmed misses.
6. Confirm   — tracks with hits >= min_hits become confirmed.
7. Return    — all confirmed TrackState objects (coasting tracks included,
              flagged with is_coasting=True).

Coasting behaviour
------------------
Confirmed tracks that miss a detection are still returned in the output list
(is_coasting=True) with a Kalman-predicted vehicle-frame position.  The stale
image bbox is carried forward unchanged.  This gives the MF assembler a
continuous per-track signal during brief occlusions without creating ID gaps.
“coasting” is visible to downstream stages via TrackState.is_coasting and
TrackState.consecutive_misses.
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
        During coasting frames this is the stale last-seen bbox.
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
    confidence : float
        Confidence score of the last matched detection [0, 1].
    ttc_s : float
        Time-to-collision in seconds computed from the Kalman state.
        ``inf`` when the target is not closing (diverging or stationary).
        Used by the TTC confidence gate and the MF assembler.
    is_coasting : bool
        True when the track was not matched to any detection this frame.
        The vehicle-frame position is Kalman-predicted, not measured.
        Downstream stages (MF assembler) should weight coasting samples
        lower or mark them as interpolated.
    consecutive_misses : int
        Number of consecutive frames since the last successful match.
        0 = matched this frame.  Increases each missed frame.
    """
    track_id:           int
    class_id:           int
    class_name:         str
    bbox_xyxy:          np.ndarray
    x_veh:              float
    y_veh:              float
    vx_veh:             float
    vy_veh:             float
    age:                int
    hits:               int
    confidence:         float = 1.0
    ttc_s:              float = float('inf')
    is_coasting:        bool  = False
    consecutive_misses: int   = 0

    def to_dict(self) -> dict:
        return {
            "track_id":           self.track_id,
            "class_id":           self.class_id,
            "class_name":         self.class_name,
            "bbox_xyxy":          self.bbox_xyxy.tolist(),
            "x_veh":              self.x_veh,
            "y_veh":              self.y_veh,
            "vx_veh":             self.vx_veh,
            "vy_veh":             self.vy_veh,
            "age":                self.age,
            "hits":               self.hits,
            "confidence":         self.confidence,
            "ttc_s":              self.ttc_s if self.ttc_s != float('inf') else None,
            "is_coasting":        self.is_coasting,
            "consecutive_misses": self.consecutive_misses,
        }


# ---------------------------------------------------------------------------
# Internal track record (not exposed outside this module)
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    track_id:        int
    class_id:        int
    class_name:      str
    kalman:          KalmanTracker
    bbox_xyxy:       np.ndarray    # last known image bbox (used for IoU matching)
    age:             int   = 1
    hits:            int   = 1
    misses:          int   = 0
    last_confidence: float = 1.0   # confidence of the last matched detection


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
    reacquire_dist_px: float,
    min_hits_confirmed: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Two-stage assignment: detections to tracks.

    Stage 1 — IoU-based Hungarian (same as before).
    Stage 2 — Centre-distance fallback for confirmed coasting tracks that
               were not matched in Stage 1.  Prevents ID jumps when IoU
               drops because the stale bbox has drifted from the object.

    Returns
    -------
    matched          : list of (track_idx, detection_idx) pairs
    unmatched_tracks : list of track indices without a match
    unmatched_dets   : list of detection indices without a match
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    n_t = len(tracks)
    n_d = len(detections)

    # ── Stage 1: IoU-based Hungarian matching ──────────────────────────────
    cost = np.zeros((n_t, n_d), dtype=np.float64)
    for ti, trk in enumerate(tracks):
        for di, det in enumerate(detections):
            cost[ti, di] = _iou(trk.bbox_xyxy, det.bbox_xyxy)

    if _HAS_SCIPY:
        row_ind, col_ind = _hungarian(-cost)   # maximise IoU
        pairs_all = list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        pairs_all = _greedy_assign(cost)

    matched:   list[tuple[int, int]] = []
    matched_t: set[int] = set()
    matched_d: set[int] = set()

    for ti, di in pairs_all:
        if cost[ti, di] >= iou_threshold:
            matched.append((ti, di))
            matched_t.add(ti)
            matched_d.add(di)

    unmatched_tracks = [i for i in range(n_t) if i not in matched_t]
    unmatched_dets   = [i for i in range(n_d) if i not in matched_d]

    # ── Stage 2: centre-distance re-acquisition for coasting confirmed tracks
    if reacquire_dist_px > 0 and unmatched_tracks and unmatched_dets:
        # Only attempt re-acquisition for confirmed tracks that are coasting.
        # Tentative tracks should not be re-acquired this way — let them die
        # cleanly so ghost tracks don’t accumulate.
        candidates = [
            ti for ti in unmatched_tracks
            if tracks[ti].hits >= min_hits_confirmed and tracks[ti].misses > 0
        ]
        remaining_dets = list(unmatched_dets)  # copy — we’ll shrink it

        for ti in candidates:
            if not remaining_dets:
                break
            trk   = tracks[ti]
            tcx   = (trk.bbox_xyxy[0] + trk.bbox_xyxy[2]) * 0.5
            tcy   = (trk.bbox_xyxy[1] + trk.bbox_xyxy[3]) * 0.5

            best_di, best_dist = -1, float("inf")
            for di in remaining_dets:
                det = detections[di]
                dcx = (det.bbox_xyxy[0] + det.bbox_xyxy[2]) * 0.5
                dcy = (det.bbox_xyxy[1] + det.bbox_xyxy[3]) * 0.5
                d   = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_di = d, di

            if best_di >= 0 and best_dist <= reacquire_dist_px:
                matched.append((ti, best_di))
                matched_t.add(ti)
                remaining_dets.remove(best_di)

        unmatched_tracks = [i for i in unmatched_tracks if i not in matched_t]
        unmatched_dets   = remaining_dets

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
        # Support both legacy single max_age and the new split params.
        self._max_age_tentative  = int(getattr(cfg, "max_age_tentative",
                                               getattr(cfg, "max_age", 3)))
        self._max_age_confirmed  = int(getattr(cfg, "max_age_confirmed",
                                               getattr(cfg, "max_age", 12)))
        self._min_hits           = int(cfg.min_hits)
        self._iou_threshold      = float(cfg.iou_threshold)
        self._reacquire_dist_px  = float(getattr(cfg, "reacquire_dist_px", 0))
        self._ttc_gate_s         = float(getattr(cfg, "ttc_gate_s", 0.0))
        self._ttc_conf_threshold = float(getattr(cfg, "ttc_conf_threshold", 0.0))
        self._default_dt         = float(cfg.default_dt)
        self._q_pos              = float(cfg.process_noise_pos)
        self._q_vel              = float(cfg.process_noise_vel)
        self._r_pos              = float(cfg.measurement_noise)

        self._tracks:   list[_Track] = []
        self._next_id:  int = 0

        log.info(
            "TrackManager ready — max_age tent/conf=%d/%d  min_hits=%d  "
            "iou_thr=%.2f  reacquire_px=%.0f",
            self._max_age_tentative, self._max_age_confirmed,
            self._min_hits, self._iou_threshold, self._reacquire_dist_px,
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
            self._tracks, detections, self._iou_threshold,
            self._reacquire_dist_px, self._min_hits,
        )

        # 3. Update matched tracks
        for ti, di in matched:
            trk     = self._tracks[ti]
            det     = detections[di]
            rw_pos  = rw_positions[di]

            trk.bbox_xyxy       = det.bbox_xyxy
            trk.class_id        = det.class_id
            trk.class_name      = det.class_name
            trk.last_confidence = det.confidence
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
                track_id        = self._next_id,
                class_id        = det.class_id,
                class_name      = det.class_name,
                kalman          = KalmanTracker(
                    initial_xy = np.array(rw_pos, dtype=np.float64),
                    dt         = effective_dt,
                    q_pos      = self._q_pos,
                    q_vel      = self._q_vel,
                    r_pos      = self._r_pos,
                ),
                bbox_xyxy       = det.bbox_xyxy,
                last_confidence = det.confidence,
            )
            self._tracks.append(trk)
            self._next_id += 1

        # 5. Kill stale tracks — different budgets for tentative vs confirmed
        def _should_kill(t: _Track) -> bool:
            budget = (
                self._max_age_confirmed
                if t.hits >= self._min_hits
                else self._max_age_tentative
            )
            return t.misses > budget

        self._tracks = [t for t in self._tracks if not _should_kill(t)]

        # 6. Return confirmed track states (including coasting tracks)
        #    Apply TTC confidence gate: tracks beyond ttc_gate_s are suppressed
        #    when their confidence is below ttc_conf_threshold.
        output: list[TrackState] = []
        for trk in self._tracks:
            if trk.hits < self._min_hits:
                continue

            state = trk.kalman.state   # [x, y, vx, vy]
            x, y, vx, vy = float(state[0]), float(state[1]), float(state[2]), float(state[3])

            # TTC computation from Kalman state
            distance = (x * x + y * y) ** 0.5
            if distance > 1e-6:
                # Radial closing speed: positive = target approaching ego
                v_radial = -(vx * x + vy * y) / distance
                ttc = distance / v_radial if v_radial > 1e-6 else float('inf')
            else:
                ttc = float('inf')  # essentially at ego position

            # TTC confidence gate
            if (self._ttc_gate_s > 0.0
                    and ttc > self._ttc_gate_s
                    and trk.last_confidence < self._ttc_conf_threshold):
                continue  # beyond safe zone and not confident enough

            output.append(TrackState(
                track_id           = trk.track_id,
                class_id           = trk.class_id,
                class_name         = trk.class_name,
                bbox_xyxy          = trk.bbox_xyxy,
                x_veh              = x,
                y_veh              = y,
                vx_veh             = vx,
                vy_veh             = vy,
                age                = trk.age,
                hits               = trk.hits,
                confidence         = trk.last_confidence,
                ttc_s              = ttc,
                is_coasting        = trk.misses > 0,
                consecutive_misses = trk.misses,
            ))

        return output
