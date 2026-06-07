"""
src/models/tracking/vehicle_track_manager.py
==============================================
VehicleTrackManager — EKF-based multi-vehicle tracker.

Design — Strategy Pattern (filter selection)
--------------------------------------------
Only class_id == CLASS_VEHICLE (0) detections are processed.  All other
classes are silently dropped so non-vehicle detections never pollute the
richer 9D state estimates.

Each track is backed by a ``VehicleEKF`` instance rather than the 4-state
``KalmanTracker`` used by the general ``TrackManager``.  The lifecycle
(predict → assign → update → birth/kill → confirm) mirrors the existing
``TrackManager`` exactly, preserving the two-stage IoU + centre-distance
assignment that handles brief occlusions without ID jumps.

Why a separate manager instead of modifying TrackManager
---------------------------------------------------------
Modifying the working TrackManager for a vehicle-only EKF would couple
unrelated concerns (all-class 2D CV tracking vs vehicle 3D EKF) and violate
the Open/Closed Principle.  VehicleTrackManager is a self-contained, additive
component; the existing TrackManager is unchanged.

Hydra config node: cfg.perception.vehicle_ekf

Public API
----------
    VehicleTrackState
        9D EKF state for one confirmed vehicle, with TTC and metadata.

    VehicleTrackManager(cfg)
        Instantiate once.  Call ``update()`` each frame.
        Call ``reset()`` between TFRecord segments.

    VehicleTrackManager.set_camera_params(K, R_vc, t_vc)
        Provide the runtime camera calibration extracted from the Waymo
        calibration proto.  Called once per segment at step == 0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from src.detectors.vehicle.detector import Detection, CLASS_VEHICLE
from .vehicle_ekf import (
    VehicleEKF,
    IDX_X, IDX_Y, IDX_Z, IDX_VX, IDX_VY, IDX_HDG, IDX_W, IDX_H, IDX_L,
)
from .vehicle_measurement import (
    SFMeasurementBundle,
    KalmanMeasurement,
    compute_sf_measurements,
    build_kalman_input,
    PRIOR_HEIGHT_M,
    PRIOR_WIDTH_M,
)

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
        "scipy not found — using greedy IoU matching for VehicleTrackManager.  "
        "Install scipy for optimal assignment: pip install scipy"
    )


# ---------------------------------------------------------------------------
# VehicleTrackState — public output dataclass
# ---------------------------------------------------------------------------

@dataclass
class VehicleTrackState:
    """
    Full state of one confirmed vehicle track after one frame update.

    All vehicle-frame coordinates: X = forward, Y = left, Z = up.

    This dataclass carries both the MF (multi-frame) EKF result and the
    single-frame (SF) measurement diagnostics so callers can inspect every
    stage of the pipeline for any active track.

    Attributes — MF / EKF result
    ----------------------------
    track_id : int
        Unique ID, monotonically increasing within a segment.
    bbox_xyxy : np.ndarray, shape (4,)
        Last associated detection bbox [x1, y1, x2, y2] in image pixels.
    x_veh : float
        Forward range (m) — EKF filtered.
    y_veh : float
        Lateral offset (m), positive = left — EKF filtered.
    z_veh : float
        Estimated centre height above ground (m).
    vx_veh : float
        Range rate (m/s), positive = moving away from ego.
    vy_veh : float
        Lateral rate (m/s), positive = moving left.
    speed_mps : float
        Scalar ground speed |v| (m/s).
    heading_rad : float
        Yaw angle relative to ego X-axis (rad).
    width_m : float
        Estimated lateral width (m).
    height_m : float
        Estimated vertical height (m).
    length_m : float
        Estimated longitudinal length (m).
    age : int
        Total frames since track birth.
    hits : int
        Matched-detection count.
    confidence : float
        Detector confidence of the last matched detection [0, 1].
    ttc_s : float
        Time-to-collision (s), inf when not closing.
    is_coasting : bool
        True when no detection was matched this frame.
    consecutive_misses : int
        Frames since last successful match.
    class_id : int
        Always 0 (vehicle).
    class_name : str
        Always "vehicle".

    Attributes — SF diagnostics (this frame only)
    -----------------------------------------------
    sf_bundle : SFMeasurementBundle or None
        All single-frame geometric measurements computed this frame:
        y0_proj, height_proj, width_proj, h_aspect, w_aspect.
        None when the track coasted (no detection matched).
    kalman_meas : KalmanMeasurement or None
        The 4D measurement vector that was fed into the EKF update.
        None when the track coasted or y0_proj was invalid.
    """
    track_id:           int
    bbox_xyxy:          np.ndarray
    x_veh:              float
    y_veh:              float
    z_veh:              float
    vx_veh:             float
    vy_veh:             float
    speed_mps:          float
    heading_rad:        float
    width_m:            float
    height_m:           float
    length_m:           float
    age:                int
    hits:               int
    confidence:         float               = 1.0
    ttc_s:              float               = float('inf')
    is_coasting:        bool                = False
    consecutive_misses: int                 = 0
    class_id:           int                 = CLASS_VEHICLE
    class_name:         str                 = "vehicle"
    sf_bundle:          SFMeasurementBundle | None = None
    kalman_meas:        KalmanMeasurement   | None = None

    def to_dict(self) -> dict:
        return {
            "track_id":           self.track_id,
            "class_id":           self.class_id,
            "class_name":         self.class_name,
            "bbox_xyxy":          self.bbox_xyxy.tolist(),
            # ── MF / EKF state ────────────────────────────────────────────────
            "x_veh":              round(self.x_veh,       3),
            "y_veh":              round(self.y_veh,       3),
            "z_veh":              round(self.z_veh,       3),
            "vx_veh":             round(self.vx_veh,      3),
            "vy_veh":             round(self.vy_veh,      3),
            "speed_mps":          round(self.speed_mps,   3),
            "heading_rad":        round(self.heading_rad, 4),
            "width_m":            round(self.width_m,     3),
            "height_m":           round(self.height_m,    3),
            "length_m":           round(self.length_m,    3),
            "age":                self.age,
            "hits":               self.hits,
            "confidence":         round(self.confidence,  3),
            "ttc_s":              round(self.ttc_s, 2) if self.ttc_s != float('inf') else None,
            "is_coasting":        self.is_coasting,
            "consecutive_misses": self.consecutive_misses,
            # ── SF diagnostics ────────────────────────────────────────────────
            "sf_measurements":    self.sf_bundle.to_dict()  if self.sf_bundle   is not None else None,
            "kalman_input":       self.kalman_meas.to_dict() if self.kalman_meas is not None else None,
        }


# ---------------------------------------------------------------------------
# Internal track record (not exposed outside this module)
# ---------------------------------------------------------------------------

@dataclass
class _VehicleTrack:
    track_id:        int
    ekf:             VehicleEKF
    bbox_xyxy:       np.ndarray   # last known image bbox for IoU matching
    age:             int                        = 1
    hits:            int                        = 1
    misses:          int                        = 0
    last_confidence: float                      = 1.0
    # Per-frame SF + Kalman diagnostics (populated by update(), None on coast)
    last_sf_bundle:  SFMeasurementBundle | None = None
    last_kalman_meas: KalmanMeasurement  | None = None


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]);  iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]);  iy2 = min(a[3], b[3])
    iw  = max(0.0, ix2 - ix1)
    ih  = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# VehicleTrackManager
# ---------------------------------------------------------------------------

class VehicleTrackManager:
    """
    Multi-vehicle EKF tracker (vehicle class only).

    Accepts all detections from TargetDetector; vehicles are filtered
    internally so callers do not need to pre-filter.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config node at ``cfg.perception.vehicle_ekf``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._min_hits          = int(cfg.min_hits)
        self._max_age_tentative = int(cfg.max_age_tentative)
        self._max_age_confirmed = int(cfg.max_age_confirmed)
        self._iou_threshold     = float(cfg.iou_threshold)
        self._reacquire_dist_px = float(cfg.reacquire_dist_px)
        self._default_dt        = float(cfg.default_dt)

        # EKF noise parameters
        self._q_pos     = float(cfg.process_noise_pos)
        self._q_vel     = float(cfg.process_noise_vel)
        self._q_heading = float(cfg.process_noise_heading)
        self._q_size    = float(cfg.process_noise_size)
        self._q_length  = float(cfg.process_noise_length)
        self._r_pos     = float(cfg.measurement_noise_pos)
        self._r_aspect  = float(cfg.measurement_noise_aspect)

        # Camera calibration — set via set_camera_params() at segment start.
        # fx/fy are kept as separate members for direct use in compute_sf_measurements().
        self._K:   np.ndarray | None = None
        self._R_vc: np.ndarray | None = None
        self._t_vc: np.ndarray | None = None
        self._fx = float(getattr(cfg, "fx", 2000.0))
        self._fy = float(getattr(cfg, "fy", 2000.0))

        # SF measurement config knobs
        self._height_prior_m = PRIOR_HEIGHT_M
        self._width_prior_m  = PRIOR_WIDTH_M

        self._tracks:  list[_VehicleTrack] = []
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tracks.  Call between TFRecord segments."""
        self._tracks  = []
        self._next_id = 0

    def set_camera_params(
        self,
        K:    np.ndarray,
        R_vc: np.ndarray,
        t_vc: np.ndarray,
    ) -> None:
        """
        Provide the runtime camera calibration extracted from the Waymo proto.

        Must be called once per segment at step == 0, before the first
        ``update()`` call.  All three parameters are required for y0_proj
        ground-plane projection; fx/fy are derived from K internally.

        Parameters
        ----------
        K : np.ndarray, shape (3, 3)
            Camera intrinsic matrix.
        R_vc : np.ndarray, shape (3, 3)
            Rotation: Vehicle Frame → Camera Frame.
        t_vc : np.ndarray, shape (3,)
            Translation: P_cam = R_vc @ P_veh + t_vc.
        """
        self._K   = np.asarray(K,    dtype=np.float64)
        self._R_vc = np.asarray(R_vc, dtype=np.float64)
        self._t_vc = np.asarray(t_vc, dtype=np.float64)
        self._fx  = float(self._K[0, 0])
        self._fy  = float(self._K[1, 1])
        log.debug(
            "VehicleTrackManager: camera params updated  fx=%.1f  fy=%.1f",
            self._fx, self._fy,
        )

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[Detection],
        dt:         float | None = None,
    ) -> list[VehicleTrackState]:
        """
        Run one full predict-assign-update cycle for all vehicle tracks.

        The per-detection measurement pipeline follows four explicit steps:

            Step 1 — compute_sf_measurements(): raw SF geometry from bbox.
            Step 2 — build_kalman_input():       fuse SF into EKF input vector.
            Step 3 — ekf.update():               Kalman predict + update.
            Step 4 — store diagnostics:          SF bundle + EKF state on track.

        Parameters
        ----------
        detections : list[Detection]
            All detections from TargetDetector; non-vehicles are filtered here.
        dt : float | None
            Inter-frame time (s).  Falls back to ``default_dt`` from config.

        Returns
        -------
        list[VehicleTrackState]
            All confirmed vehicle tracks (hits >= min_hits) this frame,
            each carrying both the EKF state and the SF measurement bundle.
        """
        if self._K is None:
            log.error(
                "VehicleTrackManager.update() called before set_camera_params(). "
                "Call set_camera_params(K, R_vc, t_vc) at segment start."
            )
            return []

        eff_dt = dt if dt is not None else self._default_dt

        # ── Filter to vehicle class only ──────────────────────────────────────
        veh_dets: list[Detection] = [
            d for d in detections if d.class_id == CLASS_VEHICLE
        ]

        # ── Kalman predict: advance all tracks by dt ──────────────────────────
        for trk in self._tracks:
            trk.ekf.update_dt(eff_dt)
            trk.ekf.predict()
            trk.age    += 1
            trk.misses += 1          # cleared to 0 on successful match
            trk.last_sf_bundle   = None   # clear stale diagnostics
            trk.last_kalman_meas = None

        # ── Two-stage IoU + centre-distance assignment ────────────────────────
        matched, _unmatched_trks, unmatched_dets = self._assign(veh_dets)

        # ── Update matched tracks (Steps 1 → 4) ──────────────────────────────
        for ti, di in matched:
            trk = self._tracks[ti]
            det = veh_dets[di]

            trk.misses          = 0
            trk.hits           += 1
            trk.bbox_xyxy       = det.bbox_xyxy.copy()
            trk.last_confidence = det.confidence

            # Step 1: compute all single-frame SF measurements for this detection
            sf_bundle = compute_sf_measurements(
                bbox_xyxy      = det.bbox_xyxy,
                fx             = self._fx,
                fy             = self._fy,
                K              = self._K,
                R_vc           = self._R_vc,
                t_vc           = self._t_vc,
                height_prior_m = self._height_prior_m,
                width_prior_m  = self._width_prior_m,
            )

            # Step 2: build clean Kalman input from SF bundle
            kalman_meas = build_kalman_input(sf_bundle)

            # Step 3: Kalman update (skipped when y0_proj is invalid)
            if kalman_meas is not None:
                trk.ekf.update(kalman_meas)

            # Step 4: store frame diagnostics so VehicleTrackState can expose them
            trk.last_sf_bundle   = sf_bundle
            trk.last_kalman_meas = kalman_meas

        # ── Birth new tracks from unmatched detections ────────────────────────
        for di in unmatched_dets:
            det = veh_dets[di]

            # Step 1: SF measurements for the birth detection
            sf_bundle = compute_sf_measurements(
                bbox_xyxy      = det.bbox_xyxy,
                fx             = self._fx,
                fy             = self._fy,
                K              = self._K,
                R_vc           = self._R_vc,
                t_vc           = self._t_vc,
                height_prior_m = self._height_prior_m,
                width_prior_m  = self._width_prior_m,
            )

            if not sf_bundle.y0_proj.valid:
                continue   # cannot birth a track without a valid ground position

            x0 = sf_bundle.y0_proj.x_m
            y0 = sf_bundle.y0_proj.y_m

            # Initial 3D size: back-compute from aspect-ratio × ground range
            #   h_aspect = H_real / x  →  H_real = h_aspect * x
            init_h = (sf_bundle.h_aspect * x0) if (not math.isnan(sf_bundle.h_aspect) and x0 > 1.0) else None
            init_w = (sf_bundle.w_aspect * x0) if (not math.isnan(sf_bundle.w_aspect) and x0 > 1.0) else None

            ekf = VehicleEKF(
                initial_xy = [x0, y0],
                dt         = eff_dt,
                q_pos      = self._q_pos,
                q_vel      = self._q_vel,
                q_heading  = self._q_heading,
                q_size     = self._q_size,
                q_length   = self._q_length,
                r_pos      = self._r_pos,
                r_aspect   = self._r_aspect,
                initial_w  = init_w,
                initial_h  = init_h,
            )
            self._tracks.append(_VehicleTrack(
                track_id         = self._next_id,
                ekf              = ekf,
                bbox_xyxy        = det.bbox_xyxy.copy(),
                last_confidence  = det.confidence,
                last_sf_bundle   = sf_bundle,
                last_kalman_meas = None,   # no EKF update at birth frame
            ))
            self._next_id += 1

        # ── Kill tracks that exceeded their miss budget ───────────────────────
        self._tracks = [t for t in self._tracks if t.misses <= self._miss_budget(t)]

        # ── Assemble output: confirmed tracks with full diagnostics ───────────
        output: list[VehicleTrackState] = []
        for trk in self._tracks:
            if trk.hits < self._min_hits:
                continue

            st    = trk.ekf.state
            x_veh = float(st[IDX_X])
            vx    = float(st[IDX_VX])
            ttc   = float(-x_veh / vx) if (vx < -0.1 and x_veh > 0.0) else float('inf')

            output.append(VehicleTrackState(
                track_id           = trk.track_id,
                bbox_xyxy          = trk.bbox_xyxy.copy(),
                x_veh              = x_veh,
                y_veh              = float(st[IDX_Y]),
                z_veh              = float(st[IDX_Z]),
                vx_veh             = vx,
                vy_veh             = float(st[IDX_VY]),
                speed_mps          = trk.ekf.speed_mps,
                heading_rad        = trk.ekf.heading_from_velocity,
                width_m            = float(st[IDX_W]),
                height_m           = float(st[IDX_H]),
                length_m           = float(st[IDX_L]),
                age                = trk.age,
                hits               = trk.hits,
                confidence         = trk.last_confidence,
                ttc_s              = ttc,
                is_coasting        = trk.misses > 0,
                consecutive_misses = trk.misses,
                sf_bundle          = trk.last_sf_bundle,
                kalman_meas        = trk.last_kalman_meas,
            ))
        return output

    # ------------------------------------------------------------------
    # Private: assignment
    # ------------------------------------------------------------------

    def _assign(
        self,
        dets: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """
        Two-stage assignment (IoU + centre-distance fallback).

        Stage 1: IoU-based Hungarian matching.  Inflates stale bboxes for
                 coasting tracks proportionally to the miss count.
        Stage 2: Centre-distance re-acquisition for confirmed coasting tracks
                 not matched in Stage 1.  Prevents ID jumps during occlusion.
        """
        tracks = self._tracks
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))

        n_t, n_d = len(tracks), len(dets)
        cost = np.zeros((n_t, n_d), dtype=np.float64)

        for ti, trk in enumerate(tracks):
            bbox = trk.bbox_xyxy
            if trk.misses > 0:
                # Inflate stale bbox to maintain IoU matching through occlusion
                scale = 1.0 + 0.20 * trk.misses
                cx = (bbox[0] + bbox[2]) * 0.5
                cy = (bbox[1] + bbox[3]) * 0.5
                hw = (bbox[2] - bbox[0]) * scale * 0.5
                hh = (bbox[3] - bbox[1]) * scale * 0.5
                bbox = np.array([cx - hw, cy - hh, cx + hw, cy + hh])
            for di, det in enumerate(dets):
                cost[ti, di] = _iou(bbox, det.bbox_xyxy)

        if _HAS_SCIPY:
            rows, cols = _hungarian(-cost)   # minimise cost → maximise IoU
            pairs_all  = list(zip(rows.tolist(), cols.tolist()))
        else:
            pairs_all = self._greedy(cost)

        matched:   list[tuple[int, int]] = []
        matched_t: set[int] = set()
        matched_d: set[int] = set()

        for ti, di in pairs_all:
            if cost[ti, di] >= self._iou_threshold:
                matched.append((ti, di))
                matched_t.add(ti)
                matched_d.add(di)

        unmatched_t = [i for i in range(n_t) if i not in matched_t]
        unmatched_d = [i for i in range(n_d) if i not in matched_d]

        # Stage 2: centre-distance re-acquisition
        if self._reacquire_dist_px > 0 and unmatched_t and unmatched_d:
            candidates = [
                ti for ti in unmatched_t
                if tracks[ti].hits >= self._min_hits and tracks[ti].misses > 0
            ]
            remaining_d = list(unmatched_d)

            for ti in candidates:
                if not remaining_d:
                    break
                b   = tracks[ti].bbox_xyxy
                tcx = (b[0] + b[2]) * 0.5
                tcy = (b[1] + b[3]) * 0.5

                best_di, best_dist = -1, float("inf")
                for di in remaining_d:
                    db  = dets[di].bbox_xyxy
                    dcx = (db[0] + db[2]) * 0.5
                    dcy = (db[1] + db[3]) * 0.5
                    d   = float(np.hypot(dcx - tcx, dcy - tcy))
                    if d < best_dist:
                        best_dist, best_di = d, di

                if best_di >= 0 and best_dist <= self._reacquire_dist_px:
                    matched.append((ti, best_di))
                    unmatched_t.remove(ti)
                    remaining_d.remove(best_di)

            unmatched_d = remaining_d

        return matched, unmatched_t, unmatched_d

    def _miss_budget(self, t: _VehicleTrack) -> int:
        """
        Adaptive miss budget matching the maturity-tiered policy of TrackManager.
        Tentative tracks die quickly; mature tracks coast through long occlusions.
        """
        if t.hits < self._min_hits:
            return self._max_age_tentative
        if t.hits < 5:
            return 1    # just-confirmed: tolerate 1 missed frame
        if t.hits < 10:
            return 2    # young-confirmed: bridge single-frame holes
        return self._max_age_confirmed

    @staticmethod
    def _greedy(cost: np.ndarray) -> list[tuple[int, int]]:
        """Greedy fallback assignment when scipy is unavailable."""
        pairs:  list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        flat = np.dstack(np.unravel_index(np.argsort(-cost, axis=None), cost.shape))[0]
        for ti, di in flat:
            if ti not in used_t and di not in used_d:
                pairs.append((int(ti), int(di)))
                used_t.add(int(ti))
                used_d.add(int(di))
        return pairs
