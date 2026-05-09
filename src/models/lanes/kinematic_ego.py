"""
src/models/lanes/kinematic_ego.py
==================================
Strategy — Path 1: Kinematic Ego Path.

Encapsulates all state and computation required to produce a Constant Turn
Rate (CTR) trajectory prediction for one frame:

    • Yaw-rate derivation from consecutive Waymo pose transforms.
    • Exponential Moving Average (EMA) smoothing of the raw finite-difference
      yaw rate (suppresses single-frame GPS noise spikes).
    • Delegation to KinematicPathPredictor for the actual CTR arc geometry.

All inter-frame state (previous transform, previous timestamp, EMA
accumulator) is owned by this class.  Call ``reset_segment_state()``
between TFRecord segments to prevent stale state from one recording
bleeding into the next.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# Kinematic helpers (moved from src/models/lane_detector.py)
# ---------------------------------------------------------------------------

def compute_yaw_rate(
    prev_transform: list[float],
    curr_transform: list[float],
    dt: float,
) -> float:
    """
    Compute the ego vehicle's yaw rate in radians per second.

    Waymo ``frame.pose.transform`` is a flat, row-major 4×4 homogeneous
    transformation matrix that maps points from the Vehicle Frame to the
    Global Frame (world coordinates).

    Layout (0-indexed):
        [ R00  R01  R02  tx ]     indices  0  1  2  3
        [ R10  R11  R12  ty ]              4  5  6  7
        [ R20  R21  R22  tz ]              8  9 10 11
        [  0    0    0    1 ]             12 13 14 15

    The vehicle heading in the global X-Y plane is atan2(R10, R00), i.e.
    atan2(transform[4], transform[0]).

    Parameters
    ----------
    prev_transform : list[float]
        Flat 16-element transform from the previous frame.
    curr_transform : list[float]
        Flat 16-element transform from the current frame.
    dt : float
        Time elapsed between the two frames in seconds.  Must be > 0.

    Returns
    -------
    float
        Yaw rate in rad/s.  Positive = left turn (counter-clockwise when
        viewed from above), negative = right turn.  Returns 0.0 if dt <= 0.
    """
    if dt <= 0.0:
        return 0.0

    # Extract heading angle from the rotation sub-matrix
    # heading = atan2(R10, R00) = atan2(transform[4], transform[0])
    heading_prev = np.arctan2(prev_transform[4], prev_transform[0])
    heading_curr = np.arctan2(curr_transform[4], curr_transform[0])

    # Wrap angle difference to [-π, π] to handle heading wrap-around
    delta_heading = np.arctan2(
        np.sin(heading_curr - heading_prev),
        np.cos(heading_curr - heading_prev),
    )

    return float(delta_heading / dt)


# ---------------------------------------------------------------------------
# Temporal-smoothing constants
# ---------------------------------------------------------------------------

# EMA alpha for the raw finite-difference yaw rate.
# alpha = 0.15 means each new raw sample contributes 15 % to the smoothed
# output; the remaining 85 % comes from the running average.  This is a
# strong low-pass filter well suited for noisy pose-matrix derivatives.

_YAW_RATE_EMA_ALPHA: float = 0.15

# EMA alpha for polynomial lane/path coefficients (IPM and YOLOPv2).
# At 0.25 the smoothed polynomial tracks genuine road curvature changes
# while rejecting per-frame sliding-window or segmentation mask noise.

class KinematicPathPredictor:
    """
    Predict the drivable path using a Constant Turn Rate (CTR) motion model.

    The CTR model assumes the vehicle travels at a constant forward speed and
    a constant yaw rate.  This gives a circular arc trajectory which is a good
    approximation for short-horizon highway and urban driving.

    All predicted coordinates are expressed in the **Vehicle Frame**:
        - X axis: forward (longitudinal)
        - Y axis: left  (lateral)
        - Origin: rear axle centre at the current time step

    Parameters
    ----------
    vehicle_width : float
        Full vehicle width in metres.  Used to offset the centre-line
        trajectory to left/right wheel positions.  Default: 2.1 m.
    horizon_s : float
        Prediction horizon in seconds.  Default: 3.0 s.
    n_points : int
        Number of trajectory sample points over the horizon.  Default: 30.
    min_speed_mps : float
        Below this speed [m/s] the predictor returns a straight-line
        prediction to avoid numerical instability.  Default: 0.5 m/s.
    """

    def __init__(
        self,
        vehicle_width: float = 2.1,
        horizon_s: float = 3.0,
        n_points: int = 30,
        min_speed_mps: float = 8.0,
    ) -> None:
        self.vehicle_width  = vehicle_width
        self.horizon_s      = horizon_s
        self.n_points       = n_points
        self.min_speed_mps  = min_speed_mps

        # Half-width offsets to place left/right wheel trajectories
        self._half_width = vehicle_width / 2.0

        # EMA state for yaw-rate smoothing across frames.
        # None = first call; initialised with the raw value on first invocation.
        self._ema_yaw_rate: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        speed_mps: float,
        yaw_rate_rads: float,
    ) -> dict[str, np.ndarray]:
        """
        Predict the ego drivable path in the Vehicle Frame.

        The centre line extends to 2× the wheel-track horizon so it reaches
        toward the vanishing point while the L/R boundary corridor stays
        close (high-confidence near-field zone).

        Returns
        -------
        dict with keys:
            "centre_line"    : np.ndarray (N_centre, 2)  — long-range look-ahead
            "left_boundary"  : np.ndarray (N_bnd,    2)  — near-field wheel track
            "right_boundary" : np.ndarray (N_bnd,    2)
            "timestamps"     : np.ndarray (N_bnd,)   — seconds (boundary sample)
        """
        # ── EMA smoothing on raw yaw rate ────────────────────────────────────
        # The finite-difference yaw rate from consecutive Waymo pose matrices
        # is very noisy at typical 10 Hz frame rates.  An EMA filter blends
        # the current raw estimate toward a running average so the CTR arc
        # does not jitter violently frame-to-frame.
        # Bootstrap: on the very first call initialise from the raw value so
        # there is no lag spike at startup.
        if self._ema_yaw_rate is None:
            self._ema_yaw_rate = float(yaw_rate_rads)
        else:
            self._ema_yaw_rate = (
                _YAW_RATE_EMA_ALPHA * float(yaw_rate_rads)
                + (1.0 - _YAW_RATE_EMA_ALPHA) * self._ema_yaw_rate
            )
        yaw_rate_rads = self._ema_yaw_rate

        # ── Yaw rate cap ────────────────────────────────────────────────────
        MAX_YAW_RATE = 0.35   # rad/s ≈ 20 deg/s; filters pose-diff noise spikes
        yaw_rate_rads = float(np.clip(yaw_rate_rads, -MAX_YAW_RATE, MAX_YAW_RATE))

        # ── Speed-dependent yaw rate damping ────────────────────────────────
        # At low speeds the CTR turning radius R = v/ω collapses to zero for
        # any non-zero ω, producing spiral or wildly oscillating trajectories.
        # We linearly ramp the effective yaw rate toward zero as speed drops
        # below _LOW_SPEED_YAW_DAMP_THRESH so the prediction degrades
        # gracefully to a straight line rather than an implausible tight arc.
        # At full threshold speed the factor is 1.0 (no damping at all).
        _LOW_SPEED_YAW_DAMP_THRESH = 5.0  # m/s
        speed_abs = abs(speed_mps)
        if speed_abs < _LOW_SPEED_YAW_DAMP_THRESH:
            # damp_factor: 0.0 at standstill → 1.0 at the threshold
            damp_factor = speed_abs / _LOW_SPEED_YAW_DAMP_THRESH
            yaw_rate_rads *= damp_factor

        # ── Speed floor (visibility) ─────────────────────────────────────────
        display_speed = float(np.sign(speed_mps) or 1.0) * max(
            abs(speed_mps), self.min_speed_mps
        )

        # ── Boundary (near-field, high confidence) ──────────────────────────
        ts_bnd = np.linspace(0.0, self.horizon_s, self.n_points)
        # ── Centre line (long look-ahead, 2× horizon) ──────────────────────
        ts_ctr = np.linspace(0.0, self.horizon_s * 2.0, self.n_points)

        def _arc_or_straight(ts):
            if abs(yaw_rate_rads) < 1e-4:
                return self._straight_line(display_speed, ts)
            return self._ctr_arc(display_speed, yaw_rate_rads, ts)

        centre = _arc_or_straight(ts_ctr)
        centre_bnd = _arc_or_straight(ts_bnd)   # same arc, short range for offsets
        left, right = self._offset_to_wheels(centre_bnd, yaw_rate_rads, ts_bnd)

        return {
            "centre_line":    centre,
            "left_boundary":  left,
            "right_boundary": right,
            "timestamps":     ts_bnd,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _straight_line(speed_mps: float, timestamps: np.ndarray) -> np.ndarray:
        """Return a straight-forward centre-line trajectory."""
        x = speed_mps * timestamps        # forward displacement
        y = np.zeros_like(timestamps)     # no lateral movement
        return np.column_stack([x, y])    # shape (N, 2)

    @staticmethod
    def _ctr_arc(
        speed_mps: float,
        yaw_rate_rads: float,
        timestamps: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the CTR arc centre-line trajectory.

        For a constant turn rate ω and constant speed v, the turning radius is:
            R = v / ω

        Position at time t starting from origin heading along +X:
            X(t) =  R * sin(ω * t)
            Y(t) =  R * (1 - cos(ω * t))   [left positive]
        """
        omega = yaw_rate_rads
        radius = speed_mps / omega  # signed radius (negative = right turn)

        theta = omega * timestamps          # cumulative heading change
        x = radius * np.sin(theta)
        y = radius * (1.0 - np.cos(theta))

        return np.column_stack([x, y])     # shape (N, 2)

    def _offset_to_wheels(
        self,
        centre: np.ndarray,
        yaw_rate_rads: float,
        timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Offset the centre-line laterally to produce left/right wheel paths.

        At each point the local heading is approximated by the finite
        difference of the centre-line, then the normal direction is used
        to shift the trajectory by ±half_width.
        """
        n = len(centre)

        # Local forward direction at each point (unit vector)
        if n > 1:
            diffs = np.diff(centre, axis=0)   # (N-1, 2)
            # Pad last difference so arrays stay length N
            diffs = np.vstack([diffs, diffs[-1]])
        else:
            diffs = np.array([[1.0, 0.0]])

        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        # Avoid division by zero for zero-displacement steps
        norms = np.where(norms < 1e-9, 1.0, norms)
        forward = diffs / norms             # (N, 2)  unit vectors

        # Left normal: rotate forward by +90° → (-dy, dx)
        left_normal = np.column_stack([-forward[:, 1], forward[:, 0]])

        left  = centre + self._half_width * left_normal
        right = centre - self._half_width * left_normal

        return left, right


# ---------------------------------------------------------------------------
# IPM lane-detection constants
# ---------------------------------------------------------------------------



from .base import AbstractLaneDetector, VehicleState


class KinematicEgoStrategy:
    """
    Strategy: compute the CTR kinematic ego path for one frame.

    Owns
    ----
    KinematicPathPredictor
        CTR arc model with internal speed-dependent yaw damping and a
        second-stage EMA filter (alpha = 0.15, added for temporal smoothing).
    EMA accumulator
        A coarser pipeline-level EMA (alpha = 0.25, ~4-frame effective window)
        applied to the raw finite-difference yaw rate before passing it to the
        predictor.  Both EMA stages are intentionally preserved; the coarser
        one removes large spikes before the predictor's finer filter sees them.
    prev_transform / prev_timestamp
        Cached from the previous frame to compute the finite-difference yaw rate.
    """

    # Pipeline-level EMA alpha: fraction of the new raw sample blended in.
    # alpha = 0.25 → effective smoothing window of ~4 frames at 10 Hz.
    _EMA_ALPHA: float = 0.25

    def __init__(self, cfg: DictConfig) -> None:
        kp = cfg.perception.kinematic_path
        self._predictor = KinematicPathPredictor(
            vehicle_width = float(kp.vehicle_width),
            horizon_s     = float(kp.horizon_s),
            n_points      = int(kp.n_points),
            min_speed_mps = float(kp.min_speed_mps),
        )
        # Inter-frame state — reset between segments.
        self._prev_transform: list[float] | None = None
        self._prev_timestamp: float | None       = None
        self._smoothed_yaw_rate: float           = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_segment_state(self) -> None:
        """
        Reset all inter-frame state.

        Must be called once before the first frame of each new TFRecord
        segment so that EMA accumulators and the pose cache from the
        previous segment do not contaminate the current one.
        """
        self._prev_transform     = None
        self._prev_timestamp     = None
        self._smoothed_yaw_rate  = 0.0
        # Also reset the predictor's internal EMA so it re-bootstraps cleanly.
        self._predictor._ema_yaw_rate = None

    def compute(
        self,
        speed_mps:      float,
        curr_transform: list[float],
        curr_timestamp: float,
    ) -> dict:
        """
        Compute the kinematic ego path for one frame.

        Parameters
        ----------
        speed_mps : float
            Ego longitudinal speed in m/s (positive = forward).
        curr_transform : list[float]
            Flat 16-element row-major 4×4 Waymo pose transform for this frame
            (Vehicle Frame → Global Frame).
        curr_timestamp : float
            Timestamp of this frame in seconds.

        Returns
        -------
        dict
            Two keys:

            "raw" : dict
                Direct output of ``KinematicPathPredictor.predict()``
                (numpy arrays) — passed to the visualizer and used for
                HD-map projection (ego_center_veh).
            "serialized" : dict
                JSON-serializable path entry for the GT output file.
        """
        # --- Derive raw yaw rate from finite-difference of pose matrices ---
        yaw_rate_raw = 0.0
        if self._prev_transform is not None and self._prev_timestamp is not None:
            dt = curr_timestamp - self._prev_timestamp
            if dt > 0.0:
                yaw_rate_raw = compute_yaw_rate(
                    self._prev_transform, curr_transform, dt,
                )

        # --- Pipeline-level EMA smoothing ---
        # The raw finite-difference yaw rate from GPS/IMU pose matrices can
        # contain single-frame spikes of several rad/s.  This EMA retains 75 %
        # of the previous estimate so outliers are attenuated before reaching
        # the predictor's own (finer) EMA filter.
        self._smoothed_yaw_rate = (
            self._EMA_ALPHA * yaw_rate_raw
            + (1.0 - self._EMA_ALPHA) * self._smoothed_yaw_rate
        )

        # --- CTR prediction ---
        path_data = self._predictor.predict(
            speed_mps     = speed_mps,
            yaw_rate_rads = self._smoothed_yaw_rate,
        )

        # --- Advance per-frame state ---
        self._prev_transform = curr_transform
        self._prev_timestamp = curr_timestamp

        # --- Build JSON-serializable dict ---
        def _pts(arr) -> list:
            if arr is None:
                return []
            a = arr.tolist() if hasattr(arr, "tolist") else list(arr)
            return a if len(a) >= 2 else []

        serialized = {
            "center":            _pts(path_data["centre_line"]),
            "left":              _pts(path_data["left_boundary"]),
            "right":             _pts(path_data["right_boundary"]),
            "valid_center":      True,
            "valid_left":        len(_pts(path_data["left_boundary"])) >= 2,
            "valid_right":       len(_pts(path_data["right_boundary"])) >= 2,
            "confidence_center": 1.0,
            "confidence_left":   1.0,
            "confidence_right":  1.0,
            "timestamps_s":      path_data["timestamps"].tolist(),
            "source":            "kinematic_ctr",
            "is_gt":             False,
        }
        return {"raw": path_data, "serialized": serialized}


# ---------------------------------------------------------------------------
# Plugin wrapper — Strategy Pattern
# ---------------------------------------------------------------------------

class KinematicPlugin(AbstractLaneDetector):
    """
    Adapter: exposes ``KinematicEgoStrategy`` as an ``AbstractLaneDetector``
    plugin.

    Design pattern: Adapter.  The heavy logic lives in ``KinematicEgoStrategy``
    (CTR prediction, EMA, finite-difference yaw rate); this class maps the
    generic plugin API (``process`` / ``reset``) to that interface without
    altering the underlying strategy.  Image dimensions are accepted by the
    constructor to satisfy the plugin contract but are intentionally unused —
    kinematic prediction is purely physics-based.
    """

    def __init__(
        self,
        cfg:          DictConfig,
        image_width:  int = 1920,
        image_height: int = 1280,
    ) -> None:
        self._strategy = KinematicEgoStrategy(cfg)

    def process(
        self,
        frame_bgr:     Optional[np.ndarray],
        vehicle_state: VehicleState,
    ) -> dict:
        result = self._strategy.compute(
            speed_mps      = vehicle_state.speed_mps,
            curr_transform = vehicle_state.curr_transform,
            curr_timestamp = vehicle_state.curr_timestamp,
        )
        return {
            "kinematic_raw": result["raw"],
            "kinematic":     result["serialized"],
        }

    def reset(self) -> None:
        self._strategy.reset_segment_state()
