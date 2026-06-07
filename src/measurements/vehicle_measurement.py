"""
src/models/tracking/vehicle_measurement.py
============================================
Single-Frame (SF) measurement pipeline for vehicle tracking.

This module is the geometric front-end of the EKF pipeline.  It translates a
2D bounding-box detection into 3D vehicle-frame measurements using three
independent geometric methods, then fuses them into a clean KalmanMeasurement
that VehicleEKF.update() consumes directly.

Pipeline (called once per detection per frame)
----------------------------------------------

    Step 1 — compute_sf_measurements(bbox_xyxy, fx, fy, K, R_vc, t_vc)
        Compute three independent single-frame (SF) position estimates:

        a) y0_proj    : Bottom-centre pixel back-projected to Z_veh = 0.
                        Provides (x_m, y_m).  Most geometrically accurate
                        when the vehicle contact point lies on the road.

        b) height_proj: Range from apparent bbox height + height prior.
                        x = f_y * H_prior / bbox_h_px.
                        Provides x_m only; independent of ground-plane tilt.

        c) width_proj : Range from apparent bbox width + width prior.
                        x = f_x * W_prior / bbox_w_px.
                        Provides x_m only; symmetric cross-check to h_proj.

        Also computes the raw EKF aspect-ratio observations:
            h_aspect = bbox_h_px / fy  ≈  H_real / x  (nonlinear in x, H)
            w_aspect = bbox_w_px / fx  ≈  W_real / x  (nonlinear in x, W)

    Step 2 — build_kalman_input(bundle) → KalmanMeasurement | None
        Map the SFMeasurementBundle to the 4D EKF measurement vector
        [x_gnd, y_gnd, h_aspect, w_aspect].

        • Primary position source: y0_proj (required).
        • Aspect measurements included only when bbox is large enough
          AND the vehicle is beyond the near-field singularity threshold.
        • Returns None when y0_proj is invalid → caller skips EKF update.

Coordinate conventions (Vehicle Frame)
---------------------------------------
    X = forward  (positive = ahead of ego)
    Y = left     (positive = to the left of ego)
    Z = up       (positive = above ground)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# NOTE: project_bbox_to_ground is NOT imported from rw_coordinates here;
# the ground-projection math is duplicated below for self-containment.

# ---------------------------------------------------------------------------
# Ground-plane projection (self-contained to avoid circular imports)
# ---------------------------------------------------------------------------

def _project_bottom_centre_to_ground(
    bbox_xyxy: np.ndarray,
    K:         np.ndarray,
    R_vc:      np.ndarray,
    t_vc:      np.ndarray,
) -> tuple[float, float] | None:
    """
    Back-project the bottom-centre pixel of a bounding box to Z_veh = 0.

    Math (see also src/features/rw_coordinates.py for the full derivation):
        1. d_cam = K⁻¹ · [u, v, 1]ᵀ      (camera-frame ray)
        2. d_veh = R_vc.T · d_cam          (vehicle-frame ray)
        3. O_veh = -R_vc.T · t_vc          (camera origin in vehicle frame)
        4. t = -O_veh[Z] / d_veh[Z]        (intersect with Z = 0)
        5. P_veh = O_veh + t · d_veh       (ground point)

    Returns (range_m, lateral_m) or None when the intersection is invalid.
    """
    u = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)   # horizontal centre
    v = float(bbox_xyxy[3])                            # bottom edge

    K_inv = np.linalg.inv(K)
    R_cv  = R_vc.T   # camera frame → vehicle frame

    d_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    d_veh = R_cv @ d_cam
    O_veh = -R_cv @ t_vc

    if abs(d_veh[2]) < 1e-6:
        return None   # ray nearly parallel to ground — no valid intersection

    t = -O_veh[2] / d_veh[2]
    if t <= 0.0:
        return None   # intersection behind the camera

    P_veh = O_veh + t * d_veh
    return float(P_veh[0]), float(P_veh[1])   # (range_m, lateral_m)


# ---------------------------------------------------------------------------
# Physical priors (shared with VehicleEKF for consistency)
# ---------------------------------------------------------------------------

PRIOR_HEIGHT_M: float = 1.5   # typical passenger vehicle height (m)
PRIOR_WIDTH_M:  float = 2.0   # typical passenger vehicle width  (m)

# Minimum bounding-box pixel dimension for reliable aspect-ratio observations
_MIN_BBOX_PX: float = 10.0

# Range below which perspective aspect-ratio measurements are unreliable
_NEAR_FIELD_M: float = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SFPositionEstimate:
    """
    One single-frame position estimate from a single geometric method.

    Attributes
    ----------
    method : str
        Source identifier: ``"y0_proj"`` | ``"height_proj"`` | ``"width_proj"``.
    x_m : float
        Forward range estimate (m), Vehicle Frame (X-axis).
        ``nan`` on failure.
    y_m : float
        Lateral offset (m), Vehicle Frame (Y-axis), positive = left.
        ``nan`` when the method produces no lateral information.
    valid : bool
        True when x_m > 0 and all sanity checks passed.
    """
    method: str
    x_m:    float
    y_m:    float
    valid:  bool

    @classmethod
    def invalid(cls, method: str) -> "SFPositionEstimate":
        """Construct a sentinel invalid estimate for the given method name."""
        return cls(method=method, x_m=float("nan"), y_m=float("nan"), valid=False)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "x_m":    None if math.isnan(self.x_m) else round(self.x_m, 3),
            "y_m":    None if math.isnan(self.y_m) else round(self.y_m, 3),
            "valid":  self.valid,
        }


@dataclass
class SFMeasurementBundle:
    """
    All single-frame geometric measurements for one detection in one frame.

    Stores three independent SF position estimates plus the raw EKF
    aspect-ratio observations so every source can be logged, visualised,
    or compared independently.

    Attributes
    ----------
    bbox_xyxy : np.ndarray, shape (4,)
        Detection bounding box [x1, y1, x2, y2] in image pixels.
    bbox_w_px, bbox_h_px : float
        Width and height of the bounding box in pixels.
    y0_proj : SFPositionEstimate
        Ground-plane (Z_veh = 0) projection of the bbox bottom-centre.
        Method: back-project pixel ray and intersect with Z = 0 plane.
    height_proj : SFPositionEstimate
        Range from apparent bbox height + vehicle height prior.
        Method: x = fy * H_prior / bbox_h_px.
    width_proj : SFPositionEstimate
        Range from apparent bbox width + vehicle width prior.
        Method: x = fx * W_prior / bbox_w_px.
    h_aspect : float
        Raw EKF observation: ``bbox_h_px / fy``.  Approximates ``H_real / x``.
        ``nan`` when bbox is below the minimum size threshold.
    w_aspect : float
        Raw EKF observation: ``bbox_w_px / fx``.  Approximates ``W_real / x``.
        ``nan`` when bbox is below the minimum size threshold.
    """
    bbox_xyxy:   np.ndarray
    bbox_w_px:   float
    bbox_h_px:   float
    y0_proj:     SFPositionEstimate
    height_proj: SFPositionEstimate
    width_proj:  SFPositionEstimate
    h_aspect:    float   # nan when bbox too small
    w_aspect:    float   # nan when bbox too small

    def to_dict(self) -> dict:
        return {
            "bbox_w_px":   round(self.bbox_w_px, 1),
            "bbox_h_px":   round(self.bbox_h_px, 1),
            "y0_proj":     self.y0_proj.to_dict(),
            "height_proj": self.height_proj.to_dict(),
            "width_proj":  self.width_proj.to_dict(),
            "h_aspect":    None if math.isnan(self.h_aspect) else round(self.h_aspect, 4),
            "w_aspect":    None if math.isnan(self.w_aspect) else round(self.w_aspect, 4),
        }


@dataclass
class KalmanMeasurement:
    """
    4D EKF measurement vector derived from one SFMeasurementBundle.

    This is the clean, validated interface between the geometric measurement
    layer (Step 1) and the Kalman filter update (Step 3).

    Measurement model:
        z = [x_gnd, y_gnd, h_aspect, w_aspect]  (full 4D when use_size=True)
        z = [x_gnd, y_gnd]                       (2D position when use_size=False)

    Attributes
    ----------
    x_gnd : float
        Forward range from ground-plane projection (m).
    y_gnd : float
        Lateral offset from ground-plane projection (m).
    h_aspect : float
        ``bbox_h_px / fy`` — perspective height-to-range ratio.
        Meaningful only when ``use_size`` is True.
    w_aspect : float
        ``bbox_w_px / fx`` — perspective width-to-range ratio.
        Meaningful only when ``use_size`` is True.
    use_size : bool
        True → full 4D EKF update (position + size).
        False → 2D position-only update (aspect measurements unreliable).
    """
    x_gnd:    float
    y_gnd:    float
    h_aspect: float
    w_aspect: float
    use_size: bool

    def to_dict(self) -> dict:
        return {
            "x_gnd":    round(self.x_gnd, 3),
            "y_gnd":    round(self.y_gnd, 3),
            "h_aspect": None if math.isnan(self.h_aspect) else round(self.h_aspect, 4),
            "w_aspect": None if math.isnan(self.w_aspect) else round(self.w_aspect, 4),
            "use_size": self.use_size,
        }


# ---------------------------------------------------------------------------
# Step 1: compute_sf_measurements
# ---------------------------------------------------------------------------

def compute_sf_measurements(
    bbox_xyxy:      np.ndarray,
    fx:             float,
    fy:             float,
    K:              np.ndarray,
    R_vc:           np.ndarray,
    t_vc:           np.ndarray,
    height_prior_m: float = PRIOR_HEIGHT_M,
    width_prior_m:  float = PRIOR_WIDTH_M,
    min_bbox_px:    float = _MIN_BBOX_PX,
) -> SFMeasurementBundle:
    """
    Step 1: Compute all single-frame (SF) measurements from one bounding box.

    Three independent geometric methods are computed and stored as separate
    ``SFPositionEstimate`` objects so each result can be inspected, logged,
    or fused independently.

    Parameters
    ----------
    bbox_xyxy : array-like, shape (4,)
        Bounding box [x1, y1, x2, y2] in image pixels.
    fx, fy : float
        Camera focal lengths (pixels).  Extracted from K[:,0] and K[:,1].
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    R_vc : np.ndarray, shape (3, 3)
        Rotation: Vehicle Frame → Camera Frame.
    t_vc : np.ndarray, shape (3,)
        Translation: P_cam = R_vc @ P_veh + t_vc.
    height_prior_m : float
        Vehicle height prior (m) for height_proj.
    width_prior_m : float
        Vehicle width prior (m) for width_proj.
    min_bbox_px : float
        Minimum bbox dimension (px) below which aspect measurements are NaN.

    Returns
    -------
    SFMeasurementBundle
        All SF estimates and aspect-ratio observations for this detection.
    """
    bbox      = np.asarray(bbox_xyxy, dtype=np.float64)
    bbox_w_px = float(bbox[2] - bbox[0])
    bbox_h_px = float(bbox[3] - bbox[1])

    # ── Method a: y=0 ground-plane projection ────────────────────────────────
    # Back-project the bottom-centre pixel to the Z_veh = 0 road surface.
    # This uses the full pinhole model + extrinsics and gives (x_m, y_m).
    gnd = _project_bottom_centre_to_ground(bbox, K, R_vc, t_vc)
    if gnd is not None and gnd[0] > 0.0:
        y0_proj = SFPositionEstimate(
            method="y0_proj",
            x_m=float(gnd[0]),
            y_m=float(gnd[1]),
            valid=True,
        )
    else:
        y0_proj = SFPositionEstimate.invalid("y0_proj")

    # ── Method b: height-prior range projection ───────────────────────────────
    # Perspective depth from apparent height:  x = fy * H_prior / bbox_h_px
    # Provides range only; y_m = NaN because vertical size carries no lateral info.
    if fy > 1.0 and bbox_h_px >= min_bbox_px:
        x_from_h = fy * height_prior_m / bbox_h_px
        height_proj = SFPositionEstimate(
            method="height_proj",
            x_m=float(x_from_h),
            y_m=float("nan"),
            valid=x_from_h > 0.0,
        )
    else:
        height_proj = SFPositionEstimate.invalid("height_proj")

    # ── Method c: width-prior range projection ────────────────────────────────
    # Perspective depth from apparent width:  x = fx * W_prior / bbox_w_px
    # Provides range only; y_m = NaN because horizontal size carries no unique
    # lateral info without also knowing the camera pointing direction.
    if fx > 1.0 and bbox_w_px >= min_bbox_px:
        x_from_w = fx * width_prior_m / bbox_w_px
        width_proj = SFPositionEstimate(
            method="width_proj",
            x_m=float(x_from_w),
            y_m=float("nan"),
            valid=x_from_w > 0.0,
        )
    else:
        width_proj = SFPositionEstimate.invalid("width_proj")

    # ── EKF perspective aspect-ratio observations ─────────────────────────────
    # h_aspect = bbox_h_px / fy  ≈  H_real / x   (EKF measurement h₃)
    # w_aspect = bbox_w_px / fx  ≈  W_real / x   (EKF measurement h₄)
    # These are the raw nonlinear observations fed directly to the EKF update;
    # they are distinct from the prior-based range estimates above.
    if bbox_h_px >= min_bbox_px and bbox_w_px >= min_bbox_px:
        h_aspect = bbox_h_px / fy
        w_aspect = bbox_w_px / fx
    else:
        h_aspect = float("nan")
        w_aspect = float("nan")

    return SFMeasurementBundle(
        bbox_xyxy=bbox,
        bbox_w_px=bbox_w_px,
        bbox_h_px=bbox_h_px,
        y0_proj=y0_proj,
        height_proj=height_proj,
        width_proj=width_proj,
        h_aspect=h_aspect,
        w_aspect=w_aspect,
    )


# ---------------------------------------------------------------------------
# Step 2: build_kalman_input
# ---------------------------------------------------------------------------

def build_kalman_input(
    bundle:       SFMeasurementBundle,
    near_field_m: float = _NEAR_FIELD_M,
) -> KalmanMeasurement | None:
    """
    Step 2: Convert an SFMeasurementBundle into a KalmanMeasurement.

    Uses y0_proj as the primary position source.  Aspect-ratio measurements
    are promoted to the EKF only when the bbox is large enough (both h_aspect
    and w_aspect are valid) AND the vehicle is beyond the near-field threshold.

    Parameters
    ----------
    bundle : SFMeasurementBundle
        Output of :func:`compute_sf_measurements`.
    near_field_m : float
        Minimum range (m) below which aspect measurements are excluded from
        the EKF update to prevent near-field perspective instability.

    Returns
    -------
    KalmanMeasurement | None
        Ready-to-use EKF measurement, or ``None`` when y0_proj is invalid
        (signals the manager to skip EKF update for this detection this frame).
    """
    # y0_proj is required — no ground-plane fix means no EKF update.
    if not bundle.y0_proj.valid:
        return None

    x_gnd = bundle.y0_proj.x_m
    y_gnd = bundle.y0_proj.y_m

    # Include size measurements only when they are both geometrically reliable.
    h_aspect = bundle.h_aspect
    w_aspect = bundle.w_aspect
    use_size = (
        not math.isnan(h_aspect)
        and not math.isnan(w_aspect)
        and x_gnd > near_field_m
    )

    return KalmanMeasurement(
        x_gnd=x_gnd,
        y_gnd=y_gnd,
        h_aspect=h_aspect,
        w_aspect=w_aspect,
        use_size=use_size,
    )
