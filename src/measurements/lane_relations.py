"""
src/features/lane_relations.py
================================
LaneRelationMeasurer — per-vehicle spatial relation to every active path type.

For each (vehicle_track × path_type) pair the measurer produces two classes
of measurement:

Ground-plane distance (Vehicle Frame, metres)
---------------------------------------------
The lateral signed offset from the path centerline.  Computed by finding the
nearest point on the centerline polyline and projecting the vehicle's position
onto the local tangent:
    + positive → vehicle is LEFT  of the path
    - negative → vehicle is RIGHT of the path
Paths with pixel-coordinate polylines (drivable_path, host_lane, hdmap) are
first converted to Vehicle Frame via the camera ground-plane projection.

Image-plane distance (pixels)
------------------------------
Euclidean pixel distance from the vehicle bounding-box centre to the nearest
point in the path polyline.  Useful for image-space proximity alerts and
visual debugging.

Path types handled
------------------
    kinematic     — ego CTR predicted path.  Raw data already in Vehicle Frame
                    (x=forward, y=left, metres) from KinematicPathPredictor.
    drivable_path — YOLOPv2 drivable-area centerline + left/right bounds (px).
    host_lane     — YOLOPv2 host-lane left/right markings (pixels).
    hdmap         — HD-map projected lanes (pixels).

Inside-bounds detection
-----------------------
For paths with both left and right boundaries (drivable_path, host_lane,
hdmap) the measurer interpolates each boundary at the vehicle's forward range
and tests whether the vehicle's lateral position lies between them.

Coordinate system
-----------------
Vehicle Frame: X = forward, Y = left, Z = up.  Origin at rear axle (Waymo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from omegaconf import DictConfig

from src.visualization.visualizer import CameraCalibration
from .vehicle_track_manager import VehicleTrackState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signed-distance threshold for "on" classification
# ---------------------------------------------------------------------------

_ON_THRESHOLD_M: float = 0.5   # m — within this lateral offset → "on"


# ---------------------------------------------------------------------------
# LaneRelation — result container for one (track × path) pair
# ---------------------------------------------------------------------------

@dataclass
class LaneRelation:
    """
    Spatial relation between one vehicle track and one path type.

    Attributes
    ----------
    path_type : str
        One of: "kinematic", "drivable_path", "host_lane", "hdmap".
    valid : bool
        False when the path has no usable polyline (no inference run,
        failed detection, etc.).  All distance fields are 0.0 when False.
    dist_lateral_m : float
        Signed lateral offset from the nearest point on the path centerline.
        Positive  = vehicle is LEFT  of the path.
        Negative  = vehicle is RIGHT of the path.
    dist_to_center_m : float
        Unsigned planar distance to the nearest centerline point (m).
    dist_bbox_px : float
        Minimum pixel distance from vehicle bbox centre to path polyline.
    inside_bounds : bool
        True when the vehicle lateral position lies between the path's left
        and right boundaries at the vehicle's forward range.
        Always False for paths without boundaries (kinematic).
    side : str
        "left" | "right" | "on"  — derived from dist_lateral_m sign and
        the _ON_THRESHOLD_M constant.
    """
    path_type:        str
    valid:            bool
    dist_lateral_m:   float
    dist_to_center_m: float
    dist_bbox_px:     float
    inside_bounds:    bool
    side:             str

    def to_dict(self) -> dict:
        return {
            "path_type":        self.path_type,
            "valid":            self.valid,
            "dist_lateral_m":   round(self.dist_lateral_m,   2),
            "dist_to_center_m": round(self.dist_to_center_m, 2),
            "dist_bbox_px":     round(self.dist_bbox_px,      1),
            "inside_bounds":    self.inside_bounds,
            "side":             self.side,
        }


def _null_relation(path_type: str) -> LaneRelation:
    return LaneRelation(
        path_type=path_type, valid=False,
        dist_lateral_m=0.0, dist_to_center_m=0.0,
        dist_bbox_px=0.0, inside_bounds=False, side="unknown",
    )


# ---------------------------------------------------------------------------
# LaneRelationMeasurer
# ---------------------------------------------------------------------------

class LaneRelationMeasurer:
    """
    Computes lane relations for all vehicle tracks in one frame.

    Parameters
    ----------
    calib : CameraCalibration
        Camera intrinsics and extrinsics.  Used to convert pixel polylines
        (drivable_path, host_lane, hdmap) into Vehicle Frame coordinates for
        ground-plane distance measurement.
        Update between segments via ``update_calib()``.
    """

    def __init__(self, calib: CameraCalibration) -> None:
        self._calib = calib

    def update_calib(self, calib: CameraCalibration) -> None:
        """Refresh calibration when it changes between segments (once per segment)."""
        self._calib = calib

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        vehicle_tracks: list[VehicleTrackState],
        lane_results:   dict,
        gt_data:        dict,
    ) -> list[dict]:
        """
        Compute lane relations for every vehicle track against all path types.

        Parameters
        ----------
        vehicle_tracks : list[VehicleTrackState]
            Output of VehicleTrackManager.update() for this frame.
        lane_results : dict
            Output of LaneManager.process() — includes kinematic_raw,
            drivable_path, host_lane, kinematic serialized dicts.
        gt_data : dict
            Pipeline frame dict — used for gt_data["hdmap"].

        Returns
        -------
        list[dict]
            One entry per vehicle track::

                {
                    "track_id": int,
                    "x_veh":    float,
                    "y_veh":    float,
                    "relations": {
                        "kinematic":     { ... LaneRelation.to_dict() },
                        "drivable_path": { ... },
                        "host_lane":     { ... },
                        "hdmap":         { ... },
                    }
                }
        """
        if not vehicle_tracks:
            return []

        # ── Pre-process path polylines once per frame (not per track) ────────
        # kinematic_raw is already in Vehicle Frame (x forward, y left, metres)
        kin_vf = self._kinematic_centerline_vf(lane_results)

        # drivable_path: center/left/right in image pixels → convert to VF
        dp_ser = lane_results.get("drivable_path", {})
        dp_center_vf = self._pixels_to_vf(dp_ser.get("center", []))
        dp_left_vf   = self._pixels_to_vf(dp_ser.get("left",   []))
        dp_right_vf  = self._pixels_to_vf(dp_ser.get("right",  []))

        # host_lane: left/right boundaries in pixels → VF; center derived
        hl_ser = lane_results.get("host_lane", {})
        hl_left_vf  = self._pixels_to_vf(hl_ser.get("left",  []))
        hl_right_vf = self._pixels_to_vf(hl_ser.get("right", []))
        hl_center_vf = self._boundary_midline(hl_left_vf, hl_right_vf)

        # hdmap: center/left/right in pixels → VF
        hm_ser = gt_data.get("hdmap", {})
        hm_center_vf = self._pixels_to_vf(hm_ser.get("center", []))
        hm_left_vf   = self._pixels_to_vf(hm_ser.get("left",   []))
        hm_right_vf  = self._pixels_to_vf(hm_ser.get("right",  []))

        # Pixel-space polylines for image-plane dist_bbox_px measurement
        kin_px      = self._to_px_array(lane_results.get("kinematic",    {}).get("center", []))
        dp_center_px = self._to_px_array(dp_ser.get("center", []))
        # For host_lane, use the concatenation of both boundary pixel arrays
        hl_px       = self._concat_px(
            self._to_px_array(hl_ser.get("left",  [])),
            self._to_px_array(hl_ser.get("right", [])),
        )
        hm_px       = self._to_px_array(hm_ser.get("center", []))

        # ── Compute per-track relations ───────────────────────────────────────
        results: list[dict] = []
        for trk in vehicle_tracks:
            vx, vy  = trk.x_veh, trk.y_veh
            bbox_cx = float((trk.bbox_xyxy[0] + trk.bbox_xyxy[2]) / 2.0)
            bbox_cy = float((trk.bbox_xyxy[1] + trk.bbox_xyxy[3]) / 2.0)

            relations = {
                "kinematic": self._measure_center_only(
                    vx, vy, bbox_cx, bbox_cy,
                    kin_vf, kin_px,
                    "kinematic",
                ).to_dict(),

                "drivable_path": self._measure_with_bounds(
                    vx, vy, bbox_cx, bbox_cy,
                    dp_center_vf, dp_center_px,
                    dp_left_vf, dp_right_vf,
                    "drivable_path",
                ).to_dict(),

                "host_lane": self._measure_with_bounds(
                    vx, vy, bbox_cx, bbox_cy,
                    hl_center_vf, hl_px,
                    hl_left_vf, hl_right_vf,
                    "host_lane",
                ).to_dict(),

                "hdmap": self._measure_with_bounds(
                    vx, vy, bbox_cx, bbox_cy,
                    hm_center_vf, hm_px,
                    hm_left_vf, hm_right_vf,
                    "hdmap",
                ).to_dict(),
            }

            results.append({
                "track_id": trk.track_id,
                "x_veh":    round(vx, 2),
                "y_veh":    round(vy, 2),
                "relations": relations,
            })

        return results

    # ------------------------------------------------------------------
    # Path preparation helpers
    # ------------------------------------------------------------------

    def _kinematic_centerline_vf(self, lane_results: dict) -> np.ndarray:
        """
        Extract the kinematic path centerline already in Vehicle Frame (m).
        kinematic_raw["centre_line"] is (N, 2) with columns (x_fwd, y_left).
        """
        raw = lane_results.get("kinematic_raw", {})
        pts = raw.get("centre_line", None)
        if pts is None:
            return np.empty((0, 2), dtype=np.float64)
        arr = np.asarray(pts, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float64)
        return arr[:, :2]

    def _pixels_to_vf(self, pixel_list: list) -> np.ndarray:
        """
        Convert [[u, v], ...] image pixels to Vehicle Frame (x, y) in metres
        using the camera ground-plane ray intersection (Z_veh = 0 plane).
        Points where the ray is parallel to the ground or behind the camera
        are silently dropped.
        """
        if not pixel_list:
            return np.empty((0, 2), dtype=np.float64)

        K    = self._calib.K
        R_cv = self._calib.R_vc.T          # camera frame → vehicle frame
        t_vc = self._calib.t_vc
        K_inv = np.linalg.inv(K)
        O_veh = -R_cv @ t_vc               # camera origin in vehicle frame

        pts: list[list[float]] = []
        for p in pixel_list:
            if len(p) < 2:
                continue
            d_cam = K_inv @ np.array([float(p[0]), float(p[1]), 1.0], dtype=np.float64)
            d_veh = R_cv @ d_cam
            if abs(d_veh[2]) < 1e-6:
                continue
            t = -O_veh[2] / d_veh[2]
            if t <= 0.0:
                continue
            P = O_veh + t * d_veh
            pts.append([float(P[0]), float(P[1])])

        if not pts:
            return np.empty((0, 2), dtype=np.float64)
        return np.array(pts, dtype=np.float64)

    @staticmethod
    def _to_px_array(pixel_list: list) -> np.ndarray:
        """Convert [[u, v], ...] to float64 (N, 2) array."""
        if not pixel_list:
            return np.empty((0, 2), dtype=np.float64)
        arr = np.asarray(pixel_list, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float64)
        return arr[:, :2]

    @staticmethod
    def _concat_px(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Concatenate two pixel arrays; return empty if both empty."""
        if len(a) == 0 and len(b) == 0:
            return np.empty((0, 2), dtype=np.float64)
        if len(a) == 0:
            return b
        if len(b) == 0:
            return a
        return np.concatenate([a, b], axis=0)

    @staticmethod
    def _boundary_midline(
        left_vf: np.ndarray, right_vf: np.ndarray
    ) -> np.ndarray:
        """
        Derive a centerline from left/right boundary arrays in Vehicle Frame.
        Interpolates both boundaries onto a common forward-range axis.
        """
        if len(left_vf) < 2 or len(right_vf) < 2:
            return np.empty((0, 2), dtype=np.float64)

        ls = left_vf[np.argsort(left_vf[:, 0])]
        rs = right_vf[np.argsort(right_vf[:, 0])]

        x_lo = max(float(ls[0, 0]),  float(rs[0, 0]))
        x_hi = min(float(ls[-1, 0]), float(rs[-1, 0]))
        if x_hi <= x_lo:
            return np.empty((0, 2), dtype=np.float64)

        x_c = np.linspace(x_lo, x_hi, 20)
        yl  = np.interp(x_c, ls[:, 0], ls[:, 1])
        yr  = np.interp(x_c, rs[:, 0], rs[:, 1])
        return np.column_stack([x_c, (yl + yr) / 2.0])

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------

    def _measure_center_only(
        self,
        vx: float, vy: float,
        bbox_cx: float, bbox_cy: float,
        center_vf: np.ndarray,
        center_px: np.ndarray,
        path_type: str,
    ) -> LaneRelation:
        """Measure relation to a path that only provides a centerline."""
        if len(center_vf) < 2:
            return _null_relation(path_type)

        lat, dist = _lateral_offset(vx, vy, center_vf)
        px_d      = _pixel_dist(bbox_cx, bbox_cy, center_px)

        return LaneRelation(
            path_type=path_type, valid=True,
            dist_lateral_m=lat,   dist_to_center_m=dist,
            dist_bbox_px=px_d,    inside_bounds=False,
            side=_side(lat),
        )

    def _measure_with_bounds(
        self,
        vx: float, vy: float,
        bbox_cx: float, bbox_cy: float,
        center_vf: np.ndarray,
        center_px: np.ndarray,
        left_vf:   np.ndarray,
        right_vf:  np.ndarray,
        path_type: str,
    ) -> LaneRelation:
        """Measure relation to a path that provides left/right boundaries."""
        # Fall back to a derived midline when explicit center is empty
        eff_center = (
            center_vf if len(center_vf) >= 2
            else self._boundary_midline(left_vf, right_vf)
        )
        if len(eff_center) < 2:
            return _null_relation(path_type)

        lat, dist = _lateral_offset(vx, vy, eff_center)
        px_d      = _pixel_dist(bbox_cx, bbox_cy, center_px)
        inside    = _check_inside(vx, vy, left_vf, right_vf)

        return LaneRelation(
            path_type=path_type, valid=True,
            dist_lateral_m=lat,  dist_to_center_m=dist,
            dist_bbox_px=px_d,   inside_bounds=inside,
            side=_side(lat),
        )


# ---------------------------------------------------------------------------
# Pure functions (no instance state — testable in isolation)
# ---------------------------------------------------------------------------

def _lateral_offset(
    px: float, py: float, polyline: np.ndarray
) -> tuple[float, float]:
    """
    Signed lateral offset and unsigned distance from (px, py) to the polyline.

    Strategy: iterate over every segment, project the point onto it (clamped
    to the segment endpoints), and keep the closest foot-of-perpendicular.
    The signed lateral is computed from the 2D cross-product against the
    local segment direction at the winning foot point.

    Returns
    -------
    (signed_lateral_m, unsigned_dist_m)
        signed_lateral_m > 0 → vehicle is LEFT  of the path direction.
        signed_lateral_m < 0 → vehicle is RIGHT of the path direction.
    """
    point = np.array([px, py], dtype=np.float64)

    best_dist:   float = float("inf")
    best_signed: float = 0.0

    for i in range(len(polyline) - 1):
        p0  = polyline[i]
        p1  = polyline[i + 1]
        seg = p1 - p0
        seg_len_sq = float(np.dot(seg, seg))

        if seg_len_sq < 1e-12:
            # Degenerate (zero-length) segment — fall back to vertex distance
            d = float(np.linalg.norm(point - p0))
            if d < best_dist:
                best_dist   = d
                best_signed = 0.0
            continue

        # Project point onto segment, clamp t ∈ [0, 1] to stay on the segment
        t    = float(np.dot(point - p0, seg)) / seg_len_sq
        foot = p0 + np.clip(t, 0.0, 1.0) * seg
        diff = point - foot
        d    = float(np.linalg.norm(diff))

        if d < best_dist:
            best_dist = d
            # 2D cross-product z-component: seg.x*diff.y − seg.y*diff.x
            # Positive → point is counter-clockwise (LEFT) of the segment direction
            cross       = float(seg[0] * diff[1] - seg[1] * diff[0])
            best_signed = float(np.copysign(d, cross))

    if best_dist == float("inf"):
        return 0.0, 0.0

    return best_signed, best_dist


def _pixel_dist(cx: float, cy: float, polyline_px: np.ndarray) -> float:
    """Minimum Euclidean pixel distance from (cx, cy) to any polyline point."""
    if len(polyline_px) == 0:
        return 0.0
    diffs = polyline_px - np.array([cx, cy], dtype=np.float64)
    return float(np.min(np.hypot(diffs[:, 0], diffs[:, 1])))


def _check_inside(
    vx: float, vy: float,
    left_vf:  np.ndarray,
    right_vf: np.ndarray,
) -> bool:
    """
    Test whether the vehicle (vx, vy) lies between the left and right
    boundaries at its forward range, using linear interpolation.
    """
    if len(left_vf) < 2 or len(right_vf) < 2:
        return False

    ls = left_vf[np.argsort(left_vf[:, 0])]
    rs = right_vf[np.argsort(right_vf[:, 0])]

    x_lo = max(float(ls[0, 0]),  float(rs[0, 0]))
    x_hi = min(float(ls[-1, 0]), float(rs[-1, 0]))

    if vx < x_lo or vx > x_hi:
        return False

    y_left  = float(np.interp(vx, ls[:, 0], ls[:, 1]))
    y_right = float(np.interp(vx, rs[:, 0], rs[:, 1]))

    # Ensure left > right in Y (left is positive in vehicle frame)
    if y_right > y_left:
        y_left, y_right = y_right, y_left

    return bool(y_right <= vy <= y_left)


def _side(lat: float) -> str:
    """Classify lateral offset into "left" / "right" / "on"."""
    if abs(lat) < _ON_THRESHOLD_M:
        return "on"
    return "left" if lat > 0.0 else "right"
