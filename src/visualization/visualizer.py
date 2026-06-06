"""
src/visualization/visualizer.py
================================
Unified perception visualizer for the ADAS/AV pipeline.

Combines four annotation layers onto a single BGR image:

    Layer 1 — GT Bounding Boxes (from JSON ground truth)
    Layer 2 — Kinematic Path    (3D Vehicle Frame → projected to 2D pixels)
    Layer 3 — Visual Ego-Lanes  (2D pixel polylines from VisualLaneDetector)
    Layer 4 — HUD overlay       (frame index, ego speed, object count)

Core design rules:
    • The input image is NEVER mutated. All drawing is done on a copy.
    • 3D→2D projection uses a standard pinhole model: u = K · [R|t] · P_veh
    • Points behind the camera plane (Z_cam ≤ 0) are silently skipped.
    • All public methods accept and return BGR np.ndarray (uint8).

Typical usage:
    from src.visualization.visualizer import CameraCalibration, PerceptionVisualizer

    calib = CameraCalibration.default_front()           # or .from_waymo_camera(cam)
    vis   = PerceptionVisualizer(calib)

    canvas = vis.draw_all(
        img       = bgr_frame,
        gt_data   = gt_dict,          # from JSON / waymo_parser
        path_data = predictor.predict(...),   # KinematicPathPredictor output
        lane_data = detector.detect(...),     # VisualLaneDetector output
        frame_idx = step,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Color palette (BGR)
# ---------------------------------------------------------------------------

# Ground-truth bounding boxes
_GT_TYPE_COLORS: dict[int, tuple[int, int, int]] = {
    1: (255, 100,  50),   # Vehicle    — blue-orange
    2: ( 50, 220,  80),   # Pedestrian — green
    3: (200, 200, 200),   # Sign       — grey
    4: ( 50, 220, 255),   # Cyclist    — yellow
}
_GT_DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)
_GT_TYPE_NAMES: dict[int, str] = {
    1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist",
}

# Kinematic path
# Centre line: bright yellow — the primary ego trajectory.
# L/R boundaries: dark amber — narrower, clearly subordinate to the centre,
# and visually distinct from both the DP teal bounds and host-lane purple.
_PATH_COLOR_LEFT   = (  0, 110, 210)   # dark amber-orange — left wheel track
_PATH_COLOR_RIGHT  = (  0, 110, 210)   # dark amber-orange — right wheel track
_PATH_COLOR_CENTRE = (  0, 220, 255)   # bright yellow     — ego centre line
_PATH_LINE_THICK   = 2                 # centre-line stroke width (px)
_PATH_BOUND_THICK  = 1                 # boundary stroke width (px, thinner)
_PATH_DOT_RADIUS   = 4

# Visual lanes — blue (CLRNet / IPM)
_LANE_COLOR_LEFT   = (220,  80,   0)   # blue
_LANE_COLOR_RIGHT  = (255, 140,   0)   # lighter blue
_LANE_LINE_THICK   = 3

# HD Map lanes — green (thick so it reads clearly)
_HDMAP_COLOR_LEFT  = ( 20, 230,  20)   # bright green (BGR)
_HDMAP_COLOR_RIGHT = ( 10, 160,  10)   # darker green (BGR)
_HDMAP_LINE_THICK  = 3

# Drivable Path — teal/cyan center line (Path 3)
_DRIVABLE_COLOR        = (200, 200,   0)   # bright cyan  — drivable centre
_DRIVABLE_BOUND_COLOR  = (140, 200,   0)   # olive-cyan   — drivable left/right
_DRIVABLE_LINE_THICK   = 3
_DRIVABLE_BOUND_THICK  = 2

# Host Lane markings — purple / magenta (Path 4, only when valid)
_HOST_LANE_COLOR_LEFT  = (210,  30, 210)   # purple-left  (BGR)
_HOST_LANE_COLOR_RIGHT = (180,   0, 255)   # magenta-right (BGR)
_HOST_LANE_LINE_THICK  = 3
# Dim color used to draw host-lane lines that are below the validity threshold.
# Makes failed detections visible for debugging without implying they are trusted.
_HOST_LANE_INVALID_COLOR = (80, 40, 80)   # very dark purple (BGR)

# Legend / label font
_LABEL_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE = 0.50
_LABEL_THICKNESS  = 1

# HUD
_HUD_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_HUD_FONT_SCALE = 0.60
_HUD_THICKNESS  = 1

# YOLO detections / Kalman tracks — visually distinct from GT colors
_DET_RAW_COLOR: tuple = (200,   0, 200)   # magenta — raw unconfirmed YOLO box
_DET_COAST_COLOR: tuple = (100, 100, 200) # muted blue — coasting track (no meas)
_DET_TRACK_COLORS: dict[int, tuple] = {
    0: (  0, 200, 255),   # vehicle    — amber
    1: (255, 200,   0),   # pedestrian — sky blue
    2: (100, 255, 200),   # cyclist    — mint
    3: (150, 150, 150),   # other      — grey
}


def _draw_dashed_rect(
    canvas: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thickness: int,
    dash: int = 10,
) -> None:
    """Draw a dashed rectangle (used for coasting tracks)."""
    def _dashed_line(p1, p2):
        dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1:
            return
        step = 2 * dash
        n    = max(1, int(dist / step))
        for i in range(n):
            t0 = i * step / dist
            t1 = min((i * step + dash) / dist, 1.0)
            s  = (int(p1[0] + dx * t0), int(p1[1] + dy * t0))
            e  = (int(p1[0] + dx * t1), int(p1[1] + dy * t1))
            cv2.line(canvas, s, e, color, thickness, cv2.LINE_AA)
    _dashed_line((x1, y1), (x2, y1))
    _dashed_line((x2, y1), (x2, y2))
    _dashed_line((x2, y2), (x1, y2))
    _dashed_line((x1, y2), (x1, y1))
_HUD_LINE_H     = 22
_HUD_MARGIN     = 10
_HUD_WIDTH      = 255
_HUD_ALPHA      = 0.55


# ---------------------------------------------------------------------------
# CameraCalibration
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    """
    Pinhole camera model calibration parameters.

    Coordinate conventions
    ----------------------
    Vehicle Frame (input space for kinematic trajectories):
        X = forward, Y = left, Z = up.  Origin at rear-axle centre.

    Camera Frame (OpenCV convention):
        X = right, Y = down, Z = into the scene.

    The extrinsic transform (vehicle → camera) is:

        P_cam = R_vc @ P_veh + t_vc

    where:
        R_vc  — 3×3 rotation matrix, vehicle frame → camera frame.
        t_vc  — (3,) translation vector expressed in the camera frame.
                Represents the camera origin in vehicle coordinates,
                rotated into the camera frame.

    Intrinsic matrix K (3×3):
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

    Distortion coefficients (OpenCV order):
        [k1, k2, p1, p2, k3]  — pass np.zeros(5) for an ideal camera.
    """

    K: np.ndarray           # shape (3, 3), float64
    R_vc: np.ndarray        # shape (3, 3), float64 — vehicle → camera
    t_vc: np.ndarray        # shape (3,),   float64 — translation in camera frame
    dist_coeffs: np.ndarray # shape (5,),   float64 — [k1, k2, p1, p2, k3]
    image_width:  int = 1920
    image_height: int = 1280

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def default_front(
        cls,
        image_width: int  = 1920,
        image_height: int = 1280,
    ) -> "CameraCalibration":
        """
        Return approximate Waymo-like front camera calibration.

        Intrinsics are estimated from the Waymo Open Dataset segment
        ``segment-10017090168044687777``.  Extrinsics place the camera
        ~1.55 m forward and ~1.48 m above the vehicle-frame origin (rear
        axle), looking straight ahead with a slight downward tilt (~2°).

        Use this for local testing / visualisation when the real proto
        calibration is not available.
        """
        # Approximate intrinsics (Waymo front, 1920×1280)
        fx, fy = 2002.4, 2002.4
        cx, cy = image_width / 2.0, image_height / 2.0
        K = np.array([[fx,  0, cx],
                      [ 0, fy, cy],
                      [ 0,  0,  1]], dtype=np.float64)

        # Rotation: vehicle (X-fwd, Y-left, Z-up) → camera (X-right, Y-down, Z-fwd)
        # Ideal (no tilt):
        #   cam_X =  -veh_Y
        #   cam_Y =  -veh_Z
        #   cam_Z =   veh_X
        # With ~2° downward pitch (rotation around cam_X axis):
        pitch = np.deg2rad(-2.0)       # negative = camera tilts downward
        R_ideal = np.array([
            [ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0],
        ], dtype=np.float64)
        # Pitch around camera X axis
        R_pitch = np.array([
            [1,             0,              0],
            [0,  np.cos(pitch), -np.sin(pitch)],
            [0,  np.sin(pitch),  np.cos(pitch)],
        ], dtype=np.float64)
        R_vc = R_pitch @ R_ideal

        # Camera position in vehicle frame: 1.55 m forward, 1.48 m up
        # t_vc = R_vc @ (-P_cam_in_veh)
        cam_pos_veh = np.array([1.55, 0.0, 1.48], dtype=np.float64)
        t_vc = -R_vc @ cam_pos_veh

        return cls(
            K            = K,
            R_vc         = R_vc,
            t_vc         = t_vc,
            dist_coeffs  = np.zeros(5, dtype=np.float64),
            image_width  = image_width,
            image_height = image_height,
        )

    @classmethod
    def from_waymo_camera(cls, camera_calibration) -> "CameraCalibration":
        """
        Build a ``CameraCalibration`` from a Waymo ``CameraCalibration`` proto.

        Parameters
        ----------
        camera_calibration :
            A ``waymo_open_dataset.dataset_pb2.CameraCalibration`` proto
            object (obtained from ``frame.context.camera_calibrations``).

        Waymo intrinsic layout (9 floats):
            [f_u, f_v, c_u, c_v, k_1, k_2, p_1, p_2, k_3]

        Waymo extrinsic:
            ``camera_calibration.extrinsic.transform`` is a flat row-major
            4×4 matrix that maps points from the **camera** frame to the
            **vehicle** frame (sensor→vehicle).  We invert it to get the
            vehicle→camera transform needed for projection.
        """
        intr = list(camera_calibration.intrinsic)
        K = np.array([[intr[0],     0, intr[2]],
                      [    0, intr[1], intr[3]],
                      [    0,     0,       1  ]], dtype=np.float64)
        dist_coeffs = np.array(intr[4:9], dtype=np.float64)

        # 4×4 sensor→vehicle extrinsic
        T_sensor2veh = np.array(
            camera_calibration.extrinsic.transform, dtype=np.float64
        ).reshape(4, 4)

        # Invert to get vehicle→sensor (Waymo sensor frame: X=fwd, Y=left, Z=up)
        T_veh2sensor = np.linalg.inv(T_sensor2veh)
        R_vs = T_veh2sensor[:3, :3]  # vehicle → Waymo sensor
        t_vs = T_veh2sensor[:3,  3]

        # Waymo sensor frame (X=fwd, Y=left, Z=up) → OpenCV camera frame
        # (X=right, Y=down, Z=into-scene).
        #   opencv_X =  -sensor_Y   (right  = -left)
        #   opencv_Y =  -sensor_Z   (down   = -up)
        #   opencv_Z =   sensor_X   (depth  =  fwd)
        R_s2c = np.array([[ 0, -1,  0],
                           [ 0,  0, -1],
                           [ 1,  0,  0]], dtype=np.float64)
        R_vc = R_s2c @ R_vs
        t_vc = R_s2c @ t_vs

        w = camera_calibration.width
        h = camera_calibration.height

        return cls(
            K            = K,
            R_vc         = R_vc,
            t_vc         = t_vc,
            dist_coeffs  = dist_coeffs,
            image_width  = w,
            image_height = h,
        )


# ---------------------------------------------------------------------------
# PerceptionVisualizer
# ---------------------------------------------------------------------------

class PerceptionVisualizer:
    """
    Draws all perception layers onto a BGR camera frame.

    Parameters
    ----------
    calibration : CameraCalibration
        Camera intrinsics and vehicle→camera extrinsic transform used
        to project 3D Vehicle Frame points onto the image plane.
    trajectory_z_ground : float
        Assumed Z coordinate (metres, Vehicle Frame) of wheel contact
        points with the ground.  Default: 0.0 (flat ground plane).
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        trajectory_z_ground: float = 0.0,
    ) -> None:
        self.cal = calibration
        self._z_ground = trajectory_z_ground

        # Pre-compute rvec / tvec for cv2.projectPoints
        # cv2.projectPoints expects rvec (Rodrigues) and tvec (3,1)
        self._rvec, _ = cv2.Rodrigues(calibration.R_vc)
        self._tvec     = calibration.t_vc.reshape(3, 1)

    # ------------------------------------------------------------------
    # Public: individual layers
    # ------------------------------------------------------------------

    def draw_gt_boxes(
        self,
        canvas: np.ndarray,
        gt_data: dict,
    ) -> np.ndarray:
        """
        Draw 2D ground-truth bounding boxes from the GT JSON onto *canvas*.

        Boxes are drawn with class-specific colors and a short label
        ``<ClassName> <first-8-chars-of-id>``.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image, shape (H, W, 3), uint8.  Must be a writable copy.
        gt_data : dict
            Frame entry from the GT JSON, with key ``"boxes_2d"`` containing
            a list of box dicts:
                {id, type, center_x, center_y, length, width}
            Optionally also contains ``"ego_speed_kmh"`` and ``"timestamp"``.

        Returns
        -------
        np.ndarray
            The same *canvas* with boxes drawn in-place (no copy made here;
            caller owns the copy via ``draw_all`` or their own ``img.copy()``).
        """
        for box in gt_data.get("boxes_2d", []):
            cx, cy = box["center_x"], box["center_y"]
            bw, bh = box["length"],   box["width"]

            x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
            x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

            obj_type   = box.get("type", 0)
            color      = _GT_TYPE_COLORS.get(obj_type, _GT_DEFAULT_COLOR)
            class_name = _GT_TYPE_NAMES.get(obj_type, "Unknown")
            label      = f"{class_name} {box['id'][:8]}"

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas, label,
                (x1, max(y1 - 5, 12)),
                _HUD_FONT, 0.45, color, _HUD_THICKNESS, cv2.LINE_AA,
            )
        return canvas

    def draw_detections_and_tracks(
        self,
        canvas: np.ndarray,
        gt_data: dict,
    ) -> np.ndarray:
        """
        Draw YOLO detections and confirmed Kalman tracks onto *canvas*.

        Two layers:
          1. Raw YOLO detections (``gt_data["detections"]``) — thin magenta
             boxes with a confidence badge.  These are pre-tracking, so they
             include both confirmed and still-tentative objects.
          2. Confirmed Kalman tracks (``gt_data["tracks"]``) — thick
             class-colored boxes.  Coasting tracks (is_coasting=True) are drawn
             with a dashed border so they are distinguishable from measured
             observations.  Each box carries a label:
               T{id}  {conf%}  [{TTC}s]
        """
        # -- Layer 1: raw YOLO detections (thin magenta, conf badge) ----------
        for det in gt_data.get("detections", []):
            bb = det["bbox_xyxy"]
            x1, y1 = int(bb[0]), int(bb[1])
            x2, y2 = int(bb[2]), int(bb[3])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), _DET_RAW_COLOR, 1)
            conf = det.get("confidence", 0.0)
            cv2.putText(
                canvas, f"{conf:.0%}",
                (x1, max(y1 - 4, 12)),
                _HUD_FONT, 0.38, _DET_RAW_COLOR, 1, cv2.LINE_AA,
            )

        # -- Layer 2: confirmed Kalman tracks ---------------------------------
        for trk in gt_data.get("tracks", []):
            bb  = trk["bbox_xyxy"]
            x1, y1 = int(bb[0]), int(bb[1])
            x2, y2 = int(bb[2]), int(bb[3])
            cls_id     = trk.get("class_id", 3)
            is_coasting = trk.get("is_coasting", False)
            color       = _DET_COAST_COLOR if is_coasting else _DET_TRACK_COLORS.get(cls_id, _DET_TRACK_COLORS[3])

            if is_coasting:
                _draw_dashed_rect(canvas, x1, y1, x2, y2, color, 2)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Label: T{id}  {conf%}  [{TTC}s]
            conf = trk.get("confidence", 1.0)
            ttc  = trk.get("ttc_s")
            label = f"T{trk['track_id']}  {conf:.0%}"
            if ttc is not None:
                label += f"  {ttc:.1f}s"

            (tw, th), _ = cv2.getTextSize(label, _HUD_FONT, 0.42, 1)
            lx, ly = x1, max(y1 - 6, 14)
            cv2.rectangle(
                canvas,
                (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                (10, 10, 20), -1,
            )
            cv2.putText(
                canvas, label,
                (lx, ly), _HUD_FONT, 0.42, color, 1, cv2.LINE_AA,
            )
        return canvas

    def draw_kinematic_path(
        self,
        canvas: np.ndarray,
        path_data: dict,
        skip_wheels: bool = False,
    ) -> np.ndarray:
        """
        Project the 3D kinematic wheel trajectories onto the image plane
        and draw them as polylines.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image to draw on (in-place).
        path_data : dict
            Output of ``KinematicPathPredictor.predict()``.
        skip_wheels : bool
            When True only the centre line is drawn (wheel-track boundaries
            are suppressed to avoid visual duplication when HD map or host
            lane already provides boundaries).

        Returns
        -------
        np.ndarray
            canvas with path overlays drawn.
        """
        # Each entry: (path_data key,  draw color,           stroke px,        draw anchor dot)
        # The centre line receives a thicker stroke and an anchor dot at t=0.
        # The L/R boundaries use a thinner, darker color so they are visually
        # subordinate to the centre and clearly distinct from the DP teal bounds.
        if skip_wheels:
            specs = [
                ("centre_line",   _PATH_COLOR_CENTRE, _PATH_LINE_THICK,  True),
            ]
        else:
            specs = [
                ("left_boundary",  _PATH_COLOR_LEFT,   _PATH_BOUND_THICK, False),
                ("right_boundary", _PATH_COLOR_RIGHT,  _PATH_BOUND_THICK, False),
                ("centre_line",    _PATH_COLOR_CENTRE, _PATH_LINE_THICK,  True),
            ]

        for key, color, thickness, draw_dot in specs:
            pts_veh_2d = path_data.get(key)
            if pts_veh_2d is None or len(pts_veh_2d) == 0:
                continue

            # Add ground-plane Z to form 3D Vehicle Frame points.
            z_col  = np.full((len(pts_veh_2d), 1), self._z_ground, dtype=np.float64)
            pts_3d = np.hstack([pts_veh_2d.astype(np.float64), z_col])  # (N, 3)

            # Project while preserving contiguous connectivity.
            # _project_path_segments applies Camera Frame Z ≥ 1.0 m clipping
            # (preventing near-lens distortion) and Vehicle Frame X ≥ 0.1 m
            # clipping (preventing frustum wrapping), then splits the result
            # into separate arrays at every gap so cv2.polylines never draws
            # a line across a clipped discontinuity.
            segments = self._project_path_segments(pts_3d)

            if not segments:
                continue

            for seg in segments:
                # seg is already (K, 1, 2) int32 — ready for cv2.polylines.
                cv2.polylines(canvas, [seg], isClosed=False,
                              color=color, thickness=thickness,
                              lineType=cv2.LINE_AA)

            # Draw a filled anchor dot only for the centre line so the
            # boundary curves do not add visual clutter at their near end.
            if draw_dot:
                anchor = tuple(segments[0][0, 0].tolist())
                cv2.circle(canvas, anchor, _PATH_DOT_RADIUS, color, -1, cv2.LINE_AA)

        return canvas

    def draw_drivable_path(
        self,
        canvas: np.ndarray,
        drivable_data: dict,
    ) -> np.ndarray:
        """
        Draw the Drivable Path center line (Path 3) in teal/cyan.

        The drivable path is always present (confidence may be 0 when
        no markings are detected); it represents the best visual estimate
        of where the vehicle can drive.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image to draw on (in-place).
        drivable_data : dict
            Output of ``VisualPerceptionDetector.detect()`` first element.
            Keys: "center_path" (np.ndarray), "confidence", "source".

        Returns
        -------
        np.ndarray
            canvas with drivable-path overlay drawn.
        """
        center = drivable_data.get("center_path")
        if center is None or len(center) < 2:
            return canvas

        # Draw left and right drivable-area boundaries (thinner, darker cyan)
        for key in ("left_path", "right_path"):
            bnd = drivable_data.get(key)
            if bnd is not None and len(bnd) >= 2:
                cv2.polylines(
                    canvas,
                    [bnd.astype(np.int32).reshape(-1, 1, 2)],
                    isClosed=False,
                    color=_DRIVABLE_BOUND_COLOR,
                    thickness=_DRIVABLE_BOUND_THICK,
                    lineType=cv2.LINE_AA,
                )

        # Draw centre line (thicker)
        pts_cv = center.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            canvas, [pts_cv], isClosed=False,
            color=_DRIVABLE_COLOR, thickness=_DRIVABLE_LINE_THICK,
            lineType=cv2.LINE_AA,
        )
        # Dot at the bottom anchor (vehicle centre)
        cv2.circle(canvas, tuple(center[0].astype(int)),
                   6, _DRIVABLE_COLOR, -1, cv2.LINE_AA)
        return canvas

    def draw_host_lane(
        self,
        canvas: np.ndarray,
        host_lane_data: dict,
    ) -> np.ndarray:
        """
        Draw the Host Lane boundaries (Path 4) in purple/magenta.

        Each side is drawn independently based on ``valid_left`` / ``valid_right``
        so a single detected marking is still visualised even when the other side
        is missing or below the confidence threshold.

        **Debug mode**: Both sides are always drawn, even when they fall below
        the validity threshold.  Invalid sides are rendered in a dim dark-purple
        color (``_HOST_LANE_INVALID_COLOR``) to distinguish them from trusted
        detections.  A confidence badge is always overlaid so the raw
        ``confidence_left`` / ``confidence_right`` scores are readable on every
        frame regardless of the validity flag.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image to draw on (in-place).
        host_lane_data : dict
            Output of ``VisualPerceptionDetector.detect()`` second element.
            Keys: "left_lane", "right_lane", "confidence", "source",
                  "valid_left", "valid_right", "confidence_left",
                  "confidence_right"  (falls back to "valid" for both).

        Returns
        -------
        np.ndarray
            canvas with host-lane overlay drawn (or unchanged if neither side valid).
        """
        valid_left  = host_lane_data.get("valid_left",  host_lane_data.get("valid", False))
        valid_right = host_lane_data.get("valid_right", host_lane_data.get("valid", False))
        conf_l = float(host_lane_data.get("confidence_left",  host_lane_data.get("confidence", 0.0)))
        conf_r = float(host_lane_data.get("confidence_right", host_lane_data.get("confidence", 0.0)))

        left_pts  = host_lane_data.get("left_lane")
        right_pts = host_lane_data.get("right_lane")

        # Draw left lane — full color when valid, dim when not.
        # Invalid lines are still rendered so failed detections are visible.
        if left_pts is not None and len(left_pts) >= 2:
            color_l = _HOST_LANE_COLOR_LEFT if valid_left else _HOST_LANE_INVALID_COLOR
            pts_cv = left_pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_cv], isClosed=False,
                          color=color_l, thickness=_HOST_LANE_LINE_THICK,
                          lineType=cv2.LINE_AA)
            for pt in left_pts[::max(1, len(left_pts) // 8)]:
                cv2.circle(canvas, tuple(pt.astype(int)),
                           4, color_l, -1, cv2.LINE_AA)

        # Draw right lane.
        if right_pts is not None and len(right_pts) >= 2:
            color_r = _HOST_LANE_COLOR_RIGHT if valid_right else _HOST_LANE_INVALID_COLOR
            pts_cv = right_pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_cv], isClosed=False,
                          color=color_r, thickness=_HOST_LANE_LINE_THICK,
                          lineType=cv2.LINE_AA)
            for pt in right_pts[::max(1, len(right_pts) // 8)]:
                cv2.circle(canvas, tuple(pt.astype(int)),
                           4, color_r, -1, cv2.LINE_AA)

        # Shade corridor with purple tint only when both sides are present and valid.
        if (valid_left and valid_right
                and left_pts is not None and len(left_pts) >= 2
                and right_pts is not None and len(right_pts) >= 2):
            poly = np.vstack([
                left_pts.astype(np.int32),
                right_pts[::-1].astype(np.int32),
            ]).copy()
            if poly.shape[0] >= 3:
                overlay = canvas.copy()
                cv2.fillPoly(overlay, [poly], color=(120, 0, 120))
                cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)

        # Always draw the confidence badge so scores are readable on every frame.
        self._draw_host_conf_badge(canvas, conf_l, conf_r, valid_left, valid_right)

        return canvas

    def draw_host_lane_debug_pip(
        self,
        canvas: np.ndarray,
        host_lane_data: dict,
        pip_width:  int = 360,
        pip_height: int = 360,
        margin:     int = 10,
    ) -> np.ndarray:
        """
        Composite a Picture-in-Picture panel showing the raw binary detection
        mask onto *canvas*.

        The mask is sourced from ``host_lane_data["debug_mask"]`` which is
        populated by the detection backends:

        - **IPM**: the BEV threshold output — a top-down 640×640 binary image
          where white pixels are candidate lane-marking pixels before the
          sliding-window stage.  Seeing this mask tells you immediately whether
          the HLS colour filter / Sobel gradient is extracting anything from
          the warped frame.
        - **YOLOPv2**: the ``ll_seg_out`` sigmoid thresholded to source
          resolution — white pixels are positions the network believed to be
          lane lines.

        The PiP is placed in the **bottom-right** corner (away from the
        bottom-left legend) with a cyan border.  Lit-pixel count and source
        name are printed inside the panel.

        If ``debug_mask`` is absent or ``None`` in *host_lane_data*, this
        method is a no-op and returns *canvas* unchanged.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image to draw on (in-place).
        host_lane_data : dict
            Host-lane result dict, expected to contain ``"debug_mask"``.
        pip_width : int
            Width of the PiP panel in pixels on the output canvas.
        pip_height : int
            Height of the PiP panel in pixels on the output canvas.
        margin : int
            Pixel margin from the canvas edges.

        Returns
        -------
        np.ndarray
            canvas with PiP composited in-place.
        """
        raw_mask = host_lane_data.get("debug_mask")
        if raw_mask is None:
            return canvas

        # Normalise to uint8 0/255 regardless of whether the backend stored
        # values as 0/1 (YOLOPv2) or 0/255 (IPM).
        if raw_mask.max() <= 1:
            mask_vis = (raw_mask.astype(np.uint8)) * 255
        else:
            mask_vis = raw_mask.astype(np.uint8)

        # Convert grayscale binary → 3-channel BGR so it composites cleanly.
        mask_bgr = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        # Resize to the requested PiP dimensions using nearest-neighbour so
        # individual binary pixels remain crisp and are not interpolated away.
        pip_frame = cv2.resize(
            mask_bgr, (pip_width, pip_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Compute diagnostic text.
        lit_pix   = int((mask_vis > 0).sum())
        src_name  = host_lane_data.get("source", "?")
        conf_l    = float(host_lane_data.get("confidence_left",  0.0))
        conf_r    = float(host_lane_data.get("confidence_right", 0.0))
        valid     = host_lane_data.get("valid", False)
        status    = "VALID" if valid else "NO DETECT"
        status_color = (0, 220, 80) if valid else (0, 80, 220)   # green / red (BGR)

        # Semi-transparent title bar at top of PiP panel.
        title_h = 18
        overlay = pip_frame.copy()
        cv2.rectangle(overlay, (0, 0), (pip_width, title_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, pip_frame, 0.25, 0, pip_frame)

        cv2.putText(
            pip_frame,
            f"Host Mask [{src_name}]  lit={lit_pix}",
            (4, 13), _LABEL_FONT, 0.40, (0, 220, 220), 1, cv2.LINE_AA,
        )

        # Stats row at bottom of PiP panel.
        stats_y = pip_height - 5
        overlay2 = pip_frame.copy()
        cv2.rectangle(overlay2, (0, pip_height - 20), (pip_width, pip_height),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay2, 0.75, pip_frame, 0.25, 0, pip_frame)

        cv2.putText(
            pip_frame,
            f"L:{conf_l:.3f}  R:{conf_r:.3f}  {status}",
            (4, stats_y), _LABEL_FONT, 0.38, status_color, 1, cv2.LINE_AA,
        )

        # Cyan border around the PiP panel.
        cv2.rectangle(pip_frame, (0, 0), (pip_width - 1, pip_height - 1),
                      (200, 200, 0), 2)

        # Composite onto canvas — bottom-right corner.
        H, W = canvas.shape[:2]
        x0 = W - pip_width  - margin
        y0 = H - pip_height - margin
        # Guard against mask being larger than canvas area (shouldn't happen).
        if x0 >= 0 and y0 >= 0:
            canvas[y0 : y0 + pip_height, x0 : x0 + pip_width] = pip_frame

        return canvas

    def draw_visual_lanes(
        self,
        canvas: np.ndarray,
        lane_data: dict,
    ) -> np.ndarray:
        """
        Draw the ego-lane boundaries detected by VisualLaneDetector.

        The detector returns pixel coordinates directly, so no projection
        is needed.  Each boundary is drawn as a thick polyline.

        For the stub detector (2 points per boundary), a straight line is
        drawn.  When a real model provides denser polylines the same code
        draws a smooth curve.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image to draw on (in-place).
        lane_data : dict
            Output of ``VisualLaneDetector.detect()``, with keys:
                "left_lane"  : np.ndarray (N, 2)  pixel (x, y) coords
                "right_lane" : np.ndarray (N, 2)

        Returns
        -------
        np.ndarray
            canvas with lane overlays drawn.
        """
        lane_specs = []
        source = lane_data.get("source", "")
        if source == "hdmap":
            lane_specs = [
                ("left_lane",  _HDMAP_COLOR_LEFT),
                ("right_lane", _HDMAP_COLOR_RIGHT),
            ]
        else:
            lane_specs = [
                ("left_lane",  _LANE_COLOR_LEFT),
                ("right_lane", _LANE_COLOR_RIGHT),
            ]

        for key, color in lane_specs:
            pts = lane_data.get(key)
            if pts is None or len(pts) < 2:
                continue

            thick = _HDMAP_LINE_THICK if source == "hdmap" else _LANE_LINE_THICK
            pts_cv = pts.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts_cv], isClosed=False,
                          color=color, thickness=thick,
                          lineType=cv2.LINE_AA)

            # Endpoint markers
            for pt in pts:
                cv2.circle(canvas, tuple(pt.astype(int)),
                           5, color, -1, cv2.LINE_AA)

        # Optionally shade the ego-lane region between the two boundaries
        left_pts  = lane_data.get("left_lane")
        right_pts = lane_data.get("right_lane")
        if left_pts is not None and right_pts is not None:
            canvas = self._shade_ego_lane(canvas, left_pts, right_pts)

        return canvas

    def draw_hud(
        self,
        canvas: np.ndarray,
        gt_data: dict,
        frame_idx: int,
    ) -> np.ndarray:
        """
        Draw a semi-transparent heads-up display (HUD) in the top-left corner.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image to draw on (in-place).
        gt_data : dict
            Frame dict; used for ``ego_speed_kmh``, ``timestamp``, ``boxes_2d``.
        frame_idx : int
            Current frame index displayed in the HUD.

        Returns
        -------
        np.ndarray
            canvas with HUD drawn.
        """
        lines = [
            f"Frame:   {frame_idx}",
            f"Speed:   {gt_data.get('ego_speed_kmh', 0):.1f} km/h",
            f"Objects: {len(gt_data.get('boxes_2d', []))}",
            f"TS:      {gt_data.get('timestamp', 0):.3f}",
        ]

        hud_h = _HUD_MARGIN * 2 + len(lines) * _HUD_LINE_H
        x0, y0 = _HUD_MARGIN, _HUD_MARGIN

        # Semi-transparent dark background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0),
                      (x0 + _HUD_WIDTH, y0 + hud_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, _HUD_ALPHA, canvas, 1 - _HUD_ALPHA, 0, canvas)

        for i, line in enumerate(lines):
            text_y = y0 + _HUD_MARGIN + (i + 1) * _HUD_LINE_H - 4
            cv2.putText(
                canvas, line, (x0 + 8, text_y),
                _HUD_FONT, _HUD_FONT_SCALE, (255, 255, 255),
                _HUD_THICKNESS, cv2.LINE_AA,
            )
        return canvas

    # ------------------------------------------------------------------
    # Public: Vehicle EKF track visualisation
    # ------------------------------------------------------------------

    def draw_vehicle_ekf_tracks(
        self,
        canvas:         np.ndarray,
        ekf_tracks:     list[dict],
        lane_relations: list[dict],
    ) -> np.ndarray:
        """
        Draw all confirmed vehicle EKF tracks onto *canvas*.

        Per-track layers
        ----------------
        1. **Bbox outline** — color-coded by lane-relation "side":
               green  = inside drivable / host-lane bounds
               orange = adjacent (left or right, within ~1 lane width)
               red    = outside all bounds (or no relation available)
        2. **Velocity arrow** — 3D velocity vector [vx, vy] projected from the
           track position in Vehicle Frame and drawn as an arrow to the
           predicted position 1 s in the future.
        3. **Info panel** — semi-transparent label below/above the bbox with:
               EKF{id}  {speed:.1f}m/s  [{TTC}s]
               {x:.1f}m fwd  {y:.1f}m lat
               {W:.1f}x{H:.1f}x{L:.1f}m  |  {side} of {best_path}

        Parameters
        ----------
        canvas : np.ndarray
            BGR image to draw on (caller must pass a copy if immutability
            is needed).
        ekf_tracks : list[dict]
            List of VehicleTrackState.to_dict() for this frame
            (from ``gt_data["vehicle_ekf_tracks"]``).
        lane_relations : list[dict]
            List of lane-relation dicts for this frame
            (from ``gt_data["lane_relations"]``).

        Returns
        -------
        np.ndarray
            *canvas* with EKF overlays drawn in-place.
        """
        # Build a lookup from track_id → relation dict for O(1) access
        rel_by_id: dict[int, dict] = {
            r["track_id"]: r for r in lane_relations
        }

        for trk in ekf_tracks:
            tid   = trk["track_id"]
            bb    = trk["bbox_xyxy"]          # [x1,y1,x2,y2] image pixels
            x1, y1, x2, y2 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])

            x_veh = float(trk["x_veh"])
            y_veh = float(trk["y_veh"])
            vx    = float(trk["vx_veh"])
            vy    = float(trk["vy_veh"])
            spd   = float(trk["speed_mps"])
            ttc   = trk.get("ttc_s")          # None when not closing
            w_m   = float(trk["width_m"])
            h_m   = float(trk["height_m"])
            l_m   = float(trk["length_m"])

            # ── 1. Determine lane-relation color ─────────────────────────────
            rel   = rel_by_id.get(tid, {})
            rels  = rel.get("relations", {})

            # Priority: host_lane > drivable_path > kinematic > hdmap
            _priority = ["host_lane", "drivable_path", "kinematic", "hdmap"]
            best_path = "none"
            best_side = "unknown"
            best_rel  = {}
            for p in _priority:
                r = rels.get(p, {})
                if r.get("valid", False):
                    best_path = p
                    best_side = r.get("side", "unknown")
                    best_rel  = r
                    break

            inside = any(
                rels.get(p, {}).get("inside_bounds", False)
                for p in ("host_lane", "drivable_path", "hdmap")
            )
            if inside:
                box_color = (20, 200, 20)    # green — inside ego lane
            elif best_side in ("left", "right"):
                dist_m = abs(best_rel.get("dist_lateral_m", 99.0))
                if dist_m < 4.0:
                    box_color = (0, 160, 255)   # orange — adjacent
                else:
                    box_color = (0, 50, 220)    # red — far outside
            else:
                box_color = (120, 120, 120)  # grey — no valid relation

            # ── 2. Draw the EKF bounding box ─────────────────────────────────
            thickness = 3 if not trk.get("is_coasting", False) else 1
            if trk.get("is_coasting", False):
                _draw_dashed_rect(canvas, x1, y1, x2, y2, box_color, thickness)
            else:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness)

            # Corner tick marks (distinguish from the plain Kalman bbox)
            tick = 10
            for cx_, cy_, dx, dy in [
                (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(canvas, (cx_, cy_),
                         (cx_ + dx * tick, cy_), box_color, 2, cv2.LINE_AA)
                cv2.line(canvas, (cx_, cy_),
                         (cx_, cy_ + dy * tick), box_color, 2, cv2.LINE_AA)

            # ── 3. Velocity arrow (project predicted position 1 s ahead) ────
            pred_x = x_veh + vx * 1.0
            pred_y = y_veh + vy * 1.0
            # Only draw when range is in front of the camera
            if x_veh > 1.0 and pred_x > 1.0:
                origin_pts = self._project_points(
                    np.array([[x_veh, y_veh, 0.75]], dtype=np.float64)
                )
                pred_pts   = self._project_points(
                    np.array([[pred_x, pred_y, 0.75]], dtype=np.float64)
                )
                if len(origin_pts) > 0 and len(pred_pts) > 0:
                    op = tuple(origin_pts[0].astype(int))
                    pp = tuple(pred_pts[0].astype(int))
                    cv2.arrowedLine(
                        canvas, op, pp, box_color, 2,
                        cv2.LINE_AA, tipLength=0.35,
                    )

            # ── 4. Info panel ─────────────────────────────────────────────────
            ttc_str = f"  {ttc:.1f}s" if ttc is not None else ""
            coast   = " [C]" if trk.get("is_coasting", False) else ""
            lines   = [
                f"EKF{tid}{coast}  {spd:.1f}m/s{ttc_str}",
                f"{x_veh:.1f}m  {y_veh:+.1f}m lat",
                f"{w_m:.1f}x{h_m:.1f}x{l_m:.1f}m",
                f"{best_side} / {best_path.replace('_', ' ')}",
            ]
            line_h  = 15
            pad_x   = 4
            panel_w = 185
            panel_h = len(lines) * line_h + pad_x * 2

            # Place panel above the bbox if it fits, else below
            if y1 - panel_h - 4 >= 0:
                panel_y0 = y1 - panel_h - 4
            else:
                panel_y0 = y2 + 4
            panel_x0 = max(0, min(x1, canvas.shape[1] - panel_w))

            overlay = canvas.copy()
            cv2.rectangle(overlay,
                          (panel_x0, panel_y0),
                          (panel_x0 + panel_w, panel_y0 + panel_h),
                          (10, 10, 20), -1)
            cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
            cv2.rectangle(canvas,
                          (panel_x0, panel_y0),
                          (panel_x0 + panel_w, panel_y0 + panel_h),
                          box_color, 1)

            for li, text in enumerate(lines):
                ty = panel_y0 + pad_x + (li + 1) * line_h - 2
                cv2.putText(
                    canvas, text,
                    (panel_x0 + pad_x + 2, ty),
                    _HUD_FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA,
                )

        return canvas

    # ------------------------------------------------------------------
    # Public: convenience all-in-one method
    # ------------------------------------------------------------------

    def draw_all(
        self,
        img: np.ndarray,
        gt_data: dict,
        path_data: dict | None = None,
        hdmap_data: dict | None = None,
        frame_idx: int = 0,
        drivable_data: dict | None = None,
        host_lane_data: dict | None = None,
        enabled_paths: "set[str] | None" = None,
    ) -> np.ndarray:
        """
        Composite all active perception layers onto a fresh copy of *img*.

        Drawing order (bottom -> top):
            1. HD Map lane boundaries        (green,  Path 2)
            2. Host Lane markings            (purple, Path 4, per-side validity)
            3. Drivable Path center + bounds (cyan,   Path 3)
            4. Kinematic Ego Path            (yellow, Path 1)
            5. GT bounding boxes
            6. HUD overlay
            7. Legend

        Parameters
        ----------
        img : np.ndarray
            Raw BGR frame.  Never mutated.
        gt_data : dict
            Ground-truth frame dict.
        path_data : dict | None
            Output of ``KinematicPathPredictor.predict()`` (Path 1).
        hdmap_data : dict | None
            Output of ``project_hdmap_lanes()`` (Path 2).
        frame_idx : int
            Current frame index for the HUD.
        drivable_data : dict | None
            First element from ``VisualPerceptionDetector.detect()`` (Path 3).
        host_lane_data : dict | None
            Second element from ``VisualPerceptionDetector.detect()`` (Path 4).
        enabled_paths : set[str] | None
            Which path types to draw.  Keys: ``"kinematic"``, ``"hdmap"``,
            ``"drivable_path"``, ``"host_lane"``.  ``None`` draws all layers
            (fully backward-compatible default).

        Returns
        -------
        np.ndarray
            New annotated BGR image; the original *img* is unchanged.
        """
        canvas = img.copy()   # ← original is NEVER mutated

        def _show(key: str) -> bool:
            return enabled_paths is None or key in enabled_paths

        skip_wheels = False  # draw all three kinematic curves (centre + L/R boundaries)

        # Layer 1: HD Map (optional — enable via enabled_paths)
        if hdmap_data is not None and _show("hdmap"):
            self.draw_visual_lanes(canvas, hdmap_data)

        # Layer 2: Host Lane markings — per-side validity
        if host_lane_data is not None and _show("host_lane"):
            self.draw_host_lane(canvas, host_lane_data)
            # Composite the raw binary mask as a PiP in the bottom-right corner.
            # This is always rendered regardless of validity so failed frames
            # expose the pixel-extraction output for diagnostics.
            self.draw_host_lane_debug_pip(canvas, host_lane_data)

        # Layer 3: Drivable Path center + L/R bounds
        if drivable_data is not None and _show("drivable_path"):
            self.draw_drivable_path(canvas, drivable_data)

        # Layer 4: Kinematic Ego Path
        if path_data is not None and _show("kinematic"):
            self.draw_kinematic_path(canvas, path_data, skip_wheels=skip_wheels)

        # Layer 5: GT bounding boxes (always drawn)
        self.draw_gt_boxes(canvas, gt_data)

        # Layer 6: HUD
        self.draw_hud(canvas, gt_data, frame_idx)

        # Layer 7: Legend — only for enabled path types
        self._draw_legend(
            canvas,
            hdmap_data      if _show("hdmap")         else None,
            path_data       if _show("kinematic")     else None,
            drivable_data   if _show("drivable_path") else None,
            host_lane_data  if _show("host_lane")     else None,
        )

        return canvas

    def _draw_host_conf_badge(
        self,
        canvas: np.ndarray,
        conf_l: float,
        conf_r: float,
        valid_left: bool,
        valid_right: bool,
    ) -> None:
        """
        Draw a small semi-transparent confidence badge in the top-right corner.

        The badge always shows the raw ``confidence_left`` / ``confidence_right``
        scores and a VALID / INVALID status line.  It is rendered even when
        both sides are below threshold so failed frames are immediately visible
        without inspecting log files.

        Parameters
        ----------
        canvas : np.ndarray
            BGR image, modified in-place.
        conf_l, conf_r : float
            Raw per-side confidence scores from the detection backend.
        valid_left, valid_right : bool
            Per-side validity flags (score above threshold AND line detected).
        """
        H, W = canvas.shape[:2]
        badge_w, badge_h = 250, 66
        x0 = W - badge_w - 10
        y0 = 10

        # Semi-transparent dark background.
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + badge_w, y0 + badge_h),
                      (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        # Title row.
        cv2.putText(canvas, "HOST LANE",
                    (x0 + 8, y0 + 16),
                    _HUD_FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

        # Left confidence — green when valid, red when not.
        l_color = (30, 210, 30) if valid_left  else (40, 40, 210)   # BGR
        r_color = (30, 210, 30) if valid_right else (40, 40, 210)
        cv2.putText(canvas, f"L: {conf_l:.3f}",
                    (x0 + 8, y0 + 38),
                    _HUD_FONT, 0.55, l_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"R: {conf_r:.3f}",
                    (x0 + 130, y0 + 38),
                    _HUD_FONT, 0.55, r_color, 1, cv2.LINE_AA)

        # Overall status line.
        both_valid   = valid_left and valid_right
        status_text  = "VALID" if both_valid else "INVALID"
        status_color = (30, 210, 30) if both_valid else (40, 40, 210)
        cv2.putText(canvas, status_text,
                    (x0 + 8, y0 + 58),
                    _HUD_FONT, 0.45, status_color, 1, cv2.LINE_AA)

    def _draw_legend(
        self,
        canvas: np.ndarray,
        hdmap_data: dict | None,
        path_data: dict | None,
        drivable_data: dict | None = None,
        host_lane_data: dict | None = None,
    ) -> None:
        """
        Draw a colour legend in the bottom-left corner labelling every
        active detection layer.
        """
        entries: list[tuple[tuple[int, int, int], str]] = []

        # Path 2: HD Map
        if hdmap_data is not None:
            if len(hdmap_data.get("left_lane",  np.empty((0, 2)))) >= 2:
                entries.append((_HDMAP_COLOR_LEFT,  "HD Map - Left lane"))
            if len(hdmap_data.get("right_lane", np.empty((0, 2)))) >= 2:
                entries.append((_HDMAP_COLOR_RIGHT, "HD Map - Right lane"))

        # Path 3: Drivable Path
        if drivable_data is not None:
            conf = drivable_data.get("confidence", 0.0)
            label = f"Drivable Path (conf={conf:.2f})"
            entries.append((_DRIVABLE_COLOR, label))

        # Path 4: Host Lane — per-side validity
        if host_lane_data is not None:
            valid_left  = host_lane_data.get("valid_left",  host_lane_data.get("valid", False))
            valid_right = host_lane_data.get("valid_right", host_lane_data.get("valid", False))
            conf_l = host_lane_data.get("confidence_left",  host_lane_data.get("confidence", 0.0))
            conf_r = host_lane_data.get("confidence_right", host_lane_data.get("confidence", 0.0))
            if valid_left:
                entries.append((_HOST_LANE_COLOR_LEFT,  f"Host Lane L  (conf={conf_l:.3f})"))
            else:
                entries.append(((80, 80, 80), f"Host Lane L  invalid (conf={conf_l:.3f})"))
            if valid_right:
                entries.append((_HOST_LANE_COLOR_RIGHT, f"Host Lane R  (conf={conf_r:.3f})"))
            else:
                entries.append(((80, 80, 80), f"Host Lane R  invalid (conf={conf_r:.3f})"))

        # Path 1: Kinematic Ego Path
        if path_data is not None:
            entries.append((_PATH_COLOR_CENTRE, "Kinematic - Ego centre"))
            entries.append((_PATH_COLOR_LEFT,   "Kinematic - L/R boundary"))

        if not entries:
            return

        H = canvas.shape[0]
        pad, row_h, swatch_w = 8, 20, 14
        legend_h = pad * 2 + len(entries) * row_h
        x0 = pad
        y0 = H - legend_h - pad

        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0),
                      (x0 + 260, y0 + legend_h), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, (color, label) in enumerate(entries):
            row_y = y0 + pad + i * row_h
            cv2.rectangle(canvas,
                          (x0 + 4, row_y + 2),
                          (x0 + 4 + swatch_w, row_y + row_h - 4),
                          color, -1)
            cv2.putText(canvas, label,
                        (x0 + 4 + swatch_w + 6, row_y + row_h - 5),
                        _LABEL_FONT, _LABEL_FONT_SCALE,
                        (255, 255, 255), _LABEL_THICKNESS, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _project_points(self, pts_3d: np.ndarray) -> np.ndarray:
        """
        Project an (N, 3) array of Vehicle Frame 3D points to pixel coords.

        Steps:
            1. Clip points behind the vehicle (Vehicle Frame X < 0.1 m) to
               prevent frustum-wrapping: when X is negative the point is
               physically behind the ego vehicle and will always project to
               an implausible screen location.
            2. Filter out points behind the camera plane (Z_cam ≤ 0.01).
            3. Call cv2.projectPoints with the pre-computed rvec/tvec/K/dist.
            4. Clip resulting pixels to the image bounds.
            5. Return only the pixels that fall within the image.

        Note: this method discards filtered points silently and returns a
        compact (M, 2) array.  Callers that need to preserve point-to-point
        connectivity (polylines) should use ``_project_path_segments``
        instead, which returns separately-connected segments.

        Parameters
        ----------
        pts_3d : np.ndarray, shape (N, 3)
            Points in Vehicle Frame [X, Y, Z], float64.

        Returns
        -------
        np.ndarray, shape (M, 2)
            Visible, in-bounds integer pixel coordinates (x, y).
            M ≤ N.
        """
        if len(pts_3d) == 0:
            return np.empty((0, 2), dtype=np.float64)

        # --- Step 1: Vehicle-Frame X clip ---
        # Any point with X < 0.1 m lies at or behind the vehicle origin.
        # Projecting such points via the pinhole model causes the homogeneous
        # divide to flip sign, mapping them to the opposite side of the screen
        # (the classic "frustum wrapping" artefact).  Reject them first,
        # before any camera-space transformation.
        veh_x_ok = pts_3d[:, 0] >= 0.1
        pts_3d   = pts_3d[veh_x_ok]
        if len(pts_3d) == 0:
            return np.empty((0, 2), dtype=np.float64)

        # --- Step 2: pre-filter points behind camera ---
        # Transform to camera frame manually for the depth check
        pts_cam = (self.cal.R_vc @ pts_3d.T).T + self.cal.t_vc  # (N, 3)
        visible_mask = pts_cam[:, 2] > 0.01   # Z_cam > 0 = in front of lens

        pts_visible = pts_3d[visible_mask]
        if len(pts_visible) == 0:
            return np.empty((0, 2), dtype=np.float64)

        # --- Step 3: project with OpenCV (handles distortion) ---
        img_pts, _ = cv2.projectPoints(
            pts_visible.astype(np.float64),
            self._rvec,
            self._tvec,
            self.cal.K,
            self.cal.dist_coeffs,
        )
        px = img_pts.reshape(-1, 2)   # (M, 2) float

        # --- Step 4: clip to image bounds ---
        w, h = self.cal.image_width, self.cal.image_height
        in_bounds = (
            (px[:, 0] >= 0) & (px[:, 0] < w) &
            (px[:, 1] >= 0) & (px[:, 1] < h)
        )
        return px[in_bounds]

    def _project_path_segments(
        self,
        pts_3d: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Project an ordered (N, 3) Vehicle Frame trajectory to a list of
        contiguous 2-D pixel segments, splitting the polyline wherever a
        point is clipped (behind-vehicle, behind-camera, or out-of-bounds).

        Motivation
        ----------
        ``_project_points`` compacts the surviving pixels into a single flat
        array.  Passing that array directly to ``cv2.polylines`` re-connects
        the gap left by clipped points, producing spurious lines that jump
        across the image.  This method instead returns *separate* arrays for
        each run of consecutive valid points so ``cv2.polylines`` only ever
        draws truly neighbouring point pairs.

        Algorithm
        ---------
        For each input point, compute a boolean "valid" flag applying the
        same two-stage filter as ``_project_points``:
            1. Vehicle Frame X ≥ 0.1 m  (rejects behind-vehicle points).
            2. Camera Frame  Z ≥ 1.0 m  (rejects near/behind-lens points).
            3. Projected pixel falls within image bounds.
        Walk the validity array and collect runs of consecutive True values.
        Each run is projected (as a batch) and returned as one segment array.

        Why Camera Frame Z ≥ 1.0 m?
        The front camera is mounted roughly 1.55 m forward of the vehicle-
        frame origin (rear axle).  The CTR arc starts at [0, 0, 0] in the
        Vehicle Frame, so the first ~1.5 m of the trajectory transforms to a
        Camera Frame Z between −0.5 m and +0.5 m (directly behind or under
        the lens).  Passing those near-zero-Z points to cv2.projectPoints
        causes the pinhole division (u = fx * X_cam / Z_cam) to blow up,
        projecting them to extreme pixel coordinates far off-screen or, worse,
        to the opposite side of the image (frustum wrapping).  A 1.0 m
        Camera-Z floor guarantees that only points well in front of the lens
        enter the projection pipeline.

        Parameters
        ----------
        pts_3d : np.ndarray, shape (N, 3)
            Ordered Vehicle Frame points [X, Y, Z], float64.

        Returns
        -------
        list[np.ndarray]
            Each element has shape (K, 1, 2) int32 — ready for
            ``cv2.polylines([segment], ...)`` — where K ≥ 2.
        """
        if len(pts_3d) == 0:
            return []

        # --- Build per-point validity mask ---
        # Stage 1: Vehicle Frame X clip.
        veh_x_ok = pts_3d[:, 0] >= 0.1

        # Stage 2: Camera Frame Z clip.
        # The camera is mounted ~1.55 m forward of the vehicle-frame origin.
        # Points on the CTR arc that have Camera Frame Z < 1.0 m are either
        # physically behind the lens or so close to it that the pinhole
        # division produces extreme, unreliable pixel coordinates.  A 1.0 m
        # hard floor ensures only well-in-front-of-lens points are projected.
        pts_cam  = (self.cal.R_vc @ pts_3d.T).T + self.cal.t_vc   # (N, 3)
        cam_z_ok = pts_cam[:, 2] > 1.0

        # Tentatively project ALL points (invalid ones will be discarded by
        # the bounds check; projecting everything in one call is faster than
        # slicing and re-calling per segment).
        img_pts_all, _ = cv2.projectPoints(
            pts_3d.astype(np.float64),
            self._rvec,
            self._tvec,
            self.cal.K,
            self.cal.dist_coeffs,
        )
        px_all = img_pts_all.reshape(-1, 2)   # (N, 2)

        # Stage 3: image-bounds check.
        W, H = self.cal.image_width, self.cal.image_height
        in_bounds = (
            (px_all[:, 0] >= 0) & (px_all[:, 0] < W) &
            (px_all[:, 1] >= 0) & (px_all[:, 1] < H)
        )

        valid = veh_x_ok & cam_z_ok & in_bounds   # (N,) bool

        # --- Split into contiguous segments ---
        segments: list[np.ndarray] = []
        run: list[np.ndarray] = []

        for i, ok in enumerate(valid):
            if ok:
                run.append(px_all[i])
            else:
                if len(run) >= 2:
                    arr = np.array(run, dtype=np.float64)
                    segments.append(arr.reshape(-1, 1, 2).astype(np.int32))
                run = []

        # Flush the final run.
        if len(run) >= 2:
            arr = np.array(run, dtype=np.float64)
            segments.append(arr.reshape(-1, 1, 2).astype(np.int32))

        return segments

    @staticmethod
    def _shade_ego_lane(
        canvas: np.ndarray,
        left_pts:  np.ndarray,
        right_pts: np.ndarray,
        alpha: float = 0.15,
    ) -> np.ndarray:
        """
        Fill the region between left and right lane boundaries with a
        semi-transparent blue tint.

        The polygon is formed by:  left_pts (top→bottom) + right_pts[::-1] (bottom→top).

        Parameters
        ----------
        canvas : np.ndarray
            BGR image (modified in-place).
        left_pts : np.ndarray, shape (N, 2)
        right_pts : np.ndarray, shape (N, 2)
        alpha : float
            Blend factor for the fill (0 = invisible, 1 = opaque).

        Returns
        -------
        np.ndarray
            canvas with shaded ego-lane region.
        """
        poly = np.vstack([
            left_pts.astype(np.int32),
            right_pts[::-1].astype(np.int32),
        ]).copy()

        if poly.shape[0] < 3:
            return canvas

        overlay = canvas.copy()
        cv2.fillPoly(overlay, [poly], color=(150, 50, 0))   # dark blue fill
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
        return canvas
