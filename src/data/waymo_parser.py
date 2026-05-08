import tensorflow as tf
import numpy as np
import cv2
from waymo_open_dataset import dataset_pb2 as open_dataset

def extract_front_camera_image(frame):
    """Extracts and decodes the front camera image from a Waymo frame."""
    for camera in frame.images:
        if camera.name == open_dataset.CameraName.FRONT:
            img = tf.image.decode_jpeg(camera.image).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
    return None

def calculate_ego_speed(frame, prev_pos, prev_time):
    """Calculates ego vehicle speed using pose displacement over time."""
    mat = frame.pose.transform  
    curr_x, curr_y = mat[3], mat[7]  
    curr_time = frame.timestamp_micros / 1e6
    
    if prev_pos is None:
        return 0.0, (curr_x, curr_y), curr_time
        
    dt = curr_time - prev_time
    if dt <= 0:
        return 0.0, (curr_x, curr_y), curr_time
        
    speed_mps = np.sqrt((curr_x - prev_pos[0])**2 + (curr_y - prev_pos[1])**2) / dt
    return speed_mps * 3.6, (curr_x, curr_y), curr_time

def extract_ground_truth_boxes(frame): 
    """Extracts 2D bounding boxes from the front camera labels for evaluation."""
    gt_data = {
        "timestamp": frame.timestamp_micros / 1e6,
        "boxes_2d": []
    }

    for camera_labels in frame.camera_labels:
        if camera_labels.name == open_dataset.CameraName.FRONT:
            for label in camera_labels.labels:
                box = label.box
                gt_data["boxes_2d"].append({
                    "id": label.id,
                    "type": label.type, # 1: Vehicle, 2: Pedestrian, 4: Cyclist
                    "center_x": box.center_x,
                    "center_y": box.center_y,
                    "length": box.length,
                    "width": box.width
                })
    return gt_data


def extract_gt_3d_boxes(frame) -> list:
    """
    Extract 3D bounding boxes from Waymo laser_labels (vehicle frame).

    Laser labels contain the authoritative 3D ground-truth boxes used for
    GT derivation (CIPV, Lane Assignment).  All coordinates are in the
    Vehicle Frame: X = forward, Y = left, Z = up.

    Parameters
    ----------
    frame : waymo_open_dataset.dataset_pb2.Frame
        A single parsed Waymo frame proto.

    Returns
    -------
    list[dict]
        One dict per object:
            id              : str   — persistent object ID across frames
            type            : int   — 1=vehicle, 2=pedestrian, 3=sign, 4=cyclist
            center_x        : float — range (m, forward positive)
            center_y        : float — lateral (m, left positive)
            center_z        : float — height above ground (m)
            length          : float — extent along X (m)
            width           : float — extent along Y (m)
            height          : float — extent along Z (m)
            heading         : float — yaw relative to X-axis (rad)
            num_lidar_points: int   — lidar points in box (quality indicator)
    """
    boxes = []
    for label in frame.laser_labels:
        box = label.box
        boxes.append({
            "id":               label.id,
            "type":             int(label.type),
            "center_x":         float(box.center_x),
            "center_y":         float(box.center_y),
            "center_z":         float(box.center_z),
            "length":           float(box.length),
            "width":            float(box.width),
            "height":           float(box.height),
            "heading":          float(box.heading),
            "num_lidar_points": int(label.num_lidar_points_in_box),
        })
    return boxes


def parse_map_features_global(frame):
    """
    Extract road_line and road_edge polylines from Waymo map_features in global
    (GPS) coordinates.  map_features are only populated in the first frame of
    each segment — call this once and cache the result.

    Returns
    -------
    list[dict]
        Each entry: {"kind": "road_line" | "road_edge", "pts": np.ndarray(N, 3)}
        where pts columns are [X_global, Y_global, Z_global].
    """
    features = []
    for feat in frame.map_features:
        if feat.HasField('road_line'):
            polyline = feat.road_line.polyline
            kind = "road_line"
        elif feat.HasField('road_edge'):
            polyline = feat.road_edge.polyline
            kind = "road_edge"
        else:
            continue
        pts = np.array([[p.x, p.y, p.z] for p in polyline], dtype=np.float64)
        if len(pts) >= 2:
            features.append({"kind": kind, "pts": pts})
    return features


def project_hdmap_lanes(global_polylines, frame, calib, ego_center_veh=None):
    """
    Transform HD map polylines to the current Vehicle Frame using the ego pose
    and project onto the front camera image.

    Coordinate pipeline
    -------------------
    Global (GPS)  →[T_g2v = inv(frame.pose.transform)]→  Vehicle Frame
                  →[calib.R_vc, t_vc, K]→  Image pixels

    Ego-lane selection
    ------------------
    Vehicle frame: X = forward, Y = left (+), Z = up.
    Left  boundary = road_line/edge with smallest positive median Y.
    Right boundary = road_line/edge with smallest negative median Y.
    Falls back from road_line to road_edge when a side has no painted marking.

    Parameters
    ----------
    global_polylines : list[dict]
        Cached output of parse_map_features_global().
    frame : waymo_open_dataset.dataset_pb2.Frame
        Current frame (only frame.pose.transform is used).
    calib : CameraCalibration
        Real front-camera calibration (from CameraCalibration.from_waymo_camera).
    ego_center_veh : np.ndarray | None, shape (N, 2)
        Optional kinematic ego center path in Vehicle Frame [X, Y] from
        KinematicPathPredictor.  When provided, lateral offsets are computed
        relative to the predicted ego path instead of the vehicle centre-line
        (Y = 0).  This is more robust in curves and off-centre lane positions.

    Returns
    -------
    dict | None
        Keys: "left_lane", "right_lane", "source".  Compatible with
        VisualLaneDetector.detect() output format.  None if nothing projects.
    """
    if not global_polylines:
        return None

    # ── Global → Vehicle Frame ────────────────────────────────────────────────
    # frame.pose.transform is a row-major 4×4 that maps Vehicle → Global;
    # invert it to get Global → Vehicle.
    T_v2g = np.array(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    T_g2v = np.linalg.inv(T_v2g)
    R_g2v = T_g2v[:3, :3]
    t_g2v = T_g2v[:3, 3]

    # ── Transform every polyline → Vehicle Frame ─────────────────────────────
    # Candidate search window: only features ahead of ego
    X_MIN, X_MAX = 0.0, 60.0    # strictly forward
    Y_ABS_MIN    = 1.0           # must be outside the car body (half-width ~1.05 m)
    Y_ABS_MAX    = 5.0           # allow up to ~1.4 lane widths laterally (was 7 m)
    # Side-consistency: reject polylines whose points cross significantly into
    # the opposite side (junction outlines, intersection kerbs, wraparound edges)
    Y_CROSS_MAX  = 1.5

    road_line_cands = []   # (abs_lateral_offset_from_ego, signed_offset, pts_veh)
    road_edge_cands = []

    # Pre-extract ego path arrays for fast per-polyline interpolation
    _ego_xs = _ego_ys = None
    if ego_center_veh is not None and len(ego_center_veh) >= 2:
        _ego_xs = ego_center_veh[:, 0].astype(np.float64)
        _ego_ys = ego_center_veh[:, 1].astype(np.float64)

    for feat in global_polylines:
        pts_v = (R_g2v @ feat["pts"].T).T + t_g2v         # (N, 3) vehicle
        front = (pts_v[:, 0] >= X_MIN) & (pts_v[:, 0] <= X_MAX)
        if front.sum() < 2:
            continue
        front_y = pts_v[front, 1]
        # Compute lateral offset relative to the kinematic ego path when available.
        # This correctly handles curves and off-centre lane positions:
        #   med_y > 0  → polyline is to the LEFT  of the ego path
        #   med_y < 0  → polyline is to the RIGHT of the ego path
        if _ego_xs is not None:
            front_x = pts_v[front, 0]
            x_lo, x_hi = float(_ego_xs.min()), float(_ego_xs.max())
            overlap = (front_x >= x_lo) & (front_x <= x_hi)
            if overlap.sum() >= 2:
                ego_y_interp = np.interp(front_x[overlap], _ego_xs, _ego_ys)
                rel_offsets  = front_y[overlap] - ego_y_interp
                med_y        = float(np.median(rel_offsets))
            else:
                med_y = float(np.median(front_y))   # no overlap → absolute Y
        else:
            med_y = float(np.median(front_y))       # fallback: absolute Y
        abs_med = abs(med_y)
        if abs_med < Y_ABS_MIN or abs_med > Y_ABS_MAX:
            continue
        # Reject features that stray significantly onto the wrong side of ego
        if med_y > 0 and float(front_y.min()) < -Y_CROSS_MAX:
            continue
        if med_y < 0 and float(front_y.max()) >  Y_CROSS_MAX:
            continue

        pts_fwd = pts_v[front]   # forward-filtered 3-D points reused below

        # ── Filter 1: Arc length ─────────────────────────────────────────────
        # Reject very short stubs — stop lines, zebra markings, intersection
        # kerbs that happen to fall in the lateral corridor but are not lane
        # boundaries.  Require at least 8 m of continuous road marking ahead.
        arc_len = float(np.linalg.norm(np.diff(pts_fwd[:, :2], axis=0), axis=1).sum())
        if arc_len < 8.0:
            continue

        # ── Filter 2: Heading alignment ──────────────────────────────────────
        # Lane boundaries must run roughly parallel to the vehicle forward
        # direction (X-axis in vehicle frame).  Crossing-street polylines and
        # perpendicular road edges are rejected here.
        # Use SVD on the 2-D XY projection of forward-filtered points.
        if len(pts_fwd) >= 4:
            xy_c = pts_fwd[:, :2] - pts_fwd[:, :2].mean(0)
            _, _, Vt = np.linalg.svd(xy_c, full_matrices=False)
            principal = Vt[0]   # principal direction unit vector
            angle_deg = abs(np.degrees(np.arctan2(principal[1], principal[0])))
            # Accept only polylines running within ±25° of the X-axis.
            # angle_deg is in [0°, 180°]; forward-parallel = 0° or 180°.
            # Tightened from ±35° to ±25° to reject diagonal intersection
            # turn-lane features that pass the looser threshold at complex junctions.
            if 25.0 < angle_deg < 155.0:
                continue   # perpendicular / crossing road feature → reject

        # ── Filter 3: Proximity to ego path ─────────────────────────────────
        # Require the candidate polyline to physically pass close to the ego
        # predicted path.  This kills parallel roads one or two lanes over
        # that accidentally have the right median lateral offset.
        if _ego_xs is not None:
            ego_path_xy = np.column_stack([_ego_xs, _ego_ys])   # (M, 2)
            poly_xy     = pts_fwd[:, :2]                          # (N, 2)
            # Vectorised min distance: (M, N) → scalar
            diff_mat  = poly_xy[np.newaxis, :, :] - ego_path_xy[:, np.newaxis, :]
            min_dist  = float(np.linalg.norm(diff_mat, axis=2).min())
            if min_dist > 3.5:   # more than ~1 lane-width at closest point (was 4.5 m)
                continue

        entry = (abs_med, med_y, pts_v)
        if feat["kind"] == "road_line":
            road_line_cands.append(entry)
        else:
            road_edge_cands.append(entry)

    def pick_sides(cands):
        """Return (left_pts_veh, right_pts_veh) — closest on each side."""
        left  = [(d, p) for d, y, p in cands if y >  0]
        right = [(d, p) for d, y, p in cands if y <= 0]
        lp = min(left,  key=lambda x: x[0])[1] if left  else None
        rp = min(right, key=lambda x: x[0])[1] if right else None
        return lp, rp

    left_veh,  right_veh  = pick_sides(road_line_cands)
    left_edge, right_edge = pick_sides(road_edge_cands)
    if left_veh  is None: left_veh  = left_edge
    if right_veh is None: right_veh = right_edge

    if left_veh is None and right_veh is None:
        return None

    # ── Project Vehicle Frame → Image Pixels ─────────────────────────────────
    # NOTE: calib.R_vc already encodes vehicle → OpenCV camera frame
    # (X=right, Y=down, Z=forward) after CameraCalibration.from_waymo_camera.
    rvec, _ = cv2.Rodrigues(calib.R_vc)
    tvec    = calib.t_vc.reshape(3, 1)
    W, H    = calib.image_width, calib.image_height

    def project(pts_v, y_sign: int):
        """
        Project vehicle-frame polyline to image pixels.

        y_sign: +1 for the left boundary  (keep Y > 0 portion only)
                -1 for the right boundary (keep Y < 0 portion only)

        Filtering rules applied before OpenCV projection:
          • X_veh ∈ [0.5, 70]  — forward-only, max 70 m look-ahead
          • Y_veh sign matches the selected side
          • |Y_veh| ∈ [0.3, 8]  — lateral lane corridor (0.3–8 m from centre)
        """
        if pts_v is None or len(pts_v) < 2:
            return np.empty((0, 2), dtype=np.int32)
        # Lateral corridor on the correct side
        y_col = pts_v[:, 1]
        lat_mask = (y_col * y_sign > 0.3) & (np.abs(y_col) < 8.0)
        # Minimum forward distance 3 m: very near points project to extreme
        # pixel positions and create distracting lines at the image edges.
        fwd_mask = (pts_v[:, 0] > 3.0) & (pts_v[:, 0] < 70.0)
        mask = lat_mask & fwd_mask
        if mask.sum() < 2:
            return np.empty((0, 2), dtype=np.int32)
        pts_fwd = pts_v[mask]
        # ── Sort by vehicle-frame X (forward depth, near→far) ────────────────
        # The HD map polyline is ordered along the road which may meander,
        # backtrack, or wrap around intersection geometry.  Sorting by the
        # forward depth axis (X) guarantees the projected 2-D sequence is
        # monotone bottom-to-top in the image so cv2.polylines draws a smooth
        # continuous arc instead of a zigzag / fan pattern.
        depth_order = np.argsort(pts_fwd[:, 0])
        pts_fwd = pts_fwd[depth_order]
        pts_c = (calib.R_vc @ pts_fwd.T).T + calib.t_vc   # OpenCV camera frame
        vis   = pts_c[:, 2] > 0.05                          # Z = depth into scene
        if vis.sum() < 2:
            return np.empty((0, 2), dtype=np.int32)
        pxy, _ = cv2.projectPoints(
            pts_fwd[vis].astype(np.float64), rvec, tvec,
            calib.K, calib.dist_coeffs,
        )
        px = pxy.reshape(-1, 2)
        ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
        valid = px[ok].astype(np.int32)
        # Sanity: left boundary pixels should be on the left image half;
        # right boundary pixels on the right image half.  Allow a 15 % margin
        # past the centre so lane lines near the horizon aren't clipped.
        cx_limit = int(W * 0.65) if y_sign > 0 else int(W * 0.35)
        side_ok = (valid[:, 0] <= cx_limit) if y_sign > 0 else (valid[:, 0] >= cx_limit)
        valid = valid[side_ok]
        if len(valid) < 2:
            return np.empty((0, 2), dtype=np.int32)
        # ── Final sort by image-y descending (bottom = near field first) ─────
        # After the ok/side_ok masks may have removed interior points, re-sort
        # to restore monotone bottom-to-top ordering for clean polyline drawing.
        valid = valid[np.argsort(-valid[:, 1])]
        return valid

    left_img  = project(left_veh,  y_sign=+1)
    right_img = project(right_veh, y_sign=-1)
    _empty    = np.empty((0, 2), dtype=np.int32)
    return {
        "left_lane":  left_img  if len(left_img)  >= 2 else _empty,
        "right_lane": right_img if len(right_img) >= 2 else _empty,
        "source":     "hdmap",
    }