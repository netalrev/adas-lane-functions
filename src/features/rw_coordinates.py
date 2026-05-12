"""
src/features/rw_coordinates.py
================================
Projects image-space bounding boxes onto the vehicle-frame ground plane
using the pinhole camera model.

Public API
----------
    project_bbox_to_ground(bbox_xyxy, calib) -> tuple[float, float] | None
        Back-project the bottom-centre pixel of a bounding box onto the
        Z_veh = 0 plane.  Returns (range_m, lateral_m) or None if the
        ray is geometrically invalid (parallel to ground or behind camera).

Design
------
The pipeline uses bottom-centre projection because the feet / lowest contact
point of a vehicle or pedestrian is the most stable ground-plane contact
approximation available from a 2D bounding box.

Coordinate conventions (consistent with the rest of the codebase):
    Vehicle Frame: X = forward, Y = left, Z = up.  Origin at rear axle.
    Camera Frame:  X = right, Y = down, Z = into the scene.

Math
----
Given pixel (u, v):
    1. Back-project to camera-frame ray:
           d_cam = K⁻¹ · [u, v, 1]ᵀ
    2. Rotate to vehicle frame:
           d_veh = R_vcᵀ · d_cam        (R_vc maps vehicle→camera)
    3. Camera origin in vehicle frame:
           O_veh = −R_vcᵀ · t_vc
    4. Intersect ray O_veh + t·d_veh with the plane Z_veh = 0:
           t = −O_veh[Z] / d_veh[Z]
    5. Ground point:
           P_veh = O_veh + t · d_veh
           range   = P_veh[X]   (forward distance)
           lateral = P_veh[Y]   (left positive)
"""

from __future__ import annotations

import numpy as np


def project_bbox_to_ground(
    bbox_xyxy: np.ndarray,
    K: np.ndarray,
    R_vc: np.ndarray,
    t_vc: np.ndarray,
) -> tuple[float, float] | None:
    """
    Project the bottom-centre of a bounding box to the vehicle-frame ground
    plane (Z_veh = 0).

    Parameters
    ----------
    bbox_xyxy : array-like, shape (4,)
        Bounding box [x1, y1, x2, y2] in image pixels.
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.
    R_vc : np.ndarray, shape (3, 3)
        Rotation from vehicle frame to camera frame.
    t_vc : np.ndarray, shape (3,)
        Translation (camera origin in vehicle coordinates, rotated to camera
        frame).  Satisfies: P_cam = R_vc @ P_veh + t_vc.

    Returns
    -------
    tuple[float, float] | None
        (range_m, lateral_m) in vehicle frame, or None when the projection
        is geometrically invalid (ray parallel to ground, or intersection
        is behind the camera).
    """
    u = float((bbox_xyxy[0] + bbox_xyxy[2]) / 2.0)  # horizontal centre
    v = float(bbox_xyxy[3])                           # bottom edge (ground contact)
    return _pixel_to_ground(u, v, K, R_vc, t_vc)


def _pixel_to_ground(
    u: float,
    v: float,
    K: np.ndarray,
    R_vc: np.ndarray,
    t_vc: np.ndarray,
) -> tuple[float, float] | None:
    """
    Core ray-casting implementation.  See module docstring for the math.
    """
    K_inv = np.linalg.inv(K)
    R_cv  = R_vc.T   # camera frame → vehicle frame

    # 1. Camera-frame ray direction (un-normalised)
    d_cam = K_inv @ np.array([u, v, 1.0], dtype=np.float64)

    # 2. Ray direction in vehicle frame
    d_veh = R_cv @ d_cam

    # 3. Camera origin in vehicle frame
    O_veh = -R_cv @ t_vc

    # 4. Intersect with Z_veh = 0 ground plane
    if abs(d_veh[2]) < 1e-6:
        return None  # ray is nearly parallel to the ground — no valid intersection

    t = -O_veh[2] / d_veh[2]
    if t <= 0.0:
        return None  # intersection is behind the camera

    P_veh = O_veh + t * d_veh
    return float(P_veh[0]), float(P_veh[1])  # (range, lateral)
