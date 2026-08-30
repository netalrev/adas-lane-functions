"""Unit tests for the bbox -> ground-plane back-projection math.

Correctness is checked by round-tripping through an independent forward
pinhole-projection reference (not the code under test), plus hand-derived
edge cases for the horizon and behind-camera rejection branches.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.measurements.rw_coordinates import project_bbox_to_ground
from src.visualization.visualizer import CameraCalibration


def _project_point_to_pixel(p_veh, K, R_vc, t_vc):
    """Reference forward pinhole projection (independent of the code under test)."""
    p_cam = R_vc @ np.asarray(p_veh, dtype=np.float64) + t_vc
    assert p_cam[2] > 0, "test point must be in front of the camera"
    uvw = K @ p_cam
    return uvw[0] / uvw[2], uvw[1] / uvw[2]


@pytest.mark.parametrize("range_m, lateral_m", [
    (10.0, 0.0),
    (20.0, 3.0),
    (50.0, -2.0),
    (5.0, 1.0),
    (100.0, -8.0),
])
def test_roundtrip_recovers_known_ground_point(range_m, lateral_m):
    calib = CameraCalibration.default_front()
    u, v = _project_point_to_pixel([range_m, lateral_m, 0.0], calib.K, calib.R_vc, calib.t_vc)
    bbox = np.array([u - 5.0, v - 20.0, u + 5.0, v])  # bottom edge at v

    result = project_bbox_to_ground(bbox, calib.K, calib.R_vc, calib.t_vc)

    assert result is not None
    got_range, got_lateral = result
    assert got_range   == pytest.approx(range_m,   abs=1e-6)
    assert got_lateral == pytest.approx(lateral_m, abs=1e-6)


def test_returns_none_at_the_horizon():
    # No-pitch camera: at v == cy the ray is exactly parallel to the ground.
    K    = np.array([[1000.0, 0, 960.0], [0, 1000.0, 640.0], [0, 0, 1.0]])
    R_vc = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t_vc = -R_vc @ np.array([1.5, 0.0, 1.5])

    bbox = np.array([950.0, 640.0, 970.0, 640.0])  # bottom edge at v == cy
    assert project_bbox_to_ground(bbox, K, R_vc, t_vc) is None


def test_returns_none_when_intersection_is_behind_camera():
    # Same no-pitch camera; v < cy looks above the horizon (sky), so the
    # ground-plane intersection is behind the camera (t <= 0).
    K    = np.array([[1000.0, 0, 960.0], [0, 1000.0, 640.0], [0, 0, 1.0]])
    R_vc = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    t_vc = -R_vc @ np.array([1.5, 0.0, 1.5])

    bbox = np.array([950.0, 400.0, 970.0, 400.0])  # v < cy -> above horizon
    assert project_bbox_to_ground(bbox, K, R_vc, t_vc) is None
