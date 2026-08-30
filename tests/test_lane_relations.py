"""Unit tests for the pure geometry helpers in LaneRelationMeasurer.

These functions have no instance state (see the module's own "Pure
functions -- testable in isolation" section) so every case here uses
hand-derived expected values for simple, exactly-computable polylines.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.measurements.lane_relations import (
    _lateral_offset, _pixel_dist, _check_inside, _side, LaneRelationMeasurer,
)
from src.visualization.visualizer import CameraCalibration

STRAIGHT_AHEAD = np.array([[0.0, 0.0], [10.0, 0.0]])  # path running due forward


def test_lateral_offset_sign_convention():
    # Point 2 m LEFT of the path centreline, 5 m along it.
    signed, dist = _lateral_offset(5.0, 2.0, STRAIGHT_AHEAD)
    assert signed == pytest.approx(2.0)
    assert dist   == pytest.approx(2.0)

    # Point 3 m RIGHT of the path.
    signed, dist = _lateral_offset(5.0, -3.0, STRAIGHT_AHEAD)
    assert signed == pytest.approx(-3.0)
    assert dist   == pytest.approx(3.0)


def test_lateral_offset_clamps_beyond_segment_end():
    # Query point is beyond the last vertex (x=15 > path end x=10). The true
    # nearest-point distance clamps to the final vertex (10, 0), but the
    # signed LATERAL offset must stay the pure perpendicular component (the
    # path runs along +X here, so that's just the Y difference = 1.0), not
    # the full clamped-point distance.
    signed, dist = _lateral_offset(15.0, 1.0, STRAIGHT_AHEAD)
    expected_dist = float(np.hypot(15.0 - 10.0, 1.0 - 0.0))  # sqrt(26)
    assert dist   == pytest.approx(expected_dist)
    assert signed == pytest.approx(1.0)


def test_lateral_offset_does_not_leak_longitudinal_distance_when_extrapolating():
    # A point 15 m behind the path's start (x=-15 < path start x=0), dead
    # centre laterally (y=0): the lateral offset must read ~0, not the 15 m
    # longitudinal gap to the clamped nearest point (a real bug found via
    # the perception evaluation harness -- a track behind the ego reported
    # a 15+ m "lateral offset" for a vehicle that was laterally aligned).
    signed, dist = _lateral_offset(-15.0, 0.0, STRAIGHT_AHEAD)
    assert signed == pytest.approx(0.0, abs=1e-9)
    assert dist   == pytest.approx(15.0)  # true nearest-point distance is unchanged

    # Same case, 2 m left: lateral must read +2.0, not hypot(15, 2).
    signed, dist = _lateral_offset(-15.0, 2.0, STRAIGHT_AHEAD)
    assert signed == pytest.approx(2.0)
    assert dist   == pytest.approx(float(np.hypot(15.0, 2.0)))


def test_pixel_dist_is_nearest_neighbour():
    polyline_px = np.array([[0.0, 0.0], [10.0, 0.0]])
    d = _pixel_dist(5.0, 3.0, polyline_px)
    assert d == pytest.approx(float(np.hypot(5.0, 3.0)))  # equidistant to both ends


def test_check_inside_bounds():
    left  = np.array([[0.0, 2.0], [10.0, 2.0]])
    right = np.array([[0.0, -2.0], [10.0, -2.0]])

    assert _check_inside(5.0, 0.0, left, right) is True
    assert _check_inside(5.0, 3.0, left, right) is False   # outside laterally
    assert _check_inside(15.0, 0.0, left, right) is False  # outside forward range


@pytest.mark.parametrize("lat, expected", [
    (0.0,   "on"),
    (0.49,  "on"),
    (-0.49, "on"),
    (0.5,   "left"),   # abs(lat) < threshold is strict, so exactly-0.5 is NOT "on"
    (-0.5,  "right"),
    (0.6,   "left"),
    (-0.6,  "right"),
])
def test_side_classification(lat, expected):
    assert _side(lat) == expected


def test_boundary_midline_is_the_exact_average_of_symmetric_boundaries():
    left  = np.array([[0.0, 2.0], [10.0, 2.0]])
    right = np.array([[0.0, -2.0], [10.0, -2.0]])
    mid = LaneRelationMeasurer._boundary_midline(left, right)
    assert mid.shape[0] == 20
    np.testing.assert_allclose(mid[:, 1], 0.0, atol=1e-9)
    assert mid[0, 0]  == pytest.approx(0.0)
    assert mid[-1, 0] == pytest.approx(10.0)


def test_pixels_to_vf_roundtrips_a_known_ground_point():
    calib = CameraCalibration.default_front()
    measurer = LaneRelationMeasurer(calib)

    p_cam = calib.R_vc @ np.array([20.0, -1.5, 0.0]) + calib.t_vc
    uvw = calib.K @ p_cam
    u, v = uvw[0] / uvw[2], uvw[1] / uvw[2]

    vf = measurer._pixels_to_vf([[u, v]])
    assert vf.shape == (1, 2)
    assert vf[0, 0] == pytest.approx(20.0, abs=1e-6)
    assert vf[0, 1] == pytest.approx(-1.5, abs=1e-6)
