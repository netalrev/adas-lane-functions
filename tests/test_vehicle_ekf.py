"""Unit tests for VehicleEKF: the 9D EKF used for per-vehicle state estimation.

Where the state-transition/update equations are closed-form, expected values
are hand-derived independently of the implementation. The nonlinear
measurement Jacobian is checked against a numerical finite-difference
Jacobian -- the standard way to validate an EKF's analytic H matrix.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.measurements.vehicle_ekf import (
    VehicleEKF, IDX_X, IDX_Y, IDX_VX, IDX_VY, IDX_HDG, IDX_W, IDX_H, IDX_L,
    _MIN_WIDTH_M, _MIN_HEIGHT_M, _MIN_LENGTH_M,
)
from src.measurements.vehicle_measurement import KalmanMeasurement


def test_process_matrix_only_couples_position_to_velocity():
    ekf = VehicleEKF(initial_xy=[10.0, 2.0], dt=0.1)
    expected = np.eye(9)
    expected[IDX_X, IDX_VX] = 0.1
    expected[IDX_Y, IDX_VY] = 0.1
    np.testing.assert_array_equal(ekf._F, expected)


def test_predict_matches_closed_form_constant_velocity():
    ekf = VehicleEKF(initial_xy=[10.0, 2.0], dt=0.1)
    ekf._x[IDX_VX] = 3.0    # m/s forward
    ekf._x[IDX_VY] = -0.5   # m/s lateral

    for _ in range(5):
        ekf.predict()

    # Constant-velocity model: position = x0 + v * (n * dt), exactly.
    assert ekf.state[IDX_X] == pytest.approx(10.0 + 3.0 * 5 * 0.1, abs=1e-9)
    assert ekf.state[IDX_Y] == pytest.approx(2.0 + (-0.5) * 5 * 0.1, abs=1e-9)
    # Velocity itself is untouched by a constant-velocity F (no acceleration state).
    assert ekf.state[IDX_VX] == pytest.approx(3.0, abs=1e-9)
    assert ekf.state[IDX_VY] == pytest.approx(-0.5, abs=1e-9)


def test_2d_update_moves_state_toward_measurement_by_hand_computed_gain():
    ekf = VehicleEKF(initial_xy=[10.0, 2.0], dt=0.1)  # defaults: r_pos=0.5

    # At t=0, P is diagonal (P_xx = P_yy = 2.0) and R[:2,:2] = diag(0.25).
    # For the position-only update, K reduces to a diagonal gain of
    # P_xx / (P_xx + r_pos^2) = 2.0 / 2.25 on both x and y (hand-derived).
    meas = KalmanMeasurement(x_gnd=12.0, y_gnd=2.5, h_aspect=float("nan"),
                              w_aspect=float("nan"), use_size=False)
    assert ekf.update(meas) is True

    gain = 2.0 / 2.25
    assert ekf.state[IDX_X] == pytest.approx(10.0 + gain * (12.0 - 10.0), abs=1e-9)
    assert ekf.state[IDX_Y] == pytest.approx(2.0 + gain * (2.5 - 2.0),   abs=1e-9)
    # Unobserved components must be untouched by the position-only update.
    assert ekf.state[IDX_VX]  == pytest.approx(0.0, abs=1e-9)
    assert ekf.state[IDX_HDG] == pytest.approx(0.0, abs=1e-9)


def test_2d_update_rejects_a_statistical_outlier_and_leaves_state_unchanged():
    ekf = VehicleEKF(initial_xy=[10.0, 2.0], dt=0.1)
    state_before = ekf.state.copy()

    # Innovation of 15 m at S_xx=2.25 gives Mahalanobis^2 = 15^2/2.25 = 100,
    # far past the chi-squared(2 dof, p=0.999) gate of 13.816 -- an
    # implausible jump consistent with a bad detection / wrong association.
    outlier = KalmanMeasurement(x_gnd=25.0, y_gnd=2.0, h_aspect=float("nan"),
                                 w_aspect=float("nan"), use_size=False)
    accepted = ekf.update(outlier)

    assert accepted is False
    np.testing.assert_array_equal(ekf.state, state_before)


def test_2d_update_accepts_a_plausible_measurement():
    ekf = VehicleEKF(initial_xy=[10.0, 2.0], dt=0.1)
    meas = KalmanMeasurement(x_gnd=12.0, y_gnd=2.5, h_aspect=float("nan"),
                              w_aspect=float("nan"), use_size=False)
    assert ekf.update(meas) is True


def test_measurement_jacobian_matches_numerical_finite_difference():
    ekf = VehicleEKF(initial_xy=[15.0, 1.0], dt=0.1)
    ekf._x[IDX_VX] = 2.0
    ekf._x[IDX_H]  = 1.6
    ekf._x[IDX_W]  = 1.9
    x_safe = float(ekf._x[IDX_X])

    analytic_H = ekf._measurement_jacobian(x_safe)

    def h(state):
        xs = max(0.5, state[IDX_X])
        return np.array([state[IDX_X], state[IDX_Y], state[IDX_H] / xs, state[IDX_W] / xs])

    x0  = ekf.state
    eps = 1e-6
    numeric_H = np.zeros((4, 9))
    for i in range(9):
        dx = np.zeros(9); dx[i] = eps
        numeric_H[:, i] = (h(x0 + dx) - h(x0 - dx)) / (2 * eps)

    np.testing.assert_allclose(analytic_H, numeric_H, atol=1e-4)


def test_post_update_clamp_enforces_physical_floors_and_wraps_heading():
    ekf = VehicleEKF(initial_xy=[10.0, 0.0], dt=0.1)
    ekf._x[IDX_W]   = -5.0
    ekf._x[IDX_H]   = -5.0
    ekf._x[IDX_L]   = -5.0
    ekf._x[IDX_HDG] = 4.0   # > pi, must wrap into [-pi, pi]

    ekf._post_update_clamp()

    assert ekf.state[IDX_W] == pytest.approx(_MIN_WIDTH_M)
    assert ekf.state[IDX_H] == pytest.approx(_MIN_HEIGHT_M)
    assert ekf.state[IDX_L] == pytest.approx(_MIN_LENGTH_M)
    # atan2(sin(4.0), cos(4.0)) wraps 4.0 rad down by exactly one full turn.
    assert ekf.state[IDX_HDG] == pytest.approx(4.0 - 2 * np.pi, abs=1e-9)
