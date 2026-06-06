"""
src/models/tracking/vehicle_ekf.py
====================================
Extended Kalman Filter for a single confirmed vehicle track.

State vector (9D)
-----------------
    idx  symbol   unit    description
    0    x        m       range (forward distance, Vehicle Frame)
    1    y        m       lateral offset (left positive, Vehicle Frame)
    2    z        m       centre height above ground plane
    3    vx       m/s     range rate (positive = moving away from ego)
    4    vy       m/s     lateral rate (positive = moving left)
    5    heading  rad     yaw relative to ego X-axis
    6    width    m       lateral extent of the vehicle
    7    height   m       vertical extent of the vehicle
    8    length   m       longitudinal extent (weakly observable from front camera)

Process model — Constant Velocity (linear, so F is constant)
-------------------------------------------------------------
    x(k+1) = x(k) + vx(k) * dt
    y(k+1) = y(k) + vy(k) * dt
    z, vx, vy, heading, width, height, length — constant + process noise

Measurement model — 4D, nonlinear (justifies the EKF over a plain KF)
-----------------------------------------------------------------------
    z_meas = [x_gnd, y_gnd, h_aspect, w_aspect]

    x_gnd, y_gnd : ground-plane projection of bottom-centre pixel (m)
    h_aspect     : bbox_h_px / fy  — observes height / x  (nonlinear in x, height)
    w_aspect     : bbox_w_px / fx  — observes width  / x  (nonlinear in x, width)

EKF Jacobian H = ∂h/∂x evaluated analytically at the current state x̂:

              x      y    z    vx   vy  hdg    w      h    l
    H = [[    1,    0,   0,   0,   0,   0,    0,     0,   0 ],   # h₁ = x
         [    0,    1,   0,   0,   0,   0,    0,     0,   0 ],   # h₂ = y
         [ -H/x²,  0,   0,   0,   0,   0,    0,   1/x,   0 ],   # h₃ = H/x
         [ -W/x²,  0,   0,   0,   0,   0,   1/x,   0,    0 ]]   # h₄ = W/x

Physical priors for first-detection initialisation
---------------------------------------------------
    z       = 0.75 m   (mid-height of a typical 1.5 m sedan)
    width   = 2.0  m
    height  = 1.5  m
    length  = 4.5  m
"""

from __future__ import annotations

import numpy as np

from .vehicle_measurement import KalmanMeasurement

# ---------------------------------------------------------------------------
# State / measurement index constants (module-level for use by manager)
# ---------------------------------------------------------------------------

IDX_X:   int = 0
IDX_Y:   int = 1
IDX_Z:   int = 2
IDX_VX:  int = 3
IDX_VY:  int = 4
IDX_HDG: int = 5
IDX_W:   int = 6
IDX_H:   int = 7
IDX_L:   int = 8
STATE_DIM: int = 9

MEAS_X:        int = 0
MEAS_Y:        int = 1
MEAS_H_ASPECT: int = 2
MEAS_W_ASPECT: int = 3
MEAS_DIM: int = 4

# Vehicle dimension priors for track birth
PRIOR_Z_M:      float = 0.75
PRIOR_WIDTH_M:  float = 2.0
PRIOR_HEIGHT_M: float = 1.5
PRIOR_LENGTH_M: float = 4.5

# Physically plausible size floor values (prevent numerical collapse)
_MIN_WIDTH_M:  float = 0.5
_MIN_HEIGHT_M: float = 0.3
_MIN_LENGTH_M: float = 1.0

# Range floor: prevents division-by-zero in perspective measurement Jacobian
_MIN_RANGE_M: float = 0.5


class VehicleEKF:
    """
    9-state Extended Kalman Filter for one vehicle track.

    The nonlinear perspective size measurement h₃ = height/x and
    h₄ = width/x require a linearised Jacobian → Extended KF framework.

    Parameters
    ----------
    initial_xy : array-like, shape (2,)
        (x_veh, y_veh) from the ground-projection of the first detection.
    dt : float
        Inter-frame time step (seconds).
    q_pos : float
        Process noise std-dev for horizontal position components (x, y).
    q_vel : float
        Process noise std-dev for velocity components (vx, vy).
    q_heading : float
        Process noise std-dev for heading (rad).
    q_size : float
        Process noise std-dev for width and height (m).
    q_length : float
        Process noise std-dev for length (m) — weakly observable.
    r_pos : float
        Measurement noise std-dev for ground-projection position (m).
    r_aspect : float
        Measurement noise std-dev for perspective aspect-ratio observations.
    initial_w : float | None
        First-frame width estimate (m); falls back to PRIOR_WIDTH_M.
    initial_h : float | None
        First-frame height estimate (m); falls back to PRIOR_HEIGHT_M.
    """

    def __init__(
        self,
        initial_xy: np.ndarray | list,
        dt:         float,
        q_pos:      float = 0.10,
        q_vel:      float = 1.00,
        q_heading:  float = 0.05,
        q_size:     float = 0.05,
        q_length:   float = 0.10,
        r_pos:      float = 0.50,
        r_aspect:   float = 0.05,
        initial_w:  float | None = None,
        initial_h:  float | None = None,
    ) -> None:
        self._dt = dt

        # ── Initial state ─────────────────────────────────────────────────
        x0 = float(initial_xy[0])
        y0 = float(initial_xy[1])
        w0 = float(initial_w) if initial_w is not None else PRIOR_WIDTH_M
        h0 = float(initial_h) if initial_h is not None else PRIOR_HEIGHT_M

        self._x: np.ndarray = np.array(
            [x0, y0, PRIOR_Z_M, 0.0, 0.0, 0.0, w0, h0, PRIOR_LENGTH_M],
            dtype=np.float64,
        )

        # ── Initial covariance ────────────────────────────────────────────
        # High uncertainty on velocity and heading (not observed directly).
        # Modest uncertainty on position and sizes (priors are reasonable).
        self._P: np.ndarray = np.diag([
            2.00,                    # x
            2.00,                    # y
            0.50,                    # z
            9.00,                    # vx
            9.00,                    # vy
            (np.pi / 2.0) ** 2,      # heading — quarter-circle uncertainty
            0.50,                    # width
            0.25,                    # height
            1.00,                    # length
        ]).astype(np.float64)

        # ── Process noise Q (diagonal) ────────────────────────────────────
        self._Q: np.ndarray = np.diag([
            q_pos    ** 2,           # x
            q_pos    ** 2,           # y
            0.01     ** 2,           # z  — flat-ground assumption; nearly constant
            q_vel    ** 2,           # vx
            q_vel    ** 2,           # vy
            q_heading ** 2,          # heading
            q_size   ** 2,           # width
            q_size   ** 2,           # height
            q_length ** 2,           # length
        ]).astype(np.float64)

        # ── Measurement noise R (diagonal) ───────────────────────────────
        self._R: np.ndarray = np.diag([
            r_pos    ** 2,           # x_gnd
            r_pos    ** 2,           # y_gnd
            r_aspect ** 2,           # h_aspect
            r_aspect ** 2,           # w_aspect
        ]).astype(np.float64)

        # ── Identity + constant process matrix ───────────────────────────
        self._I = np.eye(STATE_DIM, dtype=np.float64)
        self._F = np.eye(STATE_DIM, dtype=np.float64)
        self._update_F(dt)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """
        Time-update step: propagate state and covariance by dt.

        Returns
        -------
        np.ndarray, shape (9,)
            Predicted state [x, y, z, vx, vy, heading, w, h, l] (copy).
        """
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def update(self, meas: KalmanMeasurement) -> None:
        """
        Measurement-update: fuse one KalmanMeasurement using EKF equations.

        Delegates to the full 4D update or the 2D position-only fallback
        based on ``meas.use_size``.  All geometry and noise handling is
        encapsulated in the private helpers below.

        Parameters
        ----------
        meas : KalmanMeasurement
            Output of ``build_kalman_input()``.  Contains validated position
            and (optionally) aspect-ratio observations.
        """
        if meas.use_size:
            self._update_4d(meas)
        else:
            self._update_2d(meas.x_gnd, meas.y_gnd)

    # ------------------------------------------------------------------
    # Private: EKF update implementations
    # ------------------------------------------------------------------

    def _update_4d(self, meas: KalmanMeasurement) -> None:
        """
        Full 4D EKF update using position + perspective size measurements.

        Measurement vector: z = [x_gnd, y_gnd, h_aspect, w_aspect]
            h_aspect = bbox_h_px / fy  ≈  H_real / x
            w_aspect = bbox_w_px / fx  ≈  W_real / x

        The aspect-ratio terms are nonlinear in state (x, H, W), which is
        why we need the EKF Jacobian rather than a plain KF.
        """
        x_safe = max(_MIN_RANGE_M, float(self._x[IDX_X]))

        z_meas = np.array(
            [meas.x_gnd, meas.y_gnd, meas.h_aspect, meas.w_aspect],
            dtype=np.float64,
        )

        # Predicted measurement h(x̂) at current state
        z_pred = np.array([
            self._x[IDX_X],
            self._x[IDX_Y],
            self._x[IDX_H] / x_safe,    # h₃ = H/x
            self._x[IDX_W] / x_safe,    # h₄ = W/x
        ], dtype=np.float64)

        H = self._measurement_jacobian(x_safe)  # ∂h/∂x evaluated at x̂

        S = H @ self._P @ H.T + self._R
        K = self._P @ H.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z_meas - z_pred)
        self._P = (self._I - K @ H) @ self._P

        self._post_update_clamp()

    def _update_2d(self, x_gnd: float, y_gnd: float) -> None:
        """
        2D position-only EKF update using the ground-plane projection.

        Used when the bbox is too small for reliable aspect-ratio measurements
        or the vehicle is inside the near-field threshold.

        Measurement vector: z = [x_gnd, y_gnd]
        Observation matrix H₂ selects [x, y] directly from the state.
        """
        H2 = np.zeros((2, STATE_DIM), dtype=np.float64)
        H2[MEAS_X, IDX_X] = 1.0
        H2[MEAS_Y, IDX_Y] = 1.0

        z  = np.array([x_gnd, y_gnd], dtype=np.float64)
        zp = H2 @ self._x
        S  = H2 @ self._P @ H2.T + self._R[:2, :2]
        K  = self._P @ H2.T @ np.linalg.inv(S)
        self._x = self._x + K @ (z - zp)
        self._P = (self._I - K @ H2) @ self._P

        self._post_update_clamp()

    def update_dt(self, new_dt: float) -> None:
        """Re-synchronise the process matrix when the frame interval changes."""
        if abs(new_dt - self._dt) > 1e-6:
            self._dt = new_dt
            self._update_F(new_dt)

    @property
    def state(self) -> np.ndarray:
        """Current state estimate (copy): [x,y,z,vx,vy,heading,w,h,l]."""
        return self._x.copy()

    @property
    def speed_mps(self) -> float:
        """Scalar ground speed in m/s: sqrt(vx² + vy²)."""
        return float(np.hypot(self._x[IDX_VX], self._x[IDX_VY]))

    @property
    def heading_from_velocity(self) -> float:
        """
        Heading derived from the velocity vector (rad).
        More reliable than the state heading component during low-speed frames.
        Falls back to stored heading when the vehicle is nearly stationary.
        """
        vx = float(self._x[IDX_VX])
        vy = float(self._x[IDX_VY])
        if abs(vx) < 0.5 and abs(vy) < 0.5:
            return float(self._x[IDX_HDG])
        return float(np.arctan2(vy, vx))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_F(self, dt: float) -> None:
        """Rebuild the (linear) process matrix for the given time step."""
        self._F = np.eye(STATE_DIM, dtype=np.float64)
        self._F[IDX_X, IDX_VX] = dt
        self._F[IDX_Y, IDX_VY] = dt

    def _measurement_jacobian(self, x_safe: float) -> np.ndarray:
        """
        Analytical Jacobian H = ∂h/∂x evaluated at the current state.

        Derivation:
            h₁ = x            → ∂h₁/∂x = 1
            h₂ = y            → ∂h₂/∂y = 1
            h₃ = height / x   → ∂h₃/∂x = -height/x², ∂h₃/∂height = 1/x
            h₄ = width  / x   → ∂h₄/∂x = -width /x², ∂h₄/∂width  = 1/x
        """
        H_val  = float(self._x[IDX_H])
        W_val  = float(self._x[IDX_W])
        inv_x  = 1.0 / x_safe
        inv_x2 = inv_x * inv_x

        H_jac = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)

        # h₁: observe x
        H_jac[MEAS_X, IDX_X] = 1.0

        # h₂: observe y
        H_jac[MEAS_Y, IDX_Y] = 1.0

        # h₃: observe height/x
        H_jac[MEAS_H_ASPECT, IDX_X] = -H_val * inv_x2
        H_jac[MEAS_H_ASPECT, IDX_H] = inv_x

        # h₄: observe width/x
        H_jac[MEAS_W_ASPECT, IDX_X] = -W_val * inv_x2
        H_jac[MEAS_W_ASPECT, IDX_W] = inv_x

        return H_jac

    def _post_update_clamp(self) -> None:
        """
        Clamp physically implausible state values after any update step.
        Prevents EKF divergence from turning into unbounded size estimates.
        """
        # Keep heading in [-π, π]
        self._x[IDX_HDG] = float(
            np.arctan2(np.sin(self._x[IDX_HDG]), np.cos(self._x[IDX_HDG]))
        )
        # Clamp dimensions to physically plausible ranges
        self._x[IDX_W] = max(_MIN_WIDTH_M,  float(self._x[IDX_W]))
        self._x[IDX_H] = max(_MIN_HEIGHT_M, float(self._x[IDX_H]))
        self._x[IDX_L] = max(_MIN_LENGTH_M, float(self._x[IDX_L]))
