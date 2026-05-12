"""
src/models/tracking/kalman_tracker.py
=======================================
Per-track Constant Velocity Kalman Filter in vehicle frame.

Public API
----------
    KalmanTracker(initial_xy, dt, q_pos, q_vel, r_pos)
        Initialise a new track at (x_veh, y_veh) with zero velocity.

    KalmanTracker.predict() -> np.ndarray
        Advance the state by dt without a measurement update.
        Returns the predicted state [x, y, vx, vy].

    KalmanTracker.update(measurement)
        Incorporate a new (x, y) measurement and update the posterior.

    KalmanTracker.state -> np.ndarray
        Current state estimate [x, y, vx, vy].

State / coordinate conventions
-------------------------------
All quantities are in the Vehicle Frame:
    x  — range (m),          positive = forward
    y  — lateral offset (m), positive = left
    vx — range rate (m/s),   positive = moving away from ego
    vy — lateral rate (m/s), positive = moving left

Model matrices
--------------
Transition F (Constant Velocity):
    [1  0  dt  0 ]
    [0  1   0 dt ]
    [0  0   1  0 ]
    [0  0   0  1 ]

Measurement H (observe position only):
    [1  0  0  0]
    [0  1  0  0]

Process noise Q = diag(q_pos, q_pos, q_vel, q_vel)
Measurement noise R = diag(r_pos, r_pos)
"""

from __future__ import annotations

import numpy as np


class KalmanTracker:
    """
    Constant Velocity Kalman filter for one track in vehicle frame.

    Parameters
    ----------
    initial_xy : array-like, shape (2,)
        Initial (x_veh, y_veh) position in metres.
    dt : float
        Inter-frame time step in seconds.
    q_pos : float
        Process noise standard deviation for position components.
    q_vel : float
        Process noise standard deviation for velocity components.
    r_pos : float
        Measurement noise standard deviation for position observations.
    """

    def __init__(
        self,
        initial_xy: np.ndarray | list,
        dt: float,
        q_pos: float,
        q_vel: float,
        r_pos: float,
    ) -> None:
        self._dt = dt

        # State transition matrix
        self._F = np.array([
            [1.0, 0.0,  dt, 0.0],
            [0.0, 1.0, 0.0,  dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Measurement matrix (observe x, y only)
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Process noise covariance
        self._Q = np.diag([q_pos ** 2, q_pos ** 2,
                           q_vel ** 2, q_vel ** 2]).astype(np.float64)

        # Measurement noise covariance
        self._R = np.diag([r_pos ** 2, r_pos ** 2]).astype(np.float64)

        # Identity (reused in update step)
        self._I = np.eye(4, dtype=np.float64)

        # Initial state: position from measurement, velocity = 0
        x0, y0 = float(initial_xy[0]), float(initial_xy[1])
        self._x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)

        # Initial covariance: high uncertainty on velocity
        self._P = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float64)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """
        Time-update step: propagate state and covariance by dt.

        Returns
        -------
        np.ndarray, shape (4,)
            Predicted state [x, y, vx, vy].
        """
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x.copy()

    def update(self, measurement: np.ndarray) -> None:
        """
        Measurement-update step: fuse a new (x, y) observation.

        Parameters
        ----------
        measurement : array-like, shape (2,)
            Observed (x_veh, y_veh) position in metres.
        """
        z = np.asarray(measurement, dtype=np.float64)
        y = z - self._H @ self._x                         # innovation
        S = self._H @ self._P @ self._H.T + self._R       # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)        # Kalman gain
        self._x = self._x + K @ y
        self._P = (self._I - K @ self._H) @ self._P

    @property
    def state(self) -> np.ndarray:
        """
        Current state estimate [x, y, vx, vy] (copy).
        """
        return self._x.copy()

    def update_dt(self, new_dt: float) -> None:
        """
        Update the inter-frame dt (e.g. when derived from actual timestamps).
        Updates the transition matrix in-place.
        """
        self._dt        = new_dt
        self._F[0, 2]   = new_dt
        self._F[1, 3]   = new_dt
