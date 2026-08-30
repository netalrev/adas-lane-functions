"""Unit tests for the pose-derived yaw-rate estimator feeding the kinematic
ego-path strategy (Path 1)."""
from __future__ import annotations

import numpy as np
import pytest

from src.detectors.lane.kinematic_ego import compute_yaw_rate


def _transform_for_heading(heading_rad: float) -> list:
    """Minimal 16-element pose transform with the given XY heading.

    Only indices [0] (R00) and [4] (R10) are read by compute_yaw_rate, but a
    full 4x4 is provided for realism.
    """
    c, s = float(np.cos(heading_rad)), float(np.sin(heading_rad))
    return [c, -s, 0.0, 0.0,
            s,  c, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0]


def test_quarter_turn_over_one_second_is_a_quarter_pi_per_second():
    prev = _transform_for_heading(0.0)
    curr = _transform_for_heading(np.pi / 2.0)
    assert compute_yaw_rate(prev, curr, dt=1.0) == pytest.approx(np.pi / 2.0)


def test_zero_or_negative_dt_returns_zero():
    prev = _transform_for_heading(0.0)
    curr = _transform_for_heading(0.5)
    assert compute_yaw_rate(prev, curr, dt=0.0)  == 0.0
    assert compute_yaw_rate(prev, curr, dt=-0.1) == 0.0


def test_heading_wraparound_takes_the_short_way():
    # Heading crosses the +-pi branch cut; the true rotation is +0.2 rad,
    # not the -2*pi+0.2 the raw difference would naively suggest.
    prev = _transform_for_heading(np.pi - 0.1)
    curr = _transform_for_heading(-np.pi + 0.1)
    assert compute_yaw_rate(prev, curr, dt=1.0) == pytest.approx(0.2, abs=1e-9)
