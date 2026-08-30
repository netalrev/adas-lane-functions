"""Unit tests for HostLaneStrategy.package() -- a pure, stateless function.

Covers the bug found via the perception evaluation harness: "valid_center"
was hardcoded False and "center" hardcoded empty even when both boundaries
were valid, so no consumer checking "valid_center" ever saw a host-lane
result even though LaneRelationMeasurer was already deriving a midline
from the same left/right arrays internally.
"""
from __future__ import annotations

from src.detectors.lane.visual_host import HostLaneStrategy


def test_package_derives_center_when_both_boundaries_valid():
    raw = {
        "left_lane":  [[100, 0], [100, 100]],
        "right_lane": [[200, 0], [200, 100]],
        "valid_left":  True,
        "valid_right": True,
        "confidence_left":  0.5,
        "confidence_right": 0.6,
        "source": "test",
    }
    result = HostLaneStrategy.package(raw)

    assert result["valid_center"] is True
    assert len(result["center"]) == 30
    assert all(p[0] == 150 for p in result["center"])  # exact midpoint of x=100/x=200
    assert result["center"][0][1]  == 0
    assert result["center"][-1][1] == 100


def test_package_center_stays_empty_when_only_one_boundary_valid():
    raw = {
        "left_lane":  [[100, 0], [100, 100]],
        "right_lane": [[200, 0], [200, 100]],
        "valid_left":  True,
        "valid_right": False,   # right marking not confidently detected
        "confidence_left":  0.5,
        "confidence_right": 0.1,
        "source": "test",
    }
    result = HostLaneStrategy.package(raw)

    assert result["valid_center"] is False
    assert result["center"] == []
    assert result["valid_left"]  is True
    assert result["valid_right"] is False


def test_package_handles_missing_data():
    result = HostLaneStrategy.package(None)
    assert result["center"] == [] and result["left"] == [] and result["right"] == []
    assert result["valid_center"] is False
    assert result["valid_left"]  is False
    assert result["valid_right"] is False
    assert result["source"] == "none"


def test_package_supports_legacy_single_valid_key():
    # CLRNet / IPM style: one "valid" flag covers both sides.
    raw = {
        "left_lane":  [[100, 0], [100, 100]],
        "right_lane": [[200, 0], [200, 100]],
        "valid":      True,
        "confidence": 0.7,
        "source":     "ipm",
    }
    result = HostLaneStrategy.package(raw)

    assert result["valid_left"]  is True
    assert result["valid_right"] is True
    assert result["valid_center"] is True
    assert len(result["center"]) == 30
