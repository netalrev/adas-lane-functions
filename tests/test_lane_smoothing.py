"""Unit tests for temporal smoothing of lane/drivable-path polylines.

_smooth_polyline_temporal() damps frame-to-frame jitter in the drivable-path
(center/left/right) and host-lane (left/right) polylines published by
YOLOPv2DrivableDetector, while a divergence guard bypasses smoothing when the
new frame differs too much from history (fast ego-rotation, lane change,
reacquisition) -- this guards against the "ghosting" lag that a previous,
now-removed EMA implementation of the same idea suffered from (see
_smooth_polyline_temporal's docstring). All cases here use hand-derived
expected values; no ONNX/TF import required.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.detectors.lane.yolopv2_drivable import (
    YOLOPv2DrivableDetector, _smooth_polyline_temporal,
    _LANE_EMA_ALPHA, _LANE_MAX_SHIFT_PX,
)


def _line(xs, ys) -> np.ndarray:
    return np.column_stack([np.asarray(xs, dtype=np.int32), np.asarray(ys, dtype=np.int32)])


# ---------------------------------------------------------------------------
# Pure _smooth_polyline_temporal
# ---------------------------------------------------------------------------

def test_seeds_history_on_first_frame():
    new = _line([100, 110, 120], [300, 200, 100])
    out = _smooth_polyline_temporal(None, new, alpha=0.6, max_shift_px=70.0)
    np.testing.assert_array_equal(out, new)


def test_blends_a_small_shift_toward_history():
    prev = _line([100, 100, 100], [300, 200, 100])
    new  = _line([110, 110, 110], [300, 200, 100])   # 10px jitter, same y-grid

    out = _smooth_polyline_temporal(prev, new, alpha=0.6, max_shift_px=70.0)

    np.testing.assert_allclose(out[:, 0], 0.6 * 110 + 0.4 * 100)   # = 106
    np.testing.assert_array_equal(out[:, 1], new[:, 1])


def test_bypasses_smoothing_when_shift_exceeds_the_guard():
    # Simulates a fast ego-rotation / lane change: the true line really did
    # move a long way in one frame.  Blending toward stale history here
    # would reproduce the "ghosting" bug this replaces -- must pass through.
    prev = _line([100, 100, 100], [300, 200, 100])
    new  = _line([250, 250, 250], [300, 200, 100])   # 150px jump > max_shift_px

    out = _smooth_polyline_temporal(prev, new, alpha=0.6, max_shift_px=70.0)

    np.testing.assert_array_equal(out, new)


def test_resamples_history_onto_the_new_frames_y_grid():
    # History only covers y in [150, 300]; the new frame also samples y=50,
    # which has no prior estimate and must be left unsmoothed.
    prev = _line([100, 120], [300, 150])
    new  = _line([110, 130, 999], [300, 150, 50])

    out = _smooth_polyline_temporal(prev, new, alpha=0.5, max_shift_px=70.0)

    assert out[0, 0] == pytest.approx(0.5 * 110 + 0.5 * 100)   # y=300: history exists
    assert out[1, 0] == pytest.approx(0.5 * 130 + 0.5 * 120)   # y=150: history exists
    assert out[2, 0] == 999                                     # y=50: no history, passthrough


def test_returns_new_unchanged_when_new_is_none_or_empty():
    prev = _line([100], [300])
    assert _smooth_polyline_temporal(prev, None, 0.6, 70.0) is None
    empty = np.empty((0, 2), dtype=np.int32)
    assert _smooth_polyline_temporal(prev, empty, 0.6, 70.0) is empty


def test_returns_new_unchanged_when_history_too_short():
    new = _line([100, 110], [300, 200])
    one_point_prev = _line([50], [300])
    out = _smooth_polyline_temporal(one_point_prev, new, 0.6, 70.0)
    np.testing.assert_array_equal(out, new)


# ---------------------------------------------------------------------------
# Integration: drivable-path center/boundaries persist + blend real history
# across successive calls.  _mask_to_centerline/_mask_to_boundaries only use
# mask-processing attributes (not the ONNX session), so __init__ is bypassed
# to avoid needing a real yolopv2.onnx file in the test environment.
# ---------------------------------------------------------------------------

def _bare_detector() -> YOLOPv2DrivableDetector:
    det = object.__new__(YOLOPv2DrivableDetector)
    det.min_drivable_pix   = 5
    det._lane_ema_alpha    = _LANE_EMA_ALPHA
    det._lane_max_shift_px = _LANE_MAX_SHIFT_PX
    det._prev_dp_points    = {"center": None, "left": None, "right": None}
    return det


def _drivable_mask(h: int, w: int, band_x: int, band_w: int = 40) -> np.ndarray:
    """A synthetic drivable-area mask: a single vertical band `band_w` px
    wide centred at `band_x`, spanning every row (well above min_drivable_pix)."""
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, x1 = max(0, band_x - band_w // 2), min(w, band_x + band_w // 2)
    mask[:, x0:x1] = 1
    return mask


def test_mask_to_centerline_damps_a_one_frame_shift():
    det = _bare_detector()
    h, w = 720, 1280

    for _ in range(3):
        center = det._mask_to_centerline(_drivable_mask(h, w, band_x=640), h, w)
    steady_x = float(center[:, 0].mean())
    assert steady_x == pytest.approx(640, abs=2.0)

    # One noisy frame: the mask shifts by 40px -- small enough to be damped,
    # not treated as a genuine lane change.
    shifted = det._mask_to_centerline(_drivable_mask(h, w, band_x=680), h, w)
    shifted_x = float(shifted[:, 0].mean())
    assert steady_x < shifted_x < 678.0   # damped: strictly between old and new


def test_mask_to_centerline_does_not_lag_a_genuine_large_shift():
    det = _bare_detector()
    h, w = 720, 1280

    for _ in range(3):
        det._mask_to_centerline(_drivable_mask(h, w, band_x=640), h, w)

    # A big jump (200px, >> lane_max_shift_px) simulates a real lane change /
    # sharp turn -- must be published as-is, not dragged toward stale history.
    moved = det._mask_to_centerline(_drivable_mask(h, w, band_x=840), h, w)
    assert float(moved[:, 0].mean()) == pytest.approx(840, abs=2.0)


def test_mask_to_boundaries_persists_and_blends_across_calls():
    det = _bare_detector()
    h, w = 720, 1280

    for _ in range(3):
        left, right = det._mask_to_boundaries(_drivable_mask(h, w, band_x=640), h, w)
    steady_width = float((right[:, 0] - left[:, 0]).mean())
    steady_left_x = float(left[:, 0].mean())

    left2, right2 = det._mask_to_boundaries(_drivable_mask(h, w, band_x=680), h, w)
    # Both boundaries follow the (damped) centre shift together, so the lane
    # width they imply stays essentially unchanged.
    assert float((right2[:, 0] - left2[:, 0]).mean()) == pytest.approx(steady_width, abs=3.0)
    assert float(left2[:, 0].mean()) > steady_left_x
