"""Unit tests for per-track bounding-box EMA smoothing.

Both TrackManager (general 2D tracker) and VehicleTrackManager (vehicle EKF
tracker) blend the box they report/draw each frame via an identical
`_smooth_bbox` helper, so a single noisy detector frame -- e.g. a partially
occluded vehicle whose box only spans its visible lower half instead of its
full height -- is damped instead of being reported verbatim. All cases here
use hand-derived expected values; no TF/ONNX import required.
"""
from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

from src.detectors.vehicle.detector import Detection, CLASS_VEHICLE
from src.detectors.vehicle.track_manager import (
    TrackManager, _smooth_bbox as _smooth_bbox_general,
)
from src.measurements.vehicle_track_manager import (
    VehicleTrackManager, _smooth_bbox as _smooth_bbox_ekf,
)
from src.visualization.visualizer import CameraCalibration

_BOTH_SMOOTHERS = [_smooth_bbox_general, _smooth_bbox_ekf]


def _det(bbox: list[float], conf: float = 0.9) -> Detection:
    return Detection(
        bbox_xyxy=np.array(bbox, dtype=np.float64),
        confidence=conf, class_id=CLASS_VEHICLE, class_name="vehicle",
    )


# ---------------------------------------------------------------------------
# Pure _smooth_bbox helper (both modules define an identical copy)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("smooth_bbox", _BOTH_SMOOTHERS)
def test_smooth_bbox_damps_a_bad_half_height_frame(smooth_bbox):
    prev = np.array([100.0, 200.0, 300.0, 500.0])   # height = 300 px
    new  = np.array([100.0, 350.0, 300.0, 500.0])   # height = 150 px (bad frame)

    out = smooth_bbox(prev, new, was_fresh=True, alpha=0.4)

    np.testing.assert_allclose(out, 0.4 * new + 0.6 * prev)
    out_h = out[3] - out[1]
    # Damped, not a hard snap to the bad value: strictly between the two,
    # much closer to the true (previous) height than a verbatim pass-through.
    assert 150.0 < out_h < 300.0
    assert out_h == pytest.approx(0.4 * 150.0 + 0.6 * 300.0)


@pytest.mark.parametrize("smooth_bbox", _BOTH_SMOOTHERS)
def test_smooth_bbox_passes_raw_through_after_a_coasting_gap(smooth_bbox):
    # A stale pre-gap box must not anchor a fresh post-occlusion detection.
    prev = np.array([100.0, 200.0, 300.0, 500.0])
    new  = np.array([120.0, 210.0, 320.0, 520.0])

    out = smooth_bbox(prev, new, was_fresh=False, alpha=0.4)

    np.testing.assert_array_equal(out, new)
    assert out is not new   # defensive copy, not an alias into detector output


@pytest.mark.parametrize("smooth_bbox", _BOTH_SMOOTHERS)
def test_smooth_bbox_alpha_one_disables_smoothing(smooth_bbox):
    prev = np.array([100.0, 200.0, 300.0, 500.0])
    new  = np.array([100.0, 350.0, 300.0, 500.0])
    out  = smooth_bbox(prev, new, was_fresh=True, alpha=1.0)
    np.testing.assert_allclose(out, new)


# ---------------------------------------------------------------------------
# TrackManager integration: this bbox_xyxy also feeds MFAssembler's
# bbox_width_norm / bbox_height_norm / bbox_aspect_norm training features.
# ---------------------------------------------------------------------------

def _tracker_cfg(**overrides) -> OmegaConf:
    base = dict(
        max_age_tentative=3, max_age_confirmed=20, min_hits=1,
        iou_threshold=0.1,   reacquire_dist_px=0,   default_dt=0.1,
        process_noise_pos=0.10, process_noise_vel=1.00, measurement_noise=0.50,
        bbox_ema_alpha=0.4,
    )
    base.update(overrides)
    return OmegaConf.create(base)


def test_track_manager_output_box_damps_one_bad_frame_then_recovers():
    tm = TrackManager(_tracker_cfg())
    rw_pos = [(50.0, 0.0)]

    good_box = [100.0, 200.0, 300.0, 500.0]   # height 300 px
    bad_box  = [100.0, 350.0, 300.0, 500.0]   # height 150 px, single bad frame

    for _ in range(3):
        [track] = tm.update([_det(good_box)], rw_pos)
    assert (track.bbox_xyxy[3] - track.bbox_xyxy[1]) == pytest.approx(300.0, abs=1e-6)

    [track] = tm.update([_det(bad_box)], rw_pos)
    bad_h = track.bbox_xyxy[3] - track.bbox_xyxy[1]
    assert 150.0 < bad_h < 300.0

    [track] = tm.update([_det(good_box)], rw_pos)
    recovered_h = track.bbox_xyxy[3] - track.bbox_xyxy[1]
    assert recovered_h > bad_h


def test_track_manager_does_not_lag_on_reacquisition_after_occlusion():
    tm = TrackManager(_tracker_cfg())
    rw_pos = [(50.0, 0.0)]
    box = [100.0, 200.0, 300.0, 500.0]

    for _ in range(3):
        [track] = tm.update([_det(box)], rw_pos)

    tm.update([], [])   # one missed frame (occlusion) -- track keeps coasting

    moved_box = [140.0, 220.0, 340.0, 520.0]
    [track] = tm.update([_det(moved_box)], rw_pos)
    # Reacquisition after a gap must use the fresh box as-is, not blend
    # against the stale pre-gap position.
    np.testing.assert_allclose(track.bbox_xyxy, moved_box)


# ---------------------------------------------------------------------------
# VehicleTrackManager integration (drives the debug_viewer EKF panel and the
# vehicle_ekf_tracks JSON output).  The EKF's own filtered state (x_veh,
# y_veh, width_m, height_m, ...) is computed from the RAW per-frame
# detection independently of this smoothing, so it is not re-tested here.
# ---------------------------------------------------------------------------

def _vehicle_ekf_cfg(**overrides) -> OmegaConf:
    base = dict(
        max_age_tentative=3, max_age_confirmed=20, min_hits=1,
        iou_threshold=0.1,   reacquire_dist_px=0,   default_dt=0.1,
        process_noise_pos=0.10, process_noise_vel=1.00,
        process_noise_heading=0.05, process_noise_size=0.05, process_noise_length=0.10,
        measurement_noise_pos=0.50, measurement_noise_aspect=0.05,
        bbox_ema_alpha=0.4,
    )
    base.update(overrides)
    return OmegaConf.create(base)


def test_vehicle_track_manager_output_box_damps_one_bad_frame():
    vtm = VehicleTrackManager(_vehicle_ekf_cfg())
    calib = CameraCalibration.default_front()
    vtm.set_camera_params(calib.K, calib.R_vc, calib.t_vc)

    # Forward-project a known ground point (independent of the code under
    # test, same reference technique as test_rw_coordinates.py) to get a
    # bottom-centre pixel guaranteed to back-project validly, then build
    # boxes of different heights around it.
    p_cam = calib.R_vc @ np.array([15.0, 0.0, 0.0]) + calib.t_vc
    uvw   = calib.K @ p_cam
    u, v_bottom = uvw[0] / uvw[2], uvw[1] / uvw[2]

    good_box = [u - 50.0, v_bottom - 100.0, u + 50.0, v_bottom]   # height 100 px
    bad_box  = [u - 50.0, v_bottom - 50.0,  u + 50.0, v_bottom]   # height 50 px

    for _ in range(3):
        [track] = vtm.update([_det(good_box)])
    assert (track.bbox_xyxy[3] - track.bbox_xyxy[1]) == pytest.approx(100.0, abs=1e-6)

    [track] = vtm.update([_det(bad_box)])
    bad_h = track.bbox_xyxy[3] - track.bbox_xyxy[1]
    assert 50.0 < bad_h < 100.0

