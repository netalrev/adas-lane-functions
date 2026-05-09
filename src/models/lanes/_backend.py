"""
src/models/lanes/_backend.py
=============================
Internal shared backend components for the lane-detection plugin system.

This module is an *internal* package detail (underscore-prefixed) and must
only be imported by plugin files within ``src/models/lanes/``.  External
callers interact with the public API via ``src/models/lanes/__init__.py``.

Classes
-------
LaneDetectionResult
    Typed dataclass returned by every LaneDetectorBase implementation.
LaneDetectorBase
    Abstract interface that all algorithm backends must implement.
VisualPerceptionDetector
    Temporal-persistence wrapper used by IPMPlugin and CLRNetPlugin.
    Derives both the drivable-area path and the host-lane boundaries from
    a single call to the underlying backend.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_HALF_LANE_WIDTH_PX_FRACTION: float = 0.094  # ≈ 180 px / 1920 px for ~3.6 m lane



@dataclass
class LaneDetectionResult:
    """
    Typed output produced by every LaneDetectorBase implementation.

    Pixel coordinates follow the OpenCV convention:
        origin = top-left corner,  x = rightward,  y = downward.
    All ndarray fields have shape (N, 2) and dtype int32.

    Attributes
    ----------
    left_lane : np.ndarray | None
        Ordered (x, y) sequence for the left ego-lane boundary, sampled
        from the bottom of the image toward the horizon.  None if not detected.
    right_lane : np.ndarray | None
        Ordered (x, y) sequence for the right ego-lane boundary.
        None if not detected.
    lane_center : np.ndarray | None
        Point-wise mean of left_lane and right_lane at matching y positions.
        None if either boundary is missing.
    confidence : float
        Detection confidence in [0, 1].  Derived from the fraction of
        sliding windows that contained sufficient inlier pixels.
    source : str
        Backend identifier: "ipm_classical" | "clrnet_onnx" | "none".
    """

    left_lane:        Optional[np.ndarray] = None
    right_lane:       Optional[np.ndarray] = None
    lane_center:      Optional[np.ndarray] = None
    confidence:       float = 0.0
    confidence_left:  float = 0.0   # score of the selected left-side lane
    confidence_right: float = 0.0   # score of the selected right-side lane
    source:           str   = "none"
    # Raw binary mask produced by the detection backend.
    # IPM: the BEV threshold output (dtype uint8, values 0/255).
    # YOLOPv2: the ll_seg_out sigmoid mask at source resolution.
    # None when the backend did not produce a mask (e.g. CLRNet).
    debug_mask: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation (arrays → nested lists)."""
        def _to_list(a: Optional[np.ndarray]):
            return a.tolist() if a is not None else None

        return {
            "left_lane":   _to_list(self.left_lane),
            "right_lane":  _to_list(self.right_lane),
            "lane_center": _to_list(self.lane_center),
            "confidence":  round(self.confidence, 4),
            "source":      self.source,
        }

    def as_legacy_dict(self) -> dict:
        """
        Return the dict format consumed by PerceptionVisualizer.draw_visual_lanes().

        Keys: "left_lane", "right_lane" (ndarray), "source" (str).
        Missing lanes are replaced with an empty (0, 2) array so callers
        never receive None and require no null-guards.
        """
        _empty = np.empty((0, 2), dtype=np.int32)
        return {
            "left_lane":  self.left_lane  if self.left_lane  is not None else _empty,
            "right_lane": self.right_lane if self.right_lane is not None else _empty,
            "source":     self.source,
        }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class LaneDetectorBase(ABC):
    """
    Abstract interface for lane-detection backends.

    Implement ``detect()`` to plug in any model or algorithm.
    All implementations must produce a LaneDetectionResult so that
    backends are interchangeable at runtime without changing any caller.
    """

    @abstractmethod
    def detect(
        self,
        image_bgr: np.ndarray,
        speed_mps: float = 10.0,
    ) -> LaneDetectionResult:
        """
        Run lane detection on a single front-camera BGR frame.

        Parameters
        ----------
        image_bgr : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.
            The implementation must never modify this array in-place.
        speed_mps : float
            Ego-vehicle longitudinal speed in m/s.  Used by gating logic
            to suppress unreliable output at near-stop / intersection speeds.
            Default 10.0 (highway cruise — no speed-based suppression).

        Returns
        -------
        LaneDetectionResult
        """


# ---------------------------------------------------------------------------
# Backend 1 — IPM classical CV (active default)
# ---------------------------------------------------------------------------


class VisualPerceptionDetector:
    """
    Runs one visual inference per frame and derives two independent outputs:

    **Path 3 — Drivable Path**
        A center path representing where the vehicle *can* drive.  Always
        returns a result (confidence = 0 when no markings are found).
        When both lane markings are detected, the center is their midpoint.
        When only one marking is found, the center is offset by half a
        standard lane width.  When nothing is detected, the image-center
        column is returned as a null-confidence fallback.

    **Path 4 — Host Lane**
        Left / right road lane-marking boundaries from visual evidence.
        Only marked ``valid=True`` when *both* markings are detected above
        ``host_lane_confidence_threshold``.  This represents what the
        camera actually sees on the road surface.

    Running both outputs from a single backend inference avoids the cost
    of calling the neural network (or IPM pipeline) twice per frame.

    Parameters
    ----------
    image_width : int
        Expected source image width in pixels.
    image_height : int
        Expected source image height in pixels.
    backend : LaneDetectorBase | None
        Detection backend.  Defaults to IPMLaneDetector.
    host_lane_confidence_threshold : float
        Minimum ``LaneDetectionResult.confidence`` (and both lanes present)
        required to mark host-lane output as ``valid``.
    """

    def __init__(
        self,
        image_width:  int   = 1920,
        image_height: int   = 1280,
        backend: Optional[LaneDetectorBase] = None,
        host_lane_confidence_threshold: float = 0.01,
    ) -> None:
        self.image_width  = image_width
        self.image_height = image_height
        if backend is None:
            raise ValueError(
                "VisualPerceptionDetector requires an explicit backend. "
                "Pass an IPMLaneDetector or CLRNetLaneDetector instance."
            )
        self._backend: LaneDetectorBase = backend
        self.host_conf_thresh = host_lane_confidence_threshold
        self.last_result: Optional[LaneDetectionResult] = None
        # ── Temporal persistence cache ─────────────────────────────────────────────
        # Production ADAS requirement: the host-lane path must not blink off
        # at every intersection or low-contrast frame.  We hold the last valid
        # detection for up to _MAX_PERSIST_FRAMES with linearly-decayed
        # confidence so downstream arbitration can distinguish fresh vs stale.
        self._cached_result: Optional[LaneDetectionResult] = None
        self._cache_age: int = 0
        self._MAX_PERSIST_FRAMES: int = 7
        # Drivable-path cache (center path only) — same persistence window.
        self._cached_drivable_center: Optional[np.ndarray] = None
        self._cached_drivable_conf: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_segment_state(self) -> None:
        """
        Flush all inter-frame temporal caches.

        Call once at the start of each new TFRecord segment.  Resets the
        host-lane persistence window and the drivable-center cache, and
        propagates the reset to the backend if it supports it.
        """
        if hasattr(self._backend, "reset_segment_state"):
            self._backend.reset_segment_state()
        self._cached_result          = None
        self._cache_age              = 0
        self._cached_drivable_center = None
        self._cached_drivable_conf   = 0.0

    def detect(
        self,
        image_bgr: np.ndarray,
        speed_mps: float = 10.0,
    ) -> tuple[dict, dict]:
        """
        Run detection once and return ``(drivable_data, host_lane_data)``.

        Parameters
        ----------
        image_bgr : np.ndarray
            BGR image, shape (H, W, 3), uint8.
        speed_mps : float
            Ego-vehicle speed in m/s forwarded to the backend's speed-based
            gating logic.  Default 10.0 (no gating at highway cruise).

        Returns
        -------
        drivable_data : dict
            "center_path" : np.ndarray (N, 2) pixel coords — always present
            "confidence"  : float
            "source"      : str
        host_lane_data : dict
            "left_lane"   : np.ndarray (N, 2) or empty (0, 2)
            "right_lane"  : np.ndarray (N, 2) or empty (0, 2)
            "confidence"  : float
            "source"      : str
            "valid"       : bool — True only when both markings above threshold
        """
        result = self._backend.detect(image_bgr, speed_mps=speed_mps)
        self.last_result = result
        # Capture the current frame's raw mask BEFORE persistence logic may
        # replace `result` with a cached LaneDetectionResult from a prior
        # frame.  We always want to expose what the detector saw THIS frame
        # so the caller can diagnose extraction failures independently of
        # whether the geometry was promoted from cache.
        fresh_debug_mask: Optional[np.ndarray] = result.debug_mask

        # ── Temporal persistence for host lane ──────────────────────────────
        # A detection is "valid" when BOTH lanes are found above threshold.
        # If the current frame is invalid (intersection, occlusion, faded
        # paint), promote the cached valid result with linearly-decayed
        # confidence for up to _MAX_PERSIST_FRAMES frames.
        is_current_valid = (
            result.left_lane  is not None
            and result.right_lane is not None
            and result.confidence_left  >= self.host_conf_thresh
            and result.confidence_right >= self.host_conf_thresh
        )
        if is_current_valid:
            self._cached_result = result
            self._cache_age = 0
        elif (
            self._cached_result is not None
            and self._cache_age < self._MAX_PERSIST_FRAMES
        ):
            decay  = 1.0 - self._cache_age / self._MAX_PERSIST_FRAMES
            cached = self._cached_result
            result = LaneDetectionResult(
                left_lane        = cached.left_lane,
                right_lane       = cached.right_lane,
                lane_center      = cached.lane_center,
                confidence       = cached.confidence       * decay,
                confidence_left  = cached.confidence_left  * decay,
                confidence_right = cached.confidence_right * decay,
                source           = cached.source + "_persisted",
            )
            self._cache_age += 1
        else:
            self._cache_age = min(self._cache_age + 1, self._MAX_PERSIST_FRAMES + 1)

        # Restore the current frame's mask so _to_host_lane() always exposes
        # what was extracted this frame, not what was cached for the geometry.
        result.debug_mask = fresh_debug_mask

        drivable_data  = self._to_drivable(result, image_bgr.shape)
        host_lane_data = self._to_host_lane(result)

        # ── Temporal persistence for drivable center path ───────────────────
        # When the current drivable confidence is meaningful, update the cache.
        # When it degrades to 0 (no markings found), substitute the last valid
        # center path at half confidence rather than snapping to image-centre.
        if drivable_data["confidence"] > 0.0:
            self._cached_drivable_center = drivable_data["center_path"]
            self._cached_drivable_conf   = drivable_data["confidence"]
        elif self._cached_drivable_center is not None:
            drivable_data = {
                "center_path": self._cached_drivable_center,
                "confidence":  self._cached_drivable_conf * 0.5,
                "source":      drivable_data["source"] + "_persisted",
            }

        return drivable_data, host_lane_data

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_drivable(
        self,
        result: LaneDetectionResult,
        img_shape: tuple[int, ...],
    ) -> dict:
        """Derive the drivable center path from a LaneDetectionResult."""
        h, w = img_shape[:2]
        half_lane_px = int(w * _HALF_LANE_WIDTH_PX_FRACTION)

        if result.lane_center is not None:
            # Both markings detected: center is their midpoint (best quality)
            center = result.lane_center
            d_conf = result.confidence

        elif result.left_lane is not None and result.right_lane is None:
            # Only left marking: shift right by half lane width
            shifted      = result.left_lane.copy()
            shifted[:, 0] = np.clip(shifted[:, 0] + half_lane_px, 0, w - 1)
            center = shifted
            d_conf = result.confidence * 0.5

        elif result.right_lane is not None and result.left_lane is None:
            # Only right marking: shift left by half lane width
            shifted      = result.right_lane.copy()
            shifted[:, 0] = np.clip(shifted[:, 0] - half_lane_px, 0, w - 1)
            center = shifted
            d_conf = result.confidence * 0.5

        else:
            # No markings at all: image-centre column, lower half of frame
            ys     = np.linspace(h - 1, h // 2, 20, dtype=np.int32)
            xs     = np.full_like(ys, w // 2)
            center = np.column_stack([xs, ys])
            d_conf = 0.0

        return {
            "center_path": center,
            "confidence":  float(d_conf),
            "source":      "drivable_" + result.source,
        }

    def _to_host_lane(self, result: LaneDetectionResult) -> dict:
        """Derive host lane boundaries from a LaneDetectionResult."""
        _empty = np.empty((0, 2), dtype=np.int32)
        valid_left  = (
            result.left_lane  is not None
            and result.confidence_left  >= self.host_conf_thresh
        )
        valid_right = (
            result.right_lane is not None
            and result.confidence_right >= self.host_conf_thresh
        )
        return {
            "left_lane":        result.left_lane  if result.left_lane  is not None else _empty,
            "right_lane":       result.right_lane if result.right_lane is not None else _empty,
            "confidence":       float(result.confidence),
            "confidence_left":  float(result.confidence_left),
            "confidence_right": float(result.confidence_right),
            "source":           result.source,
            "valid":            valid_left and valid_right,  # backward compat
            "valid_left":       valid_left,
            "valid_right":      valid_right,
            "debug_mask":       result.debug_mask,
        }
