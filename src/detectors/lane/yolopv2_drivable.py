"""
src/models/lanes/yolopv2_drivable.py
=====================================
Plugin — YOLOPv2 Multi-Task Drivable-Area + Lane-Line Detector (Strategy Pattern).

Wraps ``YOLOPv2DrivableDetector`` in the ``AbstractLaneDetector`` interface.
YOLOPv2 produces both Drivable-Area (Path 3) and Lane-Line (Path 4) outputs
from a **single ONNX forward pass** — the backbone runs exactly once per
frame regardless of how many output keys are requested.

The ``contributes`` config key lets the caller choose which of the two output
pairs this plugin publishes.  Example hybrid config (YOLOPv2 for DA, IPM for LL):

    active_plugins: [kinematic, yolopv2, ipm]
    yolopv2:
      contributes: drivable   # only publish drivable_* keys; LL handled by IPM
    ipm:
      contributes: host_lane  # only override host_* keys from YOLOPv2

Config keys consumed from ``cfg.perception.lane``
----------------------------------------------------
host_lane_confidence_threshold  : float   (default 0.01)
yolopv2.onnx_path               : str
yolopv2.device                  : str     "cpu" | "cuda"
yolopv2.min_drivable_pix        : int     (default 30)
yolopv2.ll_conf_threshold       : float   (default 0.30)
yolopv2.contributes             : str     "drivable" | "host_lane" | "all"  (default "all")

Output keys
-----------
"all"        -> drivable_raw, drivable_path, host_raw, host_lane
"drivable"   -> drivable_raw, drivable_path
"host_lane"  -> host_raw, host_lane

ONNX inference is always executed exactly once per frame, regardless of
the ``contributes`` setting — both heads are derived from the same tensor.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

import cv2

from ._backend import LaneDetectionResult

# ---------------------------------------------------------------------------
# YOLOPv2 backend (moved from src/models/lane_detector.py)
# ---------------------------------------------------------------------------

# EMA alpha for polynomial coefficients (shared with IPMLaneDetector).
_COEFF_EMA_ALPHA: float = 0.25

class YOLOPv2DrivableDetector:
    """
    Drivable-area detector backed by YOLOPv2 ONNX (Path 3).

    YOLOPv2 is a multi-task network trained on BDD100K that jointly predicts
    object detections, a drivable-area segmentation mask, and lane lines in a
    single forward pass.  This class uses **only** the drivable-area head.

    Unlike lane-marking detectors (CLRNet, IPM), this works on wet roads,
    faded markings, construction zones, and any surface the model has seen,
    because it segments drivable road *texture* — not painted lines.

    ONNX weights
    ------------
    Download yolopv2.onnx (~36 MB) from:
        https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.onnx
    Set the path in conf/config.yaml → perception.lane.yolopv2.onnx_path.

    Network I/O
    -----------
    Input  : "images"   — float32 NCHW (1, 3, 384, 640), values in [0, 1]
                          (letterbox-resized, BGR→RGB, divided by 255)
    Output : "da_seg_out" — float32 (1, 2, H_net, W_net)
                          2-class logit map (ch0=background, ch1=drivable)
             "ll_seg_out" — lane-line logit map (unused here)
             "det_out"    — detection boxes       (unused here)

    Center path extraction
    ----------------------
    1. argmax over class dim → binary mask (H_net, W_net)
    2. Remove letterbox padding → resize to source image resolution
    3. For each row in the lower 50 % of the image, find drivable pixels
       and take their column centroid.
    4. Return ordered (x, y) pairs from bottom toward horizon.

    Parameters
    ----------
    onnx_path : str
        Path to the yolopv2.onnx file.
    image_width : int
    image_height : int
    device : str
        "cpu" or "cuda".
    min_drivable_pix : int
        Minimum drivable pixels in a row to include that row's centroid.
        Lower = more sensitive, higher = more conservative.
    """

    _NET_W: int = 640
    _NET_H: int = 384

    def __init__(
        self,
        onnx_path:        str,
        image_width:      int   = 1920,
        image_height:     int   = 1280,
        device:           str   = "cpu",
        min_drivable_pix: int   = 30,
        host_conf_thresh: float = 0.35,
        ll_threshold:     float = 0.30,
        track_half:       int   = 80,
        coeff_ema_alpha:  float = 0.25,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for YOLOPv2DrivableDetector. "
                "Install: pip install onnxruntime   (CPU) or "
                "pip install onnxruntime-gpu         (CUDA)"
            ) from exc

        self.image_width       = image_width
        self.image_height      = image_height
        self.min_drivable_pix  = min_drivable_pix
        self._track_half       = track_half
        self._coeff_ema_alpha  = coeff_ema_alpha

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session    = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        # Locate the drivable-area output by shape: (N, 2, H, W) — 2-class
        # segmentation at network resolution.  This is robust to the output
        # names changing between different ONNX export methods (TorchScript
        # exports lose the original output names from the model code).
        # We also store the lane-line output name for optional future use.
        self._da_output_name  = None
        self._ll_output_name  = None
        for o in self._session.get_outputs():
            # Skip scalar/list outputs (det_out has shape [])
            if len(o.shape) != 4:
                continue
            # Drivable-area head: 2-class logit/sigmoid at net resolution
            if o.shape[1] == 2:
                self._da_output_name = o.name
            # Lane-line head: 1-class sigmoid at net resolution
            elif o.shape[1] == 1:
                self._ll_output_name = o.name

        if self._da_output_name is None:
            raise RuntimeError(
                "YOLOPv2DrivableDetector: could not locate drivable-area output "
                "(expected a 4-D output with 2 channels). "
                "Check that yolopv2.onnx was exported correctly."
            )
        log.info(
            "YOLOPv2DrivableDetector: loaded model  da=%s  ll=%s",
            self._da_output_name, self._ll_output_name,
        )
        # Lane-line head configuration
        self._host_conf_thresh: float = host_conf_thresh
        self._ll_threshold:     float = ll_threshold
        # Temporal lane-line persistence — hold last valid up to N frames
        self._cached_lane_result: Optional[LaneDetectionResult] = None
        self._lane_cache_age: int = 0
        self._LANE_MAX_PERSIST: int = 25

        # EMA coefficient cache — lane-line polynomials only.
        # Drivable-area centre and boundary paths now use the smoothed-polyline
        # approach (no polynomial fitting, no temporal blending in pixel space).
        self._prev_ll_coeffs: dict[str, Optional[np.ndarray]] = {
            "left":  None,
            "right": None,
        }

        # Trapezoidal forward-frustum ROI mask (lazy, rebuilt on size change).
        # Applied via bitwise AND to the raw drivable-area mask before any
        # centroid or boundary computation so that intersection bleed pixels
        # outside the ego-lane corridor are suppressed at the source.
        self._frustum_mask: Optional[np.ndarray] = None
        self._frustum_mask_hw: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flush_ll_cache(self) -> None:
        """
        Flush only the lane-line EMA and persistence caches.

        Called when the ego vehicle speed drops below the minimum threshold
        (e.g. stopped at an intersection) so that crosswalk / stop-line
        geometry does not corrupt the polynomial EMA that will be used once
        the vehicle starts moving again.
        """
        self._cached_lane_result = None
        self._lane_cache_age     = 0
        self._prev_ll_coeffs     = {"left": None, "right": None}

    def reset_segment_state(self) -> None:
        """
        Flush all inter-frame EMA and persistence caches.

        Call once at the start of each new TFRecord segment so that stale
        polynomial coefficients and lane-persistence state do not bleed into
        the first frames of the new recording.
        """
        self._cached_lane_result = None
        self._lane_cache_age     = 0
        self._prev_ll_coeffs     = {"left": None, "right": None}

    def detect_full(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[dict, dict]:
        """
        Single ONNX forward pass returning **(drivable_data, host_lane_data)**.

        Uses YOLOPv2's ``da_seg_out`` (drivable-area segmentation) for Path 3
        and ``ll_seg_out`` (lane-line segmentation) for Path 4 — both produced
        by the same model in one inference call, zero extra latency.

        Lane-line extraction strategy
        ------------------------------
        For each sampled row y (bottom → horizon, 30 steps):
          Left boundary  = rightmost lane pixel left  of image centre.
          Right boundary = leftmost  lane pixel right of image centre.
        A degree-2 polynomial x = f(y) is fitted to each side, plausibility-
        checked, then sampled uniformly and returned as pixel polylines.

        Temporal persistence
        ---------------------
        When the current frame has no valid lane detection (intersection,
        occlusion, faded paint) the last valid result is promoted for up to
        ``_LANE_MAX_PERSIST`` frames with linearly-decayed confidence.

        Returns
        -------
        drivable_data : dict
            ``center_path``, ``left_path``, ``right_path``, ``mask``,
            ``confidence``, ``source`` — same as ``detect()``.
        host_lane_data : dict
            ``left_lane``, ``right_lane``, ``confidence``,
            ``confidence_left``, ``confidence_right``, ``source``,
            ``valid``, ``valid_left``, ``valid_right``.
        """
        src_h, src_w = image_bgr.shape[:2]

        # ── 1. Letterbox pre-process ───────────────────────────────────────────────
        img_lb, _, (dw, dh) = self._letterbox(
            image_bgr, (self._NET_W, self._NET_H)
        )
        tensor = (
            img_lb[:, :, ::-1]
            .transpose(2, 0, 1)
            [np.newaxis]
            .astype(np.float32) / 255.0
        )

        # ── 2. Single inference call ─────────────────────────────────────────────────
        has_ll      = self._ll_output_name is not None
        req_outputs = [self._da_output_name]
        if has_ll:
            req_outputs.append(self._ll_output_name)
        raw_outputs = self._session.run(
            req_outputs, {self._input_name: tensor}
        )
        da_raw = raw_outputs[0]                             # (1, 2, H, W)
        ll_raw = raw_outputs[1] if has_ll else None         # (1, 1, H, W) or None

        # ── 3. Helper: strip letterbox padding and resize to source resolution ──
        pad_h  = int(round(dh))
        pad_w  = int(round(dw))
        net_h, net_w = da_raw.shape[2], da_raw.shape[3]

        def _to_src(net_mask: np.ndarray) -> np.ndarray:
            r0 = pad_h
            r1 = net_h - pad_h if pad_h > 0 else net_h
            c0 = pad_w
            c1 = net_w - pad_w if pad_w > 0 else net_w
            return cv2.resize(
                net_mask[r0:r1, c0:c1],
                (src_w, src_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # ── 4. Drivable-area path (Path 3) ─────────────────────────────────────────
        da_mask      = _to_src((da_raw[0, 1] >= 0.5).astype(np.uint8))
        # Apply the forward-frustum ROI before centroid / boundary extraction
        # so that intersection bleed pixels are removed at the source.
        da_mask      = self._apply_forward_frustum(da_mask)
        center_pts   = self._mask_to_centerline(da_mask, src_h, src_w)
        left_da, right_da = self._mask_to_boundaries(da_mask, src_h, src_w)
        lower        = da_mask[src_h // 2:, :]
        rows_ok      = int((lower.sum(axis=1) >= self.min_drivable_pix).sum())
        da_conf      = float(rows_ok) / max(1, lower.shape[0])
        drivable_data = {
            "center_path": center_pts,
            "left_path":   left_da,
            "right_path":  right_da,
            "mask":        da_mask,
            "confidence":  da_conf,
            "source":      "yolopv2",
        }

        # ── 5. Lane-line host lane (Path 4) ─────────────────────────────────────────
        if ll_raw is not None:
            ll_mask = _to_src((ll_raw[0, 0] >= self._ll_threshold).astype(np.uint8))

            lane_result = self._extract_lane_lines(ll_mask, src_h, src_w)
            lane_result.debug_mask = ll_mask
        else:
            lane_result = LaneDetectionResult(source="yolopv2_ll", confidence=0.0)

        # ── 6. Temporal persistence ───────────────────────────────────────────────────
        is_valid = (
            lane_result.left_lane  is not None
            and lane_result.right_lane is not None
            and lane_result.confidence_left  >= self._host_conf_thresh
            and lane_result.confidence_right >= self._host_conf_thresh
        )
        if is_valid:
            self._cached_lane_result = lane_result
            self._lane_cache_age     = 0
        elif (
            self._cached_lane_result is not None
            and self._lane_cache_age < self._LANE_MAX_PERSIST
        ):
            decay = 1.0 - self._lane_cache_age / self._LANE_MAX_PERSIST
            c = self._cached_lane_result
            lane_result = LaneDetectionResult(
                left_lane        = c.left_lane,
                right_lane       = c.right_lane,
                lane_center      = c.lane_center,
                confidence       = c.confidence       * decay,
                confidence_left  = c.confidence_left  * decay,
                confidence_right = c.confidence_right * decay,
                source           = c.source + "_persisted",
            )
            self._lane_cache_age += 1
        else:
            self._lane_cache_age = min(
                self._lane_cache_age + 1, self._LANE_MAX_PERSIST + 1
            )

        return drivable_data, self._lane_to_host_dict(lane_result)

    def detect(self, image_bgr: np.ndarray) -> dict:
        """
        Run YOLOPv2 drivable-area inference on *image_bgr*.

        Returns
        -------
        dict
            "center_path" : np.ndarray (N, 2) int32  — (x, y) pairs, bottom→top
            "mask"        : np.ndarray (H, W) uint8  — binary drivable mask
            "confidence"  : float  — fraction of lower-half rows with drivable pixels
            "source"      : "yolopv2"
        """
        src_h, src_w = image_bgr.shape[:2]

        # 1. Letterbox resize to (NET_W, NET_H) preserving aspect ratio
        img_lb, _, (dw, dh) = self._letterbox(
            image_bgr, (self._NET_W, self._NET_H)
        )

        # 2. BGR → RGB, HWC → CHW float32, normalise to [0, 1]
        tensor = (
            img_lb[:, :, ::-1]           # BGR → RGB
            .transpose(2, 0, 1)          # HWC → CHW
            [np.newaxis]                 # add batch dim
            .astype(np.float32) / 255.0
        )

        # 3. Inference — request only the drivable-area output to avoid
        #    pulling det_out which is a list (not an ndarray) and causes
        #    AttributeError when calling .shape on it.
        outputs = self._session.run(
            [self._da_output_name],
            {self._input_name: tensor},
        )
        da_raw = outputs[0]   # (1, 2, H_net, W_net), already sigmoid-activated

        # 4. Threshold class-1 (drivable) channel at 0.5 → binary mask.
        #    The exported model already applies Sigmoid, so values are in [0,1].
        da_net = (da_raw[0, 1] >= 0.5).astype(np.uint8)  # (H_net, W_net)

        # 5. Strip letterbox padding, then resize to source resolution.
        #    Padding amounts: dw pixels added left+right, dh top+bottom.
        pad_h = int(round(dh))
        pad_w = int(round(dw))
        net_h, net_w = da_net.shape
        # Clamp slices so they stay valid even when padding is 0
        r0 = pad_h
        r1 = net_h - pad_h if pad_h > 0 else net_h
        c0 = pad_w
        c1 = net_w - pad_w if pad_w > 0 else net_w
        da_cropped = da_net[r0:r1, c0:c1]
        mask = cv2.resize(da_cropped, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        # Apply the forward-frustum ROI before centroid extraction so that
        # intersection bleed pixels are removed at the source.
        mask = self._apply_forward_frustum(mask)

        # 6. Extract center path from drivable mask (bottom → horizon)
        center_pts = self._mask_to_centerline(mask, src_h, src_w)

        # 7. Confidence: fraction of lower-half rows with enough drivable pixels
        lower = mask[src_h // 2:, :]
        rows_ok = int((lower.sum(axis=1) >= self.min_drivable_pix).sum())
        conf    = float(rows_ok) / max(1, lower.shape[0])

        # 8. Extract left/right boundary paths from the mask edges
        left_pts, right_pts = self._mask_to_boundaries(mask, src_h, src_w)

        return {
            "center_path": center_pts,
            "left_path":   left_pts,
            "right_path":  right_pts,
            "mask":        mask,
            "confidence":  conf,
            "source":      "yolopv2",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_lane_lines(
        self,
        mask:  np.ndarray,
        src_h: int,
        src_w: int,
    ) -> LaneDetectionResult:
        """
        Convert a YOLOPv2 lane-line binary mask to ego-lane L/R boundaries.

        Pre-processing — horizontal structure suppression
        -------------------------------------------------
        At intersections the ll_seg_out head fires on crosswalk stripes and
        stop lines (wide horizontal white markings).  These span nearly the
        full image width at the bottom rows and, if not removed, make the
        per-row boundary sampler pick extreme left/right x values → the
        fitted polynomial creates a huge opening parabola that looks like
        an oval when rendered.

        Step A: Erode the full-resolution mask with a wide horizontal kernel
                (60 px).  Any blob that is continuous for < 60 px horizontally
                (i.e. a real lane marking) survives.  Crosswalk stripes that
                are ≥ 60 px wide in the horizontal direction are eroded away.
        Step B: Dilate the detected horizontal blob back to its original extent
                so we can subtract it from the mask cleanly.
        Step C: Open the remainder with a tall narrow kernel (3×25 px) to keep
                only near-vertical structures (lane markings) and discard
                salt-and-pepper noise.

        Per-row sampling with perspective corridor
        ------------------------------------------
        For each sampled row y (bottom → horizon, 30 steps):
          Left boundary  = rightmost clean pixel in the perspective corridor
                           left of image centre.
          Right boundary = leftmost  clean pixel in the perspective corridor
                           right of image centre.

        The corridor half-width tapers linearly from src_w * 0.19 at the
        bottom to src_w * 0.06 at the horizon.  This matches the expected
        pixel distance to a typical lane boundary under a ~2000 px focal-
        length camera and rejects any pixel placed there by a wide horizontal
        marking.

        Polynomial fit and plausibility checks
        ---------------------------------------
        A degree-2 polynomial x = f(y) is fitted to each side after 2.5-sigma
        outlier removal.  After fitting:
          * Side plausibility: mean x must be on the correct image half.
          * Top corridor check: the extrapolated horizon x must lie inside the
            far-field corridor (catches diverging parabola artefacts).
          * Out-of-bounds x values are clipped to [0, src_w-1] rather than
            rejecting the line; the ego-hood causes the mask to end a few rows
            above the image bottom, so bottom extrapolation often overshoots
            slightly but the detection is still valid.
          * Bottom width guard: if right_x - left_x > 60 % of image width
            at the bottom row the pair is from an intersection — discard.
        """
        # ── Step A-C: horizontal structure suppression ─────────────────────
        # Kernel sizes are for the full-resolution mask (src_w ≈ 1920 px).
        # A crosswalk stripe spans 800–1500 px horizontally; a lane marking
        # segment is 5–15 px wide.  The 60-px erosion threshold is well
        # between these two regimes.
        k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
        horiz_blobs = cv2.erode(mask, k_horiz)
        k_dilate    = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 6))
        horiz_cover = cv2.dilate(horiz_blobs, k_dilate)
        clean = cv2.bitwise_and(mask, cv2.bitwise_not(horiz_cover))
        # Keep only near-vertical structures (lane marking segments).
        # A (3, 7) kernel required 7 vertically-consecutive pixels, which
        # erased diagonal lane lines on sharp curves (the marking becomes
        # near-45° so no 7-pixel vertical run exists).  A minimal (3, 3)
        # kernel removes salt-and-pepper noise without destroying diagonals.
        k_vert  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean   = cv2.morphologyEx(clean, cv2.MORPH_OPEN, k_vert)

        # ── Histogram-anchored continuity tracking ───────────────────────────
        # Initialisation strategy: rather than trying to acquire the ego lane
        # on the very first row (which may be a dashed-line gap), sum the
        # morphologically-cleaned mask vertically over the bottom 30 % of the
        # active scan region to produce a column histogram.  The two dominant
        # peaks anywhere in the histogram are the ego lane boundaries — this
        # works even during lane changes or sharp curves where both markings
        # shift to the same side of the image centre.
        #
        # Tracking: every subsequent row is searched inside a ±TRACK_HALF
        # window around the last accepted x.  Rows with no pixels in that
        # window are skipped (gap in a dashed line) without resetting the
        # anchor, so the tracker coasts straight through gaps and can NEVER
        # jump to a bike-lane or tram-track marking further away.
        cx      = src_w // 2
        y_start = src_h - 1
        # Raised from 0.45 to 0.35: on uphill roads the vanishing point
        # migrates toward the upper image half, causing the 0.45 cutoff to
        # truncate valid far-field lane markings that appear above mid-frame.
        y_end   = int(src_h * 0.35)
        n_steps = 70
        step    = max(1, (y_start - y_end) // n_steps)

        # FAR_FRAC is still used by _fit_poly for the horizon corridor check.
        FAR_FRAC   = 0.12
        TRACK_HALF = self._track_half   # per-row tracking window half-width (px); wider for sharp curves

        # ── Histogram initialisation ─────────────────────────────────────────
        # Try increasingly tall histogram zones (bottom 30 %, 60 %, 90 % of
        # the active scan region) until at least one side yields a peak.
        # On hilly roads near-field markings migrate toward the upper portion
        # of the image; a fixed 30 % slice misses them entirely.
        smooth_w = 11
        _kernel  = np.ones(smooth_w, dtype=np.float64) / smooth_w
        peak1:    Optional[int] = None
        peak2:    Optional[int] = None
        for _hist_frac in (0.30, 0.60, 0.90):
            _hist_y0 = y_start - int((y_start - y_end) * _hist_frac)
            _hist    = clean[_hist_y0:y_start + 1, :].sum(axis=0).astype(np.float64)
            _hist    = np.convolve(_hist, _kernel, mode="same")
            _l_hist  = _hist[:cx]
            _r_hist  = _hist[cx:]
            # Nearest-to-center strategy: take the RIGHTMOST significant peak
            # on the left side and the LEFTMOST on the right.  On multi-lane
            # roads the far-side markings (oncoming / bike-lane) accumulate more
            # pixels and would win a raw argmax, anchoring the tracker to the
            # wrong line.  "Significant" = >= 15 % of that side's max after
            # smoothing (absolute floor of 2.0 suppresses thermal noise).
            if _l_hist.max() > 0:
                _l_floor = max(2.0, _l_hist.max() * 0.15)
                _l_ids   = np.where(_l_hist >= _l_floor)[0]
                _p1      = int(_l_ids[-1]) if len(_l_ids) > 0 else None  # rightmost
            else:
                _p1 = None
            if _r_hist.max() > 0:
                _r_floor = max(2.0, _r_hist.max() * 0.15)
                _r_ids   = np.where(_r_hist >= _r_floor)[0]
                _p2      = (int(_r_ids[0]) + cx) if len(_r_ids) > 0 else None  # leftmost
            else:
                _p2 = None
            if _p1 is not None or _p2 is not None:
                peak1, peak2 = _p1, _p2
                break

        prev_left_x:  Optional[int] = peak1
        prev_right_x: Optional[int] = peak2

        left_raw:  list[list[int]] = []
        right_raw: list[list[int]] = []

        # Dynamic search windows: start tight, expand during dashed-line gaps
        # so the tracker can re-acquire the next dash even after the road has
        # curved laterally.  Reset to TRACK_HALF on every successful hit.
        _max_window = int(src_w * 0.20)
        left_window  = TRACK_HALF
        right_window = TRACK_HALF

        for y in range(y_start, y_end, -step):
            # Band sampling: project the entire step-height slab into a single
            # 1-D column presence array.  A single-pixel row check misses thin
            # diagonal lane lines that fall in the gap between consecutive y
            # samples; taking the column-wise max over the full band guarantees
            # that ANY white pixel in the interval [y-step, y) is captured.
            y_lo = max(0, y - step)
            band = clean[y_lo:y, :]
            row  = np.where(band.max(axis=0) > 0)[0]
            if row.size == 0:
                continue

            # ---- Left side ----
            if prev_left_x is not None:
                lc = row[
                    (row >= max(0, prev_left_x - left_window)) &
                    (row <= min(src_w - 1, prev_left_x + left_window))
                ]
                if lc.size > 0:
                    best_l      = int(round(lc.mean()))  # centroid of window pixels
                    left_raw.append([best_l, y])
                    prev_left_x = best_l
                    left_window = TRACK_HALF             # re-anchor: shrink back
                else:
                    # Dashed-line gap: road may have curved during the gap, so
                    # widen the net for the next row to re-acquire the dash.
                    left_window = min(left_window + 15, _max_window)

            # ---- Right side ----
            if prev_right_x is not None:
                rc = row[
                    (row >= max(0, prev_right_x - right_window)) &
                    (row <= min(src_w - 1, prev_right_x + right_window))
                ]
                if rc.size > 0:
                    best_r       = int(round(rc.mean()))  # centroid of window pixels
                    right_raw.append([best_r, y])
                    prev_right_x = best_r
                    right_window = TRACK_HALF            # re-anchor: shrink back
                else:
                    # Dashed-line gap: expand window to chase lateral road shift.
                    right_window = min(right_window + 15, _max_window)

        # Fair confidence normalization for short-support lanes.
        # On sharp curves a lane physically exits the image frame before
        # accumulating n_steps points, so dividing by n_steps unfairly
        # penalizes valid detections.  Normalizing to 40 % of n_steps means
        # a lane that spans 40 % of the available vertical range scores 1.0.
        lconf = min(1.0, len(left_raw)  / (n_steps * 0.40))
        rconf = min(1.0, len(right_raw) / (n_steps * 0.40))

        def _fit_poly(pts: list[list[int]], side: str) -> Optional[np.ndarray]:
            # ARCHITECTURAL NOTE — Smoothed Polyline, no polynomial extrapolation.
            # np.polyfit / np.polyval / EMA blending were removed because:
            #   (a) A global polynomial cannot model S-curves or sharp exits;
            #       it extrapolates blindly to y_end when far-field points are
            #       missing (dashed gaps, occlusion), producing impossible shapes.
            #   (b) Image-space coefficient EMA caused "ghosting": pixel coords
            #       shift violently during ego-rotation so 75 % history weight
            #       dragged the line behind the real mask.
            # This function builds a polyline strictly from the tracker's own
            # observed points, smooths only the x-jitter with a boxcar filter,
            # and draws ONLY over the vertical range where pixels were detected.

            # The convolution window must be defined first: the minimum-points
            # guard requires at least _win inputs so that mode='valid' with
            # kernel length _win produces a non-empty output (length N - _win + 1)
            # that aligns exactly with ys_raw[_trim:-_trim] (length N - 2*_trim).
            # With N < _win, np.convolve(mode='valid') returns a shorter-than-
            # expected array whose size does NOT match the ys trim, causing
            # np.column_stack to raise a shape mismatch ValueError.
            _win  = 5
            _trim = _win // 2   # = 2 samples removed from each end

            if len(pts) < _win:
                return None

            # Sort by y descending: bottom of image first (large y → small y).
            arr = np.array(pts, dtype=np.float64)
            arr = arr[np.argsort(-arr[:, 1])]

            ys_raw = arr[:, 1]
            xs_raw = arr[:, 0]

            # Side plausibility: mean x must be on the correct image half.
            if side == "left"  and xs_raw.mean() >= src_w * 0.70:
                return None
            if side == "right" and xs_raw.mean() <= src_w * 0.30:
                return None

            # 1-D moving-average (window = 5, mode='valid') to suppress
            # single-pixel jitter without distorting the curve shape.
            # mode='valid' shrinks the output by (_win - 1) = 4 samples, so
            # trim the corresponding _trim boundary rows from ys to keep arrays
            # aligned.  The boundary rows are the least reliable tracker
            # samples (first acquisition and last far-field point), so
            # discarding them also improves geometric quality.
            xs_smooth = np.convolve(xs_raw, np.ones(_win) / _win, mode="valid")
            ys_out    = ys_raw[_trim : len(ys_raw) - _trim]

            xs_out = xs_smooth.clip(0.0, float(src_w - 1))

            return np.column_stack([xs_out.astype(np.int32), ys_out.astype(np.int32)])

        left_img  = _fit_poly(left_raw,  "left")
        right_img = _fit_poly(right_raw, "right")

        # Zero out tracker-coverage confidence for any side whose polynomial
        # fit failed.  A non-zero confidence with no geometry is misleading
        # and causes host_lane to report high confidence_left/right while
        # publishing an empty point array.
        if left_img  is None: lconf = 0.0
        if right_img is None: rconf = 0.0

        # Crossing guard + bottom width guard
        if left_img is not None and right_img is not None:
            l_bot = left_img[np.argmax(left_img[:, 1]),  0]
            r_bot = right_img[np.argmax(right_img[:, 1]), 0]
            if l_bot >= r_bot:
                log.debug("YOLOPv2 ll: lanes cross at bottom — discarding pair")
                left_img = right_img = None
            elif (r_bot - l_bot) > src_w * 0.95:
                # Lanes separated by more than 95 % of image width at the bottom
                # are intersection artefacts.  Waymo's wide FOV places near-field
                # ego-lane lines at the extreme image edges (often > 80 % apart),
                # so the previous 60 % threshold was discarding valid detections.
                log.debug(
                    "YOLOPv2 ll: bottom width %d px > %.0f — intersection artefact, discarding",
                    r_bot - l_bot, src_w * 0.95,
                )
                left_img = right_img = None
            else:
                # ── Convergence guard ────────────────────────────────────────
                # Under perspective projection, lane markings MUST narrow
                # toward the vanishing point.  A pair whose top (horizon) width
                # exceeds its bottom width is diverging — a physically impossible
                # fit that is always an artefact of noise or scattered activations.
                l_top = int(left_img[-1, 0])   # xs[−1] = horizon end (min y)
                r_top = int(right_img[-1, 0])
                # On tight curves, inner-arc geometry can make the fitted
                # polynomial appear slightly wider at the horizon than at
                # the bottom — this is physically valid, not an artefact.
                # Only discard truly extreme divergence (>1.8× bottom width).
                if r_top - l_top > (r_bot - l_bot) * 1.8:
                    log.debug(
                        "YOLOPv2 ll: lanes diverge excessively toward horizon "
                        "(top_w=%d  bot_w=%.0f) — discarding",
                        r_top - l_top, r_bot - l_bot,
                    )
                    left_img = right_img = None

        center: Optional[np.ndarray] = None
        if left_img is not None and right_img is not None:
            # The smoothed-polyline approach returns exactly as many rows as
            # were observed per side, so left_img and right_img can have
            # different lengths.  Interpolate both onto a shared y-grid that
            # spans only their overlapping vertical range so the center is
            # always backed by real observations on both sides.
            l_ys = left_img[:,  1].astype(np.float64)   # already sorted desc
            r_ys = right_img[:, 1].astype(np.float64)
            y_lo = max(l_ys.min(), r_ys.min())           # highest row (smallest y)
            y_hi = min(l_ys.max(), r_ys.max())           # lowest  row (largest  y)
            if y_hi > y_lo:
                # np.interp expects xp ascending; our ys are descending, so flip.
                n_c  = min(len(left_img), len(right_img))
                c_ys = np.linspace(y_hi, y_lo, n_c)
                l_xs = np.interp(c_ys, l_ys[::-1], left_img[:,  0][::-1].astype(np.float64))
                r_xs = np.interp(c_ys, r_ys[::-1], right_img[:, 0][::-1].astype(np.float64))
                c_xs = (l_xs + r_xs) / 2.0
                center = np.column_stack([c_xs.astype(np.int32), c_ys.astype(np.int32)])

        return LaneDetectionResult(
            left_lane        = left_img,
            right_lane       = right_img,
            lane_center      = center,
            confidence       = (lconf + rconf) / 2.0,
            confidence_left  = lconf,
            confidence_right = rconf,
            source           = "yolopv2_ll",
        )

    def _lane_to_host_dict(self, result: LaneDetectionResult) -> dict:
        """Convert a LaneDetectionResult to the host_lane_data dict format."""
        _empty = np.empty((0, 2), dtype=np.int32)
        vl = (
            result.left_lane  is not None
            and result.confidence_left  >= self._host_conf_thresh
        )
        vr = (
            result.right_lane is not None
            and result.confidence_right >= self._host_conf_thresh
        )
        return {
            "left_lane":        result.left_lane  if result.left_lane  is not None else _empty,
            "right_lane":       result.right_lane if result.right_lane is not None else _empty,
            "confidence":       float(result.confidence),
            "confidence_left":  float(result.confidence_left),
            "confidence_right": float(result.confidence_right),
            "source":           result.source,
            "valid":            vl and vr,
            "valid_left":       vl,
            "valid_right":      vr,
            "debug_mask":       result.debug_mask,
        }

    def _mask_to_boundaries(
        self,
        mask: np.ndarray,
        h:    int,
        w:    int,
    ) -> tuple:
        """
        Return left and right ego-lane boundary paths by offsetting a smoothed
        centre polyline by a perspective-tapered half-width.

        np.polyfit / EMA blending were removed: on sharp curves the polynomial
        extrapolates blindly and 2D-pixel EMA causes ghosting lag.  The centre
        is now derived directly from the per-row mask centroids, smoothed with
        a 1-D boxcar filter, and drawn only over the observed vertical range.

        Half-width tapers linearly from w*0.09 (near-field) to w*0.03
        (far-field) using the ACTUAL observed y positions.

        Returns
        -------
        (left_pts, right_pts) : each np.ndarray (N, 2) int32, or (None, None)
        """
        y_start = h - 1
        y_end   = int(h * 0.45)
        n_steps = 30
        x_lo    = int(w * 0.15)
        x_hi    = int(w * 0.85)
        step    = max(1, (y_start - y_end) // n_steps)

        # Collect per-row drivable centroids (same band as _mask_to_centerline).
        raw_ctr: list[list[int]] = []
        for y in range(y_start, y_end, -step):
            row_xs = np.where(mask[y, x_lo:x_hi] > 0)[0]
            if len(row_xs) >= self.min_drivable_pix:
                raw_ctr.append([int(row_xs.mean()) + x_lo, y])

        if len(raw_ctr) < 4:
            return None, None

        # Sort bottom-to-top (descending y), then apply a 1-D moving average
        # (window=5, mode='valid') to suppress centroid jitter.  mode='valid'
        # shortens the output by 4 samples; trim ys to match.
        arr    = np.array(raw_ctr, dtype=np.float64)
        arr    = arr[np.argsort(-arr[:, 1])]
        xs_raw = arr[:, 0]
        ys_raw = arr[:, 1]

        _win   = 5
        _trim  = _win // 2
        xs_ctr = np.convolve(xs_raw, np.ones(_win) / _win, mode="valid")
        ys_ctr = ys_raw[_trim : len(ys_raw) - _trim]

        # Perspective-tapered half-width computed on the ACTUAL observed ys.
        # frac = 1 at the bottom row (near-field), 0 at the horizon.
        frac    = (ys_ctr - y_end) / max(1, y_start - y_end)
        half_px = w * 0.03 + (w * 0.09 - w * 0.03) * frac

        ys_out   = ys_ctr.astype(np.int32)
        xs_left  = (xs_ctr - half_px).clip(0, w - 1).astype(np.int32)
        xs_right = (xs_ctr + half_px).clip(0, w - 1).astype(np.int32)

        return np.column_stack([xs_left, ys_out]), np.column_stack([xs_right, ys_out])

    def _mask_to_centerline(
        self,
        mask:  np.ndarray,
        h:     int,
        w:     int,
    ) -> np.ndarray:
        """
        Convert a binary drivable-area mask to an ordered (x, y) center path.

        Samples 30 rows uniformly between the bottom of the image and the
        estimated horizon (top 45 % is sky / far field and is skipped).

        Two-stage approach:
          1. Search for drivable pixels only within the ego-lane band
             (centre ±35 % of frame width) so that adjacent lanes and
             oncoming traffic lanes cannot pull the centroid sideways.
          2. Fit a degree-2 polynomial x = f(y) through the raw per-row
             centroids and resample it uniformly.  This eliminates
             row-by-row snaking caused by paint markings or mask noise.

        Falls back to the image-centre column when no drivable pixels are
        found (guarantees a non-None return, confidence will be 0).
        """
        y_start = h - 1             # sample from the very bottom row
        y_end   = int(h * 0.45)    # horizon cut-off
        n_steps = 30
        step    = max(1, (y_start - y_end) // n_steps)

        # --- Stage 1: collect raw per-row centroids ---
        # Restrict x search to ego-lane band so adjacent lanes are ignored.
        x_lo = int(w * 0.15)
        x_hi = int(w * 0.85)

        raw_pts: list[list[int]] = []
        for y in range(y_start, y_end, -step):
            row_xs = np.where(mask[y, x_lo:x_hi] > 0)[0]
            if len(row_xs) >= self.min_drivable_pix:
                raw_pts.append([int(row_xs.mean()) + x_lo, y])

        if len(raw_pts) < 4:
            # Fallback: straight-ahead centre column, confidence = 0.
            ys = np.linspace(h - 1, h // 2, 20, dtype=np.int32)
            return np.column_stack([np.full_like(ys, w // 2), ys])

        # --- Stage 2: smoothed polyline — no polynomial, no EMA ---
        # np.polyfit / EMA blending removed: polynomial extrapolation diverges
        # on sharp curves and 2D-pixel EMA causes ghosting lag during
        # ego-rotation.  A 1-D boxcar filter suppresses centroid jitter while
        # faithfully following the mask geometry frame-by-frame.
        raw_arr = np.array(raw_pts, dtype=np.float64)
        raw_arr = raw_arr[np.argsort(-raw_arr[:, 1])]  # sort bottom-to-top
        xs_raw  = raw_arr[:, 0]
        ys_raw  = raw_arr[:, 1]

        # mode='valid' shortens output by (win-1)=4; trim ys to match.
        _win      = 5
        _trim     = _win // 2
        xs_smooth = np.convolve(xs_raw, np.ones(_win) / _win, mode="valid").clip(0, w - 1)
        ys_smooth = ys_raw[_trim : len(ys_raw) - _trim]

        return np.column_stack([xs_smooth.astype(np.int32), ys_smooth.astype(np.int32)])

    def _apply_forward_frustum(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply a static trapezoidal Region of Interest (ROI) mask to *mask*
        using a bitwise AND, suppressing drivable-area pixels that lie
        outside the ego-lane forward frustum.

        Purpose
        -------
        At T-junctions and intersections the YOLOPv2 drivable-area head
        correctly marks the cross-road as drivable, but those lateral pixels
        lie far outside the vehicle's forward path.  When _mask_to_centerline
        computes per-row centroids the intersection spill pulls the centre
        path sharply sideways, creating a dangerous steering artefact.

        The ROI trapezoid is calibrated to the perspective geometry of a
        typical forward-facing camera:
          - Bottom row   : x spans [10 %, 90 %] of image width
          - Horizon row  : x spans [30 %, 70 %] of image width
        The taper matches the expected angular extent of a ~3.5 m ego lane
        at 0 m (near field) through ~50 m (horizon) and excludes the lateral
        roads that diverge left/right at intersections.

        The mask is rebuilt lazily whenever the image dimensions change;
        otherwise the cached version is reused at zero allocation cost.

        Parameters
        ----------
        mask : np.ndarray (H, W) uint8
            Binary drivable-area mask at source resolution.

        Returns
        -------
        np.ndarray (H, W) uint8
            Masked result with lateral intersection bleed removed.
        """
        h, w = mask.shape[:2]
        if (h, w) != self._frustum_mask_hw or self._frustum_mask is None:
            # Trapezoid corners (image coordinates, clockwise):
            #   bottom-left  → bottom-right  (wide, near field)
            #   top-right    → top-left      (narrow, horizon)
            pts = np.array([
                [int(w * 0.10), h - 1         ],   # bottom-left
                [int(w * 0.90), h - 1         ],   # bottom-right
                [int(w * 0.70), int(h * 0.45) ],   # top-right
                [int(w * 0.30), int(h * 0.45) ],   # top-left
            ], dtype=np.int32)
            fm = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(fm, pts, 1)
            self._frustum_mask    = fm
            self._frustum_mask_hw = (h, w)
        return cv2.bitwise_and(mask, self._frustum_mask)

    @staticmethod
    def _letterbox(
        img:       np.ndarray,
        new_shape: tuple,
        color:     tuple = (114, 114, 114),
    ) -> tuple:
        """
        Resize *img* to *new_shape* (W, H) with unchanged aspect ratio,
        padding the remainder with constant grey.

        Returns
        -------
        (img_padded, ratio, (dw, dh))
            img_padded : resized+padded image (NET_H, NET_W, 3)
            ratio      : scale factor applied to both dimensions
            (dw, dh)   : total padding added in each axis (pixels)
        """
        h, w    = img.shape[:2]
        new_w, new_h = new_shape
        ratio   = min(new_w / w, new_h / h)
        rw, rh  = int(round(w * ratio)), int(round(h * ratio))
        dw      = (new_w - rw) / 2
        dh      = (new_h - rh) / 2

        resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        top     = int(round(dh - 0.1))
        bottom  = int(round(dh + 0.1))
        left    = int(round(dw - 0.1))
        right   = int(round(dw + 0.1))
        padded  = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color,
        )
        return padded, ratio, (dw, dh)

# Standard lane half-width used to offset a single detected marking to the
# drivable centre.  3.65 m is the US/EU standard lane width; we use half.


from .base       import AbstractLaneDetector, VehicleState
from .visual_dp  import DrivablePathStrategy
from .visual_host import HostLaneStrategy

log = logging.getLogger(__name__)


class YOLOPv2Plugin(AbstractLaneDetector):
    """
    Adapter: runs a single YOLOPv2 ONNX inference pass and publishes the
    requested subset of drivable-area and host-lane outputs.

    ONNX single-pass guarantee
    --------------------------
    ``process()`` calls ``detect_full()`` exactly once.  Both the drivable-
    area and lane-line tensors are extracted from that one forward pass.
    No second call is made even in "all" mode.

    Disabled state
    --------------
    When ``onnx_path`` is empty or the model fails to load, the plugin
    sets ``self._detector = None`` and ``process()`` returns ``{}``
    gracefully without crashing the pipeline.
    """

    def __init__(
        self,
        cfg:          DictConfig,
        image_width:  int = 1920,
        image_height: int = 1280,
    ) -> None:
        lane_cfg          = cfg.perception.lane
        yolopv2_cfg       = getattr(lane_cfg, "yolopv2", None)
        host_conf         = float(getattr(lane_cfg, "host_lane_confidence_threshold", 0.01))
        self._contributes   = "all"
        self._min_speed_mps = float(getattr(yolopv2_cfg, "min_speed_mps", 3.0)) if yolopv2_cfg is not None else 3.0
        self._detector: Optional[YOLOPv2DrivableDetector] = None

        if yolopv2_cfg is None:
            log.warning("YOLOPv2Plugin: no yolopv2 config section found — plugin disabled.")
            return

        self._contributes = str(getattr(yolopv2_cfg, "contributes", "all"))
        onnx_path         = str(getattr(yolopv2_cfg, "onnx_path", ""))

        if not onnx_path:
            log.warning("YOLOPv2Plugin: yolopv2.onnx_path not set — plugin disabled.")
            return

        try:
            self._detector = YOLOPv2DrivableDetector(
                onnx_path        = onnx_path,
                image_width      = image_width,
                image_height     = image_height,
                device           = str(getattr(yolopv2_cfg, "device", "cpu")),
                min_drivable_pix = int(getattr(yolopv2_cfg, "min_drivable_pix", 30)),
                host_conf_thresh = host_conf,
                ll_threshold     = float(getattr(yolopv2_cfg, "ll_conf_threshold", 0.30)),
                track_half       = int(getattr(yolopv2_cfg, "track_half", 80)),
                coeff_ema_alpha  = float(getattr(yolopv2_cfg, "coeff_ema_alpha", 0.25)),
            )
            log.info(
                "YOLOPv2Plugin: ONNX loaded  ll_thresh=%.2f  host_conf=%.2f  contributes=%s",
                float(getattr(yolopv2_cfg, "ll_conf_threshold", 0.30)),
                host_conf,
                self._contributes,
            )
        except Exception as exc:  # noqa: BLE001 — propagate to warning, not crash
            log.error("YOLOPv2Plugin: failed to load ONNX model: %s", exc)
            self._detector = None

    def process(
        self,
        frame_bgr:     Optional[np.ndarray],
        vehicle_state: VehicleState,
    ) -> dict:
        if frame_bgr is None or self._detector is None:
            return {}

        # Speed gate: at intersections / stops the LL head picks up crosswalk
        # stripes as lane lines.  Below the minimum speed we flush the lane-line
        # EMA cache (so garbage geometry does not bleed into the next road
        # section) and publish nothing for the host lane.
        below_speed = vehicle_state.speed_mps < self._min_speed_mps
        if below_speed and self._contributes in ("host_lane", "all"):
            self._detector.flush_ll_cache()

        # Single ONNX forward pass — both drivable-area and lane-line
        # tensors are extracted from the same network output.
        drivable_raw, host_raw = self._detector.detect_full(frame_bgr)

        result: dict = {}
        if self._contributes in ("drivable", "all"):
            result["drivable_raw"]  = drivable_raw
            result["drivable_path"] = DrivablePathStrategy.package(drivable_raw)
        if self._contributes in ("host_lane", "all") and not below_speed:
            result["host_raw"]  = host_raw
            result["host_lane"] = HostLaneStrategy.package(host_raw)
        return result

    def reset(self) -> None:
        if self._detector is not None:
            self._detector.reset_segment_state()
