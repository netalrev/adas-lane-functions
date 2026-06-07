"""
src/models/lanes/ipm_classical.py
==================================
Plugin — IPM Classical BEV Lane Detector (Strategy Pattern).

Wraps ``VisualPerceptionDetector`` (with an ``IPMLaneDetector`` backend) in
the ``AbstractLaneDetector`` interface so the ``LaneManager`` can instantiate
it from the plugin registry without any hardcoded routing.

The ``contributes`` config key controls which output keys this plugin
publishes.  Setting ``contributes: host_lane`` in ``conf/model/lane.yaml``
lets a YOLOPv2 plugin upstream own the drivable-area keys while IPM only
overrides the host-lane output — the standard hybrid mode.

Config keys consumed from ``cfg.perception.lane``
----------------------------------------------------
host_lane_confidence_threshold : float   (default 0.01)
ipm.n_windows                  : int
ipm.win_margin                 : int
ipm.win_min_pix                : int
ipm.min_lane_pix               : int
ipm.n_sample                   : int
ipm.roi_top_y                  : float
ipm.roi_top_x_half             : float
ipm.roi_bottom_margin          : float
ipm.contributes                : str     "drivable" | "host_lane" | "all"  (default "all")

Output keys
-----------
"all"        -> drivable_raw, drivable_path, host_raw, host_lane
"drivable"   -> drivable_raw, drivable_path
"host_lane"  -> host_raw, host_lane
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

import cv2

from ._backend import LaneDetectionResult, LaneDetectorBase, VisualPerceptionDetector

# ---------------------------------------------------------------------------
# IPM constants (moved from src/models/lane_detector.py)
# ---------------------------------------------------------------------------

# EMA alpha for polynomial coefficients — same value as in YOLOPv2 detector.
_COEFF_EMA_ALPHA: float = 0.25

_BEV_W: int = 640
_BEV_H: int = 640

# Sliding-window hyperparameters
_N_WINDOWS:   int = 9    # windows per lane per frame
_WIN_MARGIN:  int = 60   # half-width of each window (BEV pixels)
_WIN_MIN_PIX: int = 40   # min inliers in a window to re-centre the next

# Polynomial degree for the lane curve  x = f(y)
_POLY_DEG: int = 2

# Number of sample points along the fitted polynomial (output resolution)
_N_SAMPLE: int = 30

# Minimum total inlier pixels required to report a lane as detected
_MIN_LANE_PIX: int = 500

# ROI trapezoid corners in normalised [0, 1] coordinates.
# Tuned for the Waymo front camera (1920 × 1280, fx ≈ fy ≈ 2002).
# Top of ROI sits just below the estimated horizon (~62 % from top);
# bottom spans nearly the full image width to capture the near field.
_ROI_SRC_NORM = np.float32([
    [0.42, 0.62],   # top-left
    [0.58, 0.62],   # top-right
    [0.95, 1.00],   # bottom-right
    [0.05, 1.00],   # bottom-left
])

# Destination corners mapped into the BEV canvas (normalised [0, 1])
_ROI_DST_NORM = np.float32([
    [0.25, 0.00],   # top-left
    [0.75, 0.00],   # top-right
    [0.75, 1.00],   # bottom-right
    [0.25, 1.00],   # bottom-left
])

# HLS colour ranges for lane-marking extraction (OpenCV H: 0–180)
# White: tightened saturation upper-bound (S ≤ 35 instead of 50) to reject
# sun-lit road surface patches that are bright but slightly saturated, while
# still capturing true white paint (near-zero saturation).
_WHITE_HLS_LO  = np.array([  0, 185,   0], dtype=np.uint8)
_WHITE_HLS_HI  = np.array([180, 255,  35], dtype=np.uint8)
# Yellow: broadened Hue (10–45), Lightness (60–255), and Saturation (60–255)
# to capture yellow paint in three failure modes:
#   • Shadow / overcast: saturation drops to ~60, lightness falls to ~60.
#   • Bright sun / overexposure: lightness climbs toward 255, saturation drops.
#   • Aged / faded paint: hue drifts slightly toward orange (Hue → 10) or
#     toward green-yellow (Hue → 45).
_YELLOW_HLS_LO = np.array([ 10,  60,  60], dtype=np.uint8)
_YELLOW_HLS_HI = np.array([ 45, 255, 255], dtype=np.uint8)

# Horizontal-structure suppression kernel (stop lines, crosswalks, text).
# We erode the binary BEV mask with a wide horizontal kernel to remove
# continuous horizontal blobs, then restore vertically-running structures
# with a tall vertical kernel.  Kernel widths are in BEV pixels.
_HORIZ_SUPPRESS_W: int = 40   # must be < lane stripe width after BEV warp
_HORIZ_SUPPRESS_H: int = 1
_VERT_RESTORE_W:   int = 3
_VERT_RESTORE_H:   int = 12

# Fraction of the BEV bottom to clip before building the histogram.
# Raised from 0.15 → 0.25 to also skip near-field curb reflections and
# building-edge features that nighttime headlights illuminate in the
# lowest portion of the BEV warp (~0–6 m ahead of the vehicle).
_HISTOGRAM_BOTTOM_CLIP: float = 0.25   # skip bottom 25 % of BEV rows

# Speed below which the soft speed gate is evaluated in IPMLaneDetector.
# 2.0 m/s ≈ 7.2 km/h — near-stop urban / intersection regime.
_SPEED_GATE_MPS: float = 2.0

# Ego-lane lateral corridor (BEV normalised x).
# Histogram search is restricted to these x-bands so parallel roads and
# wide intersections cannot steal the argmax from the ego lane.
_HIST_LEFT_BAND  = (0.10, 0.48)   # left-lane  search band  (norm BEV x)
_HIST_RIGHT_BAND = (0.52, 0.90)   # right-lane search band  (norm BEV x)

# Minimum fraction of inlier pixels in a sliding window that must be
# roughly vertically oriented (|dx/dy| < tan(50°)) to count that window
# as a quality win.  Horizontal inliers from stop-line bleed-through are
# penalised but still collected so the polynomial has enough support.
_WIN_QUALITY_ANGLE_TAN: float = 1.19   # tan(50°)


# ---------------------------------------------------------------------------
# LaneDetectionResult
# ---------------------------------------------------------------------------


class IPMLaneDetector(LaneDetectorBase):
    """
    Classical lane detector using Inverse Perspective Mapping (BEV).

    Pipeline
    --------
    1. Warp the camera frame to a bird's-eye-view (BEV) canvas via a fixed
       perspective homography tuned to the Waymo front camera geometry.
    2. Extract lane-marking pixels with an HLS colour mask (white + yellow)
       combined with a Sobel-x gradient mask.
    3. Find the left and right lane base x-positions from a histogram of
       the lower half of the thresholded BEV image.
    4. Trace each lane upward with a sliding window, collecting inlier pixels.
    5. Fit a 2nd-degree polynomial  x = f(y)  to each inlier set.
    6. Sample the polynomial at ``n_sample`` evenly-spaced y values and
       back-project the BEV coordinates to the original image plane via
       the inverse homography.
    7. Score confidence from the fraction of filled sliding windows.

    Parameters
    ----------
    image_width : int
        Expected input image width in pixels.
    image_height : int
        Expected input image height in pixels.
    n_sample : int
        Number of points to sample along each fitted polynomial curve.
    n_windows : int
        Number of sliding windows per lane.
    win_margin : int
        Half-width of each sliding window in BEV pixels.
    win_min_pix : int
        Minimum inlier pixels in a window to re-centre the next window.
    """

    def __init__(
        self,
        image_width:        int   = 1920,
        image_height:       int   = 1280,
        n_sample:           int   = _N_SAMPLE,
        n_windows:          int   = _N_WINDOWS,
        win_margin:         int   = _WIN_MARGIN,
        win_min_pix:        int   = _WIN_MIN_PIX,
        min_lane_pix:       int   = _MIN_LANE_PIX,
        roi_top_y:          float = 0.55,   # lowered from 0.62 → sees farther ahead
        roi_top_x_half:     float = 0.08,
        roi_bottom_margin:  float = 0.05,
    ) -> None:
        self.image_width  = image_width
        self.image_height = image_height
        self.n_sample     = n_sample
        self.n_windows    = n_windows
        self.win_margin   = win_margin
        self.win_min_pix  = win_min_pix
        self.min_lane_pix = min_lane_pix

        # Build the ROI trapezoid from the three intuitive parameters:
        #   roi_top_y          — horizon cut-off (lower = farther ego lane)
        #   roi_top_x_half     — half-width of the top edge around image centre
        #   roi_bottom_margin  — left/right margin at image bottom
        roi_src_norm = np.float32([
            [0.5 - roi_top_x_half,    roi_top_y],   # top-left
            [0.5 + roi_top_x_half,    roi_top_y],   # top-right
            [1.0 - roi_bottom_margin, 1.00      ],   # bottom-right
            [      roi_bottom_margin, 1.00      ],   # bottom-left
        ])

        src = roi_src_norm * np.array([image_width,  image_height], dtype=np.float32)
        dst = _ROI_DST_NORM * np.array([_BEV_W, _BEV_H],            dtype=np.float32)
        self._M     = cv2.getPerspectiveTransform(src, dst)   # image → BEV
        self._M_inv = cv2.getPerspectiveTransform(dst, src)   # BEV   → image

        # EMA coefficient caches for left and right polynomial fits.
        # Each entry holds the smoothed degree-2 coefficient array from the
        # previous frame.  None = no prior history (first frame).
        self._prev_coeffs: dict[str, Optional[np.ndarray]] = {
            "left":  None,
            "right": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        image_bgr: np.ndarray,
        speed_mps: float = 10.0,
    ) -> LaneDetectionResult:
        """
        Run the full IPM detection pipeline on *image_bgr*.

        Parameters
        ----------
        image_bgr : np.ndarray
            BGR image, shape (H, W, 3), dtype uint8.
        speed_mps : float
            Ego-vehicle speed in m/s.  When the vehicle is below
            ``_SPEED_GATE_MPS`` (near-stop / urban intersection), the
            detector applies a stricter initial quality gate before
            committing to a detection.
        """
        bev    = self._warp_to_bev(image_bgr)
        binary = self._threshold(bev)

        # ── Intersection / scene-quality gate ────────────────────────────────
        # Two complementary conditions suppress unreliable output:
        #   (a) Structural: BEV binary is dominated by horizontal features
        #       (stop lines, crosswalks) — _is_intersection_scene().
        #   (b) Speed-based: vehicle is near-stop (< 2.0 m/s ≈ 7.2 km/h) AND
        #       the initial histogram peak quality is weak, indicating no clear
        #       lane structure is visible (common in intersections at night).
        # CRITICAL: whenever we decide to suppress, flush the EMA caches so
        # stale polynomial coefficients from a good upstream frame cannot bleed
        # into this invalid frame or the frames immediately after it.
        is_invalid = self._is_intersection_scene(binary)

        if not is_invalid and speed_mps < _SPEED_GATE_MPS:
            # Soft speed gate: compute a quick histogram peak quality score.
            # If both peaks are too weak to anchor a lane, treat as invalid.
            h, w = binary.shape
            clip_rows  = int(h * _HISTOGRAM_BOTTOM_CLIP)
            hist_top   = h // 2
            hist_bot   = max(h - clip_rows, hist_top + 1)
            histogram  = binary[hist_top:hist_bot, :].sum(axis=0).astype(np.float32)
            lx0 = int(w * _HIST_LEFT_BAND[0]);  lx1 = int(w * _HIST_LEFT_BAND[1])
            rx0 = int(w * _HIST_RIGHT_BAND[0]); rx1 = int(w * _HIST_RIGHT_BAND[1])
            l_peak = float(histogram[lx0:lx1].max()) if lx1 > lx0 else 0.0
            r_peak = float(histogram[rx0:rx1].max()) if rx1 > rx0 else 0.0
            # Minimum expected peak: each row contributes ~255 per active pixel;
            # a real lane marking should light up at least 15 % of the rows.
            min_peak = (hist_bot - hist_top) * 0.15 * 255.0
            if l_peak < min_peak and r_peak < min_peak:
                is_invalid = True
                log.debug(
                    "IPMLaneDetector: speed-gate suppression "
                    "(speed=%.1f m/s, l_peak=%.0f, r_peak=%.0f, min=%.0f)",
                    speed_mps, l_peak, r_peak, min_peak,
                )

        if is_invalid:
            log.debug("IPMLaneDetector: invalid scene — suppressing output and flushing EMA cache")
            # Flush EMA so stale polynomials do not contaminate future frames.
            self._prev_coeffs["left"]  = None
            self._prev_coeffs["right"] = None
            return LaneDetectionResult(
                source="ipm_classical",
                confidence=0.0,
                debug_mask=binary,
            )

        left_pts_bev, right_pts_bev, conf, lconf, rconf = self._sliding_window(binary)

        left_img  = self._fit_and_project(left_pts_bev,  binary.shape[0], "left")
        right_img = self._fit_and_project(right_pts_bev, binary.shape[0], "right")

        # Cross-check: lanes must not cross each other (bow-tie artefact)
        if left_img is not None and right_img is not None:
            l_bot = left_img[np.argmax(left_img[:, 1]),  0]
            r_bot = right_img[np.argmax(right_img[:, 1]), 0]
            if l_bot >= r_bot:
                # Geometry is invalid — drop both
                log.debug("IPMLaneDetector: lanes cross at bottom — discarding pair")
                left_img = right_img = None
                conf = lconf = rconf = 0.0

        center: Optional[np.ndarray] = None
        if left_img is not None and right_img is not None:
            center = (
                (left_img.astype(np.float64) + right_img.astype(np.float64)) / 2.0
            ).astype(np.int32)

        return LaneDetectionResult(
            left_lane        = left_img,
            right_lane       = right_img,
            lane_center      = center,
            confidence       = conf,
            confidence_left  = lconf,
            confidence_right = rconf,
            source           = "ipm_classical",
            debug_mask       = binary,
        )

    # ------------------------------------------------------------------
    # Private: perspective warp
    # ------------------------------------------------------------------

    def reset_segment_state(self) -> None:
        """
        Flush all inter-frame EMA state.

        Call once at the start of each new TFRecord segment so that stale
        polynomial coefficients from a prior segment do not blend into the
        first frames of the new one.
        """
        self._prev_coeffs["left"]  = None
        self._prev_coeffs["right"] = None

    def _warp_to_bev(self, img: np.ndarray) -> np.ndarray:
        """Warp *img* to the BEV canvas using the pre-computed homography."""
        return cv2.warpPerspective(
            img, self._M, (_BEV_W, _BEV_H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # ------------------------------------------------------------------
    # Private: thresholding
    # ------------------------------------------------------------------

    def _threshold(self, bev_bgr: np.ndarray) -> np.ndarray:
        """
        Produce a binary mask highlighting lane-marking pixels in the BEV.

        Pipeline
        --------
        1. HLS colour filter (white + yellow paint).
        2. CLAHE on the L-channel before colour masking to boost faded,
           low-contrast, or distant markings.
        3. Sobel-x gradient filter to catch markings under varied lighting.
        4. Horizontal-structure suppression: erode with a wide horizontal
           kernel to destroy stop lines / crosswalks / painted text (all
           wide horizontal blobs), then restore the remaining near-vertical
           structures with a tall vertical kernel.  This is the key step
           that prevents intersection features from contaminating the mask.
        5. Morphological closing to bridge small gaps in dashed markings.
        """
        # ── colour filter with CLAHE-enhanced L channel ───────────────
        # CLAHE (Contrast Limited Adaptive Histogram Equalisation) boosts
        # local contrast on the Lightness channel before thresholding.
        # This is critical for distant / faded lane markings: in the BEV,
        # far-field pixels are compressed into a small area and their
        # contrast is naturally low.  CLAHE restores it adaptively.
        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hls_enhanced = hls.copy()
        hls_enhanced[:, :, 1] = clahe.apply(hls[:, :, 1])   # enhance L channel

        white_mask  = cv2.inRange(hls_enhanced, _WHITE_HLS_LO,  _WHITE_HLS_HI)
        yellow_mask = cv2.inRange(hls_enhanced, _YELLOW_HLS_LO, _YELLOW_HLS_HI)
        colour_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # ── Sobel-x gradient ─────────────────────────────────────────
        # Use Sobel-x (vertical edges = lane boundaries) and explicitly
        # suppress Sobel-y (horizontal edges = stop lines) by NOT including
        # a y-gradient channel in the combined mask.
        # Apply Sobel on the CLAHE-enhanced L channel so faint distant
        # edges that survive equalization are captured by the gradient too.
        gray    = hls_enhanced[:, :, 1]   # CLAHE-enhanced lightness
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        sobelx  = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
        abs_sx  = np.abs(sobelx)
        max_sx  = abs_sx.max()
        if max_sx > 0:
            scaled = (abs_sx / max_sx * 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(gray)
        _, grad_mask = cv2.threshold(scaled, 30, 255, cv2.THRESH_BINARY)

        # ── combine colour + gradient ─────────────────────────────────
        combined = cv2.bitwise_or(colour_mask, grad_mask)

        # ── horizontal-structure suppression ─────────────────────────
        # Step A: erode with a wide horizontal kernel.
        #   Any blob that is NOT at least _HORIZ_SUPPRESS_W pixels wide
        #   in the horizontal direction survives.  Stop lines, crosswalk
        #   stripes, and painted text are all wider than this threshold
        #   in the BEV; lane-marking segments are narrower.
        k_horiz = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (_HORIZ_SUPPRESS_W, _HORIZ_SUPPRESS_H),
        )
        horiz_blobs = cv2.erode(combined, k_horiz)

        # Step B: dilate the detected horizontal blobs back to their
        #   original extent so we can subtract them cleanly.
        k_dilate = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (_HORIZ_SUPPRESS_W * 2, _HORIZ_SUPPRESS_H * 4),
        )
        horiz_mask = cv2.dilate(horiz_blobs, k_dilate)

        # Step C: remove horizontal structure from the combined mask.
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(horiz_mask))

        # ── morphological closing (bridge dashed-line gaps) ───────────
        k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

        # ── vertical-structure restore / denoise ──────────────────────
        # Open with a tall narrow kernel: keep only blobs that are at
        # least _VERT_RESTORE_H pixels tall (lane segments) and remove
        # salt-and-pepper noise that survived the previous steps.
        k_vert   = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (_VERT_RESTORE_W, _VERT_RESTORE_H),
        )
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_vert)

        return combined

    # ------------------------------------------------------------------
    # Private: sliding window
    # ------------------------------------------------------------------

    def _sliding_window(
        self,
        binary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Locate inlier pixels for the left and right lanes via sliding windows.

        Improvements over the naive implementation
        ------------------------------------------
        • Histogram search is restricted to lateral ego-lane corridor bands
          (_HIST_LEFT_BAND, _HIST_RIGHT_BAND) so that distant parallel roads
          or wide intersection geometry cannot steal the argmax.
        • The bottom _HISTOGRAM_BOTTOM_CLIP fraction of BEV rows is excluded
          from the histogram.  That region is typically dominated by the stop
          line / crosswalk in intersection frames even after morphological
          suppression, and the actual lane starts a few rows higher.
        • Window quality score: a window only adds to the confidence score
          when its inlier pixels have a vertical aspect ratio (|dx/dy| < tan50°).
          Horizontal blobs (bleed-through from stop lines) accumulate inliers
          but do NOT raise the confidence, making the validity flag reliable.

        Returns
        -------
        left_pts   : np.ndarray (N, 2) BEV float32
        right_pts  : np.ndarray (M, 2) BEV float32
        confidence : float — combined quality score in [0, 1]
        lconf      : float — left-side quality score
        rconf      : float — right-side quality score
        """
        h, w = binary.shape

        # ── Corridor-restricted histogram (bottom-clipped) ───────────
        clip_rows = int(h * _HISTOGRAM_BOTTOM_CLIP)
        # Use the band from (h - clip_rows) upward to mid-height for the histogram
        hist_top    = h // 2
        hist_bottom = h - clip_rows
        if hist_bottom <= hist_top:
            hist_bottom = h   # safety: use full height if clip is too aggressive
        histogram = binary[hist_top:hist_bottom, :].sum(axis=0).astype(np.float32)

        # Restrict each half to its lateral corridor to prevent the other
        # side's lane or a wide horizontal blob from stealing the peak.
        lx0 = int(w * _HIST_LEFT_BAND[0])
        lx1 = int(w * _HIST_LEFT_BAND[1])
        rx0 = int(w * _HIST_RIGHT_BAND[0])
        rx1 = int(w * _HIST_RIGHT_BAND[1])

        left_hist  = histogram[lx0:lx1]
        right_hist = histogram[rx0:rx1]

        # Minimum peak energy: a genuine lane marking should activate at
        # least 15 % of the histogram rows worth of pixels in its corridor.
        # Peaks below this level are sparse noise (nighttime curb glints,
        # distant building reflections) — reject them to prevent the sliding
        # window from anchoring to the wrong feature.
        min_peak_val = (hist_bottom - hist_top) * 0.15 * 255.0
        left_valid_peak  = left_hist.max()  >= min_peak_val
        right_valid_peak = right_hist.max() >= min_peak_val

        left_base_x  = (int(np.argmax(left_hist))  + lx0) if left_valid_peak  else w // 4
        right_base_x = (int(np.argmax(right_hist)) + rx0) if right_valid_peak else w * 3 // 4

        win_height  = h // self.n_windows
        nz_y, nz_x = binary.nonzero()

        left_cx, right_cx   = left_base_x, right_base_x
        left_pts, right_pts = [], []
        left_qwins = right_qwins = 0   # quality window counts (vertically oriented)
        left_wins  = right_wins  = 0   # total windows with enough pixels

        for win_idx in range(self.n_windows):
            y_lo = h - (win_idx + 1) * win_height
            y_hi = h - win_idx * win_height

            # Adaptive minimum-pixel threshold: near-field windows (win_idx=0)
            # require the full self.win_min_pix.  Far-field windows (win_idx →
            # n_windows-1) require only 10 pixels.  This is necessary because
            # perspective warping compresses distant lane markings into
            # exponentially fewer BEV pixels — a fixed threshold would drop
            # the lane tracker long before the visible horizon.
            far_field_min = 10
            if self.n_windows > 1:
                t = win_idx / (self.n_windows - 1)   # 0.0 (near) → 1.0 (far)
            else:
                t = 0.0
            adaptive_min_pix = int(
                round(self.win_min_pix * (1.0 - t) + far_field_min * t)
            )

            l_ids = np.where(
                (nz_y >= y_lo) & (nz_y < y_hi) &
                (nz_x >= left_cx  - self.win_margin) &
                (nz_x <  left_cx  + self.win_margin)
            )[0]
            r_ids = np.where(
                (nz_y >= y_lo) & (nz_y < y_hi) &
                (nz_x >= right_cx - self.win_margin) &
                (nz_x <  right_cx + self.win_margin)
            )[0]

            if len(l_ids) >= adaptive_min_pix:
                left_cx    = int(nz_x[l_ids].mean())
                left_wins += 1
                # Quality check: are inliers spread more vertically than horizontally?
                if (nz_y[l_ids].max() - nz_y[l_ids].min()) > 0:
                    aspect = (nz_x[l_ids].max() - nz_x[l_ids].min()) / max(
                        1, nz_y[l_ids].max() - nz_y[l_ids].min()
                    )
                    if aspect < _WIN_QUALITY_ANGLE_TAN:
                        left_qwins += 1

            if len(r_ids) >= adaptive_min_pix:
                right_cx    = int(nz_x[r_ids].mean())
                right_wins += 1
                if (nz_y[r_ids].max() - nz_y[r_ids].min()) > 0:
                    aspect = (nz_x[r_ids].max() - nz_x[r_ids].min()) / max(
                        1, nz_y[r_ids].max() - nz_y[r_ids].min()
                    )
                    if aspect < _WIN_QUALITY_ANGLE_TAN:
                        right_qwins += 1

            left_pts.extend(zip(nz_x[l_ids], nz_y[l_ids]))
            right_pts.extend(zip(nz_x[r_ids], nz_y[r_ids]))

        def _to_arr(pts: list) -> np.ndarray:
            return np.array(pts, dtype=np.float32) if pts else np.empty((0, 2), dtype=np.float32)

        # Confidence is based on quality windows (vertically oriented inliers),
        # not just window fill rate, so stop-line bleed-through cannot inflate it.
        # Additionally, force confidence to 0 for any side whose histogram peak
        # was below the minimum energy threshold — the sliding window started
        # at a fallback anchor and its output is spatially untrustworthy.
        lconf = float(left_qwins)  / self.n_windows if left_valid_peak  else 0.0
        rconf = float(right_qwins) / self.n_windows if right_valid_peak else 0.0
        conf  = (lconf + rconf) / 2.0
        return _to_arr(left_pts), _to_arr(right_pts), conf, lconf, rconf

    # ------------------------------------------------------------------
    # Private: polynomial fit, sampling, and back-projection
    # ------------------------------------------------------------------

    @staticmethod
    def _is_intersection_scene(binary: np.ndarray) -> bool:
        """
        Detect whether the BEV binary is dominated by horizontal structure
        (intersection / stop-line / crosswalk scene).

        Method: compare the total horizontal-edge energy vs the vertical-edge
        energy using morphological hit-or-miss on the binary mask.

        Returns True when the horizontal-to-vertical energy ratio exceeds a
        threshold, indicating the IPM output would be unreliable.
        """
        h, w = binary.shape
        total_pix = int((binary > 0).sum())

        # Measure horizontal projection density: fraction of rows that are
        # densely filled (> 55 % of the row width set as lane pixels).
        row_fill   = (binary > 0).sum(axis=1) / w   # (H,) fraction per row
        dense_rows = int((row_fill > 0.55).sum())
        dense_frac = dense_rows / h

        # Measure vertical projection: fraction of columns that have a
        # meaningful vertical run of set pixels (lane stripe signature).
        col_fill    = (binary > 0).sum(axis=0) / h
        active_cols = int((col_fill > 0.15).sum())
        active_frac = active_cols / w

        # Night-time adaptation: when the overall mask pixel density is very
        # low (dark scene — few bright pixels in BEV), horizontal stop-line
        # and crosswalk paint does not produce the dense row fills seen in
        # daytime.  Lower the required dense_frac threshold from 0.25 → 0.15
        # so the check still fires on sparse nighttime intersection structure.
        night_scene  = (total_pix < h * w * 0.04)   # < 4 % of BEV pixels lit
        dense_thresh = 0.15 if night_scene else 0.25

        # Scene is intersection-like when rows are mostly full (stop-line
        # or crosswalk bleed-through) AND vertical lanes are sparse.
        return dense_frac > dense_thresh and active_frac < 0.15

    def _fit_and_project(
        self,
        pts: np.ndarray,
        bev_height: int,
        side: str,
    ) -> Optional[np.ndarray]:
        """
        Fit a quadratic  x = f(y)  to *pts* in BEV space, sample it, then
        back-project the result to the original image plane.

        The fitted coefficients are blended with the previous frame's
        coefficients via an EMA filter (_COEFF_EMA_ALPHA) before sampling,
        so frame-to-frame inlier fluctuations do not snap the rendered
        lane curve.

        The coefficient cache for *side* is only updated on a successful
        fit; transient failures (too few pixels, degenerate polyfit) leave
        the cache intact so the next good frame can blend from a valid prior.

        Returns None if there are too few inlier pixels or the fit fails.
        """
        if len(pts) < self.min_lane_pix:
            return None

        xs, ys = pts[:, 0], pts[:, 1]
        try:
            coeffs = np.polyfit(ys, xs, _POLY_DEG)
        except (np.linalg.LinAlgError, ValueError) as exc:
            log.debug("polyfit failed: %s", exc)
            return None

        # EMA smoothing: blend the freshly fitted coefficients with the
        # smoothed coefficients from the previous frame.  This damps jitter
        # caused by sliding-window inlier fluctuations between frames while
        # preserving genuine road-curvature changes over time.
        prev = self._prev_coeffs.get(side)
        if prev is not None:
            coeffs = (
                _COEFF_EMA_ALPHA * coeffs
                + (1.0 - _COEFF_EMA_ALPHA) * prev
            )
        # Persist the smoothed coefficients for the next frame.
        self._prev_coeffs[side] = coeffs.copy()

        # Sample the polynomial at n_sample evenly-spaced y values
        y_vals  = np.linspace(0, bev_height - 1, self.n_sample)
        x_vals  = np.clip(np.polyval(coeffs, y_vals), 0, _BEV_W - 1)
        bev_pts = np.column_stack([x_vals, y_vals]).astype(np.float32)

        # Back-project BEV → original image plane
        img_pts = cv2.perspectiveTransform(bev_pts.reshape(-1, 1, 2), self._M_inv)
        if img_pts is None:
            return None
        return img_pts.reshape(-1, 2).astype(np.int32)


# ---------------------------------------------------------------------------
# Backend 2 — CLRNet (ONNX-Runtime, plug in when weights are ready)
# ---------------------------------------------------------------------------


from .base       import AbstractLaneDetector, VehicleState
from .visual_dp  import DrivablePathStrategy
from .visual_host import HostLaneStrategy

log = logging.getLogger(__name__)


class IPMPlugin(AbstractLaneDetector):
    """
    Adapter: exposes ``VisualPerceptionDetector`` (IPM backend) as a plugin.

    Design pattern: Adapter.  The BEV sliding-window algorithm lives in
    ``IPMLaneDetector``; ``VisualPerceptionDetector`` adds temporal persistence
    and confidence gating.  This class only maps the plugin API to that
    existing interface and reads the ``contributes`` setting.
    """

    def __init__(
        self,
        cfg:          DictConfig,
        image_width:  int = 1920,
        image_height: int = 1280,
    ) -> None:
        lane_cfg          = cfg.perception.lane
        ipm_cfg           = lane_cfg.ipm
        host_conf         = float(getattr(lane_cfg, "host_lane_confidence_threshold", 0.01))
        self._contributes = str(getattr(ipm_cfg, "contributes", "all"))

        backend = IPMLaneDetector(
            image_width       = image_width,
            image_height      = image_height,
            n_sample          = int(ipm_cfg.n_sample),
            n_windows         = int(ipm_cfg.n_windows),
            win_margin        = int(ipm_cfg.win_margin),
            win_min_pix       = int(ipm_cfg.win_min_pix),
            min_lane_pix      = int(ipm_cfg.min_lane_pix),
            roi_top_y         = float(ipm_cfg.roi_top_y),
            roi_top_x_half    = float(ipm_cfg.roi_top_x_half),
            roi_bottom_margin = float(ipm_cfg.roi_bottom_margin),
        )
        self._detector = VisualPerceptionDetector(
            image_width                    = image_width,
            image_height                   = image_height,
            backend                        = backend,
            host_lane_confidence_threshold = host_conf,
        )
        log.info(
            "IPMPlugin: initialized  host_conf=%.3f  contributes=%s",
            host_conf, self._contributes,
        )

    def process(
        self,
        frame_bgr:     Optional[np.ndarray],
        vehicle_state: VehicleState,
    ) -> dict:
        if frame_bgr is None:
            return {}

        drivable_raw, host_raw = self._detector.detect(
            frame_bgr, speed_mps=vehicle_state.speed_mps
        )

        result: dict = {}
        if self._contributes in ("drivable", "all"):
            result["drivable_raw"]  = drivable_raw
            result["drivable_path"] = DrivablePathStrategy.package(drivable_raw)
        if self._contributes in ("host_lane", "all"):
            result["host_raw"]  = host_raw
            result["host_lane"] = HostLaneStrategy.package(host_raw)
        return result

    def reset(self) -> None:
        self._detector.reset_segment_state()
