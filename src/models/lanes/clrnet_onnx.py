"""
src/models/lanes/clrnet_onnx.py
================================
Plugin — CLRNet ONNX Neural Lane Detector (Strategy Pattern).

Wraps ``VisualPerceptionDetector`` (with a ``CLRNetLaneDetector`` backend) in
the ``AbstractLaneDetector`` interface.  When no valid ``onnx_path`` is
configured, the plugin transparently falls back to an IPM backend so the
pipeline remains operational without the ONNX weights present.

Config keys consumed from ``cfg.perception.lane``
----------------------------------------------------
host_lane_confidence_threshold  : float   (default 0.01)
clrnet.onnx_path                : str
clrnet.confidence_threshold     : float   (default 0.005)
clrnet.device                   : str     "cpu" | "cuda"
clrnet.contributes              : str     "drivable" | "host_lane" | "all"  (default "all")

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

from ._backend     import LaneDetectionResult, LaneDetectorBase, VisualPerceptionDetector
from .ipm_classical import IPMLaneDetector  # used as CLRNet fallback backend

# ---------------------------------------------------------------------------
# CLRNet backend (moved from src/models/lane_detector.py)
# ---------------------------------------------------------------------------

class CLRNetLaneDetector(LaneDetectorBase):
    """
    ONNX-Runtime backend for CLRNet (Cross-Layer Refinement Network).

    CLRNet is the current state-of-the-art on the CULane benchmark
    (F1 = 81.43 %) and TuSimple benchmark (96.84 %).

    References
    ----------
    Paper      : https://arxiv.org/abs/2203.10350
    Repository : https://github.com/Turoad/CLRNet

    Integration steps
    -----------------
    1. Export a CLRNet checkpoint to ONNX::

           python tools/export_onnx.py \\
               --config configs/clrnet/clrnet_culane_r18.py \\
               --ckpt  <checkpoint.pth>

    2. Install the runtime dependency::

           pip install onnxruntime        # CPU
           pip install onnxruntime-gpu    # CUDA

    3. Construct this class with ``onnx_path`` pointing to the exported file.

    4. Implement the inference logic inside ``detect()`` following the
       TODO comments therein.

    Until step 3 is complete this class raises ``ImportError`` or
    ``FileNotFoundError`` on construction so that misconfiguration is
    caught early rather than at the first inference call.

    Parameters
    ----------
    onnx_path : str
        Absolute path to the CLRNet ONNX model file.
    image_width : int
        Source image width; used to scale network output back to pixels.
    image_height : int
        Source image height.
    confidence_threshold : float
        Minimum per-lane score to include a detection in the output.
    device : str
        ONNX Runtime execution provider — "cpu" or "cuda".
    """

    # Canonical CLRNet-CULane input resolution (img_w × img_h from config)
    _NET_W: int = 800
    _NET_H: int = 320
    # BGR mean subtraction (from CLRNet CULane config: img_norm)
    _IMG_MEAN: np.ndarray = np.array([103.939, 116.779, 123.68], dtype=np.float32)

    # CLRNet output layout per prior:  [cls0, cls1, start_y, start_x, theta,
    #                                   length, x_0 … x_71]
    # x_i are x-coordinates at the 72 row-strips, normalised by (img_w - 1)
    _N_OFFSETS: int = 72

    # CULane training image aspect ratio (width / height = 1640 / 590 ≈ 2.78).
    # Waymo images (1920×1280, ratio 1.5) must be cropped to this ratio before
    # feeding the network — otherwise the perspective is totally wrong and all
    # detected lanes appear as near-vertical lines spanning the full image.
    _CULANE_ASPECT: float = 1640.0 / 590.0  # ≈ 2.78

    def __init__(
        self,
        onnx_path:            str,
        image_width:          int   = 1920,
        image_height:         int   = 1280,
        confidence_threshold: float = 0.50,
        device:               str   = "cpu",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for CLRNetLaneDetector. "
                "Install it with:  pip install onnxruntime  (CPU)  "
                "or  pip install onnxruntime-gpu  (CUDA)."
            ) from exc

        self.image_width          = image_width
        self.image_height         = image_height
        self.confidence_threshold = confidence_threshold

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session    = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        log.info("CLRNetLaneDetector: loaded model from %s", onnx_path)

    def detect(
        self,
        image_bgr: np.ndarray,
        speed_mps: float = 10.0,   # accepted for API compatibility; unused by CLRNet
    ) -> LaneDetectionResult:
        """
        Run CLRNet inference on a single BGR frame.

        Pre-processing mirrors the CULane val_process:
          - Resize to (NET_W × NET_H) = (800 × 320)
          - Subtract BGR mean [103.939, 116.779, 123.68] (std = 1, no division)
          - Layout: float32 NCHW

        Decoding mirrors CLRHead.predictions_to_pred():
          - prior_ys = linspace(1, 0, 72) — normalised y positions
            (1 = bottom-of-frame, 0 = top-of-frame)
          - x offsets 6:78 are normalised by (img_w − 1) = 799
          - start / end from lane[2] (start_y) and lane[5] (length)
          - x values outside [start, end] or < 0 are invalid

        Ego-lane selection: left boundary = rightmost lane left of centre,
        right boundary = leftmost lane right of centre.
        """
        src_h, src_w = image_bgr.shape[:2]

        # ── 1. Pre-process ────────────────────────────────────────────────────
        # Crop the bottom of the source image to match CULane's aspect ratio
        # (≈2.78:1) before resizing to 800×320.  This preserves the correct
        # road perspective that CLRNet was trained on.  Without this crop,
        # Waymo's tall image (1920×1280, ratio 1.5) gets squished so much
        # vertically that all x offsets appear as near-vertical lines.
        crop_h = min(int(src_w / self._CULANE_ASPECT), src_h)
        crop_y0 = src_h - crop_h          # start row of the crop
        cropped  = image_bgr[crop_y0:]    # shape: (crop_h, src_w, 3)

        net_img = cv2.resize(cropped, (self._NET_W, self._NET_H))
        net_img = net_img.astype(np.float32) - self._IMG_MEAN  # BGR subtract
        tensor = net_img.transpose(2, 0, 1)[np.newaxis]        # → NCHW

        # ── 2. Infer ──────────────────────────────────────────────────────────
        raw = self._session.run(None, {self._input_name: tensor})[0]  # (1, 192, 78)
        preds = raw[0]  # (192, 78)

        # ── 3. Decode ─────────────────────────────────────────────────────────
        # Stable softmax on cls logits → lane-existence probability
        logits = preds[:, :2]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        scores = e[:, 1] / e.sum(axis=1)   # (192,)

        keep = np.where(scores >= self.confidence_threshold)[0]
        if keep.size == 0:
            log.debug("CLRNetLaneDetector: no lanes above threshold %.2f",
                      self.confidence_threshold)
            return LaneDetectionResult(source="clrnet_onnx", confidence=0.0)

        # Row-strip normalised y values: 1=bottom … 0=top of the net image
        prior_ys = np.linspace(1.0, 0.0, self._N_OFFSETS, dtype=np.float64)
        n_strips = self._N_OFFSETS - 1  # 71

        lanes: list[tuple[float, np.ndarray]] = []   # (score, (N,2) float)

        for idx in keep:
            lane   = preds[idx]
            xs_raw = lane[6:].copy().astype(np.float64)  # 72 normalised x

            # Use all x offsets that fall within the normalised image bounds
            # [0, 1].  The CLRNet length head is unreliable on out-of-domain
            # images (e.g. Waymo vs CULane), so we skip the start/end masking
            # and rely solely on x-range validity.
            valid = (xs_raw >= 0.0) & (xs_raw <= 1.0)
            if valid.sum() <= 2:
                continue

            y_norm = prior_ys[valid]    # normalised, 1=bottom → 0=top
            x_norm = xs_raw[valid]

            # Scale normalised coords to source-image pixels.
            # x: normalised by (NET_W - 1) = 799; scale to full source width.
            x_px = x_norm * (self._NET_W - 1) * src_w / self._NET_W
            # y: prior_ys 1→0 maps to the CROPPED region, then shift by crop_y0
            #    so coordinates land correctly in the full source image.
            y_px = crop_y0 + y_norm * (crop_h - 1)

            # ── Plausibility filter ──────────────────────────────────────────
            # A valid lane must:
            #   1. Span at least 12 % of image height vertically.
            #   2. Have meaningful horizontal extent (not near-vertical).
            #   3. Converge toward the horizon — x-spread at the BOTTOM (near
            #      field) must be wider than at the TOP (far field).  Real road
            #      lanes always narrow as they approach the vanishing point.
            #      Inverted detections (building edges, curbs on urban streets)
            #      show the opposite pattern and are rejected here.
            y_span = float(y_px.max() - y_px.min())
            x_span = float(x_px.max() - x_px.min())
            min_y_span = 0.12 * src_h   # at least 12% of image height
            min_x_span = 0.02 * src_w   # at least 2% of image width
            if y_span < min_y_span or x_span < min_x_span:
                continue

            # Convergence check: split into bottom-third and top-third strips
            # and verify that the x-position moves toward the image centre as
            # y decreases (i.e. the lane converges on the horizon).
            sort_idx  = np.argsort(y_px)[::-1]   # bottom → top (y descending)
            n_pts     = len(sort_idx)
            third     = max(1, n_pts // 3)
            x_near    = float(x_px[sort_idx[:third]].mean())   # near field
            x_far     = float(x_px[sort_idx[-third:]].mean())  # far field
            # Near x should be farther from image centre than far x
            cx_f = src_w / 2.0
            if abs(x_near - cx_f) <= abs(x_far - cx_f):
                # Lane diverges away from centre going up → reject
                continue

            pts = np.column_stack([x_px, y_px]).astype(np.int32)
            lanes.append((float(scores[idx]), pts))

        if not lanes:
            return LaneDetectionResult(source="clrnet_onnx", confidence=0.0)

        # ── 4. Select ego left / right lanes ─────────────────────────────────
        cx = src_w / 2.0
        # Use the x value at the bottom-most detected point for lane placement
        bottom_xs = np.array([
            pts[np.argmax(pts[:, 1]), 0]
            for _, pts in lanes
        ], dtype=np.float64)

        left_mask  = bottom_xs < cx
        right_mask = bottom_xs >= cx

        left_lane = right_lane = None
        left_score = right_score = 0.0
        if left_mask.any():
            rel_idx = int(np.argmax(bottom_xs[left_mask]))
            abs_idx = np.where(left_mask)[0][rel_idx]
            left_lane  = lanes[abs_idx][1]
            left_score = float(lanes[abs_idx][0])
        if right_mask.any():
            rel_idx = int(np.argmin(bottom_xs[right_mask]))
            abs_idx = np.where(right_mask)[0][rel_idx]
            right_lane  = lanes[abs_idx][1]
            right_score = float(lanes[abs_idx][0])

        # ── Crossing guard ────────────────────────────────────────────────────
        # If the two selected ego lanes cross at any shared y level (left x >
        # right x), the pair is geometrically invalid (bow-tie artefact from
        # domain-shifted detections).  Discard both rather than draw garbage.
        if left_lane is not None and right_lane is not None:
            l_sort = left_lane[np.argsort(left_lane[:, 1])]
            r_sort = right_lane[np.argsort(right_lane[:, 1])]
            y_lo = float(max(l_sort[0, 1], r_sort[0, 1]))
            y_hi = float(min(l_sort[-1, 1], r_sort[-1, 1]))
            if y_hi > y_lo:
                y_check = np.linspace(y_lo, y_hi, 10)
                xl = np.interp(y_check, l_sort[:, 1], l_sort[:, 0])
                xr = np.interp(y_check, r_sort[:, 1], r_sort[:, 0])
                if np.any(xl >= xr):
                    log.debug("CLRNetLaneDetector: lanes cross — discarding pair")
                    left_lane = right_lane = None

        # ── 5. Lane centre ────────────────────────────────────────────────────
        lane_center = None
        if left_lane is not None and right_lane is not None:
            y_lo = float(max(left_lane[:, 1].min(),  right_lane[:, 1].min()))
            y_hi = float(min(left_lane[:, 1].max(),  right_lane[:, 1].max()))
            if y_hi > y_lo:
                y_common = np.linspace(y_lo, y_hi, 20)
                # Both lanes are ordered bottom-to-top (y descending numerically)
                # np.interp needs xp increasing, so sort ascending
                l_sort = left_lane[np.argsort(left_lane[:, 1])]
                r_sort = right_lane[np.argsort(right_lane[:, 1])]
                x_left  = np.interp(y_common, l_sort[:, 1], l_sort[:, 0])
                x_right = np.interp(y_common, r_sort[:, 1], r_sort[:, 0])
                lane_center = np.column_stack(
                    [((x_left + x_right) / 2).astype(np.int32),
                     y_common.astype(np.int32)]
                )

        conf = float(np.mean([s for s, _ in lanes[:4]]))
        log.debug("CLRNetLaneDetector: %d lanes detected, conf=%.3f", len(lanes), conf)
        return LaneDetectionResult(
            left_lane=left_lane,
            right_lane=right_lane,
            lane_center=lane_center,
            confidence=conf,
            confidence_left=left_score,
            confidence_right=right_score,
            source="clrnet_onnx",
        )


# ---------------------------------------------------------------------------
# Public façade (backward-compatible entry point)
# ---------------------------------------------------------------------------


from .base       import AbstractLaneDetector, VehicleState
from .visual_dp  import DrivablePathStrategy
from .visual_host import HostLaneStrategy

log = logging.getLogger(__name__)


class CLRNetPlugin(AbstractLaneDetector):
    """
    Adapter: exposes ``VisualPerceptionDetector`` (CLRNet backend) as a plugin.

    When ``clrnet.onnx_path`` is empty or the ``clrnet`` config section is
    absent, the constructor silently builds an IPM backend instead.  This
    fallback guarantee means the plugin is always safe to list in
    ``active_plugins``, even on machines where the ONNX weights are absent.
    """

    def __init__(
        self,
        cfg:          DictConfig,
        image_width:  int = 1920,
        image_height: int = 1280,
    ) -> None:
        lane_cfg          = cfg.perception.lane
        clr_cfg           = getattr(lane_cfg, "clrnet", None)
        host_conf         = float(getattr(lane_cfg, "host_lane_confidence_threshold", 0.01))
        self._contributes = str(getattr(clr_cfg, "contributes", "all")) if clr_cfg else "all"

        onnx_path = str(getattr(clr_cfg, "onnx_path", "")) if clr_cfg else ""

        if onnx_path:
            backend = CLRNetLaneDetector(
                onnx_path            = onnx_path,
                image_width          = image_width,
                image_height         = image_height,
                confidence_threshold = float(getattr(clr_cfg, "confidence_threshold", 0.005)),
                device               = str(getattr(clr_cfg, "device", "cpu")),
            )
            log.info(
                "CLRNetPlugin: ONNX loaded  conf_thresh=%.3f  contributes=%s",
                float(getattr(clr_cfg, "confidence_threshold", 0.005)), self._contributes,
            )
        else:
            log.warning(
                "CLRNetPlugin: clrnet.onnx_path not set — falling back to IPM backend."
            )
            ipm_cfg = lane_cfg.ipm
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
