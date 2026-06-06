"""
src/models/detection/detector.py
==================================
Target detector wrapping YOLOv8n ONNX for the ADAS perception pipeline.

Public API
----------
    Detection
        Dataclass representing a single filtered detection.

    TargetDetector(cfg)
        Load the ONNX session once, run per-frame detection with confidence
        filtering, class whitelisting, and NMS.

Design notes
------------
- All neural-network inference is isolated here.  No detection logic leaks
  into the pipeline orchestrator or the Kalman tracker.
- COCO class IDs are remapped to an internal 4-class schema on output:
      0 = vehicle     (COCO: car, bus, truck)
      1 = pedestrian  (COCO: person)
      2 = cyclist     (COCO: bicycle, motorcycle)
      3 = other
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class mapping: COCO → internal 4-class schema
# ---------------------------------------------------------------------------

# Internal class IDs
CLASS_VEHICLE    = 0
CLASS_PEDESTRIAN = 1
CLASS_CYCLIST    = 2
CLASS_OTHER      = 3

CLASS_NAMES: dict[int, str] = {
    CLASS_VEHICLE:    "vehicle",
    CLASS_PEDESTRIAN: "pedestrian",
    CLASS_CYCLIST:    "cyclist",
    CLASS_OTHER:      "other",
}

# COCO class ID → internal class ID
_COCO_TO_CLASS: dict[int, int] = {
    0: CLASS_PEDESTRIAN,  # person
    1: CLASS_CYCLIST,     # bicycle
    2: CLASS_VEHICLE,     # car
    3: CLASS_CYCLIST,     # motorcycle
    5: CLASS_VEHICLE,     # bus
    7: CLASS_VEHICLE,     # truck
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    A single target detection after all filtering stages.

    Attributes
    ----------
    bbox_xyxy : np.ndarray, shape (4,)
        Bounding box in original image coordinates [x1, y1, x2, y2].
    confidence : float
        Detector confidence score in [0, 1].
    class_id : int
        Internal class ID (0=vehicle, 1=pedestrian, 2=cyclist, 3=other).
    class_name : str
        Human-readable class label.
    """
    bbox_xyxy:  np.ndarray
    confidence: float
    class_id:   int
    class_name: str

    def to_dict(self) -> dict:
        return {
            "bbox_xyxy":  self.bbox_xyxy.tolist(),
            "confidence": self.confidence,
            "class_id":   self.class_id,
            "class_name": self.class_name,
        }


# ---------------------------------------------------------------------------
# TargetDetector
# ---------------------------------------------------------------------------

class TargetDetector:
    """
    Wraps a YOLOv8n ONNX session and exposes a single ``detect()`` method.

    Pipeline (per frame)
    --------------------
    1. Preprocess        — resize to 640×640, normalise [0,1], convert to NCHW.
    2. Inference         — single ONNX forward pass.
    3. Decode            — transpose output, extract class scores, map cx/cy/w/h
                           back to original image coordinates.
    4. Confidence filter — drop detections below threshold (0.25 by default).
    5. Class filter      — keep only whitelisted COCO class IDs; disabled when
                           ``class_whitelist`` is empty (accept all 80 classes).
    6. NMS               — per-class OpenCV DNN NMS to remove duplicate boxes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config node at ``cfg.perception.detector``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for TargetDetector.\n"
                "Install: pip install onnxruntime   (CPU)\n"
                "      or pip install onnxruntime-gpu  (CUDA)"
            ) from exc

        self._input_size           = int(cfg.input_size)
        self._confidence_threshold = float(cfg.confidence_threshold)
        self._nms_iou_threshold    = float(cfg.nms_iou_threshold)
        self._class_whitelist      = set(int(c) for c in cfg.class_whitelist)
        providers                  = list(cfg.providers)

        self._session    = ort.InferenceSession(str(cfg.onnx_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        log.info("TargetDetector loaded: %s  providers=%s", cfg.onnx_path, providers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        img_bgr : np.ndarray
            Original-resolution BGR image from the front camera.

        Returns
        -------
        list[Detection]
            Filtered, deduplicated detections sorted by descending confidence.
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        blob, scale_x, scale_y = self._preprocess(img_bgr)
        raw = self._session.run(None, {self._input_name: blob})[0]
        detections = self._postprocess(raw, scale_x, scale_y)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self, img_bgr: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """
        Resize, normalise, and convert to NCHW float32.

        Returns
        -------
        blob : np.ndarray, shape (1, 3, 640, 640), float32
        scale_x : float  — factor to map 640-space x → original image x
        scale_y : float  — factor to map 640-space y → original image y
        """
        h, w = img_bgr.shape[:2]
        scale_x = w / self._input_size
        scale_y = h / self._input_size

        resized = cv2.resize(img_bgr, (self._input_size, self._input_size))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob    = (rgb.astype(np.float32) / 255.0)
        blob    = np.transpose(blob, (2, 0, 1))[np.newaxis]  # NCHW
        return blob, scale_x, scale_y

    def _postprocess(
        self,
        raw_output: np.ndarray,
        scale_x: float,
        scale_y: float,
    ) -> list[Detection]:
        """
        Decode the YOLOv8n ONNX output tensor into Detection objects.

        YOLOv8n ONNX output shape: [1, 84, 8400]
        Channels 0-3:  cx, cy, w, h (in 640-space pixels)
        Channels 4-83: per-class confidence scores (COCO 80 classes)
        """
        predictions = raw_output[0].T           # [8400, 84]
        boxes_raw   = predictions[:, :4]        # cx, cy, w, h
        class_scores = predictions[:, 4:]       # [8400, 80]

        class_ids    = np.argmax(class_scores, axis=1)
        confidences  = np.max(class_scores, axis=1)

        # Confidence mask; class whitelist is optional (empty = accept all classes)
        conf_mask = confidences >= self._confidence_threshold
        if self._class_whitelist:
            mask = conf_mask & np.isin(class_ids, list(self._class_whitelist))
        else:
            mask = conf_mask

        if not mask.any():
            return []

        boxes_raw   = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        # cx,cy,w,h → x1,y1,x2,y2 (still in 640-space)
        x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2.0
        y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2.0
        x2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2.0
        y2 = boxes_raw[:, 1] + boxes_raw[:, 3] / 2.0

        # Scale to original image dimensions
        x1 = x1 * scale_x;  x2 = x2 * scale_x
        y1 = y1 * scale_y;  y2 = y2 * scale_y
        boxes_xyxy = np.column_stack([x1, y1, x2, y2]).astype(np.float32)

        # NMS via OpenCV DNN — run independently per COCO class.
        # Class-agnostic NMS would allow a high-IoU vehicle box to suppress
        # a co-located pedestrian (or vice-versa).  Per-class suppression
        # ensures only same-class duplicates are collapsed, so genuinely
        # different objects that happen to overlap both survive.
        surviving: list[int] = []
        for cls in np.unique(class_ids):
            cls_mask   = np.where(class_ids == cls)[0]
            cls_boxes  = boxes_xyxy[cls_mask].tolist()
            cls_scores = confidences[cls_mask].tolist()
            cls_nms    = cv2.dnn.NMSBoxes(
                cls_boxes, cls_scores,
                self._confidence_threshold,
                self._nms_iou_threshold,
            )
            if len(cls_nms) > 0:
                surviving.extend(cls_mask[np.asarray(cls_nms).flatten()].tolist())

        if not surviving:
            return []

        detections: list[Detection] = []
        for i in surviving:
            coco_cls = int(class_ids[i])
            our_cls  = _COCO_TO_CLASS.get(coco_cls, CLASS_OTHER)
            detections.append(Detection(
                bbox_xyxy  = boxes_xyxy[i],
                confidence = float(confidences[i]),
                class_id   = our_cls,
                class_name = CLASS_NAMES[our_cls],
            ))
        return detections
