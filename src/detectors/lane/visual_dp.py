"""
src/models/lanes/visual_dp.py
==============================
Strategy — Path 3: Drivable Path (free-space detection).

This module is deliberately stateless.  Its only responsibility is to
package the raw drivable-area output dict produced by an inference engine
(YOLOPv2 drivable-area head, or a VisualPerceptionDetector fallback) into
the standard path-entry format consumed by the serializer and the visualizer.

No neural-network inference code lives here.  The inference engine is
owned and executed by ``LaneManager``; the result dict is forwarded to
``DrivablePathStrategy.package()`` for normalisation only.
"""

from __future__ import annotations


class DrivablePathStrategy:
    """
    Stateless strategy: package a raw drivable-area inference result into
    the standardised ``drivable_path`` serializable dict.
    """

    @staticmethod
    def package(drivable_data: dict | None) -> dict:
        """
        Convert the raw drivable-area dict from the inference engine into the
        standard path entry.

        Parameters
        ----------
        drivable_data : dict | None
            First element returned by ``YOLOPv2DrivableDetector.detect_full()``
            or by ``VisualPerceptionDetector.detect()``.
            Pass ``None`` when no inference was run (e.g. missing frame).

        Returns
        -------
        dict
            Standardised ``drivable_path`` entry with keys:
                center, left, right,
                valid_center, valid_left, valid_right,
                confidence_center, confidence_left, confidence_right,
                timestamps_s, source, is_gt.
        """
        def _pts(arr) -> list:
            if arr is None:
                return []
            a = arr.tolist() if hasattr(arr, "tolist") else list(arr)
            return a if len(a) >= 2 else []

        def _conf(v) -> float:
            return float(v) if v is not None else 0.0

        if drivable_data is not None:
            dp_cp   = _pts(drivable_data.get("center_path"))
            dp_lp   = _pts(drivable_data.get("left_path"))
            dp_rp   = _pts(drivable_data.get("right_path"))
            dp_conf = _conf(drivable_data.get("confidence", 0.0))
            dp_src  = drivable_data.get("source", "unknown")
        else:
            dp_cp, dp_lp, dp_rp, dp_conf, dp_src = [], [], [], 0.0, "none"

        return {
            "center":            dp_cp,
            "left":              dp_lp,
            "right":             dp_rp,
            "valid_center":      len(dp_cp) >= 2,
            "valid_left":        len(dp_lp) >= 2,
            "valid_right":       len(dp_rp) >= 2,
            "confidence_center": dp_conf,
            "confidence_left":   dp_conf,
            "confidence_right":  dp_conf,
            "timestamps_s":      [],
            "source":            dp_src,
            "is_gt":             False,
        }
