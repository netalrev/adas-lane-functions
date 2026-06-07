"""
src/models/lanes/visual_host.py
================================
Strategy — Path 4: Host Lane (painted lane-marking detection).

This module is deliberately stateless.  Its only responsibility is to
package the raw host-lane output dict produced by an inference engine
(YOLOPv2 lane-line head, CLRNet, or IPM) into the standard path-entry
format consumed by the serializer and the visualizer.

No neural-network inference code lives here.  The inference engine is
owned and executed by ``LaneManager``; the result dict is forwarded to
``HostLaneStrategy.package()`` for normalisation only.
"""

from __future__ import annotations


class HostLaneStrategy:
    """
    Stateless strategy: package a raw host-lane inference result into the
    standardised ``host_lane`` serializable dict.
    """

    @staticmethod
    def package(host_lane_data: dict | None) -> dict:
        """
        Convert the raw host-lane dict from the inference engine into the
        standard path entry.

        Parameters
        ----------
        host_lane_data : dict | None
            Second element returned by ``YOLOPv2DrivableDetector.detect_full()``
            or by ``VisualPerceptionDetector.detect()``.
            Pass ``None`` when no inference was run (e.g. missing frame).

        Returns
        -------
        dict
            Standardised ``host_lane`` entry with keys:
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

        if host_lane_data is not None:
            hl_ll    = _pts(host_lane_data.get("left_lane"))
            hl_rl    = _pts(host_lane_data.get("right_lane"))
            hl_conf  = _conf(host_lane_data.get("confidence", 0.0))
            hl_lconf = _conf(host_lane_data.get("confidence_left",  hl_conf))
            hl_rconf = _conf(host_lane_data.get("confidence_right", hl_conf))
            hl_src   = host_lane_data.get("source", "unknown")
            # Support both "valid_left"/"valid_right" (YOLOPv2 style) and
            # the legacy single "valid" key (CLRNet / IPM style).
            hl_vl    = bool(host_lane_data.get(
                "valid_left", host_lane_data.get("valid", False)
            ))
            hl_vr    = bool(host_lane_data.get(
                "valid_right", host_lane_data.get("valid", False)
            ))
        else:
            hl_ll    = hl_rl    = []
            hl_conf  = hl_lconf = hl_rconf = 0.0
            hl_src   = "none"
            hl_vl    = hl_vr   = False

        return {
            "center":            [],
            "left":              hl_ll,
            "right":             hl_rl,
            "valid_center":      False,
            "valid_left":        hl_vl and len(hl_ll) >= 2,
            "valid_right":       hl_vr and len(hl_rl) >= 2,
            "confidence_center": hl_conf,
            "confidence_left":   hl_lconf,
            "confidence_right":  hl_rconf,
            "timestamps_s":      [],
            "source":            hl_src,
            "is_gt":             False,
        }
