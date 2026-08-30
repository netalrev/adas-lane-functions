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

import numpy as np


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

        def _midline(left: list, right: list) -> list:
            """Derive a centerline as the pointwise average of left/right,
            resampled onto a shared y-grid (same approach as hdmap_serializer
            and LaneRelationMeasurer._boundary_midline)."""
            if len(left) < 2 or len(right) < 2:
                return []
            left_arr  = np.array(left,  dtype=np.float64)
            right_arr = np.array(right, dtype=np.float64)
            y_lo = float(max(left_arr[:, 1].min(),  right_arr[:, 1].min()))
            y_hi = float(min(left_arr[:, 1].max(),  right_arr[:, 1].max()))
            if y_hi <= y_lo:
                return []
            left_sorted  = left_arr[np.argsort(left_arr[:, 1])]
            right_sorted = right_arr[np.argsort(right_arr[:, 1])]
            y_c = np.linspace(y_lo, y_hi, 30)
            xl  = np.interp(y_c, left_sorted[:, 1],  left_sorted[:, 0])
            xr  = np.interp(y_c, right_sorted[:, 1], right_sorted[:, 0])
            return np.column_stack(
                [((xl + xr) / 2).astype(np.int32), y_c.astype(np.int32)]
            ).tolist()

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

        valid_left  = hl_vl and len(hl_ll) >= 2
        valid_right = hl_vr and len(hl_rl) >= 2
        # A host-lane center is only derivable once BOTH boundaries are valid;
        # it was previously hardcoded to empty/invalid even when both sides
        # were present, which silently starved every downstream consumer
        # that checks "valid_center" (e.g. lane-quality evaluation) of a
        # signal that LaneRelationMeasurer was already deriving internally.
        hl_center = _midline(hl_ll, hl_rl) if (valid_left and valid_right) else []

        return {
            "center":            hl_center,
            "left":              hl_ll,
            "right":             hl_rl,
            "valid_center":      len(hl_center) >= 2,
            "valid_left":        valid_left,
            "valid_right":       valid_right,
            "confidence_center": hl_conf,
            "confidence_left":   hl_lconf,
            "confidence_right":  hl_rconf,
            "timestamps_s":      [],
            "source":            hl_src,
            "is_gt":             False,
        }
