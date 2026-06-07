"""
src/models/lanes/manager.py
============================
LaneManager — plugin-registry orchestrator for all lane-path strategies.

Architecture — Plugin / Strategy Pattern
-----------------------------------------
Each lane-detection path is a self-contained plugin that implements the
``AbstractLaneDetector`` interface (``base.py``).  The ``LaneManager`` reads
the ``active_plugins`` list from the Hydra config, instantiates each plugin
through the ``_PLUGIN_REGISTRY`` dict, and iterates over them once per frame.

Adding a new detector in future requires **zero changes** to this file:
    1. Create a new plugin module (e.g. ``laneformer.py``) that subclasses
       ``AbstractLaneDetector``.
    2. Register it in ``_PLUGIN_REGISTRY`` below.
    3. Add its name to ``active_plugins`` in ``conf/model/lane.yaml``.

Merge strategy
--------------
``process()`` calls every plugin in the order listed in ``active_plugins``
and merges their output dicts with ``dict.update()``.  A plugin listed
*later* overrides the same key from an earlier plugin.  This enables hybrid
configurations — for example, listing YOLOPv2 before IPM lets IPM override
only the host-lane keys (set ``contributes: host_lane`` in the ipm config
section) while preserving YOLOPv2's superior drivable-area output.

Public API
----------
VehicleState
    Re-exported from ``base.py`` for backward-compatible
    ``from src.models.lanes import VehicleState`` imports in the pipeline.

LaneManager(cfg, image_width, image_height)
    Instantiate once.  All ONNX loads happen during construction.

LaneManager.reset_segment_state()
    Call between TFRecord segments; propagates ``reset()`` to every plugin.

LaneManager.process(frame_bgr, vehicle_state) -> dict
    Run all active plugins; return the merged result dict.
    Always returns all six canonical keys (missing keys are filled with safe
    empty defaults so callers never encounter a ``KeyError``).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from omegaconf import DictConfig

from .base             import AbstractLaneDetector, VehicleState  # re-exported
from .kinematic_ego    import KinematicPlugin
from .ipm_classical    import IPMPlugin
from .clrnet_onnx      import CLRNetPlugin
from .yolopv2_drivable import YOLOPv2Plugin
from .visual_dp        import DrivablePathStrategy
from .visual_host      import HostLaneStrategy

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plugin registry — maps config name -> plugin class
# ---------------------------------------------------------------------------
# To register a new detector, add one entry here.  The LaneManager needs no
# other modification.

_PLUGIN_REGISTRY: dict[str, type[AbstractLaneDetector]] = {
    "kinematic": KinematicPlugin,
    "ipm":       IPMPlugin,
    "clrnet":    CLRNetPlugin,
    "yolopv2":   YOLOPv2Plugin,
}


# ---------------------------------------------------------------------------
# Default fallback values
# ---------------------------------------------------------------------------

def _empty_kinematic_raw() -> dict:
    """Return empty numpy arrays matching the ``KinematicPathPredictor`` output shape."""
    return {
        "centre_line":    np.empty((0, 2), dtype=np.float64),
        "left_boundary":  np.empty((0, 2), dtype=np.float64),
        "right_boundary": np.empty((0, 2), dtype=np.float64),
        "timestamps":     np.empty(0,      dtype=np.float64),
    }


def _empty_kinematic() -> dict:
    """Return a zero-content serialized kinematic path dict."""
    return {
        "center":            [],
        "left":              [],
        "right":             [],
        "valid_center":      False,
        "valid_left":        False,
        "valid_right":       False,
        "confidence_center": 0.0,
        "confidence_left":   0.0,
        "confidence_right":  0.0,
        "timestamps_s":      [],
        "source":            "kinematic_ctr",
        "is_gt":             False,
    }


# ---------------------------------------------------------------------------
# LaneManager
# ---------------------------------------------------------------------------

class LaneManager:
    """
    Plugin-registry orchestrator for all active lane-path strategies.

    Responsibilities
    ----------------
    1. Build all plugin instances once from the Hydra config (expensive ONNX
       loads happen here, not per-frame).
    2. Maintain per-segment state via ``reset_segment_state()``.
    3. In ``process()``, iterate over plugins exactly once, merge their
       partial output dicts, and guarantee all six canonical keys are present.

    The manager has NO knowledge of file I/O, Comet ML, or visualisation.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config object (``conf/config.yaml``).
    image_width : int
    image_height : int
        Source camera resolution forwarded to plugin constructors.
    """

    def __init__(
        self,
        cfg:          DictConfig,
        image_width:  int = 1920,
        image_height: int = 1280,
    ) -> None:
        plugin_names: list[str] = list(cfg.perception.lane.active_plugins)

        if not plugin_names:
            log.warning("LaneManager: active_plugins is empty — no lane detection will run.")

        self._plugins: list[AbstractLaneDetector] = []
        for name in plugin_names:
            cls = _PLUGIN_REGISTRY.get(name)
            if cls is None:
                raise ValueError(
                    f"LaneManager: unknown plugin '{name}'. "
                    f"Registered plugins: {list(_PLUGIN_REGISTRY)}"
                )
            self._plugins.append(cls(cfg, image_width, image_height))
            log.debug("LaneManager: registered plugin '%s'", name)

        log.info(
            "LaneManager: active plugins = %s",
            " -> ".join(plugin_names) if plugin_names else "(none)",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_segment_state(self) -> None:
        """
        Flush all per-segment inter-frame state across every active plugin.

        Call once at the start of each new TFRecord segment so that EMA
        accumulators and pose-transform caches from the previous segment do
        not contaminate the current one.
        """
        for plugin in self._plugins:
            plugin.reset()

    def process(
        self,
        frame_bgr:     Optional[np.ndarray],
        vehicle_state: VehicleState,
    ) -> dict:
        """
        Run all active plugins for one frame and return the merged result.

        Each plugin contributes a subset of the six canonical output keys.
        Plugins are called in the order listed in ``active_plugins``; later
        plugins override the same key from earlier ones via ``dict.update()``.

        The returned dict always contains all six keys — missing keys are
        filled with safe empty defaults so that callers never encounter a
        ``KeyError``.

        Parameters
        ----------
        frame_bgr : np.ndarray | None
            Front-camera BGR image.  Pass ``None`` when frame extraction
            failed; visual plugins skip inference gracefully.
        vehicle_state : VehicleState
            Ego kinematics for this frame.

        Returns
        -------
        dict
            "kinematic_raw"  — raw KinematicPathPredictor output (numpy arrays).
            "kinematic"      — JSON-serializable kinematic path entry.
            "drivable_raw"   — raw drivable-area inference dict (numpy arrays) or None.
            "drivable_path"  — JSON-serializable drivable-path entry.
            "host_raw"       — raw host-lane inference dict (numpy arrays) or None.
            "host_lane"      — JSON-serializable host-lane entry.
        """
        merged: dict = {}
        for plugin in self._plugins:
            merged.update(plugin.process(frame_bgr, vehicle_state))

        # Guarantee all six canonical keys are always present in the output.
        merged.setdefault("kinematic_raw",  _empty_kinematic_raw())
        merged.setdefault("kinematic",      _empty_kinematic())
        merged.setdefault("drivable_raw",   None)
        merged.setdefault("drivable_path",  DrivablePathStrategy.package(None))
        merged.setdefault("host_raw",       None)
        merged.setdefault("host_lane",      HostLaneStrategy.package(None))

        return merged
