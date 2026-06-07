"""
src/models/lanes/base.py
========================
Plugin contract for all lane-detection strategy implementations.

Every lane-detection plugin (Kinematic, IPM, CLRNet, YOLOPv2) must
subclass ``AbstractLaneDetector`` and implement the three methods defined
here.  The ``LaneManager`` reads the ``active_plugins`` list from the Hydra
config and instantiates each plugin dynamically — no hardcoded routing is
required in the orchestrator.

Public surface
--------------
VehicleState          — per-frame ego-kinematics passed to every plugin.
AbstractLaneDetector  — ABC that all plugins must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from omegaconf import DictConfig


@dataclass
class VehicleState:
    """
    Per-frame ego-vehicle state consumed by all lane-detection plugins.

    All quantities are expressed in SI units.

    Attributes
    ----------
    speed_mps : float
        Longitudinal speed [m/s].  Positive = forward.
    curr_transform : list[float]
        Flat 16-element row-major 4x4 Waymo pose transform for this frame,
        mapping Vehicle Frame to Global Frame.
    curr_timestamp : float
        Frame timestamp [s].
    """

    speed_mps:      float
    curr_transform: list
    curr_timestamp: float


class AbstractLaneDetector(ABC):
    """
    Plugin contract for all lane-path detection strategies.

    Sub-classes
    -----------
    KinematicPlugin  — CTR arc prediction from pose matrices.
    IPMPlugin        — Classical BEV + sliding-window polynomial fit.
    CLRNetPlugin     — ONNX-Runtime CLRNet neural lane detector.
    YOLOPv2Plugin    — Multi-task YOLOPv2: drivable area + lane lines.

    Lifecycle
    ---------
    1. ``__init__`` is called once at pipeline startup.  All expensive
       operations (ONNX loads, kernel pre-compilation) happen here.
    2. ``process()`` is called once per frame.
    3. ``reset()`` is called once at the start of each new TFRecord segment.

    Output contract
    ---------------
    ``process()`` returns a dict containing any subset of the six canonical
    output keys.  The ``LaneManager`` merges plugin outputs via
    ``dict.update()`` in the order listed in ``active_plugins``; a plugin
    listed later overrides the same key from an earlier plugin.

    Canonical keys
    ~~~~~~~~~~~~~~
    "kinematic_raw"  — raw ``KinematicPathPredictor.predict()`` output (numpy arrays).
    "kinematic"      — JSON-serializable kinematic path entry.
    "drivable_raw"   — raw drivable-area inference dict (numpy arrays).
    "drivable_path"  — JSON-serializable drivable-path entry.
    "host_raw"       — raw host-lane inference dict (numpy arrays).
    "host_lane"      — JSON-serializable host-lane entry.
    """

    @abstractmethod
    def __init__(
        self,
        cfg:          DictConfig,
        image_width:  int = 1920,
        image_height: int = 1280,
    ) -> None:
        """
        Construct the plugin and load all weights / resources.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config root node (``conf/config.yaml``).
        image_width : int
            Source camera width [px].
        image_height : int
            Source camera height [px].
        """

    @abstractmethod
    def process(
        self,
        frame_bgr:     Optional[np.ndarray],
        vehicle_state: VehicleState,
    ) -> dict:
        """
        Run the plugin for one frame and return a partial result dict.

        Visual plugins must return ``{}`` gracefully when ``frame_bgr`` is
        ``None`` (e.g. when frame extraction fails upstream).

        Parameters
        ----------
        frame_bgr : np.ndarray | None
            BGR image (H, W, 3) uint8.
        vehicle_state : VehicleState
            Per-frame ego kinematics.

        Returns
        -------
        dict
            Any subset of the six canonical output keys described in the
            class docstring.
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Flush all inter-frame temporal state.

        Called once at the boundary between TFRecord segments so that EMA
        accumulators, pose caches, and temporal persistence windows from one
        recording do not bleed into the first frames of the next.
        """
