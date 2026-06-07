"""
src/models/lanes/__init__.py
============================
Lane strategy sub-package.

Public surface
--------------
    LaneManager   — orchestrator; owns all plugin instances.
    VehicleState  — lightweight per-frame ego-kinematics dataclass.
"""

from .base    import VehicleState
from .manager import LaneManager

__all__ = ["LaneManager", "VehicleState"]
