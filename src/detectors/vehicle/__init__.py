"""
src/detectors/vehicle/__init__.py
===================================
Vehicle detection and Kalman-based ID tracking.

TargetDetector   — YOLOv8n ONNX single-frame detector.
KalmanTracker    — per-track 4D constant-velocity Kalman filter.
TrackManager     — multi-class IoU + Hungarian assignment tracker;
                   maintains persistent track IDs and coasting across
                   missed frames.
"""
from .detector       import Detection, TargetDetector
from .kalman_tracker import KalmanTracker
from .track_manager  import TrackManager, TrackState

__all__ = [
    "Detection",
    "TargetDetector",
    "KalmanTracker",
    "TrackManager",
    "TrackState",
]
