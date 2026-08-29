"""
src/pipeline/engine_factory.py
================================
Builds every inference engine exactly once per pipeline run (expensive
ONNX/model loads happen here, not per-segment or per-frame).
"""
from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from src.detectors.lane import LaneManager
from src.detectors.vehicle import TargetDetector, TrackManager
from src.measurements import LaneRelationMeasurer, VehicleTrackManager
from src.visualization.visualizer import CameraCalibration


@dataclass
class PipelineEngines:
    """Bundle of the long-lived inference engines shared across all segments."""
    lane_manager:          LaneManager
    detector:              TargetDetector
    track_manager:         TrackManager
    vehicle_track_manager: VehicleTrackManager
    lane_measurer:         LaneRelationMeasurer


def build_engines(cfg: DictConfig) -> PipelineEngines:
    """Construct all inference engines once (expensive ONNX loads) for the whole batch run."""
    return PipelineEngines(
        lane_manager=LaneManager(cfg),
        detector=TargetDetector(cfg.perception.detector),
        track_manager=TrackManager(cfg.perception.tracker),
        vehicle_track_manager=VehicleTrackManager(cfg.perception.vehicle_ekf),
        lane_measurer=LaneRelationMeasurer(
            CameraCalibration.default_front(image_width=1920, image_height=1280)
        ),
    )
