"""
src/measurements/__init__.py
==============================
Measurements package — physics-based state estimation and spatial relation
computations that run after detection.

Modules
-------
rw_coordinates         — ground-plane back-projection of bbox bottom-centre.
vehicle_ekf            — 9D Extended Kalman Filter for a single vehicle track.
vehicle_measurement    — SF geometry front-end: bbox → 3D EKF measurement.
vehicle_track_manager  — EKF-based multi-vehicle tracker (vehicles only).
lane_relations         — per-track lateral distance to every active path type.
"""
from .rw_coordinates       import project_bbox_to_ground
from .vehicle_ekf          import VehicleEKF
from .vehicle_measurement  import (
    SFPositionEstimate,
    SFMeasurementBundle,
    KalmanMeasurement,
    compute_sf_measurements,
    build_kalman_input,
)
from .vehicle_track_manager import VehicleTrackManager, VehicleTrackState
from .lane_relations        import LaneRelationMeasurer, LaneRelation

__all__ = [
    "project_bbox_to_ground",
    "VehicleEKF",
    "SFPositionEstimate",
    "SFMeasurementBundle",
    "KalmanMeasurement",
    "compute_sf_measurements",
    "build_kalman_input",
    "VehicleTrackManager",
    "VehicleTrackState",
    "LaneRelationMeasurer",
    "LaneRelation",
]
