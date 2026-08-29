"""
src/utils/comet_logger.py
===========================
Comet ML logging helpers for the batch perception pipeline.

NullExperiment
    Drop-in replacement for comet_ml.Experiment that discards all data --
    used when COMET_API_KEY is not set so the pipeline runs fully offline.

CometFrameLogger
    Wraps the per-frame Comet ML image/metric uploads used by
    src.pipeline.segment_runner.SegmentRunner. Uploads run on a
    caller-supplied ThreadPoolExecutor so the frame loop never blocks on
    network I/O.

format_boxes_for_comet(gt_boxes)
    Converts Waymo GT 2D boxes into the Comet ML image-annotation schema.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import cv2
from PIL import Image

if TYPE_CHECKING:
    from src.visualization.visualizer import PerceptionVisualizer


class NullExperiment:
    """Drop-in replacement for comet_ml.Experiment that discards all data."""
    def set_name(self, *a, **kw):         pass
    def log_image(self, *a, **kw):        pass
    def log_metric(self, *a, **kw):       pass
    def log_asset(self, *a, **kw):        pass
    def end(self, *a, **kw):              pass


def format_boxes_for_comet(gt_boxes):
    """
    Converts Ground Truth box data (center to top-left)
    to fit the Comet ML annotation schema.
    """
    annotations = []

    # Map Waymo object types to readable strings
    type_map = {
        1: "Vehicle",
        2: "Pedestrian",
        3: "Sign",
        4: "Cyclist"
    }

    for box in gt_boxes:
        top_left_x = box["center_x"] - (box["length"] / 2.0)
        top_left_y = box["center_y"] - (box["width"] / 2.0)

        class_name = type_map.get(box["type"], "Unknown")
        short_id = box["id"][:6]

        annotations.append({
            "name": class_name,
            "data": [{
                "label": f"ID:{short_id}",
                "boxes": [[top_left_x, top_left_y, box["length"], box["width"]]]
            }]
        })
    return annotations


class CometFrameLogger:
    """Uploads per-frame Comet ML images and scalar metrics on a background executor."""

    def __init__(self, experiment, executor: ThreadPoolExecutor) -> None:
        self._experiment = experiment
        self._executor   = executor

    def log_images(
        self,
        *,
        seg_name:       str,
        step:           int,
        frame_idx:      int,
        img,
        gt_data:        dict,
        vis:            "PerceptionVisualizer",
        path_data:      dict,
        hdmap_data,
        drivable_data:  dict,
        host_lane_data: dict,
        enabled_paths:  set,
    ) -> None:
        """Draw and submit the five per-frame debug images, if a camera frame exists."""
        if img is None:
            return

        raw_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        comet_annotations = format_boxes_for_comet(gt_data.get("boxes_2d", []))
        self._executor.submit(
            self._experiment.log_image,
            raw_pil,
            name=f"[{seg_name}] Raw_Front_Camera",
            step=step,
            annotations=comet_annotations,
        )

        annotated_img = vis.draw_all(
            img, gt_data, path_data, hdmap_data,
            frame_idx=frame_idx,
            drivable_data=drivable_data,
            host_lane_data=host_lane_data,
            enabled_paths=enabled_paths,
        )
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        self._executor.submit(
            self._experiment.log_image,
            annotated_pil,
            name=f"[{seg_name}] Annotated_Front_Camera",
            step=step,
        )

        # Third image: YOLO detections + Kalman tracks on the raw frame.
        # Kept separate from Annotated_Front_Camera so GT and network
        # outputs can be compared side-by-side in the Comet image gallery.
        yolo_canvas = vis.draw_detections_and_tracks(img.copy(), gt_data)
        yolo_pil    = Image.fromarray(cv2.cvtColor(yolo_canvas, cv2.COLOR_BGR2RGB))
        self._executor.submit(
            self._experiment.log_image,
            yolo_pil,
            name=f"[{seg_name}] YOLO_Detections",
            step=step,
        )

        # Fourth image: GT boxes + lanes + YOLO detections/tracks combined.
        # This single image lets researchers compare ground-truth labels
        # against the network's online output without switching tabs in Comet.
        combined_canvas = vis.draw_detections_and_tracks(annotated_img.copy(), gt_data)
        combined_pil    = Image.fromarray(cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB))
        self._executor.submit(
            self._experiment.log_image,
            combined_pil,
            name=f"[{seg_name}] Combined_GT_and_Predictions",
            step=step,
        )

        # Fifth image: Vehicle EKF tracks with lane-relation color coding.
        # Green bbox = inside ego lane, orange = adjacent, red = far outside.
        # Each track carries a velocity arrow + 9D state panel (speed, TTC,
        # 3D size, forward/lateral position, lane side).
        ekf_canvas = vis.draw_vehicle_ekf_tracks(
            annotated_img.copy(),
            gt_data.get("vehicle_ekf_tracks", []),
            gt_data.get("lane_relations",     []),
        )
        ekf_pil = Image.fromarray(cv2.cvtColor(ekf_canvas, cv2.COLOR_BGR2RGB))
        self._executor.submit(
            self._experiment.log_image,
            ekf_pil,
            name=f"[{seg_name}] Vehicle_EKF_Tracks",
            step=step,
        )

    def log_frame_metrics(
        self, *, step: int, ego_speed_kmh: float, vehicle_track_states: list,
    ) -> None:
        """Log per-frame scalar metrics, including the CIPV time series."""
        self._experiment.log_metric("ego_speed_kmh",     ego_speed_kmh,             step=step)
        self._experiment.log_metric("ekf_vehicle_count", len(vehicle_track_states), step=step)

        # Log the CIPV (Closest In-Path Vehicle) state each frame so Comet
        # charts show range, speed, and TTC as time-series for every run.
        if vehicle_track_states:
            cipv = min(vehicle_track_states, key=lambda t: t.x_veh)
            self._experiment.log_metric("cipv_range_m",   cipv.x_veh,     step=step)
            self._experiment.log_metric("cipv_lateral_m", cipv.y_veh,     step=step)
            self._experiment.log_metric("cipv_speed_mps", cipv.speed_mps, step=step)
            if cipv.ttc_s != float('inf'):
                self._experiment.log_metric("cipv_ttc_s", cipv.ttc_s, step=step)
            self._experiment.log_metric("cipv_width_m",  cipv.width_m,  step=step)
            self._experiment.log_metric("cipv_height_m", cipv.height_m, step=step)

    def log_segment_summary(
        self, *, json_out: str, seg_name: str, seg_elapsed: float, frames_done: int, seg_idx: int,
    ) -> None:
        """Log the segment JSON asset and segment-level duration/frame-count metrics."""
        self._experiment.log_asset(json_out, file_name=seg_name + ".json")
        self._experiment.log_metric("segment_duration_s", seg_elapsed, step=seg_idx)
        self._experiment.log_metric("segment_frames",     frames_done, step=seg_idx)