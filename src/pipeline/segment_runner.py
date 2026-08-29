"""
src/pipeline/segment_runner.py
================================
SegmentRunner processes one Waymo .tfrecord segment end-to-end: frame
decode, detection/tracking, lane-path inference, HD-map projection, GT/label
serialization, and Comet ML logging.

Extracted from the former pipeline_input.py::_process_segment as a pure
structural refactor; tests/test_golden_master.py guards that its output is
byte-for-byte unchanged.
"""
from __future__ import annotations

import json
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor

import tensorflow as tf
from omegaconf import DictConfig
from waymo_open_dataset import dataset_pb2 as open_dataset

from src.data.hdmap_serializer import serialize_hdmap
from src.data.waymo_parser import (
    calculate_ego_speed, extract_front_camera_image,
    extract_ground_truth_boxes, extract_gt_3d_boxes,
    parse_map_features_global, project_hdmap_lanes,
)
from src.detectors.lane import VehicleState
from src.measurements import project_bbox_to_ground
from src.utils.comet_logger import CometFrameLogger
from src.visualization.visualizer import CameraCalibration, PerceptionVisualizer

from .engine_factory import PipelineEngines


class SegmentRunner:
    """Runs the per-frame pipeline for one .tfrecord segment."""

    def __init__(
        self,
        cfg:           DictConfig,
        experiment,
        engines:       PipelineEngines,
        enabled_paths: set,
    ) -> None:
        self._cfg           = cfg
        self._experiment    = experiment
        self._engines       = engines
        self._enabled_paths = enabled_paths

    def run(
        self,
        tfrecord_path:      str,
        seg_idx:            int,
        global_step_offset: int,
    ) -> tuple[int, str]:
        """
        Process one .tfrecord segment end-to-end.

        Returns (frames_processed, json_output_path).
        """
        cfg                   = self._cfg
        experiment            = self._experiment
        lane_manager          = self._engines.lane_manager
        detector              = self._engines.detector
        track_manager         = self._engines.track_manager
        vehicle_track_manager = self._engines.vehicle_track_manager
        lane_measurer         = self._engines.lane_measurer
        enabled_paths         = self._enabled_paths

        seg_name  = os.path.splitext(os.path.basename(tfrecord_path))[0]
        seg_label = f"seg{seg_idx:03d}/{seg_name}"

        # ── Output path ──────────────────────────────────────────────────────────
        out_dir = getattr(cfg.output, "output_dir", "") or os.path.dirname(tfrecord_path)
        os.makedirs(out_dir, exist_ok=True)
        json_out = os.path.join(out_dir, seg_name + ".json")

        # Resume: skip if output already exists
        skip_existing = getattr(cfg.dataset, "skip_existing", True)
        if skip_existing and os.path.exists(json_out):
            print(f"[batch] SKIP {seg_label}  (output exists: {json_out})")
            return 0, json_out

        print(f"\n{'='*70}")
        print(f"[batch] Processing segment {seg_idx}: {seg_name}")
        print(f"{'='*70}")
        seg_t0 = _time.time()

        # Camera calibration is extracted on step==0 inside the main loop, eliminating
        # the separate .take(1) pre-scan that opened the TFRecord file twice.
        calib = CameraCalibration.default_front(image_width=1920, image_height=1280)
        vis: PerceptionVisualizer | None = None

        # Prefetch overlaps disk reads with CPU/GPU compute at zero algorithmic cost.
        dataset = (
            tf.data.TFRecordDataset(tfrecord_path, compression_type='')
            .prefetch(tf.data.AUTOTUNE)
        )

        prev_pos, prev_time = None, None
        all_frames_gt: list = []
        hdmap_global        = None
        # Reset all per-segment inter-frame state owned by the lane manager
        # (EMA accumulators, previous-pose cache).
        lane_manager.reset_segment_state()
        track_manager.reset()
        vehicle_track_manager.reset()
        prev_frame_ts: float | None = None

        # Background thread pool for non-blocking Comet image uploads.
        # The main frame loop submits tasks and continues immediately; shutdown(wait=True)
        # at segment end ensures all uploads complete before the JSON asset is logged.
        upload_executor = ThreadPoolExecutor(max_workers=4)
        comet = CometFrameLogger(experiment, upload_executor)

        for step, data in enumerate(dataset):
            if step >= cfg.dataset.max_frames:
                break

            print(f"  Frame {step}...")
            frame = open_dataset.Frame()
            frame.ParseFromString(bytes(data.numpy()))

            # Extract real camera calibration once from the first frame's context,
            # then build the visualizer — avoids a second TFRecord open for .take(1).
            if step == 0:
                for _cam_cal in frame.context.camera_calibrations:
                    if _cam_cal.name == open_dataset.CameraName.FRONT:
                        calib = CameraCalibration.from_waymo_camera(_cam_cal)
                        break
                vis = PerceptionVisualizer(calib)
                # Propagate real focal lengths to the vehicle EKF and lane measurer
                vehicle_track_manager.set_camera_params(calib.K, calib.R_vc, calib.t_vc)
                lane_measurer.update_calib(calib)

            # 1. Raw data
            img = extract_front_camera_image(frame)
            ego_speed, prev_pos, prev_time = calculate_ego_speed(frame, prev_pos, prev_time)

            # 2. Ground truth
            gt_data = extract_ground_truth_boxes(frame)
            gt_data["ego_speed_kmh"] = ego_speed
            gt_data["segment_name"]  = seg_name

            # 3. Target detection + Kalman tracking
            curr_ts       = gt_data.get("timestamp", 0.0)
            dt            = (curr_ts - prev_frame_ts) if prev_frame_ts is not None else None
            prev_frame_ts = curr_ts

            if img is not None:
                raw_dets   = detector.detect(img)
                rw_pos     = [
                    project_bbox_to_ground(d.bbox_xyxy, calib.K, calib.R_vc, calib.t_vc)
                    for d in raw_dets
                ]
                track_list         = track_manager.update(raw_dets, rw_pos, dt=dt)
                vehicle_track_states = vehicle_track_manager.update(raw_dets, dt=dt)
            else:
                raw_dets, track_list, vehicle_track_states = [], [], []

            gt_data["detections"] = [d.to_dict() for d in raw_dets]
            gt_data["tracks"]     = [t.to_dict() for t in track_list]
            gt_data["boxes_3d"]   = extract_gt_3d_boxes(frame)

            # 4. Lane strategies (Paths 1 + 3 + 4) — single manager call.
            # The LaneManager runs all active inference engines exactly once, applies
            # per-segment EMA filtering, and packages serializable result dicts.
            speed_mps      = ego_speed / 3.6
            curr_transform = list(frame.pose.transform)
            vehicle_state  = VehicleState(
                speed_mps      = speed_mps,
                curr_transform = curr_transform,
                curr_timestamp = gt_data.get("timestamp", 0.0),
            )
            lane_results   = lane_manager.process(img, vehicle_state)
            path_data      = lane_results["kinematic_raw"]
            drivable_data  = lane_results["drivable_raw"]
            host_lane_data = lane_results["host_raw"]
            ego_center_veh = path_data["centre_line"]

            hdmap_data = None

            # HD Map (Path 2) — cached from first frame with map features
            if hdmap_global is None and len(frame.map_features) > 0:
                hdmap_global = parse_map_features_global(frame)
                print(f"    HD map cached: {len(hdmap_global)} polylines")
            if hdmap_global is not None:
                hdmap_data = project_hdmap_lanes(
                    hdmap_global, frame, calib, ego_center_veh=ego_center_veh,
                )

            # ── Serialize unified PathData ─────────────────────────────────────
            # Paths 1 (kinematic), 3 (drivable_path), 4 (host_lane) are already
            # serialized by their respective strategies inside LaneManager.
            gt_data["kinematic"] = lane_results["kinematic"]
            gt_data["hdmap"]     = serialize_hdmap(hdmap_data)

            # Path 3 — Drivable Path (serialized by DrivablePathStrategy)
            gt_data["drivable_path"] = lane_results["drivable_path"]

            # Path 4 — Host Lane (serialized by HostLaneStrategy)
            gt_data["host_lane"] = lane_results["host_lane"]

            # 5. Vehicle EKF states + lane relations
            # vehicle_ekf_tracks: per-track 9D state (x,y,z,vx,vy,heading,w,h,l)
            # lane_relations: per-track distances to every active path type
            gt_data["vehicle_ekf_tracks"] = [t.to_dict() for t in vehicle_track_states]
            gt_data["lane_relations"]      = lane_measurer.compute(
                vehicle_track_states, lane_results, gt_data
            )

            all_frames_gt.append(gt_data)

            # ── Comet ML logging ────────────────────────────────────────────────
            # Global step keeps images from different segments non-overlapping in
            # the Comet timeline.
            global_step = global_step_offset + step

            comet.log_images(
                seg_name       = seg_name,
                step           = global_step,
                frame_idx      = step,
                img            = img,
                gt_data        = gt_data,
                vis            = vis,
                path_data      = path_data,
                hdmap_data     = hdmap_data,
                drivable_data  = drivable_data,
                host_lane_data = host_lane_data,
                enabled_paths  = enabled_paths,
            )
            comet.log_frame_metrics(
                step                 = global_step,
                ego_speed_kmh        = ego_speed,
                vehicle_track_states = vehicle_track_states,
            )

        frames_done = len(all_frames_gt)

        # Drain the upload queue before logging the JSON asset, so all images for
        # this segment are committed in Comet before the run is marked complete.
        upload_executor.shutdown(wait=True)

        # ── Save JSON ─────────────────────────────────────────────────────────────
        # Compact output (no indent) reduces file size by 3-5× vs indent=4 on float arrays.
        with open(json_out, "w") as f:
            json.dump(all_frames_gt, f)
        print(f"  JSON saved → {json_out}  ({frames_done} frames)")

        seg_elapsed = _time.time() - seg_t0
        comet.log_segment_summary(
            json_out    = json_out,
            seg_name    = seg_name,
            seg_elapsed = seg_elapsed,
            frames_done = frames_done,
            seg_idx     = seg_idx,
        )
        print(f"  Segment done in {seg_elapsed:.1f}s")

        return frames_done, json_out
