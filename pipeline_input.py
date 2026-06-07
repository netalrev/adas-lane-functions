from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ before Hydra resolves ${oc.env:...}

from comet_ml import Experiment  # Must be first import


# ---------------------------------------------------------------------------
# No-op experiment stub — used when COMET_API_KEY is not set so the pipeline
# runs offline without any changes to the rest of the code.
# ---------------------------------------------------------------------------
class _NullExperiment:
    """Drop-in replacement for comet_ml.Experiment that discards all data."""
    def set_name(self, *a, **kw):         pass
    def log_image(self, *a, **kw):        pass
    def log_metric(self, *a, **kw):       pass
    def log_asset(self, *a, **kw):        pass
    def end(self, *a, **kw):              pass


 # Suppress TF info/warning/error logs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import our custom modules
from src.data.waymo_parser import (
    extract_front_camera_image, calculate_ego_speed,
    extract_ground_truth_boxes, extract_gt_3d_boxes,
    parse_map_features_global, project_hdmap_lanes,
)
from src.utils.comet_logger import format_boxes_for_comet
from src.detectors.lane import LaneManager, VehicleState
from src.visualization.visualizer import CameraCalibration, PerceptionVisualizer
from src.detectors.vehicle import TargetDetector, TrackManager
from src.measurements import VehicleTrackManager, VehicleTrackState
from src.measurements import LaneRelationMeasurer
from src.measurements import project_bbox_to_ground


# Imports
from concurrent.futures import ThreadPoolExecutor
from waymo_open_dataset import dataset_pb2 as open_dataset
from omegaconf import DictConfig, OmegaConf
import tensorflow as tf
from PIL import Image
import hydra
import json
import random
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Segment discovery
# ---------------------------------------------------------------------------

def _resolve_segments(cfg: DictConfig) -> list[str]:
    """
    Return an ordered list of .tfrecord paths to process.

    Priority (first non-empty wins):
      1. dataset.segment_list  — explicit .txt file (one path per line, # ok)
      2. dataset.segment_dir   — auto-discover every *.tfrecord in a directory
      3. dataset.tfrecord_path — single legacy path

    Applies max_segments cap and optional shuffle.
    """
    ds = cfg.dataset
    paths: list[str] = []

    seg_list = getattr(ds, "segment_list", "")
    seg_dir  = getattr(ds, "segment_dir",  "")

    if seg_list:
        # Read explicit list file; skip blank lines and comments
        with open(seg_list) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    paths.append(line)
        print(f"[batch] Segment list file: {seg_list} → {len(paths)} entries")

    elif seg_dir:
        # Auto-discover all .tfrecord files recursively
        for root, _, files in os.walk(seg_dir):
            for fname in sorted(files):
                if fname.endswith(".tfrecord"):
                    paths.append(os.path.join(root, fname))
        print(f"[batch] Discovered {len(paths)} tfrecords under: {seg_dir}")

    else:
        single = getattr(ds, "tfrecord_path", "")
        if single:
            paths = [single]
        else:
            raise ValueError(
                "No segment source configured. Set one of: "
                "dataset.tfrecord_path, dataset.segment_dir, dataset.segment_list"
            )

    if not paths:
        raise ValueError("Segment source resolved to zero files.")

    # Filter out paths that don't exist on disk yet (e.g. not downloaded yet).
    # This allows a segment_list to contain future segments without crashing.
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"[batch] WARNING: file not found, skipping — {p}")
        paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(
            "All resolved segments are missing from disk.\n"
            "Download them first — see 'dataset.gcs_prefix' in conf/config.yaml "
            "and run:\n"
            "  gsutil -m cp '<gcs_prefix>/segment-*.tfrecord' <local_dir>/"
        )

    # Optional shuffle before capping (for reproducible random sampling)
    shuffle = getattr(ds, "shuffle_segments", False)
    if shuffle:
        random.shuffle(paths)

    # Cap to max_segments
    max_seg = getattr(ds, "max_segments", 1)
    if max_seg is not None and len(paths) > max_seg:
        paths = paths[:max_seg]

    return paths


# ---------------------------------------------------------------------------
# Per-segment processor
# ---------------------------------------------------------------------------

def _process_segment(
    tfrecord_path:         str,
    seg_idx:               int,
    cfg:                   DictConfig,
    experiment:            Experiment,
    lane_manager:          LaneManager,
    detector:              TargetDetector,
    track_manager:         TrackManager,
    vehicle_track_manager: VehicleTrackManager,
    lane_measurer:         LaneRelationMeasurer,
    enabled_paths:         set,
    global_step_offset:    int,
) -> tuple[int, str]:
    """
    Process one .tfrecord segment end-to-end.

    Returns (frames_processed, json_output_path).
    """
    import time as _time
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
    _upload_executor = ThreadPoolExecutor(max_workers=4)

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

        def _pts(arr) -> list:
            if arr is None:
                return []
            a = arr if not hasattr(arr, 'tolist') else arr.tolist()
            return a if len(a) >= 2 else []

        def _conf(v) -> float:
            return float(v) if v is not None else 0.0

        # Path 2 — HD Map
        if hdmap_data is not None:
            ll = _pts(hdmap_data["left_lane"])
            rl = _pts(hdmap_data["right_lane"])
        else:
            ll, rl = [], []
        hm_center: list = []
        if len(ll) >= 2 and len(rl) >= 2:
            ll_arr = np.array(ll, dtype=np.float64)
            rl_arr = np.array(rl, dtype=np.float64)
            y_lo   = float(max(ll_arr[:, 1].min(), rl_arr[:, 1].min()))
            y_hi   = float(min(ll_arr[:, 1].max(), rl_arr[:, 1].max()))
            if y_hi > y_lo:
                ll_s = ll_arr[np.argsort(ll_arr[:, 1])]
                rl_s = rl_arr[np.argsort(rl_arr[:, 1])]
                y_c  = np.linspace(y_lo, y_hi, 30)
                xl   = np.interp(y_c, ll_s[:, 1], ll_s[:, 0])
                xr   = np.interp(y_c, rl_s[:, 1], rl_s[:, 0])
                hm_center = np.column_stack(
                    [((xl + xr) / 2).astype(np.int32), y_c.astype(np.int32)]
                ).tolist()
        gt_data["hdmap"] = {
            "center":            hm_center,
            "left":              ll,
            "right":             rl,
            "valid_center":      len(hm_center) >= 2,
            "valid_left":        len(ll) >= 2,
            "valid_right":       len(rl) >= 2,
            "confidence_center": 1.0 if len(hm_center) >= 2 else 0.0,
            "confidence_left":   1.0 if len(ll) >= 2 else 0.0,
            "confidence_right":  1.0 if len(rl) >= 2 else 0.0,
            "timestamps_s":      [],
            "source":            "hdmap",
            "is_gt":             True,
        }

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

        if img is not None:
            raw_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            comet_annotations = format_boxes_for_comet(gt_data.get("boxes_2d", []))
            _upload_executor.submit(
                experiment.log_image,
                raw_pil,
                name=f"[{seg_name}] Raw_Front_Camera",
                step=global_step,
                annotations=comet_annotations,
            )

            annotated_img = vis.draw_all(
                img, gt_data, path_data, hdmap_data,
                frame_idx=step,
                drivable_data=drivable_data,
                host_lane_data=host_lane_data,
                enabled_paths=enabled_paths,
            )
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            _upload_executor.submit(
                experiment.log_image,
                annotated_pil,
                name=f"[{seg_name}] Annotated_Front_Camera",
                step=global_step,
            )

            # Third image: YOLO detections + Kalman tracks on the raw frame.
            # Kept separate from Annotated_Front_Camera so GT and network
            # outputs can be compared side-by-side in the Comet image gallery.
            yolo_canvas = vis.draw_detections_and_tracks(img.copy(), gt_data)
            yolo_pil    = Image.fromarray(cv2.cvtColor(yolo_canvas, cv2.COLOR_BGR2RGB))
            _upload_executor.submit(
                experiment.log_image,
                yolo_pil,
                name=f"[{seg_name}] YOLO_Detections",
                step=global_step,
            )

            # Fourth image: GT boxes + lanes + YOLO detections/tracks combined.
            # This single image lets researchers compare ground-truth labels
            # against the network's online output without switching tabs in Comet.
            combined_canvas = vis.draw_detections_and_tracks(annotated_img.copy(), gt_data)
            combined_pil    = Image.fromarray(cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB))
            _upload_executor.submit(
                experiment.log_image,
                combined_pil,
                name=f"[{seg_name}] Combined_GT_and_Predictions",
                step=global_step,
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
            _upload_executor.submit(
                experiment.log_image,
                ekf_pil,
                name=f"[{seg_name}] Vehicle_EKF_Tracks",
                step=global_step,
            )

        # ── Per-frame scalar metrics ────────────────────────────────────────
        experiment.log_metric("ego_speed_kmh",    ego_speed,                    step=global_step)
        experiment.log_metric("ekf_vehicle_count", len(vehicle_track_states),   step=global_step)

        # Log the CIPV (Closest In-Path Vehicle) state each frame so Comet
        # charts show range, speed, and TTC as time-series for every run.
        if vehicle_track_states:
            cipv = min(vehicle_track_states, key=lambda t: t.x_veh)
            experiment.log_metric("cipv_range_m",   cipv.x_veh,    step=global_step)
            experiment.log_metric("cipv_lateral_m", cipv.y_veh,    step=global_step)
            experiment.log_metric("cipv_speed_mps", cipv.speed_mps, step=global_step)
            if cipv.ttc_s != float('inf'):
                experiment.log_metric("cipv_ttc_s", cipv.ttc_s, step=global_step)
            experiment.log_metric("cipv_width_m",  cipv.width_m,  step=global_step)
            experiment.log_metric("cipv_height_m", cipv.height_m, step=global_step)

    frames_done = len(all_frames_gt)

    # Drain the upload queue before logging the JSON asset, so all images for
    # this segment are committed in Comet before the run is marked complete.
    _upload_executor.shutdown(wait=True)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    # Compact output (no indent) reduces file size by 3-5× vs indent=4 on float arrays.
    with open(json_out, "w") as f:
        json.dump(all_frames_gt, f)
    print(f"  JSON saved → {json_out}  ({frames_done} frames)")
    experiment.log_asset(json_out, file_name=seg_name + ".json")

    seg_elapsed = _time.time() - seg_t0
    experiment.log_metric("segment_duration_s", seg_elapsed, step=seg_idx)
    experiment.log_metric("segment_frames",     frames_done, step=seg_idx)
    print(f"  Segment done in {seg_elapsed:.1f}s")

    return frames_done, json_out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import time as _time
    print("--- Starting ADAS/AV Batch Input Pipeline ---")
    print(OmegaConf.to_yaml(cfg))

    _api_key = cfg.comet.api_key  # empty string when COMET_API_KEY is not set
    if _api_key:
        experiment = Experiment(
            api_key      = _api_key,
            project_name = cfg.comet.project_name,
            workspace    = cfg.comet.workspace,
        )
        experiment.set_name(cfg.comet.experiment_name)
        print("[comet] Logging to Comet ML experiment.")
    else:
        experiment = _NullExperiment()
        print("[comet] COMET_API_KEY not set — running offline (no remote logging).")

    # ── Resolve segment list ─────────────────────────────────────────────────
    segments = _resolve_segments(cfg)
    print(f"\n[batch] Will process {len(segments)} segment(s):")
    for i, p in enumerate(segments):
        print(f"  [{i:03d}] {os.path.basename(p)}")

    # ── Build all inference engines once (expensive ONNX loads) ─────────────────
    lane_manager          = LaneManager(cfg)
    detector              = TargetDetector(cfg.perception.detector)
    track_manager         = TrackManager(cfg.perception.tracker)
    vehicle_track_manager = VehicleTrackManager(cfg.perception.vehicle_ekf)
    lane_measurer         = LaneRelationMeasurer(
        CameraCalibration.default_front(image_width=1920, image_height=1280)
    )

    _viz_cfg = getattr(cfg, "visualization", None)
    _ep_list = list(getattr(_viz_cfg, "enabled_paths",
                            ["kinematic", "drivable_path", "host_lane"])) \
               if _viz_cfg else ["kinematic", "drivable_path", "host_lane"]
    enabled_paths: set = set(_ep_list)

    # ── Segment loop ─────────────────────────────────────────────────────────
    total_t0     = _time.time()
    global_step  = 0
    total_frames = 0

    for seg_idx, tfrecord_path in enumerate(segments):
        frames_done, _ = _process_segment(
            tfrecord_path         = tfrecord_path,
            seg_idx               = seg_idx,
            cfg                   = cfg,
            experiment            = experiment,
            lane_manager          = lane_manager,
            detector              = detector,
            track_manager         = track_manager,
            vehicle_track_manager = vehicle_track_manager,
            lane_measurer         = lane_measurer,
            enabled_paths         = enabled_paths,
            global_step_offset    = global_step,
        )
        global_step  += cfg.dataset.max_frames   # reserve step space per segment
        total_frames += frames_done

    elapsed = _time.time() - total_t0
    print(f"\n[batch] All done — {len(segments)} segment(s), "
          f"{total_frames} frames in {elapsed:.1f}s "
          f"({total_frames / max(elapsed, 1):.1f} fps)")
    experiment.log_metric("total_frames",     total_frames)
    experiment.log_metric("total_duration_s", elapsed)
    experiment.end()
    print("Check your Comet ML dashboard!")


if __name__ == "__main__":
    main()

