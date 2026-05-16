# Architecture Overview

## System Purpose

An end-to-end batch perception pipeline that ingests Waymo Open Dataset TFRecords, runs four parallel lane-path strategies per frame, serialises results to JSON, and logs annotated imagery to Comet ML.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────┐
│  conf/  (Hydra config)                                       │
│  ├── config.yaml          ← composition root + defaults list │
│  ├── dataset/waymo.yaml   → cfg.dataset.*                    │
│  ├── model/lane.yaml      → cfg.perception.lane.*            │
│  ├── model/kinematic.yaml → cfg.perception.kinematic_path.*  │
│  └── logger/comet.yaml    → cfg.comet.*                      │
└──────────────────────┬──────────────────────────────────────┘
                       │ single cfg object (DictConfig)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  pipeline_input.py  (Hydra entry point)                      │
│  ├── _resolve_segments()   reads segments_to_run.txt         │
│  ├── LaneManager(cfg)      builds all inference engines once │
│  └── _process_segment()    frame loop per TFRecord           │
└──────┬──────────┬──────────────────────┬────────────────────┘
       │          │                      │
       ▼          ▼                      ▼
┌──────────┐ ┌────────────────────┐ ┌───────────────────────┐
│ Waymo    │ │ LaneManager        │ │ PerceptionVisualizer  │
│ Parser   │ │ src/models/lanes/  │ │ src/visualization/    │
│          │ │                    │ │ visualizer.py         │
│ • image  │ │ Path 1: Kinematic  │ │                       │
│ • GT 2D  │ │ Path 2: HD Map*    │ │ • GT boxes (2D)       │
│ • speed  │ │ Path 3: Drivable   │ │ • Kinematic path      │
│ • pose   │ │ Path 4: Host Lane  │ │ • Drivable path       │
│ • map    │ │                    │ │ • Host lane           │
└──────────┘ └─────────┬──────────┘ │ • HUD overlay         │
                        │            └──────────┬────────────┘
                        ▼                       │
              ┌─────────────────┐               │
              │  JSON output    │               ▼
              │  per segment    │    ┌──────────────────────┐
              │  (gt_data[])    │    │  Comet ML logger     │
              └─────────────────┘    │  • raw + annotated   │
                                     │    images per frame  │
                                     │  • ego speed metric  │
                                     │  • segment duration  │
                                     └──────────────────────┘
```

> \* HD Map (Path 2) is parsed directly from Waymo `map_features` proto in `waymo_parser.py`, not managed by `LaneManager`.

---

## Component Responsibilities

### `conf/`  — Configuration

- Hydra composition root. `config.yaml` uses the `defaults` list; each group YAML holds only the keys relevant to that domain.
- The entire codebase treats `cfg` as the **single source of truth**. No magic constants exist outside YAML.
- CLI overrides: `python pipeline_input.py dataset.max_frames=10 comet.experiment_name="run_v2"`

### `pipeline_input.py`  — Orchestrator

- `_resolve_segments(cfg)` — resolves `segments_to_run.txt` → list of `.tfrecord` paths. Supports three source modes (list file, directory scan, single path).
- `main(cfg)` — builds `LaneManager` once (all ONNX loads happen here), then calls `_process_segment` for each TFRecord in sequence.
- `_process_segment()` — per-segment frame loop: parse frame → call `LaneManager.process()` → serialize to `gt_data` dict → log to Comet ML. Resets `LaneManager` state between segments.

### `src/data/waymo_parser.py`  — Data Parser

| Function | Output |
|---|---|
| `extract_front_camera_image` | BGR `np.ndarray` from JPEG-encoded proto |
| `calculate_ego_speed` | speed in km/h via finite-difference of pose matrix translation |
| `extract_ground_truth_boxes` | 2D box list + timestamp from camera label proto |
| `parse_map_features_global` | HD Map polylines in Global Frame (cached once per segment) |
| `project_hdmap_lanes` | HD Map polylines transformed to Vehicle Frame + projected to BEV |

### `src/models/lanes/`  — Lane Manager (Strategy Pattern)

See [02_Lane_Calculations.md](02_Lane_Calculations.md) for algorithm detail.

| Class | File | Role |
|---|---|---|
| `LaneManager` | `manager.py` | Owns ONNX engines; single `process()` call per frame |
| `VehicleState` | `manager.py` | Dataclass: `speed_mps`, `curr_transform`, `curr_timestamp` |
| `KinematicEgoStrategy` | `kinematic_ego.py` | Stateful: EMA + CTR predictor |
| `DrivablePathStrategy` | `visual_dp.py` | Stateless packager for YOLOPv2 DA output |
| `HostLaneStrategy` | `visual_host.py` | Stateless packager for YOLOPv2 LL output |

### `src/visualization/visualizer.py`  — Visualizer

- `CameraCalibration` — holds intrinsics (`fx, fy, cx, cy`) and `[R|t]` extrinsics (Vehicle → Camera transform). Constructed from the Waymo camera calibration proto or from hardcoded defaults.
- `PerceptionVisualizer.draw_all()` — composites all annotation layers onto a copy of the frame (never mutates the original).
- **3D→2D projection pipeline**: Vehicle Frame → Camera Frame (`[R|t]`) → strict Z > 1.0 m clip → `cv2.projectPoints` → image-bounds filter.

### `src/utils/comet_logger.py`  — Logger Adapter

- Wraps Comet ML's `Experiment` API.
- `format_boxes_for_comet()` converts the internal box list to Comet annotation format for bounding-box overlays on logged images.

---

## Per-Frame Data Flow

```
TFRecord frame
    │
    ├─► extract_front_camera_image()   →  img (BGR ndarray)
    ├─► calculate_ego_speed()          →  ego_speed (km/h)
    ├─► extract_ground_truth_boxes()   →  gt_data dict
    │
    ├─► LaneManager.process(img, VehicleState)
    │       ├── KinematicEgoStrategy.compute()   → kinematic_raw + kinematic (JSON)
    │       ├── YOLOPv2.detect_full(img)          → drivable_raw, host_raw
    │       ├── DrivablePathStrategy.package()   → drivable_path (JSON)
    │       └── HostLaneStrategy.package()       → host_lane (JSON)
    │
    ├─► project_hdmap_lanes()          → hdmap_data
    │
    ├─► Assemble gt_data["kinematic" | "hdmap" | "drivable_path" | "host_lane"]
    │
    ├─► PerceptionVisualizer.draw_all()  → annotated BGR image
    │
    └─► Comet ML: log raw image, annotated image, ego_speed metric
```

---

## Key Design Constraints

- **ONNX inference engines are constructed once** in `main()` and reused across all segments. Avoid moving them into the per-frame loop.
- **YOLOPv2 runs a single forward pass** per frame — its two output heads (drivable area + lane lines) are consumed by separate packaging strategies without a second inference call.
- **`cfg` is the only config surface.** Hardcoded thresholds or paths anywhere in `src/` are a bug.
- **Coordinate frames**: Vehicle Frame = X forward, Y left, Z up. Camera Frame = Z forward, X right, Y down (standard OpenCV). All 3D path points are expressed in Vehicle Frame and projected to pixel space by the visualizer.
