# Changelog

All notable changes to **adas-lane-functions** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

---

## [0.4.0] — 2026-05-15

### Added — Phase 5: Deployment

- **`src/inference/export_onnx.py`** — `ModelExporter`: loads a `MFTransformer` checkpoint, exports to ONNX (opset 12, dynamic batch), and numerically verifies the exported graph against PyTorch (atol 1e-4).
- **`src/inference/quantize.py`** — `ModelQuantizer`: applies ONNX Runtime dynamic INT8 quantization and gates accuracy via a CIPV-F1 drop check (≤ 2 pp threshold, 2 000 test samples).
- **`src/inference/__init__.py`** — Package exports `ModelExporter`, `ModelQuantizer`.
- **`scripts/export_model.py`** — Argparse CLI: full export pipeline — checkpoint → FP32 ONNX → verify → INT8 quantize → accuracy gate.
- **`cpp/CMakeLists.txt`** — C++ build config (CMake 3.18+, C++17, `-O3`). Builds `adas_inference` static library and `adas_infer` executable. FetchContent for `nlohmann/json v3.11.3`.
- **`cpp/include/adas_inference.hpp`** — Public C++ API: `TrackInput`, `TrackPrediction`, `AdasInference` class. Constants `CIPV_THRESHOLD`, `CUT_IN_THRESHOLD`, `N_LANE_CLASSES`, lane offset map.
- **`cpp/src/adas_inference.cpp`** — `AdasInference` implementation: batches all tracks into `[N,T,D]` float32 tensor, single ORT session call, per-track sigmoid/softmax/argmax post-processing, single-CIPV assignment.
- **`cpp/src/main.cpp`** — CLI driver: reads `{frame_idx, tracks[{track_id, mf_window}]}` JSON, calls `AdasInference::run()`, writes predictions JSON to stdout.

### Fixed
- `requirements.txt`: corrected `tensorflow` pin to `2.13.0` (actual installed version).
- `.gitignore`: added `cpp/build/` and `cpp/_deps/` entries for C++ build artefacts.

---

## [0.3.0] — 2026-05-15

### Added — Phase 4: Model Training

- **`src/models/classification/transformer.py`** — `MFTransformer`: Pre-LN Transformer encoder (sinusoidal PE, learnable CLS token, 202 055 params) producing `cipv_logit [B,1]`, `lane_logit [B,5]`, `cut_in_logit [B,1]`.
- **`src/models/classification/heads.py`** — `CIPVHead`, `LaneAssignHead` (5-class softmax), `CutInHead`.
- **`src/training/dataset.py`** — `MFDataset`: segment-level train/val/test split over HDF5, preloads to RAM.
- **`src/training/trainer.py`** — `Trainer`: AdamW + cosine LR + linear warmup + gradient clipping + epoch checkpointing.
- **`src/evaluation/metrics.py`** — `binary_metrics`, `multiclass_metrics`, `confusion_matrix_counts`.
- **`src/evaluation/report.py`** — `ReportWriter`: text report, `.npy` confusion matrix, JSON metrics summary.
- **`scripts/train.py`** — Hydra entry point: train loop + held-out test evaluation.
- **`scripts/hpsearch.py`** — Optuna HP search with Hydra config overrides.
- **`conf/training/default.yaml`** — Full training config (batch, LR, scheduler, grad clip, heads).
- **`conf/training/hpsearch.yaml`** — Optuna search space (LR, depth, heads, dropout).
- **`conf/train_config.yaml`** — Hydra composition root for training.

---

## [0.2.0] — 2026-05-15

### Added — Phase 3: Feature Engineering & GT Builder

- **`src/features/mf_assembler.py`** — `MFAssembler`: rolling T=10 deque, produces 18-feature measurement vector per target per frame.
- **`src/features/gt_builder.py`** — `GTBuilder`: derives CIPV / Lane Assignment / Cut-In labels from Waymo GT 3D boxes and ego pose (no manual annotation required).
- **`src/features/dataset_writer.py`** — `DatasetWriter`: chunked HDF5 writer with resize-and-append for streaming data.
- **`src/features/__init__.py`** — Package exports `MFAssembler`, `GTBuilder`, `DatasetWriter`.
- **`scripts/build_dataset.py`** — Offline CLI: JSON segment replay → HDF5 training dataset.
- **`conf/features/mf.yaml`** — Feature config (`window_size: 10`, `feature_dim: 18`, lane-distance bins, etc.).
- **`src/data/waymo_parser.py`** — Added `extract_gt_3d_boxes(frame)` helper.

### Added — Phase 2: Detection & Tracking

- **`src/models/detection/detector.py`** — `TargetDetector`: YOLOv8n ONNX inference, letterbox pre-processing, NMS post-processing.
- **`src/models/tracking/kalman_tracker.py`** — `KalmanTracker`: 4-state linear Kalman filter (x, y, vx, vy) with Hungarian assignment.
- **`src/models/tracking/track_manager.py`** — `TrackManager`: manages tracklet lifecycle (birth/update/coasting/death) and wraps state in `Track` dataclass.
- **`src/features/rw_coordinates.py`** — `RWCoordinates`: pixel detections → vehicle-frame real-world (X forward, Y left, Z up) using camera intrinsics + ego pose.
- **`conf/model/detector.yaml`**, **`conf/model/tracker.yaml`** — Detection and tracking config.
- **`scripts/download_models.py`** — Downloads YOLOv8n and YOLOPv2 ONNX weights to `src/data/models/`.

---

## [0.1.0] — 2026-05-15

### Added — Phase 1: Perception Pipeline

- **`pipeline_input.py`** — Hydra batch orchestrator. Reads `segments_to_run.txt`, builds all inference engines once, processes each TFRecord segment in sequence.
- **`src/data/waymo_parser.py`** — Waymo TFRecord parser: front-camera image extraction, GT 2D/3D box extraction, ego speed and pose, HD Map polyline parsing and BEV projection.
- **`src/models/lanes/kinematic_ego.py`** — Constant Turn Rate arc path predictor from ego yaw rate and speed (double EMA smoothing, speed-dependent yaw damping).
- **`src/models/lanes/visual_dp.py`** — YOLOPv2 ONNX drivable-area segmentation path.
- **`src/models/lanes/visual_host.py`** — CLRNet ONNX host lane marking detector.
- **`src/models/lanes/manager.py`** — `LaneManager`: builds all lane inference engines once per run, dispatches per-frame calls, resets state between segments.
- **`src/models/lane_detector.py`** — `KinematicPathPredictor` wrapper with EMA state.
- **`src/visualization/visualizer.py`** — Frame annotator: GT boxes, lane paths, HUD overlay.
- **`src/utils/comet_logger.py`** — Comet ML integration: raw + annotated frames, ego speed metric, segment duration.
- **`scripts/download_waymo_segments.py`** — Lists, samples, and downloads Waymo TFRecords from GCS. Writes `segments_to_run.txt`.
- **`conf/`** — Full Hydra config tree: `dataset/waymo.yaml`, `model/lane.yaml`, `model/kinematic.yaml`, `logger/comet.yaml`.
- **`docs/01_Architecture_Overview.md`** — Component map and responsibility breakdown.
- **`docs/02_Lane_Calculations.md`** — Four lane-path strategies with algorithm details and config knobs.
- **`docs/03_Quickstart_and_Debug.md`** — Setup, data download, pipeline run, debug viewer.
- **`debug_viewer.py`** — Local frame-by-frame debug visualiser (no Comet ML required).

### Added — Repository Foundations (P0)

- **`README.md`** — Project storefront: pipeline diagram, phase status table, quickstart.
- **`docs/00_Roadmap.md`** — Full project vision, design decisions, per-phase deliverables.
- **`LICENSE`** — Apache 2.0.
- **`.gitignore`** — Excludes TFRecords, ONNX weights, generated JSONs, outputs, `.venv`.
- **`CHANGELOG.md`** — This file.
