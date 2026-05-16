# Project Roadmap

**Project:** adas-lane-functions
**Goal:** A fully reproducible, production-ready pipeline that takes raw driving video (Waymo Open Dataset TFRecords) and outputs per-target ADAS classification signals — starting with **CIPV** (Closest In-Path Vehicle) and **Lane Assignment** — using a Transformer trained on algorithmically-derived ground truth.

---

## The Core Idea

ADAS lane functions (CIPV, Lane Assignment, Cut-In) are fundamentally **target-to-lane relationship problems**. A vehicle ECU needs to know, for each detected object in the scene:

1. Which lane is the object in, relative to ego? (Lane Assignment)
2. Is the object the most critical threat in the ego lane? (CIPV)
3. Is the object moving into the ego lane? (Cut-In)

Solving these correctly and robustly is the difference between a highway assist system that works and one that causes accidents.

This project demonstrates the full stack required to solve them: from raw sensor data, through structured feature engineering, to a deployable classification model.

---

## Design Decisions

### 1. Why Waymo Open Dataset?

- Highest quality publicly available driving data (10 Hz, calibrated, multi-camera, HD Map, 3D GT boxes with persistent object IDs across frames).
- The HD Map + 3D GT boxes + ego pose allow **algorithmic GT derivation** for CIPV and Lane Assignment — no manual annotation needed.
- Real-world diversity: highways, urban intersections, rain, night.

### 2. Why pre-trained ONNX detector (not custom-trained)?

The innovation in this project is the **signal classification model**, not the detector. Using a stable, well-validated pre-trained detector (YOLOv8n) as a dependency:
- Eliminates detector training as a time sink.
- Produces consistent, well-understood detections.
- Allows the pipeline to focus on what matters: measurement assembly and temporal classification.

Highly occluded objects (occlusion ratio > threshold) are filtered at the detection stage before entering the Kalman tracker. This keeps the MF vector clean and prevents the model from learning on noisy inputs.

### 3. Why a Transformer (not LSTM or rule-based)?

- Attention over the T-frame window lets the model learn *which frames matter* — e.g., the frame where lateral velocity spikes is more informative for Cut-In prediction than steady-state frames.
- Transformer attention maps are inspectable: we can visualise which frames drove a CIPV prediction — critical for debugging and for demonstrating model interpretability to a hiring audience.
- The multi-head design (shared encoder, separate classification heads) means adding a new signal (e.g., Cut-In, pedestrian intent) requires only a new head, not a new model.

### 4. Why algorithmic GT (not crowdsourced labels)?

Waymo provides:
- Per-frame GT 3D bounding boxes with persistent object IDs (tracklets).
- HD Map with lane topology (which lane connects to which, lane boundaries in global frame).
- Ego pose at every frame (vehicle → global transform).

From these three, we can deterministically compute:
- **CIPV:** Project GT 3D boxes to vehicle frame → find objects in ego lane corridor → closest with positive range rate = CIPV.
- **Lane Assignment:** For each tracked object, compute lateral offset from each HD Map lane centre → assign integer lane index relative to ego.
- **Cut-In:** Lane Assignment sequence changes from ±1 → 0 within N consecutive frames.

This means the GT builder is an **algorithm, not a labeling task**. It is fast, reproducible, and consistent.

---

## Phase Breakdown

---

### Phase 1 — Perception Pipeline ✅ Complete

**Deliverables:**
- Hydra-configured batch processor (`pipeline_input.py`) that iterates `segments_to_run.txt`.
- `WaymoParser`: extracts front-camera images, GT 2D/3D boxes, ego speed and pose, HD Map polylines.
- `LaneManager` with four lane-path strategies per frame:
  - Kinematic ego path (CTR arc from ego pose).
  - HD Map path (Waymo ground-truth lanes).
  - Visual drivable path (YOLOPv2 ONNX segmentation).
  - Host lane detection (CLRNet ONNX lane marking detector).
- Per-segment JSON output serialisation.
- Comet ML logging (raw + annotated frames, metrics).

**Key files:**
- `pipeline_input.py` — orchestrator
- `src/data/waymo_parser.py` — data layer
- `src/models/lanes/` — lane strategy implementations
- `src/visualization/visualizer.py` — frame annotator

---

### Phase 2 — Detection & Tracking 🚧 In Progress

**Goal:** For each frame, produce a stable set of tracked objects with real-world state estimates (position, velocity, heading in vehicle frame).

**Deliverables:**
- `src/models/detection/detector.py` — `TargetDetector` class wrapping YOLOv8n ONNX. Filters detections by:
  - Confidence threshold.
  - Occlusion ratio (bounding box overlap with other boxes > threshold → discard).
  - Class whitelist: vehicle, pedestrian, cyclist.
- `src/models/tracking/kalman_tracker.py` — `KalmanTracker` class. Constant Velocity model in vehicle frame. One filter instance per track ID. Outputs per-track state: `[range, lateral_offset, range_rate, lateral_rate, heading_delta]`.
- `src/models/tracking/track_manager.py` — `TrackManager`. Handles birth, update, and death of tracks. Hungarian algorithm assignment.
- `src/features/rw_coordinates.py` — `RWCoordinateConverter`. Projects 2D detections to vehicle-frame real-world coordinates using camera intrinsics + ego height assumption. Feeds into Kalman state initialisation.
- Config: `conf/model/detector.yaml`, `conf/model/tracker.yaml`.

**Occlusion filtering strategy:**
- Compute pairwise IoU between all detections in a frame.
- For any detection where `sum(IoU with others) > cfg.detector.occlusion_iou_threshold`: mark as occluded and exclude from tracker input.
- This keeps MF vectors free from high-uncertainty observations.

---

### Phase 3 — Feature Engineering & GT Builder 📋 Planned

**Goal:** Assemble the per-target Measurement Feature (MF) vector over a rolling T-frame window, and generate CIPV + Lane Assignment ground-truth labels for every tracked object in every segment.

**MF vector design (per target, per frame, D ≈ 18 features):**

```
Kalman state (5):
  range, range_rate, lateral_offset, lateral_rate, heading_delta

Normalised pixel measurements (4):
  dist_to_host_lane_centre_norm   # pixels / lane_width_px
  dist_to_left_lane_line_norm
  dist_to_right_lane_line_norm
  drivable_area_overlap_ratio

Derived kinematic (2):
  ttc                             # range / range_rate, clipped at cfg.features.ttc_clip_s
  relative_speed_norm             # relative speed / ego_speed, clipped

Bounding box (3):
  bbox_width_norm, bbox_height_norm, bbox_aspect_ratio

Object class one-hot (4):
  [is_vehicle, is_pedestrian, is_cyclist, is_other]
```

Stacked over T=10 frames → shape `[T, D]` per target → Transformer input.

**GT builder design:**
- `src/features/gt_builder.py` — `GTBuilder` class.
- Input: segment TFRecord + Waymo GT 3D boxes + HD Map + tracker output.
- Output: per-frame, per-track label dict:
  - `cipv: bool` — True for at most one object per frame.
  - `lane_assignment: int` — integer in {-2, -1, 0, +1, +2}.
  - `cut_in: bool` — True if lane_assignment transitioned from ±1 → 0 in the past N frames.

**Deliverables:**
- `src/features/mf_assembler.py` — `MFAssembler`
- `src/features/gt_builder.py` — `GTBuilder`
- `src/features/dataset_writer.py` — writes assembled (MF, label) pairs to HDF5 for training
- Config: `conf/features/mf.yaml`

---

### Phase 4 — Model Training 📋 Planned

**Goal:** Train a Transformer-based classifier on the assembled dataset and produce a performance analysis report.

**Model architecture:**
```
Input: [B, T=10, D=18]
  │
TransformerEncoder
  • 4 layers, 4 heads, d_model=64, d_ff=256
  • Positional encoding over T dimension
  • [CLS] token prepended
  │
[CLS] output: [B, 64]
  ├── CIPVHead:        Linear(64, 1)  → sigmoid  → binary
  └── LaneAssignHead:  Linear(64, 5)  → softmax  → class {-2,-1,0,+1,+2}
```

**Training setup:**
- Dataset split: 80/10/10 (train/val/test), split by segment (no frame leakage across splits).
- Loss: BCE for CIPV + CrossEntropy for Lane Assignment, weighted by class frequency.
- Optimiser: AdamW, cosine LR schedule.
- HPsearch: Optuna (n_trials=50) over `{d_model, n_heads, n_layers, dropout, lr, batch_size}`.
- Output: best checkpoint + full performance report (confusion matrix, F1, precision/recall per class, attention visualisation).

**Deliverables:**
- `src/models/classification/transformer.py`
- `src/models/classification/heads.py`
- `src/training/trainer.py`
- `src/training/dataset.py`
- `src/evaluation/metrics.py`
- `src/evaluation/report.py`
- `notebooks/04_training_analysis.ipynb`
- `conf/training/default.yaml`, `conf/training/hpsearch.yaml`

---

### Phase 5 — Deployment 📋 Planned

**Goal:** Export the trained model to a quantised ONNX graph and build a C++ frame-loop runtime that produces classification signals within a single-frame time budget.

**Export pipeline:**
- `src/inference/export_onnx.py` — export PyTorch model to ONNX (opset 17).
- `src/inference/quantize.py` — PTQ INT8 calibration using a held-out calibration split.
- Accuracy validation: compare FP32 vs INT8 outputs on the test split; fail if F1 drops > 2%.

**C++ runtime (`cpp/`):**
- Input: per-frame JSON (MF vectors for all active tracks, assembled by Python pipeline or a C++ equivalent).
- ONNX Runtime C++ API for inference.
- Output: `std::vector<SignalResult>` — one entry per track ID with `cipv_score`, `lane_assignment`, `cut_in_score`.
- Frame budget target: < 5 ms per frame on a mid-range automotive SoC equivalent.

**Deliverables:**
- `src/inference/export_onnx.py`
- `src/inference/quantize.py`
- `cpp/src/signal_engine.cpp` + `cpp/include/signal_engine.h`
- `cpp/CMakeLists.txt`
- `docs/08_CPP_Deployment.md`

---

## Progress Tracker

| Phase | Owner | Started | Completed |
|---|---|---|---|
| P1 — Perception Pipeline | — | 2026-04 | 2026-05 |
| P2 — Detection & Tracking | — | 2026-05 | — |
| P3 — Feature Engineering & GT | — | — | — |
| P4 — Model Training | — | — | — |
| P5 — Deployment | — | — | — |
