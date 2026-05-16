# adas-lane-functions

> **End-to-end ADAS lane-function pipeline** — from raw camera frames (Waymo Open Dataset) to production-ready classification signals: **CIPV**, **Lane Assignment**, and beyond.

---

## What This Project Is

An end-to-end ADAS signal pipeline — currently under **active development**.

The perception input pipeline (Phases 1–2) is fully operational: it ingests Waymo TFRecords, runs lane-path detection and target tracking, and produces structured per-frame JSON outputs. The downstream stages (feature assembly, model training, deployment) have their architecture and base code in place and are the next focus of development.

The project is designed to be **transparent, reproducible, and architecture-first** — every design decision is documented, every config is externalised, and every pipeline stage is independently testable.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Phase 1 — Perception  (🔄 active)                                       │
│                                                                          │
│  Waymo TFRecord  →  WaymoParser  →  LaneManager (4 strategies)          │
│                                  →  GT 2D Boxes + HD Map + Ego Pose     │
│                                  →  Segment JSON  +  Comet ML log       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Phase 2 — Detection & Tracking  (🔄 active)                             │
│                                                                          │
│  YOLOv8n ONNX (pre-trained)  →  2D detections (vehicle/ped/cyclist)     │
│  KalmanTracker               →  Tracklet state: range, lateral, heading │
│  RWCoordinates               →  Vehicle-frame real-world state per ID   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Phase 3 — Feature Engineering  (🏗️  base written — next up)             │
│                                                                          │
│  MeasurementAssembler  →  Per-target MF vector (T=10 frames):           │
│    • Kalman state  (range, range_rate, lateral, TTC)                    │
│    • Normalized pixel dist to each lane line                            │
│    • Drivable-area overlap ratio                                        │
│  GTBuilder             →  CIPV label + Lane Assignment label from       │
│                           Waymo GT 3D boxes + HD Map (no annotation)   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Phase 4 — Model Training  (🏗️  base written — next up)                  │
│                                                                          │
│  TransformerEncoder [T×D]  →  CIPVHead (binary)                         │
│                             →  LaneAssignHead (5-class: −2…+2)          │
│  Optuna HPSearch  →  Performance analysis report  →  Final checkpoint   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────────────┐
│  Phase 5 — Deployment  (🏗️  base written — next up)                      │
│                                                                          │
│  ONNX export  →  INT8 quantization  →  C++ frame-loop runtime           │
│  Input: per-frame JSON measurements  →  Output: signal per target / ms  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Status

| Phase | Description | Status |
|---|---|---|
| **P1** | Batch perception pipeline (lane paths, GT parsing, Comet logging) | 🔄 Active development |
| **P2** | Target detection (ONNX), Kalman tracking, real-world coordinates | 🔄 Active development |
| **P3** | MF vector assembly, GT builder (CIPV + Lane Assignment labels) | 🏗️ Base written — next up |
| **P4** | Transformer model, training loop, HPsearch, eval report | 🏗️ Base written — next up |
| **P5** | ONNX export, INT8 quantization, C++ inference runtime | 🏗️ Base written — next up |

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/00_Roadmap.md](docs/00_Roadmap.md) | Full project vision, design decisions, phase breakdown |
| [docs/01_Architecture_Overview.md](docs/01_Architecture_Overview.md) | Component map and responsibilities |
| [docs/02_Lane_Calculations.md](docs/02_Lane_Calculations.md) | Four lane-path strategies explained |
| [docs/03_Quickstart_and_Debug.md](docs/03_Quickstart_and_Debug.md) | Setup, data download, pipeline run |

---

## Quickstart

```bash
# 1. Create environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Download Waymo segments (requires Waymo licence acceptance)
python scripts/download_waymo_segments.py --num-segments 5 --output-dir ./src/data

# 3. Run the batch perception pipeline
python pipeline_input.py

# 4. Override config at the CLI
python pipeline_input.py dataset.max_frames=20 comet.experiment_name="test_run"
```

See [docs/03_Quickstart_and_Debug.md](docs/03_Quickstart_and_Debug.md) for the full guide.

---

## Project Structure

```
adas-lane-functions/
├── conf/                    # Hydra config (single source of truth)
│   ├── config.yaml
│   ├── dataset/waymo.yaml
│   ├── model/lane.yaml
│   ├── model/kinematic.yaml
│   └── logger/comet.yaml
├── src/
│   ├── data/                # Waymo parser + data loader
│   ├── models/
│   │   ├── lanes/           # Lane-path strategies (Kinematic, Drivable, Host)
│   │   ├── detection/       # [P2] Target detector wrapper (YOLOv8n ONNX)
│   │   ├── tracking/        # [P2] Kalman filter state estimator
│   │   └── classification/  # [P4] Transformer + CIPV/Lane Assignment heads
│   ├── features/            # [P3] MF vector assembler + GT builder
│   ├── training/            # [P4] Dataset, training loop, HPsearch
│   ├── evaluation/          # [P4] Metrics + performance report
│   ├── inference/           # [P5] ONNX export + quantization
│   └── visualization/       # Frame annotator + Comet logger
├── cpp/                     # [P5] C++ frame-loop inference runtime
├── scripts/                 # Utility scripts (data download, etc.)
├── tests/                   # Unit tests per module
├── notebooks/               # EDA, training analysis
├── docs/                    # Architecture + design documentation
├── pipeline_input.py        # Hydra entry point — batch orchestrator
└── segments_to_run.txt      # Active segment list (path per line)
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
