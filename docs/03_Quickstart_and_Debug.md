# Quickstart and Debug Guide

---

## Prerequisites

```bash
# 1. Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Authenticate with Google Cloud (required for segment download only)
gcloud auth application-default login
# Accept the Waymo licence at https://waymo.com/open/terms/ first.
```

---

## Step 1 — Download Segments

`scripts/download_waymo_segments.py` lists available TFRecords from the public Waymo GCS bucket, draws a random sample, downloads them, and writes `segments_to_run.txt`.

### SDK mode (default)
```bash
python scripts/download_waymo_segments.py \
    --num-segments 5 \
    --output-dir ./src/data
```

### gsutil mode (resumable downloads, faster for large files)
```bash
python scripts/download_waymo_segments.py \
    --num-segments 5 \
    --use-gsutil \
    --output-dir ./src/data
```

### Useful flags
| Flag | Default | Purpose |
|---|---|---|
| `-n / --num-segments` | 5 | Number of segments to download |
| `--output-dir` | `./src/data` | Local destination directory |
| `--bucket` | `waymo_open_dataset_v_1_4_2` | GCS bucket name |
| `--seed` | *(random)* | Integer seed for reproducible sampling |
| `--dry-run` | off | Preview selection without downloading; still writes `.txt` |

After the script exits, `segments_to_run.txt` (project root) is overwritten with the absolute paths of all downloaded files.

To add segments manually, append lines to `segments_to_run.txt`:
```
# Lines starting with # are ignored
/absolute/path/to/segment-XXXXX_with_camera_labels.tfrecord
```

---

## Step 2 — Run the Batch Pipeline

The pipeline entry point is `pipeline_input.py`, driven by Hydra.

### Default run (reads `segments_to_run.txt`, up to `max_segments=1`)
```bash
python pipeline_input.py
```

### Process all segments in the list
```bash
python pipeline_input.py dataset.max_segments=null
```

### Limit frames per segment (fast smoke test)
```bash
python pipeline_input.py dataset.max_frames=10 dataset.max_segments=2
```

### Override any config value on the CLI
```bash
# Use a different segment list
python pipeline_input.py dataset.segment_list=/path/to/other.txt

# Change Comet experiment name for this run
python pipeline_input.py comet.experiment_name="ablation_v3"

# Save JSON output to a specific directory
python pipeline_input.py output.output_dir=/results/run_01

# Swap the visual backend to IPM (no ONNX weights required)
python pipeline_input.py perception.lane.visual_backend=ipm
```

### Skip already-processed segments (resume a batch)
```bash
python pipeline_input.py dataset.skip_existing=true
```

---

## Step 3 — Understand the Outputs

### JSON files
One `.json` per segment, written to `output.output_dir` (defaults to the same directory as the TFRecord). Each JSON is a list of frame dicts. Each frame contains:

```json
{
  "timestamp": 1234567890.123,
  "ego_speed_kmh": 45.2,
  "boxes_2d": [ ... ],
  "kinematic":     { "center": [[x,y],...], "left": [...], "right": [...], "valid_center": true, ... },
  "hdmap":         { "center": [...], "left": [...], "right": [...], "source": "hdmap", "is_gt": true },
  "drivable_path": { "center": [...], "left": [...], "right": [...], "source": "yolopv2_da", ... },
  "host_lane":     { "left": [...], "right": [...], "valid_left": true, "valid_right": false, ... }
}
```

Load with:
```python
import json
with open("segment-XXXX.json") as f:
    frames = json.load(f)

# Example: print all frames where host lane is valid
for i, frame in enumerate(frames):
    if frame["host_lane"]["valid_left"] and frame["host_lane"]["valid_right"]:
        print(f"Frame {i}: host lane valid, speed={frame['ego_speed_kmh']:.1f} km/h")
```

### Comet ML dashboard
Every run logs to the project defined in `conf/logger/comet.yaml`:
- **Raw front camera** image per frame (with GT bounding box annotations).
- **Annotated front camera** image per frame — all enabled lane overlays rendered.
- **`ego_speed_kmh`** metric per frame.
- **`segment_duration_s`** and **`segment_frames`** per segment.
- The full JSON file as an asset (downloadable from the Artifacts tab).

Enable or disable annotation layers without code changes:
```yaml
# conf/config.yaml
visualization:
  enabled_paths:
    - kinematic
    - drivable_path
    - host_lane
    # - hdmap      ← uncomment to enable HD Map overlay
```

---

## Step 4 — Debugging Common Issues

### Path 1 (Kinematic) is spinning or erratic at low speed
- **Cause:** noisy GPS yaw rate at near-zero speed.
- **Fix:** `perception.kinematic_path.min_speed_mps` is the floor. Raise it (e.g. to 10.0 m/s) to prevent the path from using unreliable yaw-rate estimates.

### Path 3 (Drivable Area) bleeds into oncoming lanes at intersections
- **Cause:** YOLOPv2 correctly segments all drivable surface; forward-frustum ROI is too wide.
- **Fix:** Tighten the trapezoid mask in `YOLOPv2DrivableDetector._apply_forward_frustum` (horizon line width `0.30–0.70` → narrower, e.g. `0.38–0.62`).

### Path 4 (Host Lane) is always invalid
- **Cause A:** `host_lane_confidence_threshold` too high for the active backend.
  - YOLOPv2: try 0.10–0.15. CLRNet on Waymo: try 0.005.
- **Cause B:** `ll_conf_threshold` too high — lane pixels are suppressed before polynomial fitting.
  - Try lowering from 0.30 to 0.20.
- **Debug:** Set `dataset.max_frames=5`, check Comet annotated images for any purple host-lane overlay.

### CLRNet produces no detections
- **Cause:** Domain shift — CULane-trained model scores are low on Waymo imagery (~0.001–0.015).
- **Fix:** Lower `confidence_threshold` in `conf/model/lane.yaml` to 0.001, or switch to YOLOPv2.

### Kinematic path wraps around / draws a backward arc
- **Cause:** Projection artefact from points near Camera Frame Z ≈ 0 (within ~1.5 m of camera mount).
- **Status:** Fixed by the Z > 1.0 m clip in `_project_path_segments`. If it reappears, verify `CameraCalibration.extrinsic` is loaded from the Waymo proto (not the hardcoded default) for the segment in question.

### `All resolved segments are missing from disk`
- Re-run `download_waymo_segments.py` or check that `segments_to_run.txt` contains valid absolute paths.
- Use `--dry-run` to inspect what the downloader would select before committing.

### Hydra `ConfigCompositionException`
- Run `python -c "import hydra; ..."` (see README or the verification snippet) to print the composed config and identify which group file is malformed.
- Common cause: a tab character instead of spaces in a YAML file.
