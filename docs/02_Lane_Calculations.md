# Lane Calculations

The pipeline produces four lane-path outputs per frame. Three are computed by `LaneManager`; one comes directly from the Waymo HD Map proto.

---

## Path 1 — Kinematic Ego Path

**Source:** `src/models/lanes/kinematic_ego.py` + `src/models/lane_detector.KinematicPathPredictor`

**Why:** Camera-based lane detectors fail on occluded, faded, or unmarked roads. A physics model always produces a geometrically plausible prediction from ego-motion data alone.

**How — Constant Turn Rate (CTR) arc:**

1. **Yaw rate derivation** — consecutive Waymo `frame.pose.transform` matrices (4×4, Vehicle → Global) are differenced to compute a raw finite-difference yaw rate in rad/s.
2. **Two-stage EMA smoothing** — GPS/IMU pose matrices contain single-frame noise spikes. Two cascaded Exponential Moving Averages suppress them before they reach the arc model:
   - Pipeline-level EMA: α = 0.25 (~4-frame effective window at 10 Hz) — removes large outliers.
   - Predictor-level EMA: α = 0.15 (~7-frame window) — fine temporal smoothing.
3. **Speed-dependent yaw damping** — below 5 m/s, yaw rate is linearly damped to zero. Prevents the low-speed singularity where small GPS noise produces wildly spinning paths.
4. **Hard clamp** — |yaw_rate| is capped at 0.35 rad/s (≈ 20°/s). Eliminates extreme outliers that both EMA stages miss.
5. **CTR arc geometry** — with (speed, yaw_rate) the predictor samples `n_points` along a circular arc over `horizon_s` seconds. Returns three 3D polylines in **Vehicle Frame** (X forward, Y left, Z=0): `centre_line`, `left_boundary`, `right_boundary`.

**Config knobs** (`conf/model/kinematic.yaml`):
| Key | Default | Effect |
|---|---|---|
| `vehicle_width` | 2.1 m | lateral offset from centre to each boundary |
| `horizon_s` | 3.0 s | how far ahead the arc is projected |
| `n_points` | 30 | spatial resolution of the arc |
| `min_speed_mps` | 8.0 m/s | floor speed to keep the path visible when stopped |

---

## Path 2 — HD Map

**Source:** `src/data/waymo_parser.parse_map_features_global` / `project_hdmap_lanes`

**Why:** Ground-truth reference path from Waymo's pre-built HD Map. Used for evaluation and as a gold-standard visual overlay.

**How:**
- `map_features` are cached from the **first frame** that contains them and reused for the entire segment (the map does not change per frame).
- Global-Frame polylines are transformed to Vehicle Frame using `frame.pose.transform`, then projected to BEV / image space.
- The left and right HD Map lanes are averaged to compute a centre line for comparison with kinematic and drivable predictions.

> HD Map is **not** managed by `LaneManager` — it is Waymo-proto-specific and lives entirely in `waymo_parser.py`.

---

## Path 3 — Visual Drivable Path

**Source:** `src/models/lanes/visual_dp.py` + `YOLOPv2DrivableDetector`

**Why:** Painted lane markings identify *lanes*, not *free space*. At intersections, construction zones, or on unmarked roads the host lane detector fails — but the drivable surface is still clearly visible as a free-space region. YOLOPv2 segments that surface without requiring lane markings.

**How:**
1. **YOLOPv2 ONNX inference** — a single forward pass on the resized front-camera image produces a `da_seg_out` binary segmentation mask (drivable area).
2. **Forward frustum ROI** — a trapezoid mask clips the drivable area to the ego forward corridor, removing bleed from intersecting roads and oncoming lanes seen at angles.
3. **EMA on polynomial coefficients** — a 2nd-degree polynomial is fitted to the drivable centroid column and to each boundary. Coefficients are blended with the previous frame's values (α = 0.25) for temporal smoothness.
4. **Outputs**: `center_path`, `left_path`, `right_path` — pixel-space polylines ready for the visualizer.

**Config knobs** (`conf/model/lane.yaml`, `yolopv2` block):
| Key | Default | Effect |
|---|---|---|
| `min_drivable_pix` | 30 px | min pixels per row to include a row centroid |
| `ll_conf_threshold` | 0.30 | sigmoid threshold for the lane-line head |

---

## Path 4 — Visual Host Lane

**Source:** `src/models/lanes/visual_host.py` + YOLOPv2 `ll_seg_out` head (or CLRNet / IPM fallback)

**Why:** Painted lane markings give the legal lane boundary — the space the vehicle is *permitted* to occupy. This is complementary to the drivable-area path: a host lane can exist in a construction zone (painted), while free-space detection covers unmarked roads. Together they cross-validate each other.

**How — YOLOPv2 LL head (primary):**
1. The `ll_seg_out` head of the same YOLOPv2 forward pass (shared with Path 3) produces a lane-line probability map.
2. A perspective-corridor mask restricts detection to the ego lane width in the image bottom half, blocking adjacent lanes.
3. Rows where the sigmoid output exceeds `ll_conf_threshold` are collected; a 2nd-degree polynomial is fitted separately to left and right lane columns.
4. The lane is marked **valid** only when both left and right polynomials have enough row support (`coverage ≥ host_lane_confidence_threshold`).

**How — CLRNet fallback (when YOLOPv2 is not configured):**
- CLRNet detects lane lines directly as parametric curves.
- The two highest-confidence curves bracketing the image centre are selected as host left/right.
- Due to domain shift (CULane-trained model on Waymo imagery), confidence scores are typically 0.001–0.015; set `confidence_threshold` accordingly.

**How — IPM fallback:**
- A BEV sliding-window search on a thresholded binary image. No trained weights.
- More robust to domain shift; less accurate on curved or occluded lanes.

**Config knobs** (`conf/model/lane.yaml`):
| Key | Default | Effect |
|---|---|---|
| `host_lane_confidence_threshold` | 0.01 | min coverage fraction to publish host lane as valid |
| `visual_backend` | `clrnet` | active fallback backend when YOLOPv2 is not used |

---

## Inference Priority and Single-Pass Guarantee

```
YOLOPv2 configured?
    YES → one ONNX forward pass → da_seg_out (Path 3) + ll_seg_out (Path 4)
    NO  → CLRNet or IPM → separate drivable + host-lane estimates
```

`LaneManager.process()` guarantees **at most one ONNX forward pass per frame** regardless of how many strategies consume its outputs. The packaging strategies (`DrivablePathStrategy`, `HostLaneStrategy`) are stateless and only normalise the dict format; they never trigger inference.
