#!/usr/bin/env python3
"""
Waymo ADAS Debug Viewer — EKF Measurement & Track Analysis GUI

Usage
-----
    python debug_viewer.py
    python debug_viewer.py --tfrecord <path.tfrecord>
    python debug_viewer.py --tfrecord <path.tfrecord> --json <path.json>
    python debug_viewer.py --tfrecord <path> --max-frames 150

Panels
------
  Image (left)   Annotated frame: GT boxes · YOLO+Kalman · EKF vehicle track
                 overlays with colour-coded IDs and SF y0-projection crosshairs.

  Info (tab)     Frame metadata · box display mode · lane path toggles
                 · path data quality table.

  Tracks (tab)   EKF vehicle track table (14-field state) + per-track SF
                 measurement breakdown + Lane Relations for selected track.

  Detections (tab)  Raw YOLO detections · Kalman TrackManager tracks (all
                    classes) · GT 3D bounding boxes from laser labels.

  Plot (tab)     Embedded time-series chart — any EKF/SF/lane-relation field
                 for any track_id over all frames, with a live frame-cursor.

  JSON (tab)     Raw frame JSON (populated lazily — zero cost during playback).

Keyboard shortcuts
------------------
  ← / →     step back / forward one frame
  Space      play / pause
  Home / End jump to first / last frame
  + / -      increase / decrease playback speed
  T          jump to Tracks tab
  D          jump to Detections tab
  P          jump to Plot tab
  J          jump to JSON tab
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
from src.visualization.visualizer import CameraCalibration, PerceptionVisualizer

from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QProgressDialog, QPushButton, QSizePolicy, QSlider, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigCanvas
    from matplotlib.figure import Figure as _MplFigure
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# =============================================================================
# Constants & colour palettes
# =============================================================================

# GT box colours (BGR)
_GT_COLORS_BGR: dict[int, tuple] = {
    1: (255, 100,  50),
    2: ( 50, 220,  80),
    3: (200, 200, 200),
    4: (255, 220,  50),
}
_TYPE_NAMES: dict[int, str] = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist"}
_ROW_COLORS: dict[str, str] = {
    "Vehicle": "#6699ff", "Pedestrian": "#44cc88",
    "Cyclist":  "#ffcc44", "Sign":       "#bbbbcc",
}

# 20 visually distinct BGR colours cycling through EKF track IDs
_TRACK_PALETTE_BGR: list[tuple] = [
    (255,  80,  80), ( 80, 255,  80), ( 80, 120, 255), (255, 200,  50),
    (200,  50, 255), ( 50, 220, 220), (255, 130,  50), (160, 255, 100),
    (255,  50, 180), ( 80, 200, 255), (220, 220,  50), (255, 100, 150),
    (100, 180, 255), ( 50, 255, 160), (255, 170,  80), (180,  80, 255),
    (200, 255,  50), ( 80, 255, 220), (255,  80, 130), (140, 200, 255),
]

_PATH_DRAW: dict[str, dict] = {
    "kinematic":     {"center": (  0, 220, 255), "left": (  0, 190, 255), "right": (  0, 190, 255), "thick": 2},
    "hdmap":         {"center": ( 20, 180,  20), "left": ( 20, 230,  20), "right": ( 10, 160,  10), "thick": 3},
    "drivable_path": {"center": (200, 200,   0), "left": (140, 200,   0), "right": (140, 200,   0), "thick": 2},
    "host_lane":     {"center": (210,  30, 210), "left": (210,  30, 210), "right": (180,   0, 255), "thick": 3},
}
_PATH_LABELS = {
    "kinematic": "Kinematic", "hdmap": "HD Map",
    "drivable_path": "Drivable", "host_lane": "Host Lane",
}
_PATH_ORDER = ["kinematic", "hdmap", "drivable_path", "host_lane"]

# All plottable EKF + SF fields
_PLOT_FIELDS: dict[str, str] = {
    "x_veh":       "EKF range x (m)",
    "y_veh":       "EKF lateral y (m)",
    "z_veh":       "EKF height z (m)",
    "vx_veh":      "EKF range-rate vx (m/s)",
    "vy_veh":      "EKF lateral-rate vy (m/s)",
    "speed_mps":   "EKF speed |v| (m/s)",
    "heading_rad": "EKF heading (rad)",
    "width_m":     "EKF width (m)",
    "height_m":    "EKF height (m)",
    "length_m":    "EKF length (m)",
    "ttc_s":       "TTC (s)",
    "sf_y0_x":     "SF y0-proj x (m)",
    "sf_y0_y":     "SF y0-proj y (m)",
    "sf_height_x": "SF height-prior x (m)",
    "sf_width_x":  "SF width-prior x (m)",
    "sf_h_aspect": "SF h_aspect (px/fy)",
    "sf_w_aspect": "SF w_aspect (px/fx)",
    "km_x_gnd":          "Kalman x_gnd (m)",
    "km_y_gnd":          "Kalman y_gnd (m)",
    # Lane relation — lateral offset per path (m, + = left of path)
    "lr_kinematic_lateral":  "LR kinematic lateral (m)",
    "lr_drivable_lateral":   "LR drivable lateral (m)",
    "lr_host_lateral":       "LR host-lane lateral (m)",
    "lr_hdmap_lateral":      "LR hdmap lateral (m)",
    # Lane relation — inside-bounds flag (1 = inside, 0 = outside)
    "lr_kinematic_inside":   "LR kinematic inside (0/1)",
    "lr_drivable_inside":    "LR drivable inside (0/1)",
    "lr_host_inside":        "LR host-lane inside (0/1)",
    "lr_hdmap_inside":       "LR hdmap inside (0/1)",
}


# =============================================================================
# Data loading & track history index
# =============================================================================

def load_tfrecord(path: str, max_frames: int,
                  progress_cb=None) -> tuple[list, list]:
    """Extract BGR images and minimal GT metadata from a TFRecord segment."""
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset

    dataset = tf.data.TFRecordDataset(path, compression_type="")
    images, gt_list = [], []
    prev_pos, prev_time = None, None

    for i, raw in enumerate(dataset):
        if i >= max_frames:
            break
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(raw.numpy()))

        img = None
        for cam in frame.images:
            if cam.name == open_dataset.CameraName.FRONT:
                arr = tf.image.decode_jpeg(cam.image).numpy()
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                break
        images.append(img)

        mat = frame.pose.transform
        cx2, cy2 = mat[3], mat[7]
        t2 = frame.timestamp_micros / 1e6
        if prev_pos is not None and (t2 - prev_time) > 0:
            speed = np.hypot(cx2 - prev_pos[0], cy2 - prev_pos[1]) / (t2 - prev_time) * 3.6
        else:
            speed = 0.0
        prev_pos, prev_time = (cx2, cy2), t2

        boxes_2d = []
        for cam_labels in frame.camera_labels:
            if cam_labels.name == open_dataset.CameraName.FRONT:
                for lbl in cam_labels.labels:
                    b = lbl.box
                    boxes_2d.append({
                        "id": lbl.id, "type": lbl.type,
                        "center_x": b.center_x, "center_y": b.center_y,
                        "length": b.length, "width": b.width,
                    })

        gt_list.append({"timestamp": t2, "ego_speed_kmh": speed, "boxes_2d": boxes_2d})
        if progress_cb:
            progress_cb(i + 1)

    return images, gt_list


def build_track_history(gt_data: list) -> dict[int, dict[str, list]]:
    """
    Pre-index all vehicle EKF track data for O(1) time-series access.

    Returns dict[track_id, dict[field_key, list[tuple[frame_idx, value]]]].
    Frames where a track is absent produce no entry, so the plot shows
    natural gaps rather than zeros.
    """
    history: dict[int, dict[str, list]] = {}

    for frame_idx, gt in enumerate(gt_data):
        # Build lane-relation lookup for this frame: {track_id -> relations dict}
        _lr_idx = {
            e["track_id"]: e.get("relations", {})
            for e in gt.get("lane_relations", [])
            if "track_id" in e
        }

        for t in gt.get("vehicle_ekf_tracks", []):
            tid = t.get("track_id")
            if tid is None:
                continue
            if tid not in history:
                history[tid] = {f: [] for f in _PLOT_FIELDS}

            rec = history[tid]

            def _push(field: str, val: Any) -> None:
                if val is None:
                    return
                if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                    return
                try:
                    rec[field].append((frame_idx, float(val)))
                except (TypeError, ValueError):
                    pass

            for f in ("x_veh", "y_veh", "z_veh", "vx_veh", "vy_veh",
                      "speed_mps", "heading_rad", "width_m", "height_m",
                      "length_m", "ttc_s"):
                _push(f, t.get(f))

            sf = t.get("sf_measurements") or {}
            _push("sf_y0_x",     (sf.get("y0_proj")     or {}).get("x_m"))
            _push("sf_y0_y",     (sf.get("y0_proj")     or {}).get("y_m"))
            _push("sf_height_x", (sf.get("height_proj") or {}).get("x_m"))
            _push("sf_width_x",  (sf.get("width_proj")  or {}).get("x_m"))
            _push("sf_h_aspect", sf.get("h_aspect"))
            _push("sf_w_aspect", sf.get("w_aspect"))

            km = t.get("kalman_input") or {}
            _push("km_x_gnd", km.get("x_gnd"))
            _push("km_y_gnd", km.get("y_gnd"))

            # Lane relations — lateral offset and inside-bounds per path
            lr_rels = _lr_idx.get(tid, {})
            for _pt, _fk in (
                ("kinematic",     "lr_kinematic_lateral"),
                ("drivable_path", "lr_drivable_lateral"),
                ("host_lane",     "lr_host_lateral"),
                ("hdmap",         "lr_hdmap_lateral"),
            ):
                _rel = lr_rels.get(_pt, {})
                if _rel.get("valid"):
                    _push(_fk, _rel.get("dist_lateral_m"))
            for _pt, _fk in (
                ("kinematic",     "lr_kinematic_inside"),
                ("drivable_path", "lr_drivable_inside"),
                ("host_lane",     "lr_host_inside"),
                ("hdmap",         "lr_hdmap_inside"),
            ):
                _rel = lr_rels.get(_pt, {})
                if _rel.get("valid"):
                    _push(_fk, 1.0 if _rel.get("inside_bounds") else 0.0)

    return history


# =============================================================================
# Drawing helpers
# =============================================================================

def _track_color(track_id: int) -> tuple[int, int, int]:
    return _TRACK_PALETTE_BGR[int(track_id) % len(_TRACK_PALETTE_BGR)]


def draw_gt_boxes(canvas: np.ndarray, frame_gt: dict,
                  frame_idx: int, highlight_id: str | None) -> np.ndarray:
    """Draw GT 2D boxes and a HUD overlay onto a copy of *canvas*."""
    canvas = canvas.copy()
    font   = cv2.FONT_HERSHEY_SIMPLEX
    boxes  = frame_gt.get("boxes_2d", [])
    ego    = frame_gt.get("ego_speed_kmh", 0.0)
    ts     = frame_gt.get("timestamp", 0.0)

    for box in boxes:
        cx, cy = box["center_x"], box["center_y"]
        bw, bh = box["length"],   box["width"]
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        color  = _GT_COLORS_BGR.get(box.get("type", 0), (200, 200, 200))
        name   = _TYPE_NAMES.get(box.get("type", 0), "?")
        is_hl  = (box.get("id") == highlight_id)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if is_hl else 2)
        cv2.putText(canvas, f"{name} {box['id'][:8]}", (x1, max(y1 - 5, 14)),
                    font, 0.40, color, 1, cv2.LINE_AA)
        if is_hl:
            cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 2)

    hud = [f"Frame: {frame_idx}", f"Speed: {ego:.1f} km/h",
           f"GT objects: {len(boxes)}", f"TS: {ts:.3f}"]
    m, lh = 10, 22
    ov = canvas.copy()
    cv2.rectangle(ov, (m, m), (m + 250, m + len(hud) * lh + m), (10, 10, 20), -1)
    cv2.addWeighted(ov, 0.60, canvas, 0.40, 0, canvas)
    for k, line in enumerate(hud):
        cv2.putText(canvas, line, (m + 8, m + (k + 1) * lh - 4),
                    font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_ekf_tracks(canvas: np.ndarray, frame_gt: dict,
                    highlight_tid: int | None = None,
                    show_sf_cross: bool = True) -> np.ndarray:
    """
    Draw EKF vehicle track bounding boxes and labels onto canvas.

    Each track gets a unique colour from _TRACK_PALETTE_BGR.  The selected
    track gets a white highlight border.  A crosshair marks the SF
    y0-projection bottom-centre pixel when the projection is valid.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for trk in frame_gt.get("vehicle_ekf_tracks", []):
        tid  = trk.get("track_id", 0)
        bbox = trk.get("bbox_xyxy")
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color  = _track_color(tid)
        is_sel = (tid == highlight_tid)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if is_sel else 2)
        if is_sel:
            cv2.rectangle(canvas, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (255, 255, 255), 1)
        x_m   = trk.get("x_veh", 0.0)
        coast = "~" if trk.get("is_coasting") else ""
        label = f"T{tid}{coast}  {x_m:.1f}m"
        lx, ly = x1, max(y1 - 6, 16)
        (tw, th), _ = cv2.getTextSize(label, font, 0.44, 1)
        cv2.rectangle(canvas, (lx - 1, ly - th - 3), (lx + tw + 2, ly + 2), (0, 0, 0), -1)
        cv2.putText(canvas, label, (lx, ly), font, 0.44, color, 1, cv2.LINE_AA)
        if show_sf_cross:
            sf = trk.get("sf_measurements") or {}
            y0 = sf.get("y0_proj") or {}
            if y0.get("valid"):
                bx, by = (x1 + x2) // 2, y2
                cv2.drawMarker(canvas, (bx, by), color,
                               cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    return canvas


def draw_json_paths(
    canvas: np.ndarray,
    frame_gt: dict,
    enabled_paths: set,
    vis: PerceptionVisualizer | None = None,
) -> np.ndarray:
    """Draw active lane-path overlays onto a copy of *canvas*."""
    canvas = canvas.copy()
    for path_key in _PATH_ORDER:
        if path_key not in enabled_paths:
            continue
        pd = frame_gt.get(path_key)
        if pd is None:
            continue
        pal   = _PATH_DRAW[path_key]
        thick = pal["thick"]
        if path_key == "kinematic":
            if vis is None:
                continue
            ctr = pd.get("center", [])
            if len(ctr) >= 2:
                vis.draw_kinematic_path(
                    canvas,
                    {"centre_line":    np.array(ctr, dtype=np.float64),
                     "left_boundary":  np.empty((0, 2), dtype=np.float64),
                     "right_boundary": np.empty((0, 2), dtype=np.float64)},
                    skip_wheels=True,
                )
            continue

        def _draw(pts_list, color, dot=False, valid=True):
            if len(pts_list) < 2:
                return
            arr = np.array(pts_list, dtype=np.int32).reshape(-1, 1, 2)
            t   = thick if valid else max(1, thick - 1)
            c   = color if valid else tuple(max(0, int(ch * 0.55)) for ch in color)
            cv2.polylines(canvas, [arr], False, c, t, cv2.LINE_AA)
            if dot:
                cv2.circle(canvas, tuple(np.array(pts_list[0], dtype=np.int32)), 6, c, -1)

        _draw(pd.get("center", []), pal["center"], dot=True, valid=pd.get("valid_center", False))
        _draw(pd.get("left",   []), pal["left"],             valid=pd.get("valid_left",   False))
        _draw(pd.get("right",  []), pal["right"],            valid=pd.get("valid_right",  False))

    return canvas


# =============================================================================
# Track time-series plot widget (embedded matplotlib)
# =============================================================================

class TrackPlotWidget(QWidget):
    """
    Embedded matplotlib canvas for a (track_id, field) time series.

    set_history()     — call once after data is loaded.
    set_target()      — configure which track + field to display.
    plot()            — re-draw the full series.
    update_cursor()   — move the vertical frame cursor cheaply (no full redraw).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)

        if _HAS_MPL:
            self._fig = _MplFigure(figsize=(5, 3.2), dpi=90, tight_layout=True)
            self._fig.patch.set_facecolor("#1e1e2e")
            self._ax = self._fig.add_subplot(111)
            self._canvas = _FigCanvas(self._fig)
            self._canvas.setMinimumHeight(220)
            vl.addWidget(self._canvas)
            self._cursor: Any = None
        else:
            vl.addWidget(QLabel(
                "matplotlib not found.\n"
                "Install it to enable track plots:\n  pip install matplotlib"))

        self._history:  dict[int, dict[str, list]] = {}
        self._n_frames: int        = 0
        self._track_id: int | None = None
        self._field:    str        = "x_veh"

    def set_history(self, history: dict, n_frames: int) -> None:
        self._history  = history
        self._n_frames = n_frames

    def set_target(self, track_id: int | None, field: str) -> None:
        self._track_id = track_id
        self._field    = field

    def plot(self, current_frame: int = 0) -> None:
        """Redraw the full time series for the current target."""
        if not _HAS_MPL:
            return
        ax = self._ax
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#888aaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a4e")
        ax.grid(True, color="#2a2a4e", linewidth=0.6, linestyle="--", alpha=0.7)

        tid   = self._track_id
        field = self._field
        label = _PLOT_FIELDS.get(field, field)

        if tid is not None and tid in self._history:
            pts = self._history[tid].get(field, [])
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                b, g, r = _track_color(tid)
                hex_col = f"#{r:02x}{g:02x}{b:02x}"   # BGR → RGB hex
                ax.plot(xs, ys, color=hex_col, linewidth=1.5,
                        marker=".", markersize=3, alpha=0.9)
                ax.set_xlabel("Frame", fontsize=8, color="#888aaa")
                ax.set_ylabel(label,   fontsize=8, color="#888aaa")
                ax.set_title(f"Track {tid}  \u00b7  {label}",
                             color="#cdd6f4", fontsize=9)
            else:
                ax.set_title(f"Track {tid}: no data for '{field}'",
                             color="#888aaa", fontsize=9)
        else:
            ax.set_title("Select a track in the Tracks tab, then click Plot",
                         color="#888aaa", fontsize=9)

        self._cursor = ax.axvline(x=current_frame, color="#ff7777",
                                  linewidth=1.2, linestyle="--", alpha=0.85)
        self._canvas.draw_idle()

    def update_cursor(self, frame_idx: int) -> None:
        """Slide the frame cursor without re-drawing the full series."""
        if not _HAS_MPL or self._cursor is None:
            return
        self._cursor.set_xdata([frame_idx, frame_idx])
        self._canvas.draw_idle()


# =============================================================================
# Scalable image widget
# =============================================================================

class ImageView(QLabel):
    """QLabel that always scales its pixmap to fill the available space."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 400)
        self.setStyleSheet("background:#000;")
        self._pix = None

    def set_bgr(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qi   = QImage(bytes(rgb), w, h, 3 * w, QImage.Format_RGB888)
        self._pix = QPixmap.fromImage(qi)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix:
            self.setPixmap(self._pix.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        self._rescale()
        super().resizeEvent(e)


# =============================================================================
# Main window
# =============================================================================

class DebugWindow(QMainWindow):

    _FPS_OPTS = ["1", "2", "5", "10", "15", "20", "30"]
    _TAB_INFO, _TAB_TRACKS, _TAB_DETECT, _TAB_PLOT, _TAB_JSON = 0, 1, 2, 3, 4

    def __init__(self, images: list, gt_data: list):
        super().__init__()
        self.images  = images
        self.gt_data = gt_data
        self.n       = min(len(images), len(gt_data))

        self._idx          = 0
        self._hl_gt_id:  str | None = None
        self._hl_tid:    int | None = None
        self._timer      = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

        self._enabled_paths: set = {"kinematic", "hdmap", "drivable_path", "host_lane"}
        self._box_mode: str  = "both"
        self._show_ekf: bool = True
        self._vis = PerceptionVisualizer(CameraCalibration.default_front())

        print("[viewer] Building track history index\u2026")
        self._track_history = build_track_history(gt_data)
        print(f"[viewer] Track history ready: {len(self._track_history)} unique track IDs")

        # Frame render cache — skip OpenCV when nothing visual changed
        self._cache_key: tuple | None     = None
        self._cache_img: np.ndarray | None = None

        self.setWindowTitle("Waymo ADAS Debug Viewer")
        self.resize(1640, 980)
        self._build_ui()
        self._apply_dark_theme()
        self._populate_plot_combos()
        self.render(0)

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(4)
        vbox.setContentsMargins(6, 6, 6, 6)

        ctrl = QHBoxLayout()
        vbox.addLayout(ctrl)

        def _btn(text: str, tip: str = "") -> QPushButton:
            b = QPushButton(text); b.setToolTip(tip); b.setFixedSize(40, 30); return b

        self.b_first = _btn("\u23ee", "First frame (Home)")
        self.b_back  = _btn("\u25c4",  "Previous (\u2190)")
        self.b_play  = _btn("\u25b6",  "Play/Pause (Space)")
        self.b_fwd   = _btn("\u25b6|", "Next (\u2192)")
        self.b_last  = _btn("\u23ed", "Last frame (End)")
        for b, fn in [(self.b_first, self._go_first), (self.b_back, self._step_back),
                      (self.b_play, self._toggle_play), (self.b_fwd, self._step_fwd),
                      (self.b_last, self._go_last)]:
            b.clicked.connect(fn); ctrl.addWidget(b)

        ctrl.addSpacing(10)
        ctrl.addWidget(QLabel("fps:"))
        self.fps_cb = QComboBox()
        self.fps_cb.addItems(self._FPS_OPTS)
        self.fps_cb.setCurrentText("10")
        self.fps_cb.setFixedWidth(58)
        ctrl.addWidget(self.fps_cb)

        ctrl.addSpacing(10)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.n - 1)
        self.slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self.slider, stretch=1)

        self.lbl_frame = QLabel(f"Frame  0 / {self.n - 1}")
        self.lbl_frame.setFixedWidth(130)
        ctrl.addWidget(self.lbl_frame)

        spl = QSplitter(Qt.Horizontal)
        vbox.addWidget(spl, stretch=1)
        self.img_view = ImageView()
        spl.addWidget(self.img_view)

        self._tabs = QTabWidget()
        self._tabs.setMaximumWidth(480)
        self._tabs.currentChanged.connect(self._on_tab_changed)
        spl.addWidget(self._tabs)
        spl.setStretchFactor(0, 3)
        spl.setStretchFactor(1, 1)

        self._build_tab_info()
        self._build_tab_tracks()
        self._build_tab_detect()
        self._build_tab_plot()
        self._build_tab_json()
        self.statusBar().showMessage(
            "Ready  \u2014  \u2190 \u2192 step  \u00b7  Space play/pause  \u00b7  T/P/J switch tab")

    def _build_tab_info(self) -> None:
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(6); vl.setContentsMargins(4, 4, 4, 4)

        grp = QGroupBox("Frame Info"); g = QGridLayout(grp)
        g.setSpacing(3); g.setContentsMargins(8, 14, 8, 8)

        def _row(grid, r, lbl):
            lb = QLabel(lbl + ":"); lb.setStyleSheet("color:#888aaa;")
            v  = QLabel("\u2014"); v.setStyleSheet("font-weight:600;")
            grid.addWidget(lb, r, 0); grid.addWidget(v, r, 1); return v

        self.v_frame = _row(g, 0, "Frame")
        self.v_ts    = _row(g, 1, "Timestamp")
        self.v_speed = _row(g, 2, "Ego speed")
        self.v_objs  = _row(g, 3, "GT objects")
        self.v_trks  = _row(g, 4, "EKF tracks")
        vl.addWidget(grp)

        grp_b = QGroupBox("Box Display"); bl = QHBoxLayout(grp_b)
        bl.setContentsMargins(8, 14, 8, 6); bl.addWidget(QLabel("Show:"))
        self.box_mode_cb = QComboBox()
        self.box_mode_cb.addItems(["Both", "GT Boxes", "YOLO + Tracks"])
        self.box_mode_cb.currentTextChanged.connect(self._on_box_mode_change)
        bl.addWidget(self.box_mode_cb, stretch=1)
        self.ekf_cb = QCheckBox("EKF overlay"); self.ekf_cb.setChecked(True)
        self.ekf_cb.toggled.connect(self._on_ekf_toggle); bl.addWidget(self.ekf_cb)
        vl.addWidget(grp_b)

        grp_p = QGroupBox("Active Paths"); pl = QGridLayout(grp_p)
        pl.setSpacing(4); pl.setContentsMargins(8, 14, 8, 6)
        self._path_cbs: dict = {}
        for i, (key, lbl) in enumerate([
            ("kinematic", "Kinematic"), ("hdmap", "HD Map"),
            ("drivable_path", "Drivable Path"), ("host_lane", "Host Lane"),
        ]):
            cb = QCheckBox(lbl); cb.setChecked(key in self._enabled_paths)
            cb.toggled.connect(lambda chk, k=key: self._on_path_toggle(k, chk))
            self._path_cbs[key] = cb; pl.addWidget(cb, i // 2, i % 2)
        vl.addWidget(grp_p)

        grp_pd = QGroupBox("Path Data"); pdl = QVBoxLayout(grp_pd)
        pdl.setContentsMargins(4, 14, 4, 4)
        self.path_table = QTableWidget(0, 11)
        self.path_table.setHorizontalHeaderLabels(
            ["Path","V_C","V_L","V_R","Conf_C","Conf_L","Conf_R","#Ctr","#L","#R","Source"])
        ph = self.path_table.horizontalHeader()
        ph.setSectionResizeMode(QHeaderView.ResizeToContents); ph.setStretchLastSection(True)
        self.path_table.setSelectionMode(QTableWidget.NoSelection)
        self.path_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.path_table.setAlternatingRowColors(True)
        self.path_table.verticalHeader().setVisible(False)
        self.path_table.setFixedHeight(130)
        pdl.addWidget(self.path_table); vl.addWidget(grp_pd)

        vl.addStretch()
        self._tabs.addTab(w, "Info")

    def _build_tab_tracks(self) -> None:
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(4); vl.setContentsMargins(4, 4, 4, 4)

        grp_ekf = QGroupBox("EKF Vehicle Tracks  (click \u2192 highlight + set plot target)")
        el = QVBoxLayout(grp_ekf); el.setContentsMargins(4, 14, 4, 4)
        self.ekf_table = QTableWidget(0, 14)
        self.ekf_table.setHorizontalHeaderLabels(
            ["ID","x(m)","y(m)","z(m)","vx","vy","spd","W","H","L","Hdg\u00b0","TTC","hits","~"])
        self.ekf_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.ekf_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.ekf_table.setSelectionMode(QTableWidget.SingleSelection)
        self.ekf_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ekf_table.setAlternatingRowColors(True)
        self.ekf_table.verticalHeader().setVisible(False)
        self.ekf_table.itemSelectionChanged.connect(self._on_ekf_row_select)
        el.addWidget(self.ekf_table); vl.addWidget(grp_ekf, stretch=2)

        grp_sf = QGroupBox("SF Measurements  (selected track, this frame)")
        sl = QVBoxLayout(grp_sf); sl.setContentsMargins(4, 14, 4, 4)
        self.sf_table = QTableWidget(0, 3)
        self.sf_table.setHorizontalHeaderLabels(["Source", "x (m)", "y (m)"])
        self.sf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sf_table.setSelectionMode(QTableWidget.NoSelection)
        self.sf_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sf_table.setAlternatingRowColors(True)
        self.sf_table.verticalHeader().setVisible(False)
        self.sf_table.setFixedHeight(195)
        sl.addWidget(self.sf_table); vl.addWidget(grp_sf)

        grp_lr = QGroupBox("Lane Relations  (selected track, this frame)")
        lrl = QVBoxLayout(grp_lr); lrl.setContentsMargins(4, 14, 4, 4)
        self.lr_table = QTableWidget(0, 7)
        self.lr_table.setHorizontalHeaderLabels(
            ["Path", "Valid", "Side", "Lat(m)", "Dist(m)", "BBox px", "Inside"])
        self.lr_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.lr_table.setSelectionMode(QTableWidget.NoSelection)
        self.lr_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.lr_table.setAlternatingRowColors(True)
        self.lr_table.verticalHeader().setVisible(False)
        self.lr_table.setFixedHeight(120)
        lrl.addWidget(self.lr_table); vl.addWidget(grp_lr)

        grp_det = QGroupBox("GT Detections  (click row \u2192 highlight on image)")
        dl = QVBoxLayout(grp_det); dl.setContentsMargins(4, 14, 4, 4)
        self.det_table = QTableWidget(0, 7)
        self.det_table.setHorizontalHeaderLabels(["ID","Type","Cx","Cy","W","H","Area"])
        self.det_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.det_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.det_table.setSelectionMode(QTableWidget.SingleSelection)
        self.det_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.det_table.setAlternatingRowColors(True)
        self.det_table.verticalHeader().setVisible(False)
        self.det_table.itemSelectionChanged.connect(self._on_gt_row_select)
        dl.addWidget(self.det_table); vl.addWidget(grp_det, stretch=1)

        self._tabs.addTab(w, "Tracks")

    def _build_tab_plot(self) -> None:
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(6); vl.setContentsMargins(6, 6, 6, 6)

        grp = QGroupBox("Track Field Time Series"); cl = QGridLayout(grp)
        cl.setSpacing(6); cl.setContentsMargins(8, 14, 8, 8)

        cl.addWidget(QLabel("Track ID:"), 0, 0)
        self.plot_tid_cb = QComboBox(); self.plot_tid_cb.setMinimumWidth(120)
        cl.addWidget(self.plot_tid_cb, 0, 1)

        cl.addWidget(QLabel("Field:"), 1, 0)
        self.plot_field_cb = QComboBox()
        for key, label in _PLOT_FIELDS.items():
            self.plot_field_cb.addItem(label, userData=key)
        cl.addWidget(self.plot_field_cb, 1, 1)

        self.plot_btn = QPushButton("Plot"); self.plot_btn.setFixedHeight(30)
        self.plot_btn.clicked.connect(self._on_plot_btn)
        cl.addWidget(self.plot_btn, 2, 0, 1, 2)
        vl.addWidget(grp)

        self.track_plot = TrackPlotWidget()
        self.track_plot.set_history(self._track_history, self.n)
        vl.addWidget(self.track_plot, stretch=1)
        self._tabs.addTab(w, "Plot")

    def _build_tab_json(self) -> None:
        w = QWidget(); vl = QVBoxLayout(w); vl.setContentsMargins(4, 4, 4, 4)
        self.json_edit = QTextEdit()
        self.json_edit.setReadOnly(True)
        self.json_edit.setFont(QFont("Consolas", 8))
        self.json_edit.setLineWrapMode(QTextEdit.NoWrap)
        vl.addWidget(self.json_edit)
        self._tabs.addTab(w, "JSON")

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
        QMainWindow, QWidget   { background: #1e1e2e; color: #cdd6f4; }
        QGroupBox {
            border: 1px solid #3d3d5c; border-radius: 5px;
            margin-top: 12px; padding-top: 4px;
            font-weight: bold; color: #7799ff; font-size: 11px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton {
            background: #3d3d5c; color: #cdd6f4;
            border: none; border-radius: 4px; font-size: 13px; }
        QPushButton:hover   { background: #5555aa; }
        QPushButton:pressed { background: #2233aa; }
        QTabWidget::pane    { border: 1px solid #3d3d5c; }
        QTabBar::tab {
            background: #2a2a3e; color: #888aaa;
            padding: 5px 14px; border-radius: 3px 3px 0 0; }
        QTabBar::tab:selected { background: #3d3d5c; color: #cdd6f4; }
        QTabBar::tab:hover    { background: #4a4a6a; }
        QSlider::groove:horizontal { height: 6px; background: #2a2a3e; border-radius: 3px; }
        QSlider::sub-page:horizontal { background: #7799ff; border-radius: 3px; }
        QSlider::handle:horizontal {
            width: 14px; height: 14px; margin: -4px 0;
            background: #aabbff; border-radius: 7px; }
        QTableWidget {
            background: #16213e; alternate-background-color: #1a1a2e;
            gridline-color: #2a2a4e; border: none; font-size: 11px; }
        QHeaderView::section {
            background: #2a2a3e; color: #cdd6f4;
            padding: 4px; border: none; font-weight: bold; }
        QTableWidget::item:selected { background: #4455bb; color: #fff; }
        QTextEdit   { background: #16213e; border: none; color: #b4d0f8; }
        QLabel      { color: #cdd6f4; }
        QComboBox {
            background: #3d3d5c; color: #cdd6f4;
            border: 1px solid #5555aa; border-radius: 3px; padding: 1px 5px; }
        QComboBox QAbstractItemView { background: #2a2a3e; color: #cdd6f4; }
        QStatusBar  { background: #2a2a3e; color: #888aaa; font-size: 11px; }
        QScrollBar:vertical   { background: #16213e; width: 10px; }
        QScrollBar::handle:vertical { background: #3d3d5c; border-radius: 4px; min-height: 20px; }
        QScrollBar:horizontal { background: #16213e; height: 10px; }
        QScrollBar::handle:horizontal { background: #3d3d5c; border-radius: 4px; }
        QSplitter::handle { background: #2a2a3e; width: 4px; }
        QCheckBox { color: #cdd6f4; spacing: 5px; }
        QCheckBox::indicator {
            width: 14px; height: 14px; border-radius: 3px;
            border: 1px solid #5555aa; background: #2a2a3e; }
        QCheckBox::indicator:checked { background: #7799ff; border-color: #7799ff; }
        QCheckBox::indicator:hover   { border-color: #aabbff; }
        QProgressDialog { background: #1e1e2e; color: #cdd6f4; }
        """)

    # =========================================================================
    # Rendering
    # =========================================================================

    def _render_image(self, idx: int) -> np.ndarray:
        """
        Build the annotated BGR frame, using a cache keyed on all visual state.
        Tab switching, table clicks, and SF breakdowns are free when the cache hits.
        """
        key = (idx, self._box_mode, frozenset(self._enabled_paths),
               self._hl_gt_id, self._hl_tid, self._show_ekf)
        if key == self._cache_key and self._cache_img is not None:
            return self._cache_img

        gt  = self.gt_data[idx]
        img = self.images[idx]
        if img is None:
            blank = np.zeros((600, 960, 3), dtype=np.uint8)
            cv2.putText(blank, "No image data", (350, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 2)
            return blank

        canvas = img.copy()
        if self._box_mode in ("gt", "both"):
            canvas = draw_gt_boxes(canvas, gt, idx, self._hl_gt_id)
        if self._enabled_paths:
            canvas = draw_json_paths(canvas, gt, self._enabled_paths, self._vis)
        if self._box_mode in ("pred", "both"):
            canvas = self._vis.draw_detections_and_tracks(canvas, gt)
        if self._show_ekf:
            canvas = draw_ekf_tracks(canvas, gt, highlight_tid=self._hl_tid)

        self._cache_key = key
        self._cache_img = canvas
        return canvas

    def render(self, idx: int) -> None:
        idx = max(0, min(idx, self.n - 1))
        self._idx = idx
        gt    = self.gt_data[idx]
        boxes = gt.get("boxes_2d", [])
        trks  = gt.get("vehicle_ekf_tracks", [])

        self.img_view.set_bgr(self._render_image(idx))

        self.v_frame.setText(f"{idx}  /  {self.n - 1}")
        self.v_ts.setText(f"{gt.get('timestamp', 0):.6f}")
        self.v_speed.setText(f"{gt.get('ego_speed_kmh', 0):.2f} km/h")
        self.v_objs.setText(str(len(boxes)))
        self.v_trks.setText(str(len(trks)))

        self._update_ekf_table(gt)
        self._update_lr_table(gt)
        self._update_det_table(gt)
        self._update_path_table(gt)
        if self._tabs.currentIndex() == self._TAB_DETECT:
            self._update_detect_tables(gt)

        if self._tabs.currentIndex() == self._TAB_JSON:
            self.json_edit.setPlainText(json.dumps(gt, indent=2))

        self.track_plot.update_cursor(idx)

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"Frame  {idx} / {self.n - 1}")
        n_dets = len(gt.get("detections", []))
        n_kal  = len(gt.get("tracks", []))
        self.statusBar().showMessage(
            f"Frame {idx}  \u00b7  {len(boxes)} GT  \u00b7  {n_dets} YOLO  \u00b7  "
            f"{n_kal} Kalman  \u00b7  {len(trks)} EKF  \u00b7  "
            f"{gt.get('ego_speed_kmh', 0):.1f} km/h  \u00b7  ts={gt.get('timestamp', 0):.3f}")

    # =========================================================================
    # Panel updaters
    # =========================================================================

    def _update_ekf_table(self, gt: dict) -> None:
        trks = gt.get("vehicle_ekf_tracks", [])
        self.ekf_table.blockSignals(True)
        self.ekf_table.setRowCount(len(trks))
        for r, t in enumerate(trks):
            tid   = t.get("track_id", "?")
            ttc   = t.get("ttc_s")
            coast = "\u2713" if t.get("is_coasting") else ""
            hdg_deg = math.degrees(t.get("heading_rad", 0.0))
            cells = [
                str(tid),
                f"{t.get('x_veh',    0):.1f}",
                f"{t.get('y_veh',    0):.1f}",
                f"{t.get('z_veh',    0):.2f}",
                f"{t.get('vx_veh',   0):.1f}",
                f"{t.get('vy_veh',   0):.1f}",
                f"{t.get('speed_mps',0):.1f}",
                f"{t.get('width_m',  0):.2f}",
                f"{t.get('height_m', 0):.2f}",
                f"{t.get('length_m', 0):.2f}",
                f"{hdg_deg:.1f}",
                f"{ttc:.1f}" if ttc is not None else "\u221e",
                str(t.get("hits", 0)),
                coast,
            ]
            try:
                b, g, rr = _track_color(int(tid))
                hex_col = f"#{rr:02x}{g:02x}{b:02x}"
            except (TypeError, ValueError):
                hex_col = "#aaaaaa"
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setData(Qt.UserRole, tid)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 0:
                    item.setForeground(QColor(hex_col))
                self.ekf_table.setItem(r, c, item)
        if self._hl_tid is not None:
            for r in range(self.ekf_table.rowCount()):
                it = self.ekf_table.item(r, 0)
                if it and it.data(Qt.UserRole) == self._hl_tid:
                    self.ekf_table.selectRow(r); break
        self.ekf_table.blockSignals(False)
        self._update_sf_table(gt)

    def _update_sf_table(self, gt: dict) -> None:
        """
        Show the full SF measurement bundle for the selected EKF track.

        Row colours: green = valid SF source, red = invalid, white = neutral.
        """
        trk_dict = None
        if self._hl_tid is not None:
            for t in gt.get("vehicle_ekf_tracks", []):
                if t.get("track_id") == self._hl_tid:
                    trk_dict = t; break

        sf = (trk_dict or {}).get("sf_measurements") or {}
        km = (trk_dict or {}).get("kalman_input")    or {}
        y0 = sf.get("y0_proj")     or {}
        hp = sf.get("height_proj") or {}
        wp = sf.get("width_proj")  or {}

        def _x(d): return f"{d['x_m']:.3f}" if d and d.get("x_m") is not None else "\u2014"
        def _y(d): return f"{d['y_m']:.3f}" if d and d.get("y_m") is not None else "\u2014"
        def _f(v): return f"{v:.4f}" if v is not None else "\u2014"

        # (source_label, x_val, y_val, valid_flag_or_None)
        rows: list[tuple] = [
            ("SF  y0-proj  (gnd plane)", _x(y0),               _y(y0),               y0.get("valid")),
            ("SF  height-prior  (x)",    _x(hp),               "\u2014",             hp.get("valid")),
            ("SF  width-prior  (x)",     _x(wp),               "\u2014",             wp.get("valid")),
            ("h_aspect  (px/fy)",        _f(sf.get("h_aspect")),"\u2014",            None),
            ("w_aspect  (px/fx)",        _f(sf.get("w_aspect")),"\u2014",            None),
            ("\u2500\u2500 Kalman input \u2500\u2500", "", "", None),
            ("Kalman  x_gnd",            _f(km.get("x_gnd")),  "\u2014",             None),
            ("Kalman  y_gnd",            "\u2014",             _f(km.get("y_gnd")),  None),
            ("use_size",                 str(km.get("use_size", "\u2014")), "\u2014", None),
        ]

        self.sf_table.blockSignals(True)
        self.sf_table.setRowCount(len(rows))
        for r, (src, xv, yv, valid) in enumerate(rows):
            v_col = ("#44cc88" if valid else "#cc4444") if valid is not None else None
            for c, txt in enumerate([src, xv, yv]):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 0:
                    if v_col:
                        item.setForeground(QColor(v_col))
                    elif src.startswith("\u2500"):
                        item.setForeground(QColor("#555577"))
                self.sf_table.setItem(r, c, item)
        self.sf_table.blockSignals(False)


    def _update_lr_table(self, gt: dict) -> None:
        """Show Lane Relations for the selected EKF track (4 paths x 7 cols)."""
        lr_entry = None
        if self._hl_tid is not None:
            for e in gt.get("lane_relations", []):
                if e.get("track_id") == self._hl_tid:
                    lr_entry = e; break

        relations = (lr_entry or {}).get("relations", {})
        _PATH_KEYS   = ["kinematic", "drivable_path", "host_lane", "hdmap"]
        _PATH_LABELS_LR = {
            "kinematic":     "Kinematic",
            "drivable_path": "Drivable",
            "host_lane":     "Host Lane",
            "hdmap":         "HD Map",
        }
        self.lr_table.blockSignals(True)
        self.lr_table.setRowCount(len(_PATH_KEYS))
        for r, pt in enumerate(_PATH_KEYS):
            rel    = relations.get(pt, {})
            valid  = rel.get("valid", False)
            inside = rel.get("inside_bounds", False)
            v_col  = "#44cc88" if valid   else "#cc4444"
            i_col  = "#44cc88" if inside  else "#cc4444"
            cells  = [
                _PATH_LABELS_LR[pt],
                "\u2713" if valid  else "\u2717",
                rel.get("side", "\u2014") if valid else "\u2014",
                f"{rel.get('dist_lateral_m',   0):.2f}" if valid else "\u2014",
                f"{rel.get('dist_to_center_m', 0):.2f}" if valid else "\u2014",
                f"{rel.get('dist_bbox_px',     0):.0f}" if valid else "\u2014",
                "\u2713" if inside else "\u2717",
            ]
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 1:
                    item.setForeground(QColor(v_col))
                elif c == 6:
                    item.setForeground(QColor(i_col if valid else "#555577"))
                self.lr_table.setItem(r, c, item)
        self.lr_table.blockSignals(False)

    def _build_tab_detect(self) -> None:
        """Detections tab: raw YOLO detections, Kalman tracks, 3D GT boxes."""
        w = QWidget(); vl = QVBoxLayout(w); vl.setSpacing(4); vl.setContentsMargins(4, 4, 4, 4)

        grp_yolo = QGroupBox("Raw YOLO Detections  (pre-tracking, this frame)")
        yl = QVBoxLayout(grp_yolo); yl.setContentsMargins(4, 14, 4, 4)
        self.yolo_table = QTableWidget(0, 7)
        self.yolo_table.setHorizontalHeaderLabels(
            ["#", "Class", "Conf", "x1", "y1", "x2", "y2"])
        self.yolo_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.yolo_table.setSelectionMode(QTableWidget.NoSelection)
        self.yolo_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.yolo_table.setAlternatingRowColors(True)
        self.yolo_table.verticalHeader().setVisible(False)
        self.yolo_table.setFixedHeight(150)
        yl.addWidget(self.yolo_table); vl.addWidget(grp_yolo)

        grp_kal = QGroupBox("Kalman Tracker  (all classes, confirmed tracks)")
        kl = QVBoxLayout(grp_kal); kl.setContentsMargins(4, 14, 4, 4)
        self.kalman_table = QTableWidget(0, 9)
        self.kalman_table.setHorizontalHeaderLabels(
            ["ID", "Class", "x(m)", "y(m)", "vx", "vy", "hits", "TTC", "~"])
        self.kalman_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.kalman_table.setSelectionMode(QTableWidget.NoSelection)
        self.kalman_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.kalman_table.setAlternatingRowColors(True)
        self.kalman_table.verticalHeader().setVisible(False)
        self.kalman_table.setFixedHeight(150)
        kl.addWidget(self.kalman_table); vl.addWidget(grp_kal)

        grp_3d = QGroupBox("GT 3D Boxes  (Waymo laser labels, vehicle frame)")
        gl = QVBoxLayout(grp_3d); gl.setContentsMargins(4, 14, 4, 4)
        self.gt3d_table = QTableWidget(0, 10)
        self.gt3d_table.setHorizontalHeaderLabels(
            ["ID", "Type", "X(m)", "Y(m)", "Z(m)", "L", "W", "H", "Hdg\u00b0", "LiDAR"])
        self.gt3d_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.gt3d_table.setSelectionMode(QTableWidget.NoSelection)
        self.gt3d_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.gt3d_table.setAlternatingRowColors(True)
        self.gt3d_table.verticalHeader().setVisible(False)
        gl.addWidget(self.gt3d_table); vl.addWidget(grp_3d, stretch=1)

        self._tabs.addTab(w, "Detections")

    def _update_detect_tables(self, gt: dict) -> None:
        """Refresh YOLO detections, Kalman tracks, and 3D GT box tables."""
        _DET_COLORS = {
            "vehicle":    "#6699ff",
            "pedestrian": "#44cc88",
            "cyclist":    "#ffcc44",
            "other":      "#bbbbcc",
        }
        _TYPE_COLORS_3D = {1: "#6699ff", 2: "#44cc88", 3: "#bbbbcc", 4: "#ffcc44"}
        _TYPE_NAMES_3D  = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist"}

        # ── Raw YOLO detections ─────────────────────────────────────────────
        dets = gt.get("detections", [])
        self.yolo_table.blockSignals(True)
        self.yolo_table.setRowCount(len(dets))
        for r, d in enumerate(dets):
            bbox = d.get("bbox_xyxy") or [0, 0, 0, 0]
            cells = [
                str(r),
                d.get("class_name", "?"),
                f"{d.get('confidence', 0):.3f}",
                f"{bbox[0]:.0f}", f"{bbox[1]:.0f}",
                f"{bbox[2]:.0f}", f"{bbox[3]:.0f}",
            ]
            col = _DET_COLORS.get(d.get("class_name", ""), "#bbbbcc")
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 1:
                    item.setForeground(QColor(col))
                self.yolo_table.setItem(r, c, item)
        self.yolo_table.blockSignals(False)

        # ── Kalman TrackManager tracks (all classes) ────────────────────────
        tracks = gt.get("tracks", [])
        self.kalman_table.blockSignals(True)
        self.kalman_table.setRowCount(len(tracks))
        for r, t in enumerate(tracks):
            tid   = t.get("track_id", "?")
            ttc   = t.get("ttc_s")
            coast = "\u2713" if t.get("is_coasting") else ""
            cells = [
                str(tid),
                t.get("class_name", "?"),
                f"{t.get('x_veh', 0):.1f}",
                f"{t.get('y_veh', 0):.1f}",
                f"{t.get('vx_veh', 0):.1f}",
                f"{t.get('vy_veh', 0):.1f}",
                str(t.get("hits", 0)),
                f"{ttc:.1f}" if ttc is not None else "\u221e",
                coast,
            ]
            try:
                b, g, rr = _track_color(int(tid))
                hex_col = f"#{rr:02x}{g:02x}{b:02x}"
            except (TypeError, ValueError):
                hex_col = "#aaaaaa"
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 0:
                    item.setForeground(QColor(hex_col))
                self.kalman_table.setItem(r, c, item)
        self.kalman_table.blockSignals(False)

        # ── GT 3D laser-label boxes ─────────────────────────────────────────
        boxes3d = gt.get("boxes_3d", [])
        self.gt3d_table.blockSignals(True)
        self.gt3d_table.setRowCount(len(boxes3d))
        for r, b in enumerate(boxes3d):
            type_id = b.get("type", 0)
            cells = [
                str(b.get("id", "?"))[:8],
                _TYPE_NAMES_3D.get(type_id, "?"),
                f"{b.get('center_x', 0):.1f}",
                f"{b.get('center_y', 0):.1f}",
                f"{b.get('center_z', 0):.2f}",
                f"{b.get('length',   0):.2f}",
                f"{b.get('width',    0):.2f}",
                f"{b.get('height',   0):.2f}",
                f"{math.degrees(b.get('heading', 0)):.1f}",
                str(b.get("num_lidar_points", 0)),
            ]
            col = _TYPE_COLORS_3D.get(type_id, "#bbbbcc")
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if c == 1:
                    item.setForeground(QColor(col))
                self.gt3d_table.setItem(r, c, item)
        self.gt3d_table.blockSignals(False)

    def _update_det_table(self, gt: dict) -> None:
        boxes = gt.get("boxes_2d", [])
        self.det_table.blockSignals(True)
        self.det_table.setRowCount(len(boxes))
        for r, box in enumerate(boxes):
            name  = _TYPE_NAMES.get(box.get("type", 0), "?")
            color = _ROW_COLORS.get(name, "#aaaaaa")
            cells = [
                box["id"][:8], name,
                f"{box['center_x']:.1f}", f"{box['center_y']:.1f}",
                f"{box['length']:.1f}", f"{box['width']:.1f}",
                str(round(box["length"] * box["width"])),
            ]
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setData(Qt.UserRole, box["id"])
                if c == 1:
                    item.setForeground(QColor(color))
                self.det_table.setItem(r, c, item)
        if self._hl_gt_id:
            for r in range(self.det_table.rowCount()):
                it = self.det_table.item(r, 0)
                if it and it.data(Qt.UserRole) == self._hl_gt_id:
                    self.det_table.selectRow(r); break
        self.det_table.blockSignals(False)

    def _update_path_table(self, gt: dict) -> None:
        rows = []
        for key in _PATH_ORDER:
            if key not in self._enabled_paths:
                continue
            pd = gt.get(key)
            if pd is None:
                rows.append((_PATH_LABELS[key], None, None, None,
                              None, None, None, None, None, None, "\u2014"))
                continue
            rows.append((
                _PATH_LABELS[key],
                pd.get("valid_center", False), pd.get("valid_left", False),
                pd.get("valid_right",  False),
                pd.get("confidence_center", 0.0), pd.get("confidence_left", 0.0),
                pd.get("confidence_right",  0.0),
                len(pd.get("center", [])), len(pd.get("left", [])),
                len(pd.get("right",  [])),
                pd.get("source", "\u2014"),
            ))
        self.path_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            nm, vc, vl, vr, cc, cl, cr, nc, nl, nr, src = row
            def _vs(v): return "\u2014" if v is None else ("\u2713" if v else "\u2717")
            def _vc(v): return None if v is None else ("#44cc88" if v else "#cc4444")
            def _fs(v): return "\u2014" if v is None else f"{v:.3f}"
            def _ns(v): return "\u2014" if v is None else str(v)
            cells = [nm, _vs(vc), _vs(vl), _vs(vr),
                     _fs(cc), _fs(cl), _fs(cr), _ns(nc), _ns(nl), _ns(nr), src]
            vcols = [None, _vc(vc), _vc(vl), _vc(vr),
                     None, None, None, None, None, None, None]
            for c, (txt, col) in enumerate(zip(cells, vcols)):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if col:
                    item.setForeground(QColor(col))
                self.path_table.setItem(r, c, item)

    def _populate_plot_combos(self) -> None:
        self.plot_tid_cb.clear()
        self.plot_tid_cb.addItem("\u2014 select \u2014", userData=None)
        for tid in sorted(self._track_history.keys()):
            n = max((len(v) for v in self._track_history[tid].values() if v), default=0)
            self.plot_tid_cb.addItem(f"Track {tid}  ({n} frames)", userData=tid)

    # =========================================================================
    # Callbacks
    # =========================================================================

    @pyqtSlot()
    def _on_ekf_row_select(self) -> None:
        sel = self.ekf_table.selectedItems()
        if sel:
            try:
                self._hl_tid = int(sel[0].data(Qt.UserRole))
            except (TypeError, ValueError):
                self._hl_tid = sel[0].data(Qt.UserRole)
        else:
            self._hl_tid = None
        for i in range(self.plot_tid_cb.count()):
            if self.plot_tid_cb.itemData(i) == self._hl_tid:
                self.plot_tid_cb.setCurrentIndex(i); break
        self._update_sf_table(self.gt_data[self._idx])
        self._update_lr_table(self.gt_data[self._idx])
        self._cache_key = None
        self.img_view.set_bgr(self._render_image(self._idx))

    @pyqtSlot()
    def _on_gt_row_select(self) -> None:
        sel = self.det_table.selectedItems()
        self._hl_gt_id = sel[0].data(Qt.UserRole) if sel else None
        self._cache_key = None
        self.img_view.set_bgr(self._render_image(self._idx))

    @pyqtSlot(int)
    def _on_tab_changed(self, idx: int) -> None:
        if idx == self._TAB_DETECT:
            self._update_detect_tables(self.gt_data[self._idx])
        elif idx == self._TAB_JSON:
            self.json_edit.setPlainText(json.dumps(self.gt_data[self._idx], indent=2))

    def _on_plot_btn(self) -> None:
        tid   = self.plot_tid_cb.currentData()
        field = self.plot_field_cb.currentData()
        if tid is None:
            return
        self.track_plot.set_target(tid, field)
        self.track_plot.plot(current_frame=self._idx)
        self._tabs.setCurrentIndex(self._TAB_PLOT)

    def _on_box_mode_change(self, text: str) -> None:
        self._box_mode = {"GT Boxes": "gt", "YOLO + Tracks": "pred", "Both": "both"}.get(text, "both")
        self._cache_key = None; self.img_view.set_bgr(self._render_image(self._idx))

    def _on_ekf_toggle(self, checked: bool) -> None:
        self._show_ekf = checked
        self._cache_key = None; self.img_view.set_bgr(self._render_image(self._idx))

    def _on_path_toggle(self, key: str, checked: bool) -> None:
        if checked: self._enabled_paths.add(key)
        else:       self._enabled_paths.discard(key)
        self._cache_key = None; self.img_view.set_bgr(self._render_image(self._idx))

    # =========================================================================
    # Transport
    # =========================================================================

    def _on_slider(self, val: int) -> None: self._stop(); self.render(val)
    def _go_first(self):  self._stop(); self.render(0)
    def _go_last(self):   self._stop(); self.render(self.n - 1)
    def _step_back(self): self._stop(); self.render(self._idx - 1)
    def _step_fwd(self):  self._stop(); self.render(self._idx + 1)

    def _stop(self) -> None:
        self._timer.stop(); self.b_play.setText("\u25b6")

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._stop()
        else:
            fps = int(self.fps_cb.currentText())
            self._timer.start(1000 // fps); self.b_play.setText("\u23f8")

    @pyqtSlot()
    def _on_tick(self) -> None:
        if self._idx < self.n - 1: self.render(self._idx + 1)
        else:                       self._stop()

    def keyPressEvent(self, e) -> None:
        k = e.key()
        if   k == Qt.Key_Left:                    self._step_back()
        elif k == Qt.Key_Right:                   self._step_fwd()
        elif k == Qt.Key_Space:                   self._toggle_play()
        elif k == Qt.Key_Home:                    self._go_first()
        elif k == Qt.Key_End:                     self._go_last()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):    self._change_speed(+1)
        elif k == Qt.Key_Minus:                   self._change_speed(-1)
        elif k == Qt.Key_T:  self._tabs.setCurrentIndex(self._TAB_TRACKS)
        elif k == Qt.Key_D:  self._tabs.setCurrentIndex(self._TAB_DETECT)
        elif k == Qt.Key_P:  self._tabs.setCurrentIndex(self._TAB_PLOT)
        elif k == Qt.Key_J:  self._tabs.setCurrentIndex(self._TAB_JSON)
        else:                super().keyPressEvent(e)

    def _change_speed(self, delta: int) -> None:
        i   = self._FPS_OPTS.index(self.fps_cb.currentText())
        new = max(0, min(i + delta, len(self._FPS_OPTS) - 1))
        self.fps_cb.setCurrentText(self._FPS_OPTS[new])
        if self._timer.isActive():
            self._timer.setInterval(1000 // int(self._FPS_OPTS[new]))


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Waymo ADAS Debug Viewer \u2014 EKF Measurement & Track Analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--tfrecord",   default="", help="Path to .tfrecord segment file")
    ap.add_argument("--json",       default="",
                    help="Path to pipeline output .json  (auto-detected by same basename)")
    ap.add_argument("--max-frames", dest="max_frames", type=int, default=200,
                    help="Maximum frames to load  (default: 200)")
    args = ap.parse_args()

    qapp = QApplication(sys.argv)
    qapp.setApplicationName("Waymo ADAS Debug Viewer")

    tfrecord = args.tfrecord
    if not tfrecord:
        tfrecord, _ = QFileDialog.getOpenFileName(
            None, "Open TFRecord Segment", ".",
            "TFRecord files (*.tfrecord);;All files (*)")
        if not tfrecord:
            print("No file selected \u2014 exiting.")
            sys.exit(0)

    json_path = args.json
    if not json_path:
        candidate = os.path.splitext(tfrecord)[0] + ".json"
        if os.path.exists(candidate):
            json_path = candidate
            print(f"[viewer] Auto-detected JSON: {json_path}")

    prog = QProgressDialog("Extracting frames from tfrecord\u2026", None, 0, args.max_frames)
    prog.setWindowTitle("Loading"); prog.setMinimumDuration(0)
    prog.setWindowModality(Qt.ApplicationModal); prog.setValue(0); prog.show()
    qapp.processEvents()

    def _cb(done: int):
        prog.setValue(done); qapp.processEvents()

    print(f"[viewer] Loading tfrecord: {tfrecord}")
    images, gt_list = load_tfrecord(tfrecord, args.max_frames, _cb)
    prog.close()

    if json_path and os.path.exists(json_path):
        print(f"[viewer] Loading JSON: {json_path}")
        with open(json_path) as f:
            gt_list = json.load(f)

    n = min(len(images), len(gt_list))
    n_ekf = sum(len(g.get("vehicle_ekf_tracks", [])) for g in gt_list)
    print(f"[viewer] Ready: {n} frames  \u00b7  {n_ekf} total EKF track-frames")

    win = DebugWindow(images[:n], gt_list[:n])
    win.show()
    sys.exit(qapp.exec_())


if __name__ == "__main__":
    main()
