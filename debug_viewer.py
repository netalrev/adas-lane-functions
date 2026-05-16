#!/usr/bin/env python3
"""
Waymo Debug Viewer — fully local desktop GUI

Usage:
    python debug_viewer.py
    python debug_viewer.py --tfrecord src/data/<segment>.tfrecord
    python debug_viewer.py --tfrecord src/data/<segment>.tfrecord --json src/data/<segment>.json
    python debug_viewer.py --tfrecord <path> --max-frames 100

- If --tfrecord only:   shows the clip with boxes extracted directly from the segment.
- If --json also given (or auto-detected by same basename): boxes + JSON data per frame.
- If neither argument: a file-picker dialog opens.

Keyboard:
    ←/→     step back / forward
    Space   play / pause
    Home    jump to first frame
    End     jump to last frame
    +/-     increase / decrease playback speed
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
from src.visualization.visualizer import CameraCalibration, PerceptionVisualizer
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ── color maps ────────────────────────────────────────────────
_COLORS_BGR = {
    1: (255, 100,  50),   # Vehicle    — blue
    2: ( 50, 220,  80),   # Pedestrian — green
    3: (200, 200, 200),   # Sign       — grey
    4: (255, 220,  50),   # Cyclist    — yellow
}
_TYPE_NAMES = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist"}
_ROW_COLORS = {
    "Vehicle":    "#6699ff",
    "Pedestrian": "#44cc88",
    "Cyclist":    "#ffcc44",
    "Sign":       "#bbbbcc",
}


# ── data loading ──────────────────────────────────────────────

def load_tfrecord(tfrecord_path: str, max_frames: int,
                  progress_cb=None) -> tuple[list, list]:
    """
    Extract images and ground-truth boxes directly from a tfrecord.
    Returns (images, gt_list).
      images  — list of BGR np.ndarray (or None if camera missing)
      gt_list — list of dicts {timestamp, ego_speed_kmh, boxes_2d:[...]}
    """
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    images, gt_list = [], []
    prev_pos, prev_time = None, None

    for i, raw in enumerate(dataset):
        if i >= max_frames:
            break
        frame = open_dataset.Frame()
        frame.ParseFromString(bytes(raw.numpy()))

        # Front camera image
        img = None
        for cam in frame.images:
            if cam.name == open_dataset.CameraName.FRONT:
                arr = tf.image.decode_jpeg(cam.image).numpy()
                img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                break
        images.append(img)

        # Ego speed from pose displacement
        mat = frame.pose.transform
        cx2, cy2 = mat[3], mat[7]
        t2 = frame.timestamp_micros / 1e6
        if prev_pos is not None and (t2 - prev_time) > 0:
            speed = (np.sqrt((cx2 - prev_pos[0])**2 + (cy2 - prev_pos[1])**2)
                     / (t2 - prev_time) * 3.6)
        else:
            speed = 0.0
        prev_pos, prev_time = (cx2, cy2), t2

        # 2D boxes from embedded camera labels
        boxes_2d = []
        for cam_labels in frame.camera_labels:
            if cam_labels.name == open_dataset.CameraName.FRONT:
                for lbl in cam_labels.labels:
                    b = lbl.box
                    boxes_2d.append({
                        "id":       lbl.id,
                        "type":     lbl.type,
                        "center_x": b.center_x,
                        "center_y": b.center_y,
                        "length":   b.length,
                        "width":    b.width,
                    })

        gt_list.append({
            "timestamp":     t2,
            "ego_speed_kmh": speed,
            "boxes_2d":      boxes_2d,
        })

        if progress_cb:
            progress_cb(i + 1)

    return images, gt_list


# ── path color palette (BGR) — mirrors visualizer.py ─────────
_PATH_DRAW: dict[str, dict] = {
    "kinematic": {
        "center": (  0, 220, 255),   # yellow
        "left":   (  0, 190, 255),
        "right":  (  0, 190, 255),
        "thick":  2,
    },
    "hdmap": {
        "center": ( 20, 180,  20),   # mid-green
        "left":   ( 20, 230,  20),   # bright green
        "right":  ( 10, 160,  10),   # dark green
        "thick":  3,
    },
    "drivable_path": {
        "center": (200, 200,   0),   # bright cyan
        "left":   (140, 200,   0),   # olive-cyan
        "right":  (140, 200,   0),
        "thick":  2,
    },
    "host_lane": {
        "center": (210,  30, 210),   # purple (unused — no HL centre)
        "left":   (210,  30, 210),   # purple-left
        "right":  (180,   0, 255),   # magenta-right
        "thick":  3,
    },
}
_PATH_LABELS: dict[str, str] = {
    "kinematic":     "Kinematic",
    "hdmap":         "HD Map",
    "drivable_path": "Drivable",
    "host_lane":     "Host Lane",
}
_PATH_ORDER = ["kinematic", "hdmap", "drivable_path", "host_lane"]


def draw_json_paths(
    img_bgr: np.ndarray,
    frame_gt: dict,
    enabled_paths: set,
    vis: PerceptionVisualizer | None = None,
) -> np.ndarray:
    """
    Draw unified PathData JSON paths onto *img_bgr* (in-place copy).

    Pixel-coord paths (hdmap, drivable_path, host_lane) are drawn directly.
    Kinematic path (vehicle-frame XY) is projected via *vis* when provided.
    Per-side validity is respected — invalid sides are silently skipped.
    """
    canvas = img_bgr.copy()

    for path_key in _PATH_ORDER:
        if path_key not in enabled_paths:
            continue
        pd = frame_gt.get(path_key)
        if pd is None:
            continue

        pal   = _PATH_DRAW[path_key]
        thick = pal["thick"]

        if path_key == "kinematic":
            # Vehicle-frame XY — project with PerceptionVisualizer
            if vis is None:
                continue
            ctr = pd.get("center", [])
            if len(ctr) >= 2:
                path_dict = {
                    "centre_line":    np.array(ctr, dtype=np.float64),
                    "left_boundary":  np.empty((0, 2), dtype=np.float64),
                    "right_boundary": np.empty((0, 2), dtype=np.float64),
                }
                vis.draw_kinematic_path(canvas, path_dict, skip_wheels=True)
            continue

        # Pixel-coord paths: draw any side that has points.
        # Validity is informational (shown in the Path Data table) — never
        # used as a gate here so a partially-detected path is always visible.
        valid_c = pd.get("valid_center", False)
        valid_l = pd.get("valid_left",   False)
        valid_r = pd.get("valid_right",  False)

        def _draw(pts_list, color, dot=False, is_valid=True):
            if len(pts_list) < 2:
                return
            arr = np.array(pts_list, dtype=np.int32).reshape(-1, 1, 2)
            # Draw invalid sides thinner and slightly dimmed so they're
            # visible but clearly marked as below-threshold detections.
            draw_thick = thick if is_valid else max(1, thick - 1)
            draw_color = color if is_valid else tuple(max(0, int(c * 0.55)) for c in color)
            cv2.polylines(canvas, [arr], False, draw_color, draw_thick, cv2.LINE_AA)
            if dot:
                cv2.circle(canvas, tuple(np.array(pts_list[0], dtype=np.int32)),
                           6, draw_color, -1, cv2.LINE_AA)

        _draw(pd.get("center", []), pal["center"], dot=True, is_valid=valid_c)
        _draw(pd.get("left",   []), pal["left"],              is_valid=valid_l)
        _draw(pd.get("right",  []), pal["right"],             is_valid=valid_r)

    return canvas


# ── annotation drawing ────────────────────────────────────────

def draw_boxes(img_bgr: np.ndarray, frame_gt: dict,
               frame_idx: int, highlight_id: str | None) -> np.ndarray:
    canvas = img_bgr.copy()
    boxes  = frame_gt.get("boxes_2d", [])
    ego    = frame_gt.get("ego_speed_kmh", 0.0)
    ts     = frame_gt.get("timestamp", 0.0)
    font   = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        cx, cy = box["center_x"], box["center_y"]
        bw, bh = box["length"],   box["width"]
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)

        color = _COLORS_BGR.get(box.get("type", 0), (200, 200, 200))
        name  = _TYPE_NAMES.get(box.get("type", 0), "Unknown")
        is_hl = (box["id"] == highlight_id)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if is_hl else 2)
        cv2.putText(canvas, f"{name}  {box['id'][:8]}",
                    (x1, max(y1 - 5, 12)), font, 0.45, color, 1, cv2.LINE_AA)

        if is_hl:
            cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4),
                          (255, 255, 255), 2)

    # HUD overlay
    hud = [f"Frame: {frame_idx}", f"Speed: {ego:.1f} km/h",
           f"Objects: {len(boxes)}", f"TS: {ts:.3f}"]
    m, lh = 10, 22
    ov = canvas.copy()
    cv2.rectangle(ov, (m, m), (m + 245, m + len(hud) * lh + m), (10, 10, 20), -1)
    cv2.addWeighted(ov, 0.60, canvas, 0.40, 0, canvas)
    for k, line in enumerate(hud):
        cv2.putText(canvas, line, (m + 8, m + (k + 1) * lh - 4),
                    font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ── scalable image widget ─────────────────────────────────────

class ImageView(QLabel):
    """QLabel that scales its pixmap to fill all available space."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 400)
        self.setStyleSheet("background:#000;")
        self._pix = None

    def set_bgr(self, img: np.ndarray):
        h, w = img.shape[:2]
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qi   = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self._pix = QPixmap.fromImage(qi)
        self._rescale()

    def _rescale(self):
        if self._pix:
            self.setPixmap(self._pix.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        self._rescale()
        super().resizeEvent(e)


# ── main window ───────────────────────────────────────────────

class DebugWindow(QMainWindow):

    _FPS_OPTS = ["1", "2", "5", "10", "15", "20", "30"]

    def __init__(self, images: list, gt_data: list):
        super().__init__()
        self.images  = images
        self.gt_data = gt_data
        self.n       = min(len(images), len(gt_data))
        self._idx    = 0
        self._hl_id  = None
        self._timer  = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        # Path type toggles — default: all four enabled
        self._enabled_paths: set = {"kinematic", "hdmap", "drivable_path", "host_lane"}
        # Visualizer for kinematic projection (default Waymo-like front calib)
        self._vis = PerceptionVisualizer(CameraCalibration.default_front())

        self.setWindowTitle("Waymo Debug Viewer")
        self.resize(1440, 900)
        self._build_ui()
        self._apply_dark_theme()
        self.render(0)

    # ── UI construction ───────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setSpacing(4)
        vbox.setContentsMargins(6, 6, 6, 6)

        # ── Transport bar ──────────────────────────────────
        ctrl = QHBoxLayout()
        vbox.addLayout(ctrl)

        def mkbtn(text, tip=""):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedSize(40, 30)
            return b

        self.b_first = mkbtn("⏮", "First frame  (Home)")
        self.b_back  = mkbtn("◀",  "Previous frame  (←)")
        self.b_play  = mkbtn("▶",  "Play / Pause  (Space)")
        self.b_fwd   = mkbtn("▶|", "Next frame  (→)")
        self.b_last  = mkbtn("⏭", "Last frame  (End)")

        self.b_first.clicked.connect(self._go_first)
        self.b_back.clicked.connect(self._step_back)
        self.b_play.clicked.connect(self._toggle_play)
        self.b_fwd.clicked.connect(self._step_fwd)
        self.b_last.clicked.connect(self._go_last)

        for b in (self.b_first, self.b_back, self.b_play, self.b_fwd, self.b_last):
            ctrl.addWidget(b)

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

        # ── Horizontal splitter: image | right panel ───────
        spl = QSplitter(Qt.Horizontal)
        vbox.addWidget(spl, stretch=1)

        self.img_view = ImageView()
        spl.addWidget(self.img_view)

        # Right panel
        rp = QWidget()
        rp.setMaximumWidth(430)
        rl = QVBoxLayout(rp)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        spl.addWidget(rp)
        spl.setStretchFactor(0, 3)
        spl.setStretchFactor(1, 1)

        # Frame Info card
        grp_info = QGroupBox("Frame Info")
        g = QGridLayout(grp_info)
        g.setSpacing(3)
        g.setContentsMargins(8, 14, 8, 8)

        def info_row(grid, row, label):
            lbl = QLabel(label + ":")
            lbl.setStyleSheet("color:#888aaa;")
            val = QLabel("—")
            val.setStyleSheet("font-weight:600;")
            grid.addWidget(lbl, row, 0)
            grid.addWidget(val, row, 1)
            return val

        self.v_frame = info_row(g, 0, "Frame")
        self.v_ts    = info_row(g, 1, "Timestamp")
        self.v_speed = info_row(g, 2, "Ego speed")
        self.v_objs  = info_row(g, 3, "Objects")
        rl.addWidget(grp_info)

        # ── Active Path Toggles ─────────────────────────────────────
        grp_paths = QGroupBox("Active Paths")
        pl = QGridLayout(grp_paths)
        pl.setSpacing(4)
        pl.setContentsMargins(8, 14, 8, 6)
        self._path_cbs: dict = {}
        _path_cfg = [
            ("kinematic",     "Kinematic"),
            ("hdmap",         "HD Map"),
            ("drivable_path", "Drivable Path"),
            ("host_lane",     "Host Lane"),
        ]
        for i, (key, label) in enumerate(_path_cfg):
            cb = QCheckBox(label)
            cb.setChecked(key in self._enabled_paths)
            cb.toggled.connect(lambda checked, k=key: self._on_path_toggle(k, checked))
            self._path_cbs[key] = cb
            pl.addWidget(cb, i // 2, i % 2)
        rl.addWidget(grp_paths)

        # ── Path Data Table ──────────────────────────────────────
        grp_path_data = QGroupBox("Path Data")
        pdl = QVBoxLayout(grp_path_data)
        pdl.setContentsMargins(4, 14, 4, 4)
        self.path_table = QTableWidget(0, 11)
        self.path_table.setHorizontalHeaderLabels(
            ["Path", "V_C", "V_L", "V_R",
             "Conf_C", "Conf_L", "Conf_R",
             "#Ctr", "#L", "#R", "Source"])
        ph = self.path_table.horizontalHeader()
        ph.setSectionResizeMode(QHeaderView.ResizeToContents)
        ph.setStretchLastSection(True)
        self.path_table.setSelectionMode(QTableWidget.NoSelection)
        self.path_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.path_table.setAlternatingRowColors(True)
        self.path_table.verticalHeader().setVisible(False)
        self.path_table.setFixedHeight(130)
        pdl.addWidget(self.path_table)
        rl.addWidget(grp_path_data)

        # Detections table
        grp_det = QGroupBox("Detections  (click row → highlight on image)")
        dl = QVBoxLayout(grp_det)
        dl.setContentsMargins(4, 14, 4, 4)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Type", "Cx", "Cy", "W", "H", "Area"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._on_row_select)
        dl.addWidget(self.table)
        rl.addWidget(grp_det, stretch=2)

        # JSON viewer
        grp_json = QGroupBox("Frame JSON")
        jl = QVBoxLayout(grp_json)
        jl.setContentsMargins(4, 14, 4, 4)
        self.json_edit = QTextEdit()
        self.json_edit.setReadOnly(True)
        self.json_edit.setFont(QFont("Consolas", 9))
        self.json_edit.setLineWrapMode(QTextEdit.NoWrap)
        jl.addWidget(self.json_edit)
        rl.addWidget(grp_json, stretch=1)

        self.statusBar().showMessage(
            "Ready  —  ← → step  •  Space play/pause  •  +/− speed")

    def _apply_dark_theme(self):
        self.setStyleSheet("""
        QMainWindow, QWidget   { background: #1e1e2e; color: #cdd6f4; }
        QGroupBox {
            border: 1px solid #3d3d5c; border-radius: 5px;
            margin-top: 12px; padding-top: 4px;
            font-weight: bold; color: #7799ff; font-size: 11px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 10px; padding: 0 4px;
        }
        QPushButton {
            background: #3d3d5c; color: #cdd6f4;
            border: none; border-radius: 4px; font-size: 13px;
        }
        QPushButton:hover   { background: #5555aa; }
        QPushButton:pressed { background: #2233aa; }
        QSlider::groove:horizontal {
            height: 6px; background: #2a2a3e; border-radius: 3px;
        }
        QSlider::sub-page:horizontal { background: #7799ff; border-radius: 3px; }
        QSlider::handle:horizontal {
            width: 14px; height: 14px; margin: -4px 0;
            background: #aabbff; border-radius: 7px;
        }
        QTableWidget {
            background: #16213e; alternate-background-color: #1a1a2e;
            gridline-color: #2a2a4e; border: none; font-size: 11px;
        }
        QHeaderView::section {
            background: #2a2a3e; color: #cdd6f4;
            padding: 4px; border: none; font-weight: bold;
        }
        QTableWidget::item:selected { background: #4455bb; color: #fff; }
        QTextEdit  { background: #16213e; border: none; color: #b4d0f8; }
        QLabel     { color: #cdd6f4; }
        QComboBox  {
            background: #3d3d5c; color: #cdd6f4;
            border: 1px solid #5555aa; border-radius: 3px; padding: 1px 5px;
        }
        QComboBox QAbstractItemView { background: #2a2a3e; color: #cdd6f4; }
        QStatusBar  { background: #2a2a3e; color: #888aaa; font-size: 11px; }
        QScrollBar:vertical   { background: #16213e; width: 10px; }
        QScrollBar::handle:vertical {
            background: #3d3d5c; border-radius: 4px; min-height: 20px;
        }
        QScrollBar:horizontal  { background: #16213e; height: 10px; }
        QScrollBar::handle:horizontal {
            background: #3d3d5c; border-radius: 4px;
        }
        QSplitter::handle { background: #2a2a3e; width: 4px; }
        QProgressDialog { background: #1e1e2e; color: #cdd6f4; }
        QCheckBox { color: #cdd6f4; spacing: 5px; }
        QCheckBox::indicator {
            width: 14px; height: 14px; border-radius: 3px;
            border: 1px solid #5555aa; background: #2a2a3e;
        }
        QCheckBox::indicator:checked  { background: #7799ff; border-color: #7799ff; }
        QCheckBox::indicator:hover    { border-color: #aabbff; }
        """)

    # ── rendering ─────────────────────────────────────────────

    def render(self, idx: int):
        idx = max(0, min(idx, self.n - 1))
        self._idx = idx
        gt    = self.gt_data[idx]
        img   = self.images[idx]
        boxes = gt.get("boxes_2d", [])

        # Annotated image
        if img is not None:
            ann = draw_boxes(img, gt, idx, self._hl_id)
            if self._enabled_paths:
                ann = draw_json_paths(ann, gt, self._enabled_paths, self._vis)
        else:
            ann = np.zeros((600, 960, 3), dtype=np.uint8)
            cv2.putText(ann, "No image data", (340, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (80, 80, 80), 2)
        self.img_view.set_bgr(ann)

        # Metadata
        self.v_frame.setText(f"{idx}  /  {self.n - 1}")
        self.v_ts.setText(   f"{gt.get('timestamp', 0):.6f}")
        self.v_speed.setText(f"{gt.get('ego_speed_kmh', 0):.2f} km/h")
        self.v_objs.setText( str(len(boxes)))

        # Detections table (block signals to avoid recursive render)
        self.table.blockSignals(True)
        self.table.setRowCount(len(boxes))
        for r, box in enumerate(boxes):
            name  = _TYPE_NAMES.get(box.get("type", 0), "Unknown")
            color = _ROW_COLORS.get(name, "#aaaaaa")
            cells = [
                box["id"][:8], name,
                f"{box['center_x']:.1f}", f"{box['center_y']:.1f}",
                f"{box['length']:.1f}",   f"{box['width']:.1f}",
                str(round(box["length"] * box["width"])),
            ]
            for c, txt in enumerate(cells):
                item = QTableWidgetItem(txt)
                item.setData(Qt.UserRole, box["id"])  # store full id
                if c == 1:
                    item.setForeground(QColor(color))
                self.table.setItem(r, c, item)

        # Re-select highlighted row after re-populating
        if self._hl_id:
            for r in range(self.table.rowCount()):
                it = self.table.item(r, 0)
                if it and it.data(Qt.UserRole) == self._hl_id:
                    self.table.selectRow(r)
                    break
        self.table.blockSignals(False)

        # Raw JSON
        self.json_edit.setPlainText(json.dumps(gt, indent=2))

        # Path data table
        self._update_path_table(gt)

        # Slider + label (no signal loop)
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl_frame.setText(f"Frame  {idx} / {self.n - 1}")

        self.statusBar().showMessage(
            f"Frame {idx}  •  {len(boxes)} objects  •  "
            f"{gt.get('ego_speed_kmh', 0):.1f} km/h  •  "
            f"ts = {gt.get('timestamp', 0):.3f}")

    # ── table interaction ─────────────────────────────────────

    @pyqtSlot()
    def _on_row_select(self):
        sel = self.table.selectedItems()
        self._hl_id = sel[0].data(Qt.UserRole) if sel else None
        gt  = self.gt_data[self._idx]
        img = self.images[self._idx]
        if img is not None:
            ann = draw_boxes(img, gt, self._idx, self._hl_id)
            if self._enabled_paths:
                ann = draw_json_paths(ann, gt, self._enabled_paths, self._vis)
            self.img_view.set_bgr(ann)

    # ── transport controls ────────────────────────────────────

    def _on_slider(self, val: int):
        self._stop()
        self.render(val)

    def _go_first(self):  self._stop(); self.render(0)
    def _go_last(self):   self._stop(); self.render(self.n - 1)
    def _step_back(self): self._stop(); self.render(self._idx - 1)
    def _step_fwd(self):  self._stop(); self.render(self._idx + 1)

    def _stop(self):
        self._timer.stop()
        self.b_play.setText("▶")

    def _toggle_play(self):
        if self._timer.isActive():
            self._stop()
        else:
            fps = int(self.fps_cb.currentText())
            self._timer.start(1000 // fps)
            self.b_play.setText("⏸")

    @pyqtSlot()
    def _on_tick(self):
        if self._idx < self.n - 1:
            self.render(self._idx + 1)
        else:
            self._stop()

    # ── keyboard shortcuts ────────────────────────────────────

    def keyPressEvent(self, e):
        k = e.key()
        if   k == Qt.Key_Left:                    self._step_back()
        elif k == Qt.Key_Right:                   self._step_fwd()
        elif k == Qt.Key_Space:                   self._toggle_play()
        elif k == Qt.Key_Home:                    self._go_first()
        elif k == Qt.Key_End:                     self._go_last()
        elif k in (Qt.Key_Plus, Qt.Key_Equal):    self._change_speed(+1)
        elif k == Qt.Key_Minus:                   self._change_speed(-1)
        else:
            super().keyPressEvent(e)

    def _change_speed(self, delta: int):
        i   = self._FPS_OPTS.index(self.fps_cb.currentText())
        new = max(0, min(i + delta, len(self._FPS_OPTS) - 1))
        self.fps_cb.setCurrentText(self._FPS_OPTS[new])
        if self._timer.isActive():
            self._timer.setInterval(1000 // int(self._FPS_OPTS[new]))

    # ── path type controls ──────────────────────────────────────

    def _on_path_toggle(self, key: str, checked: bool):
        """Handle a path-type checkbox toggle and re-render current frame."""
        if checked:
            self._enabled_paths.add(key)
        else:
            self._enabled_paths.discard(key)
        self.render(self._idx)

    def _update_path_table(self, gt: dict):
        """Populate the Path Data table for all enabled path types."""
        rows = []
        for key in _PATH_ORDER:
            if key not in self._enabled_paths:
                continue
            pd = gt.get(key)
            if pd is None:
                # Path type enabled but not present in this frame's JSON
                rows.append((_PATH_LABELS[key],
                             None, None, None,
                             None, None, None,
                             None, None, None, "—"))
                continue
            rows.append((
                _PATH_LABELS[key],
                pd.get("valid_center",       False),
                pd.get("valid_left",         False),
                pd.get("valid_right",        False),
                pd.get("confidence_center",  0.0),
                pd.get("confidence_left",    0.0),
                pd.get("confidence_right",   0.0),
                len(pd.get("center", [])),
                len(pd.get("left",   [])),
                len(pd.get("right",  [])),
                pd.get("source", "—"),
            ))

        self.path_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            path_nm, vc, vl, vr, cc, cl, cr, nc, nl, nr, src = row

            def _vstr(v):  return "—" if v is None else ("✓" if v else "✗")
            def _vcol(v):  return None if v is None else ("#44cc88" if v else "#cc4444")
            def _fstr(v):  return "—" if v is None else f"{v:.3f}"
            def _nstr(v):  return "—" if v is None else str(v)

            cells = [path_nm,
                     _vstr(vc), _vstr(vl), _vstr(vr),
                     _fstr(cc), _fstr(cl), _fstr(cr),
                     _nstr(nc), _nstr(nl), _nstr(nr),
                     src]
            vcols = [None,
                     _vcol(vc), _vcol(vl), _vcol(vr),
                     None, None, None, None, None, None, None]

            for c, (txt, col) in enumerate(zip(cells, vcols)):
                item = QTableWidgetItem(txt)
                item.setTextAlignment(Qt.AlignCenter)
                if col:
                    item.setForeground(QColor(col))
                self.path_table.setItem(r, c, item)


# ── entry point ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Waymo Debug Viewer — fully local desktop GUI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--tfrecord", default="",
                    help="Path to .tfrecord segment file")
    ap.add_argument("--json",     default="",
                    help="Path to GT .json file  (optional; auto-detected by same basename)")
    ap.add_argument("--max-frames", dest="max_frames", type=int, default=200,
                    help="Maximum frames to load (default: 200)")
    args = ap.parse_args()

    qapp = QApplication(sys.argv)
    qapp.setApplicationName("Waymo Debug Viewer")

    # ── Pick tfrecord via dialog if not given ─────────────────
    tfrecord = args.tfrecord
    if not tfrecord:
        tfrecord, _ = QFileDialog.getOpenFileName(
            None, "Open TFRecord Segment", ".",
            "TFRecord files (*.tfrecord);;All files (*)")
        if not tfrecord:
            print("No file selected — exiting.")
            sys.exit(0)

    # ── Auto-detect JSON with same basename ───────────────────
    json_path = args.json
    if not json_path:
        candidate = os.path.splitext(tfrecord)[0] + ".json"
        if os.path.exists(candidate):
            json_path = candidate
            print(f"[viewer] Auto-detected JSON: {json_path}")

    # ── Load tfrecord with progress dialog ────────────────────
    prog = QProgressDialog(
        "Extracting frames from tfrecord…", None, 0, args.max_frames)
    prog.setWindowTitle("Loading")
    prog.setMinimumDuration(0)
    prog.setWindowModality(Qt.ApplicationModal)
    prog.setValue(0)
    prog.show()
    qapp.processEvents()

    def progress_cb(done: int):
        prog.setValue(done)
        qapp.processEvents()

    print(f"[viewer] Loading: {tfrecord}")
    images, gt_list = load_tfrecord(tfrecord, args.max_frames, progress_cb)
    prog.close()

    # ── Override GT with external JSON if available ───────────
    if json_path and os.path.exists(json_path):
        print(f"[viewer] Loading JSON: {json_path}")
        with open(json_path) as f:
            gt_list = json.load(f)

    n = min(len(images), len(gt_list))
    print(f"[viewer] Ready: {n} frames")

    win = DebugWindow(images[:n], gt_list[:n])
    win.show()
    sys.exit(qapp.exec_())


if __name__ == "__main__":
    main()
