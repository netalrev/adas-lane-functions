"""
scripts/download_models.py
===========================
One-time setup script: downloads and exports all ONNX model weights
required by the pipeline that are not bundled with the repository.

Models managed by this script
------------------------------
yolov8n.onnx — YOLOv8 nano (COCO pre-trained), used by TargetDetector.
               Placed in src/data/models/.

The following models are already present in src/data/models/ and do NOT
need to be downloaded:
    culane_r18.onnx — CLRNet host-lane detector.
    yolopv2.onnx    — YOLOPv2 drivable-area segmentor.

Prerequisites
-------------
    pip install ultralytics     # for YOLOv8 download + ONNX export

Usage
-----
    python scripts/download_models.py
    python scripts/download_models.py --output-dir src/data/models
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_OUTPUT_DIR = Path("src/data/models")
_YOLOV8N_FILENAME   = "yolov8n.onnx"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_yolov8n_onnx(output_dir: Path) -> Path:
    """
    Use the ultralytics package to download yolov8n.pt weights and export
    them to ONNX format (opset 17, input size 640×640).

    The ultralytics export returns the path of the created .onnx file.
    We then copy it to `output_dir / yolov8n.onnx`.

    Parameters
    ----------
    output_dir : Path
        Destination directory for the exported ONNX file.

    Returns
    -------
    Path
        Absolute path to the placed ONNX file.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        print(
            "\n[download_models] ERROR: 'ultralytics' is not installed.\n"
            "Install it with:\n"
            "    pip install ultralytics\n"
            "Then re-run this script.\n"
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / _YOLOV8N_FILENAME

    print("[download_models] Downloading yolov8n.pt weights...")
    model = YOLO("yolov8n.pt")  # downloads to ~/.ultralytics cache on first run

    print("[download_models] Exporting to ONNX (opset 12, imgsz=640)...")
    # opset 12 is the highest opset supported by torch 1.12.x.
    # export() returns a Path to the created file
    exported: Path = Path(str(model.export(format="onnx", imgsz=640, opset=12)))

    shutil.copy(str(exported), str(dest))
    print(f"[download_models] yolov8n.onnx saved → {dest.resolve()}")
    return dest


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify_onnx(model_path: Path) -> None:
    """Run a single dummy inference to confirm the ONNX graph is valid."""
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np

        sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        inp_name = sess.get_inputs()[0].name
        sess.run(None, {inp_name: dummy})
        out_shape = sess.get_outputs()[0].shape
        print(f"[download_models] Verification OK — output shape: {out_shape}")
    except Exception as exc:
        print(f"[download_models] WARNING: verification failed — {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and export ONNX model weights for adas-lane-functions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory to place model files. Default: {_DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip ONNX runtime verification after download.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    dest = output_dir / _YOLOV8N_FILENAME

    if dest.exists():
        print(f"[download_models] {dest} already exists — skipping download.")
    else:
        dest = _download_yolov8n_onnx(output_dir)

    if not args.skip_verify:
        _verify_onnx(dest)

    print("\n[download_models] All models ready.")
    print(f"  {dest.resolve()}")


if __name__ == "__main__":
    main()
