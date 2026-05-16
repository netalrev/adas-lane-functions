"""
scripts/export_model.py
=========================
CLI export pipeline: PyTorch checkpoint → FP32 ONNX → INT8 ONNX.

Steps
-----
  1. Load the best checkpoint from outputs/checkpoints/best_model.pt.
  2. Export to FP32 ONNX with dynamic batch axis.
  3. Verify ONNX outputs match PyTorch outputs within numerical tolerance.
  4. Quantise to INT8 using dynamic weight quantisation.
  5. (Optional) Validate INT8 accuracy against the test split; fail loudly
     if CIPV F1 drops more than 2 % vs FP32.

Usage
-----
    python scripts/export_model.py

    # Custom paths:
    python scripts/export_model.py \\
        --checkpoint outputs/checkpoints/best_model.pt \\
        --fp32_out   outputs/models/mf_transformer.onnx \\
        --int8_out   outputs/models/mf_transformer_int8.onnx

    # Skip accuracy validation (e.g. dataset not yet available):
    python scripts/export_model.py --no_validate

Output artefacts
----------------
    outputs/models/mf_transformer.onnx          FP32 ONNX model
    outputs/models/mf_transformer_int8.onnx     INT8 ONNX model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.inference.export_onnx import ModelExporter
from src.inference.quantize    import ModelQuantizer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export MFTransformer checkpoint to FP32 + INT8 ONNX"
    )
    p.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    p.add_argument("--fp32_out",   default="outputs/models/mf_transformer.onnx")
    p.add_argument("--int8_out",   default="outputs/models/mf_transformer_int8.onnx")
    p.add_argument("--dataset",    default="outputs/dataset.h5",
                   help="HDF5 dataset path used for INT8 accuracy validation.")
    p.add_argument("--train_cfg",  default="conf/training/default.yaml",
                   help="Training config needed for test-split reconstruction.")
    p.add_argument("--no_validate", action="store_true",
                   help="Skip INT8 accuracy validation step.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Step 1 + 2: Export FP32 ONNX ─────────────────────────────────────
    exporter  = ModelExporter(args.checkpoint)
    model, meta = exporter.load_model()
    fp32_path = exporter.export(args.fp32_out)

    print(
        f"[export_model] Checkpoint epoch={meta['epoch']}  "
        f"val_metrics={meta['val_metrics']}"
    )

    # ── Step 3: Verify ────────────────────────────────────────────────────
    exporter.verify(model, fp32_path)

    # ── Step 4: Quantise to INT8 ──────────────────────────────────────────
    quantizer = ModelQuantizer()
    int8_path = quantizer.quantize(fp32_path, args.int8_out)

    # ── Step 5: Accuracy validation ───────────────────────────────────────
    if not args.no_validate:
        cfg_path = Path(args.train_cfg)
        h5_path  = Path(args.dataset)

        if not cfg_path.exists():
            print(f"[export_model] Training config not found: {cfg_path} — skipping validation.")
        elif not h5_path.exists():
            print(f"[export_model] Dataset not found: {h5_path} — skipping validation.")
        else:
            training_cfg = OmegaConf.load(cfg_path)
            full_cfg     = OmegaConf.create({"training": training_cfg})
            quantizer.validate_accuracy(fp32_path, int8_path, h5_path, full_cfg)

    print(
        f"\n── Export complete ──────────────────────────────────────────\n"
        f"  FP32 ONNX : {fp32_path}\n"
        f"  INT8 ONNX : {int8_path}\n"
        f"────────────────────────────────────────────────────────────\n"
    )


if __name__ == "__main__":
    main()
