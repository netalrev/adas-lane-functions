"""
scripts/train.py
=================
Hydra entry point for training the MFTransformer.

Usage
-----
    # Standard training run:
    python scripts/train.py

    # Override any config value:
    python scripts/train.py training.training.max_epochs=100
    python scripts/train.py training.training.batch_size=32 training.model.d_model=128

    # Evaluate only (load best checkpoint and run test split):
    python scripts/train.py +eval_only=true

Config composition root: conf/train_config.yaml
    ├── conf/training/default.yaml  → cfg.training
    └── conf/features/mf.yaml       → cfg.features

Output artefacts
----------------
    outputs/checkpoints/best_model.pt   — best checkpoint by val CIPV F1
    outputs/reports/val_epoch_NNN.txt   — per-epoch val evaluation report
    outputs/reports/test.txt            — final test-set report
    outputs/reports/test_metrics.json   — machine-readable final metrics
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# Ensure the project root is on sys.path when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.classification.transformer import MFTransformer
from src.training.dataset  import MFDataset
from src.training.trainer  import Trainer
from src.evaluation.report import ReportWriter


@hydra.main(config_path="../conf", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train.py] Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    h5_path = str(cfg.training.data.dataset_path)
    if not Path(h5_path).exists():
        print(
            f"\n[ERROR] Dataset not found: {h5_path}\n"
            "  Run the pipeline first:  python pipeline_input.py\n"
            "  Then build the dataset:  python scripts/build_dataset.py\n"
        )
        sys.exit(1)

    num_workers = int(cfg.training.data.num_workers)
    batch_size  = int(cfg.training.training.batch_size)

    train_ds = MFDataset(h5_path, "train", cfg)
    val_ds   = MFDataset(h5_path, "val",   cfg)
    test_ds  = MFDataset(h5_path, "test",  cfg)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    # ── Log dataset class balance ─────────────────────────────────────────
    weights = train_ds.class_weights()
    print(
        f"[train.py] Train CIPV pos_weight (data): {weights['cipv_pos_weight']:.2f}  "
        f"CutIn pos_weight (data): {weights['cut_in_pos_weight']:.2f}"
    )
    print(f"[train.py] Lane class counts (train): {weights['lane_counts']}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = MFTransformer(cfg)
    print(f"[train.py] Model parameters: {model.count_parameters():,}")

    # ── Eval-only mode ────────────────────────────────────────────────────
    if cfg.get("eval_only", False):
        ckpt_path = Path(cfg.training.output.checkpoint_dir) / "best_model.pt"
        if not ckpt_path.exists():
            print(f"[ERROR] No checkpoint found at {ckpt_path}")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[train.py] Loaded checkpoint from epoch {ckpt['epoch']}")
        _run_test_evaluation(model, test_loader, cfg, device)
        return

    # ── Training ─────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, cfg, device)
    trainer.fit()

    # ── Final test evaluation ─────────────────────────────────────────────
    ckpt_path = Path(cfg.training.output.checkpoint_dir) / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"\n[train.py] Loaded best checkpoint (epoch {ckpt['epoch']}) for test evaluation.")

    test_metrics = _run_test_evaluation(model, test_loader, cfg, device)

    print("\n── Test Set Summary ──────────────────────────────────────────")
    for key in ("test_cipv_f1", "test_lane_macro_f1", "test_cut_in_f1", "test_loss"):
        val = test_metrics.get(key, test_metrics.get(key.replace("test_", ""), "N/A"))
        print(f"  {key:<25s}: {val}")
    print("─────────────────────────────────────────────────────────────\n")


def _run_test_evaluation(
    model:       torch.nn.Module,
    test_loader: DataLoader,
    cfg:         DictConfig,
    device:      torch.device,
) -> dict:
    """Load best checkpoint, evaluate on test set, write report."""
    trainer = Trainer.__new__(Trainer)
    # Minimal init for eval-only mode
    trainer._model        = model.to(device)
    trainer._device       = device
    trainer._cfg          = cfg
    trainer._report_writer = ReportWriter(cfg.training.output.report_dir)

    import torch.nn as nn
    cipv_pw   = torch.tensor([float(cfg.training.training.cipv_pos_weight)]).to(device)
    cut_in_pw = torch.tensor([float(cfg.training.training.cut_in_pos_weight)]).to(device)
    trainer._cipv_loss   = nn.BCEWithLogitsLoss(pos_weight=cipv_pw)
    trainer._lane_loss   = nn.CrossEntropyLoss(
        label_smoothing=float(cfg.training.training.lane_label_smoothing)
    )
    trainer._cut_in_loss  = nn.BCEWithLogitsLoss(pos_weight=cut_in_pw)
    trainer._w_cipv   = float(cfg.training.training.loss_weight_cipv)
    trainer._w_lane   = float(cfg.training.training.loss_weight_lane)
    trainer._w_cut_in = float(cfg.training.training.loss_weight_cut_in)

    return trainer._evaluate(test_loader, split="test", epoch=None, write_report=True)


if __name__ == "__main__":
    main()
