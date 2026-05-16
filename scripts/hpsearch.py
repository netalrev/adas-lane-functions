"""
scripts/hpsearch.py
====================
Optuna hyper-parameter search for the MFTransformer.

Samples trial configurations from conf/training/hpsearch.yaml, overrides
the default training config (conf/training/default.yaml), and maximises a
configurable validation metric (default: CIPV F1).

After all trials complete, the best configuration is printed and can be
manually transferred to conf/training/default.yaml for the final full-epoch
training run.

Usage
-----
    python scripts/hpsearch.py                          # uses hpsearch.yaml defaults
    python scripts/hpsearch.py n_trials=20              # override trial count
    python scripts/hpsearch.py metric=lane_macro_f1     # optimise for lane F1

Requires: optuna
    pip install optuna
"""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@hydra.main(config_path="../conf", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    try:
        import optuna
    except ImportError:
        print("[ERROR] optuna is not installed.  Run:  pip install optuna")
        sys.exit(1)

    import torch
    from torch.utils.data import DataLoader
    from src.models.classification.transformer import MFTransformer
    from src.training.dataset import MFDataset
    from src.training.trainer import Trainer

    # Load hpsearch config
    hps_path = Path("conf/training/hpsearch.yaml")
    if not hps_path.exists():
        print(f"[ERROR] HP search config not found: {hps_path}")
        sys.exit(1)
    hps_cfg = OmegaConf.load(hps_path)

    h5_path = str(cfg.training.data.dataset_path)
    if not Path(h5_path).exists():
        print(f"[ERROR] Dataset not found: {h5_path}. Build it first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[hpsearch.py] Device: {device}  n_trials: {hps_cfg.n_trials}")

    # Pre-load datasets once (shared across all trials for efficiency)
    train_ds = MFDataset(h5_path, "train", cfg)
    val_ds   = MFDataset(h5_path, "val",   cfg)

    def objective(trial: "optuna.Trial") -> float:
        # Sample a configuration from the search space
        trial_cfg = deepcopy(cfg)
        for key, choices in hps_cfg.search_space.items():
            value = trial.suggest_categorical(key, list(choices))
            # Navigate the config tree and set the value
            parts = key.split(".")
            node  = trial_cfg.training
            for part in parts[:-1]:
                node = getattr(node, part)
            OmegaConf.update(trial_cfg.training, key, value, merge=True)

        # Apply fixed overrides
        for key, value in hps_cfg.fixed.items():
            OmegaConf.update(trial_cfg.training, key, value, merge=True)

        batch_size = int(trial_cfg.training.training.batch_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False,
                                  num_workers=0)

        model   = MFTransformer(trial_cfg)
        trainer = Trainer(model, train_loader, val_loader, trial_cfg, device)
        history = trainer.fit()

        # Return the best val metric across all epochs in this trial
        metric_key = str(hps_cfg.metric)
        best = max(
            (row.get(metric_key, 0.0) for row in history), default=0.0
        )
        return float(best)

    direction = str(hps_cfg.get("direction", "maximize"))
    study     = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=int(hps_cfg.n_trials),
                   n_jobs=int(hps_cfg.get("n_jobs", 1)))

    print("\n── Best trial ───────────────────────────────────────────────")
    print(f"  Value ({hps_cfg.metric}): {study.best_trial.value:.4f}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")
    print("─────────────────────────────────────────────────────────────\n")
    print("Copy the above params into conf/training/default.yaml for the final run.")


if __name__ == "__main__":
    main()
