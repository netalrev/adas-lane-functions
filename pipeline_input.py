from dotenv import load_dotenv
load_dotenv()  # load .env into os.environ before Hydra resolves ${oc.env:...}

# comet_ml is imported lazily in main() to avoid a ~30s startup penalty
# on slow filesystems (e.g. WSL2 /mnt/c/) when no API key is configured.

# Suppress TF info/warning/error logs
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from omegaconf import DictConfig, OmegaConf
import hydra

from src.pipeline import build_engines, resolve_segments, SegmentRunner
from src.utils.comet_logger import NullExperiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    import time as _time
    print("--- Starting ADAS/AV Batch Input Pipeline ---")
    print(OmegaConf.to_yaml(cfg))

    _api_key = cfg.comet.api_key  # empty string when COMET_API_KEY is not set
    if _api_key:
        from comet_ml import Experiment  # lazy import — only when API key is present
        experiment = Experiment(
            api_key      = _api_key,
            project_name = cfg.comet.project_name,
            workspace    = cfg.comet.workspace,
        )
        experiment.set_name(cfg.comet.experiment_name)
        print("[comet] Logging to Comet ML experiment.")
    else:
        experiment = NullExperiment()
        print("[comet] COMET_API_KEY not set — running offline (no remote logging).")

    # ── Resolve segment list ─────────────────────────────────────────────────
    segments = resolve_segments(cfg)
    print(f"\n[batch] Will process {len(segments)} segment(s):")
    for i, p in enumerate(segments):
        print(f"  [{i:03d}] {os.path.basename(p)}")

    # ── Build all inference engines once (expensive ONNX loads) ─────────────────
    engines = build_engines(cfg)

    _viz_cfg = getattr(cfg, "visualization", None)
    _ep_list = list(getattr(_viz_cfg, "enabled_paths",
                            ["kinematic", "drivable_path", "host_lane"])) \
               if _viz_cfg else ["kinematic", "drivable_path", "host_lane"]
    enabled_paths: set = set(_ep_list)

    runner = SegmentRunner(cfg, experiment, engines, enabled_paths)

    # ── Segment loop ─────────────────────────────────────────────────────────
    total_t0     = _time.time()
    global_step  = 0
    total_frames = 0

    for seg_idx, tfrecord_path in enumerate(segments):
        frames_done, _ = runner.run(tfrecord_path, seg_idx, global_step)
        global_step  += cfg.dataset.max_frames   # reserve step space per segment
        total_frames += frames_done

    elapsed = _time.time() - total_t0
    print(f"\n[batch] All done — {len(segments)} segment(s), "
          f"{total_frames} frames in {elapsed:.1f}s "
          f"({total_frames / max(elapsed, 1):.1f} fps)")
    experiment.log_metric("total_frames",     total_frames)
    experiment.log_metric("total_duration_s", elapsed)
    experiment.end()
    print("Check your Comet ML dashboard!")


if __name__ == "__main__":
    main()
