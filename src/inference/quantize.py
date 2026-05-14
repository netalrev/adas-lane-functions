"""
src/inference/quantize.py
===========================
Post-Training Quantisation (PTQ) INT8 for the exported ONNX graph.

Strategy
--------
Dynamic quantisation via onnxruntime.quantization.quantize_dynamic.
  - Weights are quantised to INT8 at export time (static).
  - Activations are quantised dynamically at runtime (per-tensor).
  - No calibration data required — suitable for Transformer / Linear-dominated
    models where weight quantisation alone recovers most of the speed gain.

Accuracy gate
-------------
After quantisation, both the FP32 and INT8 graphs are run on a random sample
of the test set.  If the CIPV F1 drop exceeds MAX_CIPV_F1_DROP (2 %) the
function raises RuntimeError so the export pipeline fails loudly rather than
silently shipping a degraded model.

Usage (via scripts/export_model.py)
------------------------------------
    quantizer = ModelQuantizer()
    int8_path = quantizer.quantize(fp32_onnx_path, int8_onnx_path)
    metrics   = quantizer.validate_accuracy(fp32_onnx_path, int8_onnx_path,
                                            h5_path, cfg)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import onnxruntime as ort
from omegaconf import DictConfig

from src.evaluation.metrics import binary_metrics

# Maximum allowed CIPV F1 degradation between FP32 and INT8 (absolute).
MAX_CIPV_F1_DROP = 0.02
# Number of test samples used for accuracy validation (0 = all samples).
VALIDATION_SAMPLES = 2000


class ModelQuantizer:
    """
    Wraps onnxruntime.quantization.quantize_dynamic and provides an accuracy
    validation step comparing FP32 and INT8 ONNX graphs.
    """

    def quantize(
        self,
        fp32_path: str | Path,
        int8_path: str | Path,
    ) -> Path:
        """
        Produce an INT8 ONNX model from a FP32 ONNX model.

        Parameters
        ----------
        fp32_path : str | Path
            FP32 ONNX input produced by ModelExporter.
        int8_path : str | Path
            Destination for the INT8 ONNX model.

        Returns
        -------
        Path
            Resolved path to the INT8 model.
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError as exc:
            raise ImportError(
                "onnxruntime.quantization not found. "
                "Install: pip install onnxruntime"
            ) from exc

        fp32_path = Path(fp32_path)
        int8_path = Path(int8_path)
        int8_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[ModelQuantizer] Quantising {fp32_path.name} → INT8 ...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )

        fp32_kb = fp32_path.stat().st_size / 1024
        int8_kb = int8_path.stat().st_size / 1024
        ratio   = fp32_kb / max(int8_kb, 1.0)
        print(
            f"[ModelQuantizer] Done  "
            f"FP32={fp32_kb:.1f} KB → INT8={int8_kb:.1f} KB  "
            f"(compression ratio ≈ {ratio:.1f}×)"
        )
        return int8_path

    def validate_accuracy(
        self,
        fp32_path: str | Path,
        int8_path: str | Path,
        h5_path:   str | Path,
        cfg:       DictConfig,
    ) -> dict:
        """
        Run both models on a sample of the test set and compare CIPV F1.

        Parameters
        ----------
        fp32_path, int8_path : str | Path
            Paths to the FP32 and INT8 ONNX graphs.
        h5_path : str | Path
            Path to the HDF5 dataset file.
        cfg : DictConfig
            Hydra training config (cfg.training.*) used to reconstruct the
            test split.

        Returns
        -------
        dict with keys:
            fp32_cipv_f1, int8_cipv_f1, cipv_f1_drop, passed (bool)

        Raises
        ------
        RuntimeError
            If the CIPV F1 drop exceeds MAX_CIPV_F1_DROP.
        """
        if not Path(h5_path).exists():
            print(
                f"[ModelQuantizer] Dataset not found at {h5_path} — "
                "skipping accuracy validation."
            )
            return {"passed": True, "skipped": True}

        # Load test split (lazy: only sample a subset for speed)
        mf_arr, cipv_arr = self._load_test_samples(h5_path, cfg)

        fp32_cipv_probs = self._run_cipv_inference(str(fp32_path), mf_arr)
        int8_cipv_probs = self._run_cipv_inference(str(int8_path), mf_arr)

        fp32_m   = binary_metrics(cipv_arr, fp32_cipv_probs, label="cipv")
        int8_m   = binary_metrics(cipv_arr, int8_cipv_probs, label="cipv")
        f1_drop  = fp32_m["cipv_f1"] - int8_m["cipv_f1"]

        result = {
            "fp32_cipv_f1": fp32_m["cipv_f1"],
            "int8_cipv_f1": int8_m["cipv_f1"],
            "cipv_f1_drop": round(f1_drop, 4),
            "passed":       f1_drop <= MAX_CIPV_F1_DROP,
            "n_samples":    len(cipv_arr),
        }

        status = "PASSED" if result["passed"] else "FAILED"
        print(
            f"[ModelQuantizer] Accuracy validation {status}  "
            f"FP32_F1={result['fp32_cipv_f1']:.4f}  "
            f"INT8_F1={result['int8_cipv_f1']:.4f}  "
            f"drop={f1_drop:.4f}  "
            f"(limit={MAX_CIPV_F1_DROP})"
        )

        if not result["passed"]:
            raise RuntimeError(
                f"INT8 quantisation accuracy gate failed: "
                f"CIPV F1 dropped by {f1_drop:.4f} > {MAX_CIPV_F1_DROP}. "
                "Try a higher precision quantisation scheme."
            )
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_test_samples(
        h5_path: str | Path,
        cfg:     DictConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load MF windows and CIPV labels for the test split.
        Re-implements the split logic from MFDataset to avoid a Dataset dependency.
        """
        with h5py.File(str(h5_path), "r") as f:
            raw_names = f["segment_names"][:]
            all_names = np.array([
                s.decode("utf-8") if isinstance(s, bytes) else s
                for s in raw_names
            ])
            seg_ids = f["segment_ids"][:]

        unique_segs = sorted(set(all_names.tolist()))
        n_segs      = len(unique_segs)
        rng         = np.random.default_rng(int(cfg.training.data.seed))
        perm        = rng.permutation(n_segs)
        n_train     = int(n_segs * float(cfg.training.data.train_fraction))
        n_val       = int(n_segs * float(cfg.training.data.val_fraction))
        test_names  = {unique_segs[i] for i in perm[n_train + n_val:].tolist()}

        sample_seg_names = all_names[seg_ids]
        mask    = np.isin(sample_seg_names, list(test_names))
        indices = np.where(mask)[0]

        if VALIDATION_SAMPLES > 0 and len(indices) > VALIDATION_SAMPLES:
            rng2    = np.random.default_rng(0)
            indices = rng2.choice(indices, size=VALIDATION_SAMPLES, replace=False)

        with h5py.File(str(h5_path), "r") as f:
            mf_arr   = f["mf_sequences"][indices].astype(np.float32)
            cipv_arr = f["cipv_labels"][indices].astype(np.int32).ravel()

        return mf_arr, cipv_arr

    @staticmethod
    def _run_cipv_inference(onnx_path: str, mf_arr: np.ndarray) -> np.ndarray:
        """Run ONNX inference in batches and return CIPV sigmoid probabilities."""
        sess      = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        batch_sz  = 128
        probs_out = []
        for start in range(0, len(mf_arr), batch_sz):
            batch = mf_arr[start: start + batch_sz]
            cipv_logit = sess.run(["cipv_logit"], {"mf_input": batch})[0]
            probs_out.append(1.0 / (1.0 + np.exp(-cipv_logit.ravel())))
        return np.concatenate(probs_out)
