"""
src/inference/export_onnx.py
==============================
Export a trained MFTransformer checkpoint to an ONNX graph.

The exporter:
  1. Loads the PyTorch checkpoint produced by scripts/train.py.
  2. Reconstructs the MFTransformer from the configuration saved inside
     the checkpoint (no separate config file needed at export time).
  3. Exports to ONNX with a dynamic batch axis so the runtime can process
     any number of tracks in a single inference call.
  4. Verifies that the ONNX outputs match the PyTorch outputs within a
     configurable tolerance.

ONNX graph spec
---------------
  Input:
    name  : "mf_input"
    shape : [batch_size, T=10, D=18]   (batch_size is dynamic)
    dtype : float32

  Outputs:
    name  : "cipv_logit"     shape [batch_size, 1]   float32  (pre-sigmoid)
    name  : "lane_logit"     shape [batch_size, 5]   float32  (pre-softmax)
    name  : "cut_in_logit"   shape [batch_size, 1]   float32  (pre-sigmoid)

Opset 12 is used for maximum compatibility with ORT 1.x deployment targets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

from src.models.classification.transformer import MFTransformer

# Maximum absolute difference tolerated between PyTorch and ONNX outputs.
_VERIFY_ATOL = 1e-4
_ONNX_OPSET  = 12


class ModelExporter:
    """
    Exports a trained MFTransformer to ONNX.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the .pt checkpoint saved by Trainer._save_checkpoint().
    device : str
        Device used for model reconstruction.  Always "cpu" at export time.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self._ckpt_path = Path(checkpoint_path)
        self._device    = torch.device(device)

        if not self._ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {self._ckpt_path}\n"
                "  Run: python scripts/train.py"
            )

    def load_model(self) -> tuple[MFTransformer, dict]:
        """
        Reconstruct the MFTransformer from the checkpoint.

        Returns
        -------
        model : MFTransformer  — weights loaded, set to eval mode.
        meta  : dict           — epoch, val_metrics from checkpoint.
        """
        ckpt = torch.load(self._ckpt_path, map_location=self._device)

        # Checkpoint stores cfg as a plain dict; re-wrap for MFTransformer.
        training_cfg = OmegaConf.create(ckpt["cfg"])
        full_cfg     = OmegaConf.create({"training": training_cfg})

        model = MFTransformer(full_cfg)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(self._device)

        return model, {"epoch": ckpt.get("epoch"), "val_metrics": ckpt.get("val_metrics", {})}

    def export(self, output_path: str | Path) -> Path:
        """
        Export the model to ONNX and return the output path.

        Parameters
        ----------
        output_path : str | Path
            Destination .onnx file.  Parent directory is created if needed.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model, meta = self.load_model()
        T = int(next(iter(model.pos_enc.pe.shape[1:])))  # seq_len from PE buffer
        T = T - 1  # subtract CLS token position
        D = model.input_proj.in_features

        dummy = torch.zeros(1, T, D, dtype=torch.float32, device=self._device)

        print(
            f"[ModelExporter] Exporting  "
            f"(epoch={meta['epoch']}, T={T}, D={D}) → {output_path}"
        )

        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            opset_version=_ONNX_OPSET,
            input_names=["mf_input"],
            output_names=["cipv_logit", "lane_logit", "cut_in_logit"],
            dynamic_axes={
                "mf_input":      {0: "batch_size"},
                "cipv_logit":    {0: "batch_size"},
                "lane_logit":    {0: "batch_size"},
                "cut_in_logit":  {0: "batch_size"},
            },
            do_constant_folding=True,
        )
        print(f"[ModelExporter] ONNX saved → {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")
        return output_path

    def verify(self, model: MFTransformer, onnx_path: str | Path) -> dict:
        """
        Run the same random batch through PyTorch and ONNX Runtime and compare
        outputs.  Raises AssertionError if any output differs by more than atol.

        Returns
        -------
        dict with keys: max_cipv_delta, max_lane_delta, max_cut_in_delta  (floats)
        """
        T = int(model.pos_enc.pe.shape[1]) - 1
        D = model.input_proj.in_features
        x_np = np.random.randn(4, T, D).astype(np.float32)

        # PyTorch reference
        with torch.no_grad():
            cipv_pt, lane_pt, cut_in_pt = model(torch.from_numpy(x_np))
        cipv_pt    = cipv_pt.numpy()
        lane_pt    = lane_pt.numpy()
        cut_in_pt  = cut_in_pt.numpy()

        # ONNX Runtime
        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        cipv_ort, lane_ort, cut_in_ort = sess.run(
            ["cipv_logit", "lane_logit", "cut_in_logit"],
            {"mf_input": x_np},
        )

        deltas = {
            "max_cipv_delta":   float(np.abs(cipv_pt   - cipv_ort).max()),
            "max_lane_delta":   float(np.abs(lane_pt   - lane_ort).max()),
            "max_cut_in_delta": float(np.abs(cut_in_pt - cut_in_ort).max()),
        }

        for key, val in deltas.items():
            assert val < _VERIFY_ATOL, (
                f"[ModelExporter] Verification failed: {key}={val:.6f} > atol={_VERIFY_ATOL}"
            )

        print(
            f"[ModelExporter] Verification passed  "
            f"max_delta=cipv:{deltas['max_cipv_delta']:.2e}  "
            f"lane:{deltas['max_lane_delta']:.2e}  "
            f"cut_in:{deltas['max_cut_in_delta']:.2e}"
        )
        return deltas
