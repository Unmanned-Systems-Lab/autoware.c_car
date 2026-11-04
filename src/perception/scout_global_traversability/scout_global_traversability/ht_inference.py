"""HT student inference on canonical world grids (rows +Y, columns +X).

Raw channel numbers are preserved; no physical heading calibration or vehicle
reference-point offset is implied. Finite initial terrain is evaluated normally.
Observation count and variance are separate quality data owned by the caller.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .vendor_ht.core import RECEPTIVE_FIELD, nearest_fill_nan, valid_window_ratio
from .vendor_ht.model import load_student_model


def _world_to_model(array: np.ndarray) -> np.ndarray:
    """World [+Y,+X] -> original GridMap/model [-X,-Y]."""
    return np.ascontiguousarray(np.flip(array.T, axis=(0, 1)))


def _model_to_world(array: np.ndarray) -> np.ndarray:
    """Invert the spatial transform, retaining any leading channel axes."""
    return np.ascontiguousarray(np.flip(np.swapaxes(array, -2, -1), axis=(-2, -1)))


class HTPredictor:
    receptive_field = RECEPTIVE_FIELD
    halo_cells = (RECEPTIVE_FIELD - 1) // 2

    def __init__(self, weights_path, device="cuda", min_valid_ratio=.70):
        self.weights_path = Path(weights_path).expanduser().resolve(strict=True)
        if not self.weights_path.is_file():
            raise ValueError("HT weights must be a regular checkpoint file")
        self.min_valid_ratio = float(min_valid_ratio)
        if not math.isfinite(self.min_valid_ratio) or not 0 <= self.min_valid_ratio <= 1:
            raise ValueError("min_valid_ratio must be finite and in [0, 1]")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for HT, but PyTorch CUDA is unavailable")
        self.weights_sha256 = hashlib.sha256(self.weights_path.read_bytes()).hexdigest()
        self.model = load_student_model(str(self.weights_path), self.device)
        self.model.eval()

    def predict(self, height_world_yx):
        """Return world-axis probabilities, validity, and original finite coverage.

        Probabilities are the network's raw softmax values, including positions
        marked invalid. Consumers must apply valid before displaying or using
        them. An entirely empty input avoids inference and returns NaN scores.
        NaN filling never turns a missing center or missing support into evidence.
        """
        world = np.asarray(height_world_yx, dtype=np.float32)
        if world.ndim != 2 or min(world.shape) < self.receptive_field:
            raise ValueError("HT input must be a 2-D height grid at least 51 x 51 cells")
        height = _world_to_model(world)
        finite = np.isfinite(height)
        support = valid_window_ratio(finite, self.receptive_field).astype(np.float32, copy=False)
        h = self.halo_cells
        valid = finite[h:-h, h:-h] & (support >= self.min_valid_ratio)
        if not finite.any():
            probabilities = np.full((8, *support.shape), np.nan, dtype=np.float32)
        else:
            filled, _ = nearest_fill_nan(height)
            tensor = torch.from_numpy(np.ascontiguousarray(filled, dtype=np.float32))
            tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
            with torch.inference_mode():
                logits = self.model(tensor)
                expected = (1, 8, 2, *support.shape)
                if tuple(logits.shape) != expected:
                    raise RuntimeError(f"Unexpected HT logits shape: {tuple(logits.shape)}; expected {expected}")
                probabilities = torch.softmax(logits, dim=2)[0, :, 1].cpu().numpy()
        return {
            "probabilities": _model_to_world(np.asarray(probabilities, dtype=np.float32)),
            "valid": _model_to_world(np.asarray(valid, dtype=bool)),
            "support_fraction": _model_to_world(support),
        }
