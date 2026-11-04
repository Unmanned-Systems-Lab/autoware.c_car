#
# Copyright (c) 2022, Takahiro Miki. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#
import logging
from typing import List

import cupy as cp
import cv2 as cv
import numpy as np

from .plugin_manager import PluginBase

_LOGGER = logging.getLogger(__name__)


class Inpainting(PluginBase):
    """
    This class is used for inpainting, a process of reconstructing lost or deteriorated parts of images and videos.

    Args:
        cell_n (int): The number of cells. Default is 100.
        method (str): The inpainting method. Options are 'telea' or 'ns' (Navier-Stokes). Default is 'telea'.
        **kwargs (): Additional keyword arguments.
    """

    def __init__(
        self,
        cell_n: int = 100,
        method: str = "telea",
        input_layer_name: str = "elevation",
        max_hole_area: int = 64,
        fill_border_holes: bool = False,
        inpaint_radius: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.input_layer_name = input_layer_name
        if method == "telea":
            self.method = cv.INPAINT_TELEA
        elif method == "ns":  # Navier-Stokes
            self.method = cv.INPAINT_NS
        else:  # default method
            self.method = cv.INPAINT_TELEA
        self.max_hole_area = None if int(max_hole_area) <= 0 else int(max_hole_area)
        self.fill_border_holes = bool(fill_border_holes)
        self.inpaint_radius = float(inpaint_radius)

    def _select_holes_to_fill(self, invalid_mask: np.ndarray) -> np.ndarray:
        """Return a uint8 mask of bounded invalid components worth filling."""
        if not invalid_mask.any():
            return invalid_mask

        if self.max_hole_area is None and self.fill_border_holes:
            return invalid_mask.copy()

        height, width = invalid_mask.shape
        selected = np.zeros_like(invalid_mask, dtype=np.uint8)
        label_count, labels, stats, _ = cv.connectedComponentsWithStats(invalid_mask, connectivity=4)
        for label in range(1, label_count):
            left, top, component_width, component_height, area = stats[label]
            if self.max_hole_area is not None and area > self.max_hole_area:
                continue
            if not self.fill_border_holes:
                touches_border = (
                    left == 0
                    or top == 0
                    or left + component_width == width
                    or top + component_height == height
                )
                if touches_border:
                    continue
            selected[labels == label] = 1
        return selected

    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        *args,
    ) -> cp.ndarray:
        """

        Args:
            elevation_map (cupy._core.core.ndarray):
            layer_names (List[str]):
            plugin_layers (cupy._core.core.ndarray):
            plugin_layer_names (List[str]):
            *args ():

        Returns:
            cupy._core.core.ndarray:
        """
        valid_layer = elevation_map[2]
        if self.input_layer_name in layer_names:
            elevation = elevation_map[layer_names.index(self.input_layer_name)]
        elif self.input_layer_name in plugin_layer_names:
            elevation = plugin_layers[plugin_layer_names.index(self.input_layer_name)]
        else:
            raise ValueError(f"Inpainting could not find layer '{self.input_layer_name}'")

        finite_elevation = cp.isfinite(elevation)
        valid_mask = cp.logical_and(valid_layer > 0.5, finite_elevation)
        output = cp.full(elevation.shape, cp.nan, dtype=cp.float32)
        output = cp.where(valid_mask, elevation, output)

        if not cp.any(valid_mask):
            return output.astype(cp.float64)

        invalid_mask_np = cp.asnumpy(cp.logical_not(valid_mask).astype(cp.uint8))
        if not invalid_mask_np.any():
            return elevation.astype(cp.float64)

        fill_mask_np = self._select_holes_to_fill(invalid_mask_np)
        if not fill_mask_np.any():
            return output.astype(cp.float64)

        h_valid = elevation[valid_mask]
        h_max = float(cp.asnumpy(h_valid.max()))
        h_min = float(cp.asnumpy(h_valid.min()))
        denom = h_max - h_min
        fill_mask = cp.asarray(fill_mask_np.astype(bool))

        if denom <= 1e-6:
            _LOGGER.warning(
                "Inpainting detected near-flat terrain (h_min=%.3f, h_max=%.3f); filling only bounded holes.",
                h_min,
                h_max,
            )
            output = cp.where(fill_mask, h_max, output)
            return output.astype(cp.float64)

        # Keep the full invalid mask when running OpenCV so large unknown regions do not
        # contribute placeholder values to nearby hole filling. Only bounded components are
        # copied back into the published layer.
        safe_elevation = cp.where(valid_mask, elevation, h_min)
        scaled = cp.asnumpy(cp.clip((safe_elevation - h_min) * 255.0 / denom, 0.0, 255.0)).astype("uint8")
        dst = cv.inpaint(scaled, invalid_mask_np, self.inpaint_radius, self.method)
        h_inpainted = cp.asarray(dst.astype(np.float32) * denom / 255.0 + h_min, dtype=cp.float32)
        output = cp.where(fill_mask, h_inpainted, output)
        return output.astype(cp.float64)
