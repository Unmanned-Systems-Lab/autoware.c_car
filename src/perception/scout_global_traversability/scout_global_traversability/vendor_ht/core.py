"""ROS-independent HT preprocessing and output placement utilities."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


RECEPTIVE_FIELD = 51
DIRECTION_COUNT = 8


def validate_direction_order(order: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(int(value) for value in order)
    if len(result) != DIRECTION_COUNT or sorted(result) != list(range(DIRECTION_COUNT)):
        raise ValueError("direction_channel_order must be a permutation of 0..7")
    return result


def nearest_fill_nan(height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill unknown cells for inference and return the original finite mask."""

    array = np.asarray(height, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D elevation grid, got shape {array.shape}")
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Elevation grid contains no finite cells")
    if finite.all():
        return np.ascontiguousarray(array), finite

    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise RuntimeError("python3-scipy is required to fill unknown elevation cells") from exc

    nearest = distance_transform_edt(~finite, return_distances=False, return_indices=True)
    filled = array.copy()
    filled[~finite] = array[tuple(axis[~finite] for axis in nearest)]
    return np.ascontiguousarray(filled, dtype=np.float32), finite


def valid_window_ratio(valid_mask: np.ndarray, window: int = RECEPTIVE_FIELD) -> np.ndarray:
    """Fraction of observed input cells in every valid convolution window."""

    mask = np.asarray(valid_mask, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be 2-D")
    rows, cols = mask.shape
    if rows < window or cols < window:
        raise ValueError(f"Input {rows}x{cols} is smaller than {window}x{window}")

    integral = np.pad(mask, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    counts = (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    )
    return counts / float(window * window)


def embed_center(
    values: np.ndarray, output_shape: tuple[int, int], fill: float = np.nan
) -> np.ndarray:
    """Place a smaller 2-D layer at the exact center of the output grid."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("Only 2-D layers can be embedded")
    rows, cols = output_shape
    if source.shape[0] > rows or source.shape[1] > cols:
        raise ValueError("Source layer is larger than output grid")
    top = (rows - source.shape[0]) // 2
    left = (cols - source.shape[1]) // 2
    result = np.full((rows, cols), fill, dtype=np.float32)
    result[top : top + source.shape[0], left : left + source.shape[1]] = source
    return result


def build_output_layers(
    probabilities: np.ndarray,
    valid_output: np.ndarray,
    full_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Create eight HT probability layers plus best-direction diagnostics."""

    probs = np.asarray(probabilities, dtype=np.float32)
    valid = np.asarray(valid_output, dtype=bool)
    if probs.ndim != 3 or probs.shape[0] != DIRECTION_COUNT:
        raise ValueError(f"Expected probabilities [8,H,W], got {probs.shape}")
    if probs.shape[1:] != valid.shape:
        raise ValueError("Probability and validity shapes do not match")

    masked = np.where(valid[None, :, :], probs, np.nan).astype(np.float32)
    best_dir = np.argmax(probs, axis=0).astype(np.float32)
    best_prob = np.max(probs, axis=0).astype(np.float32)
    best_dir[~valid] = np.nan
    best_prob[~valid] = np.nan

    layers = {
        f"ht_dir_{index}": embed_center(masked[index], full_shape)
        for index in range(DIRECTION_COUNT)
    }
    layers["ht_best_dir"] = embed_center(best_dir, full_shape)
    layers["ht_best_prob"] = embed_center(best_prob, full_shape)
    layers["ht_valid"] = embed_center(valid.astype(np.float32), full_shape, fill=0.0)
    return layers


def build_direction_cost_layers(
    probabilities: np.ndarray,
    valid_output: np.ndarray,
    full_shape: tuple[int, int],
    threshold: float,
) -> dict[str, np.ndarray]:
    """Create eight binary directional cost layers using a probability threshold.

    A valid cell is 0 when its directional traversability probability is at
    least ``threshold`` and 1 otherwise. Unsupported cells remain NaN.
    """

    probs = np.asarray(probabilities, dtype=np.float32)
    valid = np.asarray(valid_output, dtype=bool)
    if probs.ndim != 3 or probs.shape[0] != DIRECTION_COUNT:
        raise ValueError(f"Expected probabilities [8,H,W], got {probs.shape}")
    if probs.shape[1:] != valid.shape:
        raise ValueError("Probability and validity shapes do not match")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("traversability threshold must be between 0 and 1")

    costs = np.where(
        valid[None, :, :], (probs < threshold).astype(np.float32), np.nan
    ).astype(np.float32)
    return {
        f"ht_cost_dir_{index}": embed_center(costs[index], full_shape)
        for index in range(DIRECTION_COUNT)
    }


def build_conservative_traversability_layer(
    probabilities: np.ndarray,
    valid_output: np.ndarray,
    full_shape: tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Return one binary layer that requires every direction to be traversable.

    A valid output cell is 1 only when all eight directional traversability
    probabilities are greater than or equal to ``threshold``.  If any one
    direction is below the threshold the cell is 0.  Unsupported cells remain
    NaN so consumers cannot mistake unknown terrain for a valid decision.
    """

    probs = np.asarray(probabilities, dtype=np.float32)
    valid = np.asarray(valid_output, dtype=bool)
    if probs.ndim != 3 or probs.shape[0] != DIRECTION_COUNT:
        raise ValueError(f"Expected probabilities [8,H,W], got {probs.shape}")
    if probs.shape[1:] != valid.shape:
        raise ValueError("Probability and validity shapes do not match")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("traversability threshold must be between 0 and 1")

    traversable = np.all(probs >= threshold, axis=0).astype(np.float32)
    traversable[~valid] = np.nan
    return embed_center(traversable, full_shape)


def bounded_median_height(filled, finite, kernel=5, max_delta=.04, max_relief=.12):
    """Suppress small local elevation jitter while retaining steps and unknown masks."""
    if kernel == 1:
        return filled
    if kernel not in (3, 5) or max_delta <= 0 or max_relief <= 0:
        raise ValueError("Invalid height denoising parameters")
    import cv2
    cv2.setNumThreads(1)
    source = np.ascontiguousarray(filled, dtype=np.float32)
    source = source.copy()
    for _ in range(3):
        median = cv2.medianBlur(source, kernel)
        positive = finite & ((source-median) > .15)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(positive.astype(np.uint8), connectivity=8)
        small = stats[:, cv2.CC_STAT_AREA] <= 8
        small[0] = False
        isolated_positive = positive & small[labels]
        if not isolated_positive.any():
            break
        source[isolated_positive] = median[isolated_positive]
        if not finite.all():
            # Refill gaps from cleaned observations, not the former positive spike.
            source[~finite] = np.nan
            source, _ = nearest_fill_nan(source)
    median = cv2.medianBlur(source, kernel)
    element = np.ones((kernel, kernel), np.uint8)
    relief = cv2.dilate(source, element) - cv2.erode(source, element)
    change = (np.abs(source-median) <= max_delta) & (relief <= max_relief)
    result = source.copy()
    result[change] = median[change]
    return result
