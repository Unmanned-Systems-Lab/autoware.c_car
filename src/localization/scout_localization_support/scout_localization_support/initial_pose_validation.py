"""Validation shared by the RViz initialization bridge and offline tests."""

import math


def validate_initial_pose(frame_id, position, quaternion, covariance, map_frame="map"):
    """Raise ValueError unless the pose is a finite map-frame estimate with PSD covariance."""
    if frame_id != map_frame:
        raise ValueError(f"initial pose must use {map_frame}, got {frame_id!r}")
    if len(position) != 3 or not all(math.isfinite(v) for v in position):
        raise ValueError("initial position must be finite")
    if len(quaternion) != 4 or not all(math.isfinite(v) for v in quaternion):
        raise ValueError("initial orientation must be finite")
    norm = math.sqrt(sum(v * v for v in quaternion))
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
        raise ValueError("initial orientation must have unit norm")
    if len(covariance) != 36 or not all(math.isfinite(v) for v in covariance):
        raise ValueError("initial pose covariance must contain 36 finite values")
    scale = max(max(abs(v) for v in covariance), 1e-12)
    tolerance = 1e-10 * scale
    for i in range(6):
        if covariance[i * 6 + i] < 0:
            raise ValueError("initial covariance has negative variance")
        for j in range(i):
            if abs(covariance[i * 6 + j] - covariance[j * 6 + i]) > tolerance:
                raise ValueError("initial covariance is not symmetric")
    # LDL^T test supports semidefinite covariance (RViz leaves z/roll/pitch
    # unobserved with zero entries) without requiring NumPy or a matrix inverse.
    lower = [[0.0] * 6 for _ in range(6)]
    diagonal = [0.0] * 6
    for k in range(6):
        pivot = covariance[k * 6 + k] - sum(lower[k][j] ** 2 * diagonal[j] for j in range(k))
        if not math.isfinite(pivot) or pivot < -tolerance:
            raise ValueError("initial covariance is not positive semidefinite")
        diagonal[k] = pivot if pivot > tolerance else 0.0
        lower[k][k] = 1.0
        for i in range(k + 1, 6):
            residual = covariance[i * 6 + k] - sum(
                lower[i][j] * lower[k][j] * diagonal[j] for j in range(k)
            )
            if not math.isfinite(residual):
                raise ValueError("initial covariance cannot be represented safely")
            if diagonal[k] == 0.0:
                if abs(residual) > tolerance:
                    raise ValueError("initial covariance is not positive semidefinite")
            else:
                lower[i][k] = residual / diagonal[k]
