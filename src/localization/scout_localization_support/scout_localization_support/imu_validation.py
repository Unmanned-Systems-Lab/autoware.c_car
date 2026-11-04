"""ROS-independent checks for a calibrated IMU QoS relay."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ImuValidationConfig:
    expected_frame: str = ""
    max_age_sec: float = 0.5
    max_future_sec: float = 0.1
    default_gyro_stddev: float = 0.03
    minimum_gyro_stddev: float = 0.001

    def validate(self):
        if not isinstance(self.expected_frame, str) or self.expected_frame.startswith("/"):
            raise ValueError("expected_frame must be a relative TF frame ID or empty")
        for name in ("max_age_sec", "default_gyro_stddev", "minimum_gyro_stddev"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            if "stddev" in name and not math.isfinite(value * value):
                raise ValueError(f"{name} variance is too large")
        if not math.isfinite(self.max_future_sec) or self.max_future_sec < 0:
            raise ValueError("max_future_sec must be finite and nonnegative")


def validate_imu_sample(
    *, stamp_ns, frame_id, orientation, angular_velocity, acceleration,
    orientation_covariance, angular_velocity_covariance, acceleration_covariance,
    now_ns, config, last_stamp_ns=None,
):
    """Return a safe gyro covariance; all other calibrated readings stay unchanged."""
    config.validate()
    if not frame_id or frame_id.startswith("/"):
        raise ValueError("IMU frame ID is missing or invalid")
    if config.expected_frame and frame_id != config.expected_frame:
        raise ValueError("unexpected IMU frame ID")
    if isinstance(stamp_ns, bool) or not isinstance(stamp_ns, int) or stamp_ns <= 0:
        raise ValueError("IMU timestamp must be positive integer nanoseconds")
    if not isinstance(now_ns, int) or now_ns <= 0:
        raise ValueError("ROS clock is not initialized")
    if now_ns - stamp_ns > int(config.max_age_sec * 1e9):
        raise ValueError("IMU timestamp is stale")
    if stamp_ns - now_ns > int(config.max_future_sec * 1e9):
        raise ValueError("IMU timestamp is in the future")
    if last_stamp_ns is not None and stamp_ns <= last_stamp_ns:
        raise ValueError("IMU timestamp did not increase")
    for name, values, length in (
        ("orientation", orientation, 4),
        ("angular velocity", angular_velocity, 3),
        ("acceleration", acceleration, 3),
        ("orientation covariance", orientation_covariance, 9),
        ("gyro covariance", angular_velocity_covariance, 9),
        ("acceleration covariance", acceleration_covariance, 9),
    ):
        if len(values) != length or not all(math.isfinite(v) for v in values):
            raise ValueError(f"IMU {name} must contain {length} finite values")
    # A missing orientation is valid for gyro-only localization. Never create a
    # fake attitude or overwrite its standard ROS covariance[0] == -1 marker.
    if orientation_covariance[0] != -1.0:
        norm = math.sqrt(sum(v * v for v in orientation))
        if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
            raise ValueError("provided IMU orientation must have unit norm")
    covariance = list(angular_velocity_covariance)
    if covariance[0] == -1.0:
        raise ValueError("IMU angular velocity is marked unavailable")
    source = "input"
    if all(v == 0.0 for v in covariance):
        covariance = [0.0] * 9
        for axis in range(3):
            covariance[axis * 3 + axis] = config.default_gyro_stddev ** 2
        source = "configured_zero_covariance"
    scale = max(max(abs(v) for v in covariance), 1e-12)
    tolerance = 1e-10 * scale
    for i in range(3):
        if covariance[i * 3 + i] < 0:
            raise ValueError("gyro covariance has negative variance")
        for j in range(i):
            if abs(covariance[i * 3 + j] - covariance[j * 3 + i]) > tolerance:
                raise ValueError("gyro covariance is not symmetric")
    lower = [[0.0] * 3 for _ in range(3)]
    diagonal = [0.0] * 3
    for k in range(3):
        pivot = covariance[k * 3 + k] - sum(lower[k][j] ** 2 * diagonal[j] for j in range(k))
        if not math.isfinite(pivot) or pivot < -tolerance:
            raise ValueError("gyro covariance is not positive semidefinite")
        diagonal[k] = pivot if pivot > tolerance else 0.0
        lower[k][k] = 1.0
        for i in range(k + 1, 3):
            residual = covariance[i * 3 + k] - sum(
                lower[i][j] * lower[k][j] * diagonal[j] for j in range(k)
            )
            if not math.isfinite(residual):
                raise ValueError("gyro covariance cannot be represented safely")
            if diagonal[k] == 0.0:
                if abs(residual) > tolerance:
                    raise ValueError("gyro covariance is not positive semidefinite")
            else:
                lower[i][k] = residual / diagonal[k]
    for i in range(3):
        covariance[i * 3 + i] = max(covariance[i * 3 + i], config.minimum_gyro_stddev ** 2)
        for j in range(i):
            average = 0.5 * (covariance[i * 3 + j] + covariance[j * 3 + i])
            covariance[i * 3 + j] = covariance[j * 3 + i] = average
    return tuple(covariance), source
