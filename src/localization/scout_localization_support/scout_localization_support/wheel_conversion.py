"""Pure, ROS-independent conversion of planar Scout wheel measurements.

Odometry twist is expressed at its child-frame origin.  Transforming just the
frame name is incorrect when base_link is at the rear axle.  This module moves
the measured planar velocity to the base origin and propagates its two measured
degrees of freedom before adding uncertainty for the four unobserved ones.
"""

from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


class InvalidWheelSample(ValueError):
    """The measurement cannot be used as a localization input."""


@dataclass(frozen=True)
class WheelSample:
    stamp_ns: int
    parent_frame: str
    child_frame: str
    linear: Tuple[float, float, float]
    angular: Tuple[float, float, float]
    covariance: Sequence[float]


@dataclass(frozen=True)
class MountTransform:
    """T_target_source: source origin and orientation expressed in target."""

    target_frame: str
    source_frame: str
    translation: Tuple[float, float, float]
    rotation_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ConversionConfig:
    source_odom_frame: str = "scout_wheel_odom"
    source_child_frame: str = "scout_chassis_link"
    output_frame: str = "base_link"
    max_age_sec: float = 0.5
    max_future_sec: float = 0.1
    default_linear_stddev: float = 0.2
    default_yaw_stddev: float = 0.1
    minimum_linear_stddev: float = 0.001
    minimum_yaw_stddev: float = 0.001
    unobserved_linear_stddev: float = 100.0
    unobserved_angular_stddev: float = 100.0
    planar_tolerance: float = 0.001
    invalid_covariance_policy: str = "reject"

    def validate(self):
        for name in ("source_odom_frame", "source_child_frame", "output_frame"):
            frame = getattr(self, name)
            if not isinstance(frame, str) or not frame or frame.startswith("/"):
                raise ValueError(f"{name} must be a nonempty relative TF frame ID")
        for name in (
            "max_age_sec", "default_linear_stddev", "default_yaw_stddev",
            "minimum_linear_stddev", "minimum_yaw_stddev",
            "unobserved_linear_stddev", "unobserved_angular_stddev",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            if "stddev" in name and not math.isfinite(value * value):
                raise ValueError(f"{name} variance is too large")
        for name in ("max_future_sec", "planar_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.invalid_covariance_policy not in ("reject", "fallback"):
            raise ValueError("invalid_covariance_policy must be reject or fallback")


@dataclass(frozen=True)
class ConvertedTwist:
    stamp_ns: int
    frame_id: str
    linear: Tuple[float, float, float]
    angular: Tuple[float, float, float]
    covariance: Tuple[float, ...]
    covariance_source: str


def _finite_vector(values, length, name):
    if len(values) != length or not all(math.isfinite(v) for v in values):
        raise InvalidWheelSample(f"{name} must contain {length} finite values")


def _rotation_matrix(quaternion):
    _finite_vector(quaternion, 4, "TF quaternion")
    norm = math.sqrt(sum(v * v for v in quaternion))
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
        raise InvalidWheelSample("TF quaternion must have unit norm")
    x, y, z, w = (v / norm for v in quaternion)
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def _measurement_covariance(raw, config):
    """Return covariance of [chassis vx, chassis wz], with a finite floor."""
    default = (
        config.default_linear_stddev ** 2, 0.0, config.default_yaw_stddev ** 2,
    )
    try:
        _finite_vector(raw, 36, "odometry covariance")
        vx, xy, yx, wz = raw[0], raw[5], raw[30], raw[35]
        if vx == xy == yx == wz == 0.0:
            xx, xw, ww = default
            source = "configured_zero_covariance"
        else:
            if vx < 0 or wz < 0:
                raise InvalidWheelSample("wheel covariance has negative variance")
            tolerance = 1e-12 + 1e-9 * max(abs(xy), abs(yx))
            if abs(xy - yx) > tolerance:
                raise InvalidWheelSample("wheel covariance is not symmetric")
            xw = 0.5 * (xy + yx)
            bound = math.sqrt(vx) * math.sqrt(wz)
            if abs(xw) > bound + 1e-12 * max(bound, 1e-12):
                raise InvalidWheelSample("wheel covariance is not positive semidefinite")
            # Clamp only floating-point roundoff at the PSD boundary.
            xw = math.copysign(min(abs(xw), bound), xw)
            xx, ww, source = vx, wz, "input"
    except InvalidWheelSample:
        if config.invalid_covariance_policy != "fallback":
            raise
        xx, xw, ww = default
        source = "configured_invalid_covariance"
    return (
        max(xx, config.minimum_linear_stddev ** 2),
        xw,
        max(ww, config.minimum_yaw_stddev ** 2),
        source,
    )


def convert_wheel_sample(
    sample: WheelSample,
    mount: MountTransform,
    config: ConversionConfig,
    now_ns: int,
    last_stamp_ns: Optional[int] = None,
) -> ConvertedTwist:
    """Convert a fresh, monotonic planar sample; never invent missing data.

    The two columns of J are [R[:,0], 0] and [r x R[:,2], R[:,2]].
    Thus J C J^T propagates only the measured vx/wz covariance.  Conservatively
    large variances in unobserved target axes are added *after* this operation;
    they are not treated as measurements and rotated into longitudinal speed.
    """
    config.validate()
    if sample.parent_frame != config.source_odom_frame:
        raise InvalidWheelSample("unexpected odometry parent frame")
    if sample.child_frame != config.source_child_frame:
        raise InvalidWheelSample("unexpected odometry child frame (twist frame)")
    if mount.target_frame != config.output_frame or mount.source_frame != sample.child_frame:
        raise InvalidWheelSample("TF frame IDs do not match the requested velocity transform")
    if isinstance(sample.stamp_ns, bool) or not isinstance(sample.stamp_ns, int) or sample.stamp_ns <= 0:
        raise InvalidWheelSample("wheel timestamp must be positive integer nanoseconds")
    if isinstance(now_ns, bool) or not isinstance(now_ns, int) or now_ns <= 0:
        raise InvalidWheelSample("ROS clock is not initialized")
    if now_ns - sample.stamp_ns > int(config.max_age_sec * 1e9):
        raise InvalidWheelSample("wheel timestamp is stale")
    if sample.stamp_ns - now_ns > int(config.max_future_sec * 1e9):
        raise InvalidWheelSample("wheel timestamp is in the future")
    if last_stamp_ns is not None and sample.stamp_ns <= last_stamp_ns:
        raise InvalidWheelSample("wheel timestamp did not increase")
    _finite_vector(sample.linear, 3, "wheel linear velocity")
    _finite_vector(sample.angular, 3, "wheel angular velocity")
    if any(abs(v) > config.planar_tolerance for v in (
        sample.linear[1], sample.linear[2], sample.angular[0], sample.angular[1],
    )):
        raise InvalidWheelSample("wheel twist is not the supported planar Scout model")
    _finite_vector(mount.translation, 3, "TF translation")
    rotation = _rotation_matrix(mount.rotation_xyzw)
    axis_x = tuple(row[0] for row in rotation)
    axis_z = tuple(row[2] for row in rotation)
    rx, ry, rz = mount.translation
    wx, wy, wz = axis_z
    lever = (ry * wz - rz * wy, rz * wx - rx * wz, rx * wy - ry * wx)
    col_v = axis_x + (0.0, 0.0, 0.0)
    col_w = lever + axis_z
    state = tuple(col_v[i] * sample.linear[0] + col_w[i] * sample.angular[2] for i in range(6))
    _finite_vector(state, 6, "transformed wheel velocity")
    xx, xw, ww, source = _measurement_covariance(sample.covariance, config)
    covariance = [
        xx * col_v[i] * col_v[j] + xw * (col_v[i] * col_w[j] + col_w[i] * col_v[j])
        + ww * col_w[i] * col_w[j]
        for i in range(6) for j in range(6)
    ]
    # These directions are orthogonal to the two measured source axes.  Rotate
    # them into base_link, but do not treat unknown angular rates as a measured
    # lever-arm velocity.  This remains conservative for a rotated mounting.
    unobserved = []
    for axis in (1, 2):
        unobserved.append((tuple(row[axis] for row in rotation) + (0.0, 0.0, 0.0),
                           config.unobserved_linear_stddev ** 2))
    for axis in (0, 1):
        unobserved.append(((0.0, 0.0, 0.0) + tuple(row[axis] for row in rotation),
                           config.unobserved_angular_stddev ** 2))
    for direction, variance in unobserved:
        for i in range(6):
            for j in range(6):
                covariance[i * 6 + j] += variance * direction[i] * direction[j]
    # Numerical protection also prevents reporting a perfectly observed output
    # axis for rotated mounts. Adding diagonal variance preserves PSD.
    for axis in range(6):
        covariance[axis * 6 + axis] += 1e-12
    _finite_vector(covariance, 36, "transformed wheel covariance")
    return ConvertedTwist(
        sample.stamp_ns, config.output_frame, state[:3], state[3:],
        tuple(covariance), source,
    )
