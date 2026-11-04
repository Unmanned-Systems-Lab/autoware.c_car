"""Validate and time-order native RoboSense PointXYZIRCAEDT without changing fields."""

from dataclasses import dataclass
import math
import numbers

import numpy as np


class CloudValidationError(ValueError):
    """The input cannot safely be used as a timed native scan."""


# sensor_msgs/PointField datatype IDs, as defined by the ROS message contract.
_FIELDS = {
    "x": (0, 7),
    "y": (4, 7),
    "z": (8, 7),
    "intensity": (12, 2),
    "return_type": (13, 2),
    "channel": (14, 4),
    "azimuth": (16, 7),
    "elevation": (20, 7),
    "distance": (24, 7),
    "time_stamp": (28, 6),
}
_POINT_STEP = 32
_INVALID_TIME = np.iinfo(np.uint32).max


@dataclass(frozen=True)
class OrderedCloud:
    data: bytes
    width: int
    height: int
    point_step: int
    row_step: int
    stamp_ns: int
    min_time_ns: int
    max_time_ns: int
    backwards_time_pairs: int
    dropped_points: int
    is_dense: bool


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise CloudValidationError(f"{label} must be an integer")
    return int(value)


def order_native_cloud(cloud, *, n_scan=32, max_scan_duration=0.2, remove_non_finite=True):
    """Return an unorganized, stably ordered scan with the exact original records.

    ``cloud`` follows the PointCloud2 attribute contract; importing ROS is not
    necessary. The header time stays the scan reference and ``time_stamp`` stays
    UINT32 nanoseconds relative to it. Row padding is excluded from output.
    By default nonfinite XYZ records are excluded from this private branch.
    Surviving records retain their complete bytes and relative acquisition time;
    the original header is never rebased to the first surviving point. Neither
    the input message nor its underlying buffer is modified.
    """
    n_scan = _integer(n_scan, "n_scan")
    if not 1 <= n_scan <= 65536:
        raise CloudValidationError("n_scan must be in [1, 65536]")
    if not isinstance(remove_non_finite, bool):
        raise CloudValidationError("remove_non_finite must be a boolean")
    if (isinstance(max_scan_duration, bool)
            or not isinstance(max_scan_duration, numbers.Real)
            or not math.isfinite(max_scan_duration)
            or max_scan_duration <= 0):
        raise CloudValidationError("max_scan_duration must be finite and positive")
    max_time_ns = int(round(max_scan_duration * 1_000_000_000))

    try:
        sec = _integer(cloud.header.stamp.sec, "header stamp seconds")
        nanosec = _integer(cloud.header.stamp.nanosec, "header stamp nanoseconds")
        frame = cloud.header.frame_id
    except AttributeError as error:
        raise CloudValidationError("missing cloud header or timestamp") from error
    if sec < 0 or not 0 <= nanosec < 1_000_000_000 or sec + nanosec == 0:
        raise CloudValidationError("invalid or unset header timestamp")
    if not isinstance(frame, str) or not frame or any(c.isspace() for c in frame):
        raise CloudValidationError("invalid or empty sensor frame")

    width = _integer(cloud.width, "width")
    height = _integer(cloud.height, "height")
    point_step = _integer(cloud.point_step, "point_step")
    row_step = _integer(cloud.row_step, "row_step")
    if width <= 0 or height <= 0:
        raise CloudValidationError("empty scan")
    if cloud.is_bigendian:
        raise CloudValidationError("big-endian PointXYZIRCAEDT is unsupported")
    if point_step != _POINT_STEP:
        raise CloudValidationError("PointXYZIRCAEDT point_step must be 32 bytes")
    if row_step < width * point_step:
        raise CloudValidationError("row_step is smaller than the point records")

    fields = list(cloud.fields)
    names = [field.name for field in fields]
    if len(names) != len(_FIELDS) or set(names) != set(_FIELDS):
        raise CloudValidationError("missing, duplicate, or unexpected native point fields")
    for field in fields:
        expected_offset, expected_type = _FIELDS[field.name]
        if (field.offset != expected_offset or field.datatype != expected_type
                or field.count != 1):
            raise CloudValidationError(
                f"invalid layout for {field.name}: expected offset "
                f"{expected_offset}, datatype {expected_type}, count 1")

    try:
        buffer = memoryview(cloud.data)
    except TypeError as error:
        raise CloudValidationError("point data is not a byte buffer") from error
    if not buffer.c_contiguous or buffer.itemsize != 1:
        raise CloudValidationError("point data must be a contiguous byte buffer")
    if buffer.nbytes != row_step * height:
        raise CloudValidationError("point data length does not match row_step * height")

    shape = (height, width)
    strides = (row_step, point_step)
    times = np.ndarray(shape, dtype="<u4", buffer=buffer, offset=28,
                       strides=strides).reshape(-1)
    channels = np.ndarray(shape, dtype="<u2", buffer=buffer, offset=14,
                          strides=strides)
    if np.any(times == _INVALID_TIME):
        raise CloudValidationError("invalid UINT32_MAX per-point timestamp")
    maximum = int(times.max())
    if maximum > max_time_ns:
        raise CloudValidationError(
            f"relative point time {maximum} ns exceeds {max_time_ns} ns scan limit")
    if np.any(channels >= n_scan):
        raise CloudValidationError(f"channel is outside [0, {n_scan - 1}]")

    records = np.ndarray(shape, dtype="V32", buffer=buffer,
                         strides=strides).reshape(-1)
    xyz = np.ndarray(shape + (3,), dtype="<f4", buffer=buffer,
                     strides=strides + (4,)).reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    if not np.any(finite):
        raise CloudValidationError("scan has no finite XYZ points")
    dropped = 0
    if remove_non_finite:
        dropped = int(finite.size - np.count_nonzero(finite))
        records = records[finite]
        times = times[finite]
    minimum, maximum = int(times.min()), int(times.max())
    backwards = int(np.count_nonzero(times[1:] < times[:-1]))
    if backwards:
        # Stable ordering preserves firing order when two points have equal time.
        records = records[np.argsort(times, kind="stable")]
    data = records.tobytes(order="C")
    count = len(records)
    return OrderedCloud(data=data, width=count, height=1, point_step=point_step,
                        row_step=count * point_step,
                        stamp_ns=sec * 1_000_000_000 + nanosec,
                        min_time_ns=minimum, max_time_ns=maximum,
                        backwards_time_pairs=backwards, dropped_points=dropped,
                        is_dense=True if remove_non_finite else bool(np.all(finite)))
