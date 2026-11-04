"""Pure checks for localization-bound elevation input, without ROS imports."""

from dataclasses import dataclass
import hashlib
import math

import numpy as np


@dataclass(frozen=True)
class GateConfig:
    map_frame: str = 'map'
    base_frame: str = 'base_link'
    cloud_frame: str = 'rslidar'
    max_age_sec: float = 0.5
    max_future_sec: float = 0.1
    minimum_map_points: int = 1000
    ego_min_x: float = -0.3135
    ego_max_x: float = 0.8115
    ego_min_y: float = -0.449795
    ego_max_y: float = 0.449795
    ego_min_z: float = -0.05
    ego_max_z: float = 1.0

    def validate(self):
        if any(not isinstance(x, str) or not x or x.startswith('/')
               for x in [self.map_frame, self.base_frame, self.cloud_frame]):
            raise ValueError('Frame names must be nonempty relative ROS frame IDs')
        if not 0 < self.max_age_sec <= 5 or not 0 <= self.max_future_sec <= 1:
            raise ValueError('Invalid freshness limits')
        bounds = self.bounds()
        if self.minimum_map_points < 1000 or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
            raise ValueError('Invalid map threshold or ego bounds')

    def bounds(self):
        return np.array([[self.ego_min_x, self.ego_min_y, self.ego_min_z],
                         [self.ego_max_x, self.ego_max_y, self.ego_max_z]], dtype=np.float64)

    def contract(self):
        return {'version': 1, 'input_frame': self.cloud_frame, 'output_frame': self.cloud_frame,
                'ego_bounds_base_link_m': self.bounds().tolist(), 'bounds_inclusive': True,
                'finite_xyz_only': True, 'header_and_record_bytes_preserved': True,
                'max_age_sec': self.max_age_sec, 'max_future_sec': self.max_future_sec,
                'requires_exact_cloud_stamp_tf': True}


def stamp_ns(sec, nanosec):
    if not isinstance(sec, int) or not isinstance(nanosec, int) or sec < 0 or not 0 <= nanosec < 1_000_000_000:
        raise ValueError('Invalid ROS timestamp fields')
    result = sec * 1_000_000_000 + nanosec
    if result <= 0:
        raise ValueError('A positive source timestamp is required')
    return result


def require_fresh(stamp, now_ns, received, now_monotonic, config):
    age = (now_ns - stamp) / 1e9
    receipt_age = now_monotonic - received
    if not (stamp > 0 and math.isfinite(age) and math.isfinite(receipt_age)
            and -config.max_future_sec <= age <= config.max_age_sec
            and 0 <= receipt_age <= config.max_age_sec):
        raise ValueError('Source timestamp or receipt is stale/future')


def rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('Invalid quaternion')
    norm = np.linalg.norm(q)
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
        raise ValueError('Quaternion is not normalized')
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def require_pose(frame, position, quaternion, covariance, config, child_frame=None):
    if frame != config.map_frame or (child_frame is not None and child_frame != config.base_frame):
        raise ValueError('Localization frame does not match the local map contract')
    p = np.asarray(position, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Non-finite localization position')
    rotation_matrix(quaternion)
    c = np.asarray(covariance, dtype=np.float64)
    if c.size != 36:
        raise ValueError('Pose covariance must contain 36 elements')
    c = c.reshape(6, 6)
    if (not np.isfinite(c).all() or not np.allclose(c, c.T, atol=1e-8, rtol=1e-6)
            or np.linalg.eigvalsh((c+c.T)*0.5).min() < -1e-8):
        raise ValueError('Invalid pose covariance')


def xyz_and_records(data, width, height, point_step, row_step, fields, is_bigendian):
    """Return float64 XYZ and exact raw records, stripping only row padding."""
    if is_bigendian:
        raise ValueError('Big-endian clouds are not supported')
    if width <= 0 or height <= 0 or point_step < 12 or row_step < width * point_step:
        raise ValueError('Invalid PointCloud2 dimensions')
    if len(data) < row_step * height:
        raise ValueError('Truncated PointCloud2 data')
    names = [f['name'] for f in fields]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate PointCloud2 field')
    lookup = {f['name']: f for f in fields}
    formats, offsets = [], []
    ranges = []
    for name in ['x', 'y', 'z']:
        f = lookup.get(name)
        if f is None or f['count'] != 1 or f['datatype'] not in (7, 8):
            raise ValueError('XYZ must be scalar FLOAT32 or FLOAT64 fields')
        size = 4 if f['datatype'] == 7 else 8
        offset = f['offset']
        if not isinstance(offset, int) or offset < 0 or offset + size > point_step:
            raise ValueError('XYZ field is outside the point record')
        ranges.append(set(range(offset, offset + size)))
        formats.append('<f4' if size == 4 else '<f8')
        offsets.append(offset)
    if any(ranges[i] & ranges[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('XYZ fields overlap')
    dtype = np.dtype({'names': ['x', 'y', 'z'], 'formats': formats, 'offsets': offsets, 'itemsize': point_step})
    points = np.ndarray((height, width), dtype=dtype, buffer=data,
                        strides=(row_step, point_step))
    xyz = np.column_stack([points[name].reshape(-1) for name in ['x', 'y', 'z']]).astype(np.float64)
    records = np.ndarray((height, width), dtype=f'V{point_step}', buffer=data,
                         strides=(row_step, point_step)).reshape(-1)
    return xyz, records


def loaded_map_digest(xyz, minimum_points=1000):
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_points or not np.isfinite(points).all():
        raise ValueError('Loaded map must contain enough finite measured geometry')
    canonical = np.asarray(points, dtype='<f4', order='C')
    if not np.isfinite(canonical).all():
        raise ValueError('Map geometry overflows FLOAT32')
    return hashlib.sha256(canonical.tobytes(order='C')).hexdigest()


def crop_ego(xyz, records, translation, quaternion, config):
    t = np.asarray(translation, dtype=np.float64)
    if t.shape != (3,) or not np.isfinite(t).all():
        raise ValueError('Invalid base-to-sensor translation')
    rotation = rotation_matrix(quaternion)
    finite = np.isfinite(xyz).all(axis=1)
    base = np.full_like(xyz, np.nan, dtype=np.float64)
    measured = xyz[finite]
    # A tall Nx3 BLAS multiply can start a large thread team on Jetson and starve
    # ROS TF delivery. Three vector expressions preserve the same rigid transform.
    for axis in range(3):
        base[finite, axis] = (measured[:, 0] * rotation[axis, 0]
                              + measured[:, 1] * rotation[axis, 1]
                              + measured[:, 2] * rotation[axis, 2] + t[axis])
    bounds = config.bounds()
    inside = np.all(base >= bounds[0] - 1e-6, axis=1) & np.all(base <= bounds[1] + 1e-6, axis=1)
    keep = finite & ~inside
    if not np.any(keep):
        raise ValueError('No measured points remain outside the ego vehicle')
    return records[keep].tobytes(), int(np.count_nonzero(keep)), {
        'dropped_nonfinite': int(np.count_nonzero(~finite)),
        'dropped_ego': int(np.count_nonzero(finite & inside)),
    }
