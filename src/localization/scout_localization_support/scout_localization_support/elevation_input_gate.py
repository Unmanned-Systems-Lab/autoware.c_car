"""Feed measured LiDAR to a local-map elevation session only while localization is sound."""

from array import array
import copy
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from .elevation_input_contract import (
    GateConfig, crop_ego, loaded_map_digest, require_fresh, require_pose,
    rotation_matrix, stamp_ns, xyz_and_records,
)
from .map_checks import require_ndt_map


class ElevationInputGate(Node):
    def __init__(self):
        super().__init__('elevation_input_gate')
        defaults = GateConfig()
        values = {f.name: self.declare_parameter(f.name, getattr(defaults, f.name)).value
                  for f in fields(defaults)}
        self.config = GateConfig(**values)
        self.config.validate()
        reference = self.declare_parameter('reference_pcd', '').value
        self.reference = Path(require_ndt_map(reference, self.config.minimum_map_points)['path'])
        before = self._file_signature()
        digest = hashlib.sha256()
        with self.reference.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        self.reference_signature = self._file_signature()
        if before != self.reference_signature:
            raise ValueError('Reference PCD changed while its identity was being read')
        self.reference_sha256 = digest.hexdigest()
        self.map_sha256 = None
        self.map_point_count = 0
        self.initialized = False
        self.localization = {}
        self.localization_last_stamp = {}
        self.sensor_transform = None
        self.last_cloud_stamp = 0
        self.last_forwarded_stamp = 0
        self.last_forwarded_receipt = None
        self.pending_cloud = None
        self.pending_replaced_count = 0
        self.pending_newer_dropped_count = 0
        self.pending_expired_count = 0
        self.received_count = self.forwarded_count = self.rejected_count = 0
        self.dropped_nonfinite = self.dropped_ego = 0
        self.sequence = 0
        self.fault = None
        self.last_error = ''
        self.waiting_input_state = 'WAITING_INPUT'
        self.last_log = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        reliable = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        source_topic = self.declare_parameter('input_topic', '/scout/localization/points_deskewed').value
        output_topic = self.declare_parameter('output_topic', '/scout/elevation/points').value
        status_topic = self.declare_parameter('status_topic', '/scout/elevation/input_status').value
        realtime_cloud = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.publisher = self.create_publisher(PointCloud2, output_topic, realtime_cloud)
        self.status_publisher = self.create_publisher(String, status_topic, reliable)
        self.create_subscription(PointCloud2, source_topic, self.on_cloud, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, '/map/pointcloud_map', self.on_map, latched)
        self.create_subscription(LocalizationInitializationState, '/localization/initialization_state',
                                 self.on_initialization, latched)
        self.create_subscription(PoseWithCovarianceStamped,
                                 '/localization/pose_estimator/pose_with_covariance',
                                 self.on_ndt, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/localization/kinematic_state',
                                 self.on_odometry, qos_profile_sensor_data)
        self.create_timer(0.2, self.publish_status)
        # Retry exact-time TF without blocking the executor that receives TF updates.
        self.create_timer(0.01, self.retry_pending_cloud)
        self.get_logger().info(f'Local elevation input is bound to {self.reference}; waiting for live map/localization')

    def _file_signature(self):
        info = self.reference.stat()
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    @staticmethod
    def _xyz(message):
        return xyz_and_records(
            message.data, message.width, message.height, message.point_step,
            message.row_step, [{'name': f.name, 'offset': f.offset,
                                'datatype': f.datatype, 'count': f.count} for f in message.fields],
            message.is_bigendian)

    @staticmethod
    def _stamp(message):
        return stamp_ns(message.header.stamp.sec, message.header.stamp.nanosec)

    @staticmethod
    def _transform_values(message):
        t, q = message.transform.translation, message.transform.rotation
        translation, quaternion = [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]
        if not np.isfinite(translation).all():
            raise ValueError('Non-finite TF translation')
        rotation_matrix(quaternion)
        # q and -q encode the same rotation; publish one stable representation.
        if quaternion[3] < 0:
            quaternion = [-x for x in quaternion]
        return {'translation': translation, 'rotation': quaternion}

    def _latch_fault(self, reason):
        if self.fault is None:
            self.fault = str(reason)
            self.get_logger().error(f'Elevation input requires restart: {self.fault}')
        self.last_error = self.fault

    def on_map(self, message):
        try:
            if message.header.frame_id != self.config.map_frame:
                raise ValueError('Loaded reference map is not in map frame')
            xyz, _ = self._xyz(message)
            digest = loaded_map_digest(xyz, self.config.minimum_map_points)
            if self.map_sha256 is not None and digest != self.map_sha256:
                self._latch_fault('Loaded NDT map geometry changed; start a new elevation session')
                return
            self.map_sha256, self.map_point_count = digest, len(xyz)
        except (ValueError, TypeError, KeyError) as error:
            # An invalid replacement cannot leave an old map approved in memory.
            self._latch_fault(f'Invalid loaded NDT reference map: {error}')

    def on_initialization(self, message):
        # ADAPI publishes this latched state only on changes; there is no age timeout.
        self.initialized = int(message.state) == 3

    def _on_pose(self, key, message, child_frame=None):
        try:
            stamp = self._stamp(message)
            now = self.get_clock().now().nanoseconds
            receipt = time.monotonic()
            require_fresh(stamp, now, receipt, receipt, self.config)
            if stamp <= self.localization_last_stamp.get(key, 0):
                raise ValueError(f'{key} source timestamp did not advance')
            pose = message.pose.pose
            p, q = pose.position, pose.orientation
            require_pose(message.header.frame_id, [p.x, p.y, p.z],
                         [q.x, q.y, q.z, q.w], message.pose.covariance,
                         self.config, child_frame)
            self.localization[key] = (stamp, receipt)
            self.localization_last_stamp[key] = stamp
        except (ValueError, TypeError) as error:
            self.localization.pop(key, None)
            self.last_error = str(error)

    def on_ndt(self, message):
        self._on_pose('ndt', message)

    def on_odometry(self, message):
        self._on_pose('ekf', message, message.child_frame_id)

    def _prerequisites(self, now_ns, now_mono):
        if self.fault:
            return 'RESTART_REQUIRED', self.fault
        if not self.map_sha256:
            return 'WAITING_MAP', 'Waiting for the real latched NDT map'
        if not self.initialized:
            return 'WAITING_INITIALIZATION', 'Universe localization is not initialized'
        for key in ['ndt', 'ekf']:
            if key not in self.localization:
                return 'WAITING_LOCALIZATION', f'Waiting for valid {key} localization'
            try:
                require_fresh(self.localization[key][0], now_ns,
                              self.localization[key][1], now_mono, self.config)
            except ValueError:
                return 'WAITING_LOCALIZATION', f'{key} localization is stale/future'
        return None, ''

    def _check_reference(self):
        try:
            if self._file_signature() != self.reference_signature:
                self._latch_fault('Reference PCD file changed while running')
        except OSError as error:
            self._latch_fault(f'Reference PCD is no longer accessible: {error}')

    def _check_sensor_transform(self, transform):
        if self.sensor_transform is not None:
            previous = self.sensor_transform
            if (not np.allclose(previous['translation'], transform['translation'], atol=1e-6, rtol=0)
                    or not np.allclose(rotation_matrix(previous['rotation']),
                                       rotation_matrix(transform['rotation']), atol=1e-6, rtol=0)):
                self._latch_fault('Fixed base-to-LiDAR TF changed while running')
                raise ValueError(self.fault)
        else:
            self.sensor_transform = transform

    def _reject_input(self, error, waiting_state='WAITING_INPUT'):
        self.rejected_count += 1
        self.last_error = str(error)
        self.waiting_input_state = waiting_state
        now_mono = time.monotonic()
        # An isolated bad/newer cloud must not revoke an older, still fresh good cloud.
        # publish_status independently enforces localization and both freshness clocks.
        if now_mono - self.last_log > 5:
            self.get_logger().warning(f'Gated elevation input: {error}')
            self.last_log = now_mono

    def _drop_pending(self, error, waiting_state='WAITING_INPUT'):
        if self.pending_cloud is not None:
            self.pending_cloud = None
            self._reject_input(error, waiting_state)

    def on_cloud(self, message):
        self.received_count += 1
        received = time.monotonic()
        try:
            self._check_reference()
            now_ns = self.get_clock().now().nanoseconds
            state, reason = self._prerequisites(now_ns, received)
            if state:
                self._drop_pending(reason)
                raise ValueError(reason)
            stamp = self._stamp(message)
            require_fresh(stamp, now_ns, received, received, self.config)
            if stamp <= self.last_cloud_stamp:
                raise ValueError('LiDAR source timestamp did not advance')
            if message.header.frame_id != self.config.cloud_frame:
                raise ValueError('LiDAR source frame changed or does not match the configured sensor')
            self.last_cloud_stamp = stamp
            if self.pending_cloud is not None:
                # Give an earlier candidate its full bounded TF wait. Replacing it every
                # 100ms can starve forever when TF arrives just beyond the scan period.
                self.retry_pending_cloud()
                if self.pending_cloud is not None:
                    self.pending_newer_dropped_count += 1
                    self._reject_input('Newer LiDAR cloud dropped while awaiting earlier exact TF', 'WAITING_TF')
                    return
            xyz, records = self._xyz(message)
            # Keep one earliest candidate, with its ORIGINAL receipt and source time.
            self.pending_cloud = (message, stamp, received, xyz, records)
            self.retry_pending_cloud()
        except (ValueError, TypeError, KeyError) as error:
            self._reject_input(error)

    def retry_pending_cloud(self):
        if self.pending_cloud is None:
            return
        message, stamp, received, xyz, records = self.pending_cloud
        now_mono = time.monotonic()
        try:
            self._check_reference()
            now_ns = self.get_clock().now().nanoseconds
            state, reason = self._prerequisites(now_ns, now_mono)
            if state:
                self._drop_pending(reason)
                return
            try:
                require_fresh(stamp, now_ns, received, now_mono, self.config)
            except ValueError:
                self.pending_expired_count += 1
                self._drop_pending('Pending cloud expired before exact timestamp TF became available',
                                   'WAITING_TF')
                return
            exact = Time.from_msg(message.header.stamp)
            # Exact timestamp lookup is intentional: never replace it with Time() / latest.
            map_transform = self.tf_buffer.lookup_transform(
                self.config.map_frame, message.header.frame_id, exact)
            self._transform_values(map_transform)
            base_transform = self.tf_buffer.lookup_transform(
                self.config.base_frame, message.header.frame_id, exact)
            sensor_transform = self._transform_values(base_transform)
            self._check_sensor_transform(sensor_transform)
            payload, count, dropped = crop_ego(
                xyz, records, sensor_transform['translation'], sensor_transform['rotation'], self.config)
            output = PointCloud2()
            output.header = copy.deepcopy(message.header)
            output.height, output.width = 1, count
            output.fields = copy.deepcopy(message.fields)
            output.is_bigendian = False
            output.point_step = message.point_step
            output.row_step = count * message.point_step
            output.is_dense = True
            output.data = array('B', payload)
            # Never extend the 0.5s source/receipt deadline while waiting or filtering.
            require_fresh(stamp, self.get_clock().now().nanoseconds, received,
                          time.monotonic(), self.config)
            self.publisher.publish(output)
            self.pending_cloud = None
            self.forwarded_count += 1
            self.dropped_nonfinite += dropped['dropped_nonfinite']
            self.dropped_ego += dropped['dropped_ego']
            self.last_forwarded_stamp, self.last_forwarded_receipt = stamp, received
            self.last_error, self.waiting_input_state = '', 'WAITING_INPUT'
        except TransformException as error:
            # A normal TF delivery delay keeps this candidate until the next timer tick.
            # No Time(0)/latest fallback and no refreshed receipt timestamp are allowed.
            self.last_error = str(error)
            self.waiting_input_state = 'WAITING_TF'
        except (ValueError, TypeError, KeyError) as error:
            self._drop_pending(error)

    def publish_status(self):
        self._check_reference()
        now_mono, now_ns = time.monotonic(), self.get_clock().now().nanoseconds
        state, reason = self._prerequisites(now_ns, now_mono)
        if state is None:
            try:
                if self.last_forwarded_receipt is None:
                    raise ValueError('Waiting for a valid measured LiDAR cloud with timestamped TF')
                require_fresh(self.last_forwarded_stamp, now_ns, self.last_forwarded_receipt,
                              now_mono, self.config)
                state = 'READY'
            except ValueError as error:
                state, reason = self.waiting_input_state, str(error)
        self.sequence += 1
        status = {
            'schema_version': 1, 'sequence': self.sequence, 'state': state,
            'readiness_reason': 'Measured LiDAR, local-map identity, exact TF and Universe localization are valid'
                                if state == 'READY' else (reason or self.last_error),
            'ros_stamp_sec': now_ns / 1e9,
            'map_identity': {
                'coordinate_kind': 'local_ndt_map', 'backend': 'universe',
                'frame': self.config.map_frame, 'base_frame': self.config.base_frame,
                'reference_pcd': str(self.reference), 'reference_pcd_sha256': self.reference_sha256,
                'loaded_map_geometry_sha256': self.map_sha256,
            },
            'sensor_transform': self.sensor_transform,
            'cloud_filter_contract': self.config.contract(),
            'loaded_map_point_count': self.map_point_count,
            'received_pointcloud_count': self.received_count,
            'forwarded_pointcloud_count': self.forwarded_count,
            'rejected_pointcloud_count': self.rejected_count,
            'pending_pointcloud_count': int(self.pending_cloud is not None),
            'pending_replaced_pointcloud_count': self.pending_replaced_count,
            'pending_newer_dropped_pointcloud_count': self.pending_newer_dropped_count,
            'pending_expired_pointcloud_count': self.pending_expired_count,
            'dropped_nonfinite_point_count': self.dropped_nonfinite,
            'dropped_ego_point_count': self.dropped_ego,
            'last_error': self.last_error or reason,
        }
        self.status_publisher.publish(String(data=json.dumps(status, allow_nan=False, separators=(',', ':'))))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ElevationInputGate()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
