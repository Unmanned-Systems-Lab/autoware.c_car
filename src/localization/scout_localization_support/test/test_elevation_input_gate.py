"""Physical/representation checks for elevation input; no hardware or ROS graph required."""

import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import time
import unittest

import numpy as np

from scout_localization_support.elevation_input_contract import (
    GateConfig, crop_ego, loaded_map_digest, require_fresh, require_pose,
    rotation_matrix, stamp_ns, xyz_and_records,
)


FIELDS = [{'name': name, 'offset': offset, 'datatype': 7, 'count': 1}
          for name, offset in [('x', 0), ('y', 4), ('z', 8)]]


def cloud_bytes(points, width=None, padding=0):
    width = width or len(points)
    height = len(points) // width
    rows = []
    original = []
    for row in range(height):
        records = []
        for index in range(row * width, (row + 1) * width):
            record = bytearray([(index + 20) % 256] * 32)
            struct.pack_into('<fff', record, 0, *points[index])
            records.append(bytes(record))
        original += records
        rows.append(b''.join(records) + b'P' * padding)
    return b''.join(rows), original, width, height, width * 32 + padding


class ElevationInputContractTest(unittest.TestCase):
    def setUp(self):
        self.config = GateConfig()
        self.config.validate()

    def test_source_and_receipt_freshness_are_both_required(self):
        now = 10_000_000_000
        require_fresh(now - 500_000_000, now, 9.5, 10., self.config)
        require_fresh(now + 100_000_000, now, 10., 10., self.config)
        for stamp, received, monotonic in [
                (0, 10., 10.), (now-500_000_001, 10., 10.),
                (now+100_000_001, 10., 10.), (now, 9.499999, 10.),
                (now, 10.001, 10.), (now, math.nan, 10.)]:
            with self.subTest(stamp=stamp, received=received):
                with self.assertRaises(ValueError):
                    require_fresh(stamp, now, received, monotonic, self.config)

    def test_invalid_ros_timestamp_fields_do_not_create_latest_tf_requests(self):
        self.assertEqual(stamp_ns(3, 25), 3_000_000_025)
        for sec, nanosec in [(0, 0), (-1, 0), (1, -1), (1, 1_000_000_000), (1.5, 0)]:
            with self.assertRaises(ValueError):
                stamp_ns(sec, nanosec)

    def test_native_32_byte_records_and_row_padding_preserved(self):
        points = [[2., 3., 4.], [3., 4., 5.], [4., 5., 6.], [5., 6., 7.]]
        raw, original, width, height, row_step = cloud_bytes(points, width=2, padding=11)
        xyz, records = xyz_and_records(raw, width, height, 32, row_step, FIELDS, False)
        np.testing.assert_array_equal(xyz, points)
        self.assertEqual(records.tobytes(), b''.join(original))
        payload, count, dropped = crop_ego(xyz, records, [0, 0, 0], [0, 0, 0, 1], self.config)
        self.assertEqual(payload, b''.join(original))
        self.assertEqual(count, 4)
        self.assertEqual(dropped, {'dropped_nonfinite': 0, 'dropped_ego': 0})
        self.assertEqual(raw, b''.join([b''.join(original[:2])+b'P'*11,
                                      b''.join(original[2:])+b'P'*11]))

    def test_crop_uses_vehicle_coordinates_with_real_lidar_lever_arm(self):
        # First sensor point is inside the vehicle; second is outside after the lever arm.
        points = [[0., 0., 0.], [.6, 0., .3], [math.nan, 0., 0.], [2., 0., 0.]]
        raw, original, w, h, row = cloud_bytes(points)
        xyz, records = xyz_and_records(raw, w, h, 32, row, FIELDS, False)
        payload, count, dropped = crop_ego(xyz, records, [.4215, 0, .78499], [0, 0, 0, 1], self.config)
        self.assertEqual(count, 2)
        self.assertEqual(payload, original[1]+original[3])
        self.assertEqual(dropped, {'dropped_nonfinite': 1, 'dropped_ego': 1})

    def test_rotated_sensor_crop_is_computed_in_base_link(self):
        raw, original, w, h, row = cloud_bytes([[.8, 0., .5], [0., .7, .5]])
        xyz, records = xyz_and_records(raw, w, h, 32, row, FIELDS, False)
        q = [0, 0, math.sqrt(.5), math.sqrt(.5)]
        payload, count, _ = crop_ego(xyz, records, [0, 0, 0], q, self.config)
        # x=.8 is within ego x but becomes y=.8 outside. y=.7 becomes x=-.7 outside too.
        self.assertEqual(count, 2)
        self.assertEqual(payload, b''.join(original))

    def test_vector_crop_matches_rigid_transform_for_general_rotation(self):
        points = np.random.default_rng(7).uniform(-1., 1., size=(80, 3)).astype(np.float32)
        raw, original, w, h, row = cloud_bytes(points)
        xyz, records = xyz_and_records(raw, w, h, 32, row, FIELDS, False)
        quaternion = np.array([.2, -.3, .1, .8])
        quaternion /= np.linalg.norm(quaternion)
        translation = [.4215, 0., .78499]
        base = xyz @ rotation_matrix(quaternion).T + translation
        bounds = self.config.bounds()
        inside = np.all(base >= bounds[0]-1e-6, axis=1) & np.all(base <= bounds[1]+1e-6, axis=1)
        payload, count, _ = crop_ego(xyz, records, translation, quaternion, self.config)
        self.assertEqual(count, int(np.count_nonzero(~inside)))
        self.assertEqual(payload, b''.join(record for record, drop in zip(original, inside) if not drop))

    def test_inclusive_boundaries_mask_float32_vehicle_edges(self):
        bounds = self.config.bounds()
        points = [bounds[0].tolist(), bounds[1].tolist(), [2., 0., 0.], [math.inf, 1., 1.]]
        raw, original, w, h, row = cloud_bytes(points)
        xyz, records = xyz_and_records(raw, w, h, 32, row, FIELDS, False)
        payload, count, dropped = crop_ego(xyz, records, [0, 0, 0], [0, 0, 0, 1], self.config)
        self.assertEqual(payload, original[2])
        self.assertEqual(count, 1)
        self.assertEqual(dropped, {'dropped_nonfinite': 1, 'dropped_ego': 2})

    def test_all_ego_or_nonfinite_cloud_is_rejected(self):
        raw, _, w, h, row = cloud_bytes([[0, 0, 0], [math.nan, 0, 0]])
        xyz, records = xyz_and_records(raw, w, h, 32, row, FIELDS, False)
        with self.assertRaises(ValueError):
            crop_ego(xyz, records, [0, 0, 0], [0, 0, 0, 1], self.config)

    def test_xyz_offsets_and_float64_layout_are_read_by_fields(self):
        raw = bytearray(40)
        fields = [{'name': n, 'offset': o, 'datatype': 8, 'count': 1}
                  for n, o in [('x', 8), ('y', 16), ('z', 24)]]
        struct.pack_into('<ddd', raw, 8, 1.25, 2.5, 5.)
        xyz, records = xyz_and_records(raw, 1, 1, 40, 40, fields, False)
        np.testing.assert_array_equal(xyz, [[1.25, 2.5, 5.]])
        self.assertEqual(records.tobytes(), raw)

    def test_malformed_cloud_layout_is_rejected(self):
        raw, _, w, h, row = cloud_bytes([[2, 0, 0]])
        bad_fields = [dict(f) for f in FIELDS]
        bad_fields[1]['offset'] = 0
        variants = [(raw, w, h, 32, row, FIELDS, True),
                    (raw[:-1], w, h, 32, row, FIELDS, False),
                    (raw, w, h, 32, 1, FIELDS, False),
                    (raw, w, h, 32, row, bad_fields, False),
                    (raw, w, h, 32, row, FIELDS[:-1], False),
                    (raw, w, h, 32, row, FIELDS+[FIELDS[0]], False)]
        for args in variants:
            with self.assertRaises(ValueError):
                xyz_and_records(*args)

    def test_map_identity_is_ordered_float32_xyz_geometry(self):
        xyz = np.arange(3000, dtype=np.float32).reshape(1000, 3) / 9
        expected = hashlib.sha256(xyz.astype('<f4').tobytes()).hexdigest()
        self.assertEqual(loaded_map_digest(xyz), expected)
        self.assertEqual(loaded_map_digest(xyz.astype(np.float64)), expected)
        self.assertNotEqual(loaded_map_digest(xyz[::-1]), expected)
        changed = xyz.copy()
        changed[15, 1] += 1
        self.assertNotEqual(loaded_map_digest(changed), expected)

    def test_loaded_map_requires_real_finite_geometry(self):
        for points in [np.zeros((999, 3)), np.zeros((1000, 4)),
                       np.full((1000, 3), np.nan)]:
            with self.assertRaises(ValueError):
                loaded_map_digest(points)

    def test_localization_frames_quaternions_and_covariance_are_checked(self):
        covariance = np.eye(6).ravel()
        require_pose('map', [0, 0, 0], [0, 0, 0, 1], covariance, self.config, 'base_link')
        bad_cov = np.eye(6)
        bad_cov[0, 1] = 2
        negative_cov = np.eye(6)
        negative_cov[0, 0] = -.1
        for frame, p, q, cov, child in [
                ('odom', [0, 0, 0], [0, 0, 0, 1], covariance, 'base_link'),
                ('map', [0, 0, math.inf], [0, 0, 0, 1], covariance, 'base_link'),
                ('map', [0, 0, 0], [0, 0, 0, 2], covariance, 'base_link'),
                ('map', [0, 0, 0], [0, 0, 0, 1], bad_cov, 'base_link'),
                ('map', [0, 0, 0], [0, 0, 0, 1], negative_cov, 'base_link'),
                ('map', [0, 0, 0], [0, 0, 0, 1], covariance, 'rslidar')]:
            with self.assertRaises(ValueError):
                require_pose(frame, p, q, cov, self.config, child)

    def test_quaternion_sign_represents_same_physical_rotation(self):
        np.testing.assert_array_equal(rotation_matrix([0, 0, 0, 1]),
                                      rotation_matrix([0, 0, 0, -1]))


@unittest.skipUnless(os.environ.get('SCOUT_RUN_ISOLATED_GATE_ROS_TESTS') == '1',
                     'Set SCOUT_RUN_ISOLATED_GATE_ROS_TESTS=1 and an isolated ROS_DOMAIN_ID for ROS checks')
class ElevationInputGateRosTest(unittest.TestCase):
    """Direct callback checks on an isolated graph; no real sensor publisher is used."""

    def setUp(self):
        if not os.environ.get('ROS_DOMAIN_ID') or os.environ.get('ROS_DOMAIN_ID') == '0':
            self.fail('ROS gate tests require a nonzero isolated ROS_DOMAIN_ID')
        import rclpy
        from scout_localization_support.elevation_input_gate import ElevationInputGate
        self.rclpy = rclpy
        self.temporary = tempfile.TemporaryDirectory()
        self.reference = Path(self.temporary.name) / 'test.pcd'
        header = ('VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
                  'WIDTH 1000\nHEIGHT 1\nPOINTS 1000\nDATA binary\n').encode()
        self.reference.write_bytes(header+np.zeros((1000, 3), dtype='<f4').tobytes())
        rclpy.init(args=['--ros-args', '-p', f'reference_pcd:={self.reference}'])
        self.gate = ElevationInputGate()
        from rclpy.qos import ReliabilityPolicy
        self.assertEqual(self.gate.publisher.qos_profile.reliability, ReliabilityPolicy.BEST_EFFORT)

    def tearDown(self):
        if hasattr(self, 'gate'):
            self.gate.destroy_node()
        if hasattr(self, 'rclpy'):
            self.rclpy.try_shutdown()
        if hasattr(self, 'temporary'):
            self.temporary.cleanup()

    def make_cloud(self, points, frame, stamp=None):
        from array import array
        from sensor_msgs.msg import PointCloud2, PointField
        message = PointCloud2()
        raw, _, w, h, row = cloud_bytes(points)
        message.header.frame_id = frame
        message.header.stamp = stamp or self.gate.get_clock().now().to_msg()
        message.width, message.height = w, h
        message.point_step, message.row_step = 32, row
        message.fields = [PointField(**f) for f in FIELDS]
        message.data = array('B', raw)
        return message

    def approve_localization(self, stamp):
        from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from nav_msgs.msg import Odometry
        init = LocalizationInitializationState()
        init.state = 3
        self.gate.on_initialization(init)
        ndt, ekf = PoseWithCovarianceStamped(), Odometry()
        for message in [ndt, ekf]:
            message.header.frame_id = 'map'
            message.header.stamp = stamp
            message.pose.pose.orientation.w = 1.
            message.pose.covariance = np.eye(6).ravel().tolist()
        ekf.child_frame_id = 'base_link'
        self.gate.on_ndt(ndt)
        self.gate.on_odometry(ekf)

    def set_exact_tf(self, stamp):
        from geometry_msgs.msg import TransformStamped
        dynamic = TransformStamped()
        dynamic.header.stamp, dynamic.header.frame_id = stamp, 'map'
        dynamic.child_frame_id = 'base_link'
        dynamic.transform.rotation.w = 1.
        self.gate.tf_buffer.set_transform(dynamic, 'isolated_test')
        static = TransformStamped()
        static.header.frame_id, static.child_frame_id = 'base_link', 'rslidar'
        static.transform.rotation.w = 1.
        static.transform.translation.x, static.transform.translation.z = .4215, .78499
        self.gate.tf_buffer.set_transform_static(static, 'isolated_test')

    def test_gate_exact_tf_sensor_header_stale_drop_and_map_latch(self):
        from unittest.mock import Mock
        gate = self.gate
        gate.on_map(self.make_cloud([[float(i), 0., 0.] for i in range(1000)], 'map'))
        cloud = self.make_cloud([[0., 0., 0.], [2., 0., .1]], 'rslidar')
        self.approve_localization(cloud.header.stamp)
        gate.publisher = Mock()
        gate.status_publisher = Mock()
        # No latest fallback exists: current localization without exact TF cannot forward.
        gate.on_cloud(cloud)
        self.assertEqual(gate.forwarded_count, 0)
        self.assertEqual(gate.waiting_input_state, 'WAITING_TF')
        time.sleep(.005)
        self.set_exact_tf(cloud.header.stamp)
        gate.retry_pending_cloud()
        self.assertEqual(gate.forwarded_count, 1)
        output = gate.publisher.publish.call_args.args[0]
        self.assertEqual(output.header, cloud.header)
        self.assertEqual(output.width, 1)
        self.assertEqual(bytes(output.data), bytes(cloud.data[32:]))
        gate.publish_status()
        import json
        status = json.loads(gate.status_publisher.publish.call_args.args[0].data)
        self.assertEqual(status['state'], 'READY')
        self.assertEqual(status['map_identity']['coordinate_kind'], 'local_ndt_map')
        self.assertEqual(status['sensor_transform']['translation'], [.4215, 0., .78499])
        # Feed the actual gate JSON into the real pure controller, without launching a worker.
        from types import SimpleNamespace
        from scout_localization_support.local_elevation_manager import LocalElevationController
        stopped = []
        controller = LocalElevationController(
            self.reference, {'resolution': .04, 'ray_origin_frame': 'rslidar',
                             'self_filter_enabled': False,
                             'subscribers': {'lidar': {'topic_name': '/scout/elevation/points',
                                                       'data_type': 'pointcloud'}}},
            Path(self.temporary.name) / 'sessions',
            lambda *_: SimpleNamespace(poll=lambda: None), stopped.append)
        controller.on_gate(status, time.monotonic())
        self.assertEqual(controller.tick(time.monotonic(), gate.get_clock().now().nanoseconds / 1e9), 'STARTING')
        database, run_id = controller.session['database'], controller.run_id
        controller.on_processed(1, time.monotonic(), controller.generation)
        controller.on_observation(25, gate.get_clock().now().nanoseconds / 1e9,
                                  time.monotonic(), controller.generation)
        self.assertEqual(controller.tick(time.monotonic(), gate.get_clock().now().nanoseconds / 1e9), 'MAPPING')
        gate.initialized = False
        gate.publish_status()
        controller.on_gate(gate.status_publisher.publish.call_args.args[0].data, time.monotonic())
        self.assertEqual(controller.tick(time.monotonic(), gate.get_clock().now().nanoseconds / 1e9),
                         'WAITING_INITIALIZATION')
        self.assertEqual(len(stopped), 1)
        gate.initialized = True
        gate.publish_status()
        controller.on_gate(gate.status_publisher.publish.call_args.args[0].data, time.monotonic())
        self.assertEqual(controller.tick(time.monotonic(), gate.get_clock().now().nanoseconds / 1e9), 'STARTING')
        self.assertEqual((controller.session['database'], controller.run_id), (database, run_id))
        self.assertEqual(controller.generation, 2)
        # Repeat timestamp cannot be accepted, but does not revoke the still fresh last good input.
        gate.on_cloud(cloud)
        self.assertEqual(gate.forwarded_count, 1)
        gate.publish_status()
        self.assertEqual(json.loads(gate.status_publisher.publish.call_args.args[0].data)['state'], 'READY')
        # Monotonic age independently stops localization even if ROS stamps appear fresh.
        gate.localization['ekf'] = (gate.get_clock().now().nanoseconds, time.monotonic()-1.)
        state, _ = gate._prerequisites(gate.get_clock().now().nanoseconds, time.monotonic())
        self.assertEqual(state, 'WAITING_LOCALIZATION')
        replacement = [[float(i), 0., 0.] for i in range(1000)]
        replacement[0][0] = 3.
        gate.on_map(self.make_cloud(replacement, 'map'))
        gate.publish_status()
        self.assertEqual(json.loads(gate.status_publisher.publish.call_args.args[0].data)['state'], 'RESTART_REQUIRED')

    def test_delayed_exact_tf_waits_without_revoking_fresh_ready(self):
        from unittest.mock import Mock
        import json
        gate = self.gate
        gate.on_map(self.make_cloud([[float(i), 0., 0.] for i in range(1000)], 'map'))
        gate.publisher, gate.status_publisher = Mock(), Mock()
        first = self.make_cloud([[2., 0., .1]], 'rslidar')
        self.approve_localization(first.header.stamp)
        self.set_exact_tf(first.header.stamp)
        gate.on_cloud(first)
        self.assertEqual(gate.forwarded_count, 1)
        time.sleep(.005)
        delayed = self.make_cloud([[3., 0., .2]], 'rslidar')
        self.approve_localization(delayed.header.stamp)
        gate.on_cloud(delayed)  # TF covers only the previous stamp: future extrapolation.
        receipt = gate.pending_cloud[2]
        self.assertEqual(gate.forwarded_count, 1)
        self.assertIs(gate.pending_cloud[0], delayed)
        gate.publish_status()
        self.assertEqual(json.loads(gate.status_publisher.publish.call_args.args[0].data)['state'], 'READY')
        # Let the real retry timer run while the TF is late; it must neither block nor publish.
        end = time.monotonic() + .04
        while time.monotonic() < end:
            self.rclpy.spin_once(gate, timeout_sec=.005)
        self.assertEqual(gate.forwarded_count, 1)
        self.assertEqual(gate.pending_cloud[2], receipt)  # Receipt is never renewed by retries.
        self.set_exact_tf(delayed.header.stamp)
        end = time.monotonic() + .05
        while gate.pending_cloud is not None and time.monotonic() < end:
            self.rclpy.spin_once(gate, timeout_sec=.005)
        self.assertEqual(gate.forwarded_count, 2)
        self.assertIsNone(gate.pending_cloud)
        output = gate.publisher.publish.call_args.args[0]
        self.assertEqual(output.header, delayed.header)
        self.assertEqual(bytes(output.data), bytes(delayed.data))
        self.assertEqual(gate.last_forwarded_receipt, receipt)
        # Malformed new input also cannot erase a healthy last-good cloud.
        malformed = self.make_cloud([[3., 0., .2]], 'wrong_frame')
        gate.on_cloud(malformed)
        gate.publish_status()
        self.assertEqual(json.loads(gate.status_publisher.publish.call_args.args[0].data)['state'], 'READY')
        # READY cannot be extended by timer retries after its original receipt deadline.
        gate.last_forwarded_receipt = time.monotonic() - .501
        gate.publish_status()
        self.assertNotEqual(json.loads(gate.status_publisher.publish.call_args.args[0].data)['state'], 'READY')

    def test_pending_kept_until_expiry_and_initialization_stop(self):
        from unittest.mock import Mock
        gate = self.gate
        gate.on_map(self.make_cloud([[float(i), 0., 0.] for i in range(1000)], 'map'))
        gate.publisher = Mock()
        first = self.make_cloud([[2., 0., .1]], 'rslidar')
        self.approve_localization(first.header.stamp)
        gate.on_cloud(first)
        time.sleep(.005)
        newer = self.make_cloud([[3., 0., .2]], 'rslidar')
        gate.on_cloud(newer)
        self.assertIs(gate.pending_cloud[0], first)
        self.assertEqual(gate.pending_newer_dropped_count, 1)
        # Only a newer TF sample cannot replace the missing exact old-time coverage.
        self.set_exact_tf(newer.header.stamp)
        gate.retry_pending_cloud()
        self.assertEqual(gate.forwarded_count, 0)
        candidate = gate.pending_cloud
        gate.pending_cloud = (candidate[0], candidate[1], time.monotonic()-.501,
                              candidate[3], candidate[4])
        self.set_exact_tf(first.header.stamp)
        gate.retry_pending_cloud()
        self.assertIsNone(gate.pending_cloud)
        self.assertEqual(gate.pending_expired_count, 1)
        self.assertEqual(gate.forwarded_count, 0)
        time.sleep(.005)
        final = self.make_cloud([[4., 0., .3]], 'rslidar')
        gate.on_cloud(final)
        gate.initialized = False
        self.set_exact_tf(final.header.stamp)
        gate.retry_pending_cloud()
        self.assertIsNone(gate.pending_cloud)
        self.assertEqual(gate.forwarded_count, 0)

    def test_tf_later_than_scan_period_does_not_starve_pending(self):
        from unittest.mock import Mock
        gate = self.gate
        gate.on_map(self.make_cloud([[float(i), 0., 0.] for i in range(1000)], 'map'))
        gate.publisher = Mock()
        first = self.make_cloud([[2., 0., .1]], 'rslidar')
        self.approve_localization(first.header.stamp)
        gate.on_cloud(first)
        receipt = gate.pending_cloud[2]
        start, next_new = time.monotonic(), 0.04
        newer_count = 0
        # Several newer scans arrive during a >100ms TF delay. The oldest valid
        # candidate must retain its original deadline rather than being replaced.
        while time.monotonic()-start < .15:
            self.rclpy.spin_once(gate, timeout_sec=.005)
            if time.monotonic()-start >= next_new and newer_count < 3:
                gate.on_cloud(self.make_cloud([[3.+newer_count, 0., .2]], 'rslidar'))
                newer_count += 1
                next_new += .04
                self.assertIs(gate.pending_cloud[0], first)
                self.assertEqual(gate.pending_cloud[2], receipt)
        self.assertEqual(newer_count, 3)
        self.assertEqual(gate.pending_newer_dropped_count, 3)
        self.assertEqual(gate.forwarded_count, 0)
        self.set_exact_tf(first.header.stamp)
        gate.retry_pending_cloud()
        self.assertIsNone(gate.pending_cloud)
        self.assertEqual(gate.forwarded_count, 1)
        output = gate.publisher.publish.call_args.args[0]
        self.assertEqual(output.header, first.header)
        self.assertEqual(bytes(output.data), bytes(first.data))
        self.assertEqual(gate.last_forwarded_receipt, receipt)

    def test_reference_file_and_fixed_extrinsics_latch_restart(self):
        self.gate._check_sensor_transform({'translation': [.4215, 0., .78499], 'rotation': [0, 0, 0, 1]})
        # Quaternion sign is not a physical extrinsic change.
        self.gate._check_sensor_transform({'translation': [.4215, 0., .78499], 'rotation': [0, 0, 0, -1]})
        with self.assertRaises(ValueError):
            self.gate._check_sensor_transform({'translation': [.5, 0., .78499], 'rotation': [0, 0, 0, 1]})
        self.assertIsNotNone(self.gate.fault)
        self.gate.fault = None
        with self.reference.open('ab') as stream:
            stream.write(b'changed')
        self.gate._check_reference()
        self.assertIn('Reference PCD file changed', self.gate.fault)


if __name__ == '__main__':
    unittest.main()
