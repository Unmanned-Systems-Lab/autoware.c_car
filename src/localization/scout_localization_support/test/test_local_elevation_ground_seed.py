"""Bounded initial ground is a frozen unobserved prior, never a fabricated scan."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

from scout_localization_support.elevation_input_contract import loaded_map_digest
from scout_localization_support.initial_ground import estimate_initial_ground
from scout_localization_support.local_elevation_manager import LocalElevationController, main as manager_main
from scout_localization_support.local_elevation_session import (
    hash_file, local_identity, require_local_parameters,
)


class Process:
    returncode = None
    def poll(self):
        return self.returncode


class GroundSeedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.reference = self.root / 'map.pcd'
        self.reference.write_bytes(b'PCD source identity fixture')
        self.reference_sha256, _ = hash_file(self.reference)
        x, y = np.meshgrid(np.linspace(-3, 3, 91), np.linspace(-3, 3, 91))
        self.xyz = np.column_stack([x.ravel(), y.ravel(), (-0.4 + 0.002*x + 0.001*y).ravel()])
        self.parameters = {
            'global_map_publish_unobserved_prior': True, 'global_map_prior_bounds_enabled': True,
            'global_map_initial_elevation': 0.0, 'global_map_initial_variance': 10000.0,
            'global_map_prior_min_x': 0.0, 'global_map_prior_max_x': 0.0,
            'global_map_prior_min_y': 0.0, 'global_map_prior_max_y': 0.0,
            'resolution': 0.04,
        }
        self.gate = {
            'schema_version': 1, 'state': 'READY', 'ros_stamp_sec': 100.0,
            'map_identity': {
                'coordinate_kind': 'local_ndt_map', 'backend': 'universe', 'frame': 'map',
                'base_frame': 'base_link', 'reference_pcd': str(self.reference),
                'reference_pcd_sha256': self.reference_sha256,
                'loaded_map_geometry_sha256': loaded_map_digest(self.xyz),
            },
            'sensor_transform': {'translation': [0.4215, 0, 0.78499], 'rotation': [0, 0, 0, 1]},
            'cloud_filter_contract': {'version': 1, 'finite_xyz_only': True},
        }
        self.spawned, self.stopped = [], []
        def spawn(metadata, parameters, generation):
            process = Process()
            self.spawned.append((deepcopy(metadata), deepcopy(parameters), generation, process))
            return process
        self.estimator = unittest.mock.Mock(wraps=estimate_initial_ground)
        self.controller = LocalElevationController(
            self.reference, self.parameters, self.root / 'maps', spawn, self.stopped.append,
            seed_estimator=self.estimator)
        self.covariance = np.diag([0.01]*6).ravel().tolist()

    def tearDown(self):
        self.temporary.cleanup()

    def pose(self, position=None, now=10.0, ros=100.0, stamp=None, frame='map', child_frame='base_link'):
        self.controller.on_seed_pose(position or [0, 0, 0], [0, 0, 0, 1], self.covariance,
                                     int(ros * 1e9) if stamp is None else stamp,
                                     now, int(ros * 1e9), frame, child_frame)

    def tick(self, now=10.0, ros=100.0):
        self.gate['ros_stamp_sec'] = ros
        self.controller.on_gate(self.gate, now)
        return self.controller.tick(now, ros)

    def seed(self):
        self.controller.on_loaded_map(self.xyz)
        self.pose()
        return self.tick()

    def test_pending_zero_bounds_never_create_session_or_worker(self):
        with self.assertRaises(ValueError):
            require_local_parameters(self.parameters)
        require_local_parameters(self.parameters, allow_pending_ground_seed=True)
        self.assertEqual(self.tick(), 'WAITING_GROUND_SEED')
        self.assertIn('/map/pointcloud_map', self.controller.reason)
        self.controller.on_loaded_map(self.xyz)
        self.assertEqual(self.tick(), 'WAITING_GROUND_SEED')
        self.assertIn('/localization/kinematic_state', self.controller.reason)
        self.assertIsNone(self.controller.session)
        self.assertFalse(self.spawned)
        self.assertFalse((self.root / 'maps').exists())

    def test_actual_ground_freezes_initial_bounds_height_and_full_source_metadata(self):
        self.assertEqual(self.seed(), 'INITIAL_GROUND')
        self.assertEqual(len(self.spawned), 1)
        parameters, seed = self.controller.parameters, self.controller.ground_seed
        self.assertAlmostEqual(seed['height'], -0.4, places=2)
        self.assertEqual(seed['center_xy'], [0.0, 0.0])
        self.assertEqual(seed['bounds_xy'], [[-5.0, -5.0], [5.0, 5.0]])
        self.assertEqual(parameters['global_map_initial_elevation'], seed['height'])
        self.assertEqual(parameters['global_map_initial_variance'], 10000.0)
        self.assertFalse(parameters['use_initializer_at_start'])
        self.assertFalse(seed['window_follows_robot'])
        self.assertEqual(seed['observation_count'], 0)
        self.assertTrue(seed['quality']['accepted'])
        self.assertEqual(seed['source']['loaded_map_geometry_sha256'], loaded_map_digest(self.xyz))
        self.assertEqual(self.controller.session['identity']['initial_ground_seed'], seed)
        identity = deepcopy(self.controller.identity)
        self.pose([1, 1, 0], now=10.2, ros=100.2)
        self.assertEqual(self.tick(10.2, 100.2), 'INITIAL_GROUND')
        self.assertEqual(self.controller.identity, identity)
        self.assertEqual(self.estimator.call_count, 1)

    def test_finite_prior_and_processing_without_observations_are_not_mapping(self):
        self.seed()
        generation = self.controller.generation
        self.controller.on_processed(1, 10.1, generation)
        self.controller.on_observation(0, 100.1, 10.1, generation)
        self.assertEqual(self.tick(10.1, 100.1), 'INITIAL_GROUND')
        self.controller.on_processed(2, 10.2, generation)
        self.controller.on_observation(25, 100.2, 10.2, generation)
        self.assertEqual(self.tick(10.2, 100.2), 'MAPPING')

    def test_pause_resume_preserves_same_seed_and_db_after_robot_moves(self):
        self.seed()
        seed, database = deepcopy(self.controller.ground_seed), self.controller.session['database']
        self.gate['state'] = 'WAITING_LOCALIZATION'
        self.assertEqual(self.tick(10.2, 100.2), 'WAITING_LOCALIZATION')
        self.assertEqual(len(self.stopped), 1)
        self.pose([2, 0, 0], now=10.4, ros=100.4)
        self.gate['state'] = 'READY'
        self.assertEqual(self.tick(10.4, 100.4), 'INITIAL_GROUND')
        self.assertEqual(self.controller.ground_seed, seed)
        self.assertEqual(self.controller.session['database'], database)
        self.assertEqual(self.estimator.call_count, 1)
        self.assertTrue(self.spawned[-1][1]['global_map_resume_from_disk'])

    def test_slow_estimate_freezes_valid_snapshot_then_waits_for_new_gate(self):
        self.controller.on_loaded_map(self.xyz)
        self.pose()
        with patch('scout_localization_support.local_elevation_manager.time.monotonic',
                   side_effect=[10.0, 10.8]):
            self.assertEqual(self.tick(), 'WAITING_INPUT_GATE')
        self.assertIsNotNone(self.controller.ground_seed)
        self.assertIsNone(self.controller.session)
        self.assertFalse(self.spawned)
        self.assertAlmostEqual(self.controller.seed_estimation_elapsed_sec, 0.8)
        # The original pose is now old, but the accepted initial snapshot remains valid.
        self.assertEqual(self.tick(10.9, 100.9), 'INITIAL_GROUND')
        self.assertEqual(self.estimator.call_count, 1)
        self.assertEqual(self.controller.ground_seed['source']['base_pose']['stamp_ns'], 100_000_000_000)

    def test_stale_future_wrong_frame_or_failed_fit_never_guess_ground(self):
        self.controller.on_loaded_map(self.xyz)
        for stamp, frame, child in [(99_000_000_000, 'map', 'base_link'),
                                    (101_000_000_000, 'map', 'base_link'),
                                    (100_000_000_000, 'odom', 'base_link'),
                                    (100_000_000_000, 'map', 'rslidar')]:
            self.pose(stamp=stamp, frame=frame, child_frame=child)
            self.assertEqual(self.tick(), 'WAITING_GROUND_SEED')
            self.assertFalse(self.spawned)
        self.pose()
        self.estimator.side_effect = ValueError('insufficient supported ground cells')
        self.assertEqual(self.tick(), 'WAITING_GROUND_SEED')
        self.assertIn('insufficient supported ground cells', self.controller.reason)
        self.assertIsNone(self.controller.ground_seed)
        self.assertIsNone(self.controller.session)
        self.assertFalse(self.spawned)

    def test_gate_geometry_mismatch_or_later_actual_map_change_requires_restart(self):
        self.controller.on_loaded_map(self.xyz)
        self.pose()
        self.gate['map_identity']['loaded_map_geometry_sha256'] = 'b' * 64
        self.assertEqual(self.tick(), 'RESTART_REQUIRED')
        self.assertFalse(self.spawned)
        # A fresh controller may seed, but changing its actual map later stops it.
        self.setUp_fresh_controller()
        self.seed()
        self.controller.on_loaded_map(self.xyz + [0, 0, 0.1])
        self.assertEqual(self.controller.state, 'RESTART_REQUIRED')
        self.assertEqual(len(self.stopped), 1)

    def setUp_fresh_controller(self):
        self.gate['map_identity']['loaded_map_geometry_sha256'] = loaded_map_digest(self.xyz)
        def spawn(metadata, parameters, generation):
            self.spawned.append((deepcopy(metadata), deepcopy(parameters), generation, Process()))
            return self.spawned[-1][-1]
        self.controller = LocalElevationController(
            self.reference, self.parameters, self.root / 'maps', spawn, self.stopped.append,
            seed_estimator=self.estimator)

    def test_prior_contract_rejects_unbounded_low_variance_or_unbound_seed(self):
        for changes in ({'global_map_prior_bounds_enabled': False},
                        {'global_map_initial_variance': 1.0},
                        {'global_map_prior_max_x': -1.0}):
            with self.assertRaises(ValueError):
                require_local_parameters(dict(self.parameters, **changes), allow_pending_ground_seed=True)
        self.seed()
        with self.assertRaises(ValueError):
            local_identity(self.gate, self.reference, self.reference_sha256, self.controller.parameters)
        changed = deepcopy(self.controller.ground_seed)
        changed['observation_count'] = 1
        with self.assertRaises(ValueError):
            local_identity(self.gate, self.reference, self.reference_sha256, self.controller.parameters, changed)
        changed = deepcopy(self.controller.ground_seed)
        changed['source']['loaded_map_geometry_sha256'] = 'c' * 64
        with self.assertRaises(ValueError):
            local_identity(self.gate, self.reference, self.reference_sha256, self.controller.parameters, changed)

    @unittest.skipUnless(os.environ.get('SCOUT_RUN_ISOLATED_MANAGER_ROS_TESTS') == '1',
                         'Run only in an isolated ROS domain; no native worker is started')
    def test_real_ros_loaded_map_and_odometry_callbacks_seed_without_native_worker(self):
        import yaml
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
        from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
        from std_msgs.msg import Header

        profile = self.root / 'mapping.yaml'
        profile.write_text(yaml.safe_dump({'/elevation_mapping_node': {
            'ros__parameters': require_local_parameters(self.parameters, allow_pending_ground_seed=True)}}))
        checked = []

        def inspect_node(node):
            self.assertEqual(node.map_subscription.qos_profile.durability, DurabilityPolicy.TRANSIENT_LOCAL)
            self.assertEqual(node.map_subscription.qos_profile.depth, 1)
            self.assertEqual(node.pose_subscription.qos_profile.reliability, ReliabilityPolicy.BEST_EFFORT)
            node.on_map(create_cloud_xyz32(Header(frame_id='map'), self.xyz.tolist()))
            pose = Odometry()
            pose.header.frame_id, pose.child_frame_id = 'map', 'base_link'
            pose.header.stamp = node.get_clock().now().to_msg()
            pose.pose.pose.orientation.w = 1.0
            pose.pose.covariance = self.covariance
            node.on_seed_pose(pose)
            gate = deepcopy(self.gate)
            gate['ros_stamp_sec'] = node.get_clock().now().nanoseconds / 1e9
            node.controller.on_gate(gate, time.monotonic())
            node.controller.spawn = lambda *args: Process()
            node.tick()
            self.assertEqual(node.controller.state, 'INITIAL_GROUND')
            self.assertAlmostEqual(node.controller.ground_seed['height'], -0.4, places=2)
            self.assertEqual(node.controller.ground_seed['source']['base_pose']['frame'], 'map')
            self.assertEqual(node.controller.observed_cells, 0)
            # The fake process has no resources; prevent the production stop routine
            # from treating it as an operating-system worker during main() cleanup.
            node.controller.process = None
            checked.append(True)

        with patch('rclpy.spin', side_effect=inspect_node), patch('subprocess.Popen') as spawn:
            manager_main(args=['--ros-args', '-p', f'reference_pcd:={self.reference}',
                               '-p', f'mapping_config_file:={profile}',
                               '-p', f'storage_root:={self.root / "ros_maps"}'])
            spawn.assert_not_called()
        self.assertEqual(checked, [True])


if __name__ == '__main__':
    unittest.main()
