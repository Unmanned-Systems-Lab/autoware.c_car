"""Local map identity and mapper lifecycle checks without ROS or robot hardware."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scout_localization_support.local_elevation_manager import (
    LocalElevationController, grid_observation, main as manager_main, stop_process,
)
from scout_localization_support.local_elevation_session import (
    hash_file, local_identity, prepare_local_session, require_local_parameters, snapshot_database,
)


class Process:
    def __init__(self):
        self.returncode = None

    def poll(self):
        return self.returncode


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pcd = self.root / 'GlobalMap.pcd'
        self.pcd.write_bytes(b'local map fixture, no geographic origin\n')
        self.sha256, self.signature = hash_file(self.pcd)
        self.parameters = require_local_parameters({
            'resolution': 0.1, 'global_map_local_topic': '/elevation_mapping_node/elevation_map_global_local',
            'subscribers': {'lidar': {'topic_name': '/scout/elevation/points'}},
        })
        self.gate = {
            'schema_version': 1, 'sequence': 1, 'state': 'READY', 'ros_stamp_sec': 100.0,
            'map_identity': {
                'coordinate_kind': 'local_ndt_map', 'backend': 'universe', 'frame': 'map',
                'base_frame': 'base_link', 'reference_pcd': str(self.pcd),
                'reference_pcd_sha256': self.sha256, 'loaded_map_geometry_sha256': 'a' * 64,
            },
            'sensor_transform': {'translation': [0.1, 0.0, 0.8], 'rotation': [0, 0, 0, 1]},
            'cloud_filter_contract': {'version': 1, 'finite_xyz_only': True,
                                      'ego_bounds_base_link_m': [[-0.3, -0.4, -0.05], [0.8, 0.4, 1.0]]},
        }
        self.spawned, self.stopped = [], []

    def tearDown(self):
        self.temporary.cleanup()

    def identity(self, gate=None, parameters=None):
        return local_identity(gate or self.gate, self.pcd, self.sha256, parameters or self.parameters)

    def controller(self):
        def spawn(session, parameters, generation):
            process = Process()
            self.spawned.append((session, parameters, generation, process))
            return process
        return LocalElevationController(self.pcd, self.parameters, self.root / 'local_maps',
                                        spawn, self.stopped.append)

    def ready(self, controller, now=10.0, ros=100.0):
        self.gate['ros_stamp_sec'] = ros
        controller.on_gate(self.gate, now)
        return controller.tick(now, ros)

    def test_fresh_runs_are_isolated_and_same_run_restarts_preserve_db(self):
        identity = self.identity()
        first = prepare_local_session(self.root, identity)
        second = prepare_local_session(self.root, identity)
        self.assertNotEqual(first['database'], second['database'])
        self.assertIn('/local_ndt/', Path(first['database']).as_posix())
        self.assertFalse(Path(first['database']).exists())
        with sqlite3.connect(first['database']) as connection:
            connection.execute('CREATE TABLE observed(value INTEGER)')
            connection.execute('INSERT INTO observed VALUES(42)')
        resumed = prepare_local_session(self.root, identity, first['run_id'])
        self.assertEqual(first, resumed)
        with sqlite3.connect(resumed['database']) as connection:
            self.assertEqual(connection.execute('SELECT value FROM observed').fetchone(), (42,))

    def test_every_map_transform_filter_and_parameter_change_changes_identity(self):
        original = self.identity()
        for key in ('reference_pcd_sha256', 'loaded_map_geometry_sha256'):
            changed = deepcopy(self.gate)
            changed['map_identity'][key] = 'b' * 64
            if key == 'reference_pcd_sha256':
                with self.assertRaises(ValueError):
                    self.identity(changed)
            else:
                self.assertNotEqual(original, self.identity(changed))
        changed = deepcopy(self.gate)
        changed['sensor_transform']['translation'][0] += 0.01
        self.assertNotEqual(original, self.identity(changed))
        changed = deepcopy(self.gate)
        changed['cloud_filter_contract']['version'] = 2
        self.assertNotEqual(original, self.identity(changed))
        parameters = dict(self.parameters, resolution=0.2)
        self.assertNotEqual(original, self.identity(parameters=parameters))
        changed = deepcopy(self.gate)
        changed['sensor_transform']['rotation'] = [0, 0, 0, -1]
        self.assertEqual(original, self.identity(changed))

    def test_local_contract_rejects_fp_identity_initializers_and_fake_prior(self):
        for key, value in [('coordinate_kind', 'fp_enu0'), ('backend', 'fixposition'), ('frame', 'odom')]:
            gate = deepcopy(self.gate)
            gate['map_identity'][key] = value
            with self.assertRaises(ValueError):
                self.identity(gate)
        for key, value in [('map_frame', 'FP_ENU0'), ('use_initializer_at_start', True),
                           ('global_map_publish_unobserved_prior', True),
                           ('enable_builtin_traversability', True), ('enable_global_map', False)]:
            with self.assertRaises(ValueError):
                require_local_parameters(dict(self.parameters, **{key: value}))
        parameters = require_local_parameters(dict(self.parameters, global_map_storage_path='/old/fp/tiles.sqlite3'))
        self.assertNotIn('global_map_storage_path', parameters)
        self.assertTrue(parameters['global_map_resume_from_disk'])

    def test_worker_input_cannot_bypass_gate_or_use_wrong_ray_origin(self):
        for topic in ('/sensing/lidar/concatenated/pointcloud', '/scout/localization/points_deskewed'):
            parameters = deepcopy(self.parameters)
            parameters['subscribers']['lidar']['topic_name'] = topic
            with self.assertRaises(ValueError):
                require_local_parameters(parameters)
        parameters = deepcopy(self.parameters)
        parameters['subscribers']['ungated'] = {'topic_name': '/raw', 'data_type': 'pointcloud'}
        with self.assertRaises(ValueError):
            require_local_parameters(parameters)
        parameters = deepcopy(self.parameters)
        parameters['subscribers']['lidar']['data_type'] = 'image'
        with self.assertRaises(ValueError):
            require_local_parameters(parameters)
        for changes in ({'ray_origin_frame': 'base_link'}, {'self_filter_enabled': True}):
            with self.assertRaises(ValueError):
                require_local_parameters(dict(self.parameters, **changes))

    def test_manifest_mismatch_and_unbound_database_are_rejected(self):
        identity = self.identity()
        metadata = prepare_local_session(self.root, identity)
        path = Path(metadata['directory']) / 'session.json'
        tampered = deepcopy(metadata)
        tampered['identity']['coordinate_kind'] = 'fp_enu0'
        path.write_text(json.dumps(tampered))
        with self.assertRaises(ValueError):
            prepare_local_session(self.root, identity, metadata['run_id'])
        path.unlink()
        Path(metadata['database']).touch()
        with self.assertRaises(ValueError):
            prepare_local_session(self.root, identity, metadata['run_id'])

    def test_wal_snapshot_preserves_committed_data_and_does_not_overwrite(self):
        source, destination = self.root / 'tiles.sqlite3', self.root / 'snapshot.sqlite3'
        with sqlite3.connect(source) as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            connection.execute('CREATE TABLE observed(value INTEGER)')
            connection.execute('INSERT INTO observed VALUES(7)')
            connection.commit()
            snapshot_database(source, destination)
            with sqlite3.connect(destination) as snapshot:
                self.assertEqual(snapshot.execute('SELECT value FROM observed').fetchone(), (7,))
            with self.assertRaises(FileExistsError):
                snapshot_database(source, destination)

    def test_invalid_stale_future_or_waiting_gate_never_starts_worker(self):
        controller = self.controller()
        controller.tick(10, 100)
        self.assertFalse(self.spawned)
        controller.on_gate('{bad json', 10)
        controller.tick(10, 100)
        self.assertFalse(self.spawned)
        invalid = dict(self.gate, state='MAPPING')
        controller.on_gate(invalid, 10)
        self.assertNotEqual(controller.tick(10, 100), 'MAPPING')
        self.assertFalse(self.spawned)
        controller.on_gate(self.gate, 10)
        controller.tick(10.8, 100.8)
        self.assertFalse(self.spawned)
        controller.on_gate(self.gate, 10)
        controller.tick(10, 99)
        self.assertFalse(self.spawned)
        for state in ('WAITING_INITIALIZATION', 'WAITING_LOCALIZATION', 'WAITING_TF'):
            self.gate['state'] = state
            controller.on_gate(self.gate, 10)
            self.assertEqual(controller.tick(10, 100), state)
        self.assertFalse(self.spawned)

    def test_process_alive_is_not_mapping_and_requires_real_progress_and_observation(self):
        controller = self.controller()
        self.assertEqual(self.ready(controller), 'STARTING')
        generation = controller.generation
        controller.on_observation(200, 99, 10.01, generation)  # Restored pre-worker window.
        self.assertEqual(controller.tick(10.02, 100.02), 'STARTING')
        controller.on_observation(200, 100.02, 10.02, generation)
        self.assertEqual(controller.tick(10.03, 100.03), 'RUNNING_AVAILABLE_MAP')
        controller.on_processed(1, 10.04, generation)
        self.assertEqual(controller.tick(10.05, 100.05), 'MAPPING')
        self.ready(controller, 11.2, 101.2)
        self.assertEqual(controller.state, 'RUNNING_AVAILABLE_MAP')
        controller.on_processed(1, 11.2, generation)  # Duplicate does not refresh progress.
        controller.on_observation(200, 101.2, 11.2, generation)
        self.assertEqual(controller.tick(11.21, 101.21), 'RUNNING_AVAILABLE_MAP')

    def test_gate_loss_stops_and_recovers_same_run_without_erasing_tiles(self):
        controller = self.controller()
        self.ready(controller)
        database = controller.session['database']
        Path(database).write_bytes(b'current run tile fixture')
        self.assertEqual(controller.tick(10.8, 100.8), 'WAITING_INPUT_GATE')
        self.assertEqual(len(self.stopped), 1)
        self.ready(controller, 11, 101)
        self.assertEqual(controller.session['database'], database)
        self.assertEqual(Path(database).read_bytes(), b'current run tile fixture')
        self.assertEqual(len(self.spawned), 2)
        self.assertTrue(self.spawned[1][1]['global_map_resume_from_disk'])
        # Late messages from the previous worker cannot produce success.
        controller.on_processed(800, 11.01, 1)
        controller.on_observation(200, 101.01, 11.01, 1)
        self.assertEqual(controller.tick(11.02, 101.02), 'STARTING')

    def test_worker_exit_has_five_second_backoff_and_new_progress_generation(self):
        controller = self.controller()
        self.ready(controller)
        self.spawned[0][3].returncode = 2
        self.assertEqual(controller.tick(10.1, 100.1), 'MAPPER_EXITED')
        self.assertEqual(self.ready(controller, 12, 102), 'RESTART_BACKOFF')
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(self.ready(controller, 15.2, 105.2), 'STARTING')
        self.assertEqual(len(self.spawned), 2)
        self.assertEqual(controller.generation, 2)
        self.assertEqual(controller.processed_count, 0)

    def test_worker_exit_with_stale_gate_still_enforces_backoff(self):
        controller = self.controller()
        self.ready(controller)
        self.spawned[0][3].returncode = 3
        self.assertEqual(controller.tick(11, 101), 'MAPPER_EXITED')
        self.assertEqual(self.ready(controller, 12, 102), 'RESTART_BACKOFF')
        self.assertEqual(len(self.spawned), 1)

    def test_changed_source_file_or_loaded_map_latches_fault_and_stops(self):
        for target in ('reference', 'loaded_map', 'transform', 'filter'):
            controller = self.controller()
            self.ready(controller)
            if target == 'reference':
                self.pcd.write_bytes(b'changed map')
            else:
                changed = deepcopy(self.gate)
                if target == 'loaded_map':
                    changed['map_identity']['loaded_map_geometry_sha256'] = 'b' * 64
                elif target == 'transform':
                    changed['sensor_transform']['translation'][0] += 0.01
                else:
                    changed['cloud_filter_contract']['version'] = 2
                controller.on_gate(changed, 10.1)
            self.assertEqual(controller.tick(10.1, 100.1), 'RESTART_REQUIRED')
            self.assertIsNone(controller.process)
            self.ready(controller, 11, 101)
            self.assertEqual(controller.state, 'RESTART_REQUIRED')
            if target == 'reference':
                self.pcd.write_bytes(b'local map fixture, no geographic origin\n')

    def test_stop_process_flushes_before_escalating(self):
        class StuckProcess(Process):
            def __init__(self):
                super().__init__()
                self.actions = []
            def send_signal(self, value):
                self.actions.append('interrupt')
            def wait(self, timeout):
                self.actions.append(timeout)
                if len(self.actions) < 6:
                    raise subprocess.TimeoutExpired('mapper', timeout)
            def terminate(self):
                self.actions.append('terminate')
            def kill(self):
                self.actions.append('kill')
        process = StuckProcess()
        stop_process(process)
        self.assertEqual(process.actions, ['interrupt', 18, 'terminate', 5, 'kill', 5])

    def test_grid_observation_uses_gridmap_header_and_only_positive_finite_measurements(self):
        message = SimpleNamespace(
            header=SimpleNamespace(frame_id='map', stamp=SimpleNamespace(sec=100, nanosec=200_000_000)),
            info=SimpleNamespace(resolution=0.04), layers=['elevation', 'observation_count'],
            data=[SimpleNamespace(data=[-1.0] * 5),
                  SimpleNamespace(data=[0.0, 2.0, 4.0, float('nan'), float('inf')])])
        self.assertEqual(grid_observation(message), (2, 100.2))
        message.header.frame_id = 'FP_ENU0'
        with self.assertRaises(ValueError):
            grid_observation(message)

    def test_real_gridmap_message_schema(self):
        try:
            from grid_map_msgs.msg import GridMap
            from std_msgs.msg import Float32MultiArray
        except ImportError:
            self.skipTest('Real ROS message types are checked in the isolated ROS environment')
        message = GridMap()
        message.header.frame_id = 'map'
        message.header.stamp.sec = 101
        message.header.stamp.nanosec = 400_000_000
        message.info.resolution = 0.04
        message.layers = ['elevation', 'observation_count']
        message.data = [Float32MultiArray(data=[0.2, 0.3, 0.4]),
                        Float32MultiArray(data=[0.0, 1.0, 5.0])]
        self.assertFalse(hasattr(message.info, 'header'))
        self.assertEqual(grid_observation(message), (2, 101.4))

    @unittest.skipUnless(os.environ.get('SCOUT_RUN_ISOLATED_MANAGER_ROS_TESTS') == '1',
                         'Enable only in an isolated ROS domain; no worker or hardware is started')
    def test_flushing_ui_precedes_blocking_stop_and_latest_gate_only(self):
        import yaml
        profile = self.root / 'mapping.yaml'
        profile.write_text(yaml.safe_dump({'/elevation_mapping_node': {'ros__parameters': self.parameters}}))
        actions = []

        class FlushingProcess(Process):
            def send_signal(self, value):
                actions.append('interrupt')
            def wait(self, timeout):
                actions.append(('wait', timeout))

        def inspect_node(node):
            self.assertEqual(node.gate_subscription.qos_profile.depth, 1)
            def published(publisher, message):
                if publisher is node.status_publisher:
                    status = message.status[0]
                    values = {value.key: value.value for value in status.values}
                    actions.append((status.message, values['reason']))
            with patch.object(type(node.status_publisher), 'publish', autospec=True, side_effect=published):
                node.stop_worker(FlushingProcess())
                self.assertEqual(actions, [
                    ('STOPPING_FLUSHING', 'Waiting for current-run SQLite flush'),
                    'interrupt', ('wait', 18),
                ])
                actions.clear()
                with patch.object(type(node.context), 'ok', return_value=False):
                    node.stop_worker(FlushingProcess())
                self.assertEqual(actions, ['interrupt', ('wait', 18)])

        with patch('rclpy.spin', side_effect=inspect_node), patch('subprocess.Popen') as spawn:
            manager_main(args=['--ros-args', '-p', f'reference_pcd:={self.pcd}',
                               '-p', f'mapping_config_file:={profile}',
                               '-p', f'storage_root:={self.root / "maps"}'])
            spawn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
