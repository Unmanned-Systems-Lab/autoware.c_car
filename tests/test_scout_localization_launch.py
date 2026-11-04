"""Resolve Scout launch contracts without executing a driver or ROS node.

Run from the actual checkout after sourcing its setup_env.bash. Only launch
configuration actions are executed; hardware/process actions are inspected.
"""

import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.actions import (
    DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
    PopEnvironment, PopLaunchConfigurations, PushEnvironment,
    PushLaunchConfigurations, SetLaunchConfiguration,
)
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.utilities import normalize_to_list_of_substitutions, perform_substitutions
from launch_ros.actions import Node

from scout_localization_support.map_checks import require_ndt_map


ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'src/launcher/autoware_launch/autoware_launch/launch'


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve(context, value):
    return perform_substitutions(context, normalize_to_list_of_substitutions(value))


def launch_context(module, **overrides):
    context = LaunchContext()
    context.launch_configurations.update({key: str(value) for key, value in overrides.items()})
    # Resolve the source launch's declarations, without its opaque function.
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    return context


def include_path(action, context):
    # Humble's location property is only an expanded filename after loading the
    # description. Parsing it constructs actions but executes none of them.
    source = action.launch_description_source
    source.get_launch_description(context)
    return Path(source.location)


def include_arguments(action, context):
    return {name: resolve(context, value) for name, value in action.launch_arguments}


def write_pcd(path, *, points=1200, encoding='binary', payload=None):
    header = (
        'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
        f'WIDTH {points}\nHEIGHT 1\nPOINTS {points}\nDATA {encoding}\n'
    ).encode('ascii')
    if payload is None:
        payload = b''.join(struct.pack('<fff', (i % 40) * 0.5,
                                      (i // 40) * 0.5, (i % 3) * 0.1)
                           for i in range(points))
    path.write_bytes(header + payload)
    return path


class ScoutLaunchContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = load_source(
            '_scout_selector_contract', LAUNCH / 'components/scout_localization_selector.launch.py')
        cls.universe = load_source(
            '_scout_universe_contract', LAUNCH / 'components/scout_universe_localization.launch.py')
        cls.standalone = load_source(
            '_scout_standalone_contract', LAUNCH / 'scout_localization.launch.py')
        cls.preflight = load_source(
            '_scout_preflight_contract', ROOT / 'scripts/check_scout_localization_start.py')

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='scout-launch-contract-')
        self.addCleanup(self.directory.cleanup)
        self.temp = Path(self.directory.name)
        self.valid_map = write_pcd(self.temp / 'real_cloud.pcd')

    def test_fixposition_selects_only_legacy_backend_and_forwards_existing_arguments(self):
        context = launch_context(
            self.selector, localization_mode='fixposition',
            input_odom_topic='/test/fixposition/odometry_enu',
            output_base_link_odom_topic='/test/odom_at_base',
            base_link_to_ins_x='0.4495158179045505', base_link_to_ins_z='0.65899',
            map_projector_info_path='/test/known_projector.yaml',
            output_frame_id='map', output_child_frame_id='base_link')
        actions = self.selector._setup(context)
        includes = [action for action in actions if isinstance(action, IncludeLaunchDescription)]
        self.assertEqual(len(includes), 1)
        self.assertEqual(include_path(includes[0], context).name,
                         'ins_localization_component.launch.xml')
        self.assertFalse(any(isinstance(action, Node) for action in actions))
        args = include_arguments(includes[0], context)
        self.assertEqual(args['input_odom_topic'], '/test/fixposition/odometry_enu')
        self.assertEqual(args['output_base_link_odom_topic'], '/test/odom_at_base')
        self.assertEqual(args['map_projector_info_path'], '/test/known_projector.yaml')
        self.assertEqual(args['base_link_to_ins_x'], '0.4495158179045505')
        self.assertEqual(args['base_link_to_ins_z'], '0.65899')
        self.assertEqual((args['output_frame_id'], args['output_child_frame_id']),
                         ('map', 'base_link'))

    def test_universe_selects_only_native_wrapper_and_forwards_sensor_choices(self):
        context = launch_context(
            self.selector, localization_mode='universe', map_path=self.temp,
            pointcloud_map_file=self.valid_map.name,
            universe_imu_topic='/test/zed_imu', universe_wheel_odometry_topic='/test/wheel_odom',
            universe_initial_pose='[1,2,3,0,0,0,1]')
        actions = self.selector._setup(context)
        includes = [action for action in actions if isinstance(action, IncludeLaunchDescription)]
        self.assertEqual(len(includes), 1)
        self.assertEqual(include_path(includes[0], context).name,
                         'scout_universe_localization.launch.py')
        self.assertFalse(any(isinstance(action, Node) for action in actions))
        args = include_arguments(includes[0], context)
        self.assertEqual(args['imu_topic'], '/test/zed_imu')
        self.assertEqual(args['wheel_odometry_topic'], '/test/wheel_odom')
        self.assertEqual(args['initial_pose'], '[1,2,3,0,0,0,1]')

    def test_universe_rejects_sparse_missing_and_truncated_maps_before_backend_actions(self):
        sparse = write_pcd(self.temp / 'placeholder.pcd', points=105)
        truncated = write_pcd(self.temp / 'truncated.pcd', payload=b'\x00' * 12)
        missing = self.temp / 'missing.pcd'
        for path in [sparse, truncated, missing]:
            with self.subTest(path=path.name):
                context = launch_context(
                    self.selector, localization_mode='universe', map_path=self.temp,
                    pointcloud_map_file=path.name)
                with self.assertRaises((ValueError, OSError)):
                    self.selector._setup(context)

    def test_invalid_mode_never_constructs_a_backend(self):
        for module in [self.selector, self.standalone]:
            with self.subTest(module=module.__name__):
                context = launch_context(module, localization_mode='unexpected-backend')
                with self.assertRaisesRegex(RuntimeError, 'localization_mode'):
                    module._setup(context)

    def test_invalid_boolean_does_not_disable_the_can_gate_or_sensor_flags(self):
        context = launch_context(self.universe, require_can_feedback='tru')
        with self.assertRaisesRegex(RuntimeError, 'require_can_feedback'):
            self.universe._setup(context)
        context = launch_context(
            self.standalone, start_sensors='definitely-not-a-bool',
            pointcloud_map=self.valid_map, rviz='false')
        with self.assertRaisesRegex(RuntimeError, 'start_sensors'):
            self.standalone._setup(context)

    def test_native_scope_selects_private_imu_and_disables_gnss_without_leaking(self):
        context = launch_context(
            self.universe, start_imu_driver='false', publish_zed_mount_tf='false',
            start_wheel_odometry_driver='false')
        context.launch_configurations['input_imu_topic'] = '/parent/other_imu'
        context.launch_configurations['gnss_enabled'] = 'true'
        groups = [action for action in self.universe._setup(context)
                  if isinstance(action, GroupAction)]
        self.assertEqual(len(groups), 1)
        native_includes = 0
        safe_actions = (SetLaunchConfiguration, PushLaunchConfigurations,
                        PopLaunchConfigurations, PushEnvironment, PopEnvironment)
        for action in groups[0].get_sub_entities():
            if isinstance(action, IncludeLaunchDescription):
                native_includes += 1
                self.assertEqual(include_path(action, context).name,
                                 'tier4_localization_component.launch.xml')
                args = include_arguments(action, context)
                self.assertEqual((args['pose_source'], args['twist_source']), ('ndt', 'gyro_odom'))
                self.assertEqual(args['input_pointcloud'], '/scout/localization/points_deskewed')
                self.assertEqual(context.launch_configurations['gnss_enabled'], 'false')
                self.assertEqual(context.launch_configurations['input_imu_topic'],
                                 '/scout/localization/imu')
                # Execute the real native declarations, not a test-side copy of
                # their fallback behavior. No native Node/Include is executed.
                native_paths = [
                    (Path(get_package_share_directory('autoware_gyro_odometer'))
                     / 'launch/gyro_odometer.launch.xml', 'input_imu_topic'),
                    (Path(get_package_share_directory('tier4_localization_launch'))
                     / 'launch/pose_twist_estimator/pose_twist_estimator.launch.xml', 'gnss_enabled'),
                ]
                for source, name in native_paths:
                    description = AnyLaunchDescriptionSource(str(source)).get_launch_description(context)
                    declarations = [entity for entity in description.entities
                                    if isinstance(entity, DeclareLaunchArgument) and entity.name == name]
                    self.assertEqual(len(declarations), 1)
                    declarations[0].execute(context)
                self.assertEqual(context.launch_configurations['gnss_enabled'], 'false')
                self.assertEqual(context.launch_configurations['input_imu_topic'],
                                 '/scout/localization/imu')
            else:
                self.assertIsInstance(action, safe_actions)
                action.execute(context)
        self.assertEqual(native_includes, 1)
        self.assertEqual(context.launch_configurations['input_imu_topic'], '/parent/other_imu')
        self.assertEqual(context.launch_configurations['gnss_enabled'], 'true')

    def test_localization_only_actions_have_no_planning_or_control_process(self):
        context = launch_context(
            self.standalone, pointcloud_map=self.valid_map, start_sensors='false',
            start_wheel_odometry_driver='false', require_can_feedback='false', rviz='false')
        actions = self.standalone._setup(context)
        nodes = [action for action in actions if isinstance(action, Node)]
        packages = [resolve(context, action.node_package) for action in nodes]
        self.assertEqual(sorted(packages), sorted([
            'robot_state_publisher', 'autoware_map_loader',
            'rclcpp_components', 'scout_localization_support']))
        includes = [action for action in actions if isinstance(action, IncludeLaunchDescription)]
        self.assertEqual([include_path(action, context).name for action in includes],
                         ['scout_universe_localization.launch.py'])
        self.assertEqual(include_arguments(includes[0], context)['start_wheel_odometry_driver'], 'false')

    def test_zed_launch_node_name_does_not_leak_into_native_localizers(self):
        context = launch_context(
            self.universe, start_imu_driver='true', publish_zed_mount_tf='false',
            start_wheel_odometry_driver='false')
        self.assertNotIn('node_name', context.launch_configurations)
        actions = self.universe._setup(context)
        # Locate the actual third-party include, not a test-generated copy.
        self.assertFalse(any(
            isinstance(action, IncludeLaunchDescription)
            and include_path(action, context).name == 'zed_camera.launch.py'
            for action in actions))
        candidates = []
        for group in [action for action in actions if isinstance(action, GroupAction)]:
            for child in group.get_sub_entities():
                if (isinstance(child, IncludeLaunchDescription)
                        and include_path(child, context).name == 'zed_camera.launch.py'):
                    candidates.append((group, child))
        self.assertEqual(len(candidates), 1)
        group, _ = candidates[0]
        scope = group.get_sub_entities()
        self.assertIsInstance(scope[0], PushLaunchConfigurations)
        self.assertIsInstance(scope[-1], PopLaunchConfigurations)
        safe_actions = (SetLaunchConfiguration, PushLaunchConfigurations,
                        PopLaunchConfigurations, PushEnvironment, PopEnvironment)
        for action in scope:
            if isinstance(action, IncludeLaunchDescription):
                description = action.launch_description_source.get_launch_description(context)
                declarations = [entity for entity in description.entities
                                if isinstance(entity, DeclareLaunchArgument)
                                and entity.name == 'node_name']
                self.assertEqual(len(declarations), 1)
                declarations[0].execute(context)
                self.assertEqual(context.launch_configurations['node_name'], 'zed_node')
            else:
                self.assertIsInstance(action, safe_actions)
                action.execute(context)
        self.assertNotIn('node_name', context.launch_configurations)
        # Each native include gets its own existing scope and actual declaration.
        for package, filename, expected in [
            ('autoware_ndt_scan_matcher', 'ndt_scan_matcher.launch.xml', 'ndt_scan_matcher'),
            ('autoware_pose_instability_detector', 'pose_instability_detector.launch.xml',
             'pose_instability_detector'),
        ]:
            with self.subTest(package=package):
                path = Path(get_package_share_directory(package)) / 'launch' / filename
                native_scope = GroupAction(actions=[IncludeLaunchDescription(
                    AnyLaunchDescriptionSource(str(path)))])
                for action in native_scope.get_sub_entities():
                    if isinstance(action, IncludeLaunchDescription):
                        description = action.launch_description_source.get_launch_description(context)
                        declarations = [entity for entity in description.entities
                                        if isinstance(entity, DeclareLaunchArgument)
                                        and entity.name == 'node_name']
                        self.assertEqual(len(declarations), 1)
                        declarations[0].execute(context)
                        self.assertEqual(context.launch_configurations['node_name'], expected)
                    else:
                        self.assertIsInstance(action, safe_actions)
                        action.execute(context)
                self.assertNotIn('node_name', context.launch_configurations)

    def test_preflight_rejects_invalid_mode_boolean_and_map_path_escape(self):
        cases = [
            ['localization_mode:=typo'],
            ['--show-args', 'require_can_feedback:=tru'],
            ['localization_mode:=universe', f'map_path:={self.temp}',
             'pointcloud_map_file:=../outside.pcd'],
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.preflight.check(arguments)

    def test_preflight_dry_descriptions_do_not_require_hardware_or_a_map(self):
        self.assertEqual(self.preflight.check(['--show-args'])[0], 'fixposition')
        self.assertEqual(self.preflight.check(['--show-args'], standalone=True)[0], 'universe')

    def test_truncated_ascii_map_is_rejected(self):
        path = write_pcd(self.temp / 'short_ascii.pcd', encoding='ascii', payload=b'0 0 0\n')
        with self.assertRaises(ValueError):
            require_ndt_map(path)

    def test_truncated_compressed_map_is_rejected(self):
        path = write_pcd(self.temp / 'short_compressed.pcd', encoding='binary_compressed',
                         payload=struct.pack('<II', 100, 1200 * 12) + b'\x00')
        with self.assertRaises(ValueError):
            require_ndt_map(path)

    def test_a_complete_binary_map_passes_the_preflight_map_contract(self):
        info = require_ndt_map(self.valid_map)
        self.assertEqual((Path(info['path']), info['points'], info['encoding']),
                         (self.valid_map.resolve(), 1200, 'binary'))

    def test_shell_entries_propagate_preflight_failure_instead_of_launching(self):
        # Even if failure propagation regresses, this DEBUG trap blocks the
        # actual exec before any launch can start a physical driver.
        guard = self.temp / 'block_launch.bash'
        guard.write_text(
            'trap \'case "$BASH_COMMAND" in exec\\ ros2\\ launch*) '
            'echo SCOUT_TEST_BLOCKED_LAUNCH >&2; exit 97;; esac\' DEBUG\n')
        environment = os.environ.copy()
        environment['BASH_ENV'] = str(guard)
        cases = [
            ('start_scout.bash', ['localization_mode:=invalid']),
            ('start_scout_localization.bash', ['localization_mode:=invalid']),
            ('start_scout.bash', ['localization_mode:=universe', f'map_path:={self.temp}',
                                  'pointcloud_map_file:=missing.pcd']),
            ('start_scout_localization.bash', ['localization_mode:=universe',
                                               f'pointcloud_map:={self.temp / "missing.pcd"}']),
        ]
        for entry, arguments in cases:
            with self.subTest(entry=entry, arguments=arguments):
                result = subprocess.run(['bash', str(ROOT / entry), *arguments],
                                        cwd=ROOT, env=environment, capture_output=True,
                                        text=True, timeout=90)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn('Scout localization preflight failed:', result.stderr)
                self.assertNotIn('SCOUT_TEST_BLOCKED_LAUNCH', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
