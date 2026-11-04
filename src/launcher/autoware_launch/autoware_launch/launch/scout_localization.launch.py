"""Localization-only Scout bringup: no planning or vehicle command interface."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _value(context, name):
    return LaunchConfiguration(name).perform(context)


def _flag(context, name):
    value = _value(context, name).lower()
    if value not in ('true', 'false', '1', '0'):
        raise RuntimeError(f'{name} must be true or false, got {value!r}')
    return value in ('true', '1')


def _setup(context):
    import subprocess
    from scout_localization_support.map_checks import require_ndt_map
    share = Path(get_package_share_directory('autoware_launch'))
    mode = _value(context, 'localization_mode')
    if mode not in ('fixposition', 'universe'):
        raise RuntimeError('localization_mode must be fixposition or universe')
    elevation = _flag(context, 'start_elevation')
    traversability = _flag(context, 'start_traversability')
    if traversability and (mode != 'universe' or not elevation):
        raise RuntimeError('Combined traversability requires Universe and start_elevation:=true; attach to an existing session with start_scout_traversability.bash')
    if elevation and mode != 'universe':
        raise RuntimeError('start_elevation requires localization_mode:=universe for this local-map pipeline')
    vehicle_share = Path(get_package_share_directory('scout_vehicle_description'))
    description = subprocess.check_output(
        ['xacro', str(vehicle_share / 'urdf/vehicle.xacro')], text=True)
    actions = [Node(package='robot_state_publisher', executable='robot_state_publisher',
                    name='scout_localization_state_publisher', output='screen',
                    parameters=[{'robot_description': description}])]
    if _flag(context, 'start_sensors'):
        sensing_share = Path(get_package_share_directory('scout_sensor_kit_launch'))
        actions.append(IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(sensing_share / 'launch/sensing.launch.xml')),
            launch_arguments={'start_drivers': 'true',
                              'start_fixposition_driver': 'true' if mode == 'fixposition' else 'false'}.items()))
    if mode == 'fixposition':
        actions.append(IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(share / 'launch/components/ins_localization_component.launch.xml')),
            launch_arguments={
                'launch_ins_driver': 'false', 'input_odom_topic': '/fixposition/odometry_enu',
                'output_base_link_odom_topic': '/odom_ins_at_base_link',
                'base_link_to_ins_x': '0.4495158179045505', 'base_link_to_ins_y': '0.0',
                'base_link_to_ins_z': '0.65899', 'base_link_to_ins_roll': '0.0',
                'base_link_to_ins_pitch': '0.0', 'base_link_to_ins_yaw': '0.0',
            }.items()))
    else:
        if not _value(context, 'pointcloud_map'):
            raise RuntimeError('Universe localization-only mode requires pointcloud_map:=/absolute/path/GlobalMap.pcd')
        info = require_ndt_map(_value(context, 'pointcloud_map'))
        actions += [
            LogInfo(msg=f'Local NDT map: {info["path"]}. This launch supplies no geographic map transform.'),
            Node(package='autoware_map_loader', executable='autoware_pointcloud_map_loader',
                 namespace='map', name='pointcloud_map_loader', output='screen', parameters=[{
                     'pcd_paths_or_directory': [info['path']], 'pcd_metadata_path': '',
                     'enable_whole_load': True, 'enable_downsampled_whole_load': False,
                     'enable_partial_load': True, 'enable_selected_load': False, 'leaf_size': 3.0,
                 }], remappings=[
                     ('output/pointcloud_map', '/map/pointcloud_map'),
                     ('output/pointcloud_map_metadata', '/map/pointcloud_map_metadata'),
                     ('service/get_differential_pcd_map', '/map/get_differential_pointcloud_map'),
                     ('service/get_partial_pcd_map', '/map/get_partial_pointcloud_map'),
                 ]),
            Node(package='rclcpp_components', executable='component_container_mt',
                 name='pointcloud_container', output='screen'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(share / 'launch/components/scout_universe_localization.launch.py')),
                launch_arguments={
                    'input_pointcloud': _value(context, 'input_pointcloud'),
                    'imu_topic': _value(context, 'imu_topic'),
                    'wheel_odometry_topic': _value(context, 'wheel_odometry_topic'),
                    'initial_pose': _value(context, 'initial_pose'),
                    'start_imu_driver': _value(context, 'start_sensors'),
                    'start_wheel_odometry_driver': _value(context, 'start_wheel_odometry_driver'),
                    'can_interface': _value(context, 'can_interface'),
                    'require_can_feedback': _value(context, 'require_can_feedback'),
                }.items()),
            Node(package='scout_localization_support', executable='initial_pose_bridge',
                 name='scout_localization_initial_pose_bridge', output='screen'),
        ]
    if elevation:
        actions.append(GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/components/scout_local_elevation.launch.py')),
            launch_arguments={'pointcloud_map': _value(context, 'pointcloud_map'),
                              'elevation_storage_root': _value(context, 'elevation_storage_root')}.items())]))
    if traversability:
        actions.append(GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/components/scout_global_traversability.launch.py')),
            launch_arguments={'traversability_storage_root': _value(context, 'traversability_storage_root')}.items())]))
    if _flag(context, 'rviz'):
        actions.append(Node(package='rviz2', executable='rviz2', name='scout_localization_rviz',
                            arguments=['-d', str(share / 'rviz' / ('scout_localization_traversability.rviz' if traversability else ('scout_localization_elevation.rviz' if elevation else 'scout_localization.rviz')))], output='screen'))
    return actions


def generate_launch_description():
    defaults = {
        'localization_mode': 'universe', 'pointcloud_map': '', 'start_sensors': 'true',
        'start_wheel_odometry_driver': LaunchConfiguration('start_sensors'), 'can_interface': 'can10',
        'require_can_feedback': 'true', 'input_pointcloud': '/rslidar_points',
        'imu_topic': '/zed2i/zed_node/imu/data', 'wheel_odometry_topic': '/scout/wheel_odometry',
        'initial_pose': '[]', 'rviz': 'true', 'start_elevation': 'false',
        'elevation_storage_root': '/mnt/nvme0n1/autoware/scout_local_elevation_maps',
        'start_traversability': LaunchConfiguration('start_elevation'),
        'traversability_storage_root': '/mnt/nvme0n1/autoware/scout_traversability_maps',
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()],
        OpaqueFunction(function=_setup),
    ])
