"""Scout sensor adapters around the installed native Universe localization stack."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, SetLaunchConfiguration
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _enabled(context, name):
    value = LaunchConfiguration(name).perform(context).lower()
    if value not in ('true', 'false', '1', '0'):
        raise RuntimeError(f'{name} must be true or false, got {value!r}')
    return value in ('true', '1')


def _setup(context):
    share = Path(get_package_share_directory('autoware_launch'))
    actions = []
    if _enabled(context, 'start_imu_driver'):
        zed = Path(get_package_share_directory('zed_wrapper'))
        actions.append(GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(zed / 'launch/zed_camera.launch.py')),
            launch_arguments={
                'camera_model': 'zed2i', 'camera_name': 'zed2i',
                'ros_params_override_path': str(share / 'config/localization/scout_zed2i_imu.param.yaml'),
                'publish_tf': 'false', 'publish_map_tf': 'false', 'publish_imu_tf': 'true',
            }.items())]))
    if _enabled(context, 'publish_zed_mount_tf'):
        actions.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='scout_localization_zed_mount_tf', output='screen',
            arguments=['--x', '0', '--y', '0', '--z', '0', '--roll', '0', '--pitch', '0',
                       '--yaw', '0', '--frame-id', 'zed2i_link', '--child-frame-id', 'zed2i_camera_link']))
    if _enabled(context, 'start_wheel_odometry_driver'):
        actions.append(Node(
            package='scout_base', executable='scout_base_node', name='scout_localization_wheel_reader',
            output='screen', parameters=[{
                'port_name': LaunchConfiguration('can_interface'), 'read_only': True,
                'auto_reconnect': False, 'odom_frame': 'scout_wheel_odom',
                'base_frame': 'scout_chassis_link', 'odom_topic_name': '/scout/wheel_odometry',
            }], remappings=[('/scout_status', '/scout/status'), ('/tf', '/scout/wheel_tf')]))
    actions.append(Node(
        package='scout_localization_support', executable='wheel_odometry_to_twist',
        name='scout_localization_wheel_twist', output='screen', parameters=[{
            'input_topic': LaunchConfiguration('wheel_odometry_topic'),
            'output_topic': '/sensing/vehicle_velocity_converter/twist_with_covariance',
            'use_sim_time': _enabled(context, 'use_sim_time'),
            'require_can_feedback': _enabled(context, 'require_can_feedback'),
            'can_interface': LaunchConfiguration('can_interface'),
        }]))
    actions.extend([
        Node(package='scout_localization_support', executable='imu_relay',
             name='scout_localization_imu_relay', output='screen', parameters=[{
                 'input_topic': LaunchConfiguration('imu_topic'),
                 'output_topic': '/scout/localization/imu',
                 'use_sim_time': _enabled(context, 'use_sim_time'),
             }]),
        Node(package='scout_localization_support', executable='pointcloud_time_order',
             name='scout_localization_point_time_order', output='screen', parameters=[{
                 'input_topic': LaunchConfiguration('input_pointcloud'),
                 'output_topic': '/scout/localization/points_ordered',
                 'use_sim_time': _enabled(context, 'use_sim_time'),
             }]),
        Node(package='autoware_pointcloud_preprocessor', executable='distortion_corrector_node',
             name='scout_localization_distortion_corrector', output='screen', parameters=[{
                 'base_frame': 'base_link', 'use_imu': True,
                 'use_3d_distortion_correction': False, 'update_azimuth_and_distance': False,
                 'processing_time_threshold_sec': 0.01,
                 'timestamp_mismatch_fraction_threshold': 0.01,
                 'use_sim_time': _enabled(context, 'use_sim_time'),
             }], remappings=[
                 ('~/input/pointcloud', '/scout/localization/points_ordered'),
                 ('~/input/twist', '/localization/twist_estimator/twist_with_covariance'),
                 ('~/input/imu', '/scout/localization/imu'),
                 ('~/output/pointcloud', '/scout/localization/points_deskewed'),
             ]),
    ])
    # DeclareLaunchArgument in the native child launches respects inherited values.
    # Use the input argument, rather than chained ROS remaps, to select this IMU.
    actions.append(GroupAction(actions=[
        SetLaunchConfiguration('input_imu_topic', '/scout/localization/imu'),
        SetLaunchConfiguration('gnss_enabled', 'false'),
        SetLaunchConfiguration('system_run_mode', 'online'),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(share / 'launch/components/tier4_localization_component.launch.xml')),
            launch_arguments={
                'pose_source': 'ndt', 'twist_source': 'gyro_odom',
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'vehicle_model': LaunchConfiguration('vehicle_model'),
                'input_pointcloud': '/scout/localization/points_deskewed',
                'initial_pose': LaunchConfiguration('initial_pose'),
                'launch_pointcloud_container': 'false',
                'localization_pointcloud_container_name': LaunchConfiguration('pointcloud_container_name'),
            }.items()),
    ]))
    return actions


def generate_launch_description():
    defaults = {
        'use_sim_time': 'false', 'vehicle_model': 'scout_vehicle',
        'input_pointcloud': '/rslidar_points',
        'pointcloud_container_name': '/pointcloud_container',
        'imu_topic': '/zed2i/zed_node/imu/data', 'wheel_odometry_topic': '/scout/wheel_odometry',
        'initial_pose': '[]', 'start_imu_driver': 'false',
        'start_wheel_odometry_driver': 'false', 'publish_zed_mount_tf': 'true',
        'can_interface': 'can10', 'require_can_feedback': 'true',
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()],
        OpaqueFunction(function=_setup),
    ])
