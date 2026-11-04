"""Select exactly one localization backend for the current launch session."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


INS_DEFAULTS = {
    'use_sim_time': 'false', 'launch_ins_driver': 'false', 'ins_driver_param_path': '',
    'ins_driver_can_interface': 'can0', 'map_projector_info_path': '',
    'input_odom_topic': '/odom_ins_at_base_link', 'output_base_link_odom_topic': '',
    'output_frame_id': 'map', 'output_child_frame_id': 'base_link',
    'base_link_to_ins_x': '0.0', 'base_link_to_ins_y': '0.0', 'base_link_to_ins_z': '0.0',
    'base_link_to_ins_roll': '0.0', 'base_link_to_ins_pitch': '0.0', 'base_link_to_ins_yaw': '0.0',
}


def _setup(context):
    mode = LaunchConfiguration('localization_mode').perform(context)
    share = Path(get_package_share_directory('autoware_launch'))
    if mode == 'fixposition':
        return [
            LogInfo(msg='Scout localization backend: fixposition'),
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(str(share / 'launch/components/ins_localization_component.launch.xml')),
                launch_arguments={key: LaunchConfiguration(key) for key in INS_DEFAULTS}.items()),
        ]
    if mode != 'universe':
        raise RuntimeError('localization_mode must be fixposition or universe')
    from scout_localization_support.map_checks import require_ndt_map
    map_dir = Path(LaunchConfiguration('map_path').perform(context)).expanduser()
    map_file = LaunchConfiguration('pointcloud_map_file').perform(context)
    info = require_ndt_map(map_dir / map_file)
    return [
        LogInfo(msg=f'Scout localization backend: universe (NDT map: {info["path"]}, {info["points"]} points)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/components/scout_universe_localization.launch.py')),
            launch_arguments={
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'vehicle_model': LaunchConfiguration('vehicle_model'),
                'pointcloud_container_name': LaunchConfiguration('localization_pointcloud_container_name'),
                'input_pointcloud': '/rslidar_points',
                'imu_topic': LaunchConfiguration('universe_imu_topic'),
                'wheel_odometry_topic': LaunchConfiguration('universe_wheel_odometry_topic'),
                'initial_pose': LaunchConfiguration('universe_initial_pose'),
                'start_imu_driver': LaunchConfiguration('localization_start_imu_driver'),
                'start_wheel_odometry_driver': LaunchConfiguration('localization_start_wheel_odometry_driver'),
                'can_interface': LaunchConfiguration('localization_wheel_can_interface'),
            }.items()),
    ]


def generate_launch_description():
    defaults = {
        **INS_DEFAULTS, 'localization_mode': 'fixposition', 'map_path': '',
        'pointcloud_map_file': 'pointcloud_map.pcd', 'vehicle_model': 'scout_vehicle',
        'localization_pointcloud_container_name': '/pointcloud_container',
        'universe_imu_topic': '/zed2i/zed_node/imu/data',
        'universe_wheel_odometry_topic': '/scout/wheel_odometry', 'universe_initial_pose': '[]',
        'localization_start_imu_driver': 'false', 'localization_start_wheel_odometry_driver': 'false',
        'localization_wheel_can_interface': 'can10',
    }
    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items()],
        OpaqueFunction(function=_setup),
    ])
