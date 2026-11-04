"""Attach guarded elevation mapping to an already running Universe localization."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context):
    from scout_localization_support.map_checks import require_ndt_map
    value = lambda name: LaunchConfiguration(name).perform(context)
    reference = require_ndt_map(value('pointcloud_map'))['path']
    return [
        Node(package='scout_localization_support', executable='elevation_input_gate',
             name='scout_elevation_input_gate', output='screen',
             parameters=[{'reference_pcd': reference}]),
        Node(package='scout_localization_support', executable='local_elevation_manager',
             name='scout_local_elevation_manager', output='screen',
             sigterm_timeout='30', sigkill_timeout='10',
             parameters=[{'reference_pcd': reference,
                          'mapping_config_file': value('elevation_config'),
                          'storage_root': value('elevation_storage_root')}]),
    ]


def generate_launch_description():
    share = Path(get_package_share_directory('autoware_launch'))
    return LaunchDescription([
        DeclareLaunchArgument('pointcloud_map', description='Exact PCD loaded by the running Universe localization'),
        DeclareLaunchArgument('elevation_config', default_value=str(share/'config/elevation/scout_local_elevation.param.yaml')),
        DeclareLaunchArgument('elevation_storage_root', default_value='/mnt/nvme0n1/autoware/scout_local_elevation_maps'),
        OpaqueFunction(function=_setup),
    ])
