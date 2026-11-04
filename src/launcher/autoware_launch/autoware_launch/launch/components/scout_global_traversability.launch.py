"""Incremental HT recognition and display; consumes the elevation session."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share=Path(get_package_share_directory('scout_global_traversability'))
    return LaunchDescription([
        DeclareLaunchArgument('traversability_config', default_value=str(share/'config/global_traversability.yaml')),
        DeclareLaunchArgument('traversability_storage_root', default_value='/mnt/nvme0n1/autoware/scout_traversability_maps'),
        Node(package='scout_global_traversability', executable='global_traversability',
             name='scout_global_traversability', output='screen',
             parameters=[LaunchConfiguration('traversability_config'),
                         {'storage_root':LaunchConfiguration('traversability_storage_root')}],
             sigterm_timeout='20', sigkill_timeout='5'),
    ])
