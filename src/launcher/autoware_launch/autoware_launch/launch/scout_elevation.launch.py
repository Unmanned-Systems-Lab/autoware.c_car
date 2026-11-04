"""Elevation-only attachment. Does not start localization, drivers or vehicle control."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('autoware_launch'))
    return LaunchDescription([
        DeclareLaunchArgument('pointcloud_map', description='Same PCD as the active Universe localization'),
        DeclareLaunchArgument('elevation_storage_root', default_value='/mnt/nvme0n1/autoware/scout_local_elevation_maps'),
        DeclareLaunchArgument('rviz', default_value='true'),
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share/'launch/components/scout_local_elevation.launch.py')),
            launch_arguments={
                'pointcloud_map': LaunchConfiguration('pointcloud_map'),
                'elevation_storage_root': LaunchConfiguration('elevation_storage_root'),
            }.items())]),
        Node(package='rviz2', executable='rviz2', name='scout_elevation_rviz', output='screen',
             arguments=['-d',str(share/'rviz/scout_localization_elevation.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
