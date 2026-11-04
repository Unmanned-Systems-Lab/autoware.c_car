"""Attach tiled recognition to an existing localization/elevation session."""
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,GroupAction,IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share=Path(get_package_share_directory('autoware_launch'))
    return LaunchDescription([
        DeclareLaunchArgument('traversability_storage_root',default_value='/mnt/nvme0n1/autoware/scout_traversability_maps'),
        DeclareLaunchArgument('rviz',default_value='true'),
        GroupAction(scoped=True,actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share/'launch/components/scout_global_traversability.launch.py')),
            launch_arguments={'traversability_storage_root':LaunchConfiguration('traversability_storage_root')}.items())]),
        Node(package='rviz2',executable='rviz2',name='scout_traversability_rviz',output='screen',
             arguments=['-d',str(share/'rviz/scout_localization_traversability.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
