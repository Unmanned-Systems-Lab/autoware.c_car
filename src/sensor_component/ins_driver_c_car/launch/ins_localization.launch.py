"""Receiver plus an explicitly enabled quality-gated Autoware bridge."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("ins_driver_c_car")
    default_params = PathJoinSubstitution([share, "config", "driver.yaml"])
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    return LaunchDescription([
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="Parameters for receiver and localization bridge."),
        DeclareLaunchArgument("can_interface", default_value="",
                              description="Empty keeps params_file; shipped default is can0."),
        DeclareLaunchArgument("enable_bridge", default_value="false",
                              description="Start bridge only after quality and extrinsics are verified."),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([share, "launch", "ins_driver.launch.py"])),
            launch_arguments={"params_file": params_file,
                              "can_interface": LaunchConfiguration("can_interface"),
                              "use_sim_time": use_sim_time}.items()),
        Node(package="ins_driver_c_car", executable="ins_localization_bridge",
             name="ins_localization_bridge", output="screen",
             condition=IfCondition(LaunchConfiguration("enable_bridge")),
             parameters=[params_file, {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)}]),
    ])