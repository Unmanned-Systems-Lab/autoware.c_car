"""Compose the c_car receiver and optional quality-gated localization bridge."""
import math

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def parse_bool(text):
    value = text.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Expected true/false, got {text!r}")


def finite_float(text):
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("INS extrinsic override must be finite")
    return value


BRIDGE_OVERRIDES = {
    "input_odom_topic": str,
    "input_quality_topic": str,
    "input_child_frame_id": str,
    "output_base_link_odom_topic": str,
    "output_odom_topic": str,
    "output_accel_topic": str,
    "output_pose_topic": str,
    "output_initialization_state_topic": str,
    "output_frame_id": str,
    "output_child_frame_id": str,
    "publish_tf": parse_bool,
    "base_link_to_ins_x": finite_float,
    "base_link_to_ins_y": finite_float,
    "base_link_to_ins_z": finite_float,
    "base_link_to_ins_roll": finite_float,
    "base_link_to_ins_pitch": finite_float,
    "base_link_to_ins_yaw": finite_float,
}


def launch_setup(context, *args, **kwargs):
    def value(name):
        return LaunchConfiguration(name).perform(context).strip()

    params_file = value("ins_driver_param_path")
    if not params_file:
        raise ValueError("ins_driver_param_path must identify the shared driver/bridge YAML")
    sim_time = parse_bool(value("use_sim_time"))
    actions = []
    if parse_bool(value("launch_ins_driver")):
        driver_arguments = {"params_file": params_file, "use_sim_time": str(sim_time).lower()}
        for source, target in (("ins_driver_can_interface", "can_interface"),
                               ("map_projector_info_path", "map_projector_info_path"),
                               ("input_quality_topic", "quality_topic"),
                               ("input_odom_topic", "topic_name")):
            text = value(source)
            if text:
                driver_arguments[target] = text
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare("ins_driver_c_car"), "launch", "ins_driver.launch.py"])),
            launch_arguments=driver_arguments.items()))
    if parse_bool(value("launch_ins_localization_bridge")):
        overrides = {"use_sim_time": sim_time}
        for name, convert in BRIDGE_OVERRIDES.items():
            text = value(name)
            if text:
                overrides[name] = convert(text)
        actions.append(Node(package="ins_driver_c_car", executable="ins_localization_bridge",
                            name="ins_localization_bridge", output="screen",
                            parameters=[params_file, overrides]))
    return actions


def generate_launch_description():
    params_file = PathJoinSubstitution([FindPackageShare("ins_driver_c_car"), "config", "driver.yaml"])
    arguments = [
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("launch_ins_driver", default_value="true"),
        DeclareLaunchArgument("launch_ins_localization_bridge", default_value="false",
                              description="Enable only with verified quality, axes, extrinsics and map origin."),
        DeclareLaunchArgument("ins_driver_param_path", default_value=params_file,
                              description="Shared YAML containing ins_driver_c_car and ins_localization_bridge."),
        DeclareLaunchArgument("ins_driver_can_interface", default_value="",
                              description="Optional override; empty preserves YAML (shipped default can0)."),
        DeclareLaunchArgument("map_projector_info_path", default_value=""),
    ]
    arguments.extend(DeclareLaunchArgument(name, default_value="",
                     description="Optional override; empty preserves calibrated YAML.")
                     for name in BRIDGE_OVERRIDES)
    return LaunchDescription(arguments + [OpaqueFunction(function=launch_setup)])
