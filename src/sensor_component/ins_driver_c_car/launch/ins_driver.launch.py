"""Start only the passive GI5651 receiver, with no localization fallback nodes."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
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


# Arrays and calibration values can be supplied in params_file without shell quoting.
OVERRIDES = {
    "can_interface": str,
    "navigation_source": str,
    "field_timeout_sec": float,
    "max_group_skew_sec": float,
    "allow_missing_roll": parse_bool,
    "axes_verified": parse_bool,
    "heading_convention": str,
    "acceleration_unit": str,
    "acceleration_units_verified": parse_bool,
    "time_source": str,
    "gps_utc_leap_seconds": int,
    "gps_rollover_weeks": int,
    "max_device_time_offset_sec": float,
    "time_reference_topic": str,
    "topic_name": str,
    "frame_id": str,
    "child_frame_id": str,
    "quality_topic": str,
    "publish_raw_nav_sat_fix": parse_bool,
    "raw_nav_sat_fix_topic": str,
    "raw_nav_sat_fix_frame_id": str,
    "publish_imu": parse_bool,
    "imu_topic": str,
    "imu_frame_id": str,
    "map_projector_info_path": str,
    "reference_origin_active": parse_bool,
    "reference_origin_latitude_deg": float,
    "reference_origin_longitude_deg": float,
    "reference_origin_altitude_m": float,
    "socket_timeout_sec": float,
    "log_enabled": parse_bool,
    "log_path": str,
    "log_name": str,
    "log_name_format": str,
    "log_format": str,
    "use_sim_time": parse_bool,
}


def launch_setup(context, *args, **kwargs):
    overrides = {}
    for name, convert in OVERRIDES.items():
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            overrides[name] = convert(value)
    parameters = [LaunchConfiguration("params_file").perform(context)]
    if overrides:
        parameters.append(overrides)
    return [Node(package="ins_driver_c_car", executable="ins_driver_node",
                 name="ins_driver_c_car", output="screen", parameters=parameters)]


def generate_launch_description():
    params = PathJoinSubstitution([FindPackageShare("ins_driver_c_car"), "config", "driver.yaml"])
    arguments = [DeclareLaunchArgument("params_file", default_value=params,
                 description="ROS parameter YAML, including explicit device calibration.")]
    arguments.extend(DeclareLaunchArgument(name, default_value="",
                     description=f"Optional {name} override; empty keeps params_file.")
                     for name in OVERRIDES)
    return LaunchDescription(arguments + [OpaqueFunction(function=launch_setup)])
