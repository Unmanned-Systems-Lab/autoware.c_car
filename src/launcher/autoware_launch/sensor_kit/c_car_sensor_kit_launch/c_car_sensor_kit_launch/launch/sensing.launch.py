"""A resource-only sensor-kit entry point, never a physical sensing driver."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _validate_mode(context):
    value = LaunchConfiguration("launch_driver").perform(context).strip().lower()
    if value not in ("true", "false", "1", "0"):
        raise RuntimeError("c_car launch_driver must be true or false")
    if value in ("true", "1"):
        raise RuntimeError(
            "c_car physical sensing is not configured. This sensor-kit package "
            "only supplies planning-simulation launch and diagnostic resources; "
            "configure real sensors and calibration before requesting drivers.")
    return [LogInfo(msg="c_car simulation sensor resources: no physical sensing nodes launched.")]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("launch_driver", default_value="true"),
        OpaqueFunction(function=_validate_mode),
    ])
