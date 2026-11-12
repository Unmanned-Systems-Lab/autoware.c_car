"""Reject a legacy option whose executable and parameter file were not provided."""
from launch import LaunchDescription
from launch.actions import OpaqueFunction


def reject_legacy_dummy(context, *args, **kwargs):
    raise RuntimeError(
        "launch_planning_dummy_publishers is unavailable in c_car: its legacy "
        "executable and config were not supplied. Use planning_simulator.launch.xml "
        "for simulated perception, or configure the actual perception pipeline.")


def generate_launch_description():
    return LaunchDescription([OpaqueFunction(function=reject_legacy_dummy)])
