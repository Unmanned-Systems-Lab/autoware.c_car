from setuptools import find_packages, setup

package_name = "scout_localization_support"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Unmanned Systems Lab",
    maintainer_email="maintainers@unmanned-systems-lab.invalid",
    description="Validated Scout wheel velocity adapter for Autoware localization",
    license="BSD-3-Clause",
    entry_points={"console_scripts": [
        "wheel_odometry_to_twist = scout_localization_support.wheel_odometry_to_twist:main",
        "initial_pose_bridge = scout_localization_support.initial_pose_bridge:main",
        "imu_relay = scout_localization_support.imu_relay:main",
        "elevation_input_gate = scout_localization_support.elevation_input_gate:main",
        "local_elevation_manager = scout_localization_support.local_elevation_manager:main",
        "pointcloud_time_order = scout_localization_support.pointcloud_time_order:main",
    ]},
)
