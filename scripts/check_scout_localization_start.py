#!/usr/bin/env python3
"""Validate the selected session before any launch process or driver is started."""
import os
import re
import subprocess
import sys
from pathlib import Path


def check(arguments, standalone=False):
    options = dict(arg.split(':=', 1) for arg in arguments if ':=' in arg)
    mode = options.get('localization_mode', 'universe' if standalone else 'fixposition')
    if mode not in ('fixposition', 'universe'):
        raise ValueError('localization_mode must be fixposition or universe')
    dry = any(arg in ['--show-args', '--show-arguments', '-s', '--print-description', '-p', '--help', '-h'] for arg in arguments)
    for key in ['start_sensors', 'start_sensor_drivers', 'start_wheel_odometry_driver', 'localization_start_imu_driver', 'localization_start_wheel_odometry_driver', 'launch_vehicle_interface', 'require_can_feedback', 'start_elevation', 'start_traversability']:
        if key in options and options[key].lower() not in ('true', 'false', '1', '0'):
            raise ValueError(key + ' must be true or false')
    if options.get('start_elevation', 'false').lower() in ('true', '1') and mode != 'universe':
        raise ValueError('start_elevation requires localization_mode:=universe')
    if not dry:
        if mode == 'universe':
            from scout_localization_support.map_checks import require_ndt_map
            if standalone:
                if not options.get('pointcloud_map'):
                    raise ValueError('Supply pointcloud_map:=/absolute/path/GlobalMap.pcd')
                require_ndt_map(options['pointcloud_map'])
            else:
                if not options.get('map_path'):
                    raise ValueError('Supply map_path:=<aligned 3D pointcloud + Lanelet map directory>')
                filename = Path(options.get('pointcloud_map_file', 'pointcloud_map.pcd'))
                if filename.is_absolute() or '..' in filename.parts:
                    raise ValueError('pointcloud_map_file must be a filename within map_path')
                require_ndt_map(Path(options['map_path']) / filename)
                for required in ('lanelet2_map.osm', 'map_projector_info.yaml'):
                    if not (Path(options['map_path']) / required).is_file():
                        raise ValueError('Full Autoware requires ' + required + '; use start_scout_localization.bash for a local PCD-only map')
        # LIO publishes a separate dynamic parent for base_link and owns USB/UDP.
        for proc in Path('/proc').glob('[0-9]*/exe'):
            try:
                if proc.resolve().name == 'lio_sam_imuPreintegration':
                    raise ValueError('LIO-SAM is running. Save its map and stop its launch before starting Autoware localization.')
            except OSError:
                pass
        result = subprocess.run(['ros2', 'node', 'list', '--no-daemon', '--spin-time', '2'],
                                capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise ValueError('ROS graph discovery failed; check the DDS configuration before starting localization')
        names = [name.strip() for name in result.stdout.splitlines()]
        if any(name.rsplit('/', 1)[-1] in ('ins_localization_bridge', 'ekf_localizer', 'ndt_scan_matcher') for name in names):
            raise ValueError('An existing localization backend is running. Stop its launch before switching startup mode.')
        sensors = options.get('start_sensors' if standalone else 'start_sensor_drivers', 'true').lower() in ('true', '1')
        if sensors:
            ports = subprocess.check_output(['ss', '-H', '-lun'], text=True)
            if re.search(r':(?:3344|5566)\s', ports):
                raise ValueError('Lidar UDP ports 3344/5566 are in use; stop the existing sensor launch or explicitly reuse sensors')
            if '/zed2i/zed_node' in names or '/rslidar_sdk' in names:
                raise ValueError('Sensor drivers are already running; stop them or explicitly reuse sensors')
    return mode, options.get('pointcloud_map_file', 'pointcloud_map.pcd')


if __name__ == '__main__':
    standalone = '--standalone' in sys.argv[1:]
    try:
        mode, filename = check([a for a in sys.argv[1:] if a != '--standalone'], standalone)
        print(mode)
    except (ValueError, OSError, ImportError, subprocess.SubprocessError) as error:
        print('Scout localization preflight failed: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
