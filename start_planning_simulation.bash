#!/usr/bin/env bash
set -eo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base_workspace="${AUTOWARE_BASE_WS:-$HOME/autoware.repos_workspace}"
workspace="${AUTOWARE_WS:-$HOME/c_car_workspace}"
map_path="${MAP_PATH:-${1:-$repo_dir/maps/planning_simulation/ring}}"
for setup_file in "$base_workspace/install/setup.bash" "$workspace/install/setup.bash"; do
  if [ ! -f "$setup_file" ]; then
    echo "Build workspace not found: $setup_file" >&2
    exit 1
  fi
done
if [ -f "$HOME/autoware.c_car_dependencies/setup.bash" ]; then
  source "$HOME/autoware.c_car_dependencies/setup.bash"
else
  source /opt/ros/humble/setup.bash
fi
# Load the existing Autoware build first, then the c_car packages.
source "$base_workspace/install/setup.bash"
source "$workspace/install/setup.bash"
for map_file in lanelet2_map.osm pointcloud_map.pcd map_projector_info.yaml; do
  if [ ! -f "$map_path/$map_file" ]; then
    echo "Map file missing: $map_path/$map_file" >&2
    exit 1
  fi
done
ulimit -c 0
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-188}"
export ROS_LOCALHOST_ONLY=1
export CYCLONEDDS_URI="file://$repo_dir/config/planning_simulation/cyclonedds-local.xml"
vehicle_model="${VEHICLE_MODEL:-c_car_vehicle}"
sensor_model="${SENSOR_MODEL:-c_car_sensor_kit}"
# The simulator drives the software model; its joint bridge only animates RViz.
exec ros2 launch autoware_launch planning_simulator.launch.xml \
  map_path:="$map_path" vehicle_model:="$vehicle_model" sensor_model:="$sensor_model" \
  vehicle_simulation:=true initial_engage_state:=true enable_all_modules_auto_mode:=false \
  scenario_simulation:=false launch_freespace_bypass_coordinator:=false \
  rviz:="${RVIZ:-true}" rviz_config:="$repo_dir/config/planning_simulation/c_car.rviz"
