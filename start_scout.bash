#!/usr/bin/env bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./setup_env.bash
SCOUT_LOCALIZATION_MODE="$(python3 scripts/check_scout_localization_start.py "$@")"
for _scout_argument in "$@"; do
  case "$_scout_argument" in pointcloud_map_file:=*) export AUTOWARE_UI_POINTCLOUD_MAP_FILE="${_scout_argument#pointcloud_map_file:=}" ;; esac
done
unset _scout_argument
export SCOUT_LOCALIZATION_MODE
exec ros2 launch autoware_launch real_scout.launch.xml "$@"
