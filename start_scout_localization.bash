#!/usr/bin/env bash
set -eo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_dir/setup_env.bash"
source "$repo_dir/scripts/scout_elevation_runtime_env.bash"
SCOUT_LOCALIZATION_MODE="$(python3 "$repo_dir/scripts/check_scout_localization_start.py" --standalone "$@")"
export SCOUT_LOCALIZATION_MODE
# Prepare the passive wheel input before its reader and CAN freshness gate start.
_scout_can_interface=can10
_scout_start_sensors=true
_scout_start_wheel=
_scout_require_can=true
_scout_can_dry=false
_scout_sim_time=false
for _scout_arg in "$@"; do
  case "$_scout_arg" in
    can_interface:=*) _scout_can_interface="${_scout_arg#*:=}" ;;
    start_sensors:=*) _scout_start_sensors="${_scout_arg#*:=}" ;;
    start_wheel_odometry_driver:=*) _scout_start_wheel="${_scout_arg#*:=}" ;;
    require_can_feedback:=*) _scout_require_can="${_scout_arg#*:=}" ;;
    use_sim_time:=*) _scout_sim_time="${_scout_arg#*:=}" ;;
    --show-args|--show-arguments|-s|--print-description|-p|--help|-h) _scout_can_dry=true ;;
  esac
done
_scout_start_wheel="${_scout_start_wheel:-$_scout_start_sensors}"
if [[ "$SCOUT_LOCALIZATION_MODE" == universe && "$_scout_can_dry" == false &&
      "${_scout_start_wheel,,}" =~ ^(true|1)$ &&
      "${_scout_require_can,,}" =~ ^(true|1)$ &&
      ! "${_scout_sim_time,,}" =~ ^(true|1)$ ]]; then
  bash "$repo_dir/scripts/setup_scout_can.bash" --interface "$_scout_can_interface"
fi
unset _scout_can_interface _scout_start_sensors _scout_start_wheel _scout_require_can _scout_can_dry _scout_sim_time _scout_arg

exec ros2 launch autoware_launch scout_localization.launch.py "$@"
