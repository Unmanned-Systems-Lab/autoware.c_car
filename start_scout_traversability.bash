#!/usr/bin/env bash
set -eo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$repo_dir/setup_env.bash"
source "$repo_dir/scripts/scout_elevation_runtime_env.bash"
exec ros2 launch autoware_launch scout_traversability.launch.py "$@"
