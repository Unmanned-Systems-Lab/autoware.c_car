#!/usr/bin/env bash
set -eo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"
source ./setup_env.bash
source ./scripts/scout_elevation_runtime_env.bash
jobs="${SCOUT_LOCALIZATION_BUILD_JOBS:-2}"
if [[ ! "$jobs" =~ ^[1-9][0-9]*$ ]]; then echo 'SCOUT_LOCALIZATION_BUILD_JOBS 必须是正整数。' >&2; exit 2; fi
if [[ ! -f "$ZED_DIR/include/sl/Camera.hpp" || ! -f "$ZED_DIR/lib/libsl_zed.so" ]]; then
  echo '请先运行 bash scripts/install_scout_zed_sdk.bash 安装官方 ZED SDK 开发包。' >&2
  exit 2
fi
export CMAKE_BUILD_PARALLEL_LEVEL="$jobs" MAKEFLAGS="-j$jobs"
colcon build --packages-select zed_msgs zed_description zed_components zed_wrapper \
  ugv_sdk scout_base elevation_map_msgs ros2_numpy elevation_mapping_cupy \
  scout_localization_support scout_global_traversability autoware_launch \
  --symlink-install --executor sequential --cmake-clean-cache \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DZED_DIR="$ZED_DIR"
source ./setup_env.bash
for package in autoware_ndt_scan_matcher autoware_gyro_odometer autoware_ekf_localizer \
  autoware_pose_initializer autoware_map_loader autoware_pointcloud_preprocessor tier4_localization_launch; do
  if ! ros2 pkg prefix "$package" >/dev/null; then
    echo "缺少 Autoware 核心定位包 $package，请先按主工程说明编译其源码。" >&2
    exit 1
  fi
done
python3 -c 'import cupy, torch; print("Elevation CUDA device count:", cupy.cuda.runtime.getDeviceCount())'
echo 'Scout 定位、高程入口和传感器支持包已编译。'
