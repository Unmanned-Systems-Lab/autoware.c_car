# Source this file to select this checkout.
_scout_ws="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$_scout_ws/install/setup.bash"
export PATH="/usr/local/cuda/bin:$PATH"
export CMAKE_PREFIX_PATH="$_scout_ws/.deps/usr/local:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$_scout_ws/.deps/usr/local/lib:$_scout_ws/install/acados/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export ACADOS_SOURCE_DIR="$_scout_ws/src/universe/external/acados"
_scout_cupti="$HOME/.local/lib/python3.10/site-packages/nvidia/cuda_cupti/lib"
if [ -d "$_scout_cupti" ]; then
  export LD_LIBRARY_PATH="$_scout_cupti:$LD_LIBRARY_PATH"
fi
unset _scout_cupti _scout_ws

# Defaults for local Scout sensor/localization traffic; explicit user settings win.
if [ -z "${RMW_IMPLEMENTATION:-}" ]; then export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; fi
if [ "$RMW_IMPLEMENTATION" = rmw_cyclonedds_cpp ] && [ "${CYCLONEDDS_URI+x}" != x ]; then
  export CYCLONEDDS_URI="file://$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/config/scout_localization_cyclonedds.xml"
fi
export ZED_DIR="${ZED_DIR:-/usr/local/zed}"
if [ -d "$ZED_DIR/lib" ]; then export LD_LIBRARY_PATH="$ZED_DIR/lib:${LD_LIBRARY_PATH:-}"; fi
