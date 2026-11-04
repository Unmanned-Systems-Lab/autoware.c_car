#!/usr/bin/env bash
# Source after the ROS/workspace setup; expose CUDA/CUPTI from this machine.
# This file never sources another project workspace or changes ROS settings.
# Small per-point 3D operations must not fan out across all Jetson CPU cores.
export OPENBLAS_NUM_THREADS="${SCOUT_ELEVATION_BLAS_THREADS:-1}"

_scout_elevation_prepend_library_dir() {
  [[ -d "$1" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$1:"*) ;;
    *) export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

_scout_elevation_cupti_dir="$(python3 - <<'PY_CUPTI'
from pathlib import Path
import site
roots = []
try:
    roots.extend(site.getsitepackages())
except AttributeError:
    pass
roots.append(site.getusersitepackages())
for root in roots:
    candidate = Path(root) / "nvidia" / "cuda_cupti" / "lib"
    if candidate.is_dir():
        print(candidate)
        break
PY_CUPTI
)"
if [[ -n "${_scout_elevation_cupti_dir}" ]]; then
  _scout_elevation_prepend_library_dir "${_scout_elevation_cupti_dir}"
fi

for _scout_elevation_cuda_root in "${SCOUT_ELEVATION_CUDA_ROOT:-}" "${CUDA_HOME:-}" /usr/local/cuda /usr/local/cuda-*; do
  [[ -n "${_scout_elevation_cuda_root}" ]] || continue
  if [[ -d "${_scout_elevation_cuda_root}/lib64" ]]; then
    _scout_elevation_prepend_library_dir "${_scout_elevation_cuda_root}/lib64"
    _scout_elevation_prepend_library_dir "${_scout_elevation_cuda_root}/extras/CUPTI/lib64"
    break
  fi
done

unset _scout_elevation_cupti_dir _scout_elevation_cuda_root
unset -f _scout_elevation_prepend_library_dir
