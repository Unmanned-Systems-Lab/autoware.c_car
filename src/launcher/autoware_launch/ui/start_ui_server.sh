#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UI_SCRIPT="${SCRIPT_DIR}/generate_autoware_map.py"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORKSPACE_SETUP="${WORKSPACE_ROOT}/install/setup.bash"
WORKSPACE_INSTALL_PREFIX="${WORKSPACE_ROOT}/install"
HOST="${1:-127.0.0.1}"
PORT="${2:-8090}"
OPEN_INDEX="${3:-false}"
RUNTIME_MAP_DIR="${4:-}"
SERVER_PID=""

if [[ -n "${RUNTIME_MAP_DIR}" ]]; then
  export AUTOWARE_UI_RUNTIME_MAP_DIR="${RUNTIME_MAP_DIR}"
fi

healthcheck_url() {
  echo "http://$(browser_host):${PORT}/health"
}

normalize_bool() {
  local value="${1,,}"
  case "${value}" in
    1|true|yes|on)
      echo "true"
      ;;
    0|false|no|off)
      echo "false"
      ;;
    *)
      echo "false"
      ;;
  esac
}

browser_host() {
  case "${HOST}" in
    0.0.0.0|::|[::])
      echo "127.0.0.1"
      ;;
    *)
      echo "${HOST}"
      ;;
  esac
}

open_index_page() {
  if [[ "$(normalize_bool "${OPEN_INDEX}")" != "true" ]]; then
    return 0
  fi

  if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is not set. Skipping automatic browser open."
    return 0
  fi

  if ! command -v xdg-open >/dev/null 2>&1; then
    echo "xdg-open is not available. Skipping automatic browser open."
    return 0
  fi

  local url="http://$(browser_host):${PORT}/index.html"
  xdg-open "${url}" >/dev/null 2>&1 &
  echo "Opened UI page: ${url}"
}

wait_for_server() {
  local url
  url="$(healthcheck_url)"
  local _=0
  for _ in $(seq 1 40); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  echo "UI server did not become ready in time: ${url}"
  return 1
}

server_is_healthy() {
  curl -fsS --max-time 1 "$(healthcheck_url)" >/dev/null 2>&1
}

require_matching_server_profile() {
  local status_url="http://$(browser_host):${PORT}/autoware_status"
  python3 - "${status_url}" <<'PY_PROFILE'
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def reject(reason):
    print("Existing UI cannot be reused: " + reason, file=sys.stderr)
    print("Please stop the old UI on this port, then restart with the requested localization profile.", file=sys.stderr)
    raise SystemExit(1)


backend = os.environ.get("SCOUT_LOCALIZATION_MODE", "fixposition").strip().lower()
if backend not in {"fixposition", "universe"}:
    reject("SCOUT_LOCALIZATION_MODE must be fixposition or universe")
raw_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
expected = {
    "localization_backend": backend,
    "runtime_map_dir": str(Path(raw_map_dir).expanduser().resolve()) if raw_map_dir else "",
    "pointcloud_map_file": os.environ.get("AUTOWARE_UI_POINTCLOUD_MAP_FILE", "pointcloud_map.pcd"),
    "ui_actions_enabled": os.environ.get("SCOUT_UI_ALLOW_VEHICLE_ACTIONS", "0") == "1",
}
try:
    with urllib.request.urlopen(sys.argv[1], timeout=3.0) as response:
        status = json.load(response)
except (OSError, ValueError, urllib.error.URLError) as error:
    reject("cannot read /autoware_status: " + str(error))
if not isinstance(status, dict) or not isinstance(status.get("runtime_profile"), dict):
    reject("server does not report runtime_profile; restart it with the updated UI")
actual = status["runtime_profile"]
for field, value in expected.items():
    if field not in actual or type(actual[field]) is not type(value) or actual[field] != value:
        reject(field + " mismatch (requested " + json.dumps(value, ensure_ascii=False)
               + ", existing " + json.dumps(actual.get(field), ensure_ascii=False) + ")")
if status.get("localization_backend") != backend:
    reject("top-level localization_backend does not match the requested backend")
if type(status.get("ui_actions_enabled")) is not bool or status["ui_actions_enabled"] != expected["ui_actions_enabled"]:
    reject("top-level UI action permissions do not match the requested permissions")
print("Existing UI profile matches: " + json.dumps(expected, ensure_ascii=False))
PY_PROFILE
}

listening_pids_on_port() {
  ss -ltnp "( sport = :${PORT} )" 2>/dev/null \
    | grep -o 'pid=[0-9]\+' \
    | cut -d= -f2 \
    | sort -u
}

pid_state() {
  local pid="${1}"
  ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]'
}

is_ui_server_pid() {
  local pid="${1}"
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/${pid}/cmdline" | grep -F -- "${UI_SCRIPT}" >/dev/null 2>&1
}

stop_stale_ui_server_on_port() {
  local pid=""
  local state=""
  local stopped_pids=()
  local any_alive=false
  local _=0

  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    if ! is_ui_server_pid "${pid}"; then
      continue
    fi
    state="$(pid_state "${pid}")"
    if [[ "${state}" == *T* || "${state}" == Z* ]]; then
      stopped_pids+=("${pid}")
    fi
  done < <(listening_pids_on_port)

  if [[ ${#stopped_pids[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Stopping stale UI server process on port ${PORT}: ${stopped_pids[*]}"
  for pid in "${stopped_pids[@]}"; do
    kill -CONT "${pid}" >/dev/null 2>&1 || true
    kill "${pid}" >/dev/null 2>&1 || true
  done

  for _ in $(seq 1 20); do
    any_alive=false
    for pid in "${stopped_pids[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        any_alive=true
        break
      fi
    done
    if [[ "${any_alive}" != "true" ]]; then
      return 0
    fi
    sleep 0.25
  done

  echo "Force killing stale UI server process on port ${PORT}: ${stopped_pids[*]}"
  for pid in "${stopped_pids[@]}"; do
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  done
}

cleanup() {
  local exit_code=$?
  local _=0
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
    if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
      kill -KILL "${SERVER_PID}" >/dev/null 2>&1 || true
    fi
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  exit "${exit_code}"
}

if server_is_healthy; then
  require_matching_server_profile || exit 1
  echo "UI server already running from ${UI_SCRIPT}. Reusing the existing process."
  open_index_page
  exit 0
fi

stop_stale_ui_server_on_port

if server_is_healthy; then
  require_matching_server_profile || exit 1
  echo "UI server already running from ${UI_SCRIPT}. Reusing the existing process."
  open_index_page
  exit 0
fi

if ss -ltn "( sport = :${PORT} )" | grep -q LISTEN; then
  echo "Port ${PORT} is already in use, and $(healthcheck_url) is not responding."
  echo "Please stop the process that owns the port, then rerun this script."
  exit 1
fi

echo "Starting UI server: http://${HOST}:${PORT}/index.html"
trap cleanup EXIT INT TERM

workspace_already_sourced=false
case ":${AMENT_PREFIX_PATH:-}:" in
  *":${WORKSPACE_INSTALL_PREFIX}:"*)
    workspace_already_sourced=true
    ;;
esac

if [[ "${workspace_already_sourced}" != "true" && -f "${WORKSPACE_SETUP}" ]]; then
  # Colcon/Ament setup scripts may read optional trace vars that are unset.
  # Temporarily disable nounset so sourcing works even under `set -u`.
  set +u
  # shellcheck disable=SC1090
  source "${WORKSPACE_SETUP}"
  set -u
fi

python3 "${UI_SCRIPT}" --host "${HOST}" --port "${PORT}" &
SERVER_PID=$!

if wait_for_server; then
  require_matching_server_profile || exit 1
  open_index_page
fi

wait "${SERVER_PID}"
