#!/usr/bin/env bash
# Fetch the official SDK separately from the vendored open-source dependencies.
set -euo pipefail

deploy_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
workspace_dir="$(cd -- "$deploy_dir/.." && pwd -P)"
sdk_version=5.5.1
sdk_prefix=/usr/local/zed
installer_name=ZED_SDK_Tegra_L4T36.5_v5.5.1.zstd.run
default_url="https://stereolabs.sfo2.cdn.digitaloceanspaces.com/zedsdk/5.5/$installer_name"
expected_sha256=a4564d9272724e523572403a0decc33169a849161c37bd1546373a17f4588a57
installer_url="$default_url"
installer_arg=''
accept_license=false
download_only=false
url_overridden=false
partial_file=''

usage() {
  cat <<'USAGE'
Install the official ZED SDK 5.5.1 for Jetson Orin / L4T 36.5 (JetPack 6.2.2).

Usage: bash deploy/install_zed_sdk.bash [options]
  --installer PATH    Use an existing official .run installer instead of downloading.
  --url HTTPS_URL     Download an installer from an explicit HTTPS URL.
  --sha256 HEX        Expected SHA256 (default: the pinned official 5.5.1 installer).
  --download-only     Fetch and verify only; do not install or accept a license.
  --accept-license    Explicitly accept the vendor license and use silent installation.
  -h, --help          Show this help.

The default mode shows the official license and uses the interactive installer.
The SDK is installed to /usr/local/zed. Downloads remain in .deps/downloads/.
Read the vendor terms before using --accept-license:
https://www.stereolabs.com/SoftwareandServicesLicenseAgreement
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --installer) need_value "$@"; installer_arg="$2"; shift 2 ;;
    --url) need_value "$@"; installer_url="$2"; url_overridden=true; shift 2 ;;
    --sha256) need_value "$@"; expected_sha256="${2,,}"; shift 2 ;;
    --accept-license) accept_license=true; shift ;;
    --download-only) download_only=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "Unknown option: $1" ;;
  esac
done
[[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || die '--sha256 must contain exactly 64 hexadecimal characters'
[[ -z "$installer_arg" || "$url_overridden" == false ]] || die 'Use either --installer or --url'
[[ "$installer_url" == https://* ]] || die 'Installer downloads must use HTTPS'

cleanup() {
  if [[ -n "$partial_file" && -f "$partial_file" ]]; then
    rm -f -- "$partial_file"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

read_installed_version() {
  python3 - "$sdk_prefix" <<'PY'
import re
import sys
from pathlib import Path

prefix = Path(sys.argv[1])
for name in ('zed-config-version.cmake', 'zed-config.cmake'):
    file = prefix / name
    if not file.is_file():
        continue
    text = file.read_text(errors='replace')
    match = re.search(r'set\s*\(\s*(?:PACKAGE_VERSION|ZED_VERSION)\s+"?([0-9]+\.[0-9]+\.[0-9]+)', text, re.I)
    if match:
        print(match.group(1))
        raise SystemExit(0)
values = {}
for name in ('include/sl/Camera.hpp', 'include/sl/defines.hpp'):
    file = prefix / name
    if file.is_file():
        values.update(re.findall(r'#\s*define\s+ZED_SDK_(MAJOR|MINOR|PATCH)_VERSION\s+([0-9]+)', file.read_text(errors='replace')))
if all(part in values for part in ('MAJOR', 'MINOR', 'PATCH')):
    print('.'.join(values[part] for part in ('MAJOR', 'MINOR', 'PATCH')))
PY
}

verify_installed_sdk() {
  [[ -f "$sdk_prefix/zed-config.cmake" ]] || die "Missing $sdk_prefix/zed-config.cmake"
  [[ -f "$sdk_prefix/include/sl/Camera.hpp" ]] || die 'The SDK development header include/sl/Camera.hpp is missing; install a complete SDK development environment'
  [[ -r "$sdk_prefix/lib/libsl_zed.so" ]] || die "Missing $sdk_prefix/lib/libsl_zed.so"
  local found_version
  found_version="$(read_installed_version)"
  [[ "$found_version" == "$sdk_version" ]] || die "Expected SDK $sdk_version, found '${found_version:-unknown}' at $sdk_prefix"
}

if [[ "$download_only" == false ]]; then
  [[ "$(uname -s)" == Linux && "$(uname -m)" == aarch64 ]] || die 'This installer targets Linux aarch64 Jetson Orin only'
  [[ -r /etc/nv_tegra_release ]] || die 'Cannot identify JetPack/L4T: /etc/nv_tegra_release is missing'
  l4t_major="$(sed -nE 's/^# R([0-9]+).*/\1/p' /etc/nv_tegra_release | head -n 1)"
  l4t_revision="$(sed -nE 's/.*REVISION: ([0-9.]+),.*/\1/p' /etc/nv_tegra_release | head -n 1)"
  [[ "$l4t_major" == 36 && "$l4t_revision" == 5.* ]] || die "Expected L4T 36.5.x / JetPack 6.2.2 (CUDA 12.6); found R${l4t_major:-?}.${l4t_revision:-?}"
  command -v python3 >/dev/null || die 'Install python3 before running this script'
  if [[ "$(read_installed_version)" == "$sdk_version" ]]; then
    verify_installed_sdk
    printf 'ZED SDK %s is already available at %s\n' "$sdk_version" "$sdk_prefix"
    exit 0
  fi
  if [[ "$accept_license" == false && ! -t 0 ]]; then
    die 'The default installer is interactive. Run in a terminal, or review the vendor license and explicitly pass --accept-license'
  fi
  command -v zstd >/dev/null || die 'Install zstd before running the official installer: sudo apt install zstd'
fi

command -v sha256sum >/dev/null || die 'sha256sum is required to verify the installer'
verify_checksum() {
  local actual
  actual="$(sha256sum -- "$1")"
  actual="${actual%% *}"
  [[ "$actual" == "$expected_sha256" ]] || die "Installer checksum mismatch for $1: expected $expected_sha256, got $actual"
}

if [[ -n "$installer_arg" ]]; then
  [[ -f "$installer_arg" && -r "$installer_arg" ]] || die "Installer is not a readable file: $installer_arg"
  installer="$(cd -- "$(dirname -- "$installer_arg")" && pwd -P)/$(basename -- "$installer_arg")"
  verify_checksum "$installer"
else
  cache_dir="$workspace_dir/.deps/downloads"
  mkdir -p -- "$cache_dir"
  installer="$cache_dir/ZED_SDK_${sdk_version}_${expected_sha256:0:12}.run"
  if [[ ! -f "$installer" ]]; then
    command -v curl >/dev/null || die 'Install curl and ca-certificates before downloading the SDK'
    partial_file="$(mktemp "$cache_dir/.zed-sdk-download.XXXXXX")"
    printf 'Downloading official ZED SDK %s installer...\n' "$sdk_version"
    curl --fail --location --proto '=https' --proto-redir '=https' --tlsv1.2 \
      --connect-timeout 20 --max-time 1800 --retry 3 --retry-delay 2 \
      --output "$partial_file" "$installer_url"
    verify_checksum "$partial_file"
    mv -- "$partial_file" "$installer"
    partial_file=''
  else
    verify_checksum "$installer"
  fi
fi
printf 'Verified installer: %s\n' "$installer"

if [[ "$download_only" == true ]]; then
  printf 'Download verified. No SDK was installed and no license was accepted.\n'
  exit 0
fi

if [[ "$accept_license" == true ]]; then
  printf 'Installing ZED SDK %s after explicit license acceptance.\n' "$sdk_version"
  # --accept handles the Makeself outer license; subsequent arguments belong
  # to the official SDK installer. Keep headers for the ROS 2 wrapper build.
  bash "$installer" --accept -- silent skip_tools skip_python skip_od_module skip_drivers
else
  printf 'Opening the official interactive installer. Review its license before accepting.\n'
  bash "$installer"
fi
verify_installed_sdk
printf 'ZED SDK %s development files verified at %s\n' "$sdk_version" "$sdk_prefix"
