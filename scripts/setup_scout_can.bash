#!/usr/bin/env bash
# Prepare the verified AGX Orin CAN0 interface; never sends CAN frames.
set -euo pipefail

fail() { printf 'CAN 准备失败：%s\n' "$*" >&2; exit 1; }
usage() { printf '用法：%s [接口名，默认 can10]\n       %s --interface 接口名\n' "${0##*/}" "${0##*/}"; }

can_interface=can10
case "$#" in
  0) ;;
  1)
    case "$1" in
      -h|--help) usage; exit 0 ;;
      -*) usage >&2; exit 2 ;;
      *) can_interface="$1" ;;
    esac ;;
  2)
    [[ "$1" == --interface ]] || { usage >&2; exit 2; }
    can_interface="$2" ;;
  *) usage >&2; exit 2 ;;
esac
[[ "$can_interface" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.:-]{0,14}$ ]] || fail '接口名无效。'

net_dir="/sys/class/net/$can_interface"
[[ -d "$net_dir" ]] || fail "找不到接口 $can_interface。"
[[ "$(cat "$net_dir/type")" == 280 ]] || fail "$can_interface 不是 CAN 网络接口。"

is_up() {
  local flags
  flags="$(cat "$net_dir/flags")" || fail "不能读取 $can_interface 的接口标志。"
  [[ "$flags" =~ ^0[xX][0-9a-fA-F]+$ ]] || fail "无法识别 $can_interface 的接口标志。"
  (( (flags & 1) != 0 ))
}
if is_up; then
  printf 'CAN 接口 %s 已 UP，保持当前配置。\n' "$can_interface"
  exit 0
fi

[[ -r /proc/device-tree/model ]] || fail '无法读取 Jetson 型号；未修改接口或引脚。'
model="$(tr -d '\000' < /proc/device-tree/model)"
[[ "$model" == 'NVIDIA Jetson AGX Orin'* ]] || fail "不支持此设备型号：$model；未修改接口或引脚。"
device_path="$(readlink -f -- "$net_dir/device")" || fail '无法识别 CAN 控制器。'
driver_path="$(readlink -f -- "$net_dir/device/driver")" || fail '无法识别 CAN 驱动。'
[[ "${device_path##*/}" == c310000.mttcan && "${driver_path##*/}" == mttcan ]] || \
  fail "$can_interface 不是已验证的 c310000.mttcan 原生 CAN0；未修改接口或引脚。"

if (( EUID != 0 )); then
  command -v sudo >/dev/null || fail '需要管理员权限，但系统没有 sudo。'
  script_path="$(readlink -f -- "${BASH_SOURCE[0]}")"
  printf '为 %s 准备原生 CAN0，需要 sudo 权限。\n' "$can_interface"
  exec sudo -- bash "$script_path" --interface "$can_interface"
fi
command -v busybox >/dev/null || fail '缺少 busybox，不能配置已验证的 CAN0 引脚。'
command -v ip >/dev/null || fail '缺少 ip，不能启用 CAN 接口。'

read_register() {
  local value
  value="$(busybox devmem "$1" 32)" || fail "读取引脚寄存器 $1 失败。"
  [[ "$value" =~ ^0[xX][0-9a-fA-F]{1,8}$ ]] || fail "寄存器 $1 返回未知值：$value。"
  printf '%s' "$value"
}
tx_pin="$(read_register 0x0c303010)"
rx_pin="$(read_register 0x0c303018)"
(( tx_pin == 0xC059 || tx_pin == 0xC400 )) || fail "TX 引脚配置未知：$tx_pin；未写入寄存器。"
(( rx_pin == 0xC055 || rx_pin == 0xC458 )) || fail "RX 引脚配置未知：$rx_pin；未写入寄存器。"

# Another caller may have enabled the interface while permissions were obtained.
if is_up; then
  printf 'CAN 接口 %s 已 UP，保持当前配置。\n' "$can_interface"
  exit 0
fi
if (( tx_pin != 0xC400 )); then
  busybox devmem 0x0c303010 32 0xC400 || fail '写入 CAN0 TX 引脚配置失败。'
fi
if (( rx_pin != 0xC458 )); then
  busybox devmem 0x0c303018 32 0xC458 || fail '写入 CAN0 RX 引脚配置失败。'
fi
tx_pin="$(read_register 0x0c303010)"
rx_pin="$(read_register 0x0c303018)"
(( tx_pin == 0xC400 && rx_pin == 0xC458 )) || fail 'CAN0 引脚配置读回不一致；未启用接口。'

ip link set dev "$can_interface" up type can bitrate 500000 || fail "启用 $can_interface（500 kbps）失败。"
is_up || fail "$can_interface 启用后未处于 UP 状态。"
printf 'CAN 接口 %s 已启用：500 kbps。\n' "$can_interface"
