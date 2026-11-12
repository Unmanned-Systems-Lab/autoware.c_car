# ins_driver.c_car

`ins_driver.c_car` 是 c_car 使用的独立 GI5651 CAN 驱动仓库，ROS 2 包名为 **`ins_driver_c_car`**。它复制自 `Unmanned-Systems-Lab/ins_driver.ez21` 的 `ez21_1` 分支、提交 `ec21f05c9ffaab53d556e44b68357e1733b43fd3`，在独立仓库中修正协议解析和定位质量处理。原仓库及原 ROS 包保持独立。

源实现说明其 CAN 代码来自 `/media/nvidia/program/xmt.tools_by_can/src/gi5651_can_odom`。本仓库保留来源归属和原包声明的 Apache License 2.0；本次复制未添加设备日志、用户定位数据或协议 PDF。

## 协议覆盖

依据用户提供的《GI5651-CAN协议说明_SQDT(1).pdf》核对，全部信号使用 Intel 小端，字节偏移从 0 开始：

| CAN ID | 有效数据 | 解码 |
| --- | --- | --- |
| `0x10B`，6 字节主姿态 | 航向、俯仰、横滚 | U16、S16、S16，均 ×0.01° |
| `0x20B` / `0x21B` | INS / GPS 纬度、经度 | S32、S32，均 ×10⁻⁷° |
| `0x30B` / `0x31B` | INS / GPS 高度、导航 Flag | S32 ×0.001 m、U8 |
| `0x40B` / `0x41B` | INS / GPS 东、北、天速度 | S16、S16、S16，均 ×0.01 m/s |
| `0x50B` | X、Y 加速度 | S32、S32，均 ×10⁻⁵，单位见下文 |
| `0x60B` | X、Y 角速度 | S32、S32，均 ×10⁻⁵°/s，ROS 输出转为 rad/s |
| `0x70B` | Z 角速度、Z 加速度 | S32、S32，均 ×10⁻⁵ |
| `0x09B` | GPS 周、周内毫秒 | U16、U32 |
| `0x10B`，4 字节 GPS 姿态 | GPS 航向、俯仰 | U16、S16，均 ×0.01°，横滚缺失 |

只接受标准 CAN 数据帧，拒绝扩展帧、RTR、错误帧及字段不足的报文；6 字节有效字段允许以 8 字节帧填充。INS 与 GPS 解缓存分开，`navigation_source` 选择其中一组，不会因为另一组报文到达而覆盖所选位置或速度。各字段记录接收时刻，过期或组间时间差超限的数据不能组成可信里程计。

协议把 `0x10B` 同时用于 6 字节主姿态和 4 字节 GPS 姿态，本实现按长度分开处理。该定义也可能是文档笔误，不能据此推定实际 GPS 姿态 ID 是另一个值。GPS 的 4 字节姿态不包含横滚，默认不会沿用旧横滚；`allow_missing_roll=false`，不完整姿态不能组成完整姿态定位。

## 默认行为与待确认项

默认配置是接收、诊断模式。能够收到 CAN 帧，不表示已得到可用于控制的定位。

- `can_interface=can0`。驱动不设置 CAN 波特率、不修改系统网络接口、不发送控制帧。接口应由系统预先配置为设备的实际波特率；本机实测为 500000 bit/s。
- `valid_nav_flags`、`fixed_nav_flags`、`float_nav_flags` 默认均为空。CAN 文档没有 Flag 码表，必须将厂家或实测确认的码值写入相应列表；未知码及 0 不作为有效定位，也不会自行认定 RTK FIX/FLOAT。
- `axes_verified=false`。文档未定义航向零方向、旋向、机体轴或欧拉角顺序。默认 `heading_convention=north_cw`、`euler_order=zyx`、`roll_sign=1`、`pitch_sign=1` 是显式配置假设，不是已验证结论。先确认静止姿态、朝向变化、正向运动与 ROS 轴向，再启用此确认项。
- `mounting_rpy_deg` 是传感器轴到输出轴的安装旋转，默认全 0，对 IMU 和里程计使用同一旋转。非零旋转时 `imu_frame_id` 必须与 `child_frame_id` 一致。不要同时在驱动和定位桥重复应用同一次安装旋转。
- `acceleration_unit=mps2` 按 CAN PDF 的单位执行；`g` 模式额外乘 9.80665。《XW-GI5651使用维护说明书20210524.pdf》的串口消息写 g，而 CAN 专用文档写 m/s²。静止输出模长约 1 只提示需核实 CAN 固件单位，不能直接证明其单位。默认 `acceleration_units_verified=false`，未确认时不把加速度标成已知测量精度。
- `time_source=receive` 使用 ROS 接收时间。若改为 `gps`，必须显式设置正确的 `gps_utc_leap_seconds`；默认 -1 表示未提供。GPS 周回绕补偿由 `gps_rollover_weeks` 显式指定，不猜测世纪或回绕次数。`max_device_time_offset_sec=5.0` 限制设备时间与接收时间的差值。设备时间缺失或无效时不伪造 GPS 时间。
- `reference_origin_active=false`，不继承旧车参考经纬高。需要固定 ENU 原点时，填写经过核实的地图原点或 `map_projector_info_path`。首个有效定位生成的局部 ENU 原点不能直接当作已有 Autoware 地图原点。
- 配置里的协方差是用户可调整的测量不确定度假设，不是 GI5651 提供的协方差。未验证或缺失的 IMU 分量按 ROS 约定标为未知，不用全零表示精确。

## 输出

| 参数 / 默认话题 | 类型与用途 |
| --- | --- |
| `raw_nav_sat_fix_topic=/sensing/ins/raw_nav_sat_fix` | `sensor_msgs/msg/NavSatFix`；观察所选来源的经纬高，状态保守映射 |
| `imu_topic=/sensing/imu/imu_raw` | `sensor_msgs/msg/Imu`；角速度转为 rad/s，加速度按显式单位转换 |
| `quality_topic=/sensing/ins/quality` | `diagnostic_msgs/msg/DiagnosticArray`；接收状态、数据时效和定位有效性 |
| `time_reference_topic=/sensing/ins/time_reference` | `sensor_msgs/msg/TimeReference`；经过显式 GPS/UTC 配置转换的设备时间 |
| `topic_name=/sensing/ins/odometry` | `nav_msgs/msg/Odometry`；只有定位质量、完整性、时效及轴向确认通过才发布 |

所选 GNSS 源和 IMU 传感器数据独立处理：在无有效 GNSS 定位时仍可观察原始 IMU，不能因此宣称定位就绪。默认里程计的 `child_frame_id=ins_link`，位置指 GI5651 的参考中心；转换到 `base_link` 需要实际测量的杆臂。

## 获取与构建

仓库为独立的 `Unmanned-Systems-Lab/ins_driver.c_car`，使用 `c_car` 分支，需要相应 GitHub 访问权限。ROS 2 Humble / Linux 示例：

```bash
mkdir -p ~/ins_driver.c_car_workspace/src
cd ~/ins_driver.c_car_workspace/src
git clone --branch c_car https://github.com/Unmanned-Systems-Lab/ins_driver.c_car.git
cd ..
source /opt/ros/humble/setup.bash
# 定位桥使用 autoware_adapi_v1_msgs；构建时 source 已有 Autoware 安装环境。
colcon build --packages-select ins_driver_c_car --symlink-install
source install/setup.bash
```

不需要把新包放进原 `ins_driver_ez21` 目录；新旧包名不同，可以并存。构建本包不会替换原包。本仓库不自动修改任何 Autoware 的 `autoware.repos` 或启动配置。

## 被动接收与检查

先查看已有接口状态，再启动接收节点：

```bash
ip -details link show can0
ros2 launch ins_driver_c_car ins_driver.launch.py
```

使用独立参数文件校准，避免直接修改已安装副本：

```bash
cp "$(ros2 pkg prefix ins_driver_c_car)/share/ins_driver_c_car/config/driver.yaml" /tmp/c_car-ins.yaml
ros2 launch ins_driver_c_car ins_driver.launch.py params_file:=/tmp/c_car-ins.yaml
```

可以直接覆盖常用标量参数：

```bash
ros2 launch ins_driver_c_car ins_driver.launch.py can_interface:=can0 navigation_source:=ins
ros2 topic echo /sensing/ins/quality --once
ros2 topic echo /sensing/ins/raw_nav_sat_fix --once
ros2 topic echo /sensing/imu/imu_raw --once
```

默认 Flag 映射为空、轴向未确认，`/sensing/ins/odometry` 没有可信定位输出是预期行为。先完成设备定位、Flag 解释、坐标和原点核验，再配置相应确认项；不能通过随意填写 Flag 或确认开关替代验证。

## 可选 Autoware 定位桥

`ins_localization.launch.py` 默认也只启动接收节点。完成定位质量、地图原点、安装外参核验后，才显式启用桥：

```bash
ros2 launch ins_driver_c_car ins_localization.launch.py \
  params_file:=/tmp/c_car-ins.yaml enable_bridge:=true
```

桥的参数在同一 YAML 的 `ins_localization_bridge` 节点下：

- `require_quality=true`，订阅同一驱动的质量话题，并检查质量与里程计的时间差和超时。
- `extrinsics_verified=false` 默认关闭；`base_link_to_ins_*` 均为 0，必须换成 c_car 的实测值。平移单位为 m，桥的欧拉角单位为 rad。
- `map_origin_verified=false` 默认关闭；未确认地图原点对齐就拒绝桥接，即使输入、输出使用同一坐标系名称，也不能跳过原点核验。
- `input_child_frame_id=ins_link` 必须与驱动输出一致；桥拒绝其他子坐标系的里程计，避免对已经转换到 `base_link` 的数据重复应用外参。
- 桥的 `acceleration_covariance_diag` 是独立的 6 维加速度方差假设。首帧或时间差不可用时不发布差分加速度。
- `publish_tf=false` 默认不发布 TF；启用之前确认地图坐标和现有 TF 发布者，避免重复发布。
- 无有效定位、质量或里程计过期时，不继续输出有效定位 TF，也不维持已初始化状态。
- 不启动 planning fallback 或 freespace bypass 节点，不用仿真替代数据填补真实定位。

## 测试与日志

协议测试不访问 CAN，不需要 ROS：

```bash
cd ~/ins_driver.c_car_workspace/src/ins_driver.c_car
python3 -m pytest test/test_protocol.py -q
```

ROS 包测试：

```bash
cd ~/ins_driver.c_car_workspace
colcon test --packages-select ins_driver_c_car --event-handlers console_direct+
colcon test-result --verbose
```

`log_enabled=false` 默认不写日志。需要时配置 `log_format=can` 或 `odom_csv`；CAN 文本为 `CANID#DATA`，CSV 只记录实际发布的里程计。设备 CAN 日志可能包含位置和车辆状态，应保存在设备或指定数据目录，不提交到 Git。
