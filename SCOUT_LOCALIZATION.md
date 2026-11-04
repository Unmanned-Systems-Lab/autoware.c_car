# Scout 定位来源选择

`localization_mode:=fixposition|universe` 在启动时选择一条定位链。切换时先停止上一轮启动，再用另一参数启动；不会同时启动两个 `map → base_link` 发布源。完整框架入口是 `start_scout.bash`，独立定位观察入口是 `start_scout_localization.bash`。

## 本次硬件与构建

RSHELIOS 32 线雷达、ZED 2i 校准 IMU、Scout V2 底盘 CAN 轮速。ZED ROS 2 源码、消息和相机描述已放在当前 `src/sensor_component/external/`，保留许可且没有嵌套 Git。来源见 `SCOUT_LOCALIZATION_SOURCES.json`。ZED SDK 5.5.1 使用标准 `/usr/local/zed` 安装；SDK 二进制不作为源码分发。

已有可用基础 Autoware 构建的车机可执行下面的支持包构建命令；全新 clone 应先完成本节后面的基础包步骤。

```bash
cd /mnt/nvme0n1/autoware/autoware.scout
# 已有完整 SDK 可跳过安装；没有时交互审阅官方安装许可。
bash scripts/install_scout_zed_sdk.bash
./build_scout_localization.bash
```

脚本编译本次新增支持包、车辆只读驱动、ZED ROS 包和启动配置，使用当前已构建的 Autoware 原生定位模块。原生 NDT、gyro odometer、EKF、pose initializer、地图加载器与点云处理器源码也在主工程；若首次安装，应先按主工程说明准备系统依赖并编译这些包。

### 首次 clone：先构建本仓库的基础 Autoware

本次定位适配、高程、HT 源码及模型权重在同一仓库。当前车机的实际运行仍复用 `/mnt/nvme0n1/autoware/autoware.ez21/install` 中预编译的基础 NDT、gyro odometer、EKF、pose initializer、地图加载、点云处理、原生定位 launch 和雷达驱动；当前仓库 `install/setup.bash` 的构建链记录了该 underlay。因此，“源码同仓库”不等于当前车机整个运行链已全部从此 checkout 构建。

全新 clone 不应复制车机的 `install/` 或继续引用旧 `autoware.ez21` 工作区。先按主工程 README 准备 ROS 2 Humble、JetPack/CUDA、ZED SDK 和系统/第三方依赖，然后在未 source 旧 Autoware 工作区的干净构建终端，仅以系统 ROS 为 underlay 编译本仓库基础包。下面是本功能相关基础包的选择入口，`--packages-up-to` 会同时构建它们在源码中的依赖；它不是完整 Autoware 所有规划、控制功能的构建清单。将示例路径替换为实际 clone 目录：

```bash
cd /path/to/autoware.scout
source /opt/ros/humble/setup.bash
colcon build --packages-up-to \
  autoware_ndt_scan_matcher autoware_gyro_odometer autoware_ekf_localizer \
  autoware_pose_initializer autoware_map_loader autoware_pointcloud_preprocessor \
  tier4_localization_launch rslidar_sdk scout_base \
  scout_vehicle_description scout_sensor_kit_description scout_sensor_kit_launch \
  --symlink-install --executor sequential \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
source install/local_setup.bash
ros2 pkg prefix autoware_ndt_scan_matcher
ros2 pkg prefix rslidar_sdk
./build_scout_localization.bash
```

两个 prefix 检查结果应落在新 checkout 的 `install/` 内。基础算法源码分别位于 `src/core/autoware_core/`、`src/universe/autoware_universe/`，原生定位启动位于 `src/launcher/autoware_launch/tier4_universe_launch/`，雷达驱动位于 `src/sensor_component/external/robosense/`。`build_scout_localization.bash` 随后构建 ZED/车辆支持包、本次定位支持、高程、HT 和入口配置，不会替代前面的基础 Autoware 构建。

ROS、CUDA、ZED SDK、PyTorch、CuPy 等系统运行依赖可安装在系统目录；它们不作为旧 ROS 工作区的 `install` 二进制拷入仓库。

默认 Cyclone DDS 配置在 `config/scout_localization_cyclonedds.xml`，只用于本机通信；显式设置的 `CYCLONEDDS_URI`、`RMW_IMPLEMENTATION`、`ROS_DOMAIN_ID` 继续生效。所有启动与检查终端应使用相同配置。

## 使用已保存的 LIO-SAM 地图验证 Universe

这是本机局部定位，地图坐标不是已确认的 Fixposition 地理坐标。只启动地图、传感器、只读底盘反馈和定位，不启动规划、底盘控制接口或网页驾驶操作。先保存并停止已有 LIO-SAM/其他雷达相机启动，避免 UDP、USB 和动态 TF 冲突。

按 `SCOUT_STARTUP.md` 配置原生 CAN0 引脚与 `can10` 的 500kbps 接口后，在 Orin 桌面终端执行：

```bash
cd /mnt/nvme0n1/autoware/autoware.scout
./start_scout_localization.bash \
  localization_mode:=universe \
  pointcloud_map:=/mnt/nvme0n1/autoware/lio_sam_ros2_ws/maps/lap_review_20261103_104358_4db4143b/map/GlobalMap.pcd \
  can_interface:=can10
```

只传 `GlobalMap.pcd` 单文件，不要把含轨迹和角点文件的整个目录作为地图。RViz 显示参考地图、NDT 配准点云与车辆位姿。车辆保持静止至少 3 秒后，在地图上用 **2D Pose Estimate** 给近似位置及朝向；桥接器调用原生 `/localization/initialize` 的 AUTO 方法，由 NDT 对齐后初始化 EKF。初始化成功以前不发布伪造的就绪状态。无桌面时可加 `rviz:=false`。

已有传感器时可显式设置 `start_sensors:=false start_wheel_odometry_driver:=false`；须确认输入话题和 TF 唯一。硬件模式默认 `require_can_feedback:=true`。仅离线数据或隔离合成测试可显式关闭此门限，不能用来绕过实车 CAN 故障。

## 完整 Autoware 的两个模式

Fixposition 保留原定位桥、外参和输出接口；以下为观察模式示例：

```bash
SCOUT_UI_ALLOW_VEHICLE_ACTIONS=0 ./start_scout.bash \
  localization_mode:=fixposition map_path:="$MAP_DIR" \
  launch_vehicle_interface:=false launch_ui:=true rviz:=true
```

Universe 使用与 Lanelet/地图投影信息同坐标系的真实稠密点云地图：

```bash
SCOUT_UI_ALLOW_VEHICLE_ACTIONS=0 ./start_scout.bash \
  localization_mode:=universe map_path:="$ALIGNED_MAP_DIR" \
  vehicle_interface_can_interface:=can10 \
  launch_vehicle_interface:=false launch_ui:=true rviz:=true
```

框架的地图目录须含真实 `pointcloud_map.pcd`、`lanelet2_map.osm`、`map_projector_info.yaml` 及对应点云元数据。网页生成的几十个占位点不能做 NDT。启动会拒绝缺失、稀疏占位或截断的 PCD；点数检查并不证明点云与车道已地理配准。当前保存的 LIO 局部图尚未完成这项配准，因此本轮先使用上面的独立定位入口。

关闭底盘接口时，Universe 按相同 CAN 参数创建只读轮速采集器；完整底盘接口已启用时复用其轮速，不再创建第二个采集器。连接实车控制时仍需现有工程的 CAN、接管、地图与控制验证流程。

网页状态显示所选后端。Universe 的路线/自动操作要求原生初始化完成、NDT 和最终位姿新鲜有效，以及固定真实地图已加载；不以 Fixposition 在线与否替代这些条件。Universe 不支持网页生成或在线热换地理地图，换图需停止后用已配准地图重新启动。复用旧网页服务时会核对后端、地图和操作权限，配置不一致则拒绝复用。

## 数据与保护

| 输入/输出 | 用途 |
|---|---|
| `/rslidar_points` | 原始 XYZIRCAEDT 点云，保持不变 |
| `/scout/localization/points_ordered` | 保留有效点原始字段，过滤非有限 XYZ，按 UINT32 纳秒逐点时间稳定排序 |
| `/scout/localization/points_deskewed` | 原生 Universe 去畸变后，进入原生 crop/voxel/random/NDT |
| `/zed2i/zed_node/imu/data` | ZED SDK 校准的原始 IMU 输入，可由 `universe_imu_topic` 自定义 |
| `/scout/localization/imu` | 保留传感器数据与时间的可靠 QoS 转发，供原生 gyro odometer 与去畸变 |
| `/scout/wheel_odometry` | 底盘里程计输入；只融合轮速，不把积分 pose 当地图绝对位置 |
| `/sensing/vehicle_velocity_converter/twist_with_covariance` | 从底盘中心换算到 `base_link` 的可靠轮速与协方差 |
| `/localization/kinematic_state` | 两后端共同的最终定位输出，`map` / `base_link` |
| `/localization/acceleration` | 两后端共同的加速度输出 |
| `/localization/initialization_state` | 原生/原有初始化状态 |

`scout_chassis_link` 位于 `base_link` 前方 0.249m；适配器正确处理杆臂、协方差与时间，未知方差不当作零噪声。底盘驱动会重发缓存状态，因此实车还监听真正的 **0x221 运动反馈**；首次真实反馈前、超过 0.5 秒、接口出错或消息/TF 无效时不发布轮速。0x211 是系统状态，不能刷新运动反馈有效期。只读采集不发送 CAN 帧。

原生 gyro odometer/EKF 使用车辆式纵向速度与 IMU 角速度模型；Scout 原地转向、滑移和动态精度仍需实车验证。初始参数沿用原生 NDT 匹配阈值，没有为通过测试降低分数门限。

本轮验证日志、截图和地图快照位于 `validation/localization_switch/`，不提交为源码。修改前的现有工作区文件备份保存在主目录旁的 `scout_localization_before_20261103_144445/`。

## 2026-11-03 车机验证记录

- 当前工作区构建了 ZED 4 包及 `ugv_sdk`、`scout_base`、`scout_localization_support`、`autoware_launch`；均退出 0。新原生算法节点复用当前 Autoware 已安装定位组件，原有 underlay 设置存在 CUDA 旧前缀提示，不是本轮新增依赖。
- 49 项支持包测试、14 项启动语义测试、原 UI 22 项与新 UI 14 项测试、9 项 UI 服务复用测试通过。两个完整框架 backend 的 launch 参数展开均退出 0。新 Universe 诊断图通过原生解析器，61 个节点引用全部可解析。
- Fixposition 通过独立 DDS 域 73 的原生启动/合成里程计回归：只启动原定位桥，标准输出 frame/child 与数值正常，无 NDT、传感器、底盘或控制进程；此项不是新增的实车 GNSS 精度测试。
- Universe 用上述真实 LIO 局部地图完成原生 NDT AUTO 初始化；实际 RViz `/initialpose` 桥调用同一原生服务也成功。ZED 启动已隔离作用域，NDT 与姿态稳定检测保留原生节点名，相关 6 个原生诊断均为 OK。
- 最终 30 秒实机采样 22 项检查全部通过：轮速与转换速度约 50Hz；ZED IMU 原始 382.84Hz、转发 381.81Hz；点云、去畸变、NDT 输入、匹配点云和 NDT 位姿约 10Hz；最终定位 1502 帧，约 50Hz。原始传感器时间保留，动态时间戳正且递增，私有点云及地图 XYZ 有限。
- 地图 287360 点、初始化状态 3。NDT 末帧在采样结束时仅落后约 0.216 秒，证明仍有真实匹配更新。NVTL 最小 2.8358、均值 2.8706、最大 2.9142；实际阈值仍为原生 2.3，TP 阈值 3.0，选择类型 1，没有降低匹配门限。
- 动态 `base_link` 的父坐标系在 1499 帧观测中仅为 `map`；TF 查询新鲜、有限、单调。此 Humble Python API 未提供消息 GID，未把父集合检查冒充发布者数量测量。
- 实际参数确认只读底盘、禁止自动重连、真实 CAN 新鲜度门开启。整个窗口真实轮速 `vx/wz` 均为 0，CAN TX 计数 0→0；没有 Fixposition、INS bridge、LIO、规划或控制节点/服务。
- RViz 已目视确认 Global Status OK，参考地图与 NDT 配准点云可见。启动最初可能出现相机 IMU TF 缓存尚未覆盖首帧的提示；原生 EKF 仍偶发高频 twist 队列提示，最终持续更新和诊断检查通过，不作无警告宣称。

上述为静止局部定位验证。没有自动驾驶、原地转向、动态去畸变精度或长距离定位精度结论；当前局部地图也尚未与车道/Fixposition 地理地图配准。

## Universe 定位同时运行局部高程图

下次从停止状态启动定位与高程图：

```bash
cd /mnt/nvme0n1/autoware/autoware.scout
./start_scout_localization.bash \
  localization_mode:=universe \
  pointcloud_map:=/mnt/nvme0n1/autoware/lio_sam_ros2_ws/maps/lap_review_20261103_104358_4db4143b/map/GlobalMap.pcd \
  can_interface:=can10 \
  start_elevation:=true
```

如果定位已经运行，在具有相同 ROS_DOMAIN_ID、RMW_IMPLEMENTATION、CYCLONEDDS_URI 的终端单独挂接高程图（不会重复启动定位或传感器）：

```bash
cd /mnt/nvme0n1/autoware/autoware.scout
./start_scout_elevation.bash \
  pointcloud_map:=/mnt/nvme0n1/autoware/lio_sam_ros2_ws/maps/lap_review_20261103_104358_4db4143b/map/GlobalMap.pcd
```

两种方式都启动 RViz。高程图显示名为 `Global elevation crop (10 m)`，按观测高度着色；原始参考点云和 NDT 对齐点云可在左侧勾选比较。`rviz:=false` 可关闭新启动的可视化窗口。单独挂接的程序用 Ctrl+C 停止，保留原有定位。

高程分辨率为 0.04 m，滚动窗口为 10 m；累计地图由 SQLite 持续保存，默认目录：

```text
/mnt/nvme0n1/autoware/scout_local_elevation_maps/local_ndt/<identity>/<run_id>/tiles.sqlite3
```

每次启动管理器生成独立会话，绑定实际加载点云的几何摘要、参考 PCD 内容、固定雷达外参、过滤规则及建图参数。短暂定位中断后，仅恢复本次运行的数据；下次全新启动不自动拼接旧会话。数据在当前 LIO-SAM/NDT `map` 坐标下，未配准地理坐标。不会读写 Fixposition/ECEF 高程库。

使用 `/scout/localization/points_deskewed` 的 RSHELIOS 32 线点云，在 base_link 中去除车体，保留 rslidar 坐标与真实射线起点。输入检查要求地图已加载、原生初始化状态 3、NDT/EKF 新鲜、严格对应点云时间的 TF。原始 GPU 平面初始化保持关闭；全局地图仅在固定起点范围提供地面先验，未观测计数保持 0；处理计数增长且真实 `observation_count > 0` 才报告 `MAPPING`。更换 PCD、地图几何或固定外参需要重启定位及高程会话。

检查状态：

```bash
ros2 topic echo /scout/elevation/status
ros2 topic echo /scout/elevation/session --once --qos-durability transient_local
```

完整累计数据是 SQLite 数据库。原生 `/elevation_mapping_cupy/save_map` 服务导出当前滚动窗口的 MCAP，不代表全部累计区域。运行时复制 SQLite 必须用数据库备份接口包含 WAL 中的数据；停止该程序并完成落盘后可复制整个会话目录。

源码已放入当前仓库的 `src/perception/external/{elevation_mapping_cupy,elevation_map_msgs,ros2_numpy}`，不依赖旧 ELE 工作区。版本、许可证、复制快照和文件校验见 `SCOUT_ELEVATION_SOURCES.json`。当前机器使用已安装的 CuPy 13.6.0、PyTorch 2.8.0、CUDA 12.6，以及系统 grid_map_msgs/grid_map_rviz_plugin；没有复制第三方 Python 安装目录。`build_scout_localization.bash` 已包含这三个源包，GPU 库路径由 `scripts/scout_elevation_runtime_env.bash` 自动发现。

组合定位入口在 `start_elevation:=true` 时默认同时启动全局分块 HT；可用 `start_traversability:=false` 单独关闭识别。单独的高程挂接入口仍只启动高程图。两者均不启动规划或车辆控制；HT 使用方法见下文。

### 本机高程实测记录（2026-11-03）

现有手动 Universe 定位保持运行，另行挂接高程图与 RViz。45 秒验收全部 9 项通过：153 帧成功处理、车旁 7561 个真实观测格子、4 cm 分辨率、map 坐标、worker_generation=1；后续 12 秒独立采样 Gate READY 100%，处理计数继续增长。SQLite 已写入 6 个数据瓦片，quick_check=ok；已用数据库备份接口生成一致快照。RViz Global Status=OK。90 项测试通过。此记录验证当前输入与建图链路，运动中的定位和地图效果可在 RViz 继续观察。

实测详情在 `validation/local_elevation/{acceptance.json,verification.json,final_tests.log,rviz_elevation.png}`。

### 重启后 CAN 与初始位姿恢复

2026-11-04 修复：`start_scout_localization.bash` 在 Universe 实车只读轮速输入启动前准备 CAN。已 UP 时保持现状；仅对已验证的 AGX Orin 原生 CAN0 配置已知引脚并以 500 kbps 启用，必要时终端会要求 sudo 密码。不会发送 CAN 数据帧。未识别的硬件/引脚配置会明确报错。

高程版 RViz 默认显示 Reference map，并以 map 为视角目标，首次未初始化时也能看到参考地图。保持车辆静止，用 2D Pose Estimate 在地图位置按住左键拖出朝向并松开。若原生初始化返回 The vehicle is not stopped，需确认真实轮速与融合速度新鲜；缺失数据不能视作静止，不应关闭原生停止检查。

本次实机恢复后：轮速、轮速转换、gyro twist 均约 50 Hz；重试用户最后一次选定位置后原生初始化状态变为 3。随后 25 秒验收持续 MAPPING，成功处理计数增加 223，真实高程格子约 7254，CAN TX packets 保持 0。

### RViz 显示全局累积地图的车旁裁剪

RViz 高程显示订阅 `/elevation_mapping_node/elevation_map_global_local`：约 10×10 m、0.04 m 分辨率、最高 5 Hz，随车移动；高度和颜色层均是 elevation。该数据由累计全局 tile store 的 extract_window 得到，优先读全局热缓存，必要时加载 SQLite 瓦片，包含历史融合观测。它不是 GPU 原始滚动窗口 `/elevation_mapping_node/elevation_map_raw`。

正常启动配置仍默认显示 Reference map 供初始位姿选择。已初始化的当前窗口使用 `validation/local_elevation/scout_global_elevation_crop_live_20261104.rviz`，隐藏参考点云和 NDT 点云，专看全局裁剪高程面。此次仅重载 RViz，原定位、建图 worker 和数据库会话保持连续。

### 初始 10×10 米地面先验

默认开启：Universe 已初始化、实际地图加载且位姿新鲜后，从起点周围参考点云稳健估计地面高度，在起点固定的 10×10 m 全局 XY 范围发布初始平地。范围不会随车辆无限扩张。未观测格子 elevation 为该地面高度、variance=10000、observation_count=0；真实观测首次到达即可更新高度与方差，观测计数从 1 开始。起点范围外尚未观测处继续保持 NaN。

地面估计失败或数据缺失会明确等待，不猜零高程。初始高度、边界、地图摘要、起点位姿及估计质量绑定会话；本次管理器中断恢复沿用同一先验。RViz 的全局裁剪话题不变，启动命令不变。已有地图文件保留，新配置使用新的会话。

2026-11-04 实机：自动估计地面 -0.8092227 m（map 坐标）。10×10 m/4 cm 窗口 62500 个格子均有限、NaN=0，其中约 55258 个为未观测先验，7242 个已由真实观测更新；实测高度覆盖约 -0.919 到 1.942 m。30×30 m 全局视图范围外仍有 500000 个未知格。165 项测试通过，实机 10 项检查通过；旧地图一致备份位于 `/mnt/nvme0n1/autoware/scout_initial_ground_before_20261104_113934/`。

## 全局分块 HT 可通过性识别（2026-11-04）

可通过性识别基于当前会话的累计全局高程图持续处理，并将八方向模型结果保存为全局分块缓存。默认 RViz 显示从这份全局结果裁出的车旁视图；车辆移动时重新裁剪缓存，不把每帧局部高程重新当作一张独立地图。

### 启动与查看

从停止状态启动时，沿用上节的 Universe 定位命令并设置 `start_elevation:=true`，现在会默认同时启动 HT；`start_traversability` 默认跟随 `start_elevation`。也可明确增加 `start_traversability:=true`。仅需定位与高程图时，增加 `start_traversability:=false`。

定位和高程会话已运行时，在相同 ROS_DOMAIN_ID、RMW_IMPLEMENTATION、CYCLONEDDS_URI 的终端挂接：

```bash
cd /mnt/nvme0n1/autoware/autoware.scout
./start_scout_traversability.bash rviz:=true
```

此入口自动订阅 `/scout/elevation/session`，读取该会话的全局高程数据库。它只附加识别和 RViz；Ctrl+C 结束本次附加程序，原定位与高程会话继续运行。`rviz:=false` 可关闭新启动的窗口。单独的 `start_scout_elevation.bash` 仍用于挂接高程图，再用上述入口附加 HT。

默认配置使用 `scout_localization_traversability.rviz`：

- `Local traversability (global crop)`：默认开启，订阅 `/scout/traversability/global_local`，约 10×10 m、0.04 m 分辨率，最高 2 Hz 随车裁剪。
- `Global traversability (overview)`：在左侧勾选查看累计全局结果，订阅 `/scout/traversability/global_overview`。概览最长每轴 512 格，随地图范围选择 LOD 分辨率，默认最高 0.2 Hz；没有订阅者时不发布概览。
- 两者高度均使用 `elevation`，颜色使用 `traversability_class`：红色 0 表示模型判为不可通，绿色 1 表示模型判为可通，分类阈值为 0.5。原始连续分数层为 `traversability`，范围固定为 0～1。空白表示尚无有效结果。

默认分数是八个原始通道通过概率的最小值，达到阈值才显示为绿色。八方向尚未完成与车体物理方向的标定，保留原始通道编号；本入口不应用未标定的前参考点偏移。

### 高度、覆盖与来源

起点固定 10×10 m 范围内的有限地面高度会真实进入 HT 神经网络，显示实际输出概率；没有将先验格强制涂为可通，也不会仅因 `observation_count=0` 就屏蔽这些模型结果。范围外缺失高度继续作为未知；最近邻填充只用于网络数值计算，不扩大原始有效覆盖。有效预测要求原中心高度有限，且 51×51 感受野内至少 70% 高度有限。

质量信息与模型判别分开保存：

| 层 | 含义 |
|---|---|
| `valid` | 原分辨率有效预测标记；概览中为汇总子格的有效预测覆盖比例 |
| `support_fraction` | 模型感受野内有限高程的覆盖比例，包含有限初始地面 |
| `observed_fraction` | 感受野内真实观测高程的比例 |
| `source_stale`、`result_age_sec` | 源数据已更新标记及结果计算后的时间 |

概览采用已知模型概率的保守最小值汇总；需结合 `valid` 覆盖比例判断大格中的有效范围。原始观测计数、方差继续保留在高程与结果缓存中。

局部车旁视图中，源已变化且超过默认 10 秒仍未重算的旧分数会隐藏；静态未变化的结果不会仅因计算后的时间增长而隐藏。位姿失效时车旁视图清空。全局概览保留历史模型概率，通过 `source_stale`、`result_age_sec` 和 `valid` 覆盖比例标记新鲜度及有效范围；粗 LOD 中有部分子格等待更新时，不会因此将整个粗格的历史概率统一隐藏。

### 分块处理与保存

高程瓦片为 128×128 格，每格 4 cm。模型感受野为 51×51，因此每个任务从全局高程读取四周 25 格 halo，共 178×178 输入，只把中心 128×128 的八方向结果写入结果库。跨块上下文按统一全局栅格获取，避免块边缘缺少模型上下文。

源 SQLite 通过实际提交的 `commit_seq` 增量发现变化，将受影响块及相邻上下文块加入持久去重队列；固定初始地面覆盖的虚拟块也会首次排队。高度输入完全相同时复用模型概率并更新质量层。处理默认采用每秒 2 个模型瓦片的软预算，近车任务优先，同时保留后台任务机会；热缓存使用有限 LRU，概览使用分级 LOD，不创建覆盖整幅大地图的稠密神经网络张量。

结果、任务队列和概览缓存保存在附加 SSD：

```text
/mnt/nvme0n1/autoware/scout_traversability_maps/<source_run_id>/<contract>/traversability.sqlite3
```

源会话、地图几何、模型权重、推理代码和配置共同约束缓存，配置不匹配时拒绝复用。HT 使用独立结果数据库，源高程数据库的高度与观测层不由识别程序改写。

源码和权重均位于当前仓库的 `src/perception/scout_global_traversability/`，运行时不依赖旧 HT/ELE 工作区。模型来源、SHA256 和原始 `Proprietary research code` 声明见该目录的 `HT_MODEL_SOURCES.json`、`HT_MODEL_LICENSE.txt`。

检查运行状态：

```bash
ros2 topic echo /scout/traversability/status
```

正常状态为 `RUNNING`；诊断包含 `model_calls`、待处理块数、源提交序号、模型耗时和错误原因。每秒 2 块是预算上限，实际模型调用率会随高度复用及任务量变化。验证日志、模型基准和实时采样保存在 `validation/global_traversability/`。

上述实机验收只增加感知识别与显示，没有自动控制实车；验收阶段没有执行 Git 提交或推送。本发布分支用于提交和上传源码、配置、随包模型及来源说明，运行地图、数据库、录包和日志保留在车机。
