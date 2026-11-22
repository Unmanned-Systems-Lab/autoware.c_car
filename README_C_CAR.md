# C-Car source workspace

Use the [C-Car manifest](repositories/autoware.repos) from this repository's `c_car` branch. It follows the C-Car component branches and pins third-party dependency versions for ROS 2 Humble. The source snapshot in the top-level `src/` and `SOURCE_VERSIONS.repos` describe an earlier workspace.

## Import and check the sources

On Ubuntu 22.04 with Git, vcstool and read access to the organization's private repositories:

```bash
git clone --branch c_car https://github.com/Unmanned-Systems-Lab/autoware.c_car.git
cd autoware.c_car
mkdir -p workspace/src
vcs import --recursive workspace/src < repositories/autoware.repos
cd workspace
colcon list --base-paths src --names-only | grep -E '^(c_car_|ins_driver_c_car$)'
git -C src/launcher/autoware_launch log -1 --oneline
git -C src/universe/autoware_universe log -1 --oneline
```

Start with an empty `workspace/src`. Keep build, install and log outputs under `workspace/` and run colcon there. On a new machine install the project's Humble development environment with `setup-dev-env.sh`. A source import does not start any ROS nodes. Capture the exact revisions of an imported workspace with `vcs export --exact src` from `workspace/`.

| Source repository | Branch | C-Car components |
| --- | --- | --- |
| `Unmanned-Systems-Lab/autoware_launch.c_car` | `c_car` | Sensor kit, ground filtering, perception and fusion launch, UI and map generation, PCFA preview and CAN vehicle interface |
| `Unmanned-Systems-Lab/autoware_universe.ez21` | `c_car` | Autoware perception changes including YOLOX ROI thresholds and lidar processing |
| `Unmanned-Systems-Lab/ins_driver.c_car` | `c_car` | GI5651 CAN decoder, navigation quality and localization bridge |
| `Unmanned-Systems-Lab/c_car_vehicle_description` | `main` | Vehicle model and motion limits |

### New-machine inputs

* **Perception.** Read the launcher [C-Car guide](https://github.com/Unmanned-Systems-Lab/autoware_launch.c_car/blob/c_car/README_C_CAR.md) for the sensor topics, calibration override and stationary preview. The private launcher includes the TIMOO ROS 2 driver and its matching SDK source. The live driver requires libpcap and a network interface configured for the lidar. Set the Sony camera stream URL locally. The Autoware setup artifacts supply the YOLOX ONNX model and labels; the guide lists the matching checksums. Check the lidar-to-vehicle mounting pose before relying on measured distances.
* **Localization and maps.** Set the map origin and verified installations for the target vehicle locally. The distributed INS configuration uses an inactive reference origin. Site maps, credentials and live recordings are separate deployment inputs.
* **CAN vehicle interface.** The formal interface defaults to receive-only. See the [vehicle interface bench manual](https://github.com/Unmanned-Systems-Lab/autoware_launch.c_car/blob/c_car/vehicle/c_car_vehicle_launch/c_car_vehicle_launch/docs/CAN_INTERFACE_BENCH.md) for a manually selected bench session.
* **Navigation preview.** The C-Car avoidance packages are distributed with the private launcher. The field PCFA preview uses a 7 m planning radius; the vehicle motion profile specifies 10 m. The preview setting does not certify a physical minimum turning radius.

## Existing-base overlay

On the development machine, `~/autoware.repos_workspace/install` supplies the previously built Autoware packages. The separate `~/c_car_workspace` supplies the new description, sensor kit, launch packages and visualization bridge. Build that overlay against the existing base, then verify its package resolution:

```bash
source ~/autoware.c_car_dependencies/setup.bash
source ~/autoware.repos_workspace/install/setup.bash
cd ~/c_car_workspace
colcon build --packages-select \
  c_car_vehicle_description c_car_vehicle_launch \
  c_car_sensor_kit_description c_car_sensor_kit_launch ins_driver_c_car autoware_launch \
  --allow-overriding autoware_launch --symlink-install
source install/setup.bash
ros2 pkg prefix c_car_vehicle_description
ros2 pkg prefix c_car_vehicle_launch
ros2 pkg prefix c_car_sensor_kit_description
ros2 pkg prefix c_car_sensor_kit_launch
ros2 pkg prefix ins_driver_c_car
ros2 pkg prefix autoware_launch
~/autoware.c_car-c_car/start_planning_simulation.bash
```

The six prefixes should resolve under `~/c_car_workspace/install`. The helper explicitly loads the dependencies, the base install, and then the new overlay. `AUTOWARE_BASE_WS` and `AUTOWARE_WS` override those workspace paths; `MAP_PATH` or the first argument selects another map. `VEHICLE_MODEL` and `SENSOR_MODEL` default to `c_car_vehicle` and `c_car_sensor_kit`, with no automatic model selection.

The helper defaults to ROS domain `188` and localhost-only discovery to isolate it from the earlier simulation on domain `187`. Its CycloneDDS profile is copied unchanged from that simulation. RViz loads `config/planning_simulation/c_car.rviz`, with the robot visible at alpha 1, a transient-local robot-description subscription, and the left Control panel visible. Set `RVIZ=false` for a headless launch.

## GI5651 GNSS/INS

Planning simulation explicitly disables `launch_ins_driver` and `launch_ins_localization_bridge` when it includes the main launch. It continues to use simulated localization; starting the ring-map helper does not open a real CAN receiver. The general Autoware entry point retains passive INS reception in its real-localization mode, with the localization bridge disabled by default. The planning auxiliary publishers and freespace coordinator are installed by `autoware_launch`, so they do not require the old INS package.

After sourcing the base and c_car overlays above, inspect only the real receiver with:

```bash
ros2 launch autoware_launch ins_localization_component.launch.xml
```

This component defaults to `launch_ins_driver=true`, `launch_ins_localization_bridge=false`, and loads `ins_driver_c_car/config/driver.yaml`. The configured interface is `can0`; it must already be configured at the receiver's bitrate. The receiver does not configure the CAN interface or transmit control frames. This command starts real CAN reception, not the Autoware full stack.

Use one parameter file for the receiver and the optional bridge:

```bash
cp "$(ros2 pkg prefix ins_driver_c_car)/share/ins_driver_c_car/config/driver.yaml" /tmp/c_car-ins.yaml
ros2 launch autoware_launch ins_localization_component.launch.xml \
  ins_driver_param_path:=/tmp/c_car-ins.yaml
```

Both nodes load their sections from that file. Empty optional launch overrides preserve the YAML values, including measured mounting parameters and quality settings. The main `autoware.launch.xml` forwards the same `ins_driver_param_path` and `launch_ins_localization_bridge` arguments. The map UI uses the new `ins_driver_c_car` node and configuration keys; saving a reference origin preserves the other driver settings and does not confirm a navigation Flag, units, or axes.

The driver accepts GI5651 navigation status `0x4B` or `0x5B` for the navigation-quality check. Fresh data, axle and IMU conventions, the INS mounting transform and the map alignment have separate checks. Configure the reference origin for the current site, verify the mounting parameters and use the driver's diagnostics to confirm which outputs are valid.

```bash
ros2 topic echo /sensing/ins/quality --once
ros2 topic echo /sensing/ins/raw_nav_sat_fix --once
ros2 topic echo /sensing/imu/imu_raw --once
```

The optional Autoware bridge is enabled with `launch_ins_localization_bridge:=true`. Enabling its process does not make localization valid: fresh driver quality, valid odometry, measured extrinsics and verified map-origin alignment are still required by the YAML policy. Missing or expired quality prevents trusted localization and TF and returns the initialization state to uninitialized. Keep these confirmations unset until the corresponding measurements and device definitions are available.

## Model and visualization

The source model is `base3`, with wheelbase `L = 1.98 m` and overall length `3.74 m`. The c_car visualization node publishes all ten model joint states, so this overlay does not require `joint_state_publisher`. The bridge animates the displayed model from simulated status; it does not command hardware. Planning and control in this launch operate on the software simulator.

Use RViz's initial-pose and goal tools, then Autoware Control / Auto, to test the ring-map planning simulation. Source the same overlays and set `ROS_DOMAIN_ID=188`, `ROS_LOCALHOST_ONLY=1`, and the helper's `CYCLONEDDS_URI` in any separate inspection shell. `ros2 topic echo /joint_states --once` should show the model's ten joints.
