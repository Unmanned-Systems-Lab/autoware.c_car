# c_car planning simulation

This `autoware.c_car` `c_car` branch provides the c_car overlay, ring-map simulation helper and independent GI5651 driver. The earlier EZ21 driver remains a separate package; c_car launch configuration uses `ins_driver_c_car`.

The three c_car repositories in `repositories/autoware.repos` are:

| Repository | Ref | Manifest path |
| --- | --- | --- |
| `Unmanned-Systems-Lab/autoware_launch.c_car` | `c_car` | `launcher/autoware_launch` |
| `Unmanned-Systems-Lab/c_car_vehicle_description` | `main` | `vehicle/c_car_vehicle_description` |
| `Unmanned-Systems-Lab/ins_driver.c_car` | `c_car` | `sensor_component/ins_driver_c_car` |

Private repositories require normal GitHub Git access. The GitHub connection in chat does not configure credentials on the remote machine. Import the manifest into a new, empty source workspace; do not import over the source snapshot or an existing checkout with local changes.

The checked-in source snapshot also contains `src/sensor_component/ins_driver_c_car` and the matching c_car launch resources, so a source-snapshot build can resolve the new package. `SOURCE_VERSIONS.repos` remains the historical provenance record. Use `repositories/autoware.repos` for the current c_car dependency references; its new driver entry is `https://github.com/Unmanned-Systems-Lab/ins_driver.c_car.git`, branch `c_car`, at the independent path `sensor_component/ins_driver_c_car`.

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

The supplied default Flag lists are empty, `axes_verified=false`, and `acceleration_units_verified=false`. Raw GNSS and IMU can be observed, but this default configuration reports `NO_FIX` and does not produce trusted odometry. Configure only verified device Flag values and measured coordinate conventions. Acceleration units and GPS-to-UTC timing are explicit driver options; neither is inferred from receiving frames. Unknown IMU quantities remain marked unknown.

```bash
ros2 topic echo /sensing/ins/quality --once
ros2 topic echo /sensing/ins/raw_nav_sat_fix --once
ros2 topic echo /sensing/imu/imu_raw --once
```

The optional Autoware bridge is enabled with `launch_ins_localization_bridge:=true`. Enabling its process does not make localization valid: fresh driver quality, valid odometry, measured extrinsics and verified map-origin alignment are still required by the YAML policy. Missing or expired quality prevents trusted localization and TF and returns the initialization state to uninitialized. Keep these confirmations unset until the corresponding measurements and device definitions are available.

## Model and visualization

The source model is `base3`, with wheelbase `L = 1.98 m` and overall length `3.74 m`. The c_car visualization node publishes all ten model joint states, so this overlay does not require `joint_state_publisher`. The bridge animates the displayed model from simulated status; it does not command hardware. Planning and control in this launch operate on the software simulator.

Use RViz's initial-pose and goal tools, then Autoware Control / Auto, to test the ring-map planning simulation. Source the same overlays and set `ROS_DOMAIN_ID=188`, `ROS_LOCALHOST_ONLY=1`, and the helper's `CYCLONEDDS_URI` in any separate inspection shell. `ros2 topic echo /joint_states --once` should show the model's ten joints.
