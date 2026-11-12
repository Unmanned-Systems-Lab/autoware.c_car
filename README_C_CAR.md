# c_car planning simulation

This `autoware.c_car` `c_car` branch keeps the earlier source snapshot intact and adds a separate c_car overlay and ring-map simulation helper.

The two c_car repositories in `repositories/autoware.repos` are:

| Repository | Ref | Manifest path |
| --- | --- | --- |
| `Unmanned-Systems-Lab/autoware_launch.c_car` | `c_car` | `launcher/autoware_launch` |
| `Unmanned-Systems-Lab/c_car_vehicle_description` | `main` | `vehicle/c_car_vehicle_description` |

Private repositories require normal GitHub Git access. The GitHub connection in chat does not configure credentials on the remote machine. Import the manifest into a new, empty source workspace; do not import over the source snapshot or an existing checkout with local changes.

## Existing-base overlay

On the development machine, `~/autoware.repos_workspace/install` supplies the previously built Autoware packages. The separate `~/c_car_workspace` supplies the new description, sensor kit, launch packages and visualization bridge. Build that overlay against the existing base, then verify its package resolution:

```bash
source ~/autoware.c_car_dependencies/setup.bash
source ~/autoware.repos_workspace/install/setup.bash
cd ~/c_car_workspace
colcon build --packages-select \
  c_car_vehicle_description c_car_vehicle_launch \
  c_car_sensor_kit_description c_car_sensor_kit_launch autoware_launch \
  --allow-overriding autoware_launch --symlink-install
source install/setup.bash
ros2 pkg prefix c_car_vehicle_description
ros2 pkg prefix c_car_vehicle_launch
ros2 pkg prefix c_car_sensor_kit_description
ros2 pkg prefix c_car_sensor_kit_launch
ros2 pkg prefix autoware_launch
~/autoware.c_car-c_car/start_planning_simulation.bash
```

The five prefixes should resolve under `~/c_car_workspace/install`. The helper explicitly loads the dependencies, the base install, and then the new overlay. `AUTOWARE_BASE_WS` and `AUTOWARE_WS` override those workspace paths; `MAP_PATH` or the first argument selects another map. `VEHICLE_MODEL` and `SENSOR_MODEL` default to `c_car_vehicle` and `c_car_sensor_kit`, with no automatic model selection.

The helper defaults to ROS domain `188` and localhost-only discovery to isolate it from the earlier simulation on domain `187`. Its CycloneDDS profile is copied unchanged from that simulation. RViz loads `config/planning_simulation/c_car.rviz`, with the robot visible at alpha 1, a transient-local robot-description subscription, and the left Control panel visible. Set `RVIZ=false` for a headless launch.

## Model and visualization

The source model is `base3`, with wheelbase `L = 1.98 m` and overall length `3.74 m`. The c_car visualization node publishes all ten model joint states, so this overlay does not require `joint_state_publisher`. The bridge animates the displayed model from simulated status; it does not command hardware. Planning and control in this launch operate on the software simulator.

Use RViz's initial-pose and goal tools, then Autoware Control / Auto, to test the ring-map planning simulation. Source the same overlays and set `ROS_DOMAIN_ID=188`, `ROS_LOCALHOST_ONLY=1`, and the helper's `CYCLONEDDS_URI` in any separate inspection shell. `ros2 topic echo /joint_states --once` should show the model's ten joints.
