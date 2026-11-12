# c_car sensor-kit launch resources

The general Autoware system and sensing launch files resolve
`<sensor_model>_launch` before planning-simulation conditions have removed
physical sensing. With `sensor_model:=c_car_sensor_kit`, this package supplies
that resource identity and the expected paths:

- `launch/sensing.launch.xml`: accepts the general sensing wrapper arguments.
- `config/dummy_diag_publisher/sensor_kit.param.yaml`: standard simulation
  diagnostic placeholder used by `tier4_system_launch`.

Planning simulation uses `launch_sensing:=false`. If the sensing wrapper is
explicitly included with `launch_driver:=false`, it launches no sensor nodes.
The default `launch_driver:=true` terminates with a clear unconfigured-hardware
error. No CAN, LiDAR, camera, IMU or GNSS driver is provided or inherited.
The physical model's IMU geometry stays in `c_car_vehicle_description`; its
sensor-kit description contributes no uncalibrated external sensor frames.
