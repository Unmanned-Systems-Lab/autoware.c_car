# C-Car planning-simulation launch resources

Select `vehicle_model:=c_car_vehicle` and `sensor_model:=c_car_sensor_kit`
with `vehicle_simulation:=true` for the planning simulator.

The independent `c_car_vehicle_description` package provides the vehicle model,
vehicle parameters, and `visual_joint_state_publisher.py`. The visualization launch
loads its vehicle-info YAML and converts simulated steering/velocity reports into
joint states. It publishes no vehicle-control commands. Do not run another
joint-state publisher for these same joints.

The hardware-interface launch fails immediately if selected. This resource package
contains no CAN driver or hardware interface. The empty C-Car sensor-kit description
adds no legacy M1 sensor; the vehicle description already includes its own IMU.
