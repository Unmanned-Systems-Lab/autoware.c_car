# C-Car planning-simulation resources

This `c_car` branch adds an independent vehicle launch resource package and an
empty sensor-kit description plus a resource-only sensor-kit launch package.
Existing EZ21 packages retain their original files.

Use `vehicle_model:=c_car_vehicle`, `sensor_model:=c_car_sensor_kit` and
`vehicle_simulation:=true` with the planning simulator. Only C-Car launches its
visual joint-state publisher. The C-Car hardware-interface launch fails closed.

The vehicle geometry, parameters, and display joint-state script are provided by
the separate `c_car_vehicle_description` repository/package. This launcher does
not duplicate that model or attach the legacy M1 sensor.

The `c_car_sensor_kit_launch` package is required even for planning simulation:
common system/sensing launch files resolve its package share and simulation
diagnostic configuration before driver conditions are applied. Build/install it
along with `c_car_sensor_kit_description`, `c_car_vehicle_launch`,
`c_car_vehicle_description` and `autoware_launch`. It provides no physical
sensing nodes; requesting `launch_driver:=true` fails explicitly.

## GI5651 INS integration

The C-Car manifest `repositories/autoware.repos` imports
`Unmanned-Systems-Lab/ins_driver.c_car` at branch `c_car` into
`src/sensor_component/ins_driver_c_car`. Build the ROS package
`ins_driver_c_car` along with this launcher. The old `ins_driver_ez21` package
is not required by the C-Car startup configuration.

Real INS localization uses the new package's `config/driver.yaml` for both the
receiver and optional localization bridge. `ins_driver_param_path` can select a
calibrated copy. The default interface is `can0`; the receiver does not configure
the CAN interface or transmit control frames.

`launch_ins_localization_bridge` defaults to `false`. Enabling the bridge does
not bypass the driver's Flag/axis checks or the bridge's quality, extrinsic and
map-origin checks. Do not replace calibration by changing verification switches.
The protocol does not supply a Flag code table, and the CAN acceleration units
and installation conventions still require confirmation for the actual device.

Planning simulation explicitly disables both the real INS receiver and bridge.
Its planning fallback and freespace helpers are installed by `autoware_launch`,
independent of either INS driver. The resource-only sensor kit remains unchanged;
the INS receiver is launched through the localization component, not through the
sensor-kit hardware launch.

The map UI now reads the `ins_driver_c_car` configuration and contacts node
`/ins_driver_c_car`. Saving a reference origin changes only
`reference_origin_active` and the three reference coordinates; it preserves all
Flag, unit, axis and bridge verification settings. A raw GPS message with
`NO_FIX`, invalid coordinates or an expired reception time is not used as the
current position or map origin. Config files must already exist; the UI does not
create a partial replacement for the driver defaults.
