# Scout localization support

`wheel_odometry_to_twist` supplies the native Autoware gyro odometer with real
Scout wheel-velocity observations. It subscribes to `/scout/wheel_odometry` and
publishes `/sensing/vehicle_velocity_converter/twist_with_covariance` in
`base_link`. There is no command publisher, odometry pose fusion, or timer that
invents a stationary vehicle when wheel feedback is missing.

The current Scout odometry has parent `scout_wheel_odom` and child
`scout_chassis_link`. Its twist belongs to the **child**, not the parent.
The vehicle URDF places the chassis origin at `[0.249, 0, 0.23479] m` in
`base_link`. For the current aligned frames, the rear-axle velocity is
`[vx, -0.249*wz, 0]`, and angular velocity is `[0, 0, wz]`.

The adapter obtains the transform at each measurement's timestamp. Its planar
measurement Jacobian is `J = [[R[:,0], r×R[:,2]], [0, R[:,2]]]`. It propagates
the reported 2×2 covariance of `[vx,wz]` through `J C J^T`. A source block of
all zeros means unspecified covariance and uses configured initial standard
deviations of 0.2 m/s and 0.1 rad/s. Four unobserved source directions receive
large variances, rotated into `base_link` after measurement propagation. The
height offset therefore does not turn unknown wheel roll/pitch into a measured
longitudinal velocity. These uncertainty defaults need real-data calibration.

Invalid covariance is rejected by default; `invalid_covariance_policy=fallback`
explicitly replaces it with configured uncertainty and emits a warning. Finite
symmetry/PSD checks, minimum positive measured variances, and output checks are
applied. Parent/child and TF IDs must match. Missing TF, nonplanar velocity,
invalid numbers, timestamps older than 0.5 seconds, over 0.1 seconds in the
future, and non-increasing timestamps produce no output. Input subscription is
best effort (compatible with reliable and best-effort publishers); output is
reliable for the native gyro odometer.

```bash
ros2 run scout_localization_support wheel_odometry_to_twist
```

ROS parameters mirror `ConversionConfig` in `wheel_conversion.py`; topics are
parameters `input_topic` and `output_topic`. Configure the standalone read-only
`scout_base` driver outside this package. Do not run it alongside a second
driver publishing the same wheel topic.

For hardware, set `require_can_feedback:=true` and `can_interface:=can10`
(or the actual chassis interface). A separate nonblocking SocketCAN listener
accepts only standard Scout V2 motion feedback ID `0x221`, DLC 8; it never sends
CAN frames or changes the chassis mode. The first real motion-feedback frame is
required, and the monotonic receive age must not exceed
`can_feedback_timeout_sec` (default 0.5 s). A missing/faulted interface blocks
output until the node is restarted after repair. Unrelated traffic cannot
extend the gate. This protects against the existing Scout driver re-stamping
cached odometry at 50 Hz when CAN feedback stops. The adapter fuses no cached
odometry pose; it exposes only measured planar velocity. The default
`require_can_feedback=false` is reserved for bag/synthetic inputs where no real
CAN interface exists. The hardware launch must explicitly enable the gate.

`initial_pose_bridge` is for the standalone localization launch only. It sends
validated RViz `/initialpose` guesses to native `/localization/initialize`
(`autoware_localization_msgs/srv/InitializeLocalization`, method `AUTO`). It
does not publish `/initialpose3d`, disable native stop checks, or declare an
initialized state. Full Autoware already has ADAPI initialization and should
not start this additional bridge. Concurrent guesses are rejected. An unavailable
service is retried for up to 10 seconds; a response timeout after 30 seconds is
reported as unknown, with further guesses blocked until the outstanding native
result arrives. Success is logged only for `response.status.success == true`.

Pure conversion and pose-validation tests do not need ROS:

```bash
PYTHONPATH=. python3 -m unittest discover -s test -v
```

`imu_relay` provides reliable output for native gyro-odometer and distortion
corrector subscriptions. It accepts actual `/zed2i/zed_node/imu/data` with sensor
best-effort QoS and publishes `/scout/localization/imu` with reliable QoS, depth
100. The input can be overridden with `input_topic`. Frame, source timestamp,
calibrated angular velocity, acceleration and orientation are preserved. There
is no bias removal, axis conversion, calibration update, or synthetic data.
Only an all-zero gyro covariance is replaced with configurable isotropic
standard deviation 0.03 rad/s (the native IMU-corrector initial default);
reported covariance must be finite, symmetric and PSD, with a positive numerical
variance floor. A missing orientation marked by covariance[0] = -1 is retained,
while a gyro marked unavailable is rejected. Invalid readings and stale,
future or repeated timestamps are rejected using the same 0.5/0.1-second default
limits. Missing input produces no output.

`pointcloud_time_order` validates native 32-byte RoboSense XYZIRCAEDT records
and stably orders them by relative `UINT32` nanosecond `time_stamp` for native
Universe deskew. It reads `/rslidar_points` and writes only the private
`/scout/localization/points_ordered` topic, preserving fields, each valid point's
bytes and the scan header reference. Non-finite XYZ points and row padding are
removed only from this private output; the result is unorganized and dense.
Raw input is never modified. Channel IDs, field layout, duration and invalid
time sentinels are checked. Header timestamps must increase; restart this node
after resetting the sensor clock. This node requires system NumPy.
