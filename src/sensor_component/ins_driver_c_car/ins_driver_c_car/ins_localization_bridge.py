from copy import deepcopy
import math
import time
from typing import Optional, Tuple

import rclpy
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from geometry_msgs.msg import AccelWithCovarianceStamped, PoseStamped, TransformStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


def _quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / norm, y / norm, z / norm, w / norm)


def _conjugate_quaternion(quaternion: Quaternion) -> Quaternion:
    x, y, z, w = quaternion
    return (-x, -y, -z, w)


def _multiply_quaternion(left: Quaternion, right: Quaternion) -> Quaternion:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quaternion_to_rotation_matrix(quaternion: Quaternion) -> Tuple[Tuple[float, float, float], ...]:
    x, y, z, w = _normalize_quaternion(quaternion)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def _transpose_matrix(matrix: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def _matrix_vector_multiply(matrix: Tuple[Tuple[float, float, float], ...], vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _vector_add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _vector_scale(vector: Vector3, scale: float) -> Vector3:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _vector_cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _stamp_to_seconds(sec: int, nanosec: int) -> float:
    return float(sec) + float(nanosec) * 1e-9


def _vector_sub(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def odometry_rejection_reason(msg, now_sec, max_age_sec):
    """Pure validation used before trusted odometry or TF is forwarded."""
    stamp = _stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
    age = now_sec - stamp
    if stamp <= 0.0 or not math.isfinite(age) or age < -0.05 or age > max_age_sec:
        return "odometry timestamp is stale, absent, or in the future"
    p, q, twist = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
    values = [p.x, p.y, p.z, q.x, q.y, q.z, q.w, twist.linear.x, twist.linear.y,
              twist.linear.z, twist.angular.x, twist.angular.y, twist.angular.z]
    if not all(math.isfinite(v) for v in values):
        return "odometry contains non-finite values"
    norm2 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w
    if abs(norm2 - 1.0) > 1e-3:
        return "odometry quaternion is not normalized"
    for covariance in (msg.pose.covariance, msg.twist.covariance):
        if len(covariance) != 36 or not all(math.isfinite(v) for v in covariance):
            return "odometry covariance invalid"
        if any(covariance[i*7] <= 0.0 for i in range(6)):
            return "odometry covariance missing or not explicitly configured"
    return ""


def diagnostic_quality_is_valid(msg):
    for status in msg.status:
        if status.name == "gi5651/navigation":
            values = {v.key: v.value for v in status.values}
            return (status.level == DiagnosticStatus.OK and values.get("valid") == "true"
                    and values.get("axes_verified") == "true")
    return False


def _skew(vector):
    x, y, z = vector
    return ((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0))


def _transform_covariance(covariance, jacobian):
    return [sum(jacobian[i][k] * covariance[k*6+l] * jacobian[j][l]
                for k in range(6) for l in range(6)) for i in range(6) for j in range(6)]


class InsLocalizationBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("ins_localization_bridge")

        self.declare_parameter("input_odom_topic", "/sensing/ins/odometry")
        self.declare_parameter("output_odom_topic", "/localization/kinematic_state")
        self.declare_parameter("output_base_link_odom_topic", "")
        self.declare_parameter("output_accel_topic", "/localization/acceleration")
        self.declare_parameter("output_pose_topic", "/localization/pose_twist_fusion_filter/pose")
        self.declare_parameter(
            "output_initialization_state_topic", "/localization/initialization_state"
        )
        self.declare_parameter("output_frame_id", "map")
        self.declare_parameter("output_child_frame_id", "base_link")
        self.declare_parameter("publish_tf", False)
        self.declare_parameter("input_quality_topic", "/sensing/ins/quality")
        self.declare_parameter("input_child_frame_id", "ins_link")
        self.declare_parameter("require_quality", True)
        self.declare_parameter("quality_timeout_sec", 0.2)
        self.declare_parameter("max_quality_skew_sec", 0.1)
        self.declare_parameter("odom_timeout_sec", 0.2)
        self.declare_parameter("extrinsics_verified", False)
        self.declare_parameter("map_origin_verified", False)
        self.declare_parameter("acceleration_covariance_diag", [25.0, 25.0, 25.0, 1.0, 1.0, 1.0])
        self.declare_parameter("min_accel_dt_sec", 1e-3)
        self.declare_parameter("max_accel_dt_sec", 1.0)
        self.declare_parameter("base_link_to_ins_x", 0.0)
        self.declare_parameter("base_link_to_ins_y", 0.0)
        self.declare_parameter("base_link_to_ins_z", 0.0)
        self.declare_parameter("base_link_to_ins_roll", 0.0)
        self.declare_parameter("base_link_to_ins_pitch", 0.0)
        self.declare_parameter("base_link_to_ins_yaw", 0.0)

        self.input_odom_topic = str(self.get_parameter("input_odom_topic").value)
        self.output_odom_topic = str(self.get_parameter("output_odom_topic").value)
        self.output_base_link_odom_topic = str(
            self.get_parameter("output_base_link_odom_topic").value
        )
        self.output_accel_topic = str(self.get_parameter("output_accel_topic").value)
        self.output_pose_topic = str(self.get_parameter("output_pose_topic").value)
        self.output_initialization_state_topic = str(
            self.get_parameter("output_initialization_state_topic").value
        )
        self.output_frame_id = str(self.get_parameter("output_frame_id").value)
        self.output_child_frame_id = str(self.get_parameter("output_child_frame_id").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.input_quality_topic = str(self.get_parameter("input_quality_topic").value)
        self.input_child_frame_id = str(self.get_parameter("input_child_frame_id").value)
        self.require_quality = bool(self.get_parameter("require_quality").value)
        self.quality_timeout_sec = float(self.get_parameter("quality_timeout_sec").value)
        self.max_quality_skew_sec = float(self.get_parameter("max_quality_skew_sec").value)
        self.odom_timeout_sec = float(self.get_parameter("odom_timeout_sec").value)
        self.extrinsics_verified = bool(self.get_parameter("extrinsics_verified").value)
        self.map_origin_verified = bool(self.get_parameter("map_origin_verified").value)
        self.acceleration_covariance_diag = list(self.get_parameter("acceleration_covariance_diag").value)
        if len(self.acceleration_covariance_diag) != 6 or any(
                not math.isfinite(v) or v <= 0.0 for v in self.acceleration_covariance_diag):
            raise ValueError("acceleration_covariance_diag requires six positive finite variances")
        if not self.require_quality:
            raise ValueError("The c_car bridge requires navigation quality gating")
        if not self.input_quality_topic or not self.input_child_frame_id:
            raise ValueError("Quality topic and input child frame must not be empty")
        for value in (self.quality_timeout_sec, self.max_quality_skew_sec, self.odom_timeout_sec):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("Quality and odometry timeouts must be finite and positive")
        self.min_accel_dt_sec = float(self.get_parameter("min_accel_dt_sec").value)
        self.max_accel_dt_sec = float(self.get_parameter("max_accel_dt_sec").value)
        self.base_link_to_ins_translation: Vector3 = (
            float(self.get_parameter("base_link_to_ins_x").value),
            float(self.get_parameter("base_link_to_ins_y").value),
            float(self.get_parameter("base_link_to_ins_z").value),
        )
        self.base_link_to_ins_rpy: Vector3 = (
            float(self.get_parameter("base_link_to_ins_roll").value),
            float(self.get_parameter("base_link_to_ins_pitch").value),
            float(self.get_parameter("base_link_to_ins_yaw").value),
        )
        if not all(math.isfinite(v) for v in (*self.base_link_to_ins_translation, *self.base_link_to_ins_rpy)):
            raise ValueError("Extrinsic values must be finite")

        if not self.input_odom_topic:
            raise ValueError("Parameter 'input_odom_topic' must not be empty.")
        if not self.output_odom_topic:
            raise ValueError("Parameter 'output_odom_topic' must not be empty.")
        if not self.output_accel_topic:
            raise ValueError("Parameter 'output_accel_topic' must not be empty.")
        if not self.output_pose_topic:
            raise ValueError("Parameter 'output_pose_topic' must not be empty.")
        if not self.output_initialization_state_topic:
            raise ValueError("Parameter 'output_initialization_state_topic' must not be empty.")
        if not self.output_frame_id:
            raise ValueError("Parameter 'output_frame_id' must not be empty.")
        if not self.output_child_frame_id:
            raise ValueError("Parameter 'output_child_frame_id' must not be empty.")
        if self.min_accel_dt_sec <= 0.0:
            raise ValueError("Parameter 'min_accel_dt_sec' must be positive.")
        if self.max_accel_dt_sec <= self.min_accel_dt_sec:
            raise ValueError(
                "Parameter 'max_accel_dt_sec' must be greater than 'min_accel_dt_sec'."
            )

        self._quaternion_base_to_ins = _normalize_quaternion(
            _quaternion_from_euler(*self.base_link_to_ins_rpy)
        )
        self._quaternion_ins_to_base = _conjugate_quaternion(self._quaternion_base_to_ins)
        self._rotation_base_to_ins = _quaternion_to_rotation_matrix(self._quaternion_base_to_ins)
        self._rotation_ins_to_base = _transpose_matrix(self._rotation_base_to_ins)
        self._translation_ins_to_base = _matrix_vector_multiply(
            self._rotation_ins_to_base, _vector_scale(self.base_link_to_ins_translation, -1.0)
        )

        self.publisher = self.create_publisher(Odometry, self.output_odom_topic, 10)
        self.acceleration_publisher = self.create_publisher(
            AccelWithCovarianceStamped, self.output_accel_topic, 10
        )
        self.pose_publisher = self.create_publisher(PoseStamped, self.output_pose_topic, 10)
        localization_state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.localization_init_state_publisher = self.create_publisher(
            LocalizationInitializationState,
            self.output_initialization_state_topic,
            localization_state_qos,
        )
        self.base_link_odom_publisher = None
        if self.output_base_link_odom_topic:
            if self.output_base_link_odom_topic == self.input_odom_topic:
                self.get_logger().warning(
                    "output_base_link_odom_topic is identical to input_odom_topic (%s); "
                    "skip extra publication to avoid self-feedback."
                    % self.input_odom_topic
                )
            else:
                self.base_link_odom_publisher = self.create_publisher(
                    Odometry, self.output_base_link_odom_topic, 10
                )

        self.subscription = self.create_subscription(
            Odometry, self.input_odom_topic, self._on_odometry, 10
        )
        self.quality_subscription = self.create_subscription(
            DiagnosticArray, self.input_quality_topic, self._on_quality, 10)
        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self._forwarded_count = 0
        self._previous_twist_stamp_sec: Optional[float] = None
        self._previous_linear_velocity: Optional[Vector3] = None
        self._previous_angular_velocity: Optional[Vector3] = None
        self._quality_valid = False
        self._quality_received = None
        self._quality_stamp_sec = None
        self._last_valid_odom_received = None
        self._initialized = False
        self._watchdog_timer = self.create_timer(0.05, self._watchdog)

        self.get_logger().info(
            "Bridging %s -> %s with frames %s -> %s"
            % (
                self.input_odom_topic,
                self.output_odom_topic,
                self.output_frame_id,
                self.output_child_frame_id,
            )
        )
        if self.base_link_odom_publisher is not None:
            self.get_logger().info(
                "Publishing converted base_link odometry on %s" % self.output_base_link_odom_topic
            )
        self.get_logger().info("Publishing acceleration on %s" % self.output_accel_topic)
        self.get_logger().info("Publishing pose on %s" % self.output_pose_topic)
        self.get_logger().info(
            "Publishing localization initialization state on %s"
            % self.output_initialization_state_topic
        )
        self.get_logger().info(
            "Using base_link->ins extrinsic xyz=(%.6f, %.6f, %.6f), rpy=(%.6f, %.6f, %.6f)"
            % (
                self.base_link_to_ins_translation[0],
                self.base_link_to_ins_translation[1],
                self.base_link_to_ins_translation[2],
                self.base_link_to_ins_rpy[0],
                self.base_link_to_ins_rpy[1],
                self.base_link_to_ins_rpy[2],
            )
        )
        self._publish_localization_initialization_state(
            self.get_clock().now().to_msg(), LocalizationInitializationState.UNINITIALIZED
        )

    def _convert_ins_odom_to_base_link(self, msg: Odometry) -> Odometry:
        output = deepcopy(msg)
        output.child_frame_id = self.output_child_frame_id

        quaternion_world_ins = _normalize_quaternion(
            (
                output.pose.pose.orientation.x,
                output.pose.pose.orientation.y,
                output.pose.pose.orientation.z,
                output.pose.pose.orientation.w,
            )
        )
        rotation_world_ins = _quaternion_to_rotation_matrix(quaternion_world_ins)
        quaternion_world_base = _normalize_quaternion(
            _multiply_quaternion(quaternion_world_ins, self._quaternion_ins_to_base)
        )

        position_world_ins = (
            output.pose.pose.position.x,
            output.pose.pose.position.y,
            output.pose.pose.position.z,
        )
        position_world_base = _vector_add(
            position_world_ins,
            _matrix_vector_multiply(rotation_world_ins, self._translation_ins_to_base),
        )
        output.pose.pose.position.x = position_world_base[0]
        output.pose.pose.position.y = position_world_base[1]
        output.pose.pose.position.z = position_world_base[2]
        output.pose.pose.orientation.x = quaternion_world_base[0]
        output.pose.pose.orientation.y = quaternion_world_base[1]
        output.pose.pose.orientation.z = quaternion_world_base[2]
        output.pose.pose.orientation.w = quaternion_world_base[3]

        angular_velocity_ins = (
            output.twist.twist.angular.x,
            output.twist.twist.angular.y,
            output.twist.twist.angular.z,
        )
        linear_velocity_ins = (
            output.twist.twist.linear.x,
            output.twist.twist.linear.y,
            output.twist.twist.linear.z,
        )
        linear_velocity_base_in_ins = _vector_add(
            linear_velocity_ins, _vector_cross(angular_velocity_ins, self._translation_ins_to_base)
        )
        linear_velocity_base = _matrix_vector_multiply(
            self._rotation_base_to_ins, linear_velocity_base_in_ins
        )
        angular_velocity_base = _matrix_vector_multiply(
            self._rotation_base_to_ins, angular_velocity_ins
        )
        output.twist.twist.linear.x = linear_velocity_base[0]
        output.twist.twist.linear.y = linear_velocity_base[1]
        output.twist.twist.linear.z = linear_velocity_base[2]
        output.twist.twist.angular.x = angular_velocity_base[0]
        output.twist.twist.angular.y = angular_velocity_base[1]
        output.twist.twist.angular.z = angular_velocity_base[2]

        # q_base_ins maps sensor-coordinate vectors to base-coordinate vectors.
        # v_base = R_base_ins (v_ins + omega_ins x t_ins_base).
        rotation = self._rotation_base_to_ins
        skew = _skew(self._translation_ins_to_base)
        twist_jacobian = [[0.0] * 6 for _ in range(6)]
        for i in range(3):
            for j in range(3):
                twist_jacobian[i][j] = rotation[i][j]
                twist_jacobian[i+3][j+3] = rotation[i][j]
                twist_jacobian[i][j+3] = -sum(rotation[i][k] * skew[k][j] for k in range(3))
        output.twist.covariance = _transform_covariance(msg.twist.covariance, twist_jacobian)
        lever_world = _matrix_vector_multiply(rotation_world_ins, self._translation_ins_to_base)
        lever_skew = _skew(lever_world)
        pose_jacobian = [[float(i == j) for j in range(6)] for i in range(6)]
        for i in range(3):
            for j in range(3):
                pose_jacobian[i][j+3] = -lever_skew[i][j]
        output.pose.covariance = _transform_covariance(msg.pose.covariance, pose_jacobian)

        return output

    def _publish_acceleration(self, base_link_msg: Odometry) -> None:
        linear_velocity: Vector3 = (
            base_link_msg.twist.twist.linear.x,
            base_link_msg.twist.twist.linear.y,
            base_link_msg.twist.twist.linear.z,
        )
        angular_velocity: Vector3 = (
            base_link_msg.twist.twist.angular.x,
            base_link_msg.twist.twist.angular.y,
            base_link_msg.twist.twist.angular.z,
        )

        stamp_sec = _stamp_to_seconds(
            base_link_msg.header.stamp.sec, base_link_msg.header.stamp.nanosec
        )
        if stamp_sec <= 0.0:
            stamp_sec = self.get_clock().now().nanoseconds / 1e9

        linear_acceleration = None
        angular_acceleration = None
        if (
            self._previous_twist_stamp_sec is not None
            and self._previous_linear_velocity is not None
            and self._previous_angular_velocity is not None
        ):
            dt = stamp_sec - self._previous_twist_stamp_sec
            if self.min_accel_dt_sec <= dt <= self.max_accel_dt_sec:
                linear_acceleration = _vector_scale(
                    _vector_sub(linear_velocity, self._previous_linear_velocity), 1.0 / dt
                )
                # Differentiate a body-expressed velocity: restore the rotating-frame term.
                linear_acceleration = _vector_add(
                    linear_acceleration, _vector_cross(angular_velocity, linear_velocity))
                angular_acceleration = _vector_scale(
                    _vector_sub(angular_velocity, self._previous_angular_velocity), 1.0 / dt
                )

        self._previous_twist_stamp_sec = stamp_sec
        self._previous_linear_velocity = linear_velocity
        self._previous_angular_velocity = angular_velocity

        if linear_acceleration is None or angular_acceleration is None:
            return

        accel_msg = AccelWithCovarianceStamped()
        accel_msg.header = deepcopy(base_link_msg.header)
        accel_msg.header.frame_id = self.output_child_frame_id
        accel_msg.accel.accel.linear.x = linear_acceleration[0]
        accel_msg.accel.accel.linear.y = linear_acceleration[1]
        accel_msg.accel.accel.linear.z = linear_acceleration[2]
        accel_msg.accel.accel.angular.x = angular_acceleration[0]
        accel_msg.accel.accel.angular.y = angular_acceleration[1]
        accel_msg.accel.accel.angular.z = angular_acceleration[2]
        for i, variance in enumerate(self.acceleration_covariance_diag):
            accel_msg.accel.covariance[i*7] = variance
        self.acceleration_publisher.publish(accel_msg)

    def _publish_localization_initialization_state(
        self, stamp, state: int
    ) -> None:
        init_msg = LocalizationInitializationState()
        init_msg.stamp = stamp
        init_msg.state = state
        self.localization_init_state_publisher.publish(init_msg)

    def _on_odometry(self, msg: Odometry) -> None:
        if self._bridge_rejection_reason(msg):
            self._invalidate()
            return
        self._last_valid_odom_received = time.monotonic()
        self._initialized = True
        self._publish_localization_initialization_state(
            msg.header.stamp, LocalizationInitializationState.INITIALIZED
        )

        base_link_msg = self._convert_ins_odom_to_base_link(msg)
        self._publish_acceleration(base_link_msg)
        if self.base_link_odom_publisher is not None:
            self.base_link_odom_publisher.publish(base_link_msg)

        out_msg = deepcopy(base_link_msg)
        out_msg.header.frame_id = self.output_frame_id
        out_msg.child_frame_id = self.output_child_frame_id
        self.publisher.publish(out_msg)

        pose_msg = PoseStamped()
        pose_msg.header = out_msg.header
        pose_msg.pose = out_msg.pose.pose
        self.pose_publisher.publish(pose_msg)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = out_msg.header
            transform.child_frame_id = out_msg.child_frame_id
            transform.transform.translation.x = out_msg.pose.pose.position.x
            transform.transform.translation.y = out_msg.pose.pose.position.y
            transform.transform.translation.z = out_msg.pose.pose.position.z
            transform.transform.rotation = out_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

        self._forwarded_count += 1
        if self._forwarded_count == 1 or self._forwarded_count % 50 == 0:
            self.get_logger().info(
                "Forwarded %d odom messages to %s"
                % (self._forwarded_count, self.output_odom_topic)
            )

    def _on_quality(self, msg):
        self._quality_received = time.monotonic()
        self._quality_stamp_sec = _stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
        self._quality_valid = diagnostic_quality_is_valid(msg)
        age = self.get_clock().now().nanoseconds / 1e9 - self._quality_stamp_sec
        if not math.isfinite(age) or age < -0.05 or age > self.quality_timeout_sec:
            self._quality_valid = False
        if not self._quality_valid:
            self._invalidate()

    def _bridge_rejection_reason(self, msg):
        if not self.extrinsics_verified:
            return "sensor extrinsics have not been verified"
        if not self.map_origin_verified:
            return "local ENU/map origin alignment has not been verified"
        if msg.child_frame_id != self.input_child_frame_id:
            return "unexpected input odometry child frame"
        reason = odometry_rejection_reason(msg, self.get_clock().now().nanoseconds / 1e9,
                                           self.odom_timeout_sec)
        if reason:
            return reason
        if not self._quality_valid or self._quality_received is None:
            return "no valid GI5651 quality"
        if time.monotonic() - self._quality_received > self.quality_timeout_sec:
            return "GI5651 quality expired"
        stamp = _stamp_to_seconds(msg.header.stamp.sec, msg.header.stamp.nanosec)
        if self._quality_stamp_sec is None or abs(stamp - self._quality_stamp_sec) > self.max_quality_skew_sec:
            return "odometry and quality timestamps are not aligned"
        return ""

    def _invalidate(self):
        if self._initialized:
            self._publish_localization_initialization_state(
                self.get_clock().now().to_msg(), LocalizationInitializationState.UNINITIALIZED)
        self._initialized = False
        self._last_valid_odom_received = None
        self._previous_twist_stamp_sec = None
        self._previous_linear_velocity = None
        self._previous_angular_velocity = None

    def _watchdog(self):
        current = time.monotonic()
        quality_stale = self._quality_received is None or current - self._quality_received > self.quality_timeout_sec
        odom_stale = self._last_valid_odom_received is None or current - self._last_valid_odom_received > self.odom_timeout_sec
        if quality_stale or odom_stale or not self._quality_valid:
            self._invalidate()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InsLocalizationBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
