"""Publish a localization velocity observation from real Scout wheel odometry."""

from dataclasses import fields
import socket
import struct
import time

from geometry_msgs.msg import TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from .wheel_conversion import (
    ConversionConfig, InvalidWheelSample, MountTransform, WheelSample, convert_wheel_sample,
)
from .can_feedback import SCOUT_MOTION_FILTER_MASK, SCOUT_V2_MOTION_ID, CanFeedbackState


class WheelOdometryToTwist(Node):
    def __init__(self, **kwargs):
        super().__init__("wheel_odometry_to_twist", **kwargs)
        defaults = ConversionConfig()
        values = {}
        for item in fields(defaults):
            values[item.name] = self.declare_parameter(item.name, getattr(defaults, item.name)).value
        self.config = ConversionConfig(**values)
        self.config.validate()
        self.require_can_feedback = self.declare_parameter("require_can_feedback", False).value
        self.can_interface = self.declare_parameter("can_interface", "can10").value
        timeout = self.declare_parameter("can_feedback_timeout_sec", 0.5).value
        self.can_feedback = CanFeedbackState(timeout)
        self._can_socket = None
        self.can_timer = None
        if self.require_can_feedback:
            self._open_can_listener()
            self.can_timer = self.create_timer(0.01, self._poll_can_feedback)
        input_topic = self.declare_parameter("input_topic", "/scout/wheel_odometry").value
        output_topic = self.declare_parameter(
            "output_topic", "/sensing/vehicle_velocity_converter/twist_with_covariance",
        ).value
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.last_stamp_ns = None
        self.publisher = self.create_publisher(
            TwistWithCovarianceStamped, output_topic,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.subscription = self.create_subscription(
            Odometry, input_topic, self.on_odometry,
            QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE),
        )
        self.get_logger().info(
            f"Wheel velocity: {input_topic} [{self.config.source_child_frame}] -> "
            f"{output_topic} [{self.config.output_frame}]; publishes only validated samples",
        )

    def _open_can_listener(self):
        listener = None
        try:
            if not isinstance(self.can_interface, str) or not self.can_interface:
                raise ValueError("can_interface must be a nonempty interface name")
            listener = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
            # Match both arbitration ID and frame kind; no extended/RTR/error
            # frames can renew the freshness gate. This socket never transmits.
            listener.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER,
                                struct.pack("=II", SCOUT_V2_MOTION_ID, SCOUT_MOTION_FILTER_MASK))
            # Error frames use their own option; CAN_ERR_FLAG in a data mask selects RX_ERR.
            listener.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_ERR_FILTER, struct.pack("=I", 0))
            listener.setblocking(False)
            listener.bind((self.can_interface,))
            self._can_socket = listener
            self.get_logger().info(
                f"Read-only CAN freshness gate: {self.can_interface}, Scout V2 ID 0x221, "
                f"timeout {self.can_feedback.timeout_sec}s",
            )
        except (AttributeError, OSError, ValueError) as error:
            if listener is not None:
                listener.close()
            self.can_feedback.mark_interface_error()
            self.get_logger().error(
                f"CAN freshness listener unavailable on {self.can_interface}: {error}; "
                "wheel observations will not be published. Restart after fixing the interface.",
            )

    def _poll_can_feedback(self):
        if self._can_socket is None:
            return
        try:
            for _ in range(256):
                try:
                    frame = self._can_socket.recv(72)
                except BlockingIOError:
                    break
                if not frame:
                    raise OSError("CAN receive socket closed")
                self.can_feedback.observe(frame, time.monotonic_ns())
        except OSError as error:
            self.can_feedback.mark_interface_error()
            self._can_socket.close()
            self._can_socket = None
            self.get_logger().error(f"CAN freshness listener failed: {error}; wheel observations blocked")

    def on_odometry(self, message):
        try:
            if self.require_can_feedback and not self.can_feedback.is_fresh(time.monotonic_ns()):
                raise InvalidWheelSample("no fresh physical Scout CAN motion feedback")
            # Explicitly check the child frame before requesting its transform.
            if message.child_frame_id != self.config.source_child_frame:
                raise InvalidWheelSample("unexpected odometry child frame (twist frame)")
            transform = self.tf_buffer.lookup_transform(
                self.config.output_frame, message.child_frame_id,
                Time.from_msg(message.header.stamp),
            )
            t, q = transform.transform.translation, transform.transform.rotation
            linear, angular = message.twist.twist.linear, message.twist.twist.angular
            result = convert_wheel_sample(
                WheelSample(
                    message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec,
                    message.header.frame_id, message.child_frame_id,
                    (linear.x, linear.y, linear.z), (angular.x, angular.y, angular.z),
                    message.twist.covariance,
                ),
                MountTransform(
                    transform.header.frame_id, transform.child_frame_id,
                    (t.x, t.y, t.z), (q.x, q.y, q.z, q.w),
                ),
                self.config, self.get_clock().now().nanoseconds, self.last_stamp_ns,
            )
        except (InvalidWheelSample, TransformException, ValueError) as error:
            self.get_logger().warning(f"Rejected wheel sample: {error}", throttle_duration_sec=5.0)
            return
        output = TwistWithCovarianceStamped()
        output.header.stamp = message.header.stamp
        output.header.frame_id = result.frame_id
        output.twist.twist.linear.x, output.twist.twist.linear.y, output.twist.twist.linear.z = result.linear
        output.twist.twist.angular.x, output.twist.twist.angular.y, output.twist.twist.angular.z = result.angular
        output.twist.covariance = list(result.covariance)
        self.publisher.publish(output)
        self.last_stamp_ns = result.stamp_ns
        if result.covariance_source == "configured_invalid_covariance":
            self.get_logger().warning(
                "Invalid wheel covariance replaced by configured uncertainty",
                throttle_duration_sec=5.0,
            )

    def destroy_node(self):
        if self._can_socket is not None:
            self._can_socket.close()
            self._can_socket = None
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = WheelOdometryToTwist()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
