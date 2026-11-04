"""Validate and relay actual calibrated IMU data with reliable output QoS."""

import copy
from dataclasses import fields

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu

from .imu_validation import ImuValidationConfig, validate_imu_sample


class ImuRelay(Node):
    def __init__(self, **kwargs):
        super().__init__("scout_localization_imu_relay", **kwargs)
        defaults = ImuValidationConfig()
        self.config = ImuValidationConfig(**{
            item.name: self.declare_parameter(item.name, getattr(defaults, item.name)).value
            for item in fields(defaults)
        })
        self.config.validate()
        input_topic = self.declare_parameter("input_topic", "/zed2i/zed_node/imu/data").value
        output_topic = self.declare_parameter("output_topic", "/scout/localization/imu").value
        self.last_stamp_ns = None
        self.publisher = self.create_publisher(
            Imu, output_topic, QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.subscription = self.create_subscription(Imu, input_topic, self.on_imu, qos_profile_sensor_data)
        self.get_logger().info(
            f"Calibrated IMU relay: {input_topic} -> {output_topic}, reliable output; "
            "preserving source frame, timestamp and readings",
        )

    def on_imu(self, message):
        q, w, a = message.orientation, message.angular_velocity, message.linear_acceleration
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        try:
            covariance, _ = validate_imu_sample(
                stamp_ns=stamp_ns, frame_id=message.header.frame_id,
                orientation=(q.x, q.y, q.z, q.w), angular_velocity=(w.x, w.y, w.z),
                acceleration=(a.x, a.y, a.z),
                orientation_covariance=message.orientation_covariance,
                angular_velocity_covariance=message.angular_velocity_covariance,
                acceleration_covariance=message.linear_acceleration_covariance,
                now_ns=self.get_clock().now().nanoseconds, config=self.config,
                last_stamp_ns=self.last_stamp_ns,
            )
        except (ValueError, OverflowError) as error:
            self.get_logger().warning(f"Rejected IMU sample: {error}", throttle_duration_sec=5.0)
            return
        output = copy.deepcopy(message)
        output.angular_velocity_covariance = list(covariance)
        self.publisher.publish(output)
        self.last_stamp_ns = stamp_ns


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ImuRelay()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
