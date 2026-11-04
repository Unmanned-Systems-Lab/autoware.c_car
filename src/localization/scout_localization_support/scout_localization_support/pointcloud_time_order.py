"""Private native-layout time validation and ordering for Universe deskew."""

from array import array

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from .cloud_time_order import CloudValidationError, order_native_cloud


class PointcloudTimeOrder(Node):
    def __init__(self):
        super().__init__("pointcloud_time_order")
        source = self.declare_parameter("input_topic", "/rslidar_points").value
        target = self.declare_parameter(
            "output_topic", "/scout/localization/points_ordered").value
        self.n_scan = self.declare_parameter("n_scan", 32).value
        self.max_scan_duration = self.declare_parameter("max_scan_duration", 0.2).value
        self.remove_non_finite = self.declare_parameter("remove_non_finite", True).value
        self.publisher = self.create_publisher(
            PointCloud2, target,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self.subscription = self.create_subscription(
            PointCloud2, source, self.on_cloud, qos_profile_sensor_data)
        self.accepted = 0
        self.rejected = 0
        self.last_stamp_ns = None
        self.get_logger().info(
            f"Native scan ordering: {source} -> {target}; "
            f"{self.n_scan} channels, max {self.max_scan_duration}s")

    def on_cloud(self, message):
        try:
            ordered = order_native_cloud(
                message, n_scan=self.n_scan, max_scan_duration=self.max_scan_duration,
                remove_non_finite=self.remove_non_finite)
            if self.last_stamp_ns is not None and ordered.stamp_ns <= self.last_stamp_ns:
                raise CloudValidationError("cloud header timestamps are not increasing")
        except (CloudValidationError, AttributeError, TypeError, ValueError) as error:
            self.rejected += 1
            self.get_logger().warning(
                f"Rejected native scan ({self.rejected}): {error}",
                throttle_duration_sec=5.0)
            return

        output = PointCloud2()
        output.header = message.header
        output.fields = list(message.fields)
        output.height = ordered.height
        output.width = ordered.width
        output.point_step = ordered.point_step
        output.row_step = ordered.row_step
        output.is_bigendian = False
        output.is_dense = ordered.is_dense
        # ROS accepts the byte array directly; avoid Python per-byte validation.
        output.data = array("B", ordered.data)
        self.publisher.publish(output)
        self.last_stamp_ns = ordered.stamp_ns
        self.accepted += 1
        self.get_logger().info(
            f"accepted={self.accepted} rejected={self.rejected}; "
            f"points={ordered.width} dropped={ordered.dropped_points}, time=[{ordered.min_time_ns}, "
            f"{ordered.max_time_ns}]ns, backward_pairs={ordered.backwards_time_pairs}",
            throttle_duration_sec=10.0)


def main(args=None):
    rclpy.init(args=args)
    node = PointcloudTimeOrder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
