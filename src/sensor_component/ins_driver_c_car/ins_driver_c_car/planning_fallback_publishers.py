from __future__ import annotations

import math

from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
import rclpy


def _finite_positive(value: float, fallback: float) -> float:
    if math.isfinite(value) and value > 0.0:
        return value
    return fallback


class PlanningFallbackPublishers(Node):
    def __init__(self) -> None:
        super().__init__("planning_fallback_publishers")

        self.declare_parameter("acceleration_topic", "/localization/acceleration")
        self.declare_parameter("occupancy_grid_topic", "/perception/occupancy_grid_map/map")
        self.declare_parameter(
            "parking_occupancy_grid_topic",
            "/planning/scenario_planning/parking/costmap_generator/occupancy_grid",
        )
        self.declare_parameter("base_link_frame_id", "base_link")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("stale_timeout_acceleration_sec", 0.5)
        self.declare_parameter("stale_timeout_occupancy_sec", 1.0)
        self.declare_parameter("timer_rate_hz", 20.0)
        self.declare_parameter("occupancy_resolution", 1.0)

        self.acceleration_topic = str(self.get_parameter("acceleration_topic").value)
        self.occupancy_grid_topic = str(self.get_parameter("occupancy_grid_topic").value)
        self.parking_occupancy_grid_topic = str(
            self.get_parameter("parking_occupancy_grid_topic").value
        )
        self.base_link_frame_id = str(self.get_parameter("base_link_frame_id").value)
        self.map_frame_id = str(self.get_parameter("map_frame_id").value)
        self.stale_timeout_acceleration_sec = _finite_positive(
            float(self.get_parameter("stale_timeout_acceleration_sec").value), 0.5
        )
        self.stale_timeout_occupancy_sec = _finite_positive(
            float(self.get_parameter("stale_timeout_occupancy_sec").value), 1.0
        )
        self.timer_rate_hz = _finite_positive(float(self.get_parameter("timer_rate_hz").value), 20.0)
        self.occupancy_resolution = _finite_positive(
            float(self.get_parameter("occupancy_resolution").value), 1.0
        )

        self._last_acceleration_rx_sec: float | None = None
        self._last_occupancy_rx_sec: float | None = None
        self._last_parking_occupancy_rx_sec: float | None = None

        self.acceleration_publisher = self.create_publisher(
            AccelWithCovarianceStamped, self.acceleration_topic, 10
        )
        self.occupancy_publisher = self.create_publisher(
            OccupancyGrid, self.occupancy_grid_topic, 10
        )
        self.parking_occupancy_publisher = self.create_publisher(
            OccupancyGrid, self.parking_occupancy_grid_topic, 10
        )

        self.create_subscription(
            AccelWithCovarianceStamped, self.acceleration_topic, self._on_acceleration, 10
        )
        self.create_subscription(
            OccupancyGrid, self.occupancy_grid_topic, self._on_occupancy, 10
        )
        self.create_subscription(
            OccupancyGrid,
            self.parking_occupancy_grid_topic,
            self._on_parking_occupancy,
            10,
        )

        self._timer = self.create_timer(1.0 / self.timer_rate_hz, self._on_timer)
        self.get_logger().info(
            "Fallback publishers active for acceleration/occupancy topics. "
            "Only publishes when input is stale."
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_acceleration(self, _msg: AccelWithCovarianceStamped) -> None:
        self._last_acceleration_rx_sec = self._now_seconds()

    def _on_occupancy(self, _msg: OccupancyGrid) -> None:
        self._last_occupancy_rx_sec = self._now_seconds()

    def _on_parking_occupancy(self, _msg: OccupancyGrid) -> None:
        self._last_parking_occupancy_rx_sec = self._now_seconds()

    def _is_stale(self, last_rx_sec: float | None, timeout_sec: float, now_sec: float) -> bool:
        if last_rx_sec is None:
            return True
        return (now_sec - last_rx_sec) > timeout_sec

    def _publish_zero_acceleration(self) -> None:
        msg = AccelWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.base_link_frame_id
        msg.accel.accel.linear.x = 0.0
        msg.accel.accel.linear.y = 0.0
        msg.accel.accel.linear.z = 0.0
        msg.accel.accel.angular.x = 0.0
        msg.accel.accel.angular.y = 0.0
        msg.accel.accel.angular.z = 0.0
        self.acceleration_publisher.publish(msg)

    def _publish_empty_occupancy_grid(self, publisher) -> None:
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame_id
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = float(self.occupancy_resolution)
        msg.info.width = 1
        msg.info.height = 1
        msg.info.origin.orientation.w = 1.0
        msg.data = [0]
        publisher.publish(msg)

    def _on_timer(self) -> None:
        now_sec = self._now_seconds()

        if self._is_stale(
            self._last_acceleration_rx_sec, self.stale_timeout_acceleration_sec, now_sec
        ):
            self._publish_zero_acceleration()

        if self._is_stale(
            self._last_occupancy_rx_sec, self.stale_timeout_occupancy_sec, now_sec
        ):
            self._publish_empty_occupancy_grid(self.occupancy_publisher)

        if self._is_stale(
            self._last_parking_occupancy_rx_sec, self.stale_timeout_occupancy_sec, now_sec
        ):
            self._publish_empty_occupancy_grid(self.parking_occupancy_publisher)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningFallbackPublishers()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
