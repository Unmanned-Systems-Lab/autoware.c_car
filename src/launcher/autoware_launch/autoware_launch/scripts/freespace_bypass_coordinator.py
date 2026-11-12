#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
import uuid
from typing import Iterable

from autoware_internal_planning_msgs.msg import PathWithLaneId
from autoware_internal_planning_msgs.msg import Scenario
from autoware_perception_msgs.msg import PredictedObjects
from autoware_planning_msgs.msg import LaneletRoute
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from std_msgs.msg import Bool


def _distance_xy(a: Pose, b: Pose) -> float:
    dx = a.position.x - b.position.x
    dy = a.position.y - b.position.y
    return math.hypot(dx, dy)


def _distance_xy_points(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _path_pose(point) -> Pose:
    return point.point.pose


def _trajectory_uuid_msg() -> list[int]:
    return list(uuid.uuid4().bytes)


class FreespaceBypassCoordinator(Node):
    def __init__(self) -> None:
        super().__init__("freespace_bypass_coordinator")

        self.declare_parameter("mission_route_topic", "/planning/mission_planning/route")
        self.declare_parameter("parking_route_topic", "/planning/freespace_bypass/route")
        self.declare_parameter("force_parking_topic", "/planning/freespace_bypass/force_parking")
        self.declare_parameter(
            "path_topic",
            "/planning/scenario_planning/lane_driving/behavior_planning/path_with_lane_id",
        )
        self.declare_parameter("objects_topic", "/perception/object_recognition/objects")
        self.declare_parameter("odometry_topic", "/localization/kinematic_state")
        self.declare_parameter(
            "parking_completed_topic", "/planning/scenario_planning/parking/is_completed"
        )
        self.declare_parameter("scenario_topic", "/planning/scenario_planning/scenario")
        self.declare_parameter("trigger_persistence_sec", 5.0)
        self.declare_parameter("stopped_velocity_mps", 0.05)
        self.declare_parameter("min_forward_object_distance_m", 1.0)
        self.declare_parameter("max_forward_object_distance_m", 20.0)
        self.declare_parameter("object_path_lateral_margin_m", 2.5)
        self.declare_parameter("obstacle_length_padding_m", 1.0)
        self.declare_parameter("rejoin_margin_after_obstacle_m", 6.0)
        self.declare_parameter("min_remaining_path_after_rejoin_m", 5.0)
        self.declare_parameter("goal_clearance_radius_m", 3.0)
        self.declare_parameter("fallback_goal_clearance_radius_m", 0.5)
        self.declare_parameter("max_object_projection_length_m", 2.0)
        self.declare_parameter("max_object_clearance_radius_m", 1.0)
        self.declare_parameter("max_rejoin_extrapolation_m", 8.0)
        self.declare_parameter("rejoin_arrived_distance_m", 1.0)
        self.declare_parameter("timer_rate_hz", 5.0)

        self.mission_route_topic = str(self.get_parameter("mission_route_topic").value)
        self.parking_route_topic = str(self.get_parameter("parking_route_topic").value)
        self.force_parking_topic = str(self.get_parameter("force_parking_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.objects_topic = str(self.get_parameter("objects_topic").value)
        self.odometry_topic = str(self.get_parameter("odometry_topic").value)
        self.parking_completed_topic = str(self.get_parameter("parking_completed_topic").value)
        self.scenario_topic = str(self.get_parameter("scenario_topic").value)
        self.trigger_persistence_sec = float(self.get_parameter("trigger_persistence_sec").value)
        self.stopped_velocity_mps = float(self.get_parameter("stopped_velocity_mps").value)
        self.min_forward_object_distance_m = float(
            self.get_parameter("min_forward_object_distance_m").value
        )
        self.max_forward_object_distance_m = float(
            self.get_parameter("max_forward_object_distance_m").value
        )
        self.object_path_lateral_margin_m = float(
            self.get_parameter("object_path_lateral_margin_m").value
        )
        self.obstacle_length_padding_m = float(
            self.get_parameter("obstacle_length_padding_m").value
        )
        self.rejoin_margin_after_obstacle_m = float(
            self.get_parameter("rejoin_margin_after_obstacle_m").value
        )
        self.min_remaining_path_after_rejoin_m = float(
            self.get_parameter("min_remaining_path_after_rejoin_m").value
        )
        self.goal_clearance_radius_m = float(
            self.get_parameter("goal_clearance_radius_m").value
        )
        self.fallback_goal_clearance_radius_m = float(
            self.get_parameter("fallback_goal_clearance_radius_m").value
        )
        self.max_object_projection_length_m = float(
            self.get_parameter("max_object_projection_length_m").value
        )
        self.max_object_clearance_radius_m = float(
            self.get_parameter("max_object_clearance_radius_m").value
        )
        self.max_rejoin_extrapolation_m = float(
            self.get_parameter("max_rejoin_extrapolation_m").value
        )
        self.rejoin_arrived_distance_m = float(
            self.get_parameter("rejoin_arrived_distance_m").value
        )
        self.timer_rate_hz = float(self.get_parameter("timer_rate_hz").value)

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.route_pub = self.create_publisher(LaneletRoute, self.parking_route_topic, transient_qos)
        self.force_parking_pub = self.create_publisher(Bool, self.force_parking_topic, transient_qos)

        self.create_subscription(LaneletRoute, self.mission_route_topic, self._on_route, transient_qos)
        self.create_subscription(PathWithLaneId, self.path_topic, self._on_path, 10)
        self.create_subscription(PredictedObjects, self.objects_topic, self._on_objects, 10)
        self.create_subscription(Odometry, self.odometry_topic, self._on_odometry, 20)
        self.create_subscription(Bool, self.parking_completed_topic, self._on_parking_completed, 10)
        self.create_subscription(Scenario, self.scenario_topic, self._on_scenario, 10)

        self._mission_route: LaneletRoute | None = None
        self._path: PathWithLaneId | None = None
        self._objects: PredictedObjects | None = None
        self._odometry: Odometry | None = None
        self._parking_completed = False
        self._current_scenario = ""

        self._trigger_since = None
        self._active = False
        self._temp_route: LaneletRoute | None = None
        self._rejoin_goal_pose: Pose | None = None
        self._last_pass_through_uuid: tuple[int, ...] | None = None

        self._publish_force(False)
        self.create_timer(1.0 / max(self.timer_rate_hz, 1.0), self._on_timer)

        self.get_logger().info(
            "Freespace bypass coordinator enabled: obstacle stop -> 5s persist -> local freespace"
        )

    def _on_route(self, msg: LaneletRoute) -> None:
        self._mission_route = msg
        if not self._active:
            self._publish_pass_through_route()

    def _on_path(self, msg: PathWithLaneId) -> None:
        self._path = msg

    def _on_objects(self, msg: PredictedObjects) -> None:
        self._objects = msg

    def _on_odometry(self, msg: Odometry) -> None:
        self._odometry = msg

    def _on_parking_completed(self, msg: Bool) -> None:
        self._parking_completed = bool(msg.data)

    def _on_scenario(self, msg: Scenario) -> None:
        self._current_scenario = msg.current_scenario

    def _publish_force(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        self.force_parking_pub.publish(msg)

    def _publish_pass_through_route(self) -> None:
        if self._mission_route is None:
            return
        route_uuid = tuple(self._mission_route.uuid.uuid)
        if route_uuid == self._last_pass_through_uuid:
            return
        self.route_pub.publish(self._mission_route)
        self._last_pass_through_uuid = route_uuid

    def _ego_pose(self) -> Pose | None:
        if self._odometry is None:
            return None
        return self._odometry.pose.pose

    def _ego_speed(self) -> float:
        if self._odometry is None:
            return float("inf")
        twist = self._odometry.twist.twist.linear
        return math.hypot(twist.x, twist.y)

    def _deduplicate_objects(self, objects: Iterable) -> list:
        unique = []
        seen = set()
        for obj in objects:
            pose = obj.kinematics.initial_pose_with_covariance.pose.position
            dims = obj.shape.dimensions
            key = (
                round(pose.x, 1),
                round(pose.y, 1),
                round(max(dims.x, dims.y), 1),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(obj)
        return unique

    def _build_path_index(self):
        if self._path is None or not self._path.points:
            return None

        poses = [_path_pose(point) for point in self._path.points]
        cumulative = [0.0]
        for prev_pose, pose in zip(poses, poses[1:]):
            cumulative.append(cumulative[-1] + _distance_xy(prev_pose, pose))
        return poses, cumulative

    def _nearest_path_index(self, poses: list[Pose], x: float, y: float) -> tuple[int, float]:
        best_idx = 0
        best_dist = float("inf")
        for idx, pose in enumerate(poses):
            dist = _distance_xy_points(x, y, pose.position.x, pose.position.y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx, best_dist

    def _object_projection_length(self, obj) -> float:
        dims = obj.shape.dimensions
        return min(max(float(dims.x), float(dims.y)), self.max_object_projection_length_m)

    def _object_clearance_radius(self, obj) -> float:
        dims = obj.shape.dimensions
        diagonal_radius = 0.5 * math.hypot(float(dims.x), float(dims.y))
        return min(max(diagonal_radius, 0.3), self.max_object_clearance_radius_m)

    def _select_obstacle_cluster_end_s(self):
        if self._objects is None:
            return None

        path_index = self._build_path_index()
        ego_pose = self._ego_pose()
        if path_index is None or ego_pose is None:
            return None

        poses, cumulative = path_index
        ego_idx, _ = self._nearest_path_index(poses, ego_pose.position.x, ego_pose.position.y)
        ego_s = cumulative[ego_idx]

        cluster_end_s = None
        for obj in self._deduplicate_objects(self._objects.objects):
            obj_pose = obj.kinematics.initial_pose_with_covariance.pose.position
            idx, dist = self._nearest_path_index(poses, obj_pose.x, obj_pose.y)
            obj_s = cumulative[idx]
            forward = obj_s - ego_s
            if forward < self.min_forward_object_distance_m:
                continue
            if forward > self.max_forward_object_distance_m:
                continue
            if dist > self.object_path_lateral_margin_m:
                continue

            object_length = self._object_projection_length(obj)
            obj_end_s = obj_s + object_length * 0.5 + self.obstacle_length_padding_m
            cluster_end_s = obj_end_s if cluster_end_s is None else max(cluster_end_s, obj_end_s)

        return cluster_end_s

    def _candidate_is_clear(self, pose: Pose, extra_clearance_radius_m: float) -> bool:
        if self._objects is None:
            return False
        for obj in self._deduplicate_objects(self._objects.objects):
            obj_pose = obj.kinematics.initial_pose_with_covariance.pose.position
            object_radius = self._object_clearance_radius(obj)
            clearance = extra_clearance_radius_m + object_radius
            if _distance_xy_points(
                pose.position.x, pose.position.y, obj_pose.x, obj_pose.y
            ) < clearance:
                return False
        return True

    def _search_candidate_pose(
        self,
        poses: list[Pose],
        cumulative: list[float],
        required_s: float,
        total_length: float,
        min_remaining_path_after_rejoin_m: float,
        extra_clearance_radius_m: float,
    ) -> Pose | None:
        for pose, s in zip(poses, cumulative):
            if s < required_s:
                continue
            if total_length - s < min_remaining_path_after_rejoin_m:
                continue
            if not self._candidate_is_clear(pose, extra_clearance_radius_m):
                continue
            return copy.deepcopy(pose)
        return None

    def _extrapolate_pose(self, poses: list[Pose], distance_m: float) -> Pose | None:
        if len(poses) < 2:
            return None
        last_pose = poses[-1]
        prev_pose = poses[-2]
        dx = last_pose.position.x - prev_pose.position.x
        dy = last_pose.position.y - prev_pose.position.y
        norm = math.hypot(dx, dy)
        if norm < 1e-3:
            return None

        pose = copy.deepcopy(last_pose)
        pose.position.x += distance_m * dx / norm
        pose.position.y += distance_m * dy / norm
        return pose

    def _build_rejoin_route(self) -> LaneletRoute | None:
        if self._mission_route is None:
            return None

        path_index = self._build_path_index()
        ego_pose = self._ego_pose()
        if path_index is None:
            return None

        poses, cumulative = path_index
        cluster_end_s = self._select_obstacle_cluster_end_s()
        if cluster_end_s is None:
            return None

        total_length = cumulative[-1]
        required_s = cluster_end_s + self.rejoin_margin_after_obstacle_m

        candidate_pose = self._search_candidate_pose(
            poses,
            cumulative,
            required_s,
            total_length,
            self.min_remaining_path_after_rejoin_m,
            self.goal_clearance_radius_m,
        )

        if candidate_pose is None:
            candidate_pose = self._search_candidate_pose(
                poses,
                cumulative,
                required_s,
                total_length,
                0.0,
                self.fallback_goal_clearance_radius_m,
            )

        if candidate_pose is None and required_s > total_length:
            extrapolation_distance_m = min(
                required_s - total_length, self.max_rejoin_extrapolation_m
            )
            candidate_pose = self._extrapolate_pose(poses, extrapolation_distance_m)
            if candidate_pose is not None and not self._candidate_is_clear(
                candidate_pose, self.fallback_goal_clearance_radius_m
            ):
                candidate_pose = None

        if candidate_pose is None:
            return None

        route = LaneletRoute()
        route.header = copy.deepcopy(self._mission_route.header)
        route.start_pose = copy.deepcopy(ego_pose if ego_pose is not None else self._mission_route.start_pose)
        route.goal_pose = candidate_pose
        route.segments = copy.deepcopy(self._mission_route.segments)
        route.uuid.uuid = _trajectory_uuid_msg()
        route.allow_modification = False
        self._rejoin_goal_pose = copy.deepcopy(candidate_pose)
        return route

    def _rejoin_goal_reached(self) -> bool:
        ego_pose = self._ego_pose()
        if ego_pose is None or self._rejoin_goal_pose is None:
            return False
        return _distance_xy(ego_pose, self._rejoin_goal_pose) <= self.rejoin_arrived_distance_m

    def _activate_freespace(self) -> None:
        route = self._build_rejoin_route()
        if route is None:
            self.get_logger().warn(
                "Unable to find a stable rejoin pose for freespace bypass; keep waiting."
            )
            return

        self._temp_route = route
        self._active = True
        self._parking_completed = False
        self.route_pub.publish(route)
        self._publish_force(True)
        goal = route.goal_pose.position
        self.get_logger().warn(
            f"Switching to temporary freespace bypass. Rejoin goal=({goal.x:.2f}, {goal.y:.2f})"
        )

    def _deactivate_freespace(self) -> None:
        self._active = False
        self._temp_route = None
        self._rejoin_goal_pose = None
        self._trigger_since = None
        self._parking_completed = False
        self._publish_force(False)
        self._last_pass_through_uuid = None
        self._publish_pass_through_route()
        self.get_logger().info("Freespace bypass completed. Returning to lane-driving trajectory.")

    def _on_timer(self) -> None:
        if self._mission_route is None:
            return

        if self._active:
            if self._temp_route is not None:
                self.route_pub.publish(self._temp_route)
            self._publish_force(True)
            if self._parking_completed and self._rejoin_goal_reached():
                self._deactivate_freespace()
                return
            if (
                self._current_scenario == Scenario.LANEDRIVING
                and self._rejoin_goal_reached()
            ):
                self._deactivate_freespace()
            return

        self._publish_pass_through_route()
        self._publish_force(False)

        if self._path is None or self._objects is None or self._odometry is None:
            return

        if self._current_scenario == Scenario.PARKING:
            self._trigger_since = None
            return

        if self._ego_speed() > self.stopped_velocity_mps:
            self._trigger_since = None
            return

        cluster_end_s = self._select_obstacle_cluster_end_s()
        if cluster_end_s is None:
            self._trigger_since = None
            return

        now = self.get_clock().now()
        if self._trigger_since is None:
            self._trigger_since = now
            return

        if now - self._trigger_since < Duration(seconds=self.trigger_persistence_sec):
            return

        self._activate_freespace()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FreespaceBypassCoordinator()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
