#!/usr/bin/env python3

import argparse
import errno
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from typing import Iterable

import requests
import yaml
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from autoware_adapi_v1_msgs.msg import OperationModeState
from autoware_adapi_v1_msgs.msg import RouteState
from autoware_adapi_v1_msgs.srv import ChangeOperationMode
from autoware_adapi_v1_msgs.srv import ClearRoute
from autoware_adapi_v1_msgs.srv import SetRoutePoints
from autoware_planning_msgs.msg import LaneletRoute
from autoware_planning_msgs.msg import Trajectory
from autoware_localization_msgs.srv import InitializeLocalization
from autoware_map_msgs.msg import LaneletMapBin
from fixposition_driver_msgs.msg import FpaOdomstatus
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from PIL import Image
from pyproj import CRS
from pyproj import Transformer
import rclpy
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.srv import SetParametersAtomically
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


SCRIPT_DIR = Path(__file__).resolve().parent
AUTOWARE_ROOT = SCRIPT_DIR.parents[3]
AUTOWARE_SETUP_BASH = AUTOWARE_ROOT / "install/setup.bash"
SCOUT_VEHICLE_INFO_PATH = (
    AUTOWARE_ROOT
    / "src/vehicle/scout_vehicle_description/config/vehicle_info.param.yaml"
)
INS_DRIVER_SOURCE_CONFIG_PATH = (
    AUTOWARE_ROOT / "src/sensor_component/ins_driver_ez21/config/driver.yaml"
)
INS_DRIVER_BUILD_CONFIG_PATH = AUTOWARE_ROOT / "build/ins_driver_ez21/config/driver.yaml"
INS_DRIVER_INSTALL_CONFIG_PATH = (
    AUTOWARE_ROOT / "install/ins_driver_ez21/share/ins_driver_ez21/config/driver.yaml"
)
DEFAULT_MAP_NAME = "autoware_init_map"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "generated_maps"
AUTOWARE_MAP_RELOAD_TOPIC = "/map/map_projector_info"
AUTOWARE_MAP_PROJECTION_LOADER_NODE = "/map/map_projection_loader"
AUTOWARE_MAP_RELOAD_TIMEOUT_S = 20.0
AUTOWARE_MAP_COMPONENT_RELOAD_TIMEOUT_S = 30.0
AUTOWARE_MAP_CONTAINER = "/map/map_container"
AUTOWARE_LANELET_MAP_LOADER_NODE = "/map/lanelet2_map_loader"
AUTOWARE_POINTCLOUD_MAP_TOPIC = "/map/pointcloud_map"
AUTOWARE_POINTCLOUD_MAP_METADATA_TOPIC = "/map/pointcloud_map_metadata"
AUTOWARE_POINTCLOUD_MAP_LOADER_NODE = "/map/pointcloud_map_loader"
AUTOWARE_POINTCLOUD_MAP_LOADER_PACKAGE = "autoware_map_loader"
AUTOWARE_POINTCLOUD_MAP_LOADER_PLUGIN = "autoware::map_loader::PointCloudMapLoaderNode"
INS_DRIVER_NODE_NAME = "/ins_driver_ez21"
INS_ORIGIN_UPDATE_TIMEOUT_S = 5.0
INS_ORIGIN_ACTIVE_PARAMETER = "reference_origin_active"
INS_ORIGIN_LATITUDE_PARAMETER = "reference_origin_latitude_deg"
INS_ORIGIN_LONGITUDE_PARAMETER = "reference_origin_longitude_deg"
INS_ORIGIN_ALTITUDE_PARAMETER = "reference_origin_altitude_m"
RVIZ_SATELLITE_MARKER_TOPIC = "/map/vector_map_marker"
RVIZ_SATELLITE_MARKER_NAMESPACE = "ui_satellite_background"
GOOGLE_STATIC_MAP_URL = "https://maps.googleapis.com/maps/api/staticmap"
DEFAULT_SATELLITE_ZOOM = 19
DEFAULT_SATELLITE_IMAGE_WIDTH_PX = 640
DEFAULT_SATELLITE_IMAGE_HEIGHT_PX = 640
MAX_SATELLITE_IMAGE_DIMENSION_PX = 640
DEFAULT_SATELLITE_SAMPLE_COLUMNS = 64
DEFAULT_SATELLITE_SAMPLE_ROWS = 64
DEFAULT_SATELLITE_MARKER_ALPHA = 0.85
DEFAULT_SATELLITE_MARKER_Z_M = -0.05
DEFAULT_LANE_EXTRA_MARGIN_M = 1.0
GENERATED_LANE_START_BACKWARD_MARGIN_M = 5.0
DEFAULT_ELEVATION_M = 0.0
DEFAULT_SPEED_LIMIT_KPH = 3.6
SCOUT_MAX_YAW_RATE_RADPS = 0.8
DEFAULT_TURN_ARC_STEP_M = 0.2
DEFAULT_PCD_LONGITUDINAL_STEP_M = 1.0
DEFAULT_PCD_LATERAL_STEP_M = 0.5
AUTOWARE_ROUTE_SET_SERVICE = "/api/routing/set_route_points"
AUTOWARE_ROUTE_CLEAR_SERVICE = "/api/routing/clear_route"
AUTOWARE_ROUTE_STATE_TOPIC = "/api/routing/state"
AUTOWARE_MISSION_PLANNING_ROUTE_TOPIC = "/planning/mission_planning/route"
AUTOWARE_PLANNING_ROUTE_TOPIC = "/planning/route"
AUTOWARE_MISSION_PLANNING_ROUTE_MARKER_TOPIC = "/planning/mission_planning/route_marker"
AUTOWARE_SCENARIO_PLANNING_TRAJECTORY_TOPIC = "/planning/scenario_planning/trajectory"
AUTOWARE_VELOCITY_SMOOTHER_TRAJECTORY_TOPIC = "/planning/scenario_planning/velocity_smoother/trajectory"
AUTOWARE_PLANNING_TRAJECTORY_TOPIC = "/planning/trajectory"
AUTOWARE_OPERATION_MODE_STATE_TOPIC = "/api/operation_mode/state"
AUTOWARE_CHANGE_TO_AUTONOMOUS_SERVICE = "/api/operation_mode/change_to_autonomous"
AUTOWARE_CHANGE_TO_STOP_SERVICE = "/api/operation_mode/change_to_stop"
AUTOWARE_ENABLE_CONTROL_SERVICE = "/api/operation_mode/enable_autoware_control"
AUTOWARE_DISABLE_CONTROL_SERVICE = "/api/operation_mode/disable_autoware_control"
AUTOWARE_LOCALIZATION_INITIALIZE_SERVICE = "/localization/initialize"
AUTOWARE_LOCALIZATION_KINEMATIC_STATE_TOPIC = "/localization/kinematic_state"
FIXPOSITION_LLH_TOPIC = "/fixposition/odometry_llh"
FIXPOSITION_STATUS_TOPIC = "/fixposition/fpa/odomstatus"
FIXPOSITION_STATIC_TF_TOPIC = "/fixposition/tf_static"
FIXPOSITION_DATA_TIMEOUT_S = 2.0
SCOUT_LOCALIZATION_MODE = os.environ.get("SCOUT_LOCALIZATION_MODE", "fixposition").strip().lower()
if SCOUT_LOCALIZATION_MODE not in {"fixposition", "universe"}:
    raise RuntimeError("SCOUT_LOCALIZATION_MODE must be fixposition or universe")
LOCALIZATION_INITIALIZATION_STATE_TOPIC = "/localization/initialization_state"
UNIVERSE_NDT_POSE_TOPIC = "/localization/pose_estimator/pose_with_covariance"
UNIVERSE_LOCALIZATION_TIMEOUT_S = 2.0
UNIVERSE_LOCALIZATION_FUTURE_TOLERANCE_S = 0.1
UNIVERSE_MIN_MAP_POINTS = 1000
SCOUT_UI_ALLOW_VEHICLE_ACTIONS = os.environ.get("SCOUT_UI_ALLOW_VEHICLE_ACTIONS", "0") == "1"
MAP_SWITCH_LOCK = threading.Lock()
MAP_SWITCH_FAILED = False
AUTOWARE_ROUTE_SERVICE_TIMEOUT_S = 10.0
AUTOWARE_OPERATION_MODE_TIMEOUT_S = 5.0
AUTOWARE_LOCALIZATION_TIMEOUT_S = 5.0
AUTOWARE_RESET_WAIT_TIMEOUT_S = 30.0
ORIGIN_SYNC_DISCOVERY_TIMEOUT_S = 0.2

OPERATION_MODE_LABELS = {
    OperationModeState.UNKNOWN: "UNKNOWN",
    OperationModeState.STOP: "STOP",
    OperationModeState.AUTONOMOUS: "AUTONOMOUS",
    OperationModeState.LOCAL: "LOCAL",
    OperationModeState.REMOTE: "REMOTE",
}

ROUTE_STATE_LABELS = {
    RouteState.UNKNOWN: "UNKNOWN",
    RouteState.UNSET: "UNSET",
    RouteState.SET: "SET",
    RouteState.ARRIVED: "ARRIVED",
    RouteState.CHANGING: "CHANGING",
}


class MapGenerationError(Exception):
    pass


SERVER_SHUTTING_DOWN = threading.Event()


def is_invalid_ros_context_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "context is not valid" in message
        or "context is invalid" in message
        or "rcl_shutdown()" in message
        or "rcl_init() was not called" in message
    )


def is_server_shutdown_error(exc: Exception) -> bool:
    return SERVER_SHUTTING_DOWN.is_set() or is_invalid_ros_context_error(exc)


@dataclass(frozen=True)
class GeoPoint:
    latitude: float
    longitude: float
    altitude: float


def geo_point_from_ecef(ecef: tuple[float, float, float]) -> GeoPoint:
    if not all(math.isfinite(value) for value in ecef):
        raise MapGenerationError("FP_ENU0 ECEF origin contains non-finite coordinates")
    if not 6_000_000.0 <= math.sqrt(sum(value * value for value in ecef)) <= 7_000_000.0:
        raise MapGenerationError("FP_ENU0 ECEF origin is outside the expected Earth radius")
    longitude, latitude, altitude = Transformer.from_crs(
        "EPSG:4978", "EPSG:4979", always_xy=True
    ).transform(*ecef)
    return GeoPoint(latitude=latitude, longitude=longitude, altitude=altitude)


@dataclass(frozen=True)
class LocalPoint:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class RuntimeMapContext:
    output_dir: Path
    projector_type: str
    origin: GeoPoint
    elevation_m: float
    origin_source: str


@dataclass(frozen=True)
class GeoLocalAnchor:
    geo: GeoPoint
    local: LocalPoint
    source: str


class RvizMarkerPublisher:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node: Node | None = None
        self._marker_publisher = None
        self._initialized = False

    def _ensure_ready(self) -> None:
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if not rclpy.ok():
                rclpy.init(args=None)

            self._node = Node("ui_rviz_satellite_overlay_publisher")
            qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._marker_publisher = self._node.create_publisher(
                MarkerArray, RVIZ_SATELLITE_MARKER_TOPIC, qos
            )
            self._initialized = True

    def publish(self, marker_array: MarkerArray) -> None:
        self._ensure_ready()
        assert self._node is not None
        assert self._marker_publisher is not None

        with self._lock:
            for _ in range(3):
                rclpy.spin_once(self._node, timeout_sec=0.05)
                time.sleep(0.05)
            self._marker_publisher.publish(marker_array)
            for _ in range(3):
                rclpy.spin_once(self._node, timeout_sec=0.05)
                time.sleep(0.05)


RVIZ_MARKER_PUBLISHER = RvizMarkerPublisher()


class PlanningResetPublisher:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node: Node | None = None
        self._mission_route_publisher = None
        self._planning_route_publisher = None
        self._route_marker_publisher = None
        self._scenario_trajectory_publisher = None
        self._velocity_smoother_trajectory_publisher = None
        self._planning_trajectory_publisher = None
        self._initialized = False

    def _ensure_ready(self) -> None:
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if not rclpy.ok():
                rclpy.init(args=None)

            self._node = Node("ui_autoware_reset_publisher")
            transient_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            volatile_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.VOLATILE,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._mission_route_publisher = self._node.create_publisher(
                LaneletRoute, AUTOWARE_MISSION_PLANNING_ROUTE_TOPIC, transient_qos
            )
            self._planning_route_publisher = self._node.create_publisher(
                LaneletRoute, AUTOWARE_PLANNING_ROUTE_TOPIC, transient_qos
            )
            self._route_marker_publisher = self._node.create_publisher(
                MarkerArray, AUTOWARE_MISSION_PLANNING_ROUTE_MARKER_TOPIC, transient_qos
            )
            self._scenario_trajectory_publisher = self._node.create_publisher(
                Trajectory, AUTOWARE_SCENARIO_PLANNING_TRAJECTORY_TOPIC, volatile_qos
            )
            self._velocity_smoother_trajectory_publisher = self._node.create_publisher(
                Trajectory, AUTOWARE_VELOCITY_SMOOTHER_TRAJECTORY_TOPIC, volatile_qos
            )
            self._planning_trajectory_publisher = self._node.create_publisher(
                Trajectory, AUTOWARE_PLANNING_TRAJECTORY_TOPIC, volatile_qos
            )
            self._initialized = True

    def publish_reset(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self._mission_route_publisher is not None
        assert self._planning_route_publisher is not None
        assert self._route_marker_publisher is not None
        assert self._scenario_trajectory_publisher is not None
        assert self._velocity_smoother_trajectory_publisher is not None
        assert self._planning_trajectory_publisher is not None

        empty_route = LaneletRoute()
        empty_route.header.frame_id = "map"

        empty_trajectory = Trajectory()
        empty_trajectory.header.frame_id = "map"

        delete_all_marker = Marker()
        delete_all_marker.header.frame_id = "map"
        delete_all_marker.action = Marker.DELETEALL

        delete_all_markers = MarkerArray()
        delete_all_markers.markers.append(delete_all_marker)

        with self._lock:
            for _ in range(5):
                self._mission_route_publisher.publish(empty_route)
                self._planning_route_publisher.publish(empty_route)
                self._route_marker_publisher.publish(delete_all_markers)
                self._scenario_trajectory_publisher.publish(empty_trajectory)
                self._velocity_smoother_trajectory_publisher.publish(empty_trajectory)
                self._planning_trajectory_publisher.publish(empty_trajectory)
                time.sleep(0.1)

        return {
            "status": "ok",
            "route_topics": [
                AUTOWARE_MISSION_PLANNING_ROUTE_TOPIC,
                AUTOWARE_PLANNING_ROUTE_TOPIC,
            ],
            "marker_topics": [AUTOWARE_MISSION_PLANNING_ROUTE_MARKER_TOPIC],
            "trajectory_topics": [
                AUTOWARE_SCENARIO_PLANNING_TRAJECTORY_TOPIC,
                AUTOWARE_VELOCITY_SMOOTHER_TRAJECTORY_TOPIC,
                AUTOWARE_PLANNING_TRAJECTORY_TOPIC,
            ],
        }


PLANNING_RESET_PUBLISHER = PlanningResetPublisher()


class InsOriginUpdater:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node: Node | None = None
        self._client = None
        self._initialized = False

    def _ensure_ready(self) -> None:
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            if not rclpy.ok():
                rclpy.init(args=None)

            self._node = Node("ui_ins_origin_updater")
            self._client = self._node.create_client(
                SetParametersAtomically, f"{INS_DRIVER_NODE_NAME}/set_parameters_atomically"
            )
            self._initialized = True

    def is_available(self, timeout_s: float = 0.0) -> bool:
        self._ensure_ready()
        assert self._client is not None

        with self._lock:
            if timeout_s > 0.0:
                return bool(self._client.wait_for_service(timeout_sec=timeout_s))
            return bool(self._client.service_is_ready())

    def update(self, point: GeoPoint) -> dict[str, Any]:
        self._ensure_ready()
        assert self._node is not None
        assert self._client is not None

        with self._lock:
            if not self._client.wait_for_service(timeout_sec=INS_ORIGIN_UPDATE_TIMEOUT_S):
                return {
                    "status": "error",
                    "error": f"INS driver parameter service is not available on {INS_DRIVER_NODE_NAME}/set_parameters_atomically",
                    "node": INS_DRIVER_NODE_NAME,
                }

            parameters = [
                Parameter(INS_ORIGIN_LATITUDE_PARAMETER, value=float(point.latitude)),
                Parameter(INS_ORIGIN_LONGITUDE_PARAMETER, value=float(point.longitude)),
                Parameter(INS_ORIGIN_ALTITUDE_PARAMETER, value=float(point.altitude)),
                Parameter(INS_ORIGIN_ACTIVE_PARAMETER, value=True),
            ]
            request = SetParametersAtomically.Request()
            request.parameters = [parameter.to_parameter_msg() for parameter in parameters]
            future = self._client.call_async(request)
            deadline = time.monotonic() + INS_ORIGIN_UPDATE_TIMEOUT_S

            while not future.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return {
                        "status": "error",
                        "error": "timed out while updating the INS driver reference origin",
                        "node": INS_DRIVER_NODE_NAME,
                    }
                rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

            response = future.result()
            if response is None:
                return {
                    "status": "error",
                    "error": "failed to receive a response from the INS driver parameter service",
                    "node": INS_DRIVER_NODE_NAME,
                }

            if not response.result.successful:
                return {
                    "status": "error",
                    "error": response.result.reason or "unknown parameter update failure",
                    "node": INS_DRIVER_NODE_NAME,
                }

            return {
                "status": "ok",
                "message": "INS driver reference origin was updated",
                "node": INS_DRIVER_NODE_NAME,
                "origin": {
                    "latitude": float(point.latitude),
                    "longitude": float(point.longitude),
                    "altitude": float(point.altitude),
                },
                "parameters": [
                    INS_ORIGIN_LATITUDE_PARAMETER,
                    INS_ORIGIN_LONGITUDE_PARAMETER,
                    INS_ORIGIN_ALTITUDE_PARAMETER,
                    INS_ORIGIN_ACTIVE_PARAMETER,
                ],
            }


INS_ORIGIN_UPDATER = InsOriginUpdater()


class InsOriginResolver:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._node: Node | None = None
        self._client = None
        self._initialized = False

    def _ensure_ready(self) -> None:
        with self._lock:
            if self._initialized:
                return

            if not rclpy.ok():
                rclpy.init(args=None)

            self._node = Node("ui_ins_origin_resolver")
            self._client = self._node.create_client(
                GetParameters, f"{INS_DRIVER_NODE_NAME}/get_parameters"
            )
            self._initialized = True

    def current_origin(self, timeout_s: float = 0.0) -> GeoPoint | None:
        self._ensure_ready()
        assert self._node is not None
        assert self._client is not None

        with self._lock:
            if timeout_s > 0.0:
                if not self._client.wait_for_service(timeout_sec=timeout_s):
                    return None
            elif not self._client.service_is_ready():
                return None

            request = GetParameters.Request()
            request.names = [
                INS_ORIGIN_ACTIVE_PARAMETER,
                INS_ORIGIN_LATITUDE_PARAMETER,
                INS_ORIGIN_LONGITUDE_PARAMETER,
                INS_ORIGIN_ALTITUDE_PARAMETER,
            ]
            future = self._client.call_async(request)
            deadline = time.monotonic() + max(timeout_s, ORIGIN_SYNC_DISCOVERY_TIMEOUT_S)

            while not future.done():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

            response = future.result()
            if response is None or len(response.values) != len(request.names):
                return None

            active = parameter_value_to_python(response.values[0])
            latitude = finite_float_or_none(parameter_value_to_python(response.values[1]))
            longitude = finite_float_or_none(parameter_value_to_python(response.values[2]))
            altitude = finite_float_or_none(parameter_value_to_python(response.values[3]))

            if not active or latitude is None or longitude is None:
                return None

            return GeoPoint(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude if altitude is not None else DEFAULT_ELEVATION_M,
            )


INS_ORIGIN_RESOLVER = InsOriginResolver()


def response_status_to_dict(status: Any) -> dict[str, Any]:
    return {
        "success": bool(status.success),
        "code": int(status.code),
        "message": str(status.message),
    }


def build_service_result(service_name: str, status: Any) -> dict[str, Any]:
    result = response_status_to_dict(status)
    result["service"] = service_name
    result["status"] = "ok" if result["success"] else "error"
    return result


class AutowareRuntimeClient:
    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._client_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._node: Node | None = None
        self._route_client = None
        self._clear_route_client = None
        self._change_to_autonomous_client = None
        self._change_to_stop_client = None
        self._enable_control_client = None
        self._disable_control_client = None
        self._localization_initialize_client = None
        self._route_state_subscription = None
        self._operation_mode_state_subscription = None
        self._kinematic_state_subscription = None
        self._raw_gps_subscription = None
        self._fixposition_status_subscription = None
        self._fixposition_static_tf_subscription = None
        self._route_state: RouteState | None = None
        self._operation_mode_state: OperationModeState | None = None
        self._kinematic_state: Odometry | None = None
        self._kinematic_state_received_monotonic: float | None = None
        self._raw_gps: NavSatFix | None = None
        self._raw_gps_received_monotonic: float | None = None
        self._fixposition_status: FpaOdomstatus | None = None
        self._fixposition_status_received_monotonic: float | None = None
        self._fixposition_enu0_ecef: tuple[float, float, float] | None = None
        self._localization_init_state = None
        self._ndt_pose = None
        self._ndt_pose_received_monotonic = None
        self._pointcloud_map_signature = None
        self._localization_state_subscriptions = []
        self._initialized = False

    def _ensure_ready(self) -> None:
        if self._initialized and self._node is not None and rclpy.ok():
            return

        with self._init_lock:
            if self._initialized and self._node is not None and rclpy.ok():
                return

            if self._node is not None:
                try:
                    self._node.destroy_node()
                except Exception:
                    pass

            self._node = None
            self._route_client = None
            self._clear_route_client = None
            self._change_to_autonomous_client = None
            self._change_to_stop_client = None
            self._enable_control_client = None
            self._disable_control_client = None
            self._localization_initialize_client = None
            self._route_state_subscription = None
            self._operation_mode_state_subscription = None
            self._kinematic_state_subscription = None
            self._raw_gps_subscription = None
            self._fixposition_status_subscription = None
            self._fixposition_static_tf_subscription = None
            self._fixposition_status = None
            self._fixposition_status_received_monotonic = None
            self._fixposition_enu0_ecef = None
            self._localization_init_state = None
            self._ndt_pose = None
            self._ndt_pose_received_monotonic = None
            self._pointcloud_map_signature = None
            self._localization_state_subscriptions = []
            self._initialized = False

            if not rclpy.ok():
                rclpy.init(args=None)

            self._node = Node("ui_autoware_runtime_client")
            status_qos = QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._route_client = self._node.create_client(
                SetRoutePoints, AUTOWARE_ROUTE_SET_SERVICE
            )
            self._clear_route_client = self._node.create_client(
                ClearRoute, AUTOWARE_ROUTE_CLEAR_SERVICE
            )
            self._change_to_autonomous_client = self._node.create_client(
                ChangeOperationMode, AUTOWARE_CHANGE_TO_AUTONOMOUS_SERVICE
            )
            self._change_to_stop_client = self._node.create_client(
                ChangeOperationMode, AUTOWARE_CHANGE_TO_STOP_SERVICE
            )
            self._enable_control_client = self._node.create_client(
                ChangeOperationMode, AUTOWARE_ENABLE_CONTROL_SERVICE
            )
            self._disable_control_client = self._node.create_client(
                ChangeOperationMode, AUTOWARE_DISABLE_CONTROL_SERVICE
            )
            self._localization_initialize_client = self._node.create_client(
                InitializeLocalization, AUTOWARE_LOCALIZATION_INITIALIZE_SERVICE
            )
            self._route_state_subscription = self._node.create_subscription(
                RouteState,
                AUTOWARE_ROUTE_STATE_TOPIC,
                self._on_route_state,
                status_qos,
            )
            self._operation_mode_state_subscription = self._node.create_subscription(
                OperationModeState,
                AUTOWARE_OPERATION_MODE_STATE_TOPIC,
                self._on_operation_mode_state,
                status_qos,
            )
            self._kinematic_state_subscription = self._node.create_subscription(
                Odometry,
                AUTOWARE_LOCALIZATION_KINEMATIC_STATE_TOPIC,
                self._on_kinematic_state,
                10,
            )
            self._raw_gps_subscription = self._node.create_subscription(
                NavSatFix,
                FIXPOSITION_LLH_TOPIC,
                self._on_raw_gps,
                qos_profile_sensor_data,
            )
            self._fixposition_status_subscription = self._node.create_subscription(
                FpaOdomstatus,
                FIXPOSITION_STATUS_TOPIC,
                self._on_fixposition_status,
                qos_profile_sensor_data,
            )
            self._fixposition_static_tf_subscription = self._node.create_subscription(
                TFMessage,
                FIXPOSITION_STATIC_TF_TOPIC,
                self._on_fixposition_static_tf,
                status_qos,
            )
            self._localization_state_subscriptions = [
                self._node.create_subscription(
                    LocalizationInitializationState, LOCALIZATION_INITIALIZATION_STATE_TOPIC,
                    self._on_localization_init_state, status_qos),
                self._node.create_subscription(
                    PoseWithCovarianceStamped, UNIVERSE_NDT_POSE_TOPIC,
                    self._on_ndt_pose, qos_profile_sensor_data),
                self._node.create_subscription(
                    PointCloud2, AUTOWARE_POINTCLOUD_MAP_TOPIC,
                    self._on_pointcloud_map, status_qos),
            ]
            self._initialized = True

    def _on_route_state(self, message: RouteState) -> None:
        with self._state_lock:
            self._route_state = message

    def _on_operation_mode_state(self, message: OperationModeState) -> None:
        with self._state_lock:
            self._operation_mode_state = message

    def _on_kinematic_state(self, message: Odometry) -> None:
        with self._state_lock:
            self._kinematic_state = message
            self._kinematic_state_received_monotonic = time.monotonic()

    def _on_raw_gps(self, message: NavSatFix) -> None:
        with self._state_lock:
            self._raw_gps = message
            self._raw_gps_received_monotonic = time.monotonic()

    def _on_fixposition_status(self, message: FpaOdomstatus) -> None:
        with self._state_lock:
            self._fixposition_status = message
            self._fixposition_status_received_monotonic = time.monotonic()

    def _on_fixposition_static_tf(self, message: TFMessage) -> None:
        for transform in message.transforms:
            if (
                transform.header.frame_id.lstrip("/") == "FP_ECEF"
                and transform.child_frame_id.lstrip("/") == "FP_ENU0"
            ):
                translation = transform.transform.translation
                with self._state_lock:
                    self._fixposition_enu0_ecef = (
                        translation.x,
                        translation.y,
                        translation.z,
                    )

    def _spin_locked(self, duration_s: float) -> None:
        assert self._node is not None
        if SERVER_SHUTTING_DOWN.is_set() or not rclpy.ok():
            raise RuntimeError("ROS 2 context is not available")
        deadline = time.monotonic() + max(0.0, duration_s)
        while time.monotonic() < deadline:
            if SERVER_SHUTTING_DOWN.is_set() or not rclpy.ok():
                raise RuntimeError("ROS 2 context is not available")
            remaining = deadline - time.monotonic()
            try:
                rclpy.spin_once(self._node, timeout_sec=min(0.05, remaining))
            except Exception as exc:
                if is_invalid_ros_context_error(exc):
                    raise RuntimeError("ROS 2 context is not available") from exc
                raise

    def _call_service_locked(
        self, client: Any, service_name: str, request: Any, timeout_s: float
    ) -> dict[str, Any]:
        assert self._node is not None

        if not client.wait_for_service(timeout_sec=timeout_s):
            return {
                "status": "error",
                "success": False,
                "service": service_name,
                "error": f"service is not available: {service_name}",
            }

        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return {
                    "status": "error",
                    "success": False,
                    "service": service_name,
                    "error": f"timed out while waiting for {service_name}",
                }
            rclpy.spin_once(self._node, timeout_sec=min(0.1, remaining))

        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - ROS client safety path
            return {
                "status": "error",
                "success": False,
                "service": service_name,
                "error": f"service call failed: {exc}",
            }

        if response is None:
            return {
                "status": "error",
                "success": False,
                "service": service_name,
                "error": f"failed to receive a response from {service_name}",
            }

        return build_service_result(service_name, response.status)

    def _operation_mode_state_to_dict(
        self, message: OperationModeState | None
    ) -> dict[str, Any]:
        if message is None:
            return {
                "received": False,
                "mode": None,
                "label": "UNKNOWN",
                "is_autoware_control_enabled": False,
                "is_in_transition": False,
                "available_modes": {
                    "stop": False,
                    "autonomous": False,
                    "local": False,
                    "remote": False,
                },
            }

        return {
            "received": True,
            "mode": int(message.mode),
            "label": OPERATION_MODE_LABELS.get(message.mode, f"UNKNOWN({message.mode})"),
            "is_autoware_control_enabled": bool(message.is_autoware_control_enabled),
            "is_in_transition": bool(message.is_in_transition),
            "available_modes": {
                "stop": bool(message.is_stop_mode_available),
                "autonomous": bool(message.is_autonomous_mode_available),
                "local": bool(message.is_local_mode_available),
                "remote": bool(message.is_remote_mode_available),
            },
        }

    def _route_state_to_dict(self, message: RouteState | None) -> dict[str, Any]:
        if message is None:
            return {
                "received": False,
                "state": None,
                "label": "UNKNOWN",
            }

        return {
            "received": True,
            "state": int(message.state),
            "label": ROUTE_STATE_LABELS.get(message.state, f"UNKNOWN({message.state})"),
        }

    def snapshot(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self._route_client is not None
        assert self._clear_route_client is not None
        assert self._change_to_autonomous_client is not None
        assert self._change_to_stop_client is not None
        assert self._enable_control_client is not None
        assert self._disable_control_client is not None
        assert self._localization_initialize_client is not None

        with self._client_lock:
            self._spin_locked(0.05)
            services = {
                "set_route_points": bool(self._route_client.service_is_ready()),
                "clear_route": bool(self._clear_route_client.service_is_ready()),
                "change_to_autonomous": bool(
                    self._change_to_autonomous_client.service_is_ready()
                ),
                "change_to_stop": bool(self._change_to_stop_client.service_is_ready()),
                "enable_autoware_control": bool(
                    self._enable_control_client.service_is_ready()
                ),
                "disable_autoware_control": bool(
                    self._disable_control_client.service_is_ready()
                ),
                "initialize_localization": bool(
                    self._localization_initialize_client.service_is_ready()
                ),
            }

        with self._state_lock:
            route_state = self._route_state
            operation_mode_state = self._operation_mode_state

        return {
            "status": "ok",
            "services": services,
            "route_state": self._route_state_to_dict(route_state),
            "operation_mode": self._operation_mode_state_to_dict(operation_mode_state),
        }

    @staticmethod
    def _is_recent(received_monotonic: float | None) -> bool:
        return (
            received_monotonic is not None
            and 0.0 <= time.monotonic() - received_monotonic < FIXPOSITION_DATA_TIMEOUT_S
        )

    def _fixposition_ready(
        self, status: FpaOdomstatus | None, received_monotonic: float | None
    ) -> bool:
        return (
            status is not None
            and self._is_recent(received_monotonic)
            and status.init_status == status.consts.INIT_STATUS_GLOBAL_INIT
            and (
                status.gnss1_status > status.consts.GNSS_STATUS_NO_FIX
                or status.gnss2_status > status.consts.GNSS_STATUS_NO_FIX
            )
        )

    def current_fp_enu0_origin(self) -> GeoPoint:
        self._ensure_ready()
        with self._client_lock:
            self._spin_locked(0.2)

        with self._state_lock:
            status = self._fixposition_status
            status_time = self._fixposition_status_received_monotonic
            ecef = self._fixposition_enu0_ecef

        if not self._fixposition_ready(status, status_time):
            raise MapGenerationError(
                "Fixposition must have recent global initialization and GNSS fix "
                "before using its FP_ENU0 origin"
            )
        if ecef is None:
            raise MapGenerationError(
                "FP_ECEF -> FP_ENU0 transform is not available on "
                f"{FIXPOSITION_STATIC_TF_TOPIC}"
            )
        return geo_point_from_ecef(ecef)

    def _raw_gps_to_geo_point(
        self, message: NavSatFix | None, received_monotonic: float | None
    ) -> GeoPoint | None:
        if (
            message is None
            or not self._is_recent(received_monotonic)
            or message.status.status == message.status.STATUS_NO_FIX
            or message.position_covariance_type == message.COVARIANCE_TYPE_UNKNOWN
        ):
            return None

        latitude = finite_float_or_none(message.latitude)
        longitude = finite_float_or_none(message.longitude)
        altitude = finite_float_or_none(message.altitude)
        if (
            latitude is None
            or longitude is None
            or not -90.0 <= latitude <= 90.0
            or not -180.0 <= longitude <= 180.0
        ):
            return None

        return GeoPoint(
            latitude=latitude,
            longitude=longitude,
            altitude=0.0 if altitude is None else altitude,
        )

    def current_raw_gps_point(self) -> GeoPoint | None:
        self._ensure_ready()

        with self._client_lock:
            self._spin_locked(0.02)

        with self._state_lock:
            raw_gps = self._raw_gps
            raw_gps_received_monotonic = self._raw_gps_received_monotonic

        return self._raw_gps_to_geo_point(raw_gps, raw_gps_received_monotonic)

    def current_geo_local_anchor(self) -> GeoLocalAnchor | None:
        self._ensure_ready()

        with self._client_lock:
            self._spin_locked(0.02)

        with self._state_lock:
            kinematic_state = self._kinematic_state
            kinematic_state_received_monotonic = self._kinematic_state_received_monotonic
            raw_gps = self._raw_gps
            raw_gps_received_monotonic = self._raw_gps_received_monotonic
            fixposition_status = self._fixposition_status
            fixposition_status_received_monotonic = self._fixposition_status_received_monotonic

        if (
            kinematic_state is None
            or not self._is_recent(kinematic_state_received_monotonic)
            or not self._fixposition_ready(
                fixposition_status, fixposition_status_received_monotonic
            )
        ):
            return None

        geo = self._raw_gps_to_geo_point(raw_gps, raw_gps_received_monotonic)
        if geo is None:
            return None

        position = kinematic_state.pose.pose.position
        local_x = finite_float_or_none(position.x)
        local_y = finite_float_or_none(position.y)
        local_z = finite_float_or_none(position.z)
        if local_x is None or local_y is None:
            return None

        return GeoLocalAnchor(
            geo=geo,
            local=LocalPoint(
                x=local_x,
                y=local_y,
                z=0.0 if local_z is None else local_z,
            ),
            source="current_vehicle_gps_and_local_pose",
        )


    def _on_localization_init_state(self, message):
        with self._state_lock:
            self._localization_init_state = int(message.state)

    def _on_ndt_pose(self, message):
        with self._state_lock:
            self._ndt_pose = message
            self._ndt_pose_received_monotonic = time.monotonic()

    def _on_pointcloud_map(self, message):
        # Keep metadata only, not the potentially very large point-cloud payload.
        with self._state_lock:
            self._pointcloud_map_signature = (
                str(message.header.frame_id).lstrip('/'), int(message.width) * int(message.height))

    def _universe_kinematic_is_valid(self, message, received_monotonic):
        now_ns = self._node.get_clock().now().nanoseconds if self._node else time.time_ns()
        return universe_pose_is_fresh(message, received_monotonic, now_ns, odometry=True)

    def localization_snapshot(self):
        self._ensure_ready()
        with self._client_lock:
            self._spin_locked(0.02)
        with self._state_lock:
            init_state = self._localization_init_state
            odom, odom_received = self._kinematic_state, self._kinematic_state_received_monotonic
            ndt, ndt_received = self._ndt_pose, self._ndt_pose_received_monotonic
            loaded_map = self._pointcloud_map_signature
            fix, fix_received = self._fixposition_status, self._fixposition_status_received_monotonic
        if SCOUT_LOCALIZATION_MODE == "fixposition":
            ready = self._fixposition_ready(fix, fix_received) and self._is_recent(odom_received)
            return {"backend": "fixposition", "ready": ready, "initialized": ready,
                    "initialization_state": init_state, "kinematic_recent": self._is_recent(odom_received)}
        now_ns = self._node.get_clock().now().nanoseconds if self._node else time.time_ns()
        return universe_localization_readiness(
            init_state, odom, odom_received, ndt, ndt_received, loaded_map, now_ns)

    def vehicle_state_snapshot(self):
        result = self._vehicle_state_snapshot()
        result["localization_backend"] = SCOUT_LOCALIZATION_MODE
        result["localization"] = self.localization_snapshot()
        if SCOUT_LOCALIZATION_MODE == "universe":
            # Raw Fixposition GPS is a diagnostic only; the selected position is NDT/EKF.
            result["pose"]["position"].update(latitude=None, longitude=None, altitude=None)
            result.update(geo_available=False, geo_source=None,
                          geo_error="Universe local pose requires a surveyed map origin for geographic display")
            if not self._universe_kinematic_is_valid(
                    self._kinematic_state, self._kinematic_state_received_monotonic):
                result.update(received=False, source_topic=AUTOWARE_LOCALIZATION_KINEMATIC_STATE_TOPIC,
                              age_ms=None, error="Universe localization has no fresh valid map/base_link odometry")
            elif result.get("runtime_map", {}).get("status") == "ok":
                origin = result["runtime_map"]["origin"]
                _, inverse = build_local_transformers(GeoPoint(**origin))
                position = result["pose"]["position"]
                lon, lat = inverse.transform(position["x"], position["y"])
                if math.isfinite(lon) and math.isfinite(lat):
                    position.update(latitude=lat, longitude=lon,
                                    altitude=origin["altitude"] + position["z"])
                    result.update(geo_available=True, geo_source="universe_map_projection", geo_error=None)
        return result

    def _vehicle_state_snapshot(self) -> dict[str, Any]:
        self._ensure_ready()

        with self._client_lock:
            self._spin_locked(0.02)

        with self._state_lock:
            kinematic_state = self._kinematic_state
            received_monotonic = self._kinematic_state_received_monotonic
            raw_gps = self._raw_gps
            raw_gps_received_monotonic = self._raw_gps_received_monotonic
            fixposition_status = self._fixposition_status
            fixposition_status_received_monotonic = self._fixposition_status_received_monotonic

        fixposition_ready = self._fixposition_ready(
            fixposition_status, fixposition_status_received_monotonic
        )
        if (
            not self._universe_kinematic_is_valid(kinematic_state, received_monotonic)
            if SCOUT_LOCALIZATION_MODE == "universe"
            else not fixposition_ready or not self._is_recent(received_monotonic)
        ):
            kinematic_state = None
        raw_geo_position = self._raw_gps_to_geo_point(raw_gps, raw_gps_received_monotonic)
        raw_gps_age_ms = None
        if raw_gps_received_monotonic is not None:
            raw_gps_age_ms = max(0, int(round((time.monotonic() - raw_gps_received_monotonic) * 1000.0)))

        if kinematic_state is None:
            return {
                "status": "ok",
                "received": raw_geo_position is not None,
                "source_topic": (
                    FIXPOSITION_LLH_TOPIC if raw_geo_position is not None else AUTOWARE_LOCALIZATION_KINEMATIC_STATE_TOPIC
                ),
                "age_ms": raw_gps_age_ms,
                "error": (
                    None
                    if raw_geo_position is not None
                    else f"no message received on {AUTOWARE_LOCALIZATION_KINEMATIC_STATE_TOPIC}"
                ),
                "pose": {
                    "position": {
                        "x": None,
                        "y": None,
                        "z": None,
                        "latitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.latitude),
                        "longitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.longitude),
                        "altitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.altitude),
                    },
                    "orientation": {
                        "x": None,
                        "y": None,
                        "z": None,
                        "w": None,
                        "roll": None,
                        "pitch": None,
                        "yaw": None,
                    },
                },
                "twist": {
                    "linear": {
                        "x": None,
                        "y": None,
                        "z": None,
                    },
                    "angular": {
                        "x": None,
                        "y": None,
                        "z": None,
                    },
                },
                "runtime_map": {
                    "status": "error",
                    "error": f"no message received on {AUTOWARE_LOCALIZATION_KINEMATIC_STATE_TOPIC}",
                },
                "raw_gps": {
                    "received": raw_geo_position is not None,
                    "age_ms": raw_gps_age_ms,
                    "source_topic": FIXPOSITION_LLH_TOPIC,
                    "latitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.latitude),
                    "longitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.longitude),
                    "altitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.altitude),
                },
                "geo_available": raw_geo_position is not None,
                "geo_error": None if raw_geo_position is not None else "Fixposition GPS 尚未就绪",
                "geo_source": "fixposition_llh" if raw_geo_position is not None else None,
                "fixposition_ready": fixposition_ready,
            }

        position = kinematic_state.pose.pose.position
        orientation = kinematic_state.pose.pose.orientation
        twist = kinematic_state.twist.twist
        local_x = finite_float_or_none(position.x)
        local_y = finite_float_or_none(position.y)
        local_z = finite_float_or_none(position.z)
        orientation_x = finite_float_or_none(orientation.x)
        orientation_y = finite_float_or_none(orientation.y)
        orientation_z = finite_float_or_none(orientation.z)
        orientation_w = finite_float_or_none(orientation.w)
        linear_x = finite_float_or_none(twist.linear.x)
        linear_y = finite_float_or_none(twist.linear.y)
        linear_z = finite_float_or_none(twist.linear.z)
        angular_x = finite_float_or_none(twist.angular.x)
        angular_y = finite_float_or_none(twist.angular.y)
        angular_z = finite_float_or_none(twist.angular.z)
        roll, pitch, yaw = quaternion_to_euler_xyz(
            0.0 if orientation_x is None else orientation_x,
            0.0 if orientation_y is None else orientation_y,
            0.0 if orientation_z is None else orientation_z,
            1.0 if orientation_w is None else orientation_w,
        )

        runtime_map_error = None
        try:
            context = resolve_vehicle_state_runtime_map_context()
            runtime_map = runtime_map_context_to_dict(context)
        except MapGenerationError as exc:
            runtime_map = {
                "status": "error",
                "error": str(exc),
            }
            runtime_map_error = str(exc)

        geo_source = "fixposition_llh" if raw_geo_position is not None else None
        geo_error = None if raw_geo_position is not None else "Fixposition GPS 尚未就绪"
        if geo_error is not None and runtime_map_error is not None:
            geo_error = f"{geo_error}；{runtime_map_error}"

        age_ms = None
        if received_monotonic is not None:
            age_ms = max(0, int(round((time.monotonic() - received_monotonic) * 1000.0)))

        return {
            "status": "ok",
            "received": True,
            "source_topic": AUTOWARE_LOCALIZATION_KINEMATIC_STATE_TOPIC,
            "age_ms": age_ms,
            "frame_id": str(kinematic_state.header.frame_id),
            "child_frame_id": str(kinematic_state.child_frame_id),
            "stamp": {
                "sec": int(kinematic_state.header.stamp.sec),
                "nanosec": int(kinematic_state.header.stamp.nanosec),
            },
            "pose": {
                "position": {
                    "x": local_x,
                    "y": local_y,
                    "z": local_z,
                    "latitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.latitude),
                    "longitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.longitude),
                    "altitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.altitude),
                },
                "orientation": {
                    "x": orientation_x,
                    "y": orientation_y,
                    "z": orientation_z,
                    "w": orientation_w,
                    "roll": finite_float_or_none(roll),
                    "pitch": finite_float_or_none(pitch),
                    "yaw": finite_float_or_none(yaw),
                },
            },
            "raw_gps": {
                "received": raw_geo_position is not None,
                "age_ms": raw_gps_age_ms,
                "source_topic": FIXPOSITION_LLH_TOPIC,
                "latitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.latitude),
                "longitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.longitude),
                "altitude": None if raw_geo_position is None else finite_float_or_none(raw_geo_position.altitude),
            },
            "twist": {
                "linear": {
                    "x": linear_x,
                    "y": linear_y,
                    "z": linear_z,
                },
                "angular": {
                    "x": angular_x,
                    "y": angular_y,
                    "z": angular_z,
                },
            },
            "runtime_map": runtime_map,
            "geo_available": raw_geo_position is not None,
            "geo_error": geo_error,
            "geo_source": geo_source,
            "fixposition_ready": fixposition_ready,
        }

    def set_route(
        self,
        poses: list[Pose],
        allow_goal_modification: bool = False,
        clear_existing: bool = True,
    ) -> dict[str, Any]:
        self._ensure_ready()
        assert self._node is not None
        assert self._route_client is not None
        assert self._clear_route_client is not None

        if len(poses) < 2:
            raise MapGenerationError("at least two poses are required to set a route")

        with self._client_lock:
            clear_result = None
            if clear_existing:
                clear_result = self._call_service_locked(
                    self._clear_route_client,
                    AUTOWARE_ROUTE_CLEAR_SERVICE,
                    ClearRoute.Request(),
                    AUTOWARE_ROUTE_SERVICE_TIMEOUT_S,
                )

            request = SetRoutePoints.Request()
            request.header.frame_id = "map"
            request.header.stamp = self._node.get_clock().now().to_msg()
            request.goal = poses[-1]
            request.waypoints = poses[1:-1]
            request.option.allow_goal_modification = bool(allow_goal_modification)

            set_result = self._call_service_locked(
                self._route_client,
                AUTOWARE_ROUTE_SET_SERVICE,
                request,
                AUTOWARE_ROUTE_SERVICE_TIMEOUT_S,
            )

        return {
            "status": set_result["status"],
            "clear_existing": bool(clear_existing),
            "clear_route": clear_result,
            "set_route": set_result,
            "waypoint_count": len(poses) - 2,
        }

    def clear_route(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self._clear_route_client is not None
        with self._client_lock:
            result = self._call_service_locked(
                self._clear_route_client,
                AUTOWARE_ROUTE_CLEAR_SERVICE,
                ClearRoute.Request(),
                AUTOWARE_ROUTE_SERVICE_TIMEOUT_S,
            )
        return {
            "status": result["status"],
            "clear_route": result,
        }

    def change_to_autonomous(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self._change_to_autonomous_client is not None
        with self._client_lock:
            result = self._call_service_locked(
                self._change_to_autonomous_client,
                AUTOWARE_CHANGE_TO_AUTONOMOUS_SERVICE,
                ChangeOperationMode.Request(),
                AUTOWARE_OPERATION_MODE_TIMEOUT_S,
            )
        return {
            "status": result["status"],
            "change_to_autonomous": result,
        }

    def change_to_stop(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self._change_to_stop_client is not None
        with self._client_lock:
            result = self._call_service_locked(
                self._change_to_stop_client,
                AUTOWARE_CHANGE_TO_STOP_SERVICE,
                ChangeOperationMode.Request(),
                AUTOWARE_OPERATION_MODE_TIMEOUT_S,
            )
        return {
            "status": result["status"],
            "change_to_stop": result,
        }

    def enable_autoware_control(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self._enable_control_client is not None
        with self._client_lock:
            result = self._call_service_locked(
                self._enable_control_client,
                AUTOWARE_ENABLE_CONTROL_SERVICE,
                ChangeOperationMode.Request(),
                AUTOWARE_OPERATION_MODE_TIMEOUT_S,
            )
        return {
            "status": result["status"],
            "enable_autoware_control": result,
        }

    def disable_autoware_control(self) -> dict[str, Any]:
        self._ensure_ready()
        assert self._disable_control_client is not None
        with self._client_lock:
            result = self._call_service_locked(
                self._disable_control_client,
                AUTOWARE_DISABLE_CONTROL_SERVICE,
                ChangeOperationMode.Request(),
                AUTOWARE_OPERATION_MODE_TIMEOUT_S,
            )
        return {
            "status": result["status"],
            "disable_autoware_control": result,
        }

    def reset_for_route_planning(
        self, wait_timeout_s: float = AUTOWARE_RESET_WAIT_TIMEOUT_S
    ) -> dict[str, Any]:
        self._ensure_ready()
        assert self._change_to_stop_client is not None
        assert self._disable_control_client is not None
        assert self._clear_route_client is not None
        assert self._node is not None

        wait_timeout_s = max(0.0, float(wait_timeout_s))
        operation_timeout_s = max(wait_timeout_s, AUTOWARE_OPERATION_MODE_TIMEOUT_S)
        route_timeout_s = max(wait_timeout_s, AUTOWARE_ROUTE_SERVICE_TIMEOUT_S)

        with self._client_lock:
            stop_result = self._call_service_locked(
                self._change_to_stop_client,
                AUTOWARE_CHANGE_TO_STOP_SERVICE,
                ChangeOperationMode.Request(),
                operation_timeout_s,
            )
            disable_control_result = self._call_service_locked(
                self._disable_control_client,
                AUTOWARE_DISABLE_CONTROL_SERVICE,
                ChangeOperationMode.Request(),
                operation_timeout_s,
            )
            clear_route_result = self._call_service_locked(
                self._clear_route_client,
                AUTOWARE_ROUTE_CLEAR_SERVICE,
                ClearRoute.Request(),
                route_timeout_s,
            )
            self._spin_locked(0.2)

        try:
            rviz_cleanup_result = PLANNING_RESET_PUBLISHER.publish_reset()
        except Exception as exc:
            rviz_cleanup_result = {
                "status": "error",
                "error": f"failed to publish reset cleanup topics: {exc}",
            }

        deadline = time.monotonic() + min(wait_timeout_s, 3.0)
        snapshot = None
        while True:
            snapshot = self.snapshot()
            route_state = snapshot.get("route_state", {})
            operation_mode = snapshot.get("operation_mode", {})
            if route_state.get("label") == "UNSET" and operation_mode.get("label") == "STOP":
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.2)

        assert snapshot is not None
        route_state = snapshot.get("route_state", {})
        operation_mode = snapshot.get("operation_mode", {})

        verification_issues = []
        if route_state.get("label") != "UNSET":
            verification_issues.append(
                f"route_state={route_state.get('label', 'UNKNOWN')}"
            )
        if operation_mode.get("label") != "STOP":
            verification_issues.append(
                f"operation_mode={operation_mode.get('label', 'UNKNOWN')}"
            )
        if rviz_cleanup_result.get("status") != "ok":
            verification_issues.append(
                rviz_cleanup_result.get("error", "rviz_cleanup_failed")
            )

        results = {
            "change_to_stop": stop_result,
            "disable_autoware_control": disable_control_result,
            "clear_route": clear_route_result,
            "rviz_cleanup": rviz_cleanup_result,
        }
        success_count = sum(1 for result in results.values() if result.get("status") == "ok")
        failed_actions = [
            action_name for action_name, result in results.items() if result.get("status") != "ok"
        ]
        overall_status = "ok"
        if success_count == 0:
            overall_status = "error"
        elif failed_actions or verification_issues:
            overall_status = "partial"

        return {
            "status": overall_status,
            "target_state": "ready_for_route_planning",
            "wait_timeout_s": wait_timeout_s,
            "change_to_stop": stop_result,
            "disable_autoware_control": disable_control_result,
            "clear_route": clear_route_result,
            "rviz_cleanup": rviz_cleanup_result,
            "failed_actions": failed_actions,
            "verification_issues": verification_issues,
            "autoware_status": snapshot,
        }

    def is_localization_initialize_available(self, timeout_s: float = 0.0) -> bool:
        self._ensure_ready()
        assert self._localization_initialize_client is not None

        with self._client_lock:
            self._spin_locked(0.05)
            if timeout_s > 0.0:
                return bool(
                    self._localization_initialize_client.wait_for_service(timeout_sec=timeout_s)
                )
            return bool(self._localization_initialize_client.service_is_ready())

    def initialize_localization(self, pose: Pose) -> dict[str, Any]:
        self._ensure_ready()
        assert self._node is not None
        assert self._localization_initialize_client is not None

        request_pose = PoseWithCovarianceStamped()
        request_pose.header.frame_id = "map"
        request_pose.header.stamp = self._node.get_clock().now().to_msg()
        request_pose.pose.pose = pose

        request = InitializeLocalization.Request()
        request.pose = [request_pose]

        with self._client_lock:
            result = self._call_service_locked(
                self._localization_initialize_client,
                AUTOWARE_LOCALIZATION_INITIALIZE_SERVICE,
                request,
                AUTOWARE_LOCALIZATION_TIMEOUT_S,
            )

        return {
            "status": result["status"],
            "initialize_localization": result,
        }


AUTOWARE_RUNTIME_CLIENT = AutowareRuntimeClient()


def sanitize_map_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or DEFAULT_MAP_NAME


def load_scout_vehicle_width_m() -> float:
    try:
        vehicle_info = yaml.safe_load(SCOUT_VEHICLE_INFO_PATH.read_text(encoding="utf-8"))
        parameters = vehicle_info["/**"]["ros__parameters"]
        return (
            float(parameters["wheel_tread"])
            + float(parameters["left_overhang"])
            + float(parameters["right_overhang"])
        )
    except Exception as exc:
        raise MapGenerationError(
            f"failed to read Scout vehicle width from {SCOUT_VEHICLE_INFO_PATH}: {exc}"
        ) from exc


DEFAULT_VEHICLE_WIDTH_M = load_scout_vehicle_width_m()
DEFAULT_LANE_WIDTH_M = DEFAULT_VEHICLE_WIDTH_M + DEFAULT_LANE_EXTRA_MARGIN_M


def load_scout_min_turn_radius_m() -> float:
    try:
        vehicle_info = yaml.safe_load(SCOUT_VEHICLE_INFO_PATH.read_text(encoding="utf-8"))
        parameters = vehicle_info["/**"]["ros__parameters"]
        wheelbase = float(parameters["wheel_base"])
        steering_limit = float(parameters["max_steer_angle"])
        if not math.isfinite(wheelbase) or wheelbase <= 0.0:
            raise ValueError("wheel_base must be finite and positive")
        if not math.isfinite(steering_limit) or not 0.0 < steering_limit < math.pi / 2.0:
            raise ValueError("max_steer_angle must be between 0 and pi/2")
        # Match both the equivalent steering model and the Scout interface's yaw-rate limit.
        return max(
            wheelbase / math.tan(steering_limit),
            (DEFAULT_SPEED_LIMIT_KPH / 3.6) / SCOUT_MAX_YAW_RATE_RADPS,
        )
    except Exception as exc:
        raise MapGenerationError(
            f"failed to read Scout turning limits from {SCOUT_VEHICLE_INFO_PATH}: {exc}"
        ) from exc


DEFAULT_MIN_TURN_RADIUS_M = load_scout_min_turn_radius_m()


def load_payload(source: Path) -> dict[str, Any]:
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MapGenerationError(f"invalid JSON in {source}: {exc}") from exc


def parse_float(payload: dict[str, Any], *keys: str, default: float) -> float:
    for key in keys:
        if key in payload and payload[key] is not None:
            return float(payload[key])
    return float(default)


def parse_bool(payload: dict[str, Any], *keys: str, default: bool) -> bool:
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    for key in keys:
        if key not in payload or payload[key] is None:
            continue

        value = payload[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in truthy:
                return True
            if lowered in falsy:
                return False
        raise MapGenerationError(f"{key} must be a boolean value")

    return default


def parse_int(payload: dict[str, Any], *keys: str, default: int) -> int:
    for key in keys:
        if key in payload and payload[key] is not None:
            return int(payload[key])
    return int(default)


def load_google_maps_api_key() -> str:
    index_path = SCRIPT_DIR / "index.html"
    try:
        html = index_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise MapGenerationError(f"failed to read Google Maps key from {index_path}: {exc}") from exc

    match = re.search(r"maps/api/js\?key=([^&\"']+)", html)
    if not match:
        raise MapGenerationError(f"Google Maps key not found in {index_path}")
    return match.group(1)


def clamp_int(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def load_active_map_origin() -> GeoPoint | None:
    snapshot_path = DEFAULT_OUTPUT_ROOT / DEFAULT_MAP_NAME / "map_generation_request.json"
    if not snapshot_path.exists():
        return None

    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        origin = snapshot.get("origin") or {}
        return GeoPoint(
            latitude=float(origin["latitude"]),
            longitude=float(origin["longitude"]),
            altitude=float(origin.get("altitude", 0.0)),
        )
    except Exception:
        return None


def iter_ins_driver_config_paths() -> list[Path]:
    candidates = [
        INS_DRIVER_BUILD_CONFIG_PATH,
        INS_DRIVER_INSTALL_CONFIG_PATH,
        INS_DRIVER_SOURCE_CONFIG_PATH,
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except Exception:
            resolved = candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() or candidate.parent.exists():
            paths.append(candidate)
    return paths


def load_configured_ins_origin() -> GeoPoint | None:
    for config_path in iter_ins_driver_config_paths():
        if not config_path.exists():
            continue

        try:
            raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            parameters = raw_config.get("ins_driver_ez21", {}).get("ros__parameters", {})
        except Exception:
            continue

        if not bool(parameters.get(INS_ORIGIN_ACTIVE_PARAMETER, False)):
            continue

        latitude = finite_float_or_none(parameters.get(INS_ORIGIN_LATITUDE_PARAMETER))
        longitude = finite_float_or_none(parameters.get(INS_ORIGIN_LONGITUDE_PARAMETER))
        altitude = finite_float_or_none(parameters.get(INS_ORIGIN_ALTITUDE_PARAMETER))
        if latitude is None or longitude is None:
            continue

        return GeoPoint(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude if altitude is not None else DEFAULT_ELEVATION_M,
        )

    return None


def persist_configured_ins_origin(point: GeoPoint) -> dict[str, Any]:
    config_paths = iter_ins_driver_config_paths()
    if not config_paths:
        return {
            "status": "error",
            "error": "no writable ins_driver_ez21 config path was found",
        }

    written_paths: list[str] = []
    for config_path in config_paths:
        try:
            if config_path.exists():
                raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            else:
                raw_config = {}

            package_config = raw_config.setdefault("ins_driver_ez21", {})
            parameters = package_config.setdefault("ros__parameters", {})
            parameters[INS_ORIGIN_ACTIVE_PARAMETER] = True
            parameters[INS_ORIGIN_LATITUDE_PARAMETER] = float(point.latitude)
            parameters[INS_ORIGIN_LONGITUDE_PARAMETER] = float(point.longitude)
            parameters[INS_ORIGIN_ALTITUDE_PARAMETER] = float(point.altitude)

            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                yaml.safe_dump(raw_config, sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )
            written_paths.append(str(config_path))
        except Exception as exc:
            return {
                "status": "error",
                "error": f"failed to persist INS reference origin to {config_path}: {exc}",
                "config_files": written_paths,
            }

    return {
        "status": "ok",
        "message": "INS reference origin was persisted to startup config",
        "config_files": written_paths,
        "origin": {
            "latitude": float(point.latitude),
            "longitude": float(point.longitude),
            "altitude": float(point.altitude),
        },
    }


def resolve_runtime_map_dir(
    payload: dict[str, Any], *, include_launch_map: bool = True
) -> Path:
    raw_output_dir = payload.get("runtime_map_dir") or payload.get("output_dir")
    if raw_output_dir:
        return Path(str(raw_output_dir)).expanduser().resolve()

    if include_launch_map:
        launch_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
        if launch_map_dir:
            return Path(launch_map_dir).expanduser().resolve()

    map_name = sanitize_map_name(str(payload.get("map_name", DEFAULT_MAP_NAME)))
    return (DEFAULT_OUTPUT_ROOT / map_name).resolve()


def parse_geo_point_mapping(
    source: dict[str, Any] | None, *, allow_missing: bool = False
) -> GeoPoint | None:
    if not isinstance(source, dict):
        if allow_missing:
            return None
        raise MapGenerationError("map origin is missing")

    latitude = source.get("latitude")
    longitude = source.get("longitude")
    altitude = source.get("altitude", DEFAULT_ELEVATION_M)
    if latitude is None or longitude is None:
        if allow_missing:
            return None
        raise MapGenerationError("map origin must include latitude and longitude")

    try:
        return GeoPoint(
            latitude=float(latitude),
            longitude=float(longitude),
            altitude=float(altitude),
        )
    except (TypeError, ValueError) as exc:
        raise MapGenerationError(f"invalid map origin: {exc}") from exc


def runtime_map_context_to_dict(context: RuntimeMapContext) -> dict[str, Any]:
    return {
        "status": "ok",
        "output_dir": str(context.output_dir),
        "projector_type": context.projector_type,
        "elevation_m": float(context.elevation_m),
        "origin_source": context.origin_source,
        "origin": {
            "latitude": float(context.origin.latitude),
            "longitude": float(context.origin.longitude),
            "altitude": float(context.origin.altitude),
        },
    }


def load_runtime_map_context(
    payload: dict[str, Any], *, include_launch_map: bool = True
) -> RuntimeMapContext:
    output_dir = resolve_runtime_map_dir(payload, include_launch_map=include_launch_map)
    projector_info_path = output_dir / "map_projector_info.yaml"
    if not projector_info_path.exists():
        raise MapGenerationError(f"runtime map projector info not found: {projector_info_path}")

    try:
        projector_info = yaml.safe_load(projector_info_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise MapGenerationError(
            f"failed to read runtime map projector info from {projector_info_path}: {exc}"
        ) from exc

    projector_type = str(projector_info.get("projector_type", "")).strip()
    if not projector_type:
        raise MapGenerationError(f"{projector_info_path} is missing projector_type")

    if projector_type not in {"Local", "LocalCartesian"}:
        raise MapGenerationError(
            f"runtime route UI currently supports only Local/LocalCartesian maps, got {projector_type}"
        )

    origin = parse_geo_point_mapping(projector_info.get("map_origin"), allow_missing=True)
    origin_source = f"{projector_info_path.name}:map_origin" if origin is not None else ""
    elevation_m = DEFAULT_ELEVATION_M

    request_snapshot_path = output_dir / "map_generation_request.json"
    if request_snapshot_path.exists():
        snapshot = load_payload(request_snapshot_path)
        elevation_m = parse_float(snapshot, "elevation_m", default=DEFAULT_ELEVATION_M)
        if origin is None:
            origin = parse_geo_point_mapping(snapshot.get("origin"), allow_missing=True)
            if origin is not None:
                origin_source = f"{request_snapshot_path.name}:origin"

    if origin is None:
        raise MapGenerationError(
            f"failed to resolve map origin from {projector_info_path} or {request_snapshot_path}"
        )

    return RuntimeMapContext(
        output_dir=output_dir,
        projector_type=projector_type,
        origin=origin,
        elevation_m=elevation_m,
        origin_source=origin_source,
    )


def resolve_viewport(payload: dict[str, Any], points: list[GeoPoint]) -> dict[str, float]:
    viewport = payload.get("viewport")
    if isinstance(viewport, dict):
        center = viewport.get("center") or {}
        if "latitude" in center and "longitude" in center and "zoom" in viewport:
            width_px = clamp_int(
                int(viewport.get("width_px", DEFAULT_SATELLITE_IMAGE_WIDTH_PX)),
                128,
                MAX_SATELLITE_IMAGE_DIMENSION_PX,
            )
            height_px = clamp_int(
                int(viewport.get("height_px", DEFAULT_SATELLITE_IMAGE_HEIGHT_PX)),
                128,
                MAX_SATELLITE_IMAGE_DIMENSION_PX,
            )
            return {
                "center_latitude": float(center["latitude"]),
                "center_longitude": float(center["longitude"]),
                "zoom": float(viewport["zoom"]),
                "width_px": float(width_px),
                "height_px": float(height_px),
            }

    min_lat = min(point.latitude for point in points)
    max_lat = max(point.latitude for point in points)
    min_lon = min(point.longitude for point in points)
    max_lon = max(point.longitude for point in points)

    return {
        "center_latitude": (min_lat + max_lat) / 2.0,
        "center_longitude": (min_lon + max_lon) / 2.0,
        "zoom": float(DEFAULT_SATELLITE_ZOOM),
        "width_px": float(DEFAULT_SATELLITE_IMAGE_WIDTH_PX),
        "height_px": float(DEFAULT_SATELLITE_IMAGE_HEIGHT_PX),
    }


def lat_lon_to_world_pixel(latitude: float, longitude: float, zoom: int) -> tuple[float, float]:
    latitude = clamp(latitude, -85.05112878, 85.05112878)
    world_scale = 256.0 * (2**zoom)
    x = (longitude + 180.0) / 360.0 * world_scale
    sin_lat = math.sin(math.radians(latitude))
    y = (
        0.5
        - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)
    ) * world_scale
    return x, y


def world_pixel_to_lat_lon(world_x: float, world_y: float, zoom: int) -> tuple[float, float]:
    world_scale = 256.0 * (2**zoom)
    longitude = world_x / world_scale * 360.0 - 180.0
    mercator_y = math.pi - (2.0 * math.pi * world_y / world_scale)
    latitude = math.degrees(math.atan(math.sinh(mercator_y)))
    return latitude, longitude


def fetch_satellite_image(viewport: dict[str, float]) -> Image.Image:
    google_api_key = load_google_maps_api_key()
    zoom = clamp_int(int(round(viewport["zoom"])), 1, 21)
    width_px = clamp_int(int(round(viewport["width_px"])), 128, MAX_SATELLITE_IMAGE_DIMENSION_PX)
    height_px = clamp_int(int(round(viewport["height_px"])), 128, MAX_SATELLITE_IMAGE_DIMENSION_PX)

    params = {
        "center": f'{viewport["center_latitude"]:.8f},{viewport["center_longitude"]:.8f}',
        "zoom": str(zoom),
        "size": f"{width_px}x{height_px}",
        "maptype": "satellite",
        "key": google_api_key,
    }

    try:
        response = requests.get(GOOGLE_STATIC_MAP_URL, params=params, timeout=20.0)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except Exception as exc:
        raise MapGenerationError(f"failed to fetch the satellite map image: {exc}") from exc


def build_satellite_overlay_marker_array(
    image: Image.Image,
    viewport: dict[str, float],
    origin: GeoPoint,
    sample_columns: int,
    sample_rows: int,
) -> MarkerArray:
    zoom = clamp_int(int(round(viewport["zoom"])), 1, 21)
    center_world_x, center_world_y = lat_lon_to_world_pixel(
        viewport["center_latitude"], viewport["center_longitude"], zoom
    )
    image_width = image.width
    image_height = image.height
    step_x = image_width / sample_columns
    step_y = image_height / sample_rows
    pixel_offset_x = image_width / 2.0
    pixel_offset_y = image_height / 2.0

    forward, _ = build_local_transformers(origin)
    marker = Marker()
    marker.header.frame_id = "map"
    marker.ns = RVIZ_SATELLITE_MARKER_NAMESPACE
    marker.id = 0
    marker.type = Marker.POINTS
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0

    local_rows: list[list[tuple[float, float]]] = []

    for row in range(sample_rows):
        local_row: list[tuple[float, float]] = []
        source_pixel_y = min(image_height - 1, max(0, int((row + 0.5) * step_y)))
        local_pixel_y = (row + 0.5) * step_y - pixel_offset_y

        for column in range(sample_columns):
            source_pixel_x = min(image_width - 1, max(0, int((column + 0.5) * step_x)))
            local_pixel_x = (column + 0.5) * step_x - pixel_offset_x
            world_x = center_world_x + local_pixel_x
            world_y = center_world_y + local_pixel_y
            latitude, longitude = world_pixel_to_lat_lon(world_x, world_y, zoom)
            local_x, local_y = forward.transform(longitude, latitude)
            red, green, blue = image.getpixel((source_pixel_x, source_pixel_y))

            point = Point()
            point.x = float(local_x)
            point.y = float(local_y)
            point.z = DEFAULT_SATELLITE_MARKER_Z_M
            marker.points.append(point)

            color = ColorRGBA()
            color.r = red / 255.0
            color.g = green / 255.0
            color.b = blue / 255.0
            color.a = DEFAULT_SATELLITE_MARKER_ALPHA
            marker.colors.append(color)

            local_row.append((local_x, local_y))

        local_rows.append(local_row)

    spacing_samples: list[float] = []
    for row in local_rows:
        for column in range(1, len(row)):
            spacing_samples.append(math.hypot(row[column][0] - row[column - 1][0], row[column][1] - row[column - 1][1]))
    for row in range(1, len(local_rows)):
        for column in range(len(local_rows[row])):
            spacing_samples.append(
                math.hypot(
                    local_rows[row][column][0] - local_rows[row - 1][column][0],
                    local_rows[row][column][1] - local_rows[row - 1][column][1],
                )
            )

    marker_spacing = max(0.2, (sum(spacing_samples) / len(spacing_samples)) if spacing_samples else 1.0)
    marker.scale.x = marker_spacing
    marker.scale.y = marker_spacing

    marker_array = MarkerArray()
    marker_array.markers.append(marker)
    return marker_array


def publish_satellite_overlay(
    payload: dict[str, Any], path_points: list[GeoPoint], origin: GeoPoint
) -> dict[str, Any]:
    viewport = resolve_viewport(payload, path_points)
    sample_columns = clamp_int(
        parse_int(payload, "satellite_sample_columns", default=DEFAULT_SATELLITE_SAMPLE_COLUMNS),
        8,
        96,
    )
    sample_rows = clamp_int(
        parse_int(payload, "satellite_sample_rows", default=DEFAULT_SATELLITE_SAMPLE_ROWS),
        8,
        96,
    )

    image = fetch_satellite_image(viewport)
    marker_array = build_satellite_overlay_marker_array(
        image=image,
        viewport=viewport,
        origin=origin,
        sample_columns=sample_columns,
        sample_rows=sample_rows,
    )
    RVIZ_MARKER_PUBLISHER.publish(marker_array)

    return {
        "status": "ok",
        "message": "satellite overlay was published to RViz",
        "topic": RVIZ_SATELLITE_MARKER_TOPIC,
        "viewport": {
            "center_latitude": viewport["center_latitude"],
            "center_longitude": viewport["center_longitude"],
            "zoom": int(round(viewport["zoom"])),
            "width_px": int(round(viewport["width_px"])),
            "height_px": int(round(viewport["height_px"])),
        },
        "sample_columns": sample_columns,
        "sample_rows": sample_rows,
    }


def publish_satellite_overlay_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    path_points = parse_path_points(payload, DEFAULT_ELEVATION_M)
    active_origin = load_active_map_origin()
    overlay_origin = active_origin if active_origin is not None else path_points[0]
    overlay_origin_source = "active_map_snapshot" if active_origin is not None else "payload_first_point"

    try:
        result = publish_satellite_overlay(payload, path_points, overlay_origin)
    except Exception as exc:
        return {"status": "error", "error": str(exc), "origin_source": overlay_origin_source}

    result["origin_source"] = overlay_origin_source
    return result


def parse_path_points(
    payload: dict[str, Any], default_elevation_m: float, min_points: int = 2
) -> list[GeoPoint]:
    raw_points = payload.get("data")
    if raw_points is None:
        raw_points = payload.get("points")

    if not isinstance(raw_points, list):
        raise MapGenerationError("payload must contain a 'data' or 'points' list")

    points: list[GeoPoint] = []
    for index, point in enumerate(raw_points):
        if not isinstance(point, dict):
            raise MapGenerationError(f"point #{index + 1} is not an object")

        try:
            latitude = float(point["latitude"])
            longitude = float(point["longitude"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MapGenerationError(
                f"point #{index + 1} must contain numeric latitude/longitude"
            ) from exc

        altitude = parse_float(point, "altitude", "alt", default=default_elevation_m)
        geo_point = GeoPoint(latitude=latitude, longitude=longitude, altitude=altitude)

        if not points:
            points.append(geo_point)
            continue

        previous = points[-1]
        if (
            math.isclose(previous.latitude, geo_point.latitude, abs_tol=1e-10)
            and math.isclose(previous.longitude, geo_point.longitude, abs_tol=1e-10)
        ):
            continue

        points.append(geo_point)

    if len(points) < min_points:
        if min_points == 1:
            raise MapGenerationError("at least one path point is required")
        raise MapGenerationError(f"at least {min_points} distinct path points are required")

    return points


def build_localization_initialize_pose(
    path_points: list[GeoPoint], context: RuntimeMapContext
) -> tuple[Pose, str]:
    forward, _ = build_local_transformers(context.origin)
    local_points = project_path(path_points, forward, context.elevation_m)
    yaw = heading_yaw(local_points, 0) if len(local_points) >= 2 else 0.0
    heading_source = "path_points" if len(local_points) >= 2 else "default_zero_yaw"
    return build_route_pose(local_points[0], yaw), heading_source


def sync_origin_from_payload(_payload: dict[str, Any]) -> dict[str, Any]:
    raise MapGenerationError(
        "Fixposition FP_ENU0 cannot be changed by the EZ21 INS origin API; "
        "configure and survey the sensor origin separately."
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parameter_value_to_python(value: Any) -> Any:
    value_type = int(value.type)
    if value_type == ParameterType.PARAMETER_BOOL:
        return bool(value.bool_value)
    if value_type == ParameterType.PARAMETER_DOUBLE:
        return float(value.double_value)
    if value_type == ParameterType.PARAMETER_INTEGER:
        return int(value.integer_value)
    if value_type == ParameterType.PARAMETER_STRING:
        return str(value.string_value)
    return None


def finite_float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def quaternion_to_euler_xyz(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def resolve_vehicle_state_runtime_map_context() -> RuntimeMapContext:
    try:
        return load_runtime_map_context({})
    except MapGenerationError as original_error:
        active_origin = load_active_map_origin()
        if active_origin is None:
            raise original_error
        return RuntimeMapContext(
            output_dir=(DEFAULT_OUTPUT_ROOT / DEFAULT_MAP_NAME).resolve(),
            projector_type="Local",
            origin=active_origin,
            elevation_m=float(active_origin.altitude),
            origin_source="map_generation_request.json:origin",
        )


@lru_cache(maxsize=8)
def build_local_transformers(origin: GeoPoint) -> tuple[Transformer, Transformer]:
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={origin.latitude} +lon_0={origin.longitude} "
        "+datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    inverse = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    return forward, inverse


def project_path(points: list[GeoPoint], forward: Transformer, elevation_m: float) -> list[LocalPoint]:
    local_points: list[LocalPoint] = []
    for point in points:
        x, y = forward.transform(point.longitude, point.latitude)
        z = point.altitude if point.altitude is not None else elevation_m
        local_points.append(LocalPoint(x=x, y=y, z=z))
    return local_points


def convert_local_points_to_geo(points: list[LocalPoint], inverse: Transformer) -> list[GeoPoint]:
    geo_points: list[GeoPoint] = []
    for point in points:
        longitude, latitude = inverse.transform(point.x, point.y)
        geo_points.append(GeoPoint(latitude=latitude, longitude=longitude, altitude=point.z))
    return geo_points


def heading_yaw(points: list[LocalPoint], index: int) -> float:
    if len(points) < 2:
        raise MapGenerationError("at least two local points are required to build route poses")

    current = points[index]
    previous = points[index - 1] if index > 0 else current
    following = points[index + 1] if index + 1 < len(points) else current

    dx = following.x - previous.x
    dy = following.y - previous.y
    if math.hypot(dx, dy) < 1e-6 and index > 0:
        dx = current.x - previous.x
        dy = current.y - previous.y
    if math.hypot(dx, dy) < 1e-6 and index + 1 < len(points):
        dx = following.x - current.x
        dy = following.y - current.y
    if math.hypot(dx, dy) < 1e-6:
        raise MapGenerationError("route points are too close to determine a heading")

    return math.atan2(dy, dx)


def build_route_pose(local_point: LocalPoint, yaw: float) -> Pose:
    pose = Pose()
    pose.position.x = float(local_point.x)
    pose.position.y = float(local_point.y)
    pose.position.z = float(local_point.z)
    pose.orientation.z = math.sin(yaw / 2.0)
    pose.orientation.w = math.cos(yaw / 2.0)
    return pose


def build_route_poses(path_points: list[GeoPoint], context: RuntimeMapContext) -> list[Pose]:
    forward, _ = build_local_transformers(context.origin)
    local_points = project_path(path_points, forward, context.elevation_m)
    return [
        build_route_pose(local_point, heading_yaw(local_points, index))
        for index, local_point in enumerate(local_points)
    ]


def move_generated_map_endpoint_goal_inside_lane(
    route_poses: list[Pose], context: RuntimeMapContext
) -> float:
    snapshot_path = context.output_dir / "map_generation_request.json"
    if not snapshot_path.exists():
        return 0.0

    snapshot = load_payload(snapshot_path)
    if snapshot.get("origin_source") != f"{FIXPOSITION_STATIC_TF_TOPIC}:FP_ECEF->FP_ENU0":
        return 0.0
    if parse_geo_point_mapping(snapshot.get("origin"), allow_missing=True) != context.origin:
        return 0.0

    map_points = parse_path_points(snapshot.get("payload", {}), context.elevation_m)
    forward, _ = build_local_transformers(context.origin)
    previous, endpoint = project_path(map_points[-2:], forward, context.elevation_m)
    goal = route_poses[-1].position
    dx, dy = endpoint.x - previous.x, endpoint.y - previous.y
    length = math.hypot(dx, dy)
    backoff_m = 1.0
    if length < 2.0 * backoff_m:
        raise MapGenerationError(
            "generated lanelet is too short to place the goal inside its boundary"
        )

    unit_x, unit_y = dx / length, dy / length
    from_endpoint_x = goal.x - endpoint.x
    from_endpoint_y = goal.y - endpoint.y
    distance_to_end_m = -(from_endpoint_x * unit_x + from_endpoint_y * unit_y)
    lateral_error_m = abs(from_endpoint_x * unit_y - from_endpoint_y * unit_x)
    if not (-0.1 <= distance_to_end_m < backoff_m and lateral_error_m <= 0.25):
        return 0.0

    adjustment_m = backoff_m - distance_to_end_m
    route_poses[-1] = build_route_pose(
        LocalPoint(
            x=goal.x - adjustment_m * unit_x,
            y=goal.y - adjustment_m * unit_y,
            z=goal.z,
        ),
        math.atan2(dy, dx),
    )
    return adjustment_m


def distance_xy(start: LocalPoint, end: LocalPoint) -> float:
    return math.hypot(end.x - start.x, end.y - start.y)


def interpolate_local_point(start: LocalPoint, end: LocalPoint, ratio: float) -> LocalPoint:
    return LocalPoint(
        x=start.x + (end.x - start.x) * ratio,
        y=start.y + (end.y - start.y) * ratio,
        z=start.z + (end.z - start.z) * ratio,
    )


def append_point_if_far(points: list[LocalPoint], point: LocalPoint, min_distance_m: float = 1e-4) -> None:
    if not points or distance_xy(points[-1], point) > min_distance_m or abs(points[-1].z - point.z) > 1e-4:
        points.append(point)


def smooth_centerline_with_turn_radius(
    points: list[LocalPoint], min_turn_radius_m: float, arc_step_m: float
) -> list[LocalPoint]:
    if len(points) <= 2:
        return points

    if min_turn_radius_m <= 0.0:
        raise MapGenerationError("min_turn_radius_m must be > 0")
    if arc_step_m <= 0.0:
        raise MapGenerationError("turn arc step must be > 0")

    corner_specs: list[dict[str, Any] | None] = [None] * len(points)
    angle_epsilon = 1e-3

    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        current = points[index]
        following = points[index + 1]

        in_dx = current.x - previous.x
        in_dy = current.y - previous.y
        out_dx = following.x - current.x
        out_dy = following.y - current.y
        in_length = math.hypot(in_dx, in_dy)
        out_length = math.hypot(out_dx, out_dy)

        if in_length < 1e-6 or out_length < 1e-6:
            raise MapGenerationError(f"path points near point #{index + 1} are too close together")

        in_unit_x = in_dx / in_length
        in_unit_y = in_dy / in_length
        out_unit_x = out_dx / out_length
        out_unit_y = out_dy / out_length

        dot = clamp(in_unit_x * out_unit_x + in_unit_y * out_unit_y, -1.0, 1.0)
        turn_angle = math.acos(dot)
        cross = in_unit_x * out_unit_y - in_unit_y * out_unit_x

        if turn_angle < angle_epsilon or abs(cross) < 1e-6:
            continue

        if math.pi - turn_angle < angle_epsilon:
            raise MapGenerationError(
                f"path contains a near U-turn at point #{index + 1}, which is not supported"
            )

        tangent_distance = min_turn_radius_m * math.tan(turn_angle / 2.0)
        corner_specs[index] = {
            "turn_angle": turn_angle,
            "turn_left": cross > 0.0,
            "in_length": in_length,
            "out_length": out_length,
            "in_unit_x": in_unit_x,
            "in_unit_y": in_unit_y,
            "out_unit_x": out_unit_x,
            "out_unit_y": out_unit_y,
            "tangent_distance": tangent_distance,
        }

    for segment_index in range(len(points) - 1):
        segment_length = distance_xy(points[segment_index], points[segment_index + 1])
        start_trim = 0.0
        end_trim = 0.0

        if 0 < segment_index < len(points) - 1 and corner_specs[segment_index] is not None:
            start_trim = float(corner_specs[segment_index]["tangent_distance"])
        if 0 < segment_index + 1 < len(points) - 1 and corner_specs[segment_index + 1] is not None:
            end_trim = float(corner_specs[segment_index + 1]["tangent_distance"])

        if start_trim + end_trim >= segment_length - 1e-6:
            raise MapGenerationError(
                "selected path is too sharp or waypoints are too close to satisfy "
                f"the minimum turning radius of {min_turn_radius_m:.2f} m near "
                f"segment #{segment_index + 1}; please spread the points around the corner"
            )

    smoothed_points: list[LocalPoint] = [points[0]]

    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        current = points[index]
        following = points[index + 1]
        corner_spec = corner_specs[index]

        if corner_spec is None:
            append_point_if_far(smoothed_points, current)
            continue

        in_length = float(corner_spec["in_length"])
        out_length = float(corner_spec["out_length"])
        tangent_distance = float(corner_spec["tangent_distance"])
        turn_left = bool(corner_spec["turn_left"])
        in_unit_x = float(corner_spec["in_unit_x"])
        in_unit_y = float(corner_spec["in_unit_y"])
        out_unit_x = float(corner_spec["out_unit_x"])
        out_unit_y = float(corner_spec["out_unit_y"])

        tangent_in_ratio = (in_length - tangent_distance) / in_length
        tangent_out_ratio = tangent_distance / out_length
        tangent_in = interpolate_local_point(previous, current, tangent_in_ratio)
        tangent_out = interpolate_local_point(current, following, tangent_out_ratio)

        if turn_left:
            normal_x = -in_unit_y
            normal_y = in_unit_x
        else:
            normal_x = in_unit_y
            normal_y = -in_unit_x

        center_x = tangent_in.x + normal_x * min_turn_radius_m
        center_y = tangent_in.y + normal_y * min_turn_radius_m
        start_angle = math.atan2(tangent_in.y - center_y, tangent_in.x - center_x)
        end_angle = math.atan2(tangent_out.y - center_y, tangent_out.x - center_x)

        if turn_left:
            if end_angle <= start_angle:
                end_angle += 2.0 * math.pi
        else:
            if end_angle >= start_angle:
                end_angle -= 2.0 * math.pi

        delta_angle = end_angle - start_angle
        arc_length = abs(delta_angle) * min_turn_radius_m
        arc_segments = max(1, math.ceil(arc_length / arc_step_m))

        append_point_if_far(smoothed_points, tangent_in)
        for step_index in range(1, arc_segments):
            ratio = step_index / arc_segments
            angle = start_angle + delta_angle * ratio
            z = tangent_in.z + (tangent_out.z - tangent_in.z) * ratio
            append_point_if_far(
                smoothed_points,
                LocalPoint(
                    x=center_x + math.cos(angle) * min_turn_radius_m,
                    y=center_y + math.sin(angle) * min_turn_radius_m,
                    z=z,
                ),
            )
        append_point_if_far(smoothed_points, tangent_out)

    append_point_if_far(smoothed_points, points[-1])
    return smoothed_points


def direction_vector(points: list[LocalPoint], index: int) -> tuple[float, float]:
    if len(points) < 2:
        raise MapGenerationError("at least two local points are required")

    previous = points[index - 1] if index > 0 else points[index]
    current = points[index]
    following = points[index + 1] if index + 1 < len(points) else points[index]

    dx = following.x - previous.x
    dy = following.y - previous.y

    if math.hypot(dx, dy) < 1e-6:
        if index > 0:
            dx = current.x - previous.x
            dy = current.y - previous.y
        if math.hypot(dx, dy) < 1e-6 and index + 1 < len(points):
            dx = following.x - current.x
            dy = following.y - current.y

    length = math.hypot(dx, dy)
    if length < 1e-6:
        raise MapGenerationError("path points are too close to build a lane")

    return dx / length, dy / length


def extend_centerline_start_for_vehicle(
    centerline: list[LocalPoint], lane_width_m: float, vehicle: dict[str, Any]
) -> tuple[list[LocalPoint], float, float]:
    if (
        not vehicle.get("received")
        or vehicle.get("frame_id") != "map"
        or vehicle.get("age_ms") is None
        or vehicle["age_ms"] > 2000
    ):
        raise MapGenerationError("fresh vehicle localization in the map frame is required")

    position = vehicle.get("pose", {}).get("position", {})
    orientation = vehicle.get("pose", {}).get("orientation", {})
    ego_x = finite_float_or_none(position.get("x"))
    ego_y = finite_float_or_none(position.get("y"))
    ego_yaw = finite_float_or_none(orientation.get("yaw"))
    if ego_x is None or ego_y is None or ego_yaw is None:
        raise MapGenerationError("vehicle position and heading are required to align the lane start")

    if len(centerline) < 2:
        raise MapGenerationError("at least two map centerline points are required")
    along = 0.0
    nearest = None
    traveled = 0.0
    for index, (start, end) in enumerate(zip(centerline, centerline[1:])):
        dx, dy = end.x - start.x, end.y - start.y
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        raw_ratio = ((ego_x - start.x) * dx + (ego_y - start.y) * dy) / (length * length)
        ratio = min(1.0, max(0.0, raw_ratio))
        nearest_x = start.x + ratio * dx
        nearest_y = start.y + ratio * dy
        distance = math.hypot(ego_x - nearest_x, ego_y - nearest_y)
        if index == 0 and raw_ratio < 0.0:
            distance = abs((ego_x - start.x) * dy - (ego_y - start.y) * dx) / length
        if nearest is None or distance < nearest[0]:
            path_distance = traveled + ratio * length
            if index == 0 and raw_ratio < 0.0:
                path_distance = raw_ratio * length
            nearest = (distance, path_distance, math.atan2(dy, dx))
        traveled += length
    if nearest is None:
        raise MapGenerationError("map centerline points are too close to build a lane")
    lateral_offset, along, heading = nearest
    if along < -2.0:
        raise MapGenerationError(
            "the first map waypoint is more than 2 m ahead of the vehicle; "
            "place it behind the vehicle before refreshing the map"
        )

    lateral_limit = lane_width_m / 2.0 - DEFAULT_VEHICLE_WIDTH_M / 2.0 - 0.1
    heading_error = abs(math.atan2(math.sin(ego_yaw - heading), math.cos(ego_yaw - heading)))
    if lateral_offset > lateral_limit or heading_error > math.radians(30.0):
        raise MapGenerationError(
            "vehicle is not aligned with the first map segment; "
            "place the first waypoint behind the vehicle and align it with the driving direction"
        )
    extension_m = max(0.0, GENERATED_LANE_START_BACKWARD_MARGIN_M - along)
    if not extension_m:
        return centerline, 0.0, along
    direction_x, direction_y = direction_vector(centerline, 0)
    first = centerline[0]
    extended = list(centerline)
    extended[0] = LocalPoint(
        x=first.x - direction_x * extension_m,
        y=first.y - direction_y * extension_m,
        z=first.z,
    )
    return extended, extension_m, along + extension_m


def build_lane_boundaries(
    centerline: list[LocalPoint], lane_width_m: float
) -> tuple[list[LocalPoint], list[LocalPoint]]:
    half_width = lane_width_m / 2.0
    left_boundary: list[LocalPoint] = []
    right_boundary: list[LocalPoint] = []

    for index, point in enumerate(centerline):
        direction_x, direction_y = direction_vector(centerline, index)
        left_normal_x = -direction_y
        left_normal_y = direction_x

        left_boundary.append(
            LocalPoint(
                x=point.x + left_normal_x * half_width,
                y=point.y + left_normal_y * half_width,
                z=point.z,
            )
        )
        right_boundary.append(
            LocalPoint(
                x=point.x - left_normal_x * half_width,
                y=point.y - left_normal_y * half_width,
                z=point.z,
            )
        )

    return left_boundary, right_boundary


def frange(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise MapGenerationError("sampling step must be > 0")

    if stop < start:
        start, stop = stop, start

    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(current)
        current += step

    if not values or not math.isclose(values[-1], stop, abs_tol=1e-9):
        values.append(stop)

    return values


def build_flat_pcd_points(
    centerline: list[LocalPoint],
    lane_width_m: float,
    longitudinal_step_m: float,
    lateral_step_m: float,
    elevation_m: float,
) -> list[LocalPoint]:
    half_width = lane_width_m / 2.0
    lateral_offsets = frange(-half_width, half_width, lateral_step_m)
    pcd_points: list[LocalPoint] = []

    for segment_index in range(len(centerline) - 1):
        start = centerline[segment_index]
        end = centerline[segment_index + 1]
        dx = end.x - start.x
        dy = end.y - start.y
        dz = end.z - start.z
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue

        direction_x = dx / length
        direction_y = dy / length
        left_normal_x = -direction_y
        left_normal_y = direction_x
        sample_distances = frange(0.0, length, longitudinal_step_m)

        for sample_index, distance in enumerate(sample_distances):
            if segment_index > 0 and sample_index == 0:
                continue

            ratio = distance / length if length > 0.0 else 0.0
            center_x = start.x + dx * ratio
            center_y = start.y + dy * ratio
            center_z = start.z + dz * ratio
            if center_z is None:
                center_z = elevation_m

            for lateral_offset in lateral_offsets:
                pcd_points.append(
                    LocalPoint(
                        x=center_x + left_normal_x * lateral_offset,
                        y=center_y + left_normal_y * lateral_offset,
                        z=center_z if center_z is not None else elevation_m,
                    )
                )

    if not pcd_points:
        raise MapGenerationError("failed to generate a non-empty pointcloud map")

    return pcd_points


def format_number(value: float, digits: int = 6) -> str:
    text = f"{value:.{digits}f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def build_osm_document(
    centerline_geo: list[GeoPoint],
    centerline_local: list[LocalPoint],
    left_boundary_local: list[LocalPoint],
    right_boundary_local: list[LocalPoint],
    inverse: Transformer,
    speed_limit_kph: float,
) -> str:
    from xml.etree.ElementTree import Element
    from xml.etree.ElementTree import SubElement
    from xml.etree.ElementTree import tostring
    from xml.dom import minidom

    next_id = 1

    def allocate_id() -> int:
        nonlocal next_id
        current = next_id
        next_id += 1
        return current

    def add_node(parent: Any, geo_point: GeoPoint, local_point: LocalPoint) -> int:
        node_id = allocate_id()
        node = SubElement(
            parent,
            "node",
            {
                "id": str(node_id),
                "lat": f"{geo_point.latitude:.10f}",
                "lon": f"{geo_point.longitude:.10f}",
            },
        )
        SubElement(node, "tag", {"k": "local_x", "v": format_number(local_point.x, 4)})
        SubElement(node, "tag", {"k": "local_y", "v": format_number(local_point.y, 4)})
        SubElement(node, "tag", {"k": "ele", "v": format_number(local_point.z, 3)})
        return node_id

    root = Element("osm", {"version": "0.6", "generator": "generate_autoware_map.py"})
    SubElement(root, "MetaInfo", {"format_version": "1", "map_version": "1"})

    center_node_ids: list[int] = []
    for geo_point, local_point in zip(centerline_geo, centerline_local):
        center_node_ids.append(add_node(root, geo_point, local_point))

    left_node_ids: list[int] = []
    right_node_ids: list[int] = []

    for local_point in left_boundary_local:
        lon, lat = inverse.transform(local_point.x, local_point.y)
        left_node_ids.append(
            add_node(
                root,
                GeoPoint(latitude=lat, longitude=lon, altitude=local_point.z),
                local_point,
            )
        )

    for local_point in right_boundary_local:
        lon, lat = inverse.transform(local_point.x, local_point.y)
        right_node_ids.append(
            add_node(
                root,
                GeoPoint(latitude=lat, longitude=lon, altitude=local_point.z),
                local_point,
            )
        )

    center_way_id = allocate_id()
    center_way = SubElement(root, "way", {"id": str(center_way_id)})
    for node_id in center_node_ids:
        SubElement(center_way, "nd", {"ref": str(node_id)})

    left_way_id = allocate_id()
    left_way = SubElement(root, "way", {"id": str(left_way_id)})
    for node_id in left_node_ids:
        SubElement(left_way, "nd", {"ref": str(node_id)})
    SubElement(left_way, "tag", {"k": "type", "v": "line_thin"})
    SubElement(left_way, "tag", {"k": "subtype", "v": "solid"})

    right_way_id = allocate_id()
    right_way = SubElement(root, "way", {"id": str(right_way_id)})
    for node_id in right_node_ids:
        SubElement(right_way, "nd", {"ref": str(node_id)})
    SubElement(right_way, "tag", {"k": "type", "v": "line_thin"})
    SubElement(right_way, "tag", {"k": "subtype", "v": "solid"})

    lanelet_relation = SubElement(root, "relation", {"id": str(allocate_id())})
    SubElement(
        lanelet_relation,
        "member",
        {"type": "way", "role": "left", "ref": str(left_way_id)},
    )
    SubElement(
        lanelet_relation,
        "member",
        {"type": "way", "role": "right", "ref": str(right_way_id)},
    )
    SubElement(
        lanelet_relation,
        "member",
        {"type": "way", "role": "centerline", "ref": str(center_way_id)},
    )
    SubElement(lanelet_relation, "tag", {"k": "type", "v": "lanelet"})
    SubElement(lanelet_relation, "tag", {"k": "subtype", "v": "road"})
    SubElement(
        lanelet_relation,
        "tag",
        {"k": "speed_limit", "v": format_number(speed_limit_kph, 2)},
    )
    SubElement(lanelet_relation, "tag", {"k": "location", "v": "urban"})
    SubElement(lanelet_relation, "tag", {"k": "one_way", "v": "yes"})

    xml_bytes = tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_bytes)
    return parsed.toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")


def write_pcd_ascii(path: Path, points: Iterable[LocalPoint]) -> None:
    point_list = list(points)
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {len(point_list)}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {len(point_list)}",
            "DATA ascii",
        ]
    )
    body = "\n".join(
        f"{point.x:.6f} {point.y:.6f} {point.z:.6f}" for point in point_list
    )
    path.write_text(f"{header}\n{body}\n", encoding="utf-8")


def build_metadata(points: list[LocalPoint]) -> dict[str, Any]:
    min_x = math.floor(min(point.x for point in points))
    min_y = math.floor(min(point.y for point in points))
    max_x = max(point.x for point in points)
    max_y = max(point.y for point in points)
    x_resolution = max(1, math.ceil(max_x - min_x) + 1)
    y_resolution = max(1, math.ceil(max_y - min_y) + 1)

    return {
        "x_resolution": int(x_resolution),
        "y_resolution": int(y_resolution),
        "pointcloud_map.pcd": [int(min_x), int(min_y)],
    }


def resolve_output_dir(payload: dict[str, Any], map_name: str, reload_autoware: bool) -> Path:
    output_dir = payload.get("output_dir")
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return (DEFAULT_OUTPUT_ROOT / map_name).resolve()


def resolve_generation_origin(payload: dict[str, Any]) -> tuple[GeoPoint, str]:
    payload_origin = parse_geo_point_mapping(payload.get("origin"), allow_missing=True)
    if parse_bool(payload, "use_live_fp_enu0_origin", default=False):
        if payload_origin is not None:
            raise MapGenerationError("choose either an explicit origin or the live FP_ENU0 origin")
        return (
            AUTOWARE_RUNTIME_CLIENT.current_fp_enu0_origin(),
            f"{FIXPOSITION_STATIC_TF_TOPIC}:FP_ECEF->FP_ENU0",
        )
    if payload_origin is not None:
        return payload_origin, "payload:origin"

    try:
        runtime_context = load_runtime_map_context(payload, include_launch_map=False)
        return runtime_context.origin, runtime_context.origin_source
    except MapGenerationError as exc:
        raise MapGenerationError(
            "Scout map generation requires a surveyed FP_ENU0 origin or an existing "
            "map_projector_info.yaml aligned with FP_ENU0"
        ) from exc


def estimate_origin_from_anchor(anchor: GeoLocalAnchor, altitude_m: float) -> GeoPoint:
    _, inverse = build_local_transformers(anchor.geo)
    longitude, latitude = inverse.transform(-anchor.local.x, -anchor.local.y)
    return GeoPoint(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude_m,
    )


def project_path_from_anchor(
    points: list[GeoPoint], anchor: GeoLocalAnchor, elevation_m: float
) -> list[LocalPoint]:
    forward, _ = build_local_transformers(anchor.geo)
    relative_points = project_path(points, forward, elevation_m)
    return [
        LocalPoint(
            x=anchor.local.x + point.x,
            y=anchor.local.y + point.y,
            z=point.z,
        )
        for point in relative_points
    ]


def anchor_to_dict(anchor: GeoLocalAnchor | None) -> dict[str, Any] | None:
    if anchor is None:
        return None

    return {
        "source": anchor.source,
        "geo": {
            "latitude": anchor.geo.latitude,
            "longitude": anchor.geo.longitude,
            "altitude": anchor.geo.altitude,
        },
        "local": {
            "x": anchor.local.x,
            "y": anchor.local.y,
            "z": anchor.local.z,
        },
    }


def wait_for_lanelet_map_after(since_ns: int, timeout_s: float = 10.0) -> bool:
    if not rclpy.ok():
        return False

    node = Node("ui_lanelet_reload_probe")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    received = False
    qos = QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )

    def on_map(message: LaneletMapBin) -> None:
        nonlocal received
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        received = received or (bool(message.data) and stamp_ns >= since_ns)

    node.create_subscription(LaneletMapBin, "/map/vector_map", on_map, qos)
    try:
        deadline = time.monotonic() + timeout_s
        while not received and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=max(0.0, min(0.2, deadline - time.monotonic())))
        return received
    finally:
        executor.remove_node(node)
        executor.shutdown()
        node.destroy_node()


def reload_autoware_lanelet_map(output_dir: Path) -> dict[str, Any]:
    lanelet2_map_path = output_dir / "lanelet2_map.osm"
    projector_info_path = output_dir / "map_projector_info.yaml"

    if not AUTOWARE_SETUP_BASH.exists():
        return {
            "status": "error",
            "error": f"Autoware setup script not found: {AUTOWARE_SETUP_BASH}",
        }

    if not lanelet2_map_path.exists():
        return {
            "status": "error",
            "error": f"lanelet2 map file not found: {lanelet2_map_path}",
        }

    if not projector_info_path.exists():
        return {
            "status": "error",
            "error": f"map projector info file not found: {projector_info_path}",
        }

    shell_prefix = f"source {shlex.quote(str(AUTOWARE_SETUP_BASH))}"
    capability_command = (
        shell_prefix + " && ros2 param get "
        + shlex.quote(AUTOWARE_MAP_PROJECTION_LOADER_NODE)
        + " supports_runtime_reload"
    )
    try:
        capability = subprocess.run(
            ["bash", "-c", capability_command],
            capture_output=True,
            text=True,
            timeout=AUTOWARE_MAP_RELOAD_TIMEOUT_S,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"status": "error", "error": f"failed to check map projection loader: {exc}"}
    if capability.returncode != 0 or not re.search(r"\btrue\b", capability.stdout, re.I):
        return {
            "status": "error",
            "error": (
                "running map projection loader does not support hot reload; "
                "install the updated package and restart Autoware once"
            ),
        }

    shell_steps = [shell_prefix]
    shell_steps.append(
        "ros2 param set "
        f"{shlex.quote(AUTOWARE_LANELET_MAP_LOADER_NODE)} lanelet2_map_path "
        f"{shlex.quote(str(lanelet2_map_path))}"
    )
    shell_steps.append(
        "ros2 param set "
        f"{shlex.quote(AUTOWARE_MAP_PROJECTION_LOADER_NODE)} map_projector_info_path "
        f"{shlex.quote(str(projector_info_path))}"
    )
    command = " && ".join(shell_steps)
    reload_started_ns = time.time_ns()

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=AUTOWARE_MAP_RELOAD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "timed out while reloading the Autoware map projector",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"failed to reload the Autoware map projector: {exc}",
        }

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        error_message = stderr or stdout or "unknown map reload failure"
        return {
            "status": "error",
            "error": error_message,
        }

    if not wait_for_lanelet_map_after(reload_started_ns):
        return {
            "status": "error",
            "error": "lanelet map loader did not publish a fresh /map/vector_map after the reload",
        }

    return {
        "status": "ok",
        "message": "Autoware lanelet map and map projector were reloaded",
        "topic": AUTOWARE_MAP_RELOAD_TOPIC,
        "lanelet2_map": str(lanelet2_map_path),
        "map_projector_info": str(projector_info_path),
    }


def reload_autoware_pointcloud_map(output_dir: Path) -> dict[str, Any]:
    pointcloud_map_path = output_dir / "pointcloud_map.pcd"
    pointcloud_map_metadata_path = output_dir / "pointcloud_map_metadata.yaml"

    if not AUTOWARE_SETUP_BASH.exists():
        return {
            "status": "error",
            "error": f"Autoware setup script not found: {AUTOWARE_SETUP_BASH}",
        }

    if not pointcloud_map_path.exists():
        return {
            "status": "error",
            "error": f"pointcloud map file not found: {pointcloud_map_path}",
        }

    if not pointcloud_map_metadata_path.exists():
        return {
            "status": "error",
            "error": f"pointcloud map metadata file not found: {pointcloud_map_metadata_path}",
        }

    prefix = f"source {shlex.quote(str(AUTOWARE_SETUP_BASH))}"

    try:
        list_result = subprocess.run(
            ["bash", "-c", prefix + f" && ros2 component list {shlex.quote(AUTOWARE_MAP_CONTAINER)}"],
            capture_output=True,
            text=True,
            timeout=AUTOWARE_MAP_RELOAD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "timed out while listing the current map container components",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"failed to inspect the current map container: {exc}",
        }

    if list_result.returncode != 0:
        error_message = (
            (list_result.stderr or "").strip()
            or (list_result.stdout or "").strip()
            or "unknown ros2 component list failure"
        )
        return {
            "status": "error",
            "error": error_message,
        }

    pointcloud_component_uid = None
    for line in list_result.stdout.splitlines():
        match = re.match(
            r"\s*(\d+)\s+(/map/pointcloud_map_loader|/pointcloud_map_loader)\s*$",
            line.strip(),
        )
        if match:
            pointcloud_component_uid = match.group(1)
            break

    reload_steps = []
    if pointcloud_component_uid is not None:
        reload_steps.append(
            "ros2 component unload "
            f"{shlex.quote(AUTOWARE_MAP_CONTAINER)} {shlex.quote(pointcloud_component_uid)}"
        )
        reload_steps.append("sleep 1")

    load_arguments = [
        "ros2 component load",
        shlex.quote(AUTOWARE_MAP_CONTAINER),
        shlex.quote(AUTOWARE_POINTCLOUD_MAP_LOADER_PACKAGE),
        shlex.quote(AUTOWARE_POINTCLOUD_MAP_LOADER_PLUGIN),
        "--node-name",
        "pointcloud_map_loader",
        "--node-namespace",
        "/map",
        "--parameter",
        "enable_whole_load:=true",
        "--parameter",
        "enable_downsampled_whole_load:=false",
        "--parameter",
        "enable_partial_load:=true",
        "--parameter",
        "enable_selected_load:=false",
        "--parameter",
        shlex.quote(f"pcd_paths_or_directory:=[{pointcloud_map_path}]"),
        "--parameter",
        shlex.quote(f"pcd_metadata_path:={pointcloud_map_metadata_path}"),
        "--remap-rule",
        "output/pointcloud_map:=/map/pointcloud_map",
        "--remap-rule",
        "output/pointcloud_map_metadata:=/map/pointcloud_map_metadata",
        "--remap-rule",
        "service/get_partial_pcd_map:=/map/get_partial_pointcloud_map",
        "--remap-rule",
        "service/get_differential_pcd_map:=/map/get_differential_pointcloud_map",
        "--remap-rule",
        "service/get_selected_pcd_map:=/map/get_selected_pointcloud_map",
    ]
    reload_steps.append(" ".join(load_arguments))

    reload_command = prefix + " && " + " && ".join(reload_steps)

    try:
        reload_result = subprocess.run(
            ["bash", "-c", reload_command],
            capture_output=True,
            text=True,
            timeout=AUTOWARE_MAP_COMPONENT_RELOAD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "timed out while reloading the pointcloud map loader component",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"failed to reload the pointcloud map loader component: {exc}",
        }

    if reload_result.returncode != 0:
        error_message = (
            (reload_result.stderr or "").strip()
            or (reload_result.stdout or "").strip()
            or "unknown ros2 component reload failure"
        )
        return {
            "status": "error",
            "error": error_message,
            "component_uid": pointcloud_component_uid,
        }

    verify_command = (
        prefix
        + " && ros2 topic info -v "
        + shlex.quote(AUTOWARE_POINTCLOUD_MAP_TOPIC)
    )
    try:
        verify_result = subprocess.run(
            ["bash", "-c", verify_command],
            capture_output=True,
            text=True,
            timeout=AUTOWARE_MAP_RELOAD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "timed out while verifying the refreshed pointcloud map topic",
        }

    if (
        verify_result.returncode != 0
        or "Publisher count: 1" not in (verify_result.stdout or "")
    ):
        error_message = (
            (verify_result.stderr or "").strip()
            or (verify_result.stdout or "").strip()
            or "pointcloud map topic did not report a publisher after the reload"
        )
        return {
            "status": "error",
            "error": error_message,
            "component_uid": pointcloud_component_uid,
        }

    return {
        "status": "ok",
        "message": "Autoware pointcloud map loader was reloaded",
        "component_uid": pointcloud_component_uid,
        "pointcloud_map": str(pointcloud_map_path),
        "pointcloud_map_metadata": str(pointcloud_map_metadata_path),
        "topic": AUTOWARE_POINTCLOUD_MAP_TOPIC,
    }



def universe_pose_is_fresh(message, received_monotonic, now_ns, *, odometry=False):
    if message is None or received_monotonic is None:
        return False
    if not 0.0 <= time.monotonic() - received_monotonic < UNIVERSE_LOCALIZATION_TIMEOUT_S:
        return False
    if str(message.header.frame_id).lstrip('/') != "map":
        return False
    if odometry and str(message.child_frame_id).lstrip('/') != "base_link":
        return False
    stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
    age = (now_ns - stamp_ns) / 1e9
    if stamp_ns <= 0 or not -UNIVERSE_LOCALIZATION_FUTURE_TOLERANCE_S <= age < UNIVERSE_LOCALIZATION_TIMEOUT_S:
        return False
    pose = message.pose.pose
    values = [pose.position.x, pose.position.y, pose.position.z,
              pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    if not all(math.isfinite(v) for v in values):
        return False
    if abs(sum(v * v for v in values[3:]) - 1.0) > 0.02:
        return False
    if odometry:
        twist = message.twist.twist
        if not all(math.isfinite(v) for v in [twist.linear.x, twist.linear.y, twist.linear.z,
                                              twist.angular.x, twist.angular.y, twist.angular.z]):
            return False
    return True


@lru_cache(maxsize=16)
def _pcd_point_count(path_string, modified_ns, file_size):
    # The cache key tracks file changes; a filename alone must not validate replaced contents.
    del modified_ns
    path = Path(path_string)
    header = {}
    with path.open('rb') as stream:
        for _ in range(64):
            line = stream.readline(8192)
            if not line or len(line) >= 8192:
                raise MapGenerationError("invalid PCD header")
            text = line.decode('ascii').strip()
            if text and not text.startswith('#'):
                key, *values = text.split()
                header[key.upper()] = values
            if text.startswith('DATA '):
                offset = stream.tell()
                break
        else:
            raise MapGenerationError("PCD DATA header not found")
    fields = header.get('FIELDS', [])
    if not {'x', 'y', 'z'}.issubset(fields):
        raise MapGenerationError("NDT pointcloud map must contain x/y/z fields")
    count = int(header['POINTS'][0])
    if count < UNIVERSE_MIN_MAP_POINTS:
        raise MapGenerationError("NDT requires a real 3D pointcloud map; sparse UI placeholder maps are not supported")
    if int(header['WIDTH'][0]) * int(header['HEIGHT'][0]) != count:
        raise MapGenerationError("PCD point count disagrees with dimensions")
    mode = header['DATA'][0]
    if mode == 'binary':
        sizes = [int(v) for v in header['SIZE']]
        counts = [int(v) for v in header.get('COUNT', ['1'] * len(fields))]
        if len(sizes) != len(fields) or len(counts) != len(fields):
            raise MapGenerationError("invalid PCD field sizes")
        if file_size - offset < count * sum(a * b for a, b in zip(sizes, counts)):
            raise MapGenerationError("NDT pointcloud map is truncated")
    elif mode == 'ascii':
        with path.open('rb') as stream:
            stream.seek(offset)
            rows = sum(1 for line in stream if line.strip() and not line.startswith(b'#'))
        if rows != count:
            raise MapGenerationError("ASCII NDT pointcloud map is truncated or has a wrong point count")
    elif mode == 'binary_compressed':
        import struct
        with path.open('rb') as stream:
            stream.seek(offset)
            packed = stream.read(8)
        if len(packed) != 8 or file_size - offset < 8 + struct.unpack('<II', packed)[0]:
            raise MapGenerationError("compressed NDT pointcloud map is truncated")
    else:
        raise MapGenerationError("unsupported NDT pointcloud map encoding")
    return count


def universe_fixed_map_status():
    active = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
    if not active:
        raise MapGenerationError("Universe UI requires the fixed map directory used at launch")
    directory = Path(active).expanduser().resolve()
    filename = os.environ.get("AUTOWARE_UI_POINTCLOUD_MAP_FILE", "pointcloud_map.pcd")
    relative = Path(filename)
    path = (directory / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(directory) or not path.is_file():
        raise MapGenerationError("Universe pointcloud map is missing or outside the launched map directory")
    stat = path.stat()
    return {"path": str(path), "directory": str(directory),
            "points": _pcd_point_count(str(path), stat.st_mtime_ns, stat.st_size)}


def universe_localization_readiness(init_state, odom, odom_received, ndt, ndt_received, loaded_map, now_ns):
    errors = []
    initialized = init_state == LocalizationInitializationState.INITIALIZED
    kinematic_recent = universe_pose_is_fresh(odom, odom_received, now_ns, odometry=True)
    ndt_recent = universe_pose_is_fresh(ndt, ndt_received, now_ns)
    map_status = None
    try:
        map_status = universe_fixed_map_status()
    except (MapGenerationError, OSError, ValueError, KeyError, UnicodeError) as exc:
        errors.append(str(exc))
    map_loaded = map_status is not None and loaded_map == ("map", map_status["points"])
    if not map_loaded:
        errors.append("the fixed pointcloud map has not been loaded on /map/pointcloud_map")
    if not initialized:
        errors.append("Universe localization is not initialized")
    if not kinematic_recent:
        errors.append("Universe map/base_link odometry is unavailable, invalid, or stale")
    if not ndt_recent:
        errors.append("Universe NDT pose is unavailable, invalid, or stale")
    return {"backend": "universe", "ready": not errors, "initialized": initialized,
            "initialization_state": init_state, "kinematic_recent": kinematic_recent,
            "ndt_recent": ndt_recent, "map_valid": map_status is not None,
            "map_loaded": map_loaded, "map": map_status,
            "error": "; ".join(errors) if errors else None}


def require_active_localization(context, *, map_label):
    if SCOUT_LOCALIZATION_MODE == "fixposition":
        require_live_fp_enu0_alignment(context.origin, map_label=map_label)
        return
    snapshot = AUTOWARE_RUNTIME_CLIENT.localization_snapshot()
    if not snapshot.get("ready"):
        raise MapGenerationError(snapshot.get("error") or "Universe localization is not ready")
    fixed = universe_fixed_map_status()
    if context.output_dir.resolve() != Path(fixed["directory"]):
        raise MapGenerationError("route map does not match the fixed Universe localization map")


def require_live_fp_enu0_alignment(origin: GeoPoint, *, map_label: str) -> None:
    live_origin = AUTOWARE_RUNTIME_CLIENT.current_fp_enu0_origin()
    forward, _ = build_local_transformers(live_origin)
    offset = project_path([origin], forward, origin.altitude)[0]
    if math.hypot(offset.x, offset.y) > 0.05 or abs(origin.altitude - live_origin.altitude) > 0.1:
        raise MapGenerationError(
            f"{map_label} origin differs from the current Fixposition FP_ENU0; "
            "load a map aligned with the live localization frame before routing or driving"
        )


def prepare_runtime_map_switch(new_origin: GeoPoint) -> None:
    active_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
    if not active_map_dir:
        raise MapGenerationError("hot map switch requires Scout UI to be started with the Autoware map_path")

    load_runtime_map_context({})
    require_live_fp_enu0_alignment(new_origin, map_label="new map")

    snapshot = AUTOWARE_RUNTIME_CLIENT.snapshot()
    mode = snapshot["operation_mode"]
    if not mode["received"] or mode["label"] != "STOP" or mode["is_autoware_control_enabled"] or mode["is_in_transition"]:
        raise MapGenerationError("stop the vehicle and disable Autoware control before changing maps")

    vehicle = AUTOWARE_RUNTIME_CLIENT.vehicle_state_snapshot()
    if not vehicle.get("received") or (vehicle.get("age_ms") or 0) > 2000:
        raise MapGenerationError("fresh vehicle localization is required before changing maps")
    linear = vehicle["twist"]["linear"]
    if math.sqrt(sum(float(linear[axis]) ** 2 for axis in ("x", "y", "z"))) > 0.05:
        raise MapGenerationError("the vehicle must be stationary before changing maps")

    route_state = snapshot["route_state"]
    if not route_state["received"]:
        raise MapGenerationError("route state is unavailable; cannot safely change maps")
    if route_state["label"] != "UNSET":
        result = AUTOWARE_RUNTIME_CLIENT.clear_route()
        if result["status"] != "ok":
            raise MapGenerationError(f"failed to clear the old route: {result['clear_route']}")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if AUTOWARE_RUNTIME_CLIENT.snapshot()["route_state"]["label"] == "UNSET":
                break
        else:
            raise MapGenerationError("old route did not reach UNSET; map was not switched")


def switch_autoware_map(output_dir: Path) -> dict[str, Any]:
    if SCOUT_LOCALIZATION_MODE == "universe":
        raise MapGenerationError(
            "Universe uses a fixed surveyed 3D pointcloud/lanelet map; restart with the aligned map "
            "instead of generating or hot-switching a map")
    context = load_runtime_map_context({"runtime_map_dir": str(output_dir)})
    for filename in ("lanelet2_map.osm", "pointcloud_map.pcd", "pointcloud_map_metadata.yaml"):
        if not (output_dir / filename).is_file():
            raise MapGenerationError(f"map file not found: {output_dir / filename}")
    prepare_runtime_map_switch(context.origin)

    global MAP_SWITCH_FAILED
    MAP_SWITCH_FAILED = True
    lanelet_result = reload_autoware_lanelet_map(output_dir)
    if lanelet_result.get("status") != "ok":
        return lanelet_result

    pointcloud_result = reload_autoware_pointcloud_map(output_dir)
    if pointcloud_result.get("status") != "ok":
        return {
            "status": "error",
            "error": (
                "lanelet map reload succeeded, but pointcloud map reload failed: "
                + str(pointcloud_result.get("error", "unknown error"))
            ),
            "lanelet_reload": lanelet_result,
            "pointcloud_reload": pointcloud_result,
        }

    os.environ["AUTOWARE_UI_RUNTIME_MAP_DIR"] = str(output_dir)
    MAP_SWITCH_FAILED = False
    return {
        "status": "ok",
        "message": "Autoware lanelet, projector and pointcloud maps were reloaded",
        "lanelet_reload": lanelet_result,
        "pointcloud_reload": pointcloud_result,
    }


def generated_map_dir(map_name: Any) -> Path:
    if not isinstance(map_name, str) or not map_name or map_name != sanitize_map_name(map_name):
        raise MapGenerationError("select a valid generated map name")
    root = DEFAULT_OUTPUT_ROOT.resolve()
    output_dir = (root / map_name).resolve()
    if output_dir.parent != root or not output_dir.is_dir():
        raise MapGenerationError(f"generated map not found: {map_name}")
    return output_dir


def list_generated_maps() -> dict[str, Any]:
    if not DEFAULT_OUTPUT_ROOT.is_dir():
        return {"status": "ok", "maps": []}
    active_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
    active_path = Path(active_map_dir).resolve() if active_map_dir else None
    try:
        live_origin = AUTOWARE_RUNTIME_CLIENT.current_fp_enu0_origin()
    except MapGenerationError:
        live_origin = None

    maps = []
    for output_dir in DEFAULT_OUTPUT_ROOT.iterdir():
        if not output_dir.is_dir() or output_dir.is_symlink():
            continue
        try:
            context = load_runtime_map_context({"runtime_map_dir": str(output_dir)})
            for filename in ("lanelet2_map.osm", "pointcloud_map.pcd", "pointcloud_map_metadata.yaml"):
                if not (output_dir / filename).is_file():
                    raise MapGenerationError("incomplete map")
        except (MapGenerationError, OSError, ValueError):
            continue
        compatible = False
        if live_origin is not None:
            forward, _ = build_local_transformers(live_origin)
            offset = project_path([context.origin], forward, context.origin.altitude)[0]
            compatible = (
                math.hypot(offset.x, offset.y) <= 0.05
                and abs(context.origin.altitude - live_origin.altitude) <= 0.1
            )
        maps.append({
            "map_name": output_dir.name,
            "output_dir": str(output_dir),
            "active": output_dir.resolve() == active_path,
            "compatible_with_fp_enu0": compatible,
            "modified_at": output_dir.stat().st_mtime,
        })
    maps.sort(key=lambda item: item["modified_at"], reverse=True)
    return {"status": "ok", "maps": maps}


def activate_generated_map_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if SCOUT_LOCALIZATION_MODE == "universe":
        raise MapGenerationError(
            "Universe uses a fixed surveyed 3D pointcloud/lanelet map; restart with the aligned map "
            "instead of generating or hot-switching a map")
    output_dir = generated_map_dir(payload.get("map_name"))
    active_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
    if active_map_dir and output_dir == Path(active_map_dir).resolve():
        context = load_runtime_map_context({"runtime_map_dir": str(output_dir)})
        require_live_fp_enu0_alignment(context.origin, map_label="active map")
        return {
            "status": "ok",
            "map_name": output_dir.name,
            "output_dir": str(output_dir),
            "autoware_reload": {"status": "ok", "message": "map is already active"},
        }
    reload_result = switch_autoware_map(output_dir)
    return {
        "status": reload_result["status"],
        "map_name": output_dir.name,
        "output_dir": str(output_dir),
        "autoware_reload": reload_result,
    }


def generate_map_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if SCOUT_LOCALIZATION_MODE == "universe":
        raise MapGenerationError(
            "Universe uses a fixed surveyed 3D pointcloud/lanelet map; restart with the aligned map "
            "instead of generating or hot-switching a map")
    requested_map_name = sanitize_map_name(str(payload.get("map_name", DEFAULT_MAP_NAME)))
    reload_autoware = parse_bool(payload, "reload_autoware", default=False)
    publish_overlay = parse_bool(payload, "publish_satellite_overlay", default=False)
    update_ins_origin = parse_bool(payload, "update_ins_origin", default=False)
    if update_ins_origin:
        raise MapGenerationError("Fixposition FP_ENU0 cannot be changed by the UI")
    map_name = requested_map_name
    elevation_m = parse_float(payload, "elevation_m", "elevation", default=DEFAULT_ELEVATION_M)
    lane_width_m = parse_float(payload, "lane_width_m", "lane_width", default=DEFAULT_LANE_WIDTH_M)
    speed_limit_kph = parse_float(
        payload,
        "speed_limit_kph",
        "speed_limit",
        default=DEFAULT_SPEED_LIMIT_KPH,
    )
    longitudinal_step_m = parse_float(
        payload,
        "pcd_longitudinal_step_m",
        "longitudinal_step_m",
        default=DEFAULT_PCD_LONGITUDINAL_STEP_M,
    )
    lateral_step_m = parse_float(
        payload,
        "pcd_lateral_step_m",
        "lateral_step_m",
        default=DEFAULT_PCD_LATERAL_STEP_M,
    )
    min_turn_radius_m = parse_float(
        payload,
        "min_turn_radius_m",
        "turn_radius_m",
        default=DEFAULT_MIN_TURN_RADIUS_M,
    )

    input_centerline_geo = parse_path_points(payload, elevation_m)
    map_origin, origin_source = resolve_generation_origin(payload)
    generation_anchor = None
    forward, _ = build_local_transformers(map_origin)
    raw_centerline_local = project_path(input_centerline_geo, forward, elevation_m)

    _, inverse = build_local_transformers(map_origin)
    centerline_local = smooth_centerline_with_turn_radius(
        raw_centerline_local, min_turn_radius_m, DEFAULT_TURN_ARC_STEP_M
    )
    start_extension_m = 0.0
    lane_start_margin_m = None
    if reload_autoware and os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip():
        live_origin = AUTOWARE_RUNTIME_CLIENT.current_fp_enu0_origin()
        live_forward, _ = build_local_transformers(live_origin)
        origin_offset = project_path([map_origin], live_forward, map_origin.altitude)[0]
        if (
            math.hypot(origin_offset.x, origin_offset.y) <= 0.05
            and abs(map_origin.altitude - live_origin.altitude) <= 0.1
        ):
            centerline_local, start_extension_m, lane_start_margin_m = extend_centerline_start_for_vehicle(
                centerline_local, lane_width_m, AUTOWARE_RUNTIME_CLIENT.vehicle_state_snapshot()
            )
    centerline_geo = convert_local_points_to_geo(centerline_local, inverse)
    left_boundary_local, right_boundary_local = build_lane_boundaries(centerline_local, lane_width_m)
    pointcloud_points = build_flat_pcd_points(
        centerline=centerline_local,
        lane_width_m=lane_width_m,
        longitudinal_step_m=longitudinal_step_m,
        lateral_step_m=lateral_step_m,
        elevation_m=elevation_m,
    )

    output_dir = resolve_output_dir(payload, map_name, reload_autoware)
    active_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
    if active_map_dir and output_dir == Path(active_map_dir).expanduser().resolve():
        raise MapGenerationError("cannot overwrite the map currently loaded by Autoware")
    output_dir.mkdir(parents=True, exist_ok=True)

    osm_path = output_dir / "lanelet2_map.osm"
    pcd_path = output_dir / "pointcloud_map.pcd"
    metadata_path = output_dir / "pointcloud_map_metadata.yaml"
    projector_info_path = output_dir / "map_projector_info.yaml"
    request_path = output_dir / "map_generation_request.json"

    osm_path.write_text(
        build_osm_document(
            centerline_geo=centerline_geo,
            centerline_local=centerline_local,
            left_boundary_local=left_boundary_local,
            right_boundary_local=right_boundary_local,
            inverse=inverse,
            speed_limit_kph=speed_limit_kph,
        ),
        encoding="utf-8",
    )
    write_pcd_ascii(pcd_path, pointcloud_points)
    metadata_path.write_text(
        yaml.safe_dump(build_metadata(pointcloud_points), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    projector_info_path.write_text(
        yaml.safe_dump(
            {
                "projector_type": "Local",
                "map_origin": {
                    "latitude": map_origin.latitude,
                    "longitude": map_origin.longitude,
                    "altitude": map_origin.altitude,
                },
            },
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )

    request_snapshot = {
        "requested_map_name": requested_map_name,
        "map_name": map_name,
        "reload_autoware": reload_autoware,
        "input_point_count": len(input_centerline_geo),
        "centerline_point_count": len(centerline_local),
        "start_extension_m": start_extension_m,
        "lane_start_margin_m": lane_start_margin_m,
        "vehicle_width_m": DEFAULT_VEHICLE_WIDTH_M,
        "lane_width_m": lane_width_m,
        "lane_extra_margin_m": DEFAULT_LANE_EXTRA_MARGIN_M,
        "min_turn_radius_m": min_turn_radius_m,
        "speed_limit_kph": speed_limit_kph,
        "elevation_m": elevation_m,
        "pcd_longitudinal_step_m": longitudinal_step_m,
        "pcd_lateral_step_m": lateral_step_m,
        "origin_source": origin_source,
        "origin": {
            "latitude": map_origin.latitude,
            "longitude": map_origin.longitude,
            "altitude": map_origin.altitude,
        },
        "generation_anchor": anchor_to_dict(generation_anchor),
        "point_count": len(centerline_geo),
        "payload": payload,
    }
    request_path.write_text(
        json.dumps(request_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if reload_autoware:
        reload_result = switch_autoware_map(output_dir)
    else:
        reload_result = {
            "status": "skipped",
            "message": "reload_autoware is false, so Autoware was not notified",
        }
    if publish_overlay:
        try:
            overlay_result = publish_satellite_overlay(payload, input_centerline_geo, map_origin)
        except Exception as exc:
            overlay_result = {"status": "error", "error": str(exc)}
    else:
        overlay_result = {
            "status": "skipped",
            "message": "publish_satellite_overlay is false, so RViz background was not updated",
        }

    ins_origin_result = {
        "status": "skipped",
        "message": "Fixposition FP_ENU0 is not modified by the UI",
    }

    return {
        "status": "error" if reload_autoware and reload_result["status"] != "ok" else "ok",
        "map_name": map_name,
        "requested_map_name": requested_map_name,
        "output_dir": str(output_dir),
        "origin_source": origin_source,
        "origin": {
            "latitude": map_origin.latitude,
            "longitude": map_origin.longitude,
            "altitude": map_origin.altitude,
        },
        "generation_anchor": anchor_to_dict(generation_anchor),
        "files": {
            "lanelet2_map": str(osm_path),
            "pointcloud_map": str(pcd_path),
            "pointcloud_map_metadata": str(metadata_path),
            "map_projector_info": str(projector_info_path),
            "request_snapshot": str(request_path),
        },
        "autoware_reload": reload_result,
        "satellite_overlay": overlay_result,
        "ins_origin_update": ins_origin_result,
        "summary": {
            "input_centerline_points": len(input_centerline_geo),
            "centerline_points": len(centerline_geo),
            "pointcloud_points": len(pointcloud_points),
            "vehicle_width_m": DEFAULT_VEHICLE_WIDTH_M,
            "lane_width_m": lane_width_m,
            "min_turn_radius_m": min_turn_radius_m,
        },
    }


def get_autoware_status() -> dict[str, Any]:
    result = AUTOWARE_RUNTIME_CLIENT.snapshot()
    result["ui_actions_enabled"] = SCOUT_UI_ALLOW_VEHICLE_ACTIONS
    active_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
    result["runtime_profile"] = {
        "localization_backend": SCOUT_LOCALIZATION_MODE,
        "runtime_map_dir": str(Path(active_map_dir).expanduser().resolve()) if active_map_dir else "",
        "pointcloud_map_file": os.environ.get("AUTOWARE_UI_POINTCLOUD_MAP_FILE", "pointcloud_map.pcd"),
        "ui_actions_enabled": SCOUT_UI_ALLOW_VEHICLE_ACTIONS,
    }
    result["map_switch_failed"] = MAP_SWITCH_FAILED
    result["localization_backend"] = SCOUT_LOCALIZATION_MODE
    result["localization"] = AUTOWARE_RUNTIME_CLIENT.localization_snapshot()
    result["map_switch_supported"] = SCOUT_LOCALIZATION_MODE == "fixposition"
    try:
        context = load_runtime_map_context({})
        result["runtime_map"] = runtime_map_context_to_dict(context)
    except MapGenerationError as exc:
        result["runtime_map"] = {
            "status": "error",
            "error": str(exc),
        }
    return result


def get_vehicle_state() -> dict[str, Any]:
    return AUTOWARE_RUNTIME_CLIENT.vehicle_state_snapshot()


def set_route_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if MAP_SWITCH_FAILED:
        raise MapGenerationError("map switch failed; route commands are blocked until a successful reload or restart")
    context = load_runtime_map_context(payload)
    active_map_dir = os.environ.get("AUTOWARE_UI_RUNTIME_MAP_DIR", "").strip()
    if active_map_dir and context.output_dir != Path(active_map_dir).expanduser().resolve():
        raise MapGenerationError("route map does not match the map currently loaded by Autoware")
    require_active_localization(context, map_label="active map")
    path_points = parse_path_points(payload, context.elevation_m, min_points=2)
    allow_goal_modification = parse_bool(
        payload, "allow_goal_modification", default=False
    )
    clear_existing = parse_bool(payload, "clear_existing", default=True)
    route_poses = build_route_poses(path_points, context)
    goal_adjustment_m = move_generated_map_endpoint_goal_inside_lane(route_poses, context)
    result = AUTOWARE_RUNTIME_CLIENT.set_route(
        route_poses,
        allow_goal_modification=allow_goal_modification,
        clear_existing=clear_existing,
    )
    result["runtime_map"] = runtime_map_context_to_dict(context)
    result["input_point_count"] = len(path_points)
    result["allow_goal_modification"] = bool(allow_goal_modification)
    result["goal_adjustment_m"] = goal_adjustment_m
    result["route_start_source"] = "current_vehicle_pose"
    return result


def clear_route_from_payload(_payload: dict[str, Any]) -> dict[str, Any]:
    return AUTOWARE_RUNTIME_CLIENT.clear_route()


def change_to_autonomous_from_payload(_payload: dict[str, Any]) -> dict[str, Any]:
    return AUTOWARE_RUNTIME_CLIENT.change_to_autonomous()


def change_to_stop_from_payload(_payload: dict[str, Any]) -> dict[str, Any]:
    return AUTOWARE_RUNTIME_CLIENT.change_to_stop()


def enable_autoware_control_from_payload(_payload: dict[str, Any]) -> dict[str, Any]:
    return AUTOWARE_RUNTIME_CLIENT.enable_autoware_control()


def disable_autoware_control_from_payload(_payload: dict[str, Any]) -> dict[str, Any]:
    return AUTOWARE_RUNTIME_CLIENT.disable_autoware_control()


def run_reset_autoware_once(wait_timeout_s: float) -> dict[str, Any]:
    client = AutowareRuntimeClient()
    return client.reset_for_route_planning(wait_timeout_s=wait_timeout_s)


def reset_autoware_via_subprocess(wait_timeout_s: float) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--reset-autoware",
        str(wait_timeout_s),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=max(wait_timeout_s + 15.0, 30.0),
        check=False,
    )

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    if not stdout:
        raise MapGenerationError(
            "reset subprocess did not return JSON"
            + (f": {stderr}" if stderr else "")
        )

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MapGenerationError(
            "invalid JSON from reset subprocess: " + stdout[-400:]
        ) from exc

    if completed.returncode not in (0, 1):
        raise MapGenerationError(
            result.get("error")
            or stderr
            or f"reset subprocess failed with exit code {completed.returncode}"
        )

    return result


def reset_autoware_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    wait_timeout_raw = payload.get("wait_timeout_s", AUTOWARE_RESET_WAIT_TIMEOUT_S)
    try:
        wait_timeout_s = float(wait_timeout_raw)
    except (TypeError, ValueError) as exc:
        raise MapGenerationError("wait_timeout_s must be a number") from exc
    return reset_autoware_via_subprocess(wait_timeout_s=wait_timeout_s)


class MapRequestHandler(SimpleHTTPRequestHandler):
    server_version = "AutowareMapBootstrap/1.1"
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".json": "application/json; charset=utf-8",
    }

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        request_path = self.path.split("?", 1)[0]
        try:
            if SERVER_SHUTTING_DOWN.is_set() and request_path != "/health":
                self.close_connection = True
                self._send_json(
                    503,
                    {"status": "error", "error": "UI server is shutting down"},
                )
                return
            if request_path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "message": (
                            "POST /generate_map, /activate_map, /publish_satellite_overlay, /update_ins_origin, "
                            "/set_route, /clear_route, /change_to_autonomous, /change_to_stop, "
                            "/enable_autoware_control, /disable_autoware_control, /reset_autoware, or GET /vehicle_state"
                        ),
                        "default_output_root": str(DEFAULT_OUTPUT_ROOT),
                    },
                )
                return
            if request_path == "/autoware_status":
                self._send_json(200, get_autoware_status())
                return
            if request_path == "/generated_maps":
                self._send_json(200, list_generated_maps())
                return
            if request_path == "/vehicle_state":
                self._send_json(200, get_vehicle_state())
                return
            super().do_GET()
        except MapGenerationError as exc:
            self._send_json(400, {"status": "error", "error": str(exc)})
        except Exception as exc:
            self._handle_request_exception(request_path, exc)

    def do_POST(self) -> None:
        request_path = self.path.split("?", 1)[0]
        try:
            if SERVER_SHUTTING_DOWN.is_set():
                self.close_connection = True
                self._send_json(
                    503,
                    {"status": "error", "error": "UI server is shutting down"},
                )
                return
            if request_path not in {
                "/generate_map",
                "/activate_map",
                "/publish_satellite_overlay",
                "/update_ins_origin",
                "/set_route",
                "/clear_route",
                "/change_to_autonomous",
                "/change_to_stop",
                "/enable_autoware_control",
                "/disable_autoware_control",
                "/reset_autoware",
            }:
                self._send_json(404, {"status": "error", "error": "unknown endpoint"})
                return
            if not SCOUT_UI_ALLOW_VEHICLE_ACTIONS and request_path in {
                "/generate_map",
                "/activate_map",
                "/set_route",
                "/change_to_autonomous",
                "/enable_autoware_control",
            }:
                self._send_json(
                    403,
                    {"status": "error", "error": "Scout UI vehicle actions are disabled"},
                )
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = (
                json.loads(raw_body.decode("utf-8"))
                if raw_body.strip()
                else {}
            )
            if request_path == "/generate_map":
                with MAP_SWITCH_LOCK:
                    result = generate_map_from_payload(payload)
            elif request_path == "/activate_map":
                with MAP_SWITCH_LOCK:
                    result = activate_generated_map_from_payload(payload)
            elif request_path == "/update_ins_origin":
                result = sync_origin_from_payload(payload)
            elif request_path == "/publish_satellite_overlay":
                result = publish_satellite_overlay_from_payload(payload)
            elif request_path == "/set_route":
                with MAP_SWITCH_LOCK:
                    result = set_route_from_payload(payload)
            elif request_path == "/clear_route":
                result = clear_route_from_payload(payload)
            elif request_path == "/change_to_autonomous":
                with MAP_SWITCH_LOCK:
                    if MAP_SWITCH_FAILED:
                        raise MapGenerationError("map switch failed; autonomous mode is blocked")
                    require_active_localization(
                        load_runtime_map_context({}), map_label="active map"
                    )
                    result = change_to_autonomous_from_payload(payload)
            elif request_path == "/change_to_stop":
                result = change_to_stop_from_payload(payload)
            elif request_path == "/enable_autoware_control":
                with MAP_SWITCH_LOCK:
                    if MAP_SWITCH_FAILED:
                        raise MapGenerationError("map switch failed; Autoware control is blocked")
                    require_active_localization(
                        load_runtime_map_context({}), map_label="active map"
                    )
                    result = enable_autoware_control_from_payload(payload)
            elif request_path == "/disable_autoware_control":
                result = disable_autoware_control_from_payload(payload)
            else:
                result = reset_autoware_from_payload(payload)
        except MapGenerationError as exc:
            self._send_json(400, {"status": "error", "error": str(exc)})
            return
        except json.JSONDecodeError as exc:
            self._send_json(400, {"status": "error", "error": f"invalid JSON: {exc}"})
            return
        except Exception as exc:  # pragma: no cover - safety path for manual use
            self._handle_request_exception(request_path, exc)
            return

        self._send_json(200, result)

    def _handle_request_exception(self, request_path: str, exc: Exception) -> None:
        if is_server_shutdown_error(exc):
            self.close_connection = True
            self._send_json(
                503,
                {
                    "status": "error",
                    "error": "ROS 2 context is not available because the UI server is shutting down",
                },
            )
            return
        print(f"Request failed for {request_path}: {exc}", file=sys.stderr, flush=True)
        self._send_json(500, {"status": "error", "error": str(exc)})

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        try:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


class UiThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(host: str, port: int) -> None:
    DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    handler = partial(MapRequestHandler, directory=str(SCRIPT_DIR))
    SERVER_SHUTTING_DOWN.clear()
    try:
        httpd = UiThreadingHTTPServer((host, port), handler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(
                f"Port {port} is already in use on {host}. "
                "Another generate_autoware_map.py server is probably already running.\n"
                f"Reuse the existing UI at http://{host}:{port}/index.html, "
                f"or stop the old process, or start this script with --port {port + 1}.",
                file=sys.stderr,
                flush=True,
            )
            return
        raise
    print(f"Serving UI and map generator on http://{host}:{port}", flush=True)
    print("Open /index.html for runtime route/HMI and map generation", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        SERVER_SHUTTING_DOWN.set()
        print("\nStopping server", flush=True)
    finally:
        SERVER_SHUTTING_DOWN.set()
        httpd.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a minimal Autoware map from web-selected path points."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Generate once from a JSON file instead of running the HTTP server.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output_dir for one-shot generation.",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        help="Override map_name for one-shot generation.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP server bind address.")
    parser.add_argument("--port", type=int, default=8090, help="HTTP server port.")
    parser.add_argument(
        "--reset-autoware",
        type=float,
        help="Run a one-shot Autoware reset in a fresh process and print JSON.",
    )
    args = parser.parse_args()

    if args.reset_autoware is not None:
        result = run_reset_autoware_once(args.reset_autoware)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("status") in {"ok", "partial"} else 1

    if args.input_json:
        payload = load_payload(args.input_json)
        if args.output_dir:
            payload["output_dir"] = str(args.output_dir.resolve())
        if args.map_name:
            payload["map_name"] = args.map_name
        result = generate_map_from_payload(payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
