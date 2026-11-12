"""Passive GI5651 receiver. Unknown quality and coordinates never create trusted odometry."""
import fcntl
import math
from pathlib import Path
import socket
import struct
import threading

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.time import Time
from sensor_msgs.msg import Imu, NavSatFix, NavSatStatus, TimeReference
import yaml

from ins_driver_c_car.protocol import (
    CAN_READ_SIZE, LOG_FORMAT_CAN, LOG_FORMAT_ODOM_CSV, NavigationState,
    decode_can_frame, format_can_line, geodetic_to_ecef, geodetic_to_enu,
    gps_time_to_unix_seconds, normalize_log_format, open_log_file,
    quaternion_from_euler, resolve_log_output_path, update_navigation_state,
)

SIOCGIFFLAGS = 0x8913
IFF_UP = 0x1
NAVIGATION_STATUS_NAME = "gi5651/navigation"
KNOWN_IDS = {0x10B, 0x20B, 0x30B, 0x40B, 0x50B, 0x60B, 0x70B, 0x09B, 0x21B, 0x31B, 0x41B}


class CanInterfaceNotReadyError(RuntimeError):
    """The requested SocketCAN interface does not exist or is down."""


def fix_rejection_reason(state, valid_nav_flags, now=None):
    """The PDF defines no Flag code table: require an explicit allowlist."""
    if state.nav_flag is None or state.nav_flag == 0 or state.nav_flag not in valid_nav_flags:
        return "navigation Flag is unknown or not explicitly allowed"
    lat, lon, alt = state.latitude_deg, state.longitude_deg, state.altitude_m
    if any(v is None or not math.isfinite(v) for v in (lat, lon, alt)):
        return "position or altitude missing/non-finite"
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return "position outside geodetic range"
    if lat == 0.0 and lon == 0.0:
        return "zero latitude/longitude rejected"
    if not state.is_fresh(["position", "altitude"], now=now):
        return "position/altitude stale or not time-aligned"
    return ""


def navigation_rejection_reason(state, valid_nav_flags, axes_verified, now=None):
    reason = fix_rejection_reason(state, valid_nav_flags, now)
    if reason:
        return reason
    if not axes_verified:
        return "heading convention and sensor axes have not been verified"
    groups = ["attitude", "position", "altitude", "velocity", "gyro_xy", "gyro_z"]
    if not state.is_fresh(groups, now=now) or not state.has_odometry(now=now):
        return "pose/velocity/gyro missing, stale, or not time-aligned"
    return ""


def _multiply_quaternion(a, b):
    x, y, z, w = a
    u, v, s, t = b
    return (w*u+x*t+y*s-z*v, w*v-x*s+y*t+z*u,
            w*s+x*v-y*u+z*t, w*t-x*u-y*v-z*s)


def _rotate_vector(q, vector, inverse=False):
    x, y, z, w = q
    if inverse:
        x, y, z = -x, -y, -z
    result = _multiply_quaternion(_multiply_quaternion((x, y, z, w), (*vector, 0.0)),
                                  (-x, -y, -z, w))
    return result[:3]


def _diagonal(values, size):
    result = [0.0] * (size * size)
    for i, value in enumerate(values):
        result[i * (size + 1)] = float(value)
    return result


class CCarInsDriverNode(Node):
    def __init__(self):
        super().__init__("ins_driver_c_car")
        defaults = {
            "can_interface": "can0", "topic_name": "/sensing/ins/odometry", "frame_id": "odom",
            "child_frame_id": "ins_link", "navigation_source": "ins",
            "field_timeout_sec": 0.2, "max_group_skew_sec": 0.1, "allow_missing_roll": False,
            "acceleration_unit": "mps2", "acceleration_units_verified": False, "axes_verified": False,
            "heading_convention": "north_cw", "roll_sign": 1.0, "pitch_sign": 1.0,
            "euler_order": "zyx", "mounting_rpy_deg": [0.0, 0.0, 0.0],
            "publish_raw_nav_sat_fix": True, "raw_nav_sat_fix_topic": "/sensing/ins/raw_nav_sat_fix",
            "raw_nav_sat_fix_frame_id": "ins_link", "publish_imu": True,
            "imu_topic": "/sensing/imu/imu_raw", "imu_frame_id": "ins_link",
            "quality_topic": "/sensing/ins/quality", "time_source": "receive",
            "gps_utc_leap_seconds": -1, "gps_rollover_weeks": 0,
            "max_device_time_offset_sec": 5.0, "time_reference_topic": "/sensing/ins/time_reference",
            "pose_covariance_diag": [100.0, 100.0, 100.0, 1.0, 1.0, 1.0],
            "twist_covariance_diag": [10.0, 10.0, 10.0, 1.0, 1.0, 1.0],
            "orientation_covariance_diag": [1.0, 1.0, 1.0],
            "angular_velocity_covariance_diag": [0.1, 0.1, 0.1],
            "linear_acceleration_covariance_diag": [1.0, 1.0, 1.0],
            "map_projector_info_path": "", "reference_origin_active": False,
            "reference_origin_latitude_deg": 0.0, "reference_origin_longitude_deg": 0.0,
            "reference_origin_altitude_m": 0.0, "socket_timeout_sec": 0.05,
            "log_enabled": False, "log_path": "~/ins_driver_c_car/logs",
            "log_name": "auto", "log_name_format": "%y%m%d%H%M%S.txt", "log_format": LOG_FORMAT_CAN,
        }
        for name, default in defaults.items():
            self.declare_parameter(name, default)
            setattr(self, name, self.get_parameter(name).value)
        for name in ("valid_nav_flags", "fixed_nav_flags", "float_nav_flags"):
            self.declare_parameter(name, [], ParameterDescriptor(dynamic_typing=True))
            values = self.get_parameter(name).value or []
            if any(not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= 255 for v in values):
                raise ValueError(f"{name} must contain integer Flag values in 0..255")
            setattr(self, name, set(values))
        if self.fixed_nav_flags & self.float_nav_flags:
            raise ValueError("fixed_nav_flags and float_nav_flags must not overlap")
        self._valid_flags = self.valid_nav_flags | self.fixed_nav_flags | self.float_nav_flags
        self._validate_parameters()
        self.log_format = normalize_log_format(self.log_format)
        self._lock = threading.RLock()
        self.state = NavigationState(
            selected_source=self.navigation_source, field_timeout_sec=self.field_timeout_sec,
            max_group_skew_sec=self.max_group_skew_sec, acceleration_unit=self.acceleration_unit,
            allow_missing_roll=self.allow_missing_roll)
        self._mounting = quaternion_from_euler(*(math.radians(v) for v in self.mounting_rpy_deg))
        self._configure_origin()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self.publisher = self.create_publisher(Odometry, self.topic_name, 10)
        self.raw_nav_sat_fix_publisher = (self.create_publisher(NavSatFix, self.raw_nav_sat_fix_topic, 10)
                                         if self.publish_raw_nav_sat_fix else None)
        self.imu_publisher = self.create_publisher(Imu, self.imu_topic, 10) if self.publish_imu else None
        self.quality_publisher = self.create_publisher(DiagnosticArray, self.quality_topic, 10)
        self.time_reference_publisher = self.create_publisher(TimeReference, self.time_reference_topic, 10)
        self._published_count = 0
        self._stop_event = threading.Event()
        self._socket = None
        self._log_file = None
        if self.log_enabled:
            path = resolve_log_output_path(self.log_path, self.log_name, self.log_name_format)
            self._log_file = open_log_file(path, self.log_format)
        self._ensure_can_interface_is_up()
        self._socket = self._open_can_socket()
        self._quality_timer = self.create_timer(0.05, self._quality_tick)
        self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receiver_thread.start()
        self.get_logger().info(f"Passive GI5651 receiver: {self.can_interface}, source={self.navigation_source}")
        if not self._valid_flags or not self.axes_verified:
            self.get_logger().warning("Trusted odometry disabled until Flag mapping and axes are explicitly verified.")
        if not self.acceleration_units_verified:
            self.get_logger().warning("CAN acceleration unit is unverified; IMU acceleration covariance is unknown.")

    def _validate_parameters(self):
        self.navigation_source = str(self.navigation_source).strip().lower()
        if self.navigation_source not in ("ins", "gps"):
            raise ValueError("navigation_source must be ins or gps")
        for name in ("can_interface", "topic_name", "frame_id", "child_frame_id", "imu_frame_id",
                     "quality_topic", "imu_topic", "raw_nav_sat_fix_topic", "time_reference_topic",
                     "raw_nav_sat_fix_frame_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in ("field_timeout_sec", "max_group_skew_sec", "socket_timeout_sec", "max_device_time_offset_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.heading_convention not in ("north_cw", "north_ccw", "east_cw", "east_ccw"):
            raise ValueError("Unsupported heading_convention")
        if self.euler_order != "zyx" or self.roll_sign not in (-1, 1) or self.pitch_sign not in (-1, 1):
            raise ValueError("Only explicit ZYX Euler composition and +/-1 pitch/roll signs are supported")
        if len(self.mounting_rpy_deg) != 3 or not all(math.isfinite(v) for v in self.mounting_rpy_deg):
            raise ValueError("mounting_rpy_deg must contain three finite angles")
        if any(v != 0.0 for v in self.mounting_rpy_deg) and self.imu_frame_id != self.child_frame_id:
            raise ValueError("Nonzero mounting rotation requires imu_frame_id == child_frame_id")
        if self.time_source not in ("receive", "gps"):
            raise ValueError("time_source must be receive or gps")
        if self.time_source == "gps" and self.gps_utc_leap_seconds < 0:
            raise ValueError("GPS time requires an explicit nonnegative gps_utc_leap_seconds")
        if self.gps_rollover_weeks < 0 or self.gps_rollover_weeks % 1024:
            raise ValueError("gps_rollover_weeks must be a nonnegative multiple of 1024")
        for name, size in (("pose_covariance_diag", 6), ("twist_covariance_diag", 6),
                           ("orientation_covariance_diag", 3), ("angular_velocity_covariance_diag", 3),
                           ("linear_acceleration_covariance_diag", 3)):
            values = getattr(self, name)
            if len(values) != size or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{name} requires {size} positive finite variances")

    def _configure_origin(self):
        if self.reference_origin_active:
            self._set_navigation_origin((self.reference_origin_latitude_deg,
                                         self.reference_origin_longitude_deg, self.reference_origin_altitude_m))
            return
        if self.map_projector_info_path:
            data = yaml.safe_load(Path(self.map_projector_info_path).expanduser().read_text()) or {}
            if str(data.get("projector_type", "Local")).lower() != "local":
                raise ValueError("Only a Local map projector is supported; refusing an unrelated map origin")
            origin = data["map_origin"]
            self._set_navigation_origin(tuple(float(origin[k]) for k in ("latitude", "longitude", "altitude")))

    def _set_navigation_origin(self, origin):
        if not all(math.isfinite(v) for v in origin) or not (-90 <= origin[0] <= 90 and -180 <= origin[1] <= 180):
            raise ValueError("Reference origin must contain valid finite geodetic coordinates")
        self.state.origin_geodetic = origin
        self.state.origin_ecef = geodetic_to_ecef(*origin)

    def _on_set_parameters(self, parameters):
        origin_names = {"reference_origin_active", "reference_origin_latitude_deg",
                        "reference_origin_longitude_deg", "reference_origin_altitude_m"}
        changed = {p.name: p.value for p in parameters}
        if any(name not in origin_names for name in changed):
            return SetParametersResult(successful=False, reason="Driver settings require a restart")
        values = {name: changed.get(name, getattr(self, name)) for name in origin_names}
        origin = tuple(values[f"reference_origin_{axis}"] for axis in
                       ("latitude_deg", "longitude_deg", "altitude_m"))
        if values["reference_origin_active"] and (not all(math.isfinite(v) for v in origin)
                or not -90 <= origin[0] <= 90 or not -180 <= origin[1] <= 180):
            return SetParametersResult(successful=False, reason="Invalid reference origin")
        with self._lock:
            for name, value in values.items():
                setattr(self, name, value)
            self.state.origin_geodetic = self.state.origin_ecef = None
            self._configure_origin()
        return SetParametersResult(successful=True)

    def _ensure_can_interface_is_up(self):
        flags = get_interface_flags(self.can_interface)
        if flags is None or not flags & IFF_UP:
            raise CanInterfaceNotReadyError(f"CAN interface {self.can_interface!r} is absent/down; configure it externally")

    def _open_can_socket(self):
        receiver = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        receiver.settimeout(self.socket_timeout_sec)
        receiver.bind((self.can_interface,))
        return receiver

    def _gps_seconds(self):
        if self.gps_utc_leap_seconds < 0 or not self.state.is_fresh(["gps_time"]):
            return None
        try:
            seconds = gps_time_to_unix_seconds(self.state.gps_week, self.state.gps_millisecond,
                                               leap_seconds=self.gps_utc_leap_seconds,
                                               rollover_weeks=self.gps_rollover_weeks)
            # builtin_interfaces/Time.sec is signed int32, regardless of GPS U16 week range.
            return seconds if math.isfinite(seconds) and 0.0 <= seconds < 2147483648.0 else None
        except (TypeError, ValueError, OverflowError):
            return None

    def _message_stamp(self):
        seconds = self._gps_seconds() if self.time_source == "gps" else None
        received = self.get_clock().now()
        if seconds is None or abs(seconds - received.nanoseconds / 1e9) > self.max_device_time_offset_sec:
            return received.to_msg()
        return Time(nanoseconds=int(seconds * 1e9)).to_msg()

    def _navigation_reason(self):
        reason = navigation_rejection_reason(self.state, self._valid_flags, self.axes_verified)
        if reason:
            return reason
        if self.time_source == "gps":
            groups = ["gps_time", "position", "altitude", "attitude", "velocity"]
            if not self.state.is_fresh(groups):
                return "GPS time missing, stale, or not aligned with navigation"
            seconds = self._gps_seconds()
            if seconds is None or abs(seconds - self.get_clock().now().nanoseconds / 1e9) > self.max_device_time_offset_sec:
                return "GPS UTC time inconsistent with ROS clock (check leap seconds/rollover/clock)"
        return ""

    def _publish_quality(self, stamp=None):
        reason = self._navigation_reason()
        status = DiagnosticStatus()
        status.name = NAVIGATION_STATUS_NAME
        status.hardware_id = f"GI5651:{self.can_interface}"
        status.level = DiagnosticStatus.WARN if reason else DiagnosticStatus.OK
        status.message = reason or "fresh navigation with explicit Flag and axis policy"
        flag = self.state.nav_flag
        label = "rtk_fixed" if flag in self.fixed_nav_flags else "rtk_float" if flag in self.float_nav_flags else "fix" if flag in self._valid_flags else "unknown"
        attributes = {"source": self.navigation_source, "nav_flag": str(flag), "quality": label,
                      "valid": str(not reason).lower(), "axes_verified": str(self.axes_verified).lower(),
                      "acceleration_unit": self.acceleration_unit,
                      "acceleration_units_verified": str(self.acceleration_units_verified).lower(),
                      "time_source": self.time_source, "gps_week": str(self.state.gps_week),
                      "gps_millisecond": str(self.state.gps_millisecond)}
        status.values = [KeyValue(key=k, value=v) for k, v in attributes.items()]
        msg = DiagnosticArray()
        # Diagnostics are a host watchdog, never stamped by an untrusted device clock.
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [status]
        self.quality_publisher.publish(msg)
        return not reason

    def _quality_tick(self):
        with self._lock:
            self._publish_quality()

    def _receive_loop(self):
        position_id, altitude_id, velocity_id = ((0x20B, 0x30B, 0x40B) if self.navigation_source == "ins"
                                                else (0x21B, 0x31B, 0x41B))
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                receiver = self._socket
                if receiver is None:
                    return
                raw = receiver.recv(CAN_READ_SIZE)
            except socket.timeout:
                continue
            except OSError as exc:
                if not self._stop_event.is_set():
                    self.get_logger().error(f"CAN receive failed: {exc}")
                return
            frame = decode_can_frame(raw)
            if frame is None:
                continue
            with self._lock:
                if self._log_file is not None and self.log_format == LOG_FORMAT_CAN:
                    self._log_file.write(format_can_line(frame))
                if frame.can_id not in KNOWN_IDS:
                    continue
                try:
                    should_publish = update_navigation_state(self.state, frame)
                    stamp = self._message_stamp()
                    if frame.can_id in (position_id, altitude_id):
                        self._publish_raw_nav_sat_fix(stamp)
                    if frame.can_id == 0x70B:
                        self._publish_imu(stamp)
                    if frame.can_id == 0x09B:
                        self._publish_time_reference()
                    if frame.can_id == velocity_id:
                        valid = self._publish_quality(stamp)
                        if should_publish and valid:
                            self.state.reset_origin_if_needed()
                            self._publish_odometry(frame, stamp)
                except Exception as exc:
                    self.get_logger().error(f"Cannot process GI5651 frame 0x{frame.can_id:X}: {exc}")

    def _publish_raw_nav_sat_fix(self, stamp):
        if self.raw_nav_sat_fix_publisher is None or self.state.latitude_deg is None or self.state.longitude_deg is None:
            return
        msg = NavSatFix()
        msg.header.stamp, msg.header.frame_id = stamp, self.raw_nav_sat_fix_frame_id
        msg.status.service = NavSatStatus.SERVICE_GPS
        msg.status.status = (NavSatStatus.STATUS_NO_FIX if fix_rejection_reason(self.state, self._valid_flags)
                             else NavSatStatus.STATUS_FIX)
        msg.latitude, msg.longitude = self.state.latitude_deg, self.state.longitude_deg
        msg.altitude = self.state.altitude_m if self.state.is_fresh(["altitude"]) else float("nan")
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN
        self.raw_nav_sat_fix_publisher.publish(msg)

    def _sensor_quaternion(self):
        heading = math.radians(self.state.heading_deg)
        yaw = {"north_cw": math.pi/2-heading, "north_ccw": math.pi/2+heading,
               "east_cw": -heading, "east_ccw": heading}[self.heading_convention]
        return quaternion_from_euler(math.radians((self.state.roll_deg or 0.0) * self.roll_sign),
                                     math.radians(self.state.pitch_deg * self.pitch_sign), yaw)

    def _output_quaternion(self):
        x, y, z, w = self._mounting
        return _multiply_quaternion(self._sensor_quaternion(), (-x, -y, -z, w))

    def _publish_imu(self, stamp):
        if self.imu_publisher is None or not self.state.has_imu():
            return
        msg = Imu()
        msg.header.stamp, msg.header.frame_id = stamp, self.imu_frame_id
        attitude_valid = (self.state.heading_deg is not None and 0.0 <= self.state.heading_deg < 360.0
                          and self.state.pitch_deg is not None and math.isfinite(self.state.pitch_deg)
                          and self.state.roll_deg is not None and math.isfinite(self.state.roll_deg))
        if (self.axes_verified and attitude_valid and self.state.is_fresh(["attitude"])
                and not fix_rejection_reason(self.state, self._valid_flags)):
            q = self._output_quaternion()
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = q
            msg.orientation_covariance = _diagonal(self.orientation_covariance_diag, 3)
        else:
            msg.orientation_covariance[0] = -1.0
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = _rotate_vector(
            self._mounting, self.state.angular_velocity_body)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = _rotate_vector(
            self._mounting, self.state.linear_acceleration_body)
        msg.angular_velocity_covariance = _diagonal(self.angular_velocity_covariance_diag, 3)
        if self.acceleration_units_verified:
            msg.linear_acceleration_covariance = _diagonal(self.linear_acceleration_covariance_diag, 3)
        else:
            msg.linear_acceleration_covariance[0] = -1.0
        self.imu_publisher.publish(msg)

    def _publish_time_reference(self):
        seconds = self._gps_seconds()
        if seconds is None:
            return
        msg = TimeReference()
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), self.raw_nav_sat_fix_frame_id
        msg.time_ref = Time(nanoseconds=int(seconds * 1e9)).to_msg()
        msg.source = f"GI5651 GPS UTC; leap_seconds={self.gps_utc_leap_seconds}"
        self.time_reference_publisher.publish(msg)

    def _publish_odometry(self, frame, stamp):
        state = self.state
        position = geodetic_to_enu(state.latitude_deg, state.longitude_deg, state.altitude_m,
                                   state.origin_geodetic, state.origin_ecef)
        world_child = self._output_quaternion()
        linear = _rotate_vector(world_child, state.vel_enu, inverse=True)
        angular = _rotate_vector(self._mounting, state.angular_velocity_body)
        msg = Odometry()
        msg.header.stamp, msg.header.frame_id, msg.child_frame_id = stamp, self.frame_id, self.child_frame_id
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = position
        q = msg.pose.pose.orientation
        q.x, q.y, q.z, q.w = world_child
        msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = linear
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = angular
        # These are configured assumptions, never a fabricated receiver covariance estimate.
        msg.pose.covariance = _diagonal(self.pose_covariance_diag, 6)
        msg.twist.covariance = _diagonal(self.twist_covariance_diag, 6)
        self.publisher.publish(msg)
        self._published_count += 1
        if self._log_file is not None and self.log_format == LOG_FORMAT_ODOM_CSV:
            values = [frame.stamp.isoformat(timespec="milliseconds"), stamp.sec, stamp.nanosec,
                      self.frame_id, self.child_frame_id, *position, *world_child, *linear, *angular]
            self._log_file.write(",".join(str(v) for v in values) + "\n")

    def destroy_node(self):
        self._stop_event.set()
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if hasattr(self, "_receiver_thread"):
            self._receiver_thread.join(timeout=1.0)
        if self._log_file is not None:
            self._log_file.close()
        return super().destroy_node()


def get_interface_flags(interface_name):
    request = struct.pack("16sH", interface_name.encode()[:15], 0)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as ioctl_socket:
            response = fcntl.ioctl(ioctl_socket.fileno(), SIOCGIFFLAGS, request)
    except OSError:
        return None
    return struct.unpack("16sH", response[:18])[1]


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CCarInsDriverNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

