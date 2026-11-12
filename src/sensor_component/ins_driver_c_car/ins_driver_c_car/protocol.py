import math
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from typing import Dict
from typing import Iterable
from typing import TextIO
from typing import Tuple


WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

ATTITUDE_FRAME_ID = 0x10B
POSITION_FRAME_IDS = frozenset({0x20B, 0x21B})
ALTITUDE_FRAME_IDS = frozenset({0x30B, 0x31B})
VELOCITY_FRAME_IDS = frozenset({0x40B, 0x41B})
GYRO_FRAME_ID = 0x60B
GYRO_ACCEL_FRAME_ID = 0x70B
ACCEL_FRAME_ID = 0x50B
GPS_TIME_FRAME_ID = 0x09B
INS_POSITION_FRAME_ID = 0x20B
GPS_POSITION_FRAME_ID = 0x21B
INS_ALTITUDE_FRAME_ID = 0x30B
GPS_ALTITUDE_FRAME_ID = 0x31B
INS_VELOCITY_FRAME_ID = 0x40B
GPS_VELOCITY_FRAME_ID = 0x41B
STANDARD_GRAVITY = 9.80665
GPS_WEEK_MILLISECONDS = 604800000
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
NAVIGATION_GROUPS = ("attitude", "position", "altitude", "velocity")
IMU_GROUPS = ("gyro_xy", "gyro_z", "accel_xy", "accel_z")
_SOLUTION_FIELDS = (
    "heading_deg", "pitch_deg", "roll_deg", "latitude_deg", "longitude_deg",
    "altitude_m", "nav_flag", "vel_enu",
)

LOG_FORMAT_CAN = "can"
LOG_FORMAT_ODOM_CSV = "odom_csv"
LOG_FORMAT_ALIASES = {
    "can": LOG_FORMAT_CAN,
    "can_signal": LOG_FORMAT_CAN,
    "raw_can": LOG_FORMAT_CAN,
    "odom": LOG_FORMAT_ODOM_CSV,
    "csv": LOG_FORMAT_ODOM_CSV,
    "odom_csv": LOG_FORMAT_ODOM_CSV,
}

CAN_FRAME_FORMAT = "=IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FORMAT)
CAN_READ_SIZE = 72
CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_ERR_FLAG = 0x20000000
CAN_SFF_MASK = 0x000007FF
CAN_EFF_MASK = 0x1FFFFFFF


@dataclass(frozen=True)
class CanFrame:
    stamp: datetime
    can_id: int
    data: bytes
    received_monotonic: float = field(default_factory=time.monotonic)


@dataclass
class SolutionState:
    """One navigation solution. INS and GPS values must never share a cache."""

    heading_deg: Optional[float] = None
    pitch_deg: Optional[float] = None
    roll_deg: Optional[float] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    nav_flag: Optional[int] = None
    vel_enu: Optional[Tuple[float, float, float]] = None
    group_times: Dict[str, float] = field(default_factory=dict)


@dataclass
class NavigationState(SolutionState):
    """Selected navigation solution plus shared inertial measurements.

    Top-level navigation fields mirror only ``selected_source``. Timestamps are
    local monotonic receive times, not device epochs. A ready state says nothing
    about Flag quality: its undocumented mapping belongs to an explicit policy
    in the publishing node. Origin selection is likewise an explicit node action.
    """

    selected_source: str = "ins"
    field_timeout_sec: float = 0.2
    max_group_skew_sec: float = 0.1
    acceleration_unit: str = "mps2"
    allow_missing_roll: bool = False
    ins: SolutionState = field(default_factory=SolutionState)
    gps: SolutionState = field(default_factory=SolutionState)
    angular_velocity_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration_body: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gps_week: Optional[int] = None
    gps_millisecond: Optional[int] = None
    origin_geodetic: Optional[Tuple[float, float, float]] = None
    origin_ecef: Optional[Tuple[float, float, float]] = None

    def __post_init__(self) -> None:
        self.selected_source = self.selected_source.strip().lower()
        if self.selected_source not in {"ins", "gps"}:
            raise ValueError("selected_source must be 'ins' or 'gps'")
        if not math.isfinite(self.field_timeout_sec) or self.field_timeout_sec <= 0:
            raise ValueError("field_timeout_sec must be finite and positive")
        if not math.isfinite(self.max_group_skew_sec) or self.max_group_skew_sec < 0:
            raise ValueError("max_group_skew_sec must be finite and nonnegative")
        if self.acceleration_unit not in {"mps2", "g"}:
            raise ValueError("acceleration_unit must be 'mps2' or 'g'")

    def select_source(self, source: str) -> None:
        source = source.strip().lower()
        if source not in {"ins", "gps"}:
            raise ValueError("selected_source must be 'ins' or 'gps'")
        self.selected_source = source
        self._copy_selected_solution()

    def _copy_selected_solution(self) -> None:
        solution = self.ins if self.selected_source == "ins" else self.gps
        for name in _SOLUTION_FIELDS:
            setattr(self, name, getattr(solution, name))
        for group in NAVIGATION_GROUPS:
            self.group_times.pop(group, None)
        self.group_times.update(solution.group_times)

    def is_fresh(
        self, groups: Iterable[str], now: Optional[float] = None,
        source: Optional[str] = None,
    ) -> bool:
        """Require every named group to be recent and mutually aligned.

        With no source, groups may include selected navigation, IMU and GPS time.
        A source explicitly checks only that source's navigation cache. A future
        timestamp and an empty group request both fail closed.
        """
        if source is None:
            times = self.group_times
        elif source in {"ins", "gps"}:
            times = (self.ins if source == "ins" else self.gps).group_times
        else:
            raise ValueError("source must be 'ins' or 'gps'")
        names = tuple(groups)
        if not names or any(name not in times for name in names):
            return False
        current = time.monotonic() if now is None else now
        stamps = [times[name] for name in names]
        if not math.isfinite(current) or any(not math.isfinite(t) for t in stamps):
            return False
        ages = [current - stamp for stamp in stamps]
        return (
            all(-1e-12 <= age <= self.field_timeout_sec + 1e-12 for age in ages)
            and max(stamps) - min(stamps) <= self.max_group_skew_sec + 1e-12
        )

    def has_pose(self, now: Optional[float] = None) -> bool:
        required = (
            self.heading_deg, self.pitch_deg, self.latitude_deg,
            self.longitude_deg, self.altitude_m,
        )
        return (
            all(value is not None and math.isfinite(value) for value in required)
            and (self.roll_deg is not None and math.isfinite(self.roll_deg)
                 or self.roll_deg is None and self.allow_missing_roll)
            and -90.0 <= self.latitude_deg <= 90.0
            and -180.0 <= self.longitude_deg <= 180.0
            and 0.0 <= self.heading_deg < 360.0
            and self.is_fresh(("attitude", "position", "altitude"), now)
        )

    def has_odometry(self, now: Optional[float] = None) -> bool:
        return (
            self.has_pose(now)
            and self.vel_enu is not None
            and all(math.isfinite(value) for value in self.vel_enu)
            and self.is_fresh(NAVIGATION_GROUPS, now)
        )

    def has_imu(self, now: Optional[float] = None) -> bool:
        return self.is_fresh(IMU_GROUPS, now)

    def reset_origin_if_needed(self) -> None:
        if self.origin_geodetic is not None:
            return
        if self.latitude_deg is None or self.longitude_deg is None or self.altitude_m is None:
            return

        self.origin_geodetic = (self.latitude_deg, self.longitude_deg, self.altitude_m)
        self.origin_ecef = geodetic_to_ecef(*self.origin_geodetic)


def decode_can_frame(raw_frame: bytes) -> Optional[CanFrame]:
    # This protocol specifies classic standard data frames, not CAN FD.
    if len(raw_frame) != CAN_FRAME_SIZE:
        return None

    raw_can_id, can_dlc, payload = struct.unpack(CAN_FRAME_FORMAT, raw_frame[:CAN_FRAME_SIZE])
    if raw_can_id & ~CAN_SFF_MASK or can_dlc > 8:
        return None
    data = payload[:can_dlc]
    if not _valid_payload_length(raw_can_id, len(data)):
        return None
    return CanFrame(stamp=datetime.now(), can_id=raw_can_id, data=data)


def _valid_payload_length(can_id: int, length: int) -> bool:
    if length > 8:
        return False
    if can_id == ATTITUDE_FRAME_ID:
        # PDF assigns the same ID to GPS DLC4 and INS DLC6. INS may pad to 8;
        # a padded GPS frame cannot be distinguished without corrected protocol.
        return length == 4 or 6 <= length <= 8
    minimum = {
        INS_POSITION_FRAME_ID: 8, GPS_POSITION_FRAME_ID: 8,
        INS_ALTITUDE_FRAME_ID: 5, GPS_ALTITUDE_FRAME_ID: 5,
        INS_VELOCITY_FRAME_ID: 6, GPS_VELOCITY_FRAME_ID: 6,
        ACCEL_FRAME_ID: 8, GYRO_FRAME_ID: 8, GYRO_ACCEL_FRAME_ID: 8,
        GPS_TIME_FRAME_ID: 6,
    }.get(can_id)
    return minimum is None or length >= minimum


def _mark_group(solution: SolutionState, group: str, stamp: float) -> bool:
    previous = solution.group_times.get(group)
    if previous is not None and stamp < previous:
        return False
    solution.group_times[group] = stamp
    return True


def update_navigation_state(state: NavigationState, frame: CanFrame) -> bool:
    """Decode a valid frame without mixing navigation solutions.

    Only a selected-source velocity frame requests an odometry publication.
    Flag validity is deliberately not inferred here. No origin is set here.
    """
    data = frame.data
    can_id = frame.can_id
    stamp = frame.received_monotonic
    if (can_id < 0 or can_id > CAN_SFF_MASK
            or not _valid_payload_length(can_id, len(data))
            or not math.isfinite(stamp) or stamp < 0):
        return False

    source = None
    group = None
    if can_id == ATTITUDE_FRAME_ID:
        source, group = ("gps" if len(data) == 4 else "ins"), "attitude"
    elif can_id in POSITION_FRAME_IDS:
        source, group = ("ins" if can_id == INS_POSITION_FRAME_ID else "gps"), "position"
    elif can_id in ALTITUDE_FRAME_IDS:
        source, group = ("ins" if can_id == INS_ALTITUDE_FRAME_ID else "gps"), "altitude"
    elif can_id in VELOCITY_FRAME_IDS:
        source, group = ("ins" if can_id == INS_VELOCITY_FRAME_ID else "gps"), "velocity"

    if source is not None:
        solution = state.ins if source == "ins" else state.gps
        if not _mark_group(solution, group, stamp):
            return False
        if group == "attitude":
            solution.heading_deg = decode_u16(data, 0) * 1e-2
            solution.pitch_deg = decode_s16(data, 2) * 1e-2
            solution.roll_deg = decode_s16(data, 4) * 1e-2 if source == "ins" else None
        elif group == "position":
            solution.latitude_deg = decode_s32(data, 0) * 1e-7
            solution.longitude_deg = decode_s32(data, 4) * 1e-7
        elif group == "altitude":
            solution.altitude_m = decode_s32(data, 0) * 1e-3
            solution.nav_flag = data[4]
        elif group == "velocity":
            solution.vel_enu = tuple(decode_s16(data, offset) * 1e-2 for offset in (0, 2, 4))
        if source == state.selected_source:
            state._copy_selected_solution()
            return group == "velocity" and state.has_odometry(stamp)
        return False

    acceleration_factor = 1e-5 * (STANDARD_GRAVITY if state.acceleration_unit == "g" else 1.0)
    if can_id == ACCEL_FRAME_ID and _mark_group(state, "accel_xy", stamp):
        _, _, az = state.linear_acceleration_body
        state.linear_acceleration_body = (
            decode_s32(data, 0) * acceleration_factor,
            decode_s32(data, 4) * acceleration_factor, az,
        )
    elif can_id == GYRO_FRAME_ID and _mark_group(state, "gyro_xy", stamp):
        _, _, wz = state.angular_velocity_body
        state.angular_velocity_body = (
            math.radians(decode_s32(data, 0) * 1e-5),
            math.radians(decode_s32(data, 4) * 1e-5), wz,
        )
    elif can_id == GYRO_ACCEL_FRAME_ID:
        # Both fields share this frame's timestamp; never accept partial DLC4.
        if not _mark_group(state, "gyro_z", stamp):
            return False
        _mark_group(state, "accel_z", stamp)
        wx, wy, _ = state.angular_velocity_body
        ax, ay, _ = state.linear_acceleration_body
        state.angular_velocity_body = (wx, wy, math.radians(decode_s32(data, 0) * 1e-5))
        state.linear_acceleration_body = (ax, ay, decode_s32(data, 4) * acceleration_factor)
    elif can_id == GPS_TIME_FRAME_ID:
        week, millisecond = decode_u16(data, 0), decode_u32(data, 2)
        if millisecond < GPS_WEEK_MILLISECONDS and _mark_group(state, "gps_time", stamp):
            state.gps_week, state.gps_millisecond = week, millisecond
    return False


def gps_time_to_unix_seconds(
    week: int, millisecond: int, *, leap_seconds: int, rollover_weeks: int = 0,
) -> float:
    """Convert full GPS week/TOW with an explicitly supplied UTC policy.

    No leap-second value is guessed. ``rollover_weeks`` explicitly adds weeks
    when the installation has established that the receiver reports a truncated
    week number. This document does not establish that it does.
    """
    for name, value in (("week", week), ("millisecond", millisecond),
                        ("leap_seconds", leap_seconds), ("rollover_weeks", rollover_weeks)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an explicit nonnegative integer")
    if week > 65535 or millisecond >= GPS_WEEK_MILLISECONDS:
        raise ValueError("GPS week or millisecond is outside the protocol range")
    try:
        utc = GPS_EPOCH + timedelta(
            weeks=week + rollover_weeks, milliseconds=millisecond, seconds=-leap_seconds,
        )
    except OverflowError as exc:
        raise ValueError("GPS UTC conversion exceeds the datetime range") from exc
    return utc.timestamp()


def resolve_log_output_path(log_path_text: str, log_name: str, log_name_format: str) -> Path:
    base_path = Path(log_path_text).expanduser()
    if base_path.suffix.lower() in {".txt", ".csv"}:
        return base_path

    if log_name.strip() and log_name.strip().lower() != "auto":
        file_name = log_name.strip()
    else:
        file_name = datetime.now().strftime(log_name_format)
    return base_path / file_name


def open_log_file(path: Path, log_format: str) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new_file = not path.exists() or path.stat().st_size == 0
    handle = path.open("a", encoding="utf-8", buffering=1)
    if is_new_file and log_format == LOG_FORMAT_ODOM_CSV:
        handle.write(
            "receive_time,stamp_sec,stamp_nanosec,frame_id,child_frame_id,"
            "position_x,position_y,position_z,"
            "orientation_x,orientation_y,orientation_z,orientation_w,"
            "linear_x,linear_y,linear_z,"
            "angular_x,angular_y,angular_z\n"
        )
    return handle


def normalize_log_format(value: str) -> str:
    normalized = value.strip().lower()
    log_format = LOG_FORMAT_ALIASES.get(normalized)
    if log_format is None:
        valid_formats = ", ".join(sorted(set(LOG_FORMAT_ALIASES.values())))
        raise ValueError(f"Parameter 'log_format' must be one of: {valid_formats}.")
    return log_format


def format_can_line(frame: CanFrame) -> str:
    can_id_width = 3 if frame.can_id <= CAN_SFF_MASK else 8
    return f"{frame.can_id:0{can_id_width}X}#{frame.data.hex().upper()}\n"


def normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def decode_u16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], byteorder="little", signed=False)


def decode_s16(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], byteorder="little", signed=True)


def decode_s32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], byteorder="little", signed=True)


def decode_u32(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 4], byteorder="little", signed=False)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (radius + alt_m) * cos_lat * cos_lon
    y = (radius + alt_m) * cos_lat * sin_lon
    z = (radius * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return (x, y, z)


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    origin_geodetic: Tuple[float, float, float],
    origin_ecef: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    dx = x - origin_ecef[0]
    dy = y - origin_ecef[1]
    dz = z - origin_ecef[2]

    lat0 = math.radians(origin_geodetic[0])
    lon0 = math.radians(origin_geodetic[1])
    sin_lat0 = math.sin(lat0)
    cos_lat0 = math.cos(lat0)
    sin_lon0 = math.sin(lon0)
    cos_lon0 = math.cos(lon0)

    east = -sin_lon0 * dx + cos_lon0 * dy
    north = -sin_lat0 * cos_lon0 * dx - sin_lat0 * sin_lon0 * dy + cos_lat0 * dz
    up = cos_lat0 * cos_lon0 * dx + cos_lat0 * sin_lon0 * dy + sin_lat0 * dz
    return (east, north, up)


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return (x, y, z, w)


def rotate_world_to_body(
    vector_enu: Tuple[float, float, float], roll: float, pitch: float, yaw: float
) -> Tuple[float, float, float]:
    sr = math.sin(roll)
    cr = math.cos(roll)
    sp = math.sin(pitch)
    cp = math.cos(pitch)
    sy = math.sin(yaw)
    cy = math.cos(yaw)

    rotation = (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )

    vx = (
        rotation[0][0] * vector_enu[0]
        + rotation[1][0] * vector_enu[1]
        + rotation[2][0] * vector_enu[2]
    )
    vy = (
        rotation[0][1] * vector_enu[0]
        + rotation[1][1] * vector_enu[1]
        + rotation[2][1] * vector_enu[2]
    )
    vz = (
        rotation[0][2] * vector_enu[0]
        + rotation[1][2] * vector_enu[1]
        + rotation[2][2] * vector_enu[2]
    )
    return (vx, vy, vz)
