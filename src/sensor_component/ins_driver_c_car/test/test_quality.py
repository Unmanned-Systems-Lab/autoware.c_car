"""Quality gates with synthetic data only; no ROS nodes or CAN sockets start."""
import math
import struct
from datetime import datetime

import pytest

# Windows has no ROS distribution. These tests run in the remote ROS workspace.
pytest.importorskip("rclpy")

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry

from ins_driver_c_car.ins_driver_node import fix_rejection_reason
from ins_driver_c_car.ins_driver_node import navigation_rejection_reason
from ins_driver_c_car.ins_localization_bridge import diagnostic_quality_is_valid
from ins_driver_c_car.ins_localization_bridge import odometry_rejection_reason
from ins_driver_c_car.protocol import CanFrame, NavigationState, update_navigation_state


NOW = 100.0
# This is a synthetic, explicit allowlist, not a manufacturer Flag definition.
ALLOWED_FLAG = 7


def receive(state, can_id, data, t=NOW):
    return update_navigation_state(state, CanFrame(datetime.now(), can_id, data, t))


def valid_state(flag=ALLOWED_FLAG, lat=-30.0, lon=-114.0, omit=()):
    state = NavigationState(field_timeout_sec=0.2, max_group_skew_sec=0.05)
    frames = [
        (0x10B, struct.pack("<Hhh", 35999, -100, 200)),
        (0x20B, struct.pack("<ii", round(lat * 1e7), round(lon * 1e7))),
        (0x30B, struct.pack("<iB", -12345, flag)),
        (0x40B, struct.pack("<hhh", 100, -200, 30)),
        (0x60B, struct.pack("<ii", 100000, -200000)),
        (0x70B, struct.pack("<ii", 300000, 400000)),
    ]
    for can_id, data in frames:
        if can_id not in omit:
            receive(state, can_id, data)
    return state


def test_explicit_flag_fresh_navigation_and_verified_axes_pass():
    state = valid_state()
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW) == ""
    assert navigation_rejection_reason(state, {ALLOWED_FLAG}, True, now=NOW) == ""
    assert state.origin_geodetic is None


@pytest.mark.parametrize("flag,allowlist", [
    (ALLOWED_FLAG, set()), (2, {ALLOWED_FLAG}), (255, {ALLOWED_FLAG}),
    (0, {ALLOWED_FLAG}), (0, {0, ALLOWED_FLAG}),
])
def test_unknown_positive_empty_allowlist_and_zero_flag_are_not_valid_fixes(flag, allowlist):
    state = valid_state(flag=flag)
    assert fix_rejection_reason(state, allowlist, now=NOW)
    assert navigation_rejection_reason(state, allowlist, True, now=NOW)


def test_missing_flag_is_rejected_even_with_other_position_fields_present():
    state = valid_state()
    state.nav_flag = None
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW)


@pytest.mark.parametrize("lat,lon", [(0.0, 0.0), (91.0, 20.0), (20.0, 181.0)])
def test_zero_pair_or_geodetic_range_violation_is_rejected(lat, lon):
    state = valid_state(lat=lat, lon=lon)
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW)


@pytest.mark.parametrize("lat,lon", [(0.0, 30.0), (20.0, 0.0)])
def test_equator_or_prime_meridian_alone_is_a_valid_coordinate(lat, lon):
    state = valid_state(lat=lat, lon=lon)
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW) == ""


@pytest.mark.parametrize("field,value", [
    ("latitude_deg", math.nan), ("longitude_deg", math.inf),
    ("altitude_m", -math.inf), ("altitude_m", None),
])
def test_missing_or_nonfinite_position_and_altitude_are_rejected(field, value):
    state = valid_state()
    setattr(state, field, value)
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW)


def test_axes_verification_is_required_after_a_fix_passes():
    state = valid_state()
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW) == ""
    assert navigation_rejection_reason(state, {ALLOWED_FLAG}, False, now=NOW)


def test_stale_position_cannot_be_rescued_by_a_current_velocity_frame():
    state = valid_state()
    receive(state, 0x40B, struct.pack("<hhh", 100, 200, 300), t=NOW + 0.3)
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW + 0.3)
    assert navigation_rejection_reason(state, {ALLOWED_FLAG}, True, now=NOW + 0.3)


def test_fix_requires_position_and_altitude_to_be_time_aligned():
    state = valid_state()
    receive(state, 0x30B, struct.pack("<iB", 12345, ALLOWED_FLAG), t=NOW + 0.1)
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW + 0.1)


@pytest.mark.parametrize("missing_id", [0x10B, 0x40B, 0x60B, 0x70B])
def test_navigation_requires_attitude_velocity_and_both_gyro_frames(missing_id):
    state = valid_state(omit={missing_id})
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW) == ""
    assert navigation_rejection_reason(state, {ALLOWED_FLAG}, True, now=NOW)


def test_gyro_must_align_with_position_even_when_every_group_is_recent():
    state = valid_state()
    receive(state, 0x60B, struct.pack("<ii", 100000, 200000), t=NOW + 0.1)
    assert fix_rejection_reason(state, {ALLOWED_FLAG}, now=NOW + 0.1) == ""
    assert navigation_rejection_reason(state, {ALLOWED_FLAG}, True, now=NOW + 0.1)


def valid_odometry():
    msg = Odometry()
    msg.header.stamp.sec = int(NOW)
    msg.header.frame_id = "odom"
    msg.child_frame_id = "ins_link"
    msg.pose.pose.orientation.w = 1.0
    msg.pose.pose.position.x = 1.0
    msg.twist.twist.linear.x = 0.5
    for covariance in (msg.pose.covariance, msg.twist.covariance):
        for index in range(6):
            covariance[index * 7] = 0.1
    return msg


def test_bridge_passes_fresh_finite_odometry_with_explicit_covariance():
    assert odometry_rejection_reason(valid_odometry(), NOW, 0.2) == ""


@pytest.mark.parametrize("stamp", [0, 99, 101])
def test_bridge_rejects_absent_stale_or_future_stamp(stamp):
    msg = valid_odometry()
    msg.header.stamp.sec = stamp
    assert odometry_rejection_reason(msg, NOW, 0.2)


@pytest.mark.parametrize("now", [math.nan, math.inf])
def test_bridge_rejects_nonfinite_current_time(now):
    assert odometry_rejection_reason(valid_odometry(), now, 0.2)


@pytest.mark.parametrize("w", [0.0, 2.0])
def test_bridge_rejects_zero_or_unnormalized_quaternion(w):
    msg = valid_odometry()
    msg.pose.pose.orientation.w = w
    assert odometry_rejection_reason(msg, NOW, 0.2)


@pytest.mark.parametrize("path,value", [
    ("pose.pose.position.x", math.nan),
    ("pose.pose.orientation.w", math.inf),
    ("twist.twist.linear.y", math.nan),
    ("twist.twist.angular.z", -math.inf),
])
def test_bridge_rejects_nonfinite_pose_or_twist(path, value):
    msg = valid_odometry()
    parts = path.split(".")
    target = msg
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    assert odometry_rejection_reason(msg, NOW, 0.2)


@pytest.mark.parametrize("which,value", [
    ("pose", 0.0), ("pose", -1.0), ("pose", math.nan),
    ("twist", 0.0), ("twist", -1.0), ("twist", math.inf),
])
def test_bridge_rejects_missing_negative_or_nonfinite_covariance(which, value):
    msg = valid_odometry()
    getattr(msg, which).covariance[0] = value
    assert odometry_rejection_reason(msg, NOW, 0.2)


def quality_message(level=0, valid="true", axes="true", name="gi5651/navigation"):
    msg = DiagnosticArray()
    status = DiagnosticStatus()
    status.name = name
    # Humble maps the diagnostic_msgs ``byte`` level to a one-byte bytes value;
    # use message constants rather than assuming the uint8/int representation.
    status.level = {
        0: DiagnosticStatus.OK, 1: DiagnosticStatus.WARN, 2: DiagnosticStatus.ERROR,
    }[level]
    status.values = [KeyValue(key="valid", value=valid), KeyValue(key="axes_verified", value=axes)]
    msg.status = [status]
    return msg


def test_bridge_accepts_only_explicit_matching_good_navigation_quality():
    assert diagnostic_quality_is_valid(quality_message())


@pytest.mark.parametrize("kwargs", [
    {"level": 1}, {"level": 2}, {"valid": "false"},
    {"axes": "false"}, {"name": "unrelated/navigation"},
])
def test_bridge_rejects_unverified_invalid_or_unrelated_quality(kwargs):
    assert not diagnostic_quality_is_valid(quality_message(**kwargs))


def test_bridge_rejects_missing_quality_or_missing_confirmation_fields():
    assert not diagnostic_quality_is_valid(DiagnosticArray())
    msg = quality_message()
    msg.status[0].values = []
    assert not diagnostic_quality_is_valid(msg)
