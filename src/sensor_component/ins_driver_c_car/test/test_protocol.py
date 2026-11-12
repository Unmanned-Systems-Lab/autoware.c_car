import math
import struct
from datetime import datetime

import pytest

from ins_driver_c_car.protocol import CAN_EFF_FLAG, CAN_ERR_FLAG, CAN_FRAME_FORMAT, CAN_RTR_FLAG
from ins_driver_c_car.protocol import GPS_WEEK_MILLISECONDS, CanFrame, NavigationState
from ins_driver_c_car.protocol import decode_can_frame, geodetic_to_ecef, geodetic_to_enu
from ins_driver_c_car.protocol import gps_time_to_unix_seconds, quaternion_from_euler
from ins_driver_c_car.protocol import update_navigation_state


def raw_frame(can_id, data, dlc=None):
    return struct.pack(CAN_FRAME_FORMAT, can_id, len(data) if dlc is None else dlc,
                       data.ljust(8, b"\x00"))


def receive(state, can_id, data, t=10.0):
    return update_navigation_state(state, CanFrame(datetime.now(), can_id, data, t))


def fill_solution(state, source="ins", t=10.0, flag=0):
    if source == "ins":
        attitude = struct.pack("<Hhh", 35999, -1234, 789)
        ids = (0x20B, 0x30B, 0x40B)
    else:
        attitude = struct.pack("<Hh", 35999, -1234)
        ids = (0x21B, 0x31B, 0x41B)
    receive(state, 0x10B, attitude, t)
    receive(state, ids[0], struct.pack("<ii", -301234567, -1147654321), t)
    receive(state, ids[1], struct.pack("<iB", -12345, flag), t)
    return receive(state, ids[2], struct.pack("<hhh", 123, -456, 78), t)


def test_classic_standard_frame_keeps_dlc_and_receive_monotonic_time():
    payload = struct.pack("<ii", -301234567, -1147654321)
    frame = decode_can_frame(raw_frame(0x20B, payload))
    assert frame is not None
    assert frame.can_id == 0x20B
    assert frame.data == payload
    assert frame.received_monotonic > 0


@pytest.mark.parametrize("flag", [CAN_EFF_FLAG, CAN_RTR_FLAG, CAN_ERR_FLAG, 0x800])
def test_transport_rejects_extended_remote_error_and_nonstandard_ids(flag):
    assert decode_can_frame(raw_frame(0x20B | flag, b"\x00" * 8)) is None


@pytest.mark.parametrize("raw", [b"", b"\x00" * 15, b"\x00" * 17, b"\x00" * 72])
def test_transport_rejects_truncated_oversized_and_can_fd_buffers(raw):
    assert decode_can_frame(raw) is None


def test_transport_rejects_classic_dlc_above_eight():
    assert decode_can_frame(raw_frame(0x20B, b"\x00" * 8, dlc=9)) is None


@pytest.mark.parametrize("can_id,length", [
    (0x10B, 3), (0x10B, 5), (0x20B, 7), (0x21B, 7),
    (0x30B, 4), (0x31B, 4), (0x40B, 5), (0x41B, 5),
    (0x50B, 7), (0x60B, 7), (0x70B, 4), (0x09B, 5),
])
def test_short_protocol_frames_rejected_in_transport_and_direct_parser(can_id, length):
    data = bytes(length)
    assert decode_can_frame(raw_frame(can_id, data)) is None
    state = NavigationState()
    assert not receive(state, can_id, data)
    assert not state.group_times
    assert not state.ins.group_times
    assert not state.gps.group_times


@pytest.mark.parametrize("padding", [b"", b"\x9a\xbc"])
def test_ins_signed_fields_and_unsigned_high_heading_with_optional_padding(padding):
    state = NavigationState()
    receive(state, 0x10B, struct.pack("<Hhh", 35999, -32768, 32767) + padding)
    receive(state, 0x20B, struct.pack("<ii", -301234567, -1147654321))
    receive(state, 0x30B, struct.pack("<iB", -12345, 255))
    receive(state, 0x40B, struct.pack("<hhh", -32768, 32767, -1) + padding)
    assert state.heading_deg == pytest.approx(359.99)
    assert state.pitch_deg == pytest.approx(-327.68)
    assert state.roll_deg == pytest.approx(327.67)
    assert state.latitude_deg == pytest.approx(-30.1234567)
    assert state.longitude_deg == pytest.approx(-114.7654321)
    assert state.altitude_m == pytest.approx(-12.345)
    assert state.nav_flag == 255
    assert state.vel_enu == pytest.approx((-327.68, 327.67, -0.01))


def test_ins_and_gps_never_overwrite_or_complete_each_others_solution():
    state = NavigationState()
    assert fill_solution(state, "ins")
    original = (state.latitude_deg, state.longitude_deg, state.roll_deg, state.vel_enu)
    receive(state, 0x21B, struct.pack("<ii", 10000000, 20000000))
    receive(state, 0x41B, struct.pack("<hhh", 999, 888, 777))
    receive(state, 0x10B, struct.pack("<Hh", 9000, -500))
    assert (state.latitude_deg, state.longitude_deg, state.roll_deg, state.vel_enu) == original
    assert state.gps.latitude_deg == pytest.approx(1.0)
    assert state.gps.roll_deg is None
    state.select_source("gps")
    assert state.latitude_deg == pytest.approx(1.0)
    assert state.heading_deg == pytest.approx(90.0)
    assert state.roll_deg is None
    assert state.altitude_m is None
    assert not state.has_odometry(now=10.0)
    state.select_source("ins")
    assert state.has_odometry(now=10.0)


def test_gps_attitude_never_inherits_ins_roll_and_requires_explicit_missing_roll_policy():
    state = NavigationState(selected_source="gps")
    receive(state, 0x10B, struct.pack("<Hhh", 1000, 100, 2345))
    assert not fill_solution(state, "gps")
    assert state.roll_deg is None
    assert state.ins.roll_deg == pytest.approx(23.45)
    assert not state.has_pose(now=10.0)
    state.allow_missing_roll = True
    assert state.has_odometry(now=10.0)


def test_gps_fields_cannot_complete_missing_ins_position():
    state = NavigationState()
    receive(state, 0x10B, struct.pack("<Hhh", 1000, 100, 200))
    receive(state, 0x21B, struct.pack("<ii", 10000000, 20000000))
    receive(state, 0x30B, struct.pack("<iB", 1000, 1))
    assert not receive(state, 0x40B, struct.pack("<hhh", 0, 0, 0))
    assert state.latitude_deg is None


def test_freshness_expires_old_fields_despite_new_velocity_and_rejects_future_time():
    state = NavigationState(field_timeout_sec=0.2, max_group_skew_sec=0.1)
    assert fill_solution(state, t=10.0)
    assert state.has_odometry(now=10.2)
    assert not state.has_odometry(now=10.201)
    assert not state.has_odometry(now=9.999)
    assert not receive(state, 0x40B, struct.pack("<hhh", 1, 2, 3), t=11.0)
    assert state.group_times["attitude"] == 10.0


def test_group_alignment_is_required_even_when_each_field_is_recent():
    state = NavigationState(field_timeout_sec=1.0, max_group_skew_sec=0.05)
    assert fill_solution(state, t=10.0)
    assert not receive(state, 0x40B, struct.pack("<hhh", 1, 2, 3), t=10.1)
    assert not state.has_odometry(now=10.1)
    assert state.is_fresh(["position"], now=10.1)
    assert not state.is_fresh([], now=10.1)
    assert not state.is_fresh(["missing_group"], now=10.1)


def test_out_of_order_frames_cannot_replace_a_newer_value_or_timestamp():
    state = NavigationState()
    receive(state, 0x20B, struct.pack("<ii", 10000000, 20000000), t=10.0)
    receive(state, 0x20B, struct.pack("<ii", 30000000, 40000000), t=9.0)
    assert state.latitude_deg == pytest.approx(1.0)
    assert state.group_times["position"] == 10.0


def test_parser_never_chooses_origin_or_guesses_flag_quality():
    state = NavigationState()
    # Only an explicit node policy can interpret Flag and select an origin.
    assert fill_solution(state, flag=0)
    assert state.nav_flag == 0
    assert state.origin_geodetic is None
    state.reset_origin_if_needed()
    assert state.origin_geodetic == pytest.approx((-30.1234567, -114.7654321, -12.345))


def test_out_of_range_geographic_position_blocks_pose_without_clipping_data():
    state = NavigationState()
    fill_solution(state)
    receive(state, 0x20B, struct.pack("<ii", 910000000, 1810000000))
    assert state.latitude_deg == pytest.approx(91.0)
    assert not state.has_pose(now=10.0)


@pytest.mark.parametrize("unit,factor", [("mps2", 1.0), ("g", 9.80665)])
def test_all_acceleration_axes_and_gyro_units_signs_and_freshness(unit, factor):
    state = NavigationState(acceleration_unit=unit)
    receive(state, 0x50B, struct.pack("<ii", 100000, -200000))
    receive(state, 0x60B, struct.pack("<ii", 9000000, -18000000))
    assert not state.has_imu(now=10.0)
    receive(state, 0x70B, struct.pack("<ii", -4500000, 300000))
    assert state.linear_acceleration_body == pytest.approx((factor, -2 * factor, 3 * factor))
    assert state.angular_velocity_body == pytest.approx((math.pi / 2, -math.pi, -math.pi / 4))
    assert state.has_imu(now=10.0)
    assert not state.has_imu(now=10.201)


def test_partial_z_gyro_frame_cannot_make_a_mixed_or_false_fresh_imu():
    state = NavigationState()
    receive(state, 0x50B, bytes(8))
    receive(state, 0x60B, bytes(8))
    receive(state, 0x70B, struct.pack("<i", 100000))
    assert "gyro_z" not in state.group_times
    assert "accel_z" not in state.group_times
    assert not state.has_imu(now=10.0)


def test_gps_time_decoding_is_unsigned_and_has_independent_freshness():
    state = NavigationState()
    receive(state, 0x09B, struct.pack("<HI", 65535, 604799999))
    assert state.gps_week == 65535
    assert state.gps_millisecond == 604799999
    assert state.is_fresh(["gps_time"], now=10.0)
    assert not state.is_fresh(["gps_time"], now=10.201)
    assert not state.has_pose(now=10.0)
    receive(state, 0x09B, struct.pack("<HI", 1, GPS_WEEK_MILLISECONDS), t=10.1)
    assert state.gps_week == 65535
    assert state.group_times["gps_time"] == 10.0


def test_gps_utc_conversion_requires_explicit_leap_and_rollover_policy():
    assert gps_time_to_unix_seconds(0, 0, leap_seconds=0) == 315964800.0
    assert gps_time_to_unix_seconds(1, 1234, leap_seconds=18) == pytest.approx(316569583.234)
    assert (gps_time_to_unix_seconds(1, 0, leap_seconds=18, rollover_weeks=1024)
            - gps_time_to_unix_seconds(1, 0, leap_seconds=18)) == 1024 * 604800
    with pytest.raises(TypeError):
        gps_time_to_unix_seconds(1, 0)


@pytest.mark.parametrize("kwargs", [
    {"week": -1, "millisecond": 0, "leap_seconds": 18},
    {"week": 65536, "millisecond": 0, "leap_seconds": 18},
    {"week": 1, "millisecond": GPS_WEEK_MILLISECONDS, "leap_seconds": 18},
    {"week": 1, "millisecond": 0, "leap_seconds": -1},
    {"week": 1, "millisecond": 0, "leap_seconds": None},
    {"week": 1, "millisecond": 0, "leap_seconds": 18.0},
    {"week": 1, "millisecond": 0, "leap_seconds": 18, "rollover_weeks": -1024},
    {"week": 1, "millisecond": 0, "leap_seconds": 18, "rollover_weeks": 10 ** 100},
])
def test_gps_utc_conversion_rejects_missing_or_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        gps_time_to_unix_seconds(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"selected_source": "automatic"}, {"acceleration_unit": "unknown"},
    {"field_timeout_sec": 0}, {"field_timeout_sec": math.inf},
    {"max_group_skew_sec": -0.1},
])
def test_ambiguous_source_units_and_invalid_timeouts_are_rejected(kwargs):
    with pytest.raises(ValueError):
        NavigationState(**kwargs)


@pytest.mark.parametrize("stamp", [math.nan, math.inf, -1.0])
def test_invalid_monotonic_timestamp_cannot_update_a_solution(stamp):
    state = NavigationState()
    receive(state, 0x20B, struct.pack("<ii", 10000000, 20000000), t=stamp)
    assert state.latitude_deg is None
    assert not state.ins.group_times


def test_gps_time_and_navigation_must_align_for_device_timestamp_selection():
    state = NavigationState(field_timeout_sec=1.0, max_group_skew_sec=0.05)
    assert fill_solution(state, t=10.0)
    receive(state, 0x09B, struct.pack("<HI", 2000, 123456), t=10.1)
    assert state.is_fresh(["gps_time"], now=10.1)
    assert not state.is_fresh(
        ["gps_time", "attitude", "position", "altitude", "velocity"], now=10.1,
    )


def test_geodetic_to_enu_returns_zero_at_origin():
    origin = (-30.1234567, -114.7654321, -12.345)
    assert geodetic_to_enu(*origin, origin, geodetic_to_ecef(*origin)) == pytest.approx((0, 0, 0))


def test_quaternion_from_euler_identity():
    assert quaternion_from_euler(0, 0, 0) == pytest.approx((0, 0, 0, 1))
