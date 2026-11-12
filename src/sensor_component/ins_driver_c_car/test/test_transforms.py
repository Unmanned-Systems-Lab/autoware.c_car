"""Physical transform scenarios for the c_car sensor and localization bridge."""
from datetime import datetime
import math
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("rclpy")

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry

from ins_driver_c_car.ins_driver_node import CCarInsDriverNode
from ins_driver_c_car.ins_localization_bridge import InsLocalizationBridgeNode
from ins_driver_c_car.protocol import geodetic_to_ecef


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def diagonal(values):
    covariance = [0.0] * 36
    for i, value in enumerate(values):
        covariance[i*7] = float(value)
    return covariance


def odometry():
    msg = Odometry()
    msg.header.stamp.sec = 100
    msg.header.frame_id = "odom"
    msg.child_frame_id = "ins_link"
    msg.pose.pose.orientation.w = 1.0
    msg.pose.covariance = diagonal([1.0, 4.0, 9.0, 1.0, 4.0, 9.0])
    msg.twist.covariance = diagonal([1.0, 4.0, 9.0, 1.0, 4.0, 9.0])
    return msg


def bridge_transformer(rotation=None, inverse_rotation=None, inverse_q=None, lever=None):
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return SimpleNamespace(
        output_child_frame_id="base_link",
        _rotation_base_to_ins=rotation or identity,
        _rotation_ins_to_base=inverse_rotation or identity,
        _quaternion_ins_to_base=inverse_q or (0.0, 0.0, 0.0, 1.0),
        _translation_ins_to_base=lever or (0.0, 0.0, 0.0),
    )


def test_sensor_mounting_rotation_is_consistent_between_imu_and_odometry():
    """+90 degree sensor-to-output yaw maps sensor +X vectors to output +Y."""
    half = math.sqrt(0.5)
    origin = (30.0, 120.0, 10.0)
    state = SimpleNamespace(
        nav_flag=1, latitude_deg=origin[0], longitude_deg=origin[1], altitude_m=origin[2],
        origin_geodetic=origin, origin_ecef=geodetic_to_ecef(*origin),
        heading_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
        angular_velocity_body=(1.0, 0.0, 0.0), linear_acceleration_body=(0.0, 1.0, 0.0),
        vel_enu=(1.0, 0.0, 0.0), has_imu=lambda: True,
        is_fresh=lambda *args, **kwargs: True,
    )
    node = SimpleNamespace(
        state=state, _mounting=(0.0, 0.0, half, half), heading_convention="east_ccw",
        roll_sign=1.0, pitch_sign=1.0, axes_verified=True, acceleration_units_verified=True,
        _valid_flags={1}, imu_frame_id="ins_link", frame_id="odom", child_frame_id="ins_link",
        orientation_covariance_diag=[1.0] * 3, angular_velocity_covariance_diag=[0.1] * 3,
        linear_acceleration_covariance_diag=[1.0] * 3,
        pose_covariance_diag=[1.0] * 6, twist_covariance_diag=[1.0] * 6,
        imu_publisher=Recorder(), publisher=Recorder(), _published_count=0, _log_file=None,
    )
    node._sensor_quaternion = MethodType(CCarInsDriverNode._sensor_quaternion, node)
    node._output_quaternion = MethodType(CCarInsDriverNode._output_quaternion, node)
    stamp = Time(sec=100)
    CCarInsDriverNode._publish_imu(node, stamp)
    CCarInsDriverNode._publish_odometry(node, SimpleNamespace(stamp=datetime.now()), stamp)
    imu = node.imu_publisher.messages[0]
    odom = node.publisher.messages[0]
    assert (imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z) == pytest.approx((0, 1, 0))
    assert (imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z) == pytest.approx((-1, 0, 0))
    assert (odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z) == pytest.approx((0, 1, 0))
    assert (odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z) == pytest.approx((0, 1, 0))
    # The world orientation is -90 degrees so the same physical vectors remain unchanged in world.
    assert (imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w) == pytest.approx((0, 0, -half, half))
    q = odom.pose.pose.orientation
    assert (q.x, q.y, q.z, q.w) == pytest.approx((0, 0, -half, half))
    assert imu.orientation_covariance[0] > 0


def test_bridge_rotates_vectors_and_anisotropic_covariance_in_same_direction():
    half = math.sqrt(0.5)
    rotation = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    inverse = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    node = bridge_transformer(rotation, inverse, (0.0, 0.0, -half, half))
    msg = odometry()
    msg.twist.twist.linear.x = msg.twist.twist.angular.x = 1.0
    out = InsLocalizationBridgeNode._convert_ins_odom_to_base_link(node, msg)
    assert (out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z) == pytest.approx((0, 1, 0))
    assert (out.twist.twist.angular.x, out.twist.twist.angular.y, out.twist.twist.angular.z) == pytest.approx((0, 1, 0))
    assert [out.twist.covariance[i*7] for i in range(6)] == pytest.approx([4, 1, 9, 4, 1, 9])
    assert list(out.pose.covariance) == pytest.approx(list(msg.pose.covariance))
    assert msg.twist.twist.linear.x == 1.0  # No mutation of the input sample.


def test_bridge_lever_arm_removes_sensor_rotational_velocity_and_propagates_uncertainty():
    """Sensor one metre ahead of a stationary rotating base has sensor vy=omega_z."""
    node = bridge_transformer(lever=(-1.0, 0.0, 0.0))
    msg = odometry()
    msg.pose.pose.position.x, msg.pose.pose.position.y = 10.0, 20.0
    msg.twist.twist.linear.y, msg.twist.twist.angular.z = 2.0, 2.0
    msg.pose.covariance = diagonal([1, 1, 1, 0.25, 0.25, 0.25])
    msg.twist.covariance = diagonal([1, 1, 1, 0.25, 0.25, 0.25])
    out = InsLocalizationBridgeNode._convert_ins_odom_to_base_link(node, msg)
    assert (out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z) == pytest.approx((9, 20, 0))
    assert (out.twist.twist.linear.x, out.twist.twist.linear.y, out.twist.twist.linear.z) == pytest.approx((0, 0, 0))
    assert out.twist.twist.angular.z == pytest.approx(2)
    for covariance in (out.pose.covariance, out.twist.covariance):
        assert covariance[7] == pytest.approx(1.25)
        assert covariance[14] == pytest.approx(1.25)
        assert covariance[1*6+5] == pytest.approx(-0.25)
        assert covariance[5*6+1] == pytest.approx(-0.25)


def test_constant_speed_turn_has_centripetal_acceleration_and_keeps_input_header():
    """Body vx=5 m/s and yaw rate=.2 rad/s implies ay=1 m/s2 even when vx is constant."""
    node = SimpleNamespace(
        min_accel_dt_sec=0.001, max_accel_dt_sec=1.0, output_child_frame_id="base_link",
        _previous_twist_stamp_sec=None, _previous_linear_velocity=None, _previous_angular_velocity=None,
        acceleration_covariance_diag=[25.0, 25.0, 25.0, 1.0, 1.0, 1.0], acceleration_publisher=Recorder(),
    )
    msg = odometry()
    msg.child_frame_id = "base_link"
    msg.twist.twist.linear.x, msg.twist.twist.angular.z = 5.0, 0.2
    InsLocalizationBridgeNode._publish_acceleration(node, msg)
    assert not node.acceleration_publisher.messages  # No derivative from a single sample.
    msg.header.stamp.nanosec = 100_000_000
    InsLocalizationBridgeNode._publish_acceleration(node, msg)
    out = node.acceleration_publisher.messages[0]
    assert (out.accel.accel.linear.x, out.accel.accel.linear.y, out.accel.accel.linear.z) == pytest.approx((0, 1, 0))
    assert (out.accel.accel.angular.x, out.accel.accel.angular.y, out.accel.accel.angular.z) == pytest.approx((0, 0, 0))
    assert out.header.frame_id == "base_link"
    assert msg.header.frame_id == "odom"
    assert all(out.accel.covariance[i*7] > 0 for i in range(6))
