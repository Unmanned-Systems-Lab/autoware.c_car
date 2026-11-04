"""Forward RViz poses through the native localization initializer service.

Only the standalone localization launch should enable this node. Full Autoware
already provides an ADAPI initialization path and must not add a second bridge.
"""

import time

from autoware_localization_msgs.srv import InitializeLocalization
from geometry_msgs.msg import PoseWithCovarianceStamped
import rclpy
from rclpy.node import Node

from .initial_pose_validation import validate_initial_pose


class InitialPoseBridge(Node):
    def __init__(self, **kwargs):
        super().__init__("scout_initial_pose_bridge", **kwargs)
        self.map_frame = self.declare_parameter("map_frame", "map").value
        input_topic = self.declare_parameter("input_topic", "/initialpose").value
        service_name = self.declare_parameter("service_name", "/localization/initialize").value
        self.service_wait_sec = self.declare_parameter("service_wait_sec", 10.0).value
        self.response_timeout_sec = self.declare_parameter("response_timeout_sec", 30.0).value
        for value in (self.service_wait_sec, self.response_timeout_sec):
            if not 0.0 < value < 600.0:
                raise ValueError("initialization timeouts must be between 0 and 600 seconds")
        self.client = self.create_client(InitializeLocalization, service_name)
        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped, input_topic, self.on_pose, 10,
        )
        self.pending_pose = None
        self.future = None
        self.deadline = 0.0
        self.response_timed_out = False
        self.timer = self.create_timer(0.2, self.on_timer)
        self.get_logger().info(
            f"RViz {input_topic} -> native {service_name}, method=AUTO; "
            "native initialization and stop checks remain authoritative",
        )

    def on_pose(self, message):
        if self.pending_pose is not None or self.future is not None:
            self.get_logger().warning("Initialization is already pending; new initial pose rejected")
            return
        p, q = message.pose.pose.position, message.pose.pose.orientation
        try:
            validate_initial_pose(
                message.header.frame_id, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w),
                message.pose.covariance, self.map_frame,
            )
        except (ValueError, OverflowError) as error:
            self.get_logger().error(f"Initial pose rejected: {error}")
            return
        self.pending_pose = message
        self.deadline = time.monotonic() + self.service_wait_sec
        self.on_timer()

    def on_timer(self):
        now = time.monotonic()
        if self.future is not None:
            if now > self.deadline and not self.response_timed_out:
                self.response_timed_out = True
                # Cancelling a client Future would not cancel the server's
                # physical initialization. Stay busy until its result is known.
                self.get_logger().error(
                    "Native initialization response timed out; result is unknown. "
                    "Waiting for the outstanding request; concurrent initialization is blocked.",
                )
            return
        if self.pending_pose is None:
            return
        if now > self.deadline:
            self.pending_pose = None
            self.get_logger().error("Initialization service unavailable; pending pose discarded")
            return
        if not self.client.service_is_ready():
            return
        request = InitializeLocalization.Request()
        request.pose_with_covariance = [self.pending_pose]
        request.method = InitializeLocalization.Request.AUTO
        self.pending_pose = None
        self.response_timed_out = False
        self.deadline = now + self.response_timeout_sec
        try:
            self.future = self.client.call_async(request)
            self.future.add_done_callback(self.on_response)
        except Exception as error:
            self.future = None
            self.get_logger().error(f"Initialization request failed: {error}")

    def on_response(self, future):
        try:
            response = future.result()
            if response is None:
                raise RuntimeError("native initializer returned no response")
            status = response.status
            if status.success:
                self.get_logger().info(
                    f"Native initializer reported success: code={status.code} {status.message}",
                )
            else:
                self.get_logger().error(
                    f"Native initializer rejected request: code={status.code} {status.message}",
                )
        except Exception as error:
            self.get_logger().error(f"Initialization response failed: {error}")
        finally:
            self.future = None
            self.response_timed_out = False


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = InitialPoseBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
