import json,os
from pathlib import Path
import pytest

@pytest.mark.skipif(os.environ.get('SCOUT_RUN_ISOLATED_TRAVERSABILITY_ROS_TESTS')!='1',reason='Requires an isolated ROS domain')
def test_real_ros_node_constructs_and_rejects_unsafe_session_without_loading_model(tmp_path):
    import rclpy
    from std_msgs.msg import String
    from scout_global_traversability.node import GlobalTraversabilityNode
    assert os.environ.get('ROS_DOMAIN_ID') not in (None,'','0')
    rclpy.init(args=['--ros-args','-p',f'storage_root:={tmp_path}'])
    node=None
    try:
        node=GlobalTraversabilityNode()
        assert node.context.ok()  # The base ROS context property must remain intact.
        assert node.source_context is None and node.predictor is None
        node.on_session(String(data=json.dumps({'coordinate_kind':'local_ndt_map','database':'/tmp/test.sqlite3','run_id':'../../escape','identity_sha256':'a'*64})))
        assert node.requested is None and node.state=='INVALID_SESSION'
        for _ in range(4):rclpy.spin_once(node,timeout_sec=.1)
        assert node.predictor is None
        node.publish_status()
    finally:
        if node is not None:node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
