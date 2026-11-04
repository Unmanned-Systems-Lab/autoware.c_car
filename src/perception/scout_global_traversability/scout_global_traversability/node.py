"""Bounded global HT work queue and local/global GridMap visualization."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from ament_index_python.packages import get_package_share_directory
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from std_msgs.msg import String
from autoware_adapi_v1_msgs.msg import LocalizationInitializationState
from .ros_grid import window_message, empty_window


class TimedPredictor:
    def __init__(self, predictor):
        self.predictor = predictor; self.calls = 0; self.attempts = 0; self.durations = []; self.last_started = 0.0
    def predict(self, height):
        start = time.monotonic(); self.last_started = start; self.attempts += 1
        result = self.predictor.predict(height)
        self.calls += 1; self.durations.append(time.monotonic() - start)
        self.durations = self.durations[-128:]
        return result


class GlobalTraversabilityNode(Node):
    def __init__(self):
        super().__init__('scout_global_traversability')
        share = Path(get_package_share_directory('scout_global_traversability'))
        defaults = {'weights_path': str(share / 'models/student_best_epoch_weights.pth'),
                    'device': 'cuda', 'minimum_valid_ratio': 0.70, 'classification_threshold': 0.5,
                    'storage_root': '/mnt/nvme0n1/autoware/scout_traversability_maps',
                    'session_topic': '/scout/elevation/session',
                    'local_topic': '/scout/traversability/global_local',
                    'overview_topic': '/scout/traversability/global_overview',
                    'local_length_m': 10.0, 'local_publish_hz': 2.0,
                    'overview_publish_hz': 0.2, 'overview_max_size': 512,
                    'max_model_tiles_per_sec': 2.0, 'processing_hz': 5.0,
                    'processing_budget_sec': 0.08, 'max_tasks_per_tick': 8,
                    'source_changes_per_poll': 64, 'max_cached_tiles': 16,
                    'pose_timeout_sec': 0.5, 'stale_result_timeout_sec': 10.0,
                    'publish_direction_probabilities': False}
        for key, value in defaults.items(): self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}
        if not (0 < self.p['minimum_valid_ratio'] <= 1 and 0 < self.p['classification_threshold'] < 1):
            raise ValueError('Invalid model coverage or probability threshold')
        if not (0 < self.p['max_model_tiles_per_sec'] <= 10 and 0 < self.p['local_length_m'] <= 30
                and 16 <= self.p['overview_max_size'] <= 512 and 1 <= self.p['max_cached_tiles'] <= 128
                and 0 < self.p['pose_timeout_sec'] <= 0.5 and 0 < self.p['processing_budget_sec'] <= 0.2):
            raise ValueError('Invalid processing or window bounds')
        self.weights = Path(self.p['weights_path']).resolve(strict=True)
        self.model_sha = hashlib.sha256(self.weights.read_bytes()).hexdigest()
        code_root = Path(__file__).resolve().parent
        code_files = ['ht_inference.py', 'vendor_ht/model.py', 'vendor_ht/core.py', 'source.py', 'store.py', 'engine.py']
        self.code_sha = hashlib.sha256(json.dumps({name: hashlib.sha256((code_root / name).read_bytes()).hexdigest() for name in code_files}, sort_keys=True).encode()).hexdigest()
        self.root = Path(self.p['storage_root']).expanduser().resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.instance_lock = (self.root / ('node_domain_' + str(os.environ.get('ROS_DOMAIN_ID', '0')) + '.lock')).open('a+')
        try: fcntl.flock(self.instance_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Global traversability is already running in this ROS domain')
        self.engine = None; self.predictor = None; self.source_context = None; self.requested = None
        self.pose = None; self.initialized = False; self.retry_after = 0.; self.next_model = 0.
        self.state = 'WAITING_SESSION'; self.reason = 'Waiting for the current global elevation session'
        self.last_log_state = None; self.last_error_log = 0.; self.local_cleared = False; self.result_db = ''
        q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.RELIABLE)
        self.local_pub = self.create_publisher(GridMap, self.p['local_topic'], q)
        self.global_pub = self.create_publisher(GridMap, self.p['overview_topic'], q)
        self.status_pub = self.create_publisher(DiagnosticArray, '/scout/traversability/status', 5)
        self.create_subscription(String, self.p['session_topic'], self.on_session, q)
        self.create_subscription(Odometry, '/localization/kinematic_state', self.on_pose, 1)
        self.create_subscription(LocalizationInitializationState, '/localization/initialization_state',
                                 lambda m: setattr(self, 'initialized', int(m.state) == 3), q)
        self.create_timer(1. / self.p['processing_hz'], self.tick)
        self.create_timer(1. / self.p['local_publish_hz'], self.publish_local)
        self.create_timer(1. / self.p['overview_publish_hz'], self.publish_overview)
        self.create_timer(1., self.publish_status)

    @staticmethod
    def context_key(context):
        return (context['database'], context['identity_sha256'], context['run_id'])

    def on_session(self, message):
        try:
            value = json.loads(message.data)
            if value.get('coordinate_kind') != 'local_ndt_map': raise ValueError('Expected local NDT map session')
            self.context_key(value)
            for key, size in [('run_id', 32), ('identity_sha256', 64)]:
                text = value[key]
                if not isinstance(text, str) or len(text) != size or any(c not in '0123456789abcdef' for c in text):
                    raise ValueError('Invalid source session identifier')
            if not Path(value['database']).is_absolute(): raise ValueError('Source database path must be absolute')
            if self.requested is None or self.context_key(value) != self.context_key(self.requested):
                self.requested = value; self.retry_after = 0.
        except (ValueError, KeyError, TypeError) as error:
            self.state, self.reason = 'INVALID_SESSION', str(error)

    def on_pose(self, message):
        p = message.pose.pose.position
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        if message.header.frame_id != 'map' or message.child_frame_id != 'base_link' or not np.isfinite([p.x, p.y, p.z, stamp]).all():
            self.pose = None; return
        self.pose = (float(p.x), float(p.y), stamp, time.monotonic())

    def fresh_pose(self):
        if not self.initialized or self.pose is None: return None
        x, y, stamp, receipt = self.pose; now = self.get_clock().now().nanoseconds / 1e9
        if not (-.1 <= now - stamp <= self.p['pose_timeout_sec'] and 0 <= time.monotonic() - receipt <= self.p['pose_timeout_sec']): return None
        return (x, y)

    def clear_windows(self):
        center = self.fresh_pose() or (0., 0.)
        self.local_pub.publish(window_message(empty_window(center), self.get_clock().now().to_msg()))
        self.global_pub.publish(window_message(empty_window(center, size_m=.04), self.get_clock().now().to_msg()))
        self.local_cleared = True

    def ensure_context(self):
        if self.requested is None: return False
        if self.source_context is not None and self.context_key(self.source_context) == self.context_key(self.requested): return self.engine is not None
        if time.monotonic() < self.retry_after: return False
        if self.engine is not None: self.engine.close(); self.engine = None
        self.source_context = None; self.clear_windows()
        try:
            resolution = float(self.requested['identity']['mapping_parameters']['resolution'])
            if not np.isclose(resolution, 0.04, rtol=0.0, atol=1e-9):
                raise ValueError('This HT checkpoint requires a 0.04 m source grid')
            if not Path(self.requested['database']).is_file(): raise FileNotFoundError('Global elevation database is not ready yet')
            if self.predictor is None:
                self.state, self.reason = 'LOADING_MODEL', 'Loading the local HT checkpoint'
                self.publish_status()
                import torch
                torch.set_num_threads(1); torch.backends.cudnn.allow_tf32 = False
                from .ht_inference import HTPredictor
                self.predictor = TimedPredictor(HTPredictor(str(self.weights), device=self.p['device'],
                                      min_valid_ratio=self.p['minimum_valid_ratio']))
            from .engine import Engine
            import torch
            config = {'torch_version': str(torch.__version__), 'cudnn_version': torch.backends.cudnn.version(),
                      'cudnn_allow_tf32': False, 'backend': self.p['device'], 'min_valid_ratio': self.p['minimum_valid_ratio'], 'aggregate': 'minimum_8_raw_channels', 'inference_code_sha256': self.code_sha}
            fingerprint = hashlib.sha256(json.dumps({'model': self.model_sha, 'config': config}, sort_keys=True).encode()).hexdigest()[:20]
            folder = self.root / self.requested['run_id'] / fingerprint; folder.mkdir(parents=True, exist_ok=True)
            self.result_db = str(folder / 'traversability.sqlite3')
            self.engine = Engine(self.requested, self.predictor, self.result_db,
                                 model_sha256=self.model_sha, config=config,
                                 max_cached_tiles=self.p['max_cached_tiles'])
            self.source_context = self.requested; self.state, self.reason = 'PROCESSING', 'Updating changed global tiles'
            return True
        except Exception as error:
            self.engine = None; self.retry_after = time.monotonic() + 2.
            self.state, self.reason = 'WAITING_SOURCE', str(error); self.log_error(error); return False

    def log_error(self, error):
        if time.monotonic() - self.last_error_log > 5.:
            self.get_logger().warning(str(error)); self.last_error_log = time.monotonic()

    def tick(self):
        if not self.ensure_context(): return
        try:
            self.engine.poll_changes(limit=int(self.p['source_changes_per_poll']), near_xy=self.fresh_pose())
            start = time.monotonic()
            for _ in range(int(self.p['max_tasks_per_tick'])):
                if time.monotonic() - start >= self.p['processing_budget_sec']: break
                before = self.predictor.attempts
                try:
                    item = self.engine.process_one(near_xy=self.fresh_pose(),
                              allow_inference=time.monotonic() >= self.next_model)
                finally:
                    # Failed inference/storage still consumed a model attempt.
                    # The rate limit is measured from its start, on every path.
                    if self.predictor.attempts > before:
                        self.next_model = self.predictor.last_started + 1. / self.p['max_model_tiles_per_sec']
                if item is None: break
                if self.predictor.attempts > before:
                    break
                if item.get('state') == 'inference_budget_wait': break
            self.state, self.reason = 'RUNNING', 'Global tile inference active; local view comes from the result cache'
        except Exception as error:
            self.state, self.reason = 'PROCESSING_ERROR', str(error); self.log_error(error)

    def publish_local(self):
        if self.engine is None: return
        center = self.fresh_pose()
        if center is None:
            if not self.local_cleared:
                self.local_pub.publish(window_message(empty_window(), self.get_clock().now().to_msg())); self.local_cleared = True
            return
        try:
            window = self.engine.local_window(center, size_m=float(self.p['local_length_m']))
            self.local_pub.publish(window_message(window, self.get_clock().now().to_msg(),
                        self.p['classification_threshold'], self.p['publish_direction_probabilities'],
                        self.p['stale_result_timeout_sec'])); self.local_cleared = False
        except Exception as error: self.log_error(error)

    def publish_overview(self):
        if self.engine is None or self.global_pub.get_subscription_count() == 0: return
        try:
            window = self.engine.overview(max_size=int(self.p['overview_max_size']))
            if window is not None:
                self.global_pub.publish(window_message(window, self.get_clock().now().to_msg(),
                        self.p['classification_threshold'], False, float('inf')))
        except Exception as error: self.log_error(error)

    def publish_status(self):
        message = DiagnosticArray(); message.header.stamp = self.get_clock().now().to_msg()
        s = DiagnosticStatus(); s.name = 'scout_global_traversability'; s.hardware_id = self.p['device']
        s.message = self.state; s.level = DiagnosticStatus.OK if self.state == 'RUNNING' else DiagnosticStatus.WARN
        stats = self.engine.stats() if self.engine is not None else {}
        values = {'reason': self.reason, 'source_run_id': self.source_context['run_id'] if self.source_context else '',
                  'result_database': self.result_db, 'model_sha256': self.model_sha,
                  'inference_code_sha256': self.code_sha,
                  'model_calls': self.predictor.calls if self.predictor else 0,
                  'model_attempts': self.predictor.attempts if self.predictor else 0,
                  'pose_fresh': self.fresh_pose() is not None,
                  'inference_limit_tiles_per_sec': self.p['max_model_tiles_per_sec'], **stats}
        if self.predictor and self.predictor.durations:
            values['model_ms_p50'] = float(np.quantile(self.predictor.durations, .5) * 1000)
            values['model_ms_p95'] = float(np.quantile(self.predictor.durations, .95) * 1000)
        s.values = [KeyValue(key=str(k), value=str(v)) for k, v in values.items()]
        message.status = [s]; self.status_pub.publish(message)
        if self.state != self.last_log_state:
            self.get_logger().info(f'{self.state}: {self.reason}'); self.last_log_state = self.state

    def destroy_node(self):
        if self.engine is not None: self.engine.close()
        if getattr(self, 'instance_lock', None): self.instance_lock.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args); node = None
    try:
        node = GlobalTraversabilityNode(); rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        if node is not None: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
