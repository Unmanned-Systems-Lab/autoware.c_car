"""Run local elevation only while the Universe input gate is healthy.

The controller is independent of ROS so failure and restart semantics can be tested
without starting DDS, a GPU worker, robot drivers, or a geographic localization source.
"""

from copy import deepcopy
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from .local_elevation_session import (
    canonical_json, file_signature, hash_file, local_identity, prepare_local_session,
    merge_parameters, require_local_parameters,
)


def grid_observation(message):
    """Read the actual ROS GridMap schema: Header is on GridMap, not GridMapInfo."""
    if message.header.frame_id != 'map':
        raise ValueError('Elevation GridMap is not in the local map frame')
    index = list(message.layers).index('observation_count')
    observed = sum(math.isfinite(v) and v > 0 for v in message.data[index].data)
    stamp = message.header.stamp.sec + message.header.stamp.nanosec / 1e9
    return int(observed), stamp


class LocalElevationController:
    def __init__(self, reference_pcd, parameters, storage_root, spawn, stop,
                 gate_timeout=0.7, progress_timeout=1.0, run_id=None, *,
                 seed_estimator=None, ground_seed_length_x=10.0, ground_seed_length_y=10.0,
                 ground_seed_pose_timeout=0.5):
        self.reference_source = Path(reference_pcd).expanduser().absolute()
        self.reference = self.reference_source.resolve(strict=True)
        self.reference_sha256, self.reference_signature = hash_file(self.reference_source)
        self.parameters = require_local_parameters(parameters, allow_pending_ground_seed=True)
        self.ground_seed_required = self.parameters['global_map_publish_unobserved_prior']
        self.ground_seed = None
        self.seed_estimation_elapsed_sec = None
        self.seed_estimator = seed_estimator
        self.ground_seed_length_x, self.ground_seed_length_y = float(ground_seed_length_x), float(ground_seed_length_y)
        self.ground_seed_pose_timeout = float(ground_seed_pose_timeout)
        if (not all(math.isfinite(value) and 0 < value <= 30
                    for value in (self.ground_seed_length_x, self.ground_seed_length_y))
                or not 0 < self.ground_seed_pose_timeout <= 0.5):
            raise ValueError('Invalid initial ground seed dimensions or pose freshness')
        self.loaded_map_xyz, self.loaded_map_sha256 = None, None
        self.seed_pose, self.seed_pose_last_stamp, self.seed_pose_error = None, 0, ''
        self.storage_root = storage_root
        self.spawn, self.stop = spawn, stop
        self.gate_timeout, self.progress_timeout = float(gate_timeout), float(progress_timeout)
        if not 0 < self.gate_timeout <= 0.7 or not 0 < self.progress_timeout <= 2.0:
            raise ValueError('Invalid gate/progress freshness limits')
        self.run_id = run_id or uuid.uuid4().hex
        self.gate, self.gate_received, self.gate_error = None, None, ''
        self.identity, self.session = None, None
        self.process, self.generation = None, 0
        self.retry_after, self.fault = 0.0, ''
        self.state, self.reason = 'WAITING_INPUT_GATE', ''
        self.worker_started_ros = 0.0
        self.processed_count, self.progress_received = 0, None
        self.observed_cells, self.observation_received, self.observation_stamp = 0, None, None

    def on_loaded_map(self, xyz, frame='map'):
        from .elevation_input_contract import loaded_map_digest
        import numpy as np
        try:
            if frame != 'map':
                raise ValueError('Loaded ground-seed map is not in map frame')
            points = np.array(xyz, dtype=np.float64, copy=True)
            digest = loaded_map_digest(points)
            if self.loaded_map_sha256 is not None and digest != self.loaded_map_sha256:
                raise ValueError('Loaded ground-seed map geometry changed; restart localization')
            points.setflags(write=False)
            self.loaded_map_xyz, self.loaded_map_sha256 = points, digest
        except (ValueError, TypeError, OverflowError) as error:
            self.latch_fault(f'Invalid actual ground-seed map: {error}')

    def on_seed_pose(self, position, quaternion, covariance, stamp, received, now_ns,
                     frame='map', child_frame='base_link'):
        from .elevation_input_contract import GateConfig, require_fresh, require_pose
        try:
            config = GateConfig(max_age_sec=self.ground_seed_pose_timeout)
            if type(stamp) is not int or stamp <= self.seed_pose_last_stamp:
                raise ValueError('Initial ground base-pose timestamp did not advance')
            require_pose(frame, position, quaternion, covariance, config, child_frame)
            require_fresh(stamp, now_ns, received, received, config)
            self.seed_pose = {
                'position': [float(value) for value in position],
                'quaternion_xyzw': [float(value) for value in quaternion],
                'covariance': [float(value) for value in covariance],
                'stamp_ns': stamp, 'frame': frame, 'child_frame': child_frame,
                'received_monotonic': received,
            }
            self.seed_pose_last_stamp, self.seed_pose_error = stamp, ''
        except (ValueError, TypeError, OverflowError) as error:
            self.seed_pose, self.seed_pose_error = None, str(error)

    def prepare_ground_seed(self, gate, now, now_ros):
        """Freeze a bounded prior once; subsequent poses never move its rectangle."""
        if not self.ground_seed_required or self.ground_seed is not None:
            return True
        from .elevation_input_contract import GateConfig, require_fresh
        self.state = 'WAITING_GROUND_SEED'
        if self.loaded_map_xyz is None:
            self.reason = 'Waiting for actual transient-local /map/pointcloud_map geometry'
            return False
        if self.loaded_map_sha256 != gate['map_identity']['loaded_map_geometry_sha256']:
            self.latch_fault('Ground-seed map geometry hash differs from the approved Gate map')
            return False
        if self.seed_pose is None:
            self.reason = self.seed_pose_error or 'Waiting for fresh /localization/kinematic_state in map/base_link'
            return False
        try:
            pose = self.seed_pose
            require_fresh(pose['stamp_ns'], int(now_ros * 1e9), pose['received_monotonic'], now,
                          GateConfig(max_age_sec=self.ground_seed_pose_timeout))
            estimator = self.seed_estimator
            if estimator is None:
                from .initial_ground import estimate_initial_ground
                estimator = estimate_initial_ground
            estimation_started = time.monotonic()
            estimate = estimator(self.loaded_map_xyz, pose['position'],
                                 length_x=self.ground_seed_length_x, length_y=self.ground_seed_length_y)
            elapsed = max(0.0, time.monotonic() - estimation_started)
            seed = deepcopy(estimate)
            seed['observation_count'] = 0
            seed['window_follows_robot'] = False
            seed['source'] = {
                'map_topic': '/map/pointcloud_map', 'pose_topic': '/localization/kinematic_state',
                'loaded_map_geometry_sha256': self.loaded_map_sha256,
                'reference_pcd_sha256': self.reference_sha256,
                'base_pose': {key: deepcopy(value) for key, value in pose.items()
                              if key != 'received_monotonic'},
                'snapshot_ros_sec': now_ros, 'estimated_at_ros_sec': now_ros + elapsed,
                'estimation_elapsed_sec': elapsed, 'max_pose_age_sec': self.ground_seed_pose_timeout,
            }
            if (abs(float(seed['length_x']) - self.ground_seed_length_x) > 1e-6
                    or abs(float(seed['length_y']) - self.ground_seed_length_y) > 1e-6):
                raise ValueError('Ground estimator returned unexpected prior dimensions')
            parameters = deepcopy(self.parameters)
            parameters['global_map_initial_elevation'] = float(seed['height'])
            for index, axis in enumerate(('x', 'y')):
                parameters['global_map_prior_min_' + axis] = float(seed['bounds_xy'][0][index])
                parameters['global_map_prior_max_' + axis] = float(seed['bounds_xy'][1][index])
            parameters = require_local_parameters(parameters)
            # Validate all source/quality/center/height/bounds before freezing or creating a DB.
            local_identity(gate, self.reference, self.reference_sha256, parameters, seed)
            self.parameters, self.ground_seed = parameters, seed
            self.seed_estimation_elapsed_sec = elapsed
            # The pose was fresh when the initial snapshot was accepted. A slow fit
            # does not discard that valid fixed seed; only worker startup waits for
            # another fresh Gate heartbeat after the executor resumes callbacks.
            if (now - self.gate_received + elapsed > self.gate_timeout
                    or now_ros - gate['ros_stamp_sec'] + elapsed > self.gate_timeout):
                self.state = 'WAITING_INPUT_GATE'
                self.reason = 'Initial ground frozen; waiting for fresh Gate after ground estimation'
                return False
            return True
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            self.reason = f'No trusted initial ground seed: {error}'
            return False

    def on_gate(self, data, received):
        try:
            value = json.loads(data) if isinstance(data, str) else deepcopy(data)
            if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
                    or value['schema_version'] != 1):
                raise ValueError('Invalid input gate schema')
            stamp = value.get('ros_stamp_sec')
            if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                raise ValueError('Invalid gate timestamp')
            if value.get('state') not in {
                'READY', 'WAITING_INITIALIZATION', 'WAITING_MAP', 'WAITING_LOCALIZATION',
                'WAITING_INPUT', 'WAITING_TF', 'RESTART_REQUIRED',
            }:
                raise ValueError('Unknown input gate state')
            self.gate, self.gate_received, self.gate_error = value, received, ''
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            self.gate, self.gate_received, self.gate_error = None, None, str(error)

    def on_processed(self, count, received, generation):
        if generation != self.generation or self.process is None:
            return
        count = int(count)
        if count < self.processed_count:
            self.latch_fault('Native mapper counter reset unexpectedly; restart localization')
        elif count > self.processed_count:
            self.processed_count, self.progress_received = count, received

    def on_observation(self, observed_cells, stamp, received, generation):
        if generation != self.generation or self.process is None:
            return
        # A restored window can contain historical observations. Its timestamp must
        # represent an input processed by this worker, and native progress is also required.
        if math.isfinite(stamp) and stamp >= self.worker_started_ros - 0.1:
            self.observed_cells = int(observed_cells)
            self.observation_received, self.observation_stamp = received, stamp

    def stop_worker(self):
        process, self.process = self.process, None
        self.progress_received, self.observation_received = None, None
        if process is not None:
            self.stop(process)

    def latch_fault(self, reason):
        self.fault = self.fault or reason
        self.stop_worker()
        self.state, self.reason = 'RESTART_REQUIRED', self.fault

    def tick(self, now, now_ros):
        if self.fault:
            self.state, self.reason = 'RESTART_REQUIRED', self.fault
            return self.state
        try:
            if file_signature(self.reference_source) != self.reference_signature:
                self.latch_fault('Reference PCD changed; restart localization and elevation mapping')
                return self.state
        except (ValueError, OSError) as error:
            self.latch_fault(f'Reference PCD unavailable: {error}')
            return self.state
        gate = self.gate
        if gate and gate['state'] == 'RESTART_REQUIRED':
            self.latch_fault(gate.get('last_error') or 'Input gate requires localization restart')
            return self.state
        if self.process is not None and self.process.poll() is not None:
            code = self.process.returncode
            self.stop_worker()
            self.retry_after = now + 5.0
            self.state, self.reason = 'MAPPER_EXITED', f'Worker exited with code {code}; retry in 5 seconds'
            return self.state
        fresh = (gate is not None and self.gate_received is not None
                 and 0 <= now - self.gate_received <= self.gate_timeout
                 and -0.1 <= now_ros - gate['ros_stamp_sec'] <= self.gate_timeout)
        if not fresh or gate['state'] != 'READY':
            self.stop_worker()
            self.state = gate['state'] if fresh else 'WAITING_INPUT_GATE'
            self.reason = ((gate.get('last_error') or 'Input gate is not READY') if fresh
                           else self.gate_error or 'Input gate missing, stale, or future-dated')
            return self.state
        try:
            if self.ground_seed_required and self.ground_seed is None:
                # Validate the approved map/TF/filter contract before fitting actual geometry.
                local_identity(gate, self.reference, self.reference_sha256, self.parameters,
                               allow_pending_ground_seed=True)
                if not self.prepare_ground_seed(gate, now, now_ros):
                    return self.state
            identity = local_identity(gate, self.reference, self.reference_sha256,
                                      self.parameters, self.ground_seed)
            if self.identity is not None and identity != self.identity:
                self.latch_fault('Map, fixed sensor transform, or mapping contract changed; restart localization')
                return self.state
            if self.identity is None:
                self.session = prepare_local_session(self.storage_root, identity, self.run_id)
                self.identity = identity
            if self.process is None:
                if now < self.retry_after:
                    self.state, self.reason = 'RESTART_BACKOFF', 'Waiting before worker restart'
                    return self.state
                self.generation += 1
                self.processed_count, self.observed_cells = 0, 0
                self.progress_received, self.observation_received = None, None
                self.observation_stamp = None
                self.worker_started_ros = now_ros
                self.process = self.spawn(self.session, self.parameters, self.generation)
                self.state = 'INITIAL_GROUND' if self.ground_seed is not None else 'STARTING'
                self.reason = ('Fixed unobserved ground prior prepared; waiting for native measured cells'
                               if self.ground_seed is not None else
                               'Waiting for native successful processing and measured cells')
                return self.state
            progress_fresh = (self.progress_received is not None
                              and 0 <= now - self.progress_received <= self.progress_timeout)
            observations_fresh = (self.observation_received is not None
                                  and 0 <= now - self.observation_received <= self.progress_timeout
                                  and -0.1 <= now_ros - self.observation_stamp <= self.progress_timeout)
            if progress_fresh and observations_fresh and self.observed_cells > 0:
                self.state, self.reason = 'MAPPING', 'Native successful processing and measured elevation cells are advancing'
            elif self.observed_cells > 0:
                self.state, self.reason = 'RUNNING_AVAILABLE_MAP', 'Measured cells available; waiting for fresh native processing'
            else:
                self.state = 'INITIAL_GROUND' if self.ground_seed is not None else 'STARTING'
                self.reason = ('Fixed unobserved ground prior only; no confirmed fresh measured elevation cells yet'
                               if self.ground_seed is not None else
                               'Worker alive; no confirmed fresh measured elevation cells yet')
        except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
            # A failed session contract is never repaired by silently choosing another DB.
            self.latch_fault(f'Local elevation session error: {error}')
        return self.state

    def diagnostic_values(self):
        return {
            'reason': self.reason, 'run_id': self.run_id,
            'identity_sha256': self.session['identity_sha256'] if self.session else '',
            'database': self.session['database'] if self.session else '',
            'worker_generation': self.generation, 'native_processed_pointcloud_count': self.processed_count,
            'observed_cells_in_last_grid': self.observed_cells,
            'coordinate_kind': 'local_ndt_map', 'resume_policy': 'current_run_only',
            'ground_seed_state': ('FIXED_UNOBSERVED_PRIOR' if self.ground_seed is not None else
                                  'WAITING' if self.ground_seed_required else 'DISABLED'),
            'ground_seed_height': self.ground_seed['height'] if self.ground_seed is not None else '',
            'ground_seed_bounds_xy': canonical_json(self.ground_seed['bounds_xy']) if self.ground_seed is not None else '',
            'ground_seed_estimation_elapsed_sec': self.seed_estimation_elapsed_sec or '',
        }


def stop_process(process, logger=None):
    """Allow the native worker to flush SQLite before escalating a stuck shutdown."""
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=18)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
            if logger:
                logger.error('Mapper required forced shutdown; current-run SQLite will recover on resume')


def main(args=None):
    # Delayed imports keep session/lifecycle tests independent of ROS and DDS.
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from grid_map_msgs.msg import GridMap
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from std_msgs.msg import String, UInt64
    import yaml

    class LocalElevationManager(Node):
        def __init__(self):
            super().__init__('scout_local_elevation_manager')
            defaults = {
                'reference_pcd': '', 'mapping_config_file': '',
                'storage_root': '/mnt/nvme0n1/autoware/scout_local_elevation_maps',
                'input_status_topic': '/scout/elevation/input_status',
                'gate_timeout_sec': 0.7, 'progress_timeout_sec': 1.0,
                'ground_seed_length_x': 10.0, 'ground_seed_length_y': 10.0,
                'ground_seed_pose_timeout_sec': 0.5,
            }
            for name, value in defaults.items():
                self.declare_parameter(name, value)
            value = lambda name: self.get_parameter(name).value
            core = Path(get_package_share_directory('elevation_mapping_cupy')) / 'config/core/core_param.yaml'
            self.mapping_config = Path(value('mapping_config_file')).resolve(strict=True)
            self.parameters = merge_parameters(core, self.mapping_config)
            self.config_signature = file_signature(self.mapping_config)
            self.executable = (Path(get_package_prefix('elevation_mapping_cupy'))
                               / 'lib/elevation_mapping_cupy/elevation_mapping_node.py')
            if not self.executable.is_file():
                raise FileNotFoundError(self.executable)
            self.worker_subscriptions = []
            self.controller = LocalElevationController(
                value('reference_pcd'), self.parameters, value('storage_root'),
                self.start_worker, self.stop_worker,
                value('gate_timeout_sec'), value('progress_timeout_sec'),
                ground_seed_length_x=value('ground_seed_length_x'),
                ground_seed_length_y=value('ground_seed_length_y'),
                ground_seed_pose_timeout=value('ground_seed_pose_timeout_sec'))
            self.status_publisher = self.create_publisher(DiagnosticArray, '/scout/elevation/status', 10)
            latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            self.session_publisher = self.create_publisher(String, '/scout/elevation/session', latched)
            self.gate_subscription = self.create_subscription(
                String, value('input_status_topic'),
                lambda msg: self.controller.on_gate(msg.data, time.monotonic()), 1)
            self.map_subscription, self.pose_subscription = None, None
            if self.controller.ground_seed_required:
                self.map_subscription = self.create_subscription(
                    PointCloud2, '/map/pointcloud_map', self.on_map, latched)
                latest_pose = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
                self.pose_subscription = self.create_subscription(
                    Odometry, '/localization/kinematic_state', self.on_seed_pose, latest_pose)
            self.last_session, self.last_state = None, None
            self.timer = self.create_timer(0.2, self.tick)

        def on_map(self, message):
            from .elevation_input_contract import xyz_and_records
            try:
                xyz, _ = xyz_and_records(
                    message.data, message.width, message.height, message.point_step,
                    message.row_step, [{'name': field.name, 'offset': field.offset,
                                        'datatype': field.datatype, 'count': field.count}
                                       for field in message.fields], message.is_bigendian)
                self.controller.on_loaded_map(xyz, message.header.frame_id)
            except (ValueError, TypeError, KeyError, OverflowError) as error:
                self.controller.latch_fault(f'Invalid actual ground-seed map message: {error}')

        def on_seed_pose(self, message):
            from .elevation_input_contract import stamp_ns
            try:
                position, quaternion = message.pose.pose.position, message.pose.pose.orientation
                self.controller.on_seed_pose(
                    [position.x, position.y, position.z],
                    [quaternion.x, quaternion.y, quaternion.z, quaternion.w], message.pose.covariance,
                    stamp_ns(message.header.stamp.sec, message.header.stamp.nanosec),
                    time.monotonic(), self.get_clock().now().nanoseconds,
                    message.header.frame_id, message.child_frame_id)
            except (ValueError, TypeError, OverflowError) as error:
                self.controller.seed_pose = None
                self.controller.seed_pose_error = str(error)

        def clear_worker_subscriptions(self):
            for subscription in self.worker_subscriptions:
                self.destroy_subscription(subscription)
            self.worker_subscriptions = []

        def stop_worker(self, process):
            try:
                if self.context.ok():
                    self.publish_status('STOPPING_FLUSHING', 'Waiting for current-run SQLite flush')
                stop_process(process, self.get_logger())
            finally:
                self.clear_worker_subscriptions()

        def start_worker(self, session, parameters, generation):
            self.clear_worker_subscriptions()
            runtime_parameters = deepcopy(parameters)
            runtime_parameters['global_map_storage_path'] = session['database']
            runtime_parameters['global_map_resume_from_disk'] = True
            filename = Path(session['directory']) / 'mapping.params.yaml'
            filename.write_text(yaml.safe_dump(
                {'/elevation_mapping_node': {'ros__parameters': runtime_parameters}},
                sort_keys=False), encoding='utf-8')
            processed = self.create_subscription(
                UInt64, '/elevation_mapping_node/processed_pointcloud_count',
                lambda msg: self.controller.on_processed(msg.data, time.monotonic(), generation), 10)
            grid_topic = parameters.get('global_map_local_topic',
                                        '/elevation_mapping_node/elevation_map_global_local')
            grid = self.create_subscription(
                GridMap, grid_topic, lambda msg: self.on_grid(msg, generation), 10)
            self.worker_subscriptions = [processed, grid]
            try:
                process = subprocess.Popen(
                    [sys.executable, str(self.executable), '--ros-args',
                     '-r', '__node:=elevation_mapping_node', '--params-file', str(filename)],
                    start_new_session=True)
            except OSError:
                self.clear_worker_subscriptions()
                raise
            self.get_logger().info(f'Local elevation worker started; run={session["run_id"]}, pid={process.pid}')
            return process

        def on_grid(self, message, generation):
            try:
                observed, stamp = grid_observation(message)
                self.controller.on_observation(observed, stamp, time.monotonic(), generation)
            except (ValueError, IndexError, TypeError, AttributeError):
                return

        def publish_status(self, state, reason=None):
            message = DiagnosticArray()
            message.header.stamp = self.get_clock().now().to_msg()
            status = DiagnosticStatus()
            status.name = 'scout_local_elevation/session'
            status.hardware_id = 'local_ndt_map'
            status.message = state
            status.level = (DiagnosticStatus.OK if state == 'MAPPING' else
                            DiagnosticStatus.ERROR if state == 'RESTART_REQUIRED' else DiagnosticStatus.WARN)
            values = self.controller.diagnostic_values()
            if reason is not None:
                values['reason'] = reason
            status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
            message.status = [status]
            self.status_publisher.publish(message)

        def tick(self):
            try:
                if file_signature(self.mapping_config) != self.config_signature:
                    self.controller.latch_fault('Mapping configuration changed; restart localization')
            except (ValueError, OSError) as error:
                self.controller.latch_fault(f'Mapping configuration unavailable: {error}')
            state = self.controller.tick(time.monotonic(), self.get_clock().now().nanoseconds / 1e9)
            metadata = self.controller.session
            if metadata is not None and metadata != self.last_session:
                self.session_publisher.publish(String(data=canonical_json(metadata)))
                self.last_session = deepcopy(metadata)
            self.publish_status(state)
            if state != self.last_state:
                self.get_logger().info(f'Local elevation: {state}; {self.controller.reason}')
                self.last_state = state

    rclpy.init(args=args)
    node = None
    try:
        node = LocalElevationManager()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.controller.stop_worker()
            node.destroy_node()
        rclpy.try_shutdown()
