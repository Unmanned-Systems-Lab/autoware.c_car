"""Content-bound local NDT elevation sessions; no ROS or geographic origin dependency."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import uuid


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def file_signature(filename):
    path = Path(filename).expanduser().resolve(strict=True)
    stat = path.stat()
    if not path.is_file():
        raise ValueError('Reference PCD must be a regular file')
    return (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def hash_file(filename):
    """Reject a file changing during hashing, rather than binding a partial version."""
    before = file_signature(filename)
    digest = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if file_signature(filename) != before:
        raise ValueError('Reference file changed while its content was being hashed')
    return digest.hexdigest(), before


def merge_parameters(*files):
    """Merge only the selected mapper node, without importing another vehicle's kit."""
    import yaml

    def merge(left, right):
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(left.get(key), dict):
                merge(left[key], value)
            else:
                left[key] = deepcopy(value)

    result = {}
    for filename in files:
        document = yaml.safe_load(Path(filename).read_text(encoding='utf-8'))
        parameters = document['/elevation_mapping_node']['ros__parameters']
        if not isinstance(parameters, dict):
            raise ValueError(f'Mapper parameters must be a dictionary: {filename}')
        merge(result, parameters)
    # The configured zero bounds are a request to estimate the initial ground,
    # not a valid worker contract. Only the manager may replace them with a seed.
    return require_local_parameters(result, allow_pending_ground_seed=True)


def require_local_parameters(parameters, *, allow_pending_ground_seed=False):
    result = deepcopy(parameters)
    required = {
        'map_frame': 'map', 'corrected_map_frame': 'map', 'base_frame': 'base_link',
        'use_initializer_at_start': False, 'enable_drift_corrected_TF_publishing': False,
        'enable_builtin_traversability': False,
        'enable_global_map': True, 'ray_origin_frame': 'rslidar', 'self_filter_enabled': False,
    }
    for key, value in required.items():
        if key in result and (type(result[key]) is not type(value) or result[key] != value):
            raise ValueError(f'Unsafe local elevation parameter {key}: expected {value!r}')
        result[key] = value
    publish_prior = result.setdefault('global_map_publish_unobserved_prior', False)
    if type(publish_prior) is not bool:
        raise ValueError('global_map_publish_unobserved_prior must be boolean')
    if publish_prior:
        if result.get('global_map_prior_bounds_enabled') is not True:
            raise ValueError('Local ground prior must have fixed bounds enabled')
        variance = result.setdefault('global_map_initial_variance', 10000.0)
        if isinstance(variance, bool) or not math.isfinite(float(variance)) or float(variance) < 10000.0:
            raise ValueError('Unobserved ground prior must retain variance >= 10000')
        height = result.get('global_map_initial_elevation', 0.0)
        if isinstance(height, bool) or not math.isfinite(float(height)):
            raise ValueError('Ground prior elevation must be finite')
        bounds = [result.get('global_map_prior_' + key, 0.0)
                  for key in ('min_x', 'min_y', 'max_x', 'max_y')]
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for value in bounds):
            raise ValueError('Ground prior bounds must be finite')
        valid = float(bounds[0]) < float(bounds[2]) and float(bounds[1]) < float(bounds[3])
        if not valid and not (allow_pending_ground_seed and all(float(value) == 0.0 for value in bounds)):
            raise ValueError('Ground prior bounds are not a fixed nonempty rectangle')
    subscribers = result.setdefault('subscribers', {
        'lidar': {'topic_name': '/scout/elevation/points', 'data_type': 'pointcloud'},
    })
    if not isinstance(subscribers, dict) or set(subscribers) != {'lidar'}:
        raise ValueError('Local elevation accepts only the gated lidar subscriber')
    lidar = subscribers['lidar']
    if not isinstance(lidar, dict):
        raise ValueError('Invalid gated lidar subscriber')
    if lidar.get('topic_name') != '/scout/elevation/points':
        raise ValueError('Local elevation pointcloud must pass through /scout/elevation/points')
    if lidar.setdefault('data_type', 'pointcloud') != 'pointcloud':
        raise ValueError('Local elevation lidar data_type must be pointcloud')
    # The first worker sees an empty, newly created run directory. Subsequent workers
    # may recover only this run after a temporary localization loss or worker exit.
    result['global_map_resume_from_disk'] = True
    result.pop('global_map_database_path', None)
    result.pop('global_map_storage_path', None)
    canonical_json(result)
    return result


def _finite_vector(value, size, name):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f'{name} must contain {size} numbers')
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f'{name} contains non-finite values')
    return result


def local_identity(gate, reference_pcd, reference_sha256, parameters, ground_seed=None,
                   *, allow_pending_ground_seed=False):
    """Bind measured map, fixed T_base_sensor, filtering, and every mapper parameter."""
    source = gate['map_identity']
    if not isinstance(source, dict):
        raise ValueError('Missing local map identity')
    for key, expected in {
        'coordinate_kind': 'local_ndt_map', 'backend': 'universe',
        'frame': 'map', 'base_frame': 'base_link',
    }.items():
        if source.get(key) != expected:
            raise ValueError(f'Unexpected local map identity: {key}')
    for key in ('reference_pcd_sha256', 'loaded_map_geometry_sha256'):
        if not isinstance(source.get(key), str) or not re.fullmatch('[0-9a-f]{64}', source[key]):
            raise ValueError(f'Invalid map hash: {key}')
    reference = str(Path(reference_pcd).expanduser().resolve(strict=True))
    if str(Path(source['reference_pcd']).expanduser().resolve(strict=True)) != reference:
        raise ValueError('Gate reference PCD differs from manager reference')
    if source['reference_pcd_sha256'] != reference_sha256:
        raise ValueError('Gate reference content differs from manager reference')
    sensor = gate['sensor_transform']
    if not isinstance(sensor, dict):
        raise ValueError('Missing fixed base-to-sensor transform')
    translation = _finite_vector(sensor['translation'], 3, 'Sensor translation')
    rotation = _finite_vector(sensor['rotation'], 4, 'Sensor quaternion')
    norm = math.sqrt(sum(v * v for v in rotation))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError('Sensor quaternion is not normalized')
    rotation = [v / norm for v in rotation]
    # Quaternion signs represent the same rotation; ignore serialization noise.
    for value in reversed(rotation):
        if abs(value) > 1e-12:
            if value < 0:
                rotation = [-v for v in rotation]
            break
    filtering = gate['cloud_filter_contract']
    if not isinstance(filtering, dict) or not filtering:
        raise ValueError('Missing measured-cloud filter contract')
    identity = {
        'schema_version': 1, 'coordinate_kind': 'local_ndt_map',
        'map_identity': dict(source, reference_pcd=reference),
        'sensor_transform': {'direction': 'base_link_from_sensor',
                             'translation': [round(v, 8) for v in translation],
                             'rotation': [round(v, 8) for v in rotation]},
        'cloud_filter_contract': deepcopy(filtering),
        'mapping_parameters': require_local_parameters(
            parameters, allow_pending_ground_seed=allow_pending_ground_seed),
        'ray_evidence_version': 2,
    }
    if ground_seed is not None:
        validate_ground_seed(ground_seed, identity['mapping_parameters'], source)
        identity['initial_ground_seed'] = deepcopy(ground_seed)
    elif parameters.get('global_map_publish_unobserved_prior') and not allow_pending_ground_seed:
        raise ValueError('Local ground prior requires a measured-map ground seed')
    canonical_json(identity)
    return identity


def validate_ground_seed(seed, parameters, map_identity):
    """A prior is explicitly unobserved and bound to one actual map and one pose."""
    if not isinstance(seed, dict) or seed.get('method') != 'pcd_lower_envelope_plane_v1':
        raise ValueError('Unknown initial ground seed method')
    if seed.get('quality', {}).get('accepted') is not True or not isinstance(seed.get('evidence'), dict):
        raise ValueError('Initial ground seed has no accepted quality evidence')
    source = seed.get('source', {})
    for key in ('loaded_map_geometry_sha256', 'reference_pcd_sha256'):
        if source.get(key) != map_identity[key]:
            raise ValueError('Initial ground seed is bound to another map')
    if source.get('map_topic') != '/map/pointcloud_map' or source.get('pose_topic') != '/localization/kinematic_state':
        raise ValueError('Initial ground seed must use the actual map and localized base pose')
    pose = source.get('base_pose', {})
    xyz = _finite_vector(pose.get('position'), 3, 'Initial base pose')
    if pose.get('frame') != 'map' or pose.get('child_frame') != 'base_link':
        raise ValueError('Initial base pose has incorrect frames')
    center = _finite_vector(seed.get('center_xy'), 2, 'Initial ground center')
    if any(abs(a - b) > 1e-6 for a, b in zip(center, xyz[:2])):
        raise ValueError('Ground prior is not centered on the initial base pose')
    bounds = seed.get('bounds_xy')
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError('Invalid fixed ground bounds')
    lower = _finite_vector(bounds[0], 2, 'Ground lower bounds')
    upper = _finite_vector(bounds[1], 2, 'Ground upper bounds')
    for index, axis in enumerate(('x', 'y')):
        length = float(seed['length_' + axis])
        if not math.isfinite(length) or length <= 0:
            raise ValueError('Invalid fixed ground prior size')
        if (abs(lower[index] - (center[index] - length / 2)) > 1e-6
                or abs(upper[index] - (center[index] + length / 2)) > 1e-6
                or abs(float(parameters['global_map_prior_min_' + axis]) - lower[index]) > 1e-6
                or abs(float(parameters['global_map_prior_max_' + axis]) - upper[index]) > 1e-6):
            raise ValueError('Worker ground bounds do not match the frozen seed')
    height = float(seed['height'])
    if (not math.isfinite(height) or abs(float(parameters['global_map_initial_elevation']) - height) > 1e-6):
        raise ValueError('Worker ground elevation does not match the frozen seed')
    if seed.get('observation_count', 0) != 0:
        raise ValueError('Ground seed cannot claim measured observations')
    canonical_json(seed)


def prepare_local_session(root, identity, run_id=None):
    """Create a new run; intentionally never discover or recover an older run."""
    run_id = run_id or uuid.uuid4().hex
    if not isinstance(run_id, str) or not re.fullmatch('[0-9a-f]{32}', run_id):
        raise ValueError('Run ID must be a UUID hex string')
    identity_sha256 = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    directory = Path(root).expanduser().resolve() / 'local_ndt' / identity_sha256[:24] / run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'session.json'
    database = directory / 'tiles.sqlite3'
    if manifest.exists():
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        if (metadata.get('identity') != identity or metadata.get('run_id') != run_id
                or metadata.get('identity_sha256') != identity_sha256):
            raise ValueError('Existing local run metadata does not match')
    else:
        if database.exists() or Path(str(database) + '-wal').exists():
            raise ValueError('Refusing a database without matching local run metadata')
        metadata = {
            'schema_version': 1, 'coordinate_kind': 'local_ndt_map', 'run_id': run_id,
            'identity_sha256': identity_sha256, 'identity': identity,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'directory': str(directory), 'database': str(database),
            'resume_policy': 'only_this_run_after_worker_restart; never_previous_run_or_fixposition',
            'unobserved_cells_are_measurements': False,
        }
        temporary = directory / f'.session-{uuid.uuid4().hex}.tmp'
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                stream.write(canonical_json(metadata) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)
    return metadata


def snapshot_database(source, destination):
    """Consistent SQLite backup including committed WAL data; never overwrite an export."""
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as original:
        with sqlite3.connect(destination) as backup:
            original.backup(backup)
            if backup.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Elevation database snapshot failed integrity check')
    return destination
