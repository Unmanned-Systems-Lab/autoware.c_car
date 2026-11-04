"""Read-only, bounded snapshots of the committed global elevation tile database."""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import threading
import zlib

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass
class Snapshot:
    tile_x: int
    tile_y: int
    height: np.ndarray
    variance: np.ndarray
    observation_count: np.ndarray
    dependencies: dict
    commit_seq: int
    metadata: dict


class SourceReader:
    """Never migrates, writes, checkpoints, or scans all source rows in memory."""
    def __init__(self, context, max_cached_tiles=16):
        self.context = json.loads(canonical(context))
        self.path = Path(context['database']).expanduser().resolve(strict=True)
        stat = self.path.stat()
        self.file_identity = stat.st_dev, stat.st_ino
        self.identity = context.get('identity', {})
        self.identity_sha256 = context.get('identity_sha256') or hashlib.sha256(canonical(self.identity).encode()).hexdigest()
        self.conn = sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True,
                                    timeout=2, check_same_thread=False, isolation_level=None)
        self.conn.execute('PRAGMA query_only=ON')
        self.conn.execute('PRAGMA busy_timeout=2000')
        self.lock = threading.RLock()
        self.cache = OrderedDict()
        self.max_cached_tiles = max(1, int(max_cached_tiles))
        columns = {row[1] for row in self.conn.execute('PRAGMA table_info(tiles)')}
        if 'commit_seq' not in columns:
            raise ValueError('Source owner must install indexed commit tracking before reading')
        if self.conn.execute('SELECT 1 FROM tiles INDEXED BY tiles_commit_seq_idx WHERE commit_seq<=0 LIMIT 1').fetchone():
            raise ValueError('Source owner must backfill all legacy tile commit sequences')
        self.metadata = self._metadata()
        self._validate_metadata(self.metadata)
        self.geometry_contract = self._geometry(self.metadata)
        self.resolution = float(self.metadata['resolution'])
        self.tile_size = int(self.metadata['tile_size_cells'])
        self.phase_x, self.phase_y = float(self.metadata['phase_x']), float(self.metadata['phase_y'])
        parameters = self.identity.get('mapping_parameters', {})
        self.publish_prior = parameters.get('global_map_publish_unobserved_prior', False) is True
        self.prior_bounds = self.metadata.get('prior_bounds') if self.publish_prior else None
        if self.publish_prior and self.prior_bounds is None:
            raise ValueError('Global traversability permits only a fixed bounded height prior')
        self.prior_tile_rectangle = self._prior_rectangle()

    @staticmethod
    def _geometry(metadata):
        return {key: metadata.get(key) for key in (
            'schema_version', 'resolution', 'tile_size_cells', 'map_frame', 'global_layers',
            'phase_x', 'phase_y', 'initial_elevation', 'initial_variance', 'prior_bounds')}

    def _metadata(self):
        return {key: json.loads(value) for key, value in self.conn.execute('SELECT key,value FROM metadata')}

    def _validate_metadata(self, metadata):
        if (metadata.get('schema_version') != 2 or metadata.get('map_frame') != 'map'
                or metadata.get('tile_size_cells') != 128
                or not math.isclose(float(metadata.get('resolution', 0)), .04, abs_tol=1e-9)
                or metadata.get('phase_x') not in (0., .5) or metadata.get('phase_y') not in (0., .5)):
            raise ValueError('Unsupported source schema, lattice, frame, tile size, or resolution')
        if metadata.get('global_layers') != ['elevation', 'variance', 'observation_count', 'last_update_revision']:
            raise ValueError('Unsupported source payload layer contract')
        for key in ('initial_elevation', 'initial_variance'):
            if not math.isfinite(float(metadata.get(key, float('nan')))):
                raise ValueError('Source initial height/variance metadata is not finite')
        parameters = self.identity.get('mapping_parameters', {})
        for parameter, field in [('resolution', 'resolution'), ('global_map_tile_size_cells', 'tile_size_cells'),
                                 ('map_frame', 'map_frame'), ('global_map_initial_elevation', 'initial_elevation'),
                                 ('global_map_initial_variance', 'initial_variance')]:
            if parameter in parameters and parameters[parameter] != metadata[field]:
                raise ValueError('Source metadata differs from the session: ' + field)
        bounds = metadata.get('prior_bounds')
        if bounds is not None:
            if (not isinstance(bounds, list) or len(bounds) != 4
                    or not all(math.isfinite(float(value)) for value in bounds)
                    or not bounds[0] < bounds[2] or not bounds[1] < bounds[3]):
                raise ValueError('Invalid source prior rectangle')
            if parameters.get('global_map_publish_unobserved_prior'):
                expected = [parameters.get('global_map_prior_' + name)
                            for name in ('min_x', 'min_y', 'max_x', 'max_y')]
                if expected != bounds:
                    raise ValueError('Source prior rectangle differs from the session')

    @contextmanager
    def transaction(self):
        with self.lock:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self.file_identity:
                raise ValueError('Source database was replaced; start a new engine/session')
            self.conn.execute('BEGIN')
            try:
                clock = self.conn.execute('SELECT seq FROM tile_commit_clock WHERE id=1').fetchone()
                if clock is None:
                    raise ValueError('Source commit clock is not installed')
                metadata = self._metadata()
                if self._geometry(metadata) != self.geometry_contract:
                    raise ValueError('Source geometry/height prior changed while reading')
                self.metadata = metadata
                yield int(clock[0]), metadata
            finally:
                self.conn.execute('ROLLBACK')

    def poll_changes(self, after=0, limit=64):
        limit = max(1, min(int(limit), 4096))
        with self.transaction() as (clock, _):
            if clock < int(after):
                raise ValueError('Source commit clock regressed; start a new source session')
            rows = self.conn.execute(
                'SELECT tile_x,tile_y,revision,commit_seq FROM tiles INDEXED BY tiles_commit_seq_idx '
                'WHERE commit_seq>? AND level=0 ORDER BY commit_seq LIMIT ?', (int(after), limit)).fetchall()
            changes = [{'tile_x': int(x), 'tile_y': int(y), 'revision': int(revision), 'commit_seq': int(seq)}
                       for x, y, revision, seq in rows]
            cursor = int(rows[-1][3]) if len(rows) == limit else clock
            return changes, cursor, clock

    def _prior_rectangle(self):
        if self.prior_bounds is None:
            return None
        xmin, ymin, xmax, ymax = self.prior_bounds
        ix0 = math.ceil(xmin/self.resolution - self.phase_x - 1e-9)
        iy0 = math.ceil(ymin/self.resolution - self.phase_y - 1e-9)
        ix1 = math.ceil(xmax/self.resolution - self.phase_x - 1e-9) - 1
        iy1 = math.ceil(ymax/self.resolution - self.phase_y - 1e-9) - 1
        if ix0 > ix1 or iy0 > iy1:
            return None
        return ix0//128, iy0//128, ix1//128, iy1//128

    def neighborhood_sequences(self, tile_x, tile_y):
        """One snapshot, 25 PK metadata reads; no payload/bootstrap scan for the ROI."""
        x, y = int(tile_x), int(tile_y)
        with self.transaction() as (clock, _):
            result = {}
            for yy in range(y-2, y+3):
                for xx in range(x-2, x+3):
                    row = self.conn.execute(
                        'SELECT commit_seq FROM tiles WHERE level=0 AND tile_x=? AND tile_y=?',
                        (xx, yy)).fetchone()
                    result[f'{xx},{yy}'] = int(row[0]) if row else 0
            return result, clock

    def prior_page(self, cursor, limit):
        """Enumerate a virtual seed rectangle without materializing all tile keys."""
        rectangle = self.prior_tile_rectangle
        if rectangle is None:
            return [], int(cursor), True
        x0, y0, x1, y1 = rectangle
        width, total = x1-x0+1, (x1-x0+1)*(y1-y0+1)
        end = min(total, int(cursor) + max(1, int(limit)))
        keys = [(x0 + index % width, y0 + index//width) for index in range(int(cursor), end)]
        return keys, end, end == total

    def _decode(self, key, seq, payload):
        cache_key = (*key, seq)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]
        expected = 4*128*128*4
        decoder = zlib.decompressobj()
        raw = decoder.decompress(payload, expected+1)
        if len(raw) != expected or not decoder.eof or decoder.unused_data:
            raise ValueError('Corrupt source tile payload: ' + str(key))
        data = np.frombuffer(raw, dtype='<f4').reshape(4, 128, 128).copy()
        if (not np.isfinite(data[2]).all() or (data[2] < 0).any()
                or not np.isfinite(data[3]).all() or (data[3] < 0).any()):
            raise ValueError('Corrupt source observation/revision layers')
        observed = data[2] > 0
        if not np.isfinite(data[0, observed]).all() or not np.isfinite(data[1, observed]).all():
            raise ValueError('Corrupt observed source height/variance')
        data.setflags(write=False)
        self.cache[cache_key] = data
        while len(self.cache) > self.max_cached_tiles:
            self.cache.popitem(last=False)
        return data

    def snapshot(self, tile_x, tile_y, halo=25):
        if halo != 25:
            raise ValueError('The model source contract requires halo=25')
        tx, ty = int(tile_x), int(tile_y)
        size = 128 + 2*halo
        ix0, iy0 = tx*128-halo, ty*128-halo
        ix1, iy1 = ix0+size, iy0+size
        xs = (np.arange(ix0, ix1)+self.phase_x)*self.resolution
        ys = (np.arange(iy0, iy1)+self.phase_y)*self.resolution
        height = np.full((size, size), np.nan, np.float32)
        variance = np.full((size, size), np.nan, np.float32)
        count = np.zeros((size, size), np.float32)
        dependencies = {}
        with self.transaction() as (clock, metadata):
            # Both coordinates must be equality predicates on the WITHOUT ROWID
            # PK. An X range with Y filtering scans arbitrarily long route columns.
            rows = []
            for y in range(iy0//128, (iy1-1)//128+1):
                for x in range(ix0//128, (ix1-1)//128+1):
                    row = self.conn.execute(
                        'SELECT tile_x,tile_y,commit_seq,payload FROM tiles '
                        'WHERE level=0 AND tile_x=? AND tile_y=?', (x, y)).fetchone()
                    if row is not None:
                        rows.append(row)
            decoded = {(int(x), int(y)): (int(seq), self._decode((int(x), int(y)), int(seq), payload))
                       for x, y, seq, payload in rows}
            for y in range(iy0//128, (iy1-1)//128+1):
                for x in range(ix0//128, (ix1-1)//128+1):
                    item = decoded.get((x, y))
                    dependencies[f'{x},{y}'] = item[0] if item else 0
                    if item is None:
                        continue
                    ax, ay, bx, by = max(ix0, x*128), max(iy0, y*128), min(ix1, (x+1)*128), min(iy1, (y+1)*128)
                    src = item[1][:, ay-y*128:by-y*128, ax-x*128:bx-x*128]
                    target = (slice(ay-iy0, by-iy0), slice(ax-ix0, bx-ix0))
                    observed = src[2] > 0
                    height[target] = np.where(observed, src[0], np.nan)
                    variance[target] = np.where(observed, src[1], np.nan)
                    count[target] = src[2]
            if self.prior_bounds is not None:
                xmin, ymin, xmax, ymax = self.prior_bounds
                prior = ((ys[:, None] >= ymin) & (ys[:, None] < ymax)
                         & (xs[None, :] >= xmin) & (xs[None, :] < xmax) & (count <= 0))
                height[prior], variance[prior] = metadata['initial_elevation'], metadata['initial_variance']
            return Snapshot(tx, ty, height, variance, count, dependencies, clock, dict(metadata))

    def dependencies_now(self, dependencies):
        with self.transaction() as (clock, _):
            result = {}
            for key in dependencies:
                x, y = map(int, key.split(','))
                row = self.conn.execute('SELECT commit_seq FROM tiles WHERE level=0 AND tile_x=? AND tile_y=?', (x, y)).fetchone()
                result[key] = int(row[0]) if row else 0
            return result, clock

    def close(self):
        self.conn.close()
