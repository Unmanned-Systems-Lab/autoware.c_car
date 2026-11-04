"""Durable deduplicated work, bounded tile cache, and an incrementally updated LOD pyramid."""
from collections import OrderedDict
from pathlib import Path
import sqlite3
import threading
import time
import zlib

import numpy as np

from .source import canonical

LAYERS = ('elevation', 'traversability', *(f'ht_dir_{index}' for index in range(8)),
          'valid', 'support_fraction', 'observed_fraction', 'variance', 'observation_count')
INDEX = {name: index for index, name in enumerate(LAYERS)}
TILE = 128


def tile_cells(level):
    return 128 if level == 0 else 64 if level == 1 else 32


def lod_key(x, y, level):
    divisor = 2**max(0, level-2)
    return x//divisor, y//divisor


def empty_tile(cells=128):
    result = np.full((len(LAYERS), cells, cells), np.nan, np.float32)
    for name in ('valid', 'support_fraction', 'observed_fraction', 'observation_count'):
        result[INDEX[name]] = 0
    return result


def reduce_tile(data):
    """Bounded overview: min known predictions, mean coverage, max known variance."""
    half = data.shape[1]//2
    blocks = data.reshape(len(LAYERS), half, 2, half, 2).transpose(0, 1, 3, 2, 4).reshape(len(LAYERS), half, half, 4)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=-1)
    sums = np.where(finite, blocks, 0).sum(axis=-1, dtype=np.float64)
    output = np.full((len(LAYERS), half, half), np.nan, np.float32)
    np.divide(sums, count, out=output, where=count > 0, casting='unsafe')
    for name in ('traversability', *(f'ht_dir_{i}' for i in range(8))):
        index = INDEX[name]
        minimum = np.where(finite[index], blocks[index], np.inf).min(axis=-1)
        output[index] = np.where(count[index] > 0, minimum, np.nan)
    index = INDEX['variance']
    maximum = np.where(finite[index], blocks[index], -np.inf).max(axis=-1)
    output[index] = np.where(count[index] > 0, maximum, np.nan)
    return output


class ResultStore:
    def __init__(self, filename, binding, max_cached_tiles=16, max_lod=24):
        self.path = Path(filename).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=2, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pending(
                tile_x INTEGER,tile_y INTEGER,generation INTEGER NOT NULL,wanted_seq INTEGER NOT NULL,
                first_order INTEGER NOT NULL,PRIMARY KEY(tile_x,tile_y)) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS pending_age_idx ON pending(first_order);
            CREATE TABLE IF NOT EXISTS results(
                level INTEGER,tile_x INTEGER,tile_y INTEGER,payload BLOB NOT NULL,
                computed_at REAL NOT NULL,stale INTEGER NOT NULL,source_seq INTEGER NOT NULL,
                dependencies TEXT NOT NULL,input_sha256 TEXT NOT NULL DEFAULT '',
                PRIMARY KEY(level,tile_x,tile_y)) WITHOUT ROWID;
        ''')
        self.lock = threading.RLock()
        if 'input_sha256' not in {row[1] for row in self.conn.execute('PRAGMA table_info(results)')}:
            self.conn.execute("ALTER TABLE results ADD COLUMN input_sha256 TEXT NOT NULL DEFAULT ''")
        self.cache = OrderedDict()
        self.max_cached_tiles, self.max_lod = max(1, int(max_cached_tiles)), int(max_lod)
        old = self.get_state('binding')
        if old is not None and old != binding:
            self.conn.close()
            raise ValueError('Result database belongs to a different source session/model/configuration')
        with self.conn:
            self._set('binding', binding)
            for key, value in {'source_cursor': 0, 'prior_cursor': 0, 'prior_done': False,
                               'pending_count': 0, 'queue_order': 0, 'result_tiles': 0,
                               'processed_jobs': 0, 'extent': None}.items():
                if self.get_state(key) is None:
                    self._set(key, value)

    def get_state(self, key):
        import json
        row = self.conn.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _set(self, key, value):
        self.conn.execute('INSERT INTO metadata VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                          (key, canonical(value)))

    def _enqueue(self, x, y, wanted_seq):
        old = self.conn.execute('SELECT generation,wanted_seq,first_order FROM pending WHERE tile_x=? AND tile_y=?', (x, y)).fetchone()
        if old:
            self.conn.execute('UPDATE pending SET generation=?,wanted_seq=? WHERE tile_x=? AND tile_y=?',
                              (old[0]+1, max(old[1], int(wanted_seq)), x, y))
        else:
            order = self.get_state('queue_order')+1
            self._set('queue_order', order)
            self._set('pending_count', self.get_state('pending_count')+1)
            self.conn.execute('INSERT INTO pending VALUES(?,?,?,?,?)', (x, y, 1, int(wanted_seq), order))
        for level in range(self.max_lod+1):
            px, py = lod_key(x, y, level)
            self.conn.execute('UPDATE results SET stale=1 WHERE level=? AND tile_x=? AND tile_y=?', (level, px, py))
            self.cache.pop((level, px, py), None)

    def enqueue_changes(self, keys, wanted_seq, *, cursor=None, prior_cursor=None, prior_done=None):
        with self.lock, self.conn:
            for x, y in set(keys):
                self._enqueue(int(x), int(y), wanted_seq)
            if cursor is not None:
                self._set('source_cursor', int(cursor))
            if prior_cursor is not None:
                self._set('prior_cursor', int(prior_cursor))
            if prior_done is not None:
                self._set('prior_done', bool(prior_done))

    def choose_job(self, near_tile=None, force_oldest=False):
        with self.lock:
            if near_tile is not None and not force_oldest:
                x, y = near_tile
                candidates = []
                # Three output tiles per axis cover the default 10m crop. The
                # wider 5x5 set belongs to source input halos, not this queue.
                for yy in range(y-1, y+2):
                    for xx in range(x-1, x+2):
                        row = self.conn.execute('SELECT tile_x,tile_y,generation,wanted_seq,first_order FROM pending '
                                                'WHERE tile_x=? AND tile_y=?', (xx, yy)).fetchone()
                        if row:
                            candidates.append(row)
                if candidates:
                    return min(candidates, key=lambda row: (row[4], (row[0]-x)**2+(row[1]-y)**2))
            return self.conn.execute('SELECT tile_x,tile_y,generation,wanted_seq,first_order '
                                     'FROM pending ORDER BY first_order LIMIT 1').fetchone()

    def _row(self, level, x, y):
        return self.conn.execute('SELECT payload,computed_at,stale,source_seq,dependencies,input_sha256 FROM results '
                                 'WHERE level=? AND tile_x=? AND tile_y=?', (level, x, y)).fetchone()

    def tile(self, level, x, y):
        with self.lock:
            key = int(level), int(x), int(y)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            row = self._row(*key)
            if row is None:
                return None
            cells = tile_cells(level)
            expected = len(LAYERS)*cells*cells*4
            decoder = zlib.decompressobj()
            raw = decoder.decompress(row[0], expected+1)
            if len(raw) != expected or not decoder.eof or decoder.unused_data:
                raise ValueError('Corrupt result tile')
            data = np.frombuffer(raw, dtype='<f4').reshape(len(LAYERS), cells, cells).copy()
            metadata = {'computed_at': row[1], 'source_stale': bool(row[2]), 'source_seq': row[3],
                        'dependencies': row[4], 'input_sha256': row[5]}
            self._cache(key, data, metadata)
            return data, metadata

    def _cache(self, key, data, metadata):
        data.setflags(write=False)
        self.cache[key] = data, metadata
        self.cache.move_to_end(key)
        while len(self.cache) > self.max_cached_tiles:
            self.cache.popitem(last=False)

    def _write(self, level, x, y, data, computed_at, stale, source_seq, dependencies, input_sha256=''):
        blob = zlib.compress(np.asarray(data, dtype='<f4', order='C').tobytes(), level=1)
        self.conn.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(level,tile_x,tile_y) '
                          'DO UPDATE SET payload=excluded.payload,computed_at=excluded.computed_at,stale=excluded.stale,'
                          'source_seq=excluded.source_seq,dependencies=excluded.dependencies,input_sha256=excluded.input_sha256',
                          (level, x, y, sqlite3.Binary(blob), computed_at, int(stale), source_seq, canonical(dependencies), input_sha256))
        self._cache((level, x, y), data.copy(), {'computed_at': computed_at, 'source_stale': bool(stale),
                                               'source_seq': source_seq, 'dependencies': canonical(dependencies),
                                               'input_sha256': input_sha256})

    def complete(self, job, data, snapshot, *, stale=False, current_seq=None, input_sha256=''):
        x, y, generation, _, _ = job
        now = time.time()
        with self.lock, self.conn:
            old = self._row(0, x, y)
            previous = self.tile(0, x, y) if old is not None else None
            old_known = previous is not None and np.isfinite(previous[0][INDEX['elevation']]).any()
            if old is None:
                self._set('result_tiles', self.get_state('result_tiles')+1)
            # A completed snapshot is useful even while its source keeps changing.
            pending = self.conn.execute('SELECT generation FROM pending WHERE tile_x=? AND tile_y=?', (x, y)).fetchone()
            dirty_again = pending is not None and pending[0] != generation
            self._write(0, x, y, data, now, stale or dirty_again, snapshot.commit_seq, snapshot.dependencies, input_sha256)
            if stale and not dirty_again:
                self._enqueue(x, y, current_seq if current_seq is not None else snapshot.commit_seq)
            if not stale and not dirty_again:
                removed = self.conn.execute('DELETE FROM pending WHERE tile_x=? AND tile_y=? AND generation=?', (x, y, generation)).rowcount
                self._set('pending_count', self.get_state('pending_count')-removed)
            else:
                # A hot tile gets its new snapshot, then yields FIFO age to jobs
                # which have never been served. Keeping its original age starves far work.
                order = self.get_state('queue_order')+1
                self._set('queue_order', order)
                self.conn.execute('UPDATE pending SET first_order=? WHERE tile_x=? AND tile_y=?', (order, x, y))
            valid_height = np.isfinite(data[INDEX['elevation']])
            if valid_height.any():
                yy, xx = np.nonzero(valid_height)
                bounds = [x*128+int(xx.min()), y*128+int(yy.min()), x*128+int(xx.max()), y*128+int(yy.max())]
                extent = self.get_state('extent')
                if extent is not None:
                    bounds = [min(extent[0], bounds[0]), min(extent[1], bounds[1]), max(extent[2], bounds[2]), max(extent[3], bounds[3])]
                self._set('extent', bounds)
            self._set('processed_jobs', self.get_state('processed_jobs')+1)
            if not valid_height.any() and not old_known:
                return
            # Each changed child updates only its quadrant in every sparse ancestor.
            child, cx, cy = data, x, y
            for level in range(1, self.max_lod+1):
                if level <= 2:
                    summary = reduce_tile(child)
                    self._write(level, cx, cy, summary, now, stale or dirty_again, snapshot.commit_seq, {})
                    child = summary
                    continue
                px, py = cx//2, cy//2
                previous = self.tile(level, px, py)
                parent = previous[0].copy() if previous else empty_tile(32)
                oy, ox = (cy % 2)*16, (cx % 2)*16
                parent[:, oy:oy+16, ox:ox+16] = reduce_tile(child)
                children = []
                for yy in (py*2, py*2+1):
                    for xx in (px*2, px*2+1):
                        row = self.conn.execute('SELECT computed_at,stale,source_seq FROM results '
                                                'WHERE level=? AND tile_x=? AND tile_y=?', (level-1, xx, yy)).fetchone()
                        if row is not None:
                            children.append(row)
                stamp = min(row[0] for row in children)
                any_stale, seq = any(row[1] for row in children), max(row[2] for row in children)
                self._write(level, px, py, parent, stamp, any_stale, seq, {})
                child, cx, cy = parent, px, py

    def stats(self):
        with self.lock:
            return {key: self.get_state(key) for key in ('source_cursor', 'prior_cursor', 'prior_done',
                    'pending_count', 'result_tiles', 'processed_jobs', 'extent')}

    def close(self):
        self.conn.close()
