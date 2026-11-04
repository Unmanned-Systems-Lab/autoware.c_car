"""Pure SQLite/model-fixture tests; no DDS, GPU, native mapper, or robot is started."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from scout_global_traversability.engine import Engine, window_fraction
from scout_global_traversability.source import SourceReader
from scout_global_traversability.store import INDEX, ResultStore


class Predictor:
    def __init__(self):
        self.calls, self.callback = 0, None
    def predict(self, height):
        self.calls += 1
        core = height[25:153, 25:153]
        support = window_fraction(np.isfinite(height))
        probabilities = np.stack([np.clip(core+1+.01*i, 0, 1) for i in range(8)]).astype(np.float32)
        if self.callback:
            self.callback()
        return {'probabilities': probabilities, 'valid': np.isfinite(core) & (support >= .7),
                'support_fraction': support}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.db = self.path/'source.sqlite3'
        self.writer = sqlite3.connect(self.db)
        self.writer.execute('PRAGMA journal_mode=WAL')
        self.writer.executescript('''
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE tiles(level INTEGER,tile_x INTEGER,tile_y INTEGER,revision INTEGER,
                payload BLOB NOT NULL,commit_seq INTEGER NOT NULL,PRIMARY KEY(level,tile_x,tile_y)) WITHOUT ROWID;
            CREATE INDEX tiles_commit_seq_idx ON tiles(commit_seq);
            CREATE TABLE tile_commit_clock(id INTEGER PRIMARY KEY,seq INTEGER NOT NULL);
            INSERT INTO tile_commit_clock VALUES(1,0);
        ''')
        self.metadata = {'schema_version': 2, 'resolution': .04, 'tile_size_cells': 128, 'map_frame': 'map',
                         'phase_x': .5, 'phase_y': .5, 'initial_elevation': -.8, 'initial_variance': 10000.,
                         'prior_bounds': [-5., -5., 5., 5.],
                         'global_layers': ['elevation', 'variance', 'observation_count', 'last_update_revision']}
        self.write_metadata()
        self.context = {'database': str(self.db), 'identity_sha256': 'a'*64,
                        'identity': {'mapping_parameters': {'global_map_publish_unobserved_prior': True,
                            'global_map_prior_min_x': -5., 'global_map_prior_min_y': -5.,
                            'global_map_prior_max_x': 5., 'global_map_prior_max_y': 5.,
                            'global_map_initial_elevation': -.8, 'global_map_initial_variance': 10000.,
                            'resolution': .04, 'map_frame': 'map'}}}
        self.resources = []

    def tearDown(self):
        for item in self.resources:
            item.close()
        self.writer.close()
        self.temp.cleanup()

    def write_metadata(self):
        with self.writer:
            self.writer.executemany('INSERT OR REPLACE INTO metadata VALUES(?,?)',
                                    [(key, json.dumps(value)) for key, value in self.metadata.items()])

    def no_prior(self):
        self.metadata['prior_bounds'] = None
        self.write_metadata()
        self.context['identity']['mapping_parameters']['global_map_publish_unobserved_prior'] = False

    def put(self, x, y, height=.2, count=1, variance=.01):
        data = np.empty((4, 128, 128), np.float32)
        data[0], data[1], data[2], data[3] = height, variance, count, 1
        with self.writer:
            seq = self.writer.execute('SELECT seq FROM tile_commit_clock WHERE id=1').fetchone()[0]+1
            self.writer.execute('UPDATE tile_commit_clock SET seq=? WHERE id=1', (seq,))
            self.writer.execute('INSERT OR REPLACE INTO tiles VALUES(0,?,?,?,?,?)',
                                (x, y, seq, sqlite3.Binary(zlib.compress(data.astype('<f4').tobytes())), seq))
        return seq

    def reader(self):
        result = SourceReader(self.context, max_cached_tiles=2)
        self.resources.append(result)
        return result

    def engine(self, result='result.sqlite3', **kwargs):
        model = kwargs.pop('model_sha256', 'b'*64)
        predictor = kwargs.pop('predictor', Predictor())
        result = Engine(self.context, predictor, self.path/result, model_sha256=model,
                        max_cached_tiles=3, **kwargs)
        self.resources.append(result)
        return result

    def process_tile(self, engine, key, near_xy, *, allow_inference=True):
        # FIFO near scheduling may first complete older neighboring output tiles.
        for _ in range(32):
            result = engine.process_one(near_xy, allow_inference=allow_inference)
            self.assertIsNotNone(result)
            if (result['tile_x'], result['tile_y']) == key:
                return result
            self.assertNotEqual(result['state'], 'inference_budget_wait')
        self.fail('Target tile did not receive its fair queue turn')

    def test_raw_count_zero_never_exposes_unbounded_default_height(self):
        self.put(0, 0, height=0., count=0, variance=10000.)
        snapshot = self.reader().snapshot(0, 0)
        self.assertEqual(snapshot.height.shape, (178, 178))
        self.assertAlmostEqual(float(snapshot.height[25, 25]), -.8, places=6)
        self.assertTrue(np.isnan(snapshot.height[25+127, 25+127]))  # Center at 5.10m, outside fixed prior.
        self.assertEqual(snapshot.observation_count.sum(), 0)
        self.assertEqual(snapshot.variance[25, 25], 10000.)

    def test_read_only_source_and_indexed_bounded_keyset(self):
        self.no_prior()
        for x in range(7):
            self.put(x, 0)
        reader = self.reader()
        with self.assertRaises(sqlite3.OperationalError):
            reader.conn.execute('DELETE FROM tiles')
        first, cursor, clock = reader.poll_changes(0, 2)
        self.assertEqual([item['commit_seq'] for item in first], [1, 2])
        self.assertEqual((cursor, clock), (2, 7))
        self.put(0, 0, height=.3)
        second, cursor, _ = reader.poll_changes(cursor, 2)
        self.assertEqual([item['commit_seq'] for item in second], [3, 4])
        remaining, cursor, _ = reader.poll_changes(cursor, 10)
        self.assertEqual([item['commit_seq'] for item in remaining], [5, 6, 7, 8])
        plan = reader.conn.execute('EXPLAIN QUERY PLAN SELECT tile_x FROM tiles INDEXED BY tiles_commit_seq_idx '
                                   'WHERE commit_seq>? ORDER BY commit_seq LIMIT ?', (0, 2)).fetchall()
        self.assertIn('tiles_commit_seq_idx', str(plan))

    def test_one_wal_snapshot_for_all_neighbors_and_dependency_watermark(self):
        self.no_prior()
        self.put(0, 0, height=.1)
        self.put(1, 0, height=.2)
        reader = self.reader()
        decode = reader._decode
        fired = []
        def concurrent_commit(*args):
            if not fired:
                fired.append(True)
                self.put(1, 0, height=.9)
            return decode(*args)
        with patch.object(reader, '_decode', side_effect=concurrent_commit):
            snapshot = reader.snapshot(0, 0)
        self.assertEqual(snapshot.commit_seq, 2)
        self.assertAlmostEqual(float(snapshot.height[25, 153]), .2, places=6)
        self.assertEqual(snapshot.dependencies['1,0'], 2)
        now, clock = reader.dependencies_now(snapshot.dependencies)
        self.assertEqual(clock, 3)
        self.assertEqual(now['1,0'], 3)

    def test_missing_backfill_and_regressed_commit_clock_are_not_silently_reused(self):
        self.put(0, 0)
        with self.writer:
            self.writer.execute('UPDATE tiles SET commit_seq=0')
        with self.assertRaises(ValueError):
            self.reader()
        with self.writer:
            self.writer.execute('UPDATE tiles SET commit_seq=1')
        reader = self.reader()
        with self.assertRaises(ValueError):
            reader.poll_changes(after=2)

    def test_halo_queries_use_full_primary_key_not_unbounded_route_columns(self):
        self.put(0, 0)
        reader = self.reader()
        traces = []
        reader.conn.set_trace_callback(traces.append)
        reader.snapshot(0, 0)
        payload_queries = [query for query in traces if 'SELECT tile_x,tile_y,commit_seq,payload' in query]
        self.assertEqual(len(payload_queries), 9)
        self.assertTrue(all('BETWEEN' not in query.upper() for query in payload_queries))
        plan = reader.conn.execute('EXPLAIN QUERY PLAN SELECT payload FROM tiles '
                                   'WHERE level=0 AND tile_x=? AND tile_y=?', (0, 0)).fetchall()
        self.assertIn('PRIMARY KEY', str(plan))
        self.assertIn('tile_y=?', str(plan))

    def test_virtual_prior_without_any_source_rows_really_enters_model(self):
        engine = self.engine()
        report = engine.poll_changes(64)
        self.assertEqual(report['source_rows'], 0)
        self.assertEqual(report['dirty_tiles'], 4)
        for _ in range(4):
            self.assertTrue(engine.process_one((0, 0))['model_called'])
        self.assertEqual(engine.predictor.calls, 4)
        local = engine.local_window((0, 0))
        self.assertEqual(local['shape'], (250, 250))
        self.assertEqual(local['origin_xy'], [-5., -5.])
        known = np.isfinite(local['layers']['traversability'])
        self.assertTrue(known.any())
        self.assertTrue(np.allclose(local['layers']['traversability'][known], .2))
        self.assertTrue(np.all(local['layers']['observation_count'] == 0))
        self.assertTrue(np.all(local['layers']['observed_fraction'] == 0))
        self.assertFalse(np.all(local['layers']['valid'] == 1))  # Padding does not invent support.

    def test_unknown_centers_skip_gpu_even_with_finite_prior_in_halo(self):
        engine = self.engine()
        engine.store.enqueue_changes([(1, 1)], 0)
        job = engine.process_one((6., 6.), allow_inference=False)
        self.assertEqual(job['state'], 'skip_unknown')
        self.assertFalse(job['model_called'])
        self.assertEqual(engine.predictor.calls, 0)

    def test_exact_height_reuses_probability_but_refreshes_observation_quality(self):
        self.no_prior()
        self.put(0, 0, height=.1, count=1)
        engine = self.engine()
        engine.poll_changes()
        self.assertTrue(self.process_tile(engine, (0, 0), (1, 1))['model_called'])
        self.put(0, 0, height=.1, count=7, variance=.003)
        engine.poll_changes()
        result = self.process_tile(engine, (0, 0), (1, 1), allow_inference=False)
        self.assertEqual(result['state'], 'reused_exact_height')
        self.assertEqual(engine.predictor.calls, 1)
        tile = engine.store.tile(0, 0, 0)[0]
        self.assertTrue(np.all(tile[INDEX['observation_count']] == 7))
        self.assertTrue(np.allclose(tile[INDEX['variance']], .003))
        self.put(0, 0, height=.11, count=8)
        engine.poll_changes()
        result = self.process_tile(engine, (0, 0), (1, 1), allow_inference=False)
        self.assertEqual(result['state'], 'inference_budget_wait')
        self.assertEqual(engine.predictor.calls, 1)

    def test_changes_during_inference_save_snapshot_and_keep_latest_dirty(self):
        self.no_prior()
        self.put(0, 0, height=-.8)
        predictor = Predictor()
        engine = self.engine(predictor=predictor)
        engine.poll_changes()
        predictor.callback = lambda: self.put(0, 0, height=-.6)
        result = self.process_tile(engine, (0, 0), (1, 1))
        self.assertTrue(result['source_stale'])
        data, metadata = engine.store.tile(0, 0, 0)
        self.assertTrue(metadata['source_stale'])
        self.assertTrue(np.allclose(data[INDEX['elevation']], -.8))
        self.assertGreater(engine.stats()['pending_tiles'], 0)
        predictor.callback = None
        result = self.process_tile(engine, (0, 0), (1, 1))
        self.assertFalse(result['source_stale'])
        data, metadata = engine.store.tile(0, 0, 0)
        self.assertFalse(metadata['source_stale'])
        self.assertTrue(np.allclose(data[INDEX['elevation']], -.6))

    def test_durable_pending_dedup_and_binding_rejects_different_model(self):
        engine = self.engine()
        engine.poll_changes(1)
        count = engine.stats()['pending_tiles']
        key = engine.store.choose_job()[:2]
        engine.store.enqueue_changes([key]*20, 10)
        self.assertEqual(engine.stats()['pending_tiles'], count)
        engine.close()
        self.resources.remove(engine)
        reopened = self.engine()
        self.assertEqual(reopened.stats()['pending_tiles'], count)
        with self.assertRaises(ValueError):
            self.engine(model_sha256='c'*64)

    def test_hot_near_work_yields_fifo_age_and_far_work_cannot_starve(self):
        self.no_prior()
        self.put(0, 0)
        engine = self.engine()
        engine.store.enqueue_changes([(100, 100)], 1)
        engine.store.enqueue_changes([(0, 0)], 1)
        original = engine.predictor.predict
        def keep_hot(height):
            prediction = original(height)
            engine.store.enqueue_changes([(0, 0)], 99)
            return prediction
        engine.predictor.predict = keep_hot
        result = engine.process_one((0, 0))
        self.assertTrue(result['source_stale'])
        oldest = engine.store.choose_job((0, 0), force_oldest=True)
        self.assertEqual(oldest[:2], (100, 100))

    def test_sparse_overview_never_materializes_large_map_and_lru_is_bounded(self):
        self.no_prior()
        self.put(0, 0, height=.1)
        self.put(1000, 1000, height=.2)
        engine = self.engine()
        engine.poll_changes()
        self.process_tile(engine, (0, 0), (1, 1))
        self.process_tile(engine, (1000, 1000), (5121, 5121))
        overview = engine.overview(512)
        self.assertLessEqual(max(overview['shape']), 512)
        self.assertGreater(overview['quality']['lod'], 0)
        self.assertGreater(np.isfinite(overview['layers']['elevation']).sum(), 0)
        self.assertLessEqual(engine.stats()['source_cached_tiles'], 3)
        self.assertLessEqual(engine.stats()['result_cached_tiles'], 3)
        with self.assertRaises(ValueError):
            engine.overview(513)
        with self.assertRaises(ValueError):
            engine.local_window((0, 0), 100000)

    def test_gpu_budget_wait_cannot_advance_past_reserved_far_turn(self):
        self.no_prior()
        engine = self.engine()
        for key in [(0, 0), (1, 0), (2, 0)]:
            engine.store.enqueue_changes([key], 0)
            self.assertEqual(engine.process_one((0, 0), allow_inference=False)['state'], 'skip_unknown')
        self.put(100, 100)
        self.put(0, 0)
        engine.store.enqueue_changes([(100, 100)], 2)
        engine.store.enqueue_changes([(0, 0)], 2)
        for _ in range(5):
            result = engine.process_one((0, 0), allow_inference=False)
            self.assertEqual(result['state'], 'inference_budget_wait')
            self.assertEqual((result['tile_x'], result['tile_y']), (100, 100))
        result = engine.process_one((0, 0), allow_inference=True)
        self.assertTrue(result['model_called'])
        self.assertEqual((result['tile_x'], result['tile_y']), (100, 100))

    def test_cold_large_source_near_latest_is_queued_before_global_bootstrap(self):
        self.no_prior()
        for x in range(200):
            self.put(x, -1000, height=.1)
        self.put(10000, 10000, height=.42)
        engine = self.engine()
        near = (10000*5.12+.1, 10000*5.12+.1)
        report = engine.poll_changes(limit=1, near_xy=near)
        self.assertEqual(report['source_cursor'], 1)
        self.assertEqual(report['source_commit_seq'], 201)
        self.assertEqual(report['near_dirty_tiles'], 9)
        result = self.process_tile(engine, (10000, 10000), near)
        self.assertEqual((result['tile_x'], result['tile_y']), (10000, 10000))
        self.assertTrue(result['model_called'])
        self.assertEqual(engine.stats()['source_cursor'], 1)
        data = engine.store.tile(0, 10000, 10000)[0]
        self.assertTrue(np.allclose(data[INDEX['elevation']], .42))

    def test_near_new_commit_bypasses_low_cursor_without_requeuing_unchanged_roi(self):
        self.no_prior()
        for x in range(8):
            self.put(x, -1000)
        self.put(1000, 1000, height=.1)
        engine = self.engine()
        near = (5120.1, 5120.1)
        self.assertEqual(engine.poll_changes(1, near)['near_dirty_tiles'], 9)
        self.process_tile(engine, (1000, 1000), near)
        self.put(1000, 1000, height=.3)
        changed = engine.poll_changes(1, near)
        self.assertEqual(changed['source_cursor'], 2)
        self.assertEqual(changed['near_dirty_tiles'], 9)
        generation = engine.store.conn.execute('SELECT generation,first_order FROM pending '
                                                'WHERE tile_x=1000 AND tile_y=1000').fetchone()
        unchanged = engine.poll_changes(1, near)
        self.assertEqual(unchanged['near_dirty_tiles'], 0)
        self.assertEqual(engine.store.conn.execute('SELECT generation,first_order FROM pending '
                                                   'WHERE tile_x=1000 AND tile_y=1000').fetchone(), generation)

    def test_foreground_metadata_is_25_pk_reads_in_one_snapshot_without_payloads(self):
        self.no_prior()
        self.put(0, 0)
        self.put(1, 0)
        reader = self.reader()
        traces, fired = [], []
        def traced(query):
            traces.append(query)
            if query.startswith('SELECT commit_seq FROM tiles') and not fired:
                fired.append(True)
                self.put(1, 0, height=.9)
        reader.conn.set_trace_callback(traced)
        sequences, clock = reader.neighborhood_sequences(0, 0)
        queries = [query for query in traces if query.startswith('SELECT commit_seq FROM tiles')]
        self.assertEqual(len(queries), 25)
        self.assertTrue(all('payload' not in query and 'BETWEEN' not in query.upper() for query in queries))
        self.assertEqual(clock, 2)
        self.assertEqual(sequences['1,0'], 2)
        self.assertEqual(len(reader.cache), 0)
        plan = reader.conn.execute('EXPLAIN QUERY PLAN SELECT commit_seq FROM tiles '
                                   'WHERE level=0 AND tile_x=? AND tile_y=?', (0, 0)).fetchall()
        self.assertIn('PRIMARY KEY', str(plan))
        self.assertIn('tile_y=?', str(plan))

    def test_foreground_seen_is_bounded_and_virtual_missing_roi_queues_only_once(self):
        engine = self.engine(near_seen_limit=9)
        first = engine.poll_changes(1, (0, 0))
        self.assertEqual(first['near_dirty_tiles'], 9)
        second = engine.poll_changes(1, (0, 0))
        self.assertEqual(second['near_dirty_tiles'], 0)
        for index in range(12):
            engine.poll_changes(1, (1000+100*index, 1000))
            self.assertLessEqual(engine.stats()['near_seen_tiles'], 9)
        self.assertEqual(len(engine.reader.cache), 0)

    def test_failed_enqueue_does_not_remember_foreground_signature(self):
        engine = self.engine()
        with patch.object(engine.store, 'enqueue_changes', side_effect=sqlite3.OperationalError('busy fixture')):
            with self.assertRaises(sqlite3.OperationalError):
                engine.poll_changes(1, (0, 0))
        self.assertEqual(engine.stats()['near_seen_tiles'], 0)
        self.assertEqual(engine.poll_changes(1, (0, 0))['near_dirty_tiles'], 9)

    def test_continuously_requeued_closest_tile_cannot_starve_other_near_tiles(self):
        self.no_prior()
        engine = self.engine()
        hot, neighbors = (0, 0), [(-1, 0), (0, 1), (1, 1)]
        engine.store.enqueue_changes([hot], 0)
        for key in neighbors:
            engine.store.enqueue_changes([key], 0)
        served = []
        for _ in range(4):
            result = engine.process_one((0, 0), allow_inference=False)
            served.append((result['tile_x'], result['tile_y']))
            engine.store.enqueue_changes([hot], 1)
        self.assertEqual(set(served), {hot, *neighbors})
        self.assertEqual(served[0], hot)

    def test_near_queue_is_nine_pk_reads_and_fifo_precedes_distance(self):
        engine = self.engine()
        engine.store.enqueue_changes([(2, 0)], 0)  # Outside output radius; oldest globally.
        engine.store.enqueue_changes([(1, 1)], 0)  # Older near tile.
        engine.store.enqueue_changes([(0, 0)], 0)  # Closest, but younger.
        traces = []
        engine.store.conn.set_trace_callback(traces.append)
        job = engine.store.choose_job((0, 0))
        self.assertEqual(job[:2], (1, 1))
        queries = [query for query in traces if 'FROM pending WHERE tile_x=' in query]
        self.assertEqual(len(queries), 9)
        self.assertTrue(all('BETWEEN' not in query.upper() for query in queries))
        plan = engine.store.conn.execute('EXPLAIN QUERY PLAN SELECT tile_x,tile_y,generation,wanted_seq,first_order '
                                         'FROM pending WHERE tile_x=? AND tile_y=?', (0, 0)).fetchall()
        self.assertIn('PRIMARY KEY', str(plan))
        self.assertIn('tile_y=?', str(plan))
        self.assertEqual(engine.store.choose_job((0, 0), force_oldest=True)[:2], (2, 0))
        with engine.store.conn:
            engine.store.conn.execute('UPDATE pending SET first_order=7 WHERE tile_x IN (0,1)')
        self.assertEqual(engine.store.choose_job((0, 0))[:2], (0, 0))  # Distance only breaks an age tie.


if __name__ == '__main__':
    unittest.main()
