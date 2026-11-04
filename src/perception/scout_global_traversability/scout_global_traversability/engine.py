"""Incremental committed-map inference; no ROS, motion outputs, or whole-map tensors."""
from collections import OrderedDict
import hashlib
import math
import threading
import time

import numpy as np

from .source import SourceReader, canonical
from .store import INDEX, LAYERS, ResultStore, empty_tile, tile_cells


def window_fraction(mask, window=51):
    values = np.asarray(mask, dtype=np.float64)
    integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return ((integral[window:, window:] - integral[:-window, window:]
             - integral[window:, :-window] + integral[:-window, :-window]) / (window*window)).astype(np.float32)


class Engine:
    def __init__(self, context, predictor, result_db, *, model_sha256, config=None, max_cached_tiles=16,
                 near_seen_limit=256):
        if not isinstance(model_sha256, str) or len(model_sha256) != 64 or any(c not in '0123456789abcdef' for c in model_sha256):
            raise ValueError('A verified model SHA256 is required for durable inference results')
        self.reader = SourceReader(context, max_cached_tiles=max_cached_tiles)
        self.predictor = predictor
        self.config = dict(config or {})
        if getattr(predictor, 'weights_sha256', model_sha256) != model_sha256:
            self.reader.close()
            raise ValueError('Predictor weights differ from the bound model SHA256')
        predictor_contract = getattr(predictor, 'contract', None)
        if callable(predictor_contract):
            predictor_contract = predictor_contract()
        if predictor_contract is None:
            predictor_contract = {'adapter': type(predictor).__module__ + '.' + type(predictor).__qualname__,
                                  'min_valid_ratio': getattr(predictor, 'min_valid_ratio', None),
                                  'receptive_field': getattr(predictor, 'receptive_field', 51)}
        contract = {
            'schema_version': 1, 'source_identity_sha256': self.reader.identity_sha256,
            'source_database': str(self.reader.path), 'source_geometry': self.reader.geometry_contract,
            'model_sha256': model_sha256, 'inference_config': self.config,
            'predictor_contract': predictor_contract,
            'tile_size': 128, 'halo': 25, 'axes': 'rows_world_positive_y_columns_world_positive_x',
            'score': 'minimum_of_eight_raw_channel_probabilities', 'front_reference_shift_m': 0.0,
            'prior_policy': 'finite_bounded_height_is_model_input; observations_are_separate_quality',
            'overview_reduction': 'min_known_probabilities_mean_coverage_max_variance_small_tiles_v2',
        }
        contract['contract_sha256'] = hashlib.sha256(canonical(contract).encode()).hexdigest()
        try:
            self.store = ResultStore(result_db, contract, max_cached_tiles=max_cached_tiles)
        except Exception:
            self.reader.close()
            raise
        self.processing_lock = threading.Lock()
        self.poll_lock = threading.Lock()
        self.near_seen_limit = int(near_seen_limit)
        if not 9 <= self.near_seen_limit <= 4096:
            self.reader.close()
            self.store.close()
            raise ValueError('near_seen_limit must hold 9..4096 output tile signatures')
        self.near_seen = OrderedDict()
        self.job_counter = 0
        self.source_clock, self.last_poll_wall, self.last_job_ms = 0, None, None
        self.last_error = ''

    def poll_changes(self, limit=64, near_xy=None):
        with self.poll_lock:
            return self._poll_changes(limit, near_xy)

    def _poll_changes(self, limit, near_xy):
        limit = max(1, min(int(limit), 4096))
        state = self.store.stats()
        changes, cursor, clock = self.reader.poll_changes(state['source_cursor'], limit)
        dirty = set()
        for item in changes:
            x, y = item['tile_x'], item['tile_y']
            dirty.update((xx, yy) for yy in range(y-1, y+2) for xx in range(x-1, x+2))
        prior, prior_cursor, prior_done = self.reader.prior_page(state['prior_cursor'], limit)
        dirty.update(prior)
        near_updates, near_dirty = [], set()
        if near_xy is not None:
            if len(near_xy) != 2 or not all(math.isfinite(float(value)) for value in near_xy):
                raise ValueError('Invalid near-robot position')
            scale = 128*self.reader.resolution
            nx, ny = math.floor(float(near_xy[0])/scale), math.floor(float(near_xy[1])/scale)
            sequences, near_clock = self.reader.neighborhood_sequences(nx, ny)
            clock = max(clock, near_clock)
            # Nine output cores depend on the union of 25 source tiles. Missing
            # rows have seq=0, so a first ROI is queued even for a virtual prior.
            for y in range(ny-1, ny+2):
                for x in range(nx-1, nx+2):
                    signature = tuple(sequences[f'{xx},{yy}'] for yy in range(y-1, y+2)
                                      for xx in range(x-1, x+2))
                    key = x, y
                    if self.near_seen.get(key) != signature:
                        near_dirty.add(key)
                    near_updates.append((key, signature))
            dirty.update(near_dirty)
        # Enqueue and advance both cursors in the same durable result transaction.
        self.store.enqueue_changes(dirty, clock, cursor=cursor, prior_cursor=prior_cursor, prior_done=prior_done)
        # Do not remember a signature until its work is durably queued. A failed
        # result transaction must leave foreground prefetch retryable next poll.
        for key, signature in near_updates:
            self.near_seen[key] = signature
            self.near_seen.move_to_end(key)
        while len(self.near_seen) > self.near_seen_limit:
            self.near_seen.popitem(last=False)
        self.source_clock, self.last_poll_wall = clock, time.time()
        return {'source_rows': len(changes), 'dirty_tiles': len(dirty), 'source_commit_seq': clock,
                'source_cursor': cursor, 'prior_done': prior_done,
                'near_dirty_tiles': len(near_dirty), 'near_seen_tiles': len(self.near_seen)}

    def process_one(self, near_xy=None, *, allow_inference=True):
        if not self.processing_lock.acquire(blocking=False):
            return None
        started = time.monotonic()
        try:
            near_tile = None
            if near_xy is not None:
                if len(near_xy) != 2 or not all(math.isfinite(float(value)) for value in near_xy):
                    raise ValueError('Invalid near-robot position')
                scale = 128*self.reader.resolution
                near_tile = math.floor(float(near_xy[0])/scale), math.floor(float(near_xy[1])/scale)
            job = self.store.choose_job(near_tile, force_oldest=(self.job_counter+1) % 4 == 0)
            if job is None:
                return None
            snapshot = self.reader.snapshot(job[0], job[1])
            input_sha256 = hashlib.sha256(np.asarray(snapshot.height, dtype='<f4', order='C').tobytes()).hexdigest()
            data = empty_tile()
            core = np.s_[25:153, 25:153]
            height, count = snapshot.height[core], snapshot.observation_count[core]
            data[INDEX['elevation']], data[INDEX['variance']] = height, snapshot.variance[core]
            data[INDEX['observation_count']] = count
            data[INDEX['observed_fraction']] = window_fraction(
                (snapshot.observation_count > 0) & np.isfinite(snapshot.height))
            model_called, job_state = False, 'skip_unknown'
            previous = self.store.tile(0, job[0], job[1])
            reusable = previous is not None and previous[1].get('input_sha256') == input_sha256
            if reusable:
                for name in ('traversability', *(f'ht_dir_{i}' for i in range(8)), 'valid', 'support_fraction'):
                    data[INDEX[name]] = previous[0][INDEX[name]]
                job_state = 'reused_exact_height'
            elif np.isfinite(height).any():
                if not allow_inference:
                    return {'state': 'inference_budget_wait', 'model_called': False,
                            'tile_x': job[0], 'tile_y': job[1]}
                prediction = self.predictor.predict(snapshot.height)
                model_called, job_state = True, 'predicted'
                probabilities = np.asarray(prediction['probabilities'], dtype=np.float32)
                valid = np.asarray(prediction['valid'], dtype=bool)
                support = np.asarray(prediction['support_fraction'], dtype=np.float32)
                if probabilities.shape != (8, 128, 128) or valid.shape != (128, 128) or support.shape != (128, 128):
                    raise ValueError('Model output shape violates the 128-core/25-halo contract')
                if not np.isfinite(support).all() or ((support < 0) | (support > 1)).any():
                    raise ValueError('Invalid finite-height support fractions')
                sane = np.isfinite(probabilities).all(axis=0) & ((probabilities >= 0) & (probabilities <= 1)).all(axis=0)
                valid &= np.isfinite(height) & sane
                probabilities = np.where(valid[None], probabilities, np.nan)
                data[INDEX['traversability']] = np.where(valid, np.min(probabilities, axis=0), np.nan)
                for index in range(8):
                    data[INDEX[f'ht_dir_{index}']] = probabilities[index]
                data[INDEX['valid']], data[INDEX['support_fraction']] = valid, support
            dependencies, current_seq = self.reader.dependencies_now(snapshot.dependencies)
            stale = dependencies != snapshot.dependencies
            self.store.complete(job, data, snapshot, stale=stale, current_seq=current_seq,
                                input_sha256=input_sha256)
            self.job_counter += 1
            stale = self.store.tile(0, job[0], job[1])[1]['source_stale']
            self.last_job_ms = (time.monotonic()-started)*1000
            self.last_error = ''
            return {'state': job_state, 'model_called': model_called,
                    'tile_x': job[0], 'tile_y': job[1], 'source_commit_seq': snapshot.commit_seq,
                    'source_stale': stale, 'valid_cells': int(np.count_nonzero(data[INDEX['valid']])),
                    'finite_height_cells': int(np.isfinite(height).sum()), 'compute_ms': self.last_job_ms}
        except Exception as error:
            self.last_error = str(error)
            raise
        finally:
            self.processing_lock.release()

    def _window(self, ix0, iy0, width, height, level=0):
        if not 0 < width <= 512 or not 0 < height <= 512:
            raise ValueError('Published windows must be bounded to 512 cells per axis')
        data = np.full((len(LAYERS), height, width), np.nan, np.float32)
        for name in ('valid', 'support_fraction', 'observed_fraction', 'observation_count'):
            data[INDEX[name]] = 0
        stale_layer = np.full((height, width), np.nan, np.float32)
        age_layer = np.full((height, width), np.nan, np.float32)
        now = time.time()
        found, stale_tiles, times = 0, 0, []
        cells = tile_cells(level)
        for ty in range(iy0//cells, (iy0+height-1)//cells+1):
            for tx in range(ix0//cells, (ix0+width-1)//cells+1):
                item = self.store.tile(level, tx, ty)
                if item is None:
                    continue
                found += 1
                tile, metadata = item
                x0, y0, x1, y1 = max(ix0, tx*cells), max(iy0, ty*cells), min(ix0+width, (tx+1)*cells), min(iy0+height, (ty+1)*cells)
                src = (slice(y0-ty*cells, y1-ty*cells), slice(x0-tx*cells, x1-tx*cells))
                dst = (slice(y0-iy0, y1-iy0), slice(x0-ix0, x1-ix0))
                data[(slice(None), *dst)] = tile[(slice(None), *src)]
                known = np.isfinite(tile[INDEX['elevation']][src])
                stale_layer[dst] = np.where(known, float(metadata['source_stale']), np.nan)
                age_layer[dst] = np.where(known, max(0, now-metadata['computed_at']), np.nan)
                times.append(metadata['computed_at'])
                stale_tiles += int(metadata['source_stale'])
        resolution = self.reader.resolution*2**level
        origin = [(ix0*2**level+self.reader.phase_x-.5)*self.reader.resolution,
                  (iy0*2**level+self.reader.phase_y-.5)*self.reader.resolution]
        layers = {name: data[index] for index, name in enumerate(LAYERS)}
        layers.update(source_stale=stale_layer, result_age_sec=age_layer)
        state = self.store.stats()
        return {'frame': 'map', 'resolution': resolution, 'origin_xy': origin,
                'center_xy': [origin[0]+width*resolution/2, origin[1]+height*resolution/2],
                'shape': (height, width), 'layers': layers,
                'quality': {'lod': level, 'cached_tiles': found, 'stale_result_tiles': stale_tiles,
                            'pending_tiles': state['pending_count'], 'source_commit_seq': self.source_clock,
                            'oldest_computed_at': min(times) if times else None,
                            'newest_computed_at': max(times) if times else None,
                            'aggregation': 'native_cells' if level == 0 else 'min_known_predictions; valid_is_coverage_fraction',
                            'axes': 'world_yx', 'score': 'min8_raw_uncalibrated_channels'}}

    def local_window(self, center_xy, size_m=10.0):
        if len(center_xy) != 2 or not all(math.isfinite(float(value)) for value in center_xy):
            raise ValueError('Invalid local-window center')
        sizes = (size_m, size_m) if isinstance(size_m, (int, float)) else size_m
        if len(sizes) != 2 or not all(math.isfinite(float(value)) and float(value) > 0 for value in sizes):
            raise ValueError('Invalid local-window dimensions')
        width, height = [int(round(float(value)/self.reader.resolution)) for value in sizes]
        ix0 = math.floor(float(center_xy[0])/self.reader.resolution-width/2-self.reader.phase_x+.5+1e-9)
        iy0 = math.floor(float(center_xy[1])/self.reader.resolution-height/2-self.reader.phase_y+.5+1e-9)
        return self._window(ix0, iy0, width, height)

    def overview(self, max_size=512):
        maximum = int(max_size)
        if not 1 <= maximum <= 512:
            raise ValueError('Overview max_size must be 1..512')
        extent = self.store.stats()['extent']
        if extent is None:
            return None
        level = 0
        while (extent[2]//(2**level)-extent[0]//(2**level)+1 > maximum
               or extent[3]//(2**level)-extent[1]//(2**level)+1 > maximum):
            level += 1
            if level > self.store.max_lod:
                raise ValueError('Map extent exceeds the maintained overview pyramid')
        x0, y0, x1, y1 = [value//(2**level) for value in extent]
        return self._window(x0, y0, x1-x0+1, y1-y0+1, level)

    def stats(self):
        state = self.store.stats()
        return dict(state, pending_tiles=state['pending_count'], source_commit_seq=self.source_clock,
                    last_poll_age_sec=None if self.last_poll_wall is None else max(0, time.time()-self.last_poll_wall),
                    last_job_ms=self.last_job_ms, last_error=self.last_error,
                    source_cached_tiles=len(self.reader.cache), result_cached_tiles=len(self.store.cache),
                    near_seen_tiles=len(self.near_seen), near_seen_limit=self.near_seen_limit)

    def close(self):
        with self.processing_lock, self.poll_lock:
            self.reader.close()
            self.store.close()
