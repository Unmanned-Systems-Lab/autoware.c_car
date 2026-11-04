"""Encode canonical world (+Y rows, +X columns) result windows as GridMap."""
from array import array
import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def window_message(window, stamp, threshold=0.5, raw_directions=False, stale_timeout=10.0):
    resolution = float(window['resolution'])
    layers = {k: np.asarray(v, dtype=np.float32) for k, v in window['layers'].items()}
    height = layers['elevation']; rows, cols = height.shape
    if rows < 1 or cols < 1 or resolution <= 0:
        raise ValueError('Invalid result window geometry')
    probability = layers['traversability'].copy()
    if 'source_stale' in layers and 'result_age_sec' in layers:
        stale = (layers['source_stale'] > 0.5) & (layers['result_age_sec'] > stale_timeout)
        probability[stale] = np.nan
    probability[~np.isfinite(height)] = np.nan
    if 'valid' in layers:
        probability[~np.isfinite(layers['valid']) | (layers['valid'] <= 0)] = np.nan
    labels = np.full_like(probability, np.nan)
    valid = np.isfinite(probability)
    labels[valid] = (probability[valid] >= threshold).astype(np.float32)
    # Preserve coarse coverage fractions, but never advertise a masked result as valid.
    layers['valid'] = np.where(valid, layers.get('valid', valid.astype(np.float32)), 0).astype(np.float32)
    if raw_directions:
        for index in range(8):
            name = f'ht_dir_{index}'
            if name in layers:
                layers[name] = np.where(valid, layers[name], np.nan).astype(np.float32)
    layers['traversability'] = probability
    layers['traversability_class'] = labels
    names = ['elevation', 'traversability', 'traversability_class', 'valid', 'support_fraction',
             'observed_fraction', 'source_stale', 'result_age_sec']
    if raw_directions:
        names += [f'ht_dir_{index}' for index in range(8)]
    message = GridMap(); message.header.frame_id = window.get('frame', 'map'); message.header.stamp = stamp
    message.info.resolution = resolution; message.info.length_x = cols * resolution; message.info.length_y = rows * resolution
    center = window.get('center_xy')
    if center is None:
        origin = window['origin_xy']; center = [origin[0] + cols * resolution / 2, origin[1] + rows * resolution / 2]
    message.info.pose.position.x = float(center[0]); message.info.pose.position.y = float(center[1]); message.info.pose.orientation.w = 1.0
    message.basic_layers = ['elevation', 'traversability']
    for name in names:
        if name not in layers: continue
        values = layers[name]
        if values.shape != height.shape: raise ValueError(f'Layer geometry mismatch: {name}')
        # GridMap buffers use rows=-X, columns=-Y and column-major storage.
        grid = np.ascontiguousarray(np.flip(values.T, axis=(0, 1)))
        a = Float32MultiArray(); a.layout.dim = [
            MultiArrayDimension(label='column_index', size=grid.shape[1], stride=grid.size),
            MultiArrayDimension(label='row_index', size=grid.shape[0], stride=grid.shape[0])]
        a.data = array('f'); a.data.frombytes(np.ascontiguousarray(grid.T).tobytes())
        message.layers.append(name); message.data.append(a)
    return message


def empty_window(center_xy=(0.0, 0.0), size_m=10.0, resolution=0.04):
    cells = max(1, round(size_m / resolution))
    return {'frame': 'map', 'resolution': resolution, 'center_xy': list(center_xy),
            'layers': {'elevation': np.full((cells, cells), np.nan, np.float32),
                       'traversability': np.full((cells, cells), np.nan, np.float32)}}
