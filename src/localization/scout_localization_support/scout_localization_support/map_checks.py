"""Checks for an NDT source map; this does not establish geographic alignment."""

from pathlib import Path
import struct


def require_ndt_map(filename, minimum_points=1000):
    path = Path(filename).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != '.pcd':
        raise ValueError('NDT requires one real .pcd file, not a directory')
    header = {}
    with path.open('rb') as stream:
        for _ in range(100):
            line = stream.readline(4096)
            if not line or len(line) >= 4096:
                raise ValueError('Incomplete or oversized PCD header')
            parts = line.decode('ascii').strip().split()
            if not parts or parts[0].startswith('#'):
                continue
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == 'DATA':
                payload_start = stream.tell()
                break
        else:
            raise ValueError('PCD DATA header not found')
    fields = header.get('FIELDS', header.get('FIELD', []))
    if not {'x', 'y', 'z'}.issubset(fields):
        raise ValueError('NDT map must contain x, y and z fields')
    points = int(header.get('POINTS', [0])[0])
    if points < minimum_points:
        raise ValueError(
            f'NDT map has only {points} points; the UI-generated placeholder is not a localization map. '
            'Supply a real 3D map aligned with the Lanelet map, or use the localization-only PCD launch.')
    width = int(header.get('WIDTH', [points])[0])
    height = int(header.get('HEIGHT', [1])[0])
    if width * height != points:
        raise ValueError('PCD dimensions do not match POINTS')
    encoding = header.get('DATA', [''])[0]
    if encoding not in ('ascii', 'binary', 'binary_compressed'):
        raise ValueError('Unsupported PCD DATA encoding')
    sizes = list(map(int, header.get('SIZE', [])))
    counts = list(map(int, header.get('COUNT', ['1'] * len(fields))))
    types = header.get('TYPE', [])
    if (len(sizes) != len(fields) or len(counts) != len(fields) or len(types) != len(fields)
            or any(x <= 0 for x in sizes + counts) or any(t not in ('F', 'I', 'U') for t in types)):
        raise ValueError('Invalid PCD field declarations')
    expected = points * sum(a * b for a, b in zip(sizes, counts))
    payload_size = path.stat().st_size - payload_start
    if encoding == 'binary':
        if payload_size < expected:
            raise ValueError('Truncated binary PCD map')
    elif encoding == 'binary_compressed':
        with path.open('rb') as stream:
            stream.seek(payload_start)
            lengths = stream.read(8)
        if len(lengths) != 8:
            raise ValueError('Truncated compressed PCD size header')
        compressed, uncompressed = struct.unpack('<II', lengths)
        if compressed == 0 or uncompressed != expected or payload_size < 8 + compressed:
            raise ValueError('Truncated or inconsistent compressed PCD map')
    else:
        records = 0
        columns = sum(counts)
        with path.open('rb') as stream:
            stream.seek(payload_start)
            for line in stream:
                text = line.decode('ascii').strip()
                if not text or text.startswith('#'):
                    continue
                values = text.split()
                if len(values) != columns:
                    raise ValueError('Invalid ASCII PCD point record')
                try:
                    for value in values:
                        float(value)
                except ValueError as error:
                    raise ValueError('Invalid numeric ASCII PCD record') from error
                records += 1
        if records != points:
            raise ValueError('Truncated or inconsistent ASCII PCD map')
    return {'path': str(path), 'points': points, 'encoding': encoding}
