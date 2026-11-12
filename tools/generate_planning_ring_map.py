#!/usr/bin/env python3
"""Generate a closed, segmented Lanelet2 ring for offline planning simulation."""
import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def generate(output_dir, radius=20.0, lane_width=4.0, samples=144, segments=12):
    if not all(math.isfinite(v) and v > 0 for v in [radius, lane_width]):
        raise ValueError('Radius and lane width must be finite positive values')
    if radius <= lane_width / 2 or segments < 4 or samples < segments * 2 or samples % segments:
        raise ValueError('Require a positive inner radius and samples divisible by segments')
    output_dir.mkdir(parents=True, exist_ok=True)
    osm = ET.Element('osm', version='0.6', generator='generate_planning_ring_map.py')
    ET.SubElement(osm, 'MetaInfo', format_version='1', map_version='1')
    tracks = [('left', radius - lane_width / 2, 1), ('right', radius + lane_width / 2, 10001), ('centerline', radius, 20001)]
    for _, track_radius, base in tracks:
        for i in range(samples):
            theta = 2 * math.pi * i / samples
            x, y = track_radius * math.cos(theta), track_radius * math.sin(theta)
            node = ET.SubElement(osm, 'node', id=str(base + i), lat=f'{y / 110574.0:.12f}', lon=f'{x / 111319.49:.12f}')
            for key, value in [('local_x', f'{x:.6f}'), ('local_y', f'{y:.6f}'), ('ele', '0')]:
                ET.SubElement(node, 'tag', k=key, v=value)
    step = samples // segments
    for segment in range(segments):
        ids = {}
        for offset, (role, _, base) in enumerate(tracks):
            way_id = 40000 + segment * 3 + offset
            ids[role] = way_id
            way = ET.SubElement(osm, 'way', id=str(way_id))
            for i in range(segment * step, (segment + 1) * step + 1):
                ET.SubElement(way, 'nd', ref=str(base + i % samples))
            if role != 'centerline':
                ET.SubElement(way, 'tag', k='type', v='line_thin')
                ET.SubElement(way, 'tag', k='subtype', v='solid')
        relation = ET.SubElement(osm, 'relation', id=str(30001 + segment))
        for role, way_id in ids.items():
            ET.SubElement(relation, 'member', type='way', role=role, ref=str(way_id))
        for key, value in [('type', 'lanelet'), ('subtype', 'road'), ('location', 'urban'), ('one_way', 'yes'), ('speed_limit', '10 km/h'), ('turn_direction', 'straight')]:
            ET.SubElement(relation, 'tag', k=key, v=value)
    ET.indent(osm, space='  ')
    ET.ElementTree(osm).write(output_dir / 'lanelet2_map.osm', encoding='utf-8', xml_declaration=True)
    points = []
    for i in range(samples):
        theta = 2 * math.pi * i / samples
        for j in range(5):
            r = radius - lane_width / 2 + lane_width * j / 4
            points.append(f'{r * math.cos(theta):.6f} {r * math.sin(theta):.6f} 0.000000 1.000000')
    header = ['# .PCD v0.7 - Point Cloud Data file format', 'VERSION 0.7', 'FIELDS x y z intensity', 'SIZE 4 4 4 4', 'TYPE F F F F', 'COUNT 1 1 1 1', f'WIDTH {len(points)}', 'HEIGHT 1', 'VIEWPOINT 0 0 0 1 0 0 0', f'POINTS {len(points)}', 'DATA ascii']
    (output_dir / 'pointcloud_map.pcd').write_text('\n'.join(header + points) + '\n')
    edge = math.ceil(radius + lane_width / 2) + 1
    (output_dir / 'pointcloud_map_metadata.yaml').write_text(f'x_resolution: {edge * 2}\ny_resolution: {edge * 2}\npointcloud_map.pcd:\n- {-edge}\n- {-edge}\n')
    (output_dir / 'map_projector_info.yaml').write_text('projector_type: Local\n')
    info = {'center': [0.0, 0.0], 'radius_m': radius, 'lane_width_m': lane_width, 'direction': 'counterclockwise', 'speed_limit_kph': 10, 'lanelets': segments, 'samples_per_ring': samples, 'lanelet_ids': list(range(30001, 30001 + segments)), 'pointcloud_points': len(points), 'closed': True}
    (output_dir / 'map_info.json').write_text(json.dumps(info, indent=2) + '\n')
    return info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'maps/planning_simulation/ring')
    parser.add_argument('--radius', type=float, default=20.0)
    parser.add_argument('--lane-width', type=float, default=4.0)
    parser.add_argument('--samples', type=int, default=144)
    parser.add_argument('--segments', type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.radius, args.lane_width, args.samples, args.segments), indent=2))
