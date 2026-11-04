import struct

import numpy as np
import pytest
from sensor_msgs.msg import PointCloud2, PointField

from elevation_mapping_cupy.elevation_mapping_node import (
    _pointcloud2_xyz_f32,
    _remove_points_inside_xy_box,
    _remove_points_inside_xyz_box,
)


def _padded_cloud() -> PointCloud2:
    message = PointCloud2()
    message.height = 2
    message.width = 2
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = 40
    message.is_dense = True
    data = bytearray(message.row_step * message.height)
    points = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (10.0, 11.0, 12.0)]
    for index, point in enumerate(points):
        row, col = divmod(index, message.width)
        offset = row * message.row_step + col * message.point_step
        struct.pack_into("<ffff", data, offset, *point, 100.0 + index)
    message.data = bytes(data)
    return message


def test_parser_honors_organized_cloud_row_padding():
    points = _pointcloud2_xyz_f32(_padded_cloud())

    np.testing.assert_array_equal(
        points,
        np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        ),
    )


def test_parser_rejects_truncated_organized_cloud():
    message = _padded_cloud()
    message.data = message.data[:-1]

    with pytest.raises(ValueError, match="expected at least"):
        _pointcloud2_xyz_f32(message)


def test_self_filter_removes_inclusive_xy_box_and_preserves_channels():
    points = np.asarray(
        [
            [-1.0, -0.5, -100.0, 10.0],
            [0.0, 0.5, 100.0, 11.0],
            [-0.5, 0.0, 0.0, 12.0],
            [-1.01, 0.0, 0.0, 20.0],
            [0.01, 0.0, 0.0, 21.0],
            [-0.5, -0.51, 0.0, 22.0],
            [-0.5, 0.51, 0.0, 23.0],
        ],
        dtype=np.float32,
    )

    filtered = _remove_points_inside_xy_box(
        points,
        min_x=-1.0,
        max_x=0.0,
        min_y=-0.5,
        max_y=0.5,
    )

    np.testing.assert_array_equal(filtered, points[3:])


def test_self_filter_can_remove_every_point_without_changing_column_count():
    points = np.asarray(
        [[-0.8, -0.2, -1.0, 1.0], [-0.2, 0.3, 1.0, 2.0]],
        dtype=np.float32,
    )

    filtered = _remove_points_inside_xy_box(points, -1.0, 0.0, -0.5, 0.5)

    assert filtered.shape == (0, 4)
    assert filtered.dtype == points.dtype


def test_self_filter_xyz_box_preserves_points_above_and_below_vehicle():
    points = np.asarray(
        [
            [-0.5, 0.0, -0.91, 1.0],
            [-0.5, 0.0, -0.89, 2.0],
            [0.2, 0.7, 0.2, 3.0],
            [-0.5, 0.0, 0.21, 4.0],
        ],
        dtype=np.float32,
    )

    filtered = _remove_points_inside_xyz_box(
        points, -1.0, 0.2, -0.7, 0.7, -0.9, 0.2
    )

    np.testing.assert_array_equal(filtered, points[[0, 3]])


@pytest.mark.parametrize(
    "bounds",
    [
        (1.0, 0.0, -0.5, 0.5),
        (-1.0, 0.0, 0.5, -0.5),
        (float("nan"), 0.0, -0.5, 0.5),
    ],
)
def test_self_filter_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        _remove_points_inside_xy_box(
            np.empty((0, 3), dtype=np.float32),
            *bounds,
        )
