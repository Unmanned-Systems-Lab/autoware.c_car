"""Native sensor-layout and acquisition-time contract tests, without ROS."""

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
import unittest

import numpy as np

from scout_localization_support.cloud_time_order import CloudValidationError, order_native_cloud


@dataclass
class Field:
    name: str
    offset: int
    datatype: int
    count: int = 1


# Fixture encodes the external 32-byte RoboSense/Autoware wire protocol.
SCHEMA = [
    ("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 2),
    ("return_type", 13, 2), ("channel", 14, 4), ("azimuth", 16, 7),
    ("elevation", 20, 7), ("distance", 24, 7), ("time_stamp", 28, 6),
]
DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "u1"),
    ("return_type", "u1"), ("channel", "<u2"), ("azimuth", "<f4"),
    ("elevation", "<f4"), ("distance", "<f4"), ("time_stamp", "<u4"),
])


def make_cloud(times, *, width=None, padding=0):
    times = np.asarray(times, dtype="<u4")
    width = len(times) if width is None else width
    height = len(times) // width
    records = np.zeros(len(times), dtype=DTYPE)
    records["x"] = np.arange(len(times))
    records["y"] = np.arange(len(times)) * -0.5
    records["z"] = 2.5
    records["intensity"] = np.arange(len(times)) % 256
    records["return_type"] = np.arange(len(times)) % 3
    records["channel"] = np.arange(len(times)) % 32
    records["azimuth"] = np.arange(len(times)) * 0.01
    records["elevation"] = -0.2
    records["distance"] = 4.5
    records["time_stamp"] = times
    payload = b"".join(records[start:start + width].tobytes() + b"\xab" * padding
                       for start in range(0, len(times), width))
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=123, nanosec=456), frame_id="rslidar"),
        fields=[Field(*entry) for entry in SCHEMA],
        height=height, width=width, point_step=32, row_step=width * 32 + padding,
        is_bigendian=False, is_dense=False, data=bytearray(payload))


class NativeCloudOrderingTests(unittest.TestCase):
    def test_acquisition_order_and_all_record_bytes_are_preserved(self):
        cloud = make_cloud([30, 10, 10, 20])
        before = deepcopy(cloud)
        result = order_native_cloud(cloud)
        output = np.frombuffer(result.data, DTYPE)
        self.assertEqual(output["x"].tolist(), [1, 2, 3, 0])
        self.assertEqual(output["time_stamp"].tolist(), [10, 10, 20, 30])
        packets = [bytes(before.data[i:i + 32]) for i in range(0, len(before.data), 32)]
        self.assertEqual(result.data, b"".join(packets[i] for i in [1, 2, 3, 0]))
        self.assertEqual((result.height, result.width, result.row_step, result.point_step),
                         (1, 4, 128, 32))
        self.assertEqual(result.stamp_ns, 123_000_000_456)
        self.assertEqual(cloud, before)

    def test_real_scan_shape_with_non_monotonic_firing_batches(self):
        times = np.arange(1800 * 32, dtype="<u4") * 1700
        # Put late-return packets before their earlier acquisition batches.
        times[320:352], times[352:384] = times[352:384].copy(), times[320:352].copy()
        cloud = make_cloud(times, width=32)
        before = bytes(cloud.data)
        result = order_native_cloud(cloud)
        output = np.frombuffer(result.data, DTYPE)
        self.assertEqual((cloud.height, cloud.width), (1800, 32))
        self.assertEqual((result.height, result.width), (1, 57600))
        self.assertTrue(np.all(output["time_stamp"][1:] >= output["time_stamp"][:-1]))
        self.assertGreater(result.backwards_time_pairs, 0)
        # The stable point identity links every reordered packet to the input.
        original = np.frombuffer(before, DTYPE)
        identity = output["x"].astype(np.int64)
        self.assertEqual(output.tobytes(), original[identity].tobytes())
        self.assertEqual(set(identity), set(range(57600)))
        self.assertEqual(bytes(cloud.data), before)

    def test_row_padding_is_not_a_point_and_is_removed(self):
        cloud = make_cloud([60, 50, 40, 30, 20, 10], width=3, padding=16)
        before = bytes(cloud.data)
        result = order_native_cloud(cloud)
        self.assertEqual(len(result.data), 6 * 32)
        self.assertEqual(result.row_step, 192)
        self.assertEqual(np.frombuffer(result.data, DTYPE)["x"].tolist(), [5, 4, 3, 2, 1, 0])
        self.assertEqual(bytes(cloud.data), before)

    def test_sorted_input_with_equal_timestamps_remains_byte_identical(self):
        cloud = make_cloud([0, 1, 1, 1, 2])
        result = order_native_cloud(cloud)
        self.assertEqual(result.data, bytes(cloud.data))
        self.assertEqual(result.backwards_time_pairs, 0)

    def test_nonfinite_geometry_is_filtered_without_rebasing_or_changing_records(self):
        cloud = make_cloud([0, 20, 10, 15, 25])
        record_view = np.frombuffer(cloud.data, DTYPE)
        record_view["x"][0] = np.nan
        record_view["y"][3] = np.inf
        record_view["z"][4] = -np.inf
        before = bytes(cloud.data)
        result = order_native_cloud(cloud)
        output = np.frombuffer(result.data, DTYPE)
        self.assertEqual(result.width, 2)
        self.assertEqual(result.dropped_points, 3)
        self.assertTrue(result.is_dense)
        self.assertEqual(output["x"].tolist(), [2, 1])
        self.assertEqual(output["time_stamp"].tolist(), [10, 20])
        self.assertEqual((result.min_time_ns, result.max_time_ns), (10, 20))
        self.assertEqual(result.stamp_ns, 123_000_000_456)
        self.assertEqual(result.data, before[64:96] + before[32:64])
        self.assertEqual(bytes(cloud.data), before)

    def test_no_valid_geometry_is_rejected(self):
        cloud = make_cloud([0, 1])
        records = np.frombuffer(cloud.data, DTYPE)
        records["x"][0] = np.nan
        records["y"][1] = np.inf
        with self.assertRaisesRegex(CloudValidationError, "no finite XYZ"):
            order_native_cloud(cloud)

    def test_explicit_unfiltered_mode_reports_non_dense_data(self):
        cloud = make_cloud([2, 1])
        np.frombuffer(cloud.data, DTYPE)["z"][0] = np.nan
        result = order_native_cloud(cloud, remove_non_finite=False)
        self.assertEqual(result.width, 2)
        self.assertEqual(result.dropped_points, 0)
        self.assertFalse(result.is_dense)

    def test_scan_duration_boundary_is_inclusive(self):
        result = order_native_cloud(make_cloud([200_000_000, 0]))
        self.assertEqual((result.min_time_ns, result.max_time_ns), (0, 200_000_000))
        with self.assertRaises(CloudValidationError):
            order_native_cloud(make_cloud([0, 200_000_001]))
        with self.assertRaises(CloudValidationError):
            order_native_cloud(make_cloud([0, 0xFFFFFFFF]))

    def test_native_schema_is_required(self):
        mutations = [
            lambda c: c.fields.pop(),
            lambda c: setattr(c.fields[3], "datatype", 7),
            lambda c: setattr(c.fields[-1], "datatype", 7),
            lambda c: setattr(c.fields[-1], "offset", 24),
            lambda c: setattr(c.fields[-1], "count", 2),
            lambda c: setattr(c.fields[-1], "name", "x"),
            lambda c: setattr(c, "point_step", 36),
            lambda c: setattr(c, "is_bigendian", True),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                cloud = make_cloud([0, 1])
                mutation(cloud)
                with self.assertRaises(CloudValidationError):
                    order_native_cloud(cloud)

    def test_channel_upper_bound(self):
        cloud = make_cloud([0, 1])
        view = np.frombuffer(cloud.data, DTYPE)
        view["channel"][0] = 31
        order_native_cloud(cloud)
        view["channel"][0] = 32
        with self.assertRaises(CloudValidationError):
            order_native_cloud(cloud)

    def test_header_stamp_is_valid_and_set(self):
        for seconds, nanoseconds in [(0, 0), (-1, 10), (1, -1), (1, 1_000_000_000), (1, 2.5)]:
            with self.subTest(stamp=(seconds, nanoseconds)):
                cloud = make_cloud([0, 1])
                cloud.header.stamp.sec = seconds
                cloud.header.stamp.nanosec = nanoseconds
                with self.assertRaises(CloudValidationError):
                    order_native_cloud(cloud)
        cloud = make_cloud([0, 1])
        cloud.header.frame_id = ""
        with self.assertRaises(CloudValidationError):
            order_native_cloud(cloud)

    def test_empty_and_inconsistent_buffers_are_rejected(self):
        mutations = [
            lambda c: setattr(c, "width", 0),
            lambda c: setattr(c, "height", 0),
            lambda c: setattr(c, "row_step", 32),
            lambda c: c.data.pop(),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                cloud = make_cloud([0, 1])
                mutation(cloud)
                with self.assertRaises(CloudValidationError):
                    order_native_cloud(cloud)


if __name__ == "__main__":
    unittest.main()
