"""Physical and input-contract regression tests; no ROS installation required."""

from dataclasses import replace
import math
import unittest

from scout_localization_support.wheel_conversion import (
    ConversionConfig, InvalidWheelSample, MountTransform, WheelSample, convert_wheel_sample,
)


class WheelConversionTest(unittest.TestCase):
    def setUp(self):
        self.config = ConversionConfig()
        self.now = 10_000_000_000
        self.sample = WheelSample(
            self.now, "scout_wheel_odom", "scout_chassis_link",
            (1.0, 0.0, 0.0), (0.0, 0.0, 0.5), [0.0] * 36,
        )
        self.mount = MountTransform(
            "base_link", "scout_chassis_link", (0.249, 0.0, 0.23479), (0.0, 0.0, 0.0, 1.0),
        )

    def convert(self, sample=None, mount=None, config=None, **kwargs):
        return convert_wheel_sample(
            sample or self.sample, mount or self.mount, config or self.config,
            kwargs.get("now_ns", self.now), kwargs.get("last_stamp_ns"),
        )

    def test_rear_axle_lateral_velocity_has_correct_sign(self):
        result = self.convert()
        self.assertEqual(result.linear[0], 1.0)
        self.assertAlmostEqual(result.linear[1], -0.249 * 0.5)
        self.assertEqual(result.linear[2], 0.0)
        self.assertEqual(result.angular, (0.0, 0.0, 0.5))
        # Moving back to chassis center must recover its no-slip lateral speed.
        self.assertAlmostEqual(result.linear[1] + result.angular[2] * 0.249, 0.0)

    def test_turn_in_place_is_not_zero_velocity_at_rear_axle(self):
        result = self.convert(replace(self.sample, linear=(0.0, 0.0, 0.0)))
        self.assertAlmostEqual(result.linear[1], -0.1245)

    def test_straight_motion_does_not_add_lateral_motion(self):
        result = self.convert(replace(self.sample, angular=(0.0, 0.0, 0.0)))
        self.assertEqual(result.linear, (1.0, 0.0, 0.0))

    def test_covariance_includes_signed_lever_arm_correlation(self):
        covariance = [0.0] * 36
        covariance[0], covariance[35] = 0.09, 0.04
        covariance[5] = covariance[30] = 0.02
        result = self.convert(replace(self.sample, covariance=covariance))
        self.assertEqual(result.covariance_source, "input")
        self.assertAlmostEqual(result.covariance[0], 0.09)
        self.assertAlmostEqual(result.covariance[1], -0.249 * 0.02)
        self.assertAlmostEqual(result.covariance[11], -0.249 * 0.04)
        self.assertAlmostEqual(result.covariance[7], 10000.0 + 0.249 ** 2 * 0.04)
        self.assertAlmostEqual(result.covariance[35], 0.04)
        for i in range(6):
            for j in range(6):
                self.assertAlmostEqual(result.covariance[i * 6 + j], result.covariance[j * 6 + i])

    def test_zero_covariance_uses_configured_uncertainty(self):
        result = self.convert()
        self.assertEqual(result.covariance_source, "configured_zero_covariance")
        self.assertAlmostEqual(result.covariance[0], 0.2 ** 2)
        self.assertAlmostEqual(result.covariance[35], 0.1 ** 2)
        self.assertGreater(result.covariance[7], 9999)
        self.assertGreater(result.covariance[14], 9999)

    def test_mount_height_does_not_invent_uncertain_longitudinal_rate(self):
        low = self.convert(mount=replace(self.mount, translation=(0.249, 0.0, 0.0)))
        high = self.convert(mount=replace(self.mount, translation=(0.249, 0.0, 1.0)))
        self.assertEqual(low.linear, high.linear)
        self.assertEqual(low.covariance[0], high.covariance[0])

    def test_rotated_mount_rotates_velocity_and_unknown_directions(self):
        half = math.sqrt(0.5)
        mount = replace(self.mount, translation=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, half, half))
        result = self.convert(mount=mount)
        self.assertAlmostEqual(result.linear[0], 0.0)
        self.assertAlmostEqual(result.linear[1], 1.0)
        self.assertGreater(result.covariance[0], 9999)
        self.assertAlmostEqual(result.covariance[7], 0.04)

    def test_positive_semidefinite_input_is_preserved(self):
        covariance = [0.0] * 36
        covariance[0], covariance[35] = 0.04, 0.01
        covariance[5] = covariance[30] = -0.02  # Rank-one PSD boundary.
        result = self.convert(replace(self.sample, covariance=covariance))
        self.assertAlmostEqual(result.covariance[5], -0.02)

    def test_singular_variance_gets_nonzero_floor(self):
        covariance = [0.0] * 36
        covariance[35] = 0.01
        result = self.convert(replace(self.sample, covariance=covariance))
        self.assertGreaterEqual(result.covariance[0], self.config.minimum_linear_stddev ** 2)

    def test_invalid_covariance_is_rejected(self):
        for bad_kind in ("negative", "asymmetric", "indefinite", "nan", "length"):
            with self.subTest(kind=bad_kind):
                covariance = [0.0] * 36
                covariance[0], covariance[35] = 0.04, 0.01
                if bad_kind == "negative": covariance[0] = -1.0
                if bad_kind == "asymmetric": covariance[5] = 0.01
                if bad_kind == "indefinite": covariance[5] = covariance[30] = 0.03
                if bad_kind == "nan": covariance[0] = math.nan
                if bad_kind == "length": covariance = covariance[:-1]
                with self.assertRaises(InvalidWheelSample):
                    self.convert(replace(self.sample, covariance=covariance))

    def test_invalid_covariance_fallback_is_explicit(self):
        covariance = [math.nan] * 36
        result = self.convert(
            replace(self.sample, covariance=covariance),
            config=replace(self.config, invalid_covariance_policy="fallback"),
        )
        self.assertEqual(result.covariance_source, "configured_invalid_covariance")
        self.assertAlmostEqual(result.covariance[0], 0.04)

    def test_parent_and_child_frames_are_checked_separately(self):
        for sample in (replace(self.sample, parent_frame="map"), replace(self.sample, child_frame="base_link")):
            with self.assertRaises(InvalidWheelSample): self.convert(sample)
        with self.assertRaises(InvalidWheelSample):
            self.convert(mount=replace(self.mount, source_frame="scout_wheel_odom"))

    def test_stale_future_zero_and_repeated_timestamps_are_rejected(self):
        for stamp in (0, self.now - 500_000_001, self.now + 100_000_001):
            with self.subTest(stamp=stamp), self.assertRaises(InvalidWheelSample):
                self.convert(replace(self.sample, stamp_ns=stamp))
        for last in (self.now, self.now + 1):
            with self.assertRaises(InvalidWheelSample): self.convert(last_stamp_ns=last)

    def test_timestamp_limits_are_inclusive(self):
        self.convert(replace(self.sample, stamp_ns=self.now - 500_000_000))
        self.convert(replace(self.sample, stamp_ns=self.now + 100_000_000))

    def test_nonfinite_motion_bad_transform_and_nonplanar_motion_are_rejected(self):
        for sample in (
            replace(self.sample, linear=(math.nan, 0.0, 0.0)),
            replace(self.sample, angular=(0.0, 0.0, math.inf)),
            replace(self.sample, linear=(1.0, 0.1, 0.0)),
        ):
            with self.assertRaises(InvalidWheelSample): self.convert(sample)
        for mount in (
            replace(self.mount, rotation_xyzw=(0.0, 0.0, 0.0, 0.0)),
            replace(self.mount, translation=(math.inf, 0.0, 0.0)),
            replace(self.mount, target_frame="odom"),
        ):
            with self.assertRaises(InvalidWheelSample): self.convert(mount=mount)

    def test_invalid_configuration_is_rejected(self):
        for config in (
            replace(self.config, default_linear_stddev=0.0),
            replace(self.config, max_age_sec=math.nan),
            replace(self.config, invalid_covariance_policy="ignore"),
        ):
            with self.assertRaises(ValueError): self.convert(config=config)


if __name__ == "__main__":
    unittest.main()
