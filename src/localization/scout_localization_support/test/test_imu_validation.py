from dataclasses import replace
import math
import unittest

from scout_localization_support.imu_validation import ImuValidationConfig, validate_imu_sample


class ImuValidationTest(unittest.TestCase):
    def setUp(self):
        self.config = ImuValidationConfig()
        self.data = dict(
            stamp_ns=10_000_000_000, now_ns=10_000_000_000, frame_id="zed2i_imu_link",
            orientation=(0.0, 0.0, 0.0, 1.0), angular_velocity=(0.1, -0.2, 0.3),
            acceleration=(0.0, 0.0, 9.81), orientation_covariance=[0.0] * 9,
            angular_velocity_covariance=[0.0] * 9, acceleration_covariance=[0.0] * 9,
            config=self.config,
        )

    def validate(self, **changes):
        return validate_imu_sample(**dict(self.data, **changes))

    def test_missing_covariance_gets_configured_positive_noise(self):
        covariance, source = self.validate()
        self.assertEqual(source, "configured_zero_covariance")
        self.assertAlmostEqual(covariance[0], 0.03 ** 2)
        self.assertEqual(covariance[0], covariance[4])
        self.assertEqual(covariance[0], covariance[8])

    def test_valid_calibrated_correlated_covariance_is_preserved(self):
        raw = [0.04, 0.01, 0.0, 0.01, 0.09, 0.0, 0.0, 0.0, 0.16]
        covariance, source = self.validate(angular_velocity_covariance=raw)
        self.assertEqual(covariance, tuple(raw))
        self.assertEqual(source, "input")

    def test_missing_orientation_marker_is_accepted_without_fake_attitude(self):
        self.validate(orientation=(0.0, 0.0, 0.0, 0.0), orientation_covariance=[-1.0] + [0.0] * 8)

    def test_unavailable_gyro_is_not_synthesized(self):
        with self.assertRaises(ValueError):
            self.validate(angular_velocity_covariance=[-1.0] + [0.0] * 8)

    def test_invalid_covariance_is_rejected(self):
        for raw in (
            [math.nan] * 9,
            [-0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            [0.01, 0.1, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            [0.01, 0.02, 0.0, 0.02, 0.01, 0.0, 0.0, 0.0, 0.01],
            [0.0] * 8,
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.validate(angular_velocity_covariance=raw)

    def test_stale_future_zero_and_repeated_stamp_are_rejected(self):
        for stamp in (0, 9_499_999_999, 10_100_000_001):
            with self.assertRaises(ValueError): self.validate(stamp_ns=stamp)
        with self.assertRaises(ValueError): self.validate(last_stamp_ns=10_000_000_000)

    def test_nonfinite_readings_bad_orientation_and_wrong_frame_are_rejected(self):
        for change in (
            {"angular_velocity": (math.nan, 0.0, 0.0)},
            {"acceleration": (0.0, 0.0, math.inf)},
            {"orientation": (0.0, 0.0, 0.0, 0.0)},
            {"frame_id": ""},
            {"config": replace(self.config, expected_frame="FP_POI")},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError): self.validate(**change)

    def test_small_reported_variance_is_not_perfect_measurement(self):
        raw = [1e-20, 0.0, 0.0, 0.0, 1e-20, 0.0, 0.0, 0.0, 1e-20]
        covariance, _ = self.validate(angular_velocity_covariance=raw)
        self.assertGreaterEqual(covariance[0], self.config.minimum_gyro_stddev ** 2)


if __name__ == "__main__":
    unittest.main()
