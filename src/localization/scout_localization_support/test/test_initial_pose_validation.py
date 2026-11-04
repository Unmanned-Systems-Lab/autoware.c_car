import math
import unittest

from scout_localization_support.initial_pose_validation import validate_initial_pose


class InitialPoseValidationTest(unittest.TestCase):
    def setUp(self):
        self.covariance = [0.0] * 36
        self.covariance[0] = self.covariance[7] = 1.0
        self.covariance[35] = 0.25

    def validate(self, frame="map", position=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 1.0), covariance=None):
        validate_initial_pose(frame, position, quaternion,
                              self.covariance if covariance is None else covariance)

    def test_rviz_planar_covariance_is_accepted(self):
        self.validate()

    def test_correlated_psd_covariance_is_accepted(self):
        self.covariance[1] = self.covariance[6] = -1.0
        self.validate()

    def test_indefinite_covariance_with_positive_diagonal_is_rejected(self):
        self.covariance[1] = self.covariance[6] = 2.0
        with self.assertRaises(ValueError): self.validate()

    def test_zero_variance_cannot_have_nonzero_correlation(self):
        self.covariance[2] = self.covariance[12] = 0.1
        with self.assertRaises(ValueError): self.validate()

    def test_invalid_geometry_is_rejected(self):
        for kwargs in (
            {"frame": "lio_odom"}, {"position": (math.nan, 0.0, 0.0)},
            {"quaternion": (0.0, 0.0, 0.0, 0.0)},
            {"quaternion": (0.0, 0.0, math.inf, 1.0)},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError): self.validate(**kwargs)

    def test_asymmetric_nonfinite_and_negative_covariance_are_rejected(self):
        for kind in ("asymmetric", "nan", "negative", "length"):
            covariance = list(self.covariance)
            if kind == "asymmetric": covariance[1] = 0.1
            if kind == "nan": covariance[0] = math.nan
            if kind == "negative": covariance[0] = -1.0
            if kind == "length": covariance.pop()
            with self.subTest(kind=kind), self.assertRaises(ValueError): self.validate(covariance=covariance)


if __name__ == "__main__":
    unittest.main()
