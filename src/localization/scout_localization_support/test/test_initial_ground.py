"""Ground seeding must use spatially supported map evidence, never guessed zero."""

import json
import math
import unittest

import numpy as np

from scout_localization_support.initial_ground import estimate_initial_ground


def floor_points(rng, count=18000, *, height=-.86, slope_x=0., center=(0., 0.), stripe=False):
    x = rng.uniform(-3.1, 3.1, count)
    y = rng.uniform(-.08 if stripe else -3.1, .08 if stripe else 3.1, count)
    z = height+slope_x*x+rng.normal(0., .006, count)
    return np.column_stack((x+center[0], y+center[1], z))


def wall_points(rng, count=50000):
    along = rng.uniform(-3., 3., count)
    z = rng.uniform(-1.5, 2.5, count)
    side = rng.integers(0, 4, count)
    cross = np.where(side % 2 == 0, -2.5, 2.5)+rng.normal(0, .004, count)
    return np.column_stack((np.where(side < 2, cross, along),
                            np.where(side < 2, along, cross), z))


class InitialGroundTest(unittest.TestCase):
    def test_negative_floor_is_measured_below_base_and_seed_is_fixed_10m(self):
        result = estimate_initial_ground(floor_points(np.random.default_rng(1)), [0., 0., -.76])
        self.assertAlmostEqual(result['height'], -.86, delta=.004)
        self.assertNotEqual(result['height'], -.76)
        self.assertEqual(result['center_xy'], [0., 0.])
        self.assertEqual(result['bounds_xy'], [[-5., -5.], [5., 5.]])
        self.assertEqual(result['method'], 'pcd_lower_envelope_plane_v1')
        self.assertTrue(result['quality']['accepted'])
        self.assertGreater(result['quality']['inner_cells'], 4)
        self.assertLessEqual(result['quality']['coverage_fraction'], 1.)
        json.dumps(result, allow_nan=False)

    def test_dense_walls_and_ceiling_do_not_outvote_floor(self):
        rng = np.random.default_rng(2)
        floor = floor_points(rng, count=10000)
        walls = wall_points(rng, count=100000)
        ceiling = floor_points(rng, count=35000, height=1.8)
        result = estimate_initial_ground(np.vstack((walls, ceiling, floor)), [0., 0., -.76])
        self.assertAlmostEqual(result['height'], -.86, delta=.006)
        self.assertGreater(result['evidence']['input_points'], 100000)

    def test_extremely_low_outliers_cannot_pull_ground_down(self):
        rng = np.random.default_rng(3)
        floor = floor_points(rng)
        very_low = floor_points(rng, count=500, height=-100.)
        low_in_interval = floor_points(rng, count=250, height=-1.35)
        nonfinite = np.array([[math.nan, 0., -.86], [0., math.inf, -.86]])
        result = estimate_initial_ground(np.vstack((floor, very_low, low_in_interval, nonfinite)), [0., 0., -.76])
        self.assertAlmostEqual(result['height'], -.86, delta=.005)
        self.assertLess(result['evidence']['finite_points'], result['evidence']['input_points'])

    def test_only_walls_or_ceiling_is_rejected_without_zero_fallback(self):
        rng = np.random.default_rng(4)
        for points in [wall_points(rng), floor_points(rng, height=1.8), np.empty((0, 3))]:
            with self.subTest(points=len(points)):
                with self.assertRaises(ValueError):
                    estimate_initial_ground(points, [0., 0., -.76])

    def test_large_tilt_is_rejected_instead_of_flattened(self):
        points = floor_points(np.random.default_rng(5), slope_x=math.tan(math.radians(9.)))
        with self.assertRaisesRegex(ValueError, 'tilt'):
            estimate_initial_ground(points, [0., 0., -.76])

    def test_near_horizontal_plane_returns_only_height_at_start_xy(self):
        points = floor_points(np.random.default_rng(6), slope_x=.015, center=(12., -7.))
        result = estimate_initial_ground(points, [12., -7., -.76], length_x=8., length_y=6.)
        self.assertAlmostEqual(result['height'], -.86, delta=.005)
        self.assertEqual(result['center_xy'], [12., -7.])
        self.assertEqual(result['bounds_xy'], [[8., -10.], [16., -4.]])
        self.assertNotIn('coefficients', result)
        self.assertGreater(result['quality']['tilt_deg'], .3)
        self.assertLess(result['quality']['tilt_deg'], 2.)

    def test_thin_floor_strip_cannot_define_a_broad_starting_plane(self):
        points = floor_points(np.random.default_rng(7), stripe=True)
        with self.assertRaises(ValueError):
            estimate_initial_ground(points, [0., 0., -.76])

    def test_floor_far_from_start_cannot_fill_missing_nearby_evidence(self):
        points = floor_points(np.random.default_rng(8))
        distance = np.linalg.norm(points[:, :2], axis=1)
        points = points[distance > 1.7]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            estimate_initial_ground(points, [0., 0., -.76])

    def test_valid_observed_zero_height_is_allowed_but_not_a_fallback(self):
        points = floor_points(np.random.default_rng(9), height=0.)
        result = estimate_initial_ground(points, [0., 0., .1])
        self.assertAlmostEqual(result['height'], 0., delta=.004)

    def test_malformed_inputs_and_weak_limits_are_rejected(self):
        points = floor_points(np.random.default_rng(10))
        for data, base, kwargs in [
            (np.zeros((100, 4)), [0., 0., -.76], {}),
            (points, [0., 0., math.nan], {}),
            (points, [0., 0., -.76], {'max_tilt_deg': 0.}),
            (points, [0., 0., -.76], {'min_cells': 2}),
            (points, [0., 0., -.76], {'min_ground_below_base': .8}),
            (points, [0., 0., -.76], {'length_x': math.inf}),
        ]:
            with self.assertRaises(ValueError):
                estimate_initial_ground(data, base, **kwargs)


if __name__ == '__main__':
    unittest.main()
