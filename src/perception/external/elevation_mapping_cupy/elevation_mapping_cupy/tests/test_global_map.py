import numpy as np
import pytest
import threading
import sqlite3

from elevation_mapping_cupy.global_map import (
    GlobalMapAccumulator,
    downsample_elevation_blocks,
    install_commit_tracking,
)


def _window_centers(window):
    rows, cols = window.elevation_yx.shape
    x = window.center_x + (np.arange(cols) + 0.5 - cols / 2) * window.resolution
    y = window.center_y + (np.arange(rows) + 0.5 - rows / 2) * window.resolution
    return x, y


def test_fixed_prior_crosses_tiles_without_allocating_or_counting_observations():
    bounds = (-5.0, -5.0, 5.0, 5.0)
    global_map = _make_global(prior_bounds=bounds, publish_unobserved_prior=True,
                              initial_elevation=-0.8)
    window = global_map.extract_window((0, 0), 12, 12)
    x, y = _window_centers(window)
    expected = ((y >= -5) & (y < 5))[:, None] & ((x >= -5) & (x < 5))[None, :]
    assert expected.sum() == 100
    np.testing.assert_array_equal(np.isfinite(window.elevation_yx), expected)
    assert (window.elevation_yx[expected] == np.float32(-0.8)).all()
    assert (window.variance_yx[expected] == 10000).all()
    assert np.isnan(window.variance_yx[~expected]).all()
    assert (window.observation_count_yx == 0).all()
    assert (window.last_update_revision_yx == 0).all()
    assert global_map.tile_count() == 0
    assert not global_map.has_observations
    global_map.close()


def test_fixed_prior_uses_global_cell_centers_and_half_open_edges():
    global_map = _make_global(prior_bounds=(-0.5, -0.5, 0.5, 0.5),
                              publish_unobserved_prior=True)
    for center in ((0, 0), (1, 0), (0, 1)):
        window = global_map.extract_window(center, 4, 4)
        x, y = _window_centers(window)
        expected = ((y >= -0.5) & (y < 0.5))[:, None] & ((x >= -0.5) & (x < 0.5))[None, :]
        np.testing.assert_array_equal(np.isfinite(window.elevation_yx), expected)
        assert expected.sum() == 1  # Includes min edge; excludes max edge.
    global_map.close()


def test_robot_following_crop_does_not_extend_fixed_start_prior():
    global_map = _make_global(prior_bounds=(-5, -5, 5, 5), publish_unobserved_prior=True)
    assert np.isfinite(global_map.extract_window((0, 0), 10, 10).elevation_yx).sum() == 100
    assert np.isfinite(global_map.extract_window((5, 0), 10, 10).elevation_yx).sum() == 50
    assert np.isnan(global_map.extract_window((20, 0), 10, 10).elevation_yx).all()
    assert global_map.tile_count() == 0
    global_map.close()


def test_measurements_replace_prior_and_remain_visible_outside_bounds():
    global_map = _make_global(prior_bounds=(-1, -1, 1, 1), publish_unobserved_prior=True,
                              initial_elevation=-0.8)
    heights = np.array([[2., np.nan], [np.nan, np.nan]], np.float32)
    global_map.fuse_snapshot((0, 0), heights, np.ones_like(heights))
    window = global_map.extract_window((0, 0), 2, 2)
    assert window.elevation_yx[0, 0] == pytest.approx(-0.8 + (2.8 * 10000 / 10001))
    assert window.observation_count_yx[0, 0] == 1
    assert window.elevation_yx[1, 1] == pytest.approx(-0.8)
    assert window.observation_count_yx[1, 1] == 0
    outside = np.array([[np.nan, 3.], [np.nan, np.nan]], np.float32)
    global_map.fuse_snapshot((5, 0), outside, np.ones_like(outside))
    outside_window = global_map.extract_window((5, 0), 2, 2)
    assert np.isfinite(outside_window.elevation_yx).sum() == 1
    assert outside_window.observation_count_yx[0, 1] == 1
    assert outside_window.elevation_yx[0, 1] == pytest.approx(-0.8 + (3.8 * 10000 / 10001))
    assert np.isnan(outside_window.elevation_yx[1]).all()
    global_map.close()


def test_bounds_do_not_enable_disabled_prior_or_change_legacy_unbounded_prior():
    bounded = _make_global(prior_bounds=(-1, -1, 1, 1))
    assert np.isnan(bounded.extract_window((0, 0), 4, 4).elevation_yx).all()
    bounded.close()
    legacy = _make_global(publish_unobserved_prior=True)
    assert (legacy.extract_window((100, -100), 4, 4).elevation_yx == 0).all()
    legacy.close()


@pytest.mark.parametrize('bounds', [
    (), (0, 0, 1), (0, 0, 0, 1), (0, 1, 1, 0),
    (0, 0, float('nan'), 1), (0, 0, 1, float('inf')), 'bad',
])
def test_invalid_prior_bounds_are_rejected(bounds):
    with pytest.raises(ValueError, match='prior_bounds'):
        _make_global(prior_bounds=bounds)


def test_bounded_prior_database_recovers_only_identical_fixed_bounds(tmp_path):
    path = str(tmp_path / 'bounded.sqlite3')
    parameters = dict(storage_path=path, prior_bounds=(-5, -5, 5, 5),
                      publish_unobserved_prior=True, initial_elevation=-0.8)
    global_map = _make_global(**parameters)
    global_map.fuse_snapshot((0, 0), np.ones((2, 2), np.float32))
    global_map.close()
    restored = _make_global(**parameters, resume_from_disk=True)
    assert restored.prior_bounds == (-5., -5., 5., 5.)
    assert (restored.extract_window((0, 0), 2, 2).observation_count_yx == 1).all()
    assert np.isnan(restored.extract_window((20, 0), 2, 2).elevation_yx).all()
    restored.close()
    for bounds in ((-6, -5, 5, 5), None):
        changed = dict(parameters, prior_bounds=bounds, resume_from_disk=True)
        with pytest.raises(ValueError, match='prior_bounds'):
            _make_global(**changed)


def test_original_four_layer_database_without_bounds_is_compatible_only_with_none(tmp_path):
    path = str(tmp_path / 'legacy.sqlite3')
    legacy = _make_global(storage_path=path)
    legacy.fuse_snapshot((0, 0), np.ones((2, 2), np.float32))
    legacy.close()
    # Emulate the original metadata rather than changing the tile data format.
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM metadata WHERE key = 'prior_bounds'")
    restored = _make_global(storage_path=path, resume_from_disk=True)
    assert restored.prior_bounds is None
    assert (restored.extract_window((0, 0), 2, 2).observation_count_yx == 1).all()
    restored.close()
    with pytest.raises(ValueError, match='prior_bounds'):
        _make_global(storage_path=path, resume_from_disk=True, prior_bounds=(-5, -5, 5, 5))


def _first_posterior(height, variance=1.0):
    # Independent scalar precision oracle for the configured h0=0, P0=10000.
    return (np.asarray(height, dtype=np.float64) / (1.0 + variance / 10000.0)).astype(np.float32)


def _make_global(**overrides):
    params = {
        "resolution": 1.0,
        "tile_size_cells": 2,
        "max_tiles_in_memory": 8,
        "storage_path": "",
        "resume_from_disk": False,
        "map_frame": "map",
    }
    params.update(overrides)
    result = GlobalMapAccumulator(**params)
    result.initialize_lattice((2, 2))
    return result


def test_variance_weighted_fusion_and_nan_does_not_erase_history():
    global_map = _make_global()
    first = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    variance = np.ones_like(first)
    result = global_map.fuse_snapshot((0.0, 0.0), first, variance)

    assert result.written_cells == 4
    assert result.touched_tiles == 4  # The origin crosses four signed tiles.
    assert set(global_map.all_tile_keys()) == {(-1, -1), (0, -1), (-1, 0), (0, 0)}

    second = np.array([[1.2, np.nan], [np.nan, 3.8]], dtype=np.float32)
    global_map.fuse_snapshot((0.0, 0.0), second, variance)
    window = global_map.extract_window((0.0, 0.0), 2.0, 2.0)
    expected = np.array([[2.2 / 2.0001, 2.0 / 1.0001],
                         [3.0 / 1.0001, 7.8 / 2.0001]], dtype=np.float32)
    assert np.allclose(window.elevation_yx, expected)
    assert np.allclose(
        window.variance_yx,
        np.array([[1 / 2.0001, 1 / 1.0001], [1 / 1.0001, 1 / 2.0001]], dtype=np.float32),
    )
    assert np.array_equal(
        window.observation_count_yx,
        np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32),
    )


def test_only_explicitly_updated_cells_are_fused():
    global_map = _make_global()
    first = np.ones((2, 2), dtype=np.float32)
    variance = np.ones_like(first)
    global_map.fuse_snapshot((0.0, 0.0), first, variance)

    second = np.full((2, 2), 1.2, dtype=np.float32)
    updated = np.array([[True, False], [False, False]])
    result = global_map.fuse_snapshot(
        (0.0, 0.0), second, variance, updated
    )
    window = global_map.extract_window((0.0, 0.0), 2.0, 2.0)

    assert result.valid_input_cells == 1
    assert result.written_cells == 1
    assert np.allclose(
        window.elevation_yx,
        np.array([[2.2 / 2.0001, 1 / 1.0001], [1 / 1.0001, 1 / 1.0001]], dtype=np.float32),
    )


def test_large_conflicting_height_is_rejected():
    global_map = _make_global()
    variance = np.full((2, 2), 0.01, dtype=np.float32)
    global_map.fuse_snapshot(
        (0.0, 0.0), np.ones((2, 2), dtype=np.float32), variance
    )
    result = global_map.fuse_snapshot(
        (0.0, 0.0), np.full((2, 2), 2.0, dtype=np.float32), variance
    )

    assert result.written_cells == 0
    assert result.rejected_cells == 4
    assert np.all(
        global_map.extract_window((0.0, 0.0), 2.0, 2.0).elevation_yx
        == _first_posterior(1.0, variance=0.01)
    )


@pytest.mark.parametrize('new_height', [-.75, 1.5])
def test_consistent_real_conflicts_correct_old_height_after_ten_updates(new_height):
    m = _make_global(conflict_confirmations=10)
    var = np.full((2, 2), .01, np.float32)
    m.fuse_snapshot((0, 0), np.full((2, 2), .2, np.float32), var)
    for _ in range(9):
        result = m.fuse_snapshot((0, 0), np.full((2, 2), new_height, np.float32), var)
        assert result.rejected_cells == 4 and result.corrected_cells == 0
        assert m.extract_window((0, 0), 2, 2).elevation_yx[0, 0] == pytest.approx(.2, abs=.001)
    result = m.fuse_snapshot((0, 0), np.full((2, 2), new_height, np.float32), var)
    assert result.corrected_cells == 4 and result.rejected_cells == 0
    window = m.extract_window((0, 0), 2, 2)
    assert np.allclose(window.elevation_yx, new_height)
    assert np.allclose(window.variance_yx, .01)  # no false independent-frame certainty
    assert (window.observation_count_yx == 2).all()


def test_single_spike_oscillating_conflicts_and_unscanned_cells_do_not_erase_history():
    m = _make_global(conflict_confirmations=3)
    var = np.full((2, 2), .01, np.float32)
    old = np.full((2, 2), .2, np.float32)
    m.fuse_snapshot((0, 0), old, var)
    for k in range(12):
        result = m.fuse_snapshot((0, 0), np.full((2, 2), -.75 if k % 2 else 1.5, np.float32), var)
        assert result.corrected_cells == 0
    for _ in range(10):
        m.fuse_snapshot((0, 0), np.full((2, 2), -.75, np.float32), var, np.zeros((2, 2), bool))
    assert np.allclose(m.extract_window((0, 0), 2, 2).elevation_yx, .2, atol=.001)


def test_accepted_old_height_resets_conflict_evidence():
    m = _make_global(conflict_confirmations=3)
    var = np.full((2, 2), .01, np.float32)
    old = np.full((2, 2), .2, np.float32)
    low = np.full((2, 2), -.75, np.float32)
    m.fuse_snapshot((0, 0), old, var)
    for _ in range(2): m.fuse_snapshot((0, 0), low, var)
    m.fuse_snapshot((0, 0), old, var)
    for _ in range(2):
        assert m.fuse_snapshot((0, 0), low, var).corrected_cells == 0
    assert m.fuse_snapshot((0, 0), low, var).corrected_cells == 4


def test_conflict_gap_expires_old_votes_and_different_cells_cannot_supply_votes():
    m = _make_global(conflict_confirmations=3, conflict_max_gap=2)
    var = np.full((2, 2), .01, np.float32)
    old = np.full((2, 2), .2, np.float32)
    low = np.full((2, 2), -.75, np.float32)
    mask = np.array([[True, False], [False, False]])
    m.fuse_snapshot((0, 0), old, var)
    for _ in range(2): m.fuse_snapshot((0, 0), low, var, mask)
    for _ in range(3): m.fuse_snapshot((0, 0), old, var, ~mask)
    assert m.fuse_snapshot((0, 0), low, var, mask).corrected_cells == 0
    assert m.fuse_snapshot((0, 0), low, var, mask).corrected_cells == 0
    assert m.fuse_snapshot((0, 0), low, var, mask).corrected_cells == 1


def test_corrected_height_persists_but_partial_conflict_votes_do_not(tmp_path):
    args = dict(storage_path=str(tmp_path / 'map.sqlite3'), conflict_confirmations=3)
    m = _make_global(**args)
    var = np.full((2, 2), .01, np.float32)
    old = np.full((2, 2), .2, np.float32)
    low = np.full((2, 2), -.75, np.float32)
    m.fuse_snapshot((0, 0), old, var)
    for _ in range(2): m.fuse_snapshot((0, 0), low, var)
    m.close()
    m = _make_global(**args, resume_from_disk=True)
    for _ in range(2): assert m.fuse_snapshot((0, 0), low, var).corrected_cells == 0
    assert m.fuse_snapshot((0, 0), low, var).corrected_cells == 4
    m.close()
    m = _make_global(**args, resume_from_disk=True)
    assert np.allclose(m.extract_window((0, 0), 2, 2).elevation_yx, -.75)
    m.close()


def test_far_jump_allocates_only_observed_tiles_not_intermediate_rectangle():
    global_map = _make_global(tile_size_cells=4)
    local = np.ones((2, 2), dtype=np.float32)
    global_map.fuse_snapshot((0.0, 0.0), local)
    keys_near_origin = set(global_map.all_tile_keys())
    global_map.fuse_snapshot((100.0, -100.0), local * 7.0)
    keys_after_jump = set(global_map.all_tile_keys())

    assert len(keys_near_origin) == 4
    assert len(keys_after_jump) == 8
    assert (12, 0) not in keys_after_jump


def test_window_xy_axes_and_gridmap_conversion_use_independent_oracle():
    global_map = _make_global(tile_size_cells=4)
    natural = np.array([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32)
    global_map.fuse_snapshot((0.0, 0.0), natural)
    window = global_map.extract_window((0.0, 0.0), 2.0, 2.0)

    assert np.array_equal(window.elevation_yx, _first_posterior(natural))
    # Natural storage is row=+Y,col=+X. GridMap's matrix axes are row=-X,
    # col=-Y, so this conversion must match the rolling-map publisher.
    expected_grid_map = np.array(
        [[22.0, 12.0], [21.0, 11.0]], dtype=np.float32
    )
    assert np.array_equal(window.elevation_grid_map_copy(), _first_posterior(expected_grid_map))


def test_local_window_is_the_same_xy_crop_as_the_global_window():
    global_map = GlobalMapAccumulator(
        resolution=1.0,
        tile_size_cells=4,
        max_tiles_in_memory=16,
    )
    natural = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    global_map.fuse_snapshot((0.0, 0.0), natural)

    full = global_map.extract_window((0.0, 0.0), 8.0, 6.0)
    local = global_map.extract_window((1.0, -1.0), 4.0, 2.0)

    expected_full = np.flip(_first_posterior(natural).T, axis=(0, 1))
    natural_local = natural[1:3, 3:7]
    expected_local = np.flip(_first_posterior(natural_local).T, axis=(0, 1))

    assert np.array_equal(full.elevation_grid_map_copy(), expected_full)
    assert np.array_equal(local.elevation_grid_map_copy(), expected_local)
    # A 4 m X by 2 m Y window must become a 4-row by 2-column GridMap layer.
    assert local.elevation_grid_map_copy().shape == (4, 2)


@pytest.mark.parametrize("shape", [(3, 3), (2, 3), (3, 2), (5, 4)])
def test_odd_and_mixed_parity_lattices_round_trip(shape):
    global_map = GlobalMapAccumulator(
        resolution=1.0,
        tile_size_cells=3,
        max_tiles_in_memory=16,
    )
    global_map.initialize_lattice(shape)
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1.0
    center = (-6.0, 4.0)

    global_map.fuse_snapshot(center, source)
    restored = global_map.extract_window(center, shape[1], shape[0])

    assert restored.elevation_yx.shape == shape
    assert np.array_equal(restored.elevation_yx, _first_posterior(source))


def test_far_negative_tile_boundaries_keep_exact_values():
    global_map = GlobalMapAccumulator(
        resolution=1.0,
        tile_size_cells=3,
        max_tiles_in_memory=16,
    )
    global_map.initialize_lattice((2, 3))
    source = np.array([[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]], dtype=np.float32)

    global_map.fuse_snapshot((-6.0, -3.0), source)

    assert set(global_map.all_tile_keys()) == {(-3, -2), (-2, -2), (-3, -1), (-2, -1)}
    restored = global_map.extract_window((-6.0, -3.0), 3.0, 2.0)
    assert np.array_equal(restored.elevation_yx, _first_posterior(source))


def test_float32_center_drift_is_snapped_to_integer_cells():
    global_map = GlobalMapAccumulator(
        resolution=0.04,
        tile_size_cells=128,
        max_tiles_in_memory=16,
    )
    global_map.initialize_lattice((250, 250))
    # Reproduces float32 accumulation after hundreds of 0.04 m shifts.
    drifted_center = np.float32(18.36004)
    start_x, start_y = global_map.validate_snapshot_geometry(
        (drifted_center, np.float32(-18.36004)), (250, 250)
    )
    assert start_x == 459 - 125
    assert start_y == -459 - 125


def test_genuine_fractional_cell_misalignment_is_rejected():
    global_map = _make_global()
    with pytest.raises(ValueError, match="too far"):
        global_map.fuse_snapshot((0.4, 0.0), np.ones((2, 2), dtype=np.float32))


def test_empty_window_read_does_not_allocate_tiles():
    global_map = _make_global(tile_size_cells=4)
    assert np.isnan(
        global_map.extract_window((100.0, -100.0), 8.0, 8.0).elevation_yx
    ).all()
    assert global_map.tile_count() == 0


def test_published_prior_is_zero_with_large_variance_but_not_an_observation():
    global_map = _make_global(publish_unobserved_prior=True)
    initial = global_map.extract_window((0, 0), 10, 10)
    assert (initial.elevation_yx == 0).all()
    assert (initial.variance_yx == 10000).all()
    assert (initial.observation_count_yx == 0).all()
    assert not global_map.has_observations
    assert global_map.tile_count() == 0
    height = np.array([[2., np.nan], [np.nan, np.nan]], np.float32)
    global_map.fuse_snapshot((0, 0), height, np.ones((2, 2), np.float32))
    updated = global_map.extract_window((0, 0), 2, 2)
    assert updated.elevation_yx[0, 0] == pytest.approx(2 / 1.0001)
    assert updated.observation_count_yx[0, 0] == 1
    assert updated.elevation_yx[1, 1] == 0
    assert updated.variance_yx[1, 1] == 10000
    assert updated.observation_count_yx[1, 1] == 0


def test_exact_fifteen_metre_view_keeps_coordinates_on_even_lattice():
    global_map = _make_global(resolution=.04, initial_elevation=-1., publish_unobserved_prior=True)
    initial = global_map.extract_window((0, 0), 15, 15, exact_size=True)
    assert initial.elevation_yx.shape == (375, 375)
    assert initial.length_x == 15 and initial.length_y == 15
    assert initial.center_x == pytest.approx(.02)
    assert initial.center_y == pytest.approx(.02)
    assert (initial.elevation_yx == -1).all()
    assert (initial.variance_yx == 10000).all()
    source = np.array([[np.nan, np.nan], [np.nan, -.75]], np.float32)
    global_map.fuse_snapshot((0, 0), source, np.ones((2, 2), np.float32))
    updated = global_map.extract_window((0, 0), 15, 15, exact_size=True)
    assert updated.elevation_yx[187, 187] == pytest.approx(-1 + .25 / 1.0001)
    assert updated.observation_count_yx[187, 187] == 1
    assert updated.center_x + (187.5 - 375/2)*.04 == pytest.approx(.02)
    assert (updated.elevation_yx[updated.observation_count_yx == 0] == -1).all()


def test_only_lidar_update_mask_changes_prior_even_when_other_heights_are_finite():
    global_map = _make_global(publish_unobserved_prior=True)
    values = np.full((2, 2), 7., np.float32)
    updated = np.array([[False, False], [False, True]])
    global_map.fuse_snapshot((0, 0), values, np.ones((2, 2), np.float32), updated)
    first = global_map.extract_window((0, 0), 2, 2)
    assert (first.elevation_yx[~updated] == 0).all()
    assert (first.variance_yx[~updated] == 10000).all()
    assert (first.observation_count_yx[~updated] == 0).all()
    measured = first.elevation_yx[1, 1]
    values[:] = 9.
    updated[:] = False
    updated[0, 0] = True
    global_map.fuse_snapshot((0, 0), values, np.ones((2, 2), np.float32), updated)
    second = global_map.extract_window((0, 0), 2, 2)
    assert second.elevation_yx[1, 1] == measured
    assert second.observation_count_yx[1, 1] == 1
    assert second.elevation_yx[0, 0] == pytest.approx(9 / 1.0001)
    assert second.elevation_yx[0, 1] == 0
    assert second.variance_yx[0, 1] == 10000


def test_block_downsampling_preserves_thin_observations_and_block_centers():
    tile = np.full((4, 4), np.nan, dtype=np.float32)
    tile[1, 1] = 2.0  # Not on the old [::2, ::2] sampling lattice.
    tile[2, 2] = 4.0
    tile[3, 3] = 8.0

    x, y, z = downsample_elevation_blocks(tile, stride=2)

    assert np.array_equal(x, np.array([0.5, 2.5], dtype=np.float32))
    assert np.array_equal(y, np.array([0.5, 2.5], dtype=np.float32))
    assert np.allclose(z, np.array([2.0, 6.0], dtype=np.float32))


def test_extract_window_preserves_old_and_new_distant_areas():
    global_map = _make_global(tile_size_cells=4)
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    global_map.fuse_snapshot((6.0, 0.0), np.full((2, 2), 7.0, dtype=np.float32))

    old_window = global_map.extract_window((0.0, 0.0), 2.0, 2.0)
    new_window = global_map.extract_window((6.0, 0.0), 2.0, 2.0)
    assert np.all(old_window.elevation_yx == _first_posterior(1.0))
    assert np.all(new_window.elevation_yx == _first_posterior(7.0))


def test_sqlite_eviction_reload_and_resume(tmp_path):
    db_path = tmp_path / "global_tiles.sqlite3"
    global_map = _make_global(
        tile_size_cells=2,
        max_tiles_in_memory=1,
        storage_path=str(db_path),
    )
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    global_map.fuse_snapshot((10.0, 10.0), np.full((2, 2), 5.0, dtype=np.float32))
    global_map.flush(wait=True)
    assert global_map.cached_tile_count <= 1
    assert global_map.tile_count() == 8
    global_map.close()

    resumed = GlobalMapAccumulator(
        resolution=1.0,
        tile_size_cells=2,
        max_tiles_in_memory=1,
        storage_path=str(db_path),
        resume_from_disk=True,
        map_frame="map",
    )
    resumed.initialize_lattice((2, 2))
    assert np.all(resumed.extract_window((0.0, 0.0), 2.0, 2.0).elevation_yx == _first_posterior(1.0))
    assert np.all(resumed.extract_window((10.0, 10.0), 2.0, 2.0).elevation_yx == _first_posterior(5.0))
    resumed.fuse_snapshot((0.0, 0.0), np.full((2, 2), 1.2, dtype=np.float32))
    assert np.allclose(resumed.extract_window((0.0, 0.0), 2.0, 2.0).elevation_yx, 2.2 / 2.0001)
    resumed.close()


def test_resume_rejects_geometry_mismatch(tmp_path):
    db_path = tmp_path / "global_tiles.sqlite3"
    global_map = _make_global(storage_path=str(db_path))
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    global_map.close()

    with pytest.raises(ValueError, match="metadata mismatch"):
        GlobalMapAccumulator(
            resolution=0.5,
            tile_size_cells=2,
            max_tiles_in_memory=8,
            storage_path=str(db_path),
            resume_from_disk=True,
            map_frame="map",
        )


def test_resume_rejects_local_lattice_parity_mismatch(tmp_path):
    db_path = tmp_path / "global_tiles.sqlite3"
    global_map = _make_global(storage_path=str(db_path))
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    global_map.close()

    resumed = GlobalMapAccumulator(
        resolution=1.0,
        tile_size_cells=2,
        max_tiles_in_memory=8,
        storage_path=str(db_path),
        resume_from_disk=True,
        map_frame="map",
    )
    with pytest.raises(ValueError, match="parity"):
        resumed.initialize_lattice((3, 2))
    resumed.close()


def test_late_visualizer_can_queue_and_replay_all_tiles():
    global_map = _make_global(tile_size_cells=4)
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    global_map.take_visualization_tiles(100)
    global_map.queue_all_tiles_for_visualization()

    replay = global_map.take_visualization_tiles(100)
    assert {key for key, _ in replay} == set(global_map.all_tile_keys())


def test_clear_removes_ram_and_disk_tiles(tmp_path):
    global_map = _make_global(storage_path=str(tmp_path / "tiles.sqlite3"))
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    global_map.flush(wait=True)
    assert global_map.has_observations
    assert global_map.tile_count() == 4

    global_map.clear()

    assert not global_map.has_observations
    assert global_map.tile_count() == 0
    assert np.isnan(
        global_map.extract_window((0.0, 0.0), 2.0, 2.0).elevation_yx
    ).all()
    global_map.close()


def test_clear_serializes_a_concurrent_fusion_after_the_delete(tmp_path, monkeypatch):
    global_map = _make_global(storage_path=str(tmp_path / "tiles.sqlite3"))
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    entered_clear = threading.Event()
    release_clear = threading.Event()
    original_clear = global_map._backend.clear

    def blocking_clear():
        entered_clear.set()
        assert release_clear.wait(timeout=5.0)
        original_clear()

    monkeypatch.setattr(global_map._backend, "clear", blocking_clear)
    clear_thread = threading.Thread(target=global_map.clear)
    clear_thread.start()
    assert entered_clear.wait(timeout=5.0)

    fuse_done = threading.Event()

    def fuse_after_clear():
        global_map.fuse_snapshot(
            (10.0, 10.0), np.full((2, 2), 7.0, dtype=np.float32)
        )
        fuse_done.set()

    fuse_thread = threading.Thread(target=fuse_after_clear)
    fuse_thread.start()
    assert not fuse_done.wait(timeout=0.1)
    release_clear.set()
    clear_thread.join(timeout=5.0)
    fuse_thread.join(timeout=5.0)

    assert fuse_done.is_set()
    assert np.all(
        global_map.extract_window((10.0, 10.0), 2.0, 2.0).elevation_yx == _first_posterior(7.0)
    )
    global_map.close()


def test_first_measurement_uses_large_prior_without_clipping_it_to_sensor_cap():
    global_map = _make_global(tile_size_cells=4)
    assert global_map.initial_elevation == 0.0
    assert global_map.initial_variance == 10000.0
    height = np.array([[2.0, np.nan], [np.nan, np.nan]], dtype=np.float32)
    variance = np.full((2, 2), 100.0, dtype=np.float32)  # R is capped at 4; P0 is not.
    result = global_map.fuse_snapshot((0.0, 0.0), height, variance)
    window = global_map.extract_window((0.0, 0.0), 2.0, 2.0)
    assert result.written_cells == 1  # First observation is not rejected against h0=0.
    assert window.elevation_yx[0, 0] == pytest.approx(20000.0 / 10004.0)
    assert window.variance_yx[0, 0] == pytest.approx(40000.0 / 10004.0)
    assert window.observation_count_yx[0, 0] == 1
    assert np.isnan(window.elevation_yx[0, 1:]).all()
    assert np.isnan(window.elevation_yx[1]).all()
    # Allocated prior-only cells remain numeric in storage but are hidden from HT/RViz.
    for entry in global_map._tiles.values():
        unknown = entry.data[global_map._OBSERVATION_COUNT] == 0
        assert (entry.data[global_map._ELEVATION][unknown] == 0).all()
        assert (entry.data[global_map._VARIANCE][unknown] == 10000).all()
    finite_visualized = sum(np.isfinite(data).sum()
                            for _, data in global_map.take_visualization_tiles(10))
    assert finite_visualized == 1


def test_resume_rejects_changed_prior_and_restores_same_prior(tmp_path):
    db_path = str(tmp_path / "prior.sqlite3")
    global_map = _make_global(storage_path=db_path)
    global_map.fuse_snapshot((0.0, 0.0), np.ones((2, 2), dtype=np.float32))
    global_map.close()
    with pytest.raises(ValueError, match="initial_variance"):
        _make_global(storage_path=db_path, resume_from_disk=True, initial_variance=1000.0)
    resumed = _make_global(storage_path=db_path, resume_from_disk=True)
    window = resumed.extract_window((0, 0), 2, 2)
    assert np.allclose(window.elevation_yx, 1 / 1.0001)
    assert np.allclose(window.variance_yx, 1 / 1.0001)
    resumed.close()


@pytest.mark.parametrize("parameters", [
    {"initial_elevation": float("nan")}, {"initial_elevation": float("inf")},
    {"initial_variance": 0}, {"initial_variance": -1},
    {"initial_variance": float("nan")}, {"initial_variance": float("inf")},
    {"initial_variance": 0.001},
])
def test_invalid_prior_is_rejected(parameters):
    with pytest.raises(ValueError, match="initial"):
        _make_global(**parameters)


def test_closed_store_rejects_further_writes():
    global_map = _make_global()
    global_map.close()
    with pytest.raises(RuntimeError, match="not open"):
        global_map.fuse_snapshot(
            (0.0, 0.0), np.ones((2, 2), dtype=np.float32)
        )


def test_memory_only_store_fails_at_cache_limit_instead_of_growing_forever():
    global_map = GlobalMapAccumulator(
        resolution=1.0,
        tile_size_cells=2,
        max_tiles_in_memory=1,
    )
    global_map.initialize_lattice((2, 2))
    with pytest.raises(RuntimeError, match="storage_path"):
        global_map.fuse_snapshot(
            (0.0, 0.0), np.ones((2, 2), dtype=np.float32)
        )


def _legacy_commit_database(path, count=0, without_rowid=True):
    """A pre-migration source DB; do not obtain its schema from new production code."""
    connection = sqlite3.connect(path)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    connection.execute('''CREATE TABLE tiles (
        level INTEGER NOT NULL, tile_x INTEGER NOT NULL, tile_y INTEGER NOT NULL,
        revision INTEGER NOT NULL, payload BLOB NOT NULL,
        PRIMARY KEY(level, tile_x, tile_y)
    )''' + (' WITHOUT ROWID' if without_rowid else ''))
    connection.executemany('INSERT INTO metadata(key, value) VALUES (?, ?)', [
        ('schema_version', '2'), ('initial_elevation', '-0.809'),
        ('prior_bounds', '[-5, -5, 5, 5]'),
    ])
    rows = [(i % 2, i - 700, -i, 10000-i, b'unchanged\x00\xff'+str(i).encode())
            for i in range(count)]
    connection.executemany('INSERT INTO tiles VALUES (?, ?, ?, ?, ?)', rows)
    connection.commit()
    connection.close()
    return sorted(rows)


@pytest.mark.parametrize('without_rowid', [True, False])
def test_source_commit_tracking_migrates_legacy_rows_in_bounded_key_pages(tmp_path, without_rowid):
    path = tmp_path / 'legacy.sqlite3'
    original = _legacy_commit_database(path, 1103, without_rowid)
    with sqlite3.connect(path) as connection:
        metadata = connection.execute('SELECT * FROM metadata ORDER BY key').fetchall()
    result = install_commit_tracking(path)
    assert result['backfilled_tiles'] == len(original)
    assert result['clock'] == len(original)
    with sqlite3.connect(path) as connection:
        assert connection.execute('SELECT level, tile_x, tile_y, revision, payload FROM tiles '
                                  'ORDER BY level, tile_x, tile_y').fetchall() == original
        assert connection.execute('SELECT * FROM metadata ORDER BY key').fetchall() == metadata
        sequences = [r[0] for r in connection.execute('SELECT commit_seq FROM tiles ORDER BY commit_seq')]
        assert sequences == list(range(1, len(original)+1))
        assert connection.execute('SELECT seq FROM tile_commit_clock WHERE id=1').fetchone()[0] == len(original)
        assert connection.execute('SELECT COUNT(*) FROM tile_commit_clock').fetchone()[0] == 1


def test_source_commit_tracking_works_on_the_already_open_old_writer_connection(tmp_path):
    path = tmp_path / 'live.sqlite3'
    _legacy_commit_database(path)
    old_writer = sqlite3.connect(path)
    cursor = old_writer.cursor()
    sql = ('INSERT INTO tiles(level,tile_x,tile_y,revision,payload) VALUES(0,?,?,?,?) '
           'ON CONFLICT(level,tile_x,tile_y) DO UPDATE SET '
           'revision=excluded.revision,payload=excluded.payload '
           'WHERE excluded.revision>=tiles.revision')
    cursor.execute(sql, (2, 3, 5, b'before'))
    old_writer.commit()
    first = install_commit_tracking(path)['clock']
    # Reuse the old prepared statement/connection: no worker restart or code reload.
    cursor.execute(sql, (2, 3, 6, b'after'))
    old_writer.commit()
    sequence, revision, payload = old_writer.execute(
        'SELECT commit_seq,revision,payload FROM tiles').fetchone()
    assert sequence == first+1
    assert (revision, payload) == (6, b'after')
    # Preserve the original writer's stale-snapshot rejection semantics.
    cursor.execute(sql, (2, 3, 4, b'stale'))
    old_writer.commit()
    assert old_writer.execute('SELECT commit_seq,revision,payload FROM tiles').fetchone() == (sequence, 6, b'after')
    old_writer.close()


def test_source_commit_tracking_finds_late_commits_with_smaller_fusion_revision(tmp_path):
    path = tmp_path / 'late.sqlite3'
    _legacy_commit_database(path)
    install_commit_tracking(path)
    with sqlite3.connect(path) as writer:
        writer.execute('INSERT INTO tiles(level,tile_x,tile_y,revision,payload) VALUES(0,0,0,200,?)', (b'new',))
    with sqlite3.connect(path) as reader:
        cursor = reader.execute('SELECT MAX(commit_seq) FROM tiles').fetchone()[0]
    with sqlite3.connect(path) as writer:
        writer.execute('INSERT INTO tiles(level,tile_x,tile_y,revision,payload) VALUES(0,1,0,5,?)', (b'late',))
    with sqlite3.connect(path) as reader:
        changed = reader.execute('SELECT tile_x,revision,commit_seq FROM tiles WHERE commit_seq>? '
                                 'ORDER BY commit_seq LIMIT 10', (cursor,)).fetchall()
    assert changed == [(1, 5, cursor+1)]
    with sqlite3.connect(path) as writer:
        writer.execute('UPDATE tiles SET revision=1,payload=? WHERE tile_x=0', (b'committed-later',))
        assert writer.execute('SELECT commit_seq FROM tiles WHERE tile_x=0').fetchone()[0] == cursor+2


def test_source_commit_tracking_is_not_recursive_and_repeated_install_is_idempotent(tmp_path):
    path = tmp_path / 'recursive.sqlite3'
    _legacy_commit_database(path)
    install_commit_tracking(path)
    with sqlite3.connect(path) as writer:
        writer.execute('PRAGMA recursive_triggers=ON')
        writer.execute('INSERT INTO tiles(level,tile_x,tile_y,revision,payload) VALUES(0,0,0,10,?)', (b'a',))
        writer.execute('UPDATE tiles SET payload=? WHERE tile_x=0', (b'b',))
        writer.execute('UPDATE tiles SET revision=11 WHERE tile_x=0')
        assert writer.execute('SELECT seq FROM tile_commit_clock WHERE id=1').fetchone()[0] == 3
        assert writer.execute('SELECT commit_seq FROM tiles').fetchone()[0] == 3
    for _ in range(2):
        result = install_commit_tracking(path)
        assert result['clock'] == 3
        assert result['backfilled_tiles'] == 0
    with sqlite3.connect(path) as writer:
        writer.execute('UPDATE tiles SET payload=?', (b'c',))
        assert writer.execute('SELECT commit_seq FROM tiles').fetchone()[0] == 4


def test_source_commit_tracking_is_transactional_and_only_commits_are_visible(tmp_path):
    path = tmp_path / 'atomic.sqlite3'
    _legacy_commit_database(path)
    install_commit_tracking(path)
    writer = sqlite3.connect(path)
    writer.execute('INSERT INTO tiles(level,tile_x,tile_y,revision,payload) VALUES(0,0,0,10,?)', (b'old',))
    writer.commit()
    writer.execute('BEGIN IMMEDIATE')
    writer.execute('UPDATE tiles SET payload=?', (b'uncommitted',))
    assert writer.execute('SELECT seq FROM tile_commit_clock').fetchone()[0] == 2
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as reader:
        assert reader.execute('SELECT seq FROM tile_commit_clock').fetchone()[0] == 1
        assert reader.execute('SELECT payload,commit_seq FROM tiles').fetchone() == (b'old', 1)
    writer.rollback()
    assert writer.execute('SELECT seq FROM tile_commit_clock').fetchone()[0] == 1
    assert writer.execute('SELECT payload,commit_seq FROM tiles').fetchone() == (b'old', 1)
    writer.close()


def test_source_commit_tracking_reader_uses_indexed_readonly_keyset_pages(tmp_path):
    path = tmp_path / 'indexed.sqlite3'
    _legacy_commit_database(path, 1103)
    install_commit_tracking(path)
    with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as reader:
        reader.execute('BEGIN')
        watermark = reader.execute('SELECT seq FROM tile_commit_clock WHERE id=1').fetchone()[0]
        query = ('SELECT level,tile_x,tile_y,revision,payload,commit_seq FROM tiles '
                 'WHERE commit_seq>? ORDER BY commit_seq LIMIT ?')
        plan = ' '.join(str(row) for row in reader.execute('EXPLAIN QUERY PLAN '+query, (0, 7)))
        assert 'tiles_commit_seq_idx' in plan
        assert 'SEARCH' in plan and 'SCAN tiles' not in plan
        cursor, seen = 0, []
        while True:
            rows = reader.execute(query, (cursor, 7)).fetchall()
            if not rows:
                break
            assert len(rows) <= 7
            seen += [row[-1] for row in rows]
            cursor = rows[-1][-1]
        assert seen == list(range(1, watermark+1))
        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            reader.execute('UPDATE tiles SET payload=?', (b'forbidden',))


def test_source_commit_tracking_busy_install_leaves_no_partial_schema(tmp_path):
    path = tmp_path / 'busy.sqlite3'
    _legacy_commit_database(path, 3)
    writer = sqlite3.connect(path)
    writer.execute('BEGIN IMMEDIATE')
    try:
        with pytest.raises(sqlite3.OperationalError, match='locked'):
            install_commit_tracking(path, busy_timeout_ms=20)
    finally:
        writer.rollback()
        writer.close()
    with sqlite3.connect(path) as reader:
        assert 'commit_seq' not in [row[1] for row in reader.execute('PRAGMA table_info(tiles)')]
        assert reader.execute("SELECT name FROM sqlite_master WHERE name='tile_commit_clock'").fetchone() is None
    assert install_commit_tracking(path)['backfilled_tiles'] == 3


def test_source_commit_tracking_rolls_back_a_failure_after_backfill_has_started(tmp_path):
    path = tmp_path / 'migration_failure.sqlite3'
    original = _legacy_commit_database(path, 700)
    almost_exhausted = 9_223_372_036_854_775_807-600
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE tile_commit_clock (id INTEGER PRIMARY KEY, seq INTEGER NOT NULL)')
        connection.execute('INSERT INTO tile_commit_clock VALUES(1,?)', (almost_exhausted,))
    # The first 512-row page is modified; a later page fails. DDL and all earlier
    # changes still belong to one transaction and must disappear together.
    with pytest.raises(OverflowError, match='exhausted'):
        install_commit_tracking(path)
    with sqlite3.connect(path) as reader:
        assert 'commit_seq' not in [row[1] for row in reader.execute('PRAGMA table_info(tiles)')]
        assert reader.execute('SELECT * FROM tiles ORDER BY level,tile_x,tile_y').fetchall() == original
        assert reader.execute('SELECT seq FROM tile_commit_clock').fetchone()[0] == almost_exhausted
        assert reader.execute("SELECT name FROM sqlite_master WHERE name='tiles_commit_seq_idx'").fetchone() is None


def test_source_commit_tracking_new_backend_installs_automatically_and_preserves_layers(tmp_path):
    import json
    import zlib
    path = tmp_path / 'new.sqlite3'
    global_map = _make_global(storage_path=str(path), prior_bounds=(-5, -5, 5, 5),
                              publish_unobserved_prior=True, initial_elevation=-.809)
    try:
        global_map.fuse_snapshot((0, 0), np.ones((2, 2), np.float32))
        global_map.flush(wait=True)
        with sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) as reader:
            columns = [row[1] for row in reader.execute('PRAGMA table_info(tiles)')]
            assert 'commit_seq' in columns
            rows = reader.execute('SELECT revision,payload,commit_seq FROM tiles').fetchall()
            assert rows and all(row[2] > 0 for row in rows)
            assert len(set(row[2] for row in rows)) == len(rows)
            for _, payload, _ in rows:
                layers = np.frombuffer(zlib.decompress(payload), dtype='<f4').reshape(4, 2, 2)
                assert set(np.unique(layers[2])).issubset({0., 1.})
            metadata = {key: json.loads(value) for key, value in reader.execute('SELECT key,value FROM metadata')}
            assert metadata['schema_version'] == 2  # Four-layer data format is unchanged.
            assert metadata['prior_bounds'] == [-5., -5., 5., 5.]
            assert metadata['initial_elevation'] == -.809
            assert metadata['initial_variance'] == 10000.
            assert reader.execute('SELECT seq FROM tile_commit_clock WHERE id=1').fetchone()[0] >= max(row[2] for row in rows)
    finally:
        global_map.close()
