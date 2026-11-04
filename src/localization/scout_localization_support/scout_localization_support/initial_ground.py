"""Estimate one measured starting-ground height from a static local NDT map.

The result describes a constant-height prior, not measured elevation cells. A
missing, spatially weak or tilted ground surface raises ValueError; neither
zero nor the vehicle's base height is a fallback.
"""

import math

import numpy as np


def _fit_plane(xy, heights):
    design = np.column_stack((xy, np.ones(len(xy))))
    coefficients, _, rank, _ = np.linalg.lstsq(design, heights, rcond=None)
    if rank != 3 or not np.isfinite(coefficients).all():
        raise ValueError('Ground support does not define a two-dimensional plane')
    return coefficients


def _residuals(xy, heights, coefficients):
    # Avoid starting a BLAS thread team for a tall three-column matrix.
    return heights - (xy[:, 0]*coefficients[0] + xy[:, 1]*coefficients[1] + coefficients[2])


def _plane_consensus(xy, heights, max_residual):
    """Deterministic robust consensus over equal-weight spatial cells."""
    proposals = [np.array([0., 0., np.median(heights)])]
    rng = np.random.default_rng(173)
    for _ in range(256):
        selected = rng.choice(len(xy), 3, replace=False)
        design = np.column_stack((xy[selected], np.ones(3)))
        # Collinear/near-collinear cell centers cannot constrain floor tilt.
        if abs(np.linalg.det(design)) < .02:
            continue
        proposals.append(np.linalg.solve(design, heights[selected]))
    best, best_score = None, None
    for plane in proposals:
        residual = np.abs(_residuals(xy, heights, plane))
        inliers = residual <= max_residual
        count = int(np.count_nonzero(inliers))
        score = (count, -float(np.median(residual[inliers])) if count else -math.inf)
        if best_score is None or score > best_score:
            best, best_score = inliers, score
    for _ in range(3):
        if np.count_nonzero(best) < 3:
            raise ValueError('No consistent ground-plane consensus')
        plane = _fit_plane(xy[best], heights[best])
        updated = np.abs(_residuals(xy, heights, plane)) <= max_residual
        if np.array_equal(updated, best):
            break
        best = updated
    return _fit_plane(xy[best], heights[best]), best


def estimate_initial_ground(
    map_xyz, base_xyz, *, length_x=10.0, length_y=10.0, search_radius=3.0,
    cell_size=0.4, min_ground_below_base=0.02, max_ground_below_base=0.8,
    min_points=80, min_cells=24, min_cell_points=6, max_tilt_deg=5.0,
    max_residual=0.04,
):
    """Return a defensible flat-prior height at ``base_xyz`` in the map frame.

    ``map_xyz`` must be the loaded reference-map geometry in that same frame.
    XY cells give dense walls no extra vote. Each cell contributes a robust
    lower-height envelope only when its near-envelope points have real 2-D XY
    extent, which rejects vertical wall edges. A spatial plane consensus then
    needs broad coverage, support near the starting vehicle, and a small tilt.

    Ground must lie within the configured interval BELOW base_link. The default
    interval includes this Scout's approximately 0.1m base-to-floor clearance;
    it deliberately does not equate base_link height with floor height. Set the
    interval explicitly for a different mounting/frame convention.

    Only the fitted height at starting XY is returned. Sloped plane coefficients
    are never returned as a seed surface. All failures raise ValueError.
    """
    values = [length_x, length_y, search_radius, cell_size, min_ground_below_base,
              max_ground_below_base, max_tilt_deg, max_residual]
    if not all(isinstance(v, (int, float, np.integer, np.floating))
               and not isinstance(v, (bool, np.bool_)) and math.isfinite(v) for v in values):
        raise ValueError('Ground estimator parameters must be finite numbers')
    if (length_x <= 0 or length_y <= 0 or search_radius < 1.0
            or not .1 <= cell_size <= 1.0 or cell_size > search_radius/2
            or not 0 <= min_ground_below_base < max_ground_below_base
            or not 0 < max_tilt_deg <= 15 or not 0 < max_residual <= .15):
        raise ValueError('Ground estimator dimensions or safety limits are invalid')
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_))
           for v in [min_points, min_cells, min_cell_points]):
        raise ValueError('Ground support thresholds must be integers')
    if min_points < 30 or min_cells < 12 or min_cell_points < 6:
        raise ValueError('Ground support thresholds are too weak')
    points, base = np.asarray(map_xyz, dtype=np.float64), np.asarray(base_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Reference-map geometry must be an Nx3 array')
    if base.shape != (3,) or not np.isfinite(base).all():
        raise ValueError('Starting base position must be a finite XYZ vector')
    finite = np.isfinite(points).all(axis=1)
    measured = points[finite]
    xy = measured[:, :2] - base[:2]
    within = np.sum(xy*xy, axis=1) <= search_radius*search_radius
    nearby = measured[within]
    below = ((nearby[:, 2] <= base[2]-min_ground_below_base)
             & (nearby[:, 2] >= base[2]-max_ground_below_base))
    candidates = nearby[below]
    if len(candidates) < min_points:
        raise ValueError(f'Insufficient measured points below base: {len(candidates)} < {min_points}')

    local_xy = candidates[:, :2]-base[:2]
    keys = np.floor(local_xy/cell_size).astype(np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, candidates, local_xy = keys[order], candidates[order], local_xy[order]
    starts = np.r_[0, np.flatnonzero(np.any(np.diff(keys, axis=0), axis=1))+1]
    stops = np.r_[starts[1:], len(keys)]
    centers, heights, support_counts = [], [], []
    dense_cells = 0
    for start, stop in zip(starts, stops):
        if stop-start < min_cell_points:
            continue
        dense_cells += 1
        cell = candidates[start:stop]
        # At least six points and q=.2 tolerate an isolated very-low sample.
        envelope = np.quantile(cell[:, 2], .2)
        band = np.abs(cell[:, 2]-envelope) <= max_residual
        support = local_xy[start:stop][band]
        if len(support) < min_cell_points:
            continue
        centered = support-np.mean(support, axis=0)
        xx = np.mean(centered[:, 0]**2)
        yy = np.mean(centered[:, 1]**2)
        cross = np.mean(centered[:, 0]*centered[:, 1])
        eigenvalues = np.linalg.eigvalsh([[xx, cross], [cross, yy]])
        # A vertical wall's horizontal trace is a thin line, even if it is dense.
        if (eigenvalues[0] < (cell_size*.04)**2
                or eigenvalues[0] < .1*eigenvalues[1]):
            continue
        centers.append(np.median(support, axis=0))
        heights.append(float(np.median(cell[band, 2])))
        support_counts.append(len(support))
    if len(centers) < min_cells:
        raise ValueError(f'Insufficient two-dimensional lower-envelope cells: {len(centers)} < {min_cells}')
    centers, heights = np.asarray(centers), np.asarray(heights)
    plane, inliers = _plane_consensus(centers, heights, max_residual)
    accepted_xy = centers[inliers]
    count = int(np.count_nonzero(inliers))
    fraction = count/len(centers)
    if count < min_cells or fraction < .65:
        raise ValueError(f'Ground plane lacks consensus: {count}/{len(centers)} cells')
    tilt = math.degrees(math.atan(math.hypot(float(plane[0]), float(plane[1]))))
    if tilt > max_tilt_deg:
        raise ValueError(f'Ground plane tilt {tilt:.3f}deg exceeds {max_tilt_deg:.3f}deg')
    height = float(plane[2])
    below_base = float(base[2]-height)
    if not min_ground_below_base <= below_base <= max_ground_below_base:
        raise ValueError('Fitted ground at starting XY is outside the below-base interval')
    residual = np.abs(_residuals(accepted_xy, heights[inliers], plane))
    residual_p95 = float(np.quantile(residual, .95))
    if residual_p95 > max_residual:
        raise ValueError('Ground plane residual is too large')

    span = np.ptp(accepted_xy, axis=0)
    min_span = min(1.6, 1.2*search_radius)
    quadrants = len(set((int(p[0] >= 0), int(p[1] >= 0)) for p in accepted_xy))
    inner_radius = min(1.2, .4*search_radius)
    inner_cells = int(np.count_nonzero(np.sum(accepted_xy**2, axis=1) <= inner_radius**2))
    # Count actual intersecting grid cells, so coverage is a bounded fraction
    # rather than dividing a cell count by a continuous circle-area estimate.
    index = np.arange(-math.ceil(search_radius/cell_size), math.ceil(search_radius/cell_size))
    nearest = np.maximum(np.abs(index+.5)*cell_size-cell_size/2, 0.)
    eligible_cells = int(np.count_nonzero(nearest[:, None]**2+nearest[None, :]**2 <= search_radius**2))
    coverage = count/eligible_cells
    centered = accepted_xy-np.mean(accepted_xy, axis=0)
    xy_cov = np.cov(centered, rowvar=False)
    xy_eigenvalues = np.linalg.eigvalsh(xy_cov)
    if (np.any(span < min_span) or quadrants < 3 or inner_cells < 4
            or coverage < .15 or xy_eigenvalues[0] < .15*xy_eigenvalues[1]):
        raise ValueError(
            f'Ground support coverage is insufficient: span={span.tolist()}, '
            f'quadrants={quadrants}, inner_cells={inner_cells}, coverage={coverage:.3f}')

    center_xy = [float(base[0]), float(base[1])]
    bounds = [[center_xy[0]-length_x/2, center_xy[1]-length_y/2],
              [center_xy[0]+length_x/2, center_xy[1]+length_y/2]]
    quality = {'accepted': True, 'tilt_deg': tilt, 'residual_p95_m': residual_p95,
               'inlier_cells': count, 'coverage_fraction': float(coverage),
               'inner_cells': inner_cells, 'quadrants': quadrants}
    evidence = {
        'input_points': len(points), 'finite_points': int(np.count_nonzero(finite)),
        'nearby_points': len(nearby), 'below_base_points': len(candidates),
        'dense_xy_cells': dense_cells, 'envelope_cells': len(centers),
        'eligible_xy_cells': eligible_cells,
        'inlier_support_points': int(np.asarray(support_counts)[inliers].sum()),
        'inlier_fraction': float(fraction), 'span_xy_m': span.tolist(),
        'search_radius_m': float(search_radius), 'grid_cell_size_m': float(cell_size),
        'height_below_base_m': below_base,
        'ground_below_base_interval_m': [float(min_ground_below_base), float(max_ground_below_base)],
        'max_tilt_deg': float(max_tilt_deg), 'max_residual_m': float(max_residual),
        'checks': ['below_base_height', 'cell_lower_envelope', 'cell_2d_extent',
                   'robust_plane_consensus', 'near_horizontal', 'spatial_and_inner_coverage'],
    }
    return {'height': height, 'method': 'pcd_lower_envelope_plane_v1',
            'center_xy': center_xy, 'bounds_xy': bounds,
            'length_x': float(length_x), 'length_y': float(length_y),
            'evidence': evidence, 'quality': quality}
