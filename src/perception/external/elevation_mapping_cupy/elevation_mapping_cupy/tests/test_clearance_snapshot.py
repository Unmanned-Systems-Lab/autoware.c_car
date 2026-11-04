"""Real CUDA ray traversal through a departed obstacle into persistent tiles."""
from pathlib import Path
import numpy as np
import cupy as cp
import pytest

from elevation_mapping_cupy.parameter import Parameter
from elevation_mapping_cupy.elevation_mapping import ElevationMap
from elevation_mapping_cupy.global_map import GlobalMapAccumulator


@pytest.fixture
def rolling():
    root=Path(__file__).resolve().parents[2]/'config/core'
    p=Parameter(use_chainer=False,enable_builtin_traversability=False,
                plugin_config_file=str(root/'plugin_config.yaml'))
    p.resolution=.1;p.map_length=4.;p.enable_normal_estimation=False
    p.enable_overlap_clearance=False;p.update()
    return ElevationMap(p)


def seed(rolling):
    row,col=rolling.cell_n//2,rolling.cell_n//2+10
    rolling.elevation_map[0,row,col]=.8
    rolling.elevation_map[1,row,col]=.01
    rolling.elevation_map[2,row,col]=1
    rolling.elevation_map[4,row,col]=5
    rolling.normal_map[0,row,col]=1
    rolling.global_update_mask[row,col]=True
    return row,col


@pytest.mark.parametrize('normal,age', [((1,0,0),5), ((0,1,0),5), ((0,0,0),5), ((1,0,0),0)])
def test_real_cuda_rays_clear_a_departed_obstacle_without_ground_endpoint_in_its_cell(rolling,normal,age):
    row,col=seed(rolling)
    rolling.normal_map[:,row,col]=cp.asarray(normal,dtype=cp.float32)
    rolling.elevation_map[4,row,col]=age
    global_map=GlobalMapAccumulator(resolution=.1,initial_elevation=-1,
        publish_unobserved_prior=True,clearance_confirmations=3)
    global_map.initialize_lattice((rolling.cell_n-2,rolling.cell_n-2))
    slot=rolling.create_elevation_snapshot_buffer()
    stream=rolling.create_elevation_snapshot_copy_stream()
    def fuse():
        rolling.capture_elevation_snapshot_gpu(slot)
        center,height,var,updated,free=rolling.materialize_elevation_snapshot(
            slot,stream,include_clearance=True)
        result=global_map.fuse_snapshot(center[:2],height,var,updated,free)
        return result,free
    fuse()
    assert global_map.has_observations
    for frame in range(3):
        # The message is already in map/base coordinates. Its actual scanner
        # is 1 m above that frame; the endpoint is beyond the old obstacle.
        points=np.array([[1.8,.05,-.7]],np.float32)
        rolling.input_pointcloud(points,['x','y','z'],np.eye(3,dtype=np.float32),
            np.zeros(3,np.float32),0,0,ray_origin=np.array([0,.05,1],np.float32))
        rolling.update_time()
        result,bound=fuse()
        assert np.isfinite(bound[row-1,col-1])
        assert result.cleared_cells==(1 if frame==2 else 0)
    view=global_map.extract_window((0,0),4,4)
    assert view.observation_count_yx[row-1,col-1]==0
    assert view.elevation_yx[row-1,col-1]==-1
    assert view.variance_yx[row-1,col-1]==10000
    # No new ray/cloud: repeated exports must not repeat the clearance event.
    assert np.isinf(rolling.export_global_observation(include_clearance=True)[4]).all()


def test_actual_endpoint_discards_queued_free_space_evidence(rolling):
    row,col=seed(rolling)
    rolling.global_clearance_bound[row,col]=-.2
    rolling.new_map.fill(0);rolling.visibility_map[2].fill(cp.inf)
    rolling.new_map[2,row,col]=1
    rolling.ray_clearance_map[row,col]=-.5
    rolling._accumulate_global_clearance()
    assert cp.isinf(rolling.global_clearance_bound[row,col])


def test_clearance_bound_stays_in_world_z_and_moves_with_cells(rolling):
    row,col=seed(rolling)
    rolling.global_clearance_bound[row,col]=.3
    rolling.shift_map_xy(cp.array([1,2],cp.int32))
    assert float(rolling.global_clearance_bound[row+2,col+1])==pytest.approx(.3)
    assert cp.isinf(rolling.global_clearance_bound[:2,:]).all()
    assert cp.isinf(rolling.global_clearance_bound[:,:1]).all()
    rolling.shift_map_z(cp.float32(.5))
    assert float(rolling.global_clearance_bound[row+2,col+1])==pytest.approx(.3)
    rolling.clear()
    assert cp.isinf(rolling.global_clearance_bound).all()
