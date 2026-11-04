"""Retraction requires positive ray evidence, never mere absence of data."""
import numpy as np
import pytest

from elevation_mapping_cupy.global_map import GlobalMapAccumulator


def make(**kw):
    args=dict(resolution=1.,tile_size_cells=4,max_tiles_in_memory=16,
              initial_elevation=-1.,publish_unobserved_prior=True,
              clearance_confirmations=3)
    args.update(kw)
    m=GlobalMapAccumulator(**args)
    m.initialize_lattice((2,2))
    return m


def observe(m, height=1.):
    return m.fuse_snapshot((0,0),np.full((2,2),height,np.float32),
                           np.full((2,2),.01,np.float32))


def ray(m, bound=0., **kw):
    return m.fuse_snapshot((0,0),np.full((2,2),np.nan,np.float32),
                           free_upper_bound_yx=np.full((2,2),bound,np.float32),**kw)


def test_person_departure_clears_without_a_new_endpoint_and_can_reobserve_ground():
    m=make();observe(m)
    for _ in range(2):
        assert ray(m).cleared_cells==0
        assert (m.extract_window((0,0),2,2).observation_count_yx==1).all()
    result=ray(m)
    assert result.cleared_cells==4 and result.written_cells==0
    w=m.extract_window((0,0),2,2)
    assert (w.elevation_yx==-1).all() and (w.variance_yx==10000).all()
    assert (w.observation_count_yx==0).all()
    assert (w.last_update_revision_yx==m.revision).all()
    observe(m,-.75)
    assert np.allclose(m.extract_window((0,0),2,2).elevation_yx,-.75,atol=.001)


def test_current_endpoint_wins_and_resets_pending_clearance():
    m=make();observe(m)
    ray(m);ray(m)
    result=m.fuse_snapshot((0,0),np.ones((2,2),np.float32),
                           np.full((2,2),.01,np.float32),
                           free_upper_bound_yx=np.zeros((2,2),np.float32))
    assert result.cleared_cells==0
    assert ray(m).cleared_cells==0 and ray(m).cleared_cells==0
    assert ray(m).cleared_cells==4


@pytest.mark.parametrize('bound',[1.,2.,np.inf,np.nan,-np.inf])
def test_rays_not_below_the_recorded_surface_and_missing_rays_preserve_it(bound):
    m=make();observe(m)
    for _ in range(6):assert ray(m,bound).cleared_cells==0
    assert (m.extract_window((0,0),2,2).observation_count_yx>0).all()


def test_free_space_does_not_allocate_tiles_or_count_as_an_observation():
    m=make()
    for _ in range(5):assert ray(m).cleared_cells==0
    assert m.tile_count()==0 and not m.has_observations


def test_a_ray_above_the_surface_resets_clearance_evidence():
    m=make();observe(m);ray(m);ray(m)
    ray(m,2.)
    assert ray(m).cleared_cells==0 and ray(m).cleared_cells==0
    assert ray(m).cleared_cells==4


def test_missing_data_does_not_count_as_a_clearance_vote():
    m=make();observe(m);ray(m)
    for _ in range(20):ray(m,np.inf)
    assert (m.extract_window((0,0),2,2).observation_count_yx==1).all()
    assert ray(m).cleared_cells==0
    assert ray(m).cleared_cells==4


def test_clearance_is_disabled_unless_enabled_and_checks_geometry():
    m=make(clearance_confirmations=0);observe(m)
    for _ in range(5):assert ray(m).cleared_cells==0
    with pytest.raises(ValueError,match='shape'):
        m.fuse_snapshot((0,0),np.ones((2,2)),free_upper_bound_yx=np.zeros((3,2)))


def test_cleared_height_and_count_persist_but_partial_votes_do_not(tmp_path):
    args=dict(storage_path=str(tmp_path/'map.sqlite3'))
    m=make(**args);observe(m);ray(m);ray(m);m.close()
    m=make(**args,resume_from_disk=True)
    assert ray(m).cleared_cells==0 and ray(m).cleared_cells==0
    assert ray(m).cleared_cells==4
    m.close()
    m=make(**args,resume_from_disk=True)
    w=m.extract_window((0,0),2,2)
    assert (w.elevation_yx==-1).all() and (w.observation_count_yx==0).all()
    m.close()


def test_only_ray_covered_cells_are_retracted():
    m=make();observe(m)
    bound=np.array([[0,np.inf],[np.inf,np.inf]],np.float32)
    for _ in range(3):
        result=m.fuse_snapshot((0,0),np.full((2,2),np.nan),free_upper_bound_yx=bound)
    assert result.cleared_cells==1
    w=m.extract_window((0,0),2,2)
    assert w.observation_count_yx[0,0]==0
    assert np.count_nonzero(w.observation_count_yx)==3
