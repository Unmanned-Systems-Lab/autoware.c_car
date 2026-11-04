import numpy as np
from builtin_interfaces.msg import Time
from scout_global_traversability.ros_grid import window_message

def decode(msg, name):
    a=msg.data[msg.layers.index(name)]; rows=a.layout.dim[1].size;cols=a.layout.dim[0].size
    return np.asarray(a.data).reshape((rows,cols),order='F').T[::-1,::-1]

def fixture():
    h=np.arange(28,dtype=np.float32).reshape(4,7)/10
    p=np.full_like(h,.8);p[1,4]=.2;p[2,5]=np.nan
    return {'frame':'map','resolution':.04,'origin_xy':[-2.,3.],
            'layers':{'elevation':h,'traversability':p,'valid':np.isfinite(p).astype(np.float32)}}

def test_world_axes_and_non_square_geometry_round_trip():
    w=fixture();m=window_message(w,Time())
    np.testing.assert_array_equal(decode(m,'elevation'),w['layers']['elevation'])
    assert m.info.length_x==.28 and m.info.length_y==.16
    np.testing.assert_allclose([m.info.pose.position.x,m.info.pose.position.y],[-1.86,3.08],atol=1e-12)
    assert m.header.frame_id=='map' and m.basic_layers==['elevation','traversability']

def test_binary_display_is_from_actual_probability_and_preserves_unknown():
    m=window_message(fixture(),Time(),threshold=.5);a=decode(m,'traversability_class')
    assert a[1,4]==0 and a[0,0]==1 and np.isnan(a[2,5])

def test_only_dirty_old_results_are_masked_not_unchanged_history():
    w=fixture();w['layers']['result_age_sec']=np.full((4,7),100,np.float32)
    w['layers']['source_stale']=np.zeros((4,7),np.float32);w['layers']['source_stale'][0,0]=1
    m=window_message(w,Time(),stale_timeout=10)
    assert np.isnan(decode(m,'traversability')[0,0])
    assert decode(m,'traversability')[0,1]>.79


def test_masked_validity_and_raw_directions_match_published_probability():
    w=fixture();w['layers']['result_age_sec']=np.full((4,7),100,np.float32)
    w['layers']['source_stale']=np.zeros((4,7),np.float32);w['layers']['source_stale'][0,0]=1
    w['layers']['valid'][0,1]=.25
    for i in range(8):w['layers'][f'ht_dir_{i}']=np.full((4,7),.8,np.float32)
    m=window_message(w,Time(),raw_directions=True)
    assert decode(m,'valid')[0,0]==0
    assert decode(m,'valid')[0,1]==.25
    for i in range(8):
        assert np.isnan(decode(m,f'ht_dir_{i}')[0,0])
        assert np.isnan(decode(m,f'ht_dir_{i}')[2,5])
    assert w['layers']['valid'][0,0]==1

def test_overview_retains_historical_probability_and_fractional_coverage():
    w=fixture();w['layers']['source_stale']=np.ones((4,7),np.float32)
    w['layers']['result_age_sec']=np.full((4,7),100,np.float32)
    w['layers']['valid'][0,0]=.25
    m=window_message(w,Time(),stale_timeout=float('inf'))
    assert decode(m,'traversability')[0,0]>.79
    assert decode(m,'valid')[0,0]==.25 and decode(m,'source_stale')[0,0]==1
