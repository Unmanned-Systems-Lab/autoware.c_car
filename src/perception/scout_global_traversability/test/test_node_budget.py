from types import SimpleNamespace
import pytest
from scout_global_traversability.node import GlobalTraversabilityNode, TimedPredictor
import scout_global_traversability.node as module

def test_model_rate_is_measured_from_call_start_not_after_storage_work(monkeypatch):
    clock=[0.0];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    predictor=SimpleNamespace(calls=0,attempts=0,last_started=0.0)
    allowed=[];polls=[]
    def process_one(near_xy=None,allow_inference=True):
        allowed.append(allow_inference)
        if not allow_inference:return {'state':'inference_budget_wait'}
        predictor.calls+=1;predictor.attempts+=1;predictor.last_started=clock[0];clock[0]+=.20
        return {'state':'computed','model_called':True}
    fake=SimpleNamespace(p={'source_changes_per_poll':64,'max_tasks_per_tick':8,'processing_budget_sec':.08,'max_model_tiles_per_sec':2.0},predictor=predictor,next_model=0.0,
        engine=SimpleNamespace(poll_changes=lambda **kw:polls.append(kw),process_one=process_one),
        ensure_context=lambda:True,fresh_pose=lambda:(1.,2.),log_error=lambda error:pytest.fail(str(error)))
    GlobalTraversabilityNode.tick(fake)
    assert fake.next_model==pytest.approx(.5) and predictor.calls==1
    clock[0]=.4;GlobalTraversabilityNode.tick(fake);assert predictor.calls==1
    clock[0]=.6;GlobalTraversabilityNode.tick(fake);assert predictor.calls==2 and fake.next_model==pytest.approx(1.1)
    assert polls[0]['near_xy']==(1.,2.) and allowed==[True,False,True]


def test_failed_model_attempt_obeys_same_start_time_limit_without_incrementing_success(monkeypatch):
    clock=[0.0];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    def fail(height):
        clock[0]+=.20
        raise RuntimeError('model fixture failed')
    predictor=TimedPredictor(SimpleNamespace(predict=fail))
    allowed=[];errors=[]
    def process_one(near_xy=None,allow_inference=True):
        allowed.append(allow_inference)
        if not allow_inference:return {'state':'inference_budget_wait'}
        predictor.predict(None)
    fake=SimpleNamespace(p={'source_changes_per_poll':64,'max_tasks_per_tick':8,
        'processing_budget_sec':.08,'max_model_tiles_per_sec':2.0},predictor=predictor,next_model=0.,
        engine=SimpleNamespace(poll_changes=lambda **kw:None,process_one=process_one),
        ensure_context=lambda:True,fresh_pose=lambda:(1.,2.),log_error=errors.append)
    GlobalTraversabilityNode.tick(fake)
    assert fake.state=='PROCESSING_ERROR'
    assert fake.next_model==pytest.approx(.5)
    assert predictor.attempts==1 and predictor.calls==0 and predictor.durations==[]
    clock[0]=.4;GlobalTraversabilityNode.tick(fake)
    assert predictor.attempts==1 and predictor.calls==0
    clock[0]=.6;GlobalTraversabilityNode.tick(fake)
    assert predictor.attempts==2 and predictor.calls==0
    assert fake.next_model==pytest.approx(1.1)
    assert allowed==[True,False,True] and len(errors)==2


def test_successful_model_then_storage_failure_also_consumes_attempt_budget(monkeypatch):
    clock=[0.0];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    predictor=TimedPredictor(SimpleNamespace(predict=lambda height: {'fixture':'prediction'}))
    errors=[]
    def process_one(near_xy=None,allow_inference=True):
        if not allow_inference:return {'state':'inference_budget_wait'}
        predictor.predict(None)
        clock[0]+=.20
        raise RuntimeError('storage fixture failed')
    fake=SimpleNamespace(p={'source_changes_per_poll':64,'max_tasks_per_tick':8,
        'processing_budget_sec':.08,'max_model_tiles_per_sec':2.0},predictor=predictor,next_model=0.,
        engine=SimpleNamespace(poll_changes=lambda **kw:None,process_one=process_one),
        ensure_context=lambda:True,fresh_pose=lambda:None,log_error=errors.append)
    GlobalTraversabilityNode.tick(fake)
    assert predictor.attempts==1 and predictor.calls==1
    assert fake.next_model==pytest.approx(.5)
    clock[0]=.4;GlobalTraversabilityNode.tick(fake)
    assert predictor.attempts==1 and predictor.calls==1
    assert len(errors)==1


def test_exact_reuse_or_unknown_jobs_do_not_consume_model_budget(monkeypatch):
    clock=[.4];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    predictor=TimedPredictor(SimpleNamespace(predict=lambda height: pytest.fail('No inference expected')))
    items=iter([{'state':'reused_exact_height'},{'state':'skip_unknown'},None]);allowed=[]
    def process_one(near_xy=None,allow_inference=True):
        allowed.append(allow_inference)
        return next(items)
    fake=SimpleNamespace(p={'source_changes_per_poll':64,'max_tasks_per_tick':8,
        'processing_budget_sec':.08,'max_model_tiles_per_sec':2.0},predictor=predictor,next_model=.5,
        engine=SimpleNamespace(poll_changes=lambda **kw:None,process_one=process_one),
        ensure_context=lambda:True,fresh_pose=lambda:None,log_error=lambda error:pytest.fail(str(error)))
    GlobalTraversabilityNode.tick(fake)
    assert fake.next_model==pytest.approx(.5)
    assert predictor.attempts==0 and predictor.calls==0
    assert allowed==[False,False,False]


def test_global_overview_keeps_historical_probabilities_while_local_retains_stale_mask(monkeypatch):
    calls=[];published=[]
    def encode(window,stamp,threshold=.5,raw_directions=False,stale_timeout=10.):
        calls.append((window['kind'],stale_timeout))
        return window['kind']
    monkeypatch.setattr(module,'window_message',encode)
    fake=SimpleNamespace(engine=SimpleNamespace(local_window=lambda *a,**kw:{'kind':'local'},
        overview=lambda **kw:{'kind':'overview'}),fresh_pose=lambda:(1.,2.),
        p={'local_length_m':10.,'classification_threshold':.5,'publish_direction_probabilities':False,
           'stale_result_timeout_sec':10.,'overview_max_size':512},
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(to_msg=lambda:None)),
        local_pub=SimpleNamespace(publish=published.append),
        global_pub=SimpleNamespace(publish=published.append,get_subscription_count=lambda:1),
        local_cleared=False,log_error=lambda error:pytest.fail(str(error)))
    GlobalTraversabilityNode.publish_local(fake)
    GlobalTraversabilityNode.publish_overview(fake)
    assert calls==[('local',10.),('overview',float('inf'))]
    assert published==['local','overview']
