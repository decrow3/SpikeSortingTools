import json
import numpy as np
import pytest
from testing.cross_dataset_fast_motion import complete, motion_metrics, seal


def test_rigid_translation_and_depth_offsets():
    t=np.arange(480)*.25
    ramp=2*t
    field=ramp[:,None]+np.array([100,-20,7,2])[None,:]
    m=motion_metrics(t,field)
    assert m['p99_local_speed_um_s']==pytest.approx(2)
    assert m['p99_local_1s_step_um']==pytest.approx(2)
    assert m['p95_nonrigid_spread_um']==pytest.approx(0,abs=1e-10)


def test_counter_motion_is_not_hidden_by_rigid_summary():
    t=np.arange(480)*.25
    field=t[:,None]*np.array([-2,-1,1,2])[None,:]
    m=motion_metrics(t,field)
    assert m['rigid_excursion_p95_p5_um']==pytest.approx(0)
    assert m['p95_nonrigid_spread_um']>100
    assert m['p99_local_speed_um_s']==pytest.approx(2)


def test_unsealed_or_corrupted_stages_refuse_reuse(tmp_path):
    stage=tmp_path/'stage'
    assert not complete(stage)
    stage.mkdir()
    with pytest.raises(RuntimeError): complete(stage)
    (stage/'data.json').write_text(json.dumps([1,2]))
    seal(stage)
    assert complete(stage)
    (stage/'data.json').write_text('changed')
    with pytest.raises(AssertionError): complete(stage)
