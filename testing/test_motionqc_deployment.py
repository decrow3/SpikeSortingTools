import numpy as np

from motionqc.deployment import compose_two_layer, sliding_windows, stitch_fields
from motionqc.field import MotionField


def test_sliding_windows_includes_terminal_cover():
    windows=sliding_windows(250,window_s=120,step_s=60)
    assert [(w["start_s"],w["stop_s"]) for w in windows]==[(0,120),(60,180),(120,240),(130,250)]


def test_stitch_chains_offsets_and_preserves_shape():
    t=np.arange(0,120,.25)
    windows=sliding_windows(180)
    fields=[]
    for i,w in enumerate(windows):
        absolute=t+w["start_s"]
        fields.append(MotionField(absolute,[100],(np.sin(absolute/10)+i*30)[:,None]))
    result,diagnostics=stitch_fields(fields,windows,duration_s=180)
    assert result.displacement_um.shape==(720,1)
    assert diagnostics["median_abs_seam_discontinuity_um"]<1e-9
    assert np.isfinite(result.displacement_um).all()


def test_two_layer_returns_slow_exactly_when_mask_empty_and_fast_inside():
    t=np.arange(0,10,.25);slow=MotionField(t,[100,200],np.column_stack([t,t+1]),source="slow")
    fast=MotionField(t,[150],np.full((len(t),1),100.),source="fast")
    result,weight=compose_two_layer(slow,fast,t,np.zeros(len(t),bool))
    assert np.array_equal(result.displacement_um,slow.displacement_um)
    mask=(t>=3)&(t<7);result,weight=compose_two_layer(slow,fast,t,mask)
    assert weight[~mask].max()==0 and weight[mask].max()==1
    assert np.allclose(np.diff(weight[np.flatnonzero(mask)[:4]]),.25)
