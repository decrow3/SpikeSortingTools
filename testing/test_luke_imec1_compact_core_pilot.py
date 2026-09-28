import numpy as np
from testing.luke_imec1_compact_core_pilot_v1 import lagged_scores,make_template

def test_lagged_core_score_recovers_shift():
    rng=np.random.default_rng(1);w=rng.normal(size=(49,16)).astype('float32');shift=np.zeros_like(w);shift[2:]=w[:-2]
    score=lagged_scores(shift[None],w/np.linalg.norm(w))
    unshifted=(shift.reshape(-1)/np.linalg.norm(shift))@(w.reshape(-1)/np.linalg.norm(w))
    assert score[0]>.97
    assert score[0]>unshifted+.8

def test_template_scale_preserves_median_norm():
    rng=np.random.default_rng(2);w=rng.normal(size=(7,49,16)).astype('float32');_,scale=make_template(w)
    assert np.isclose(scale,np.median(np.linalg.norm(w.reshape(7,-1),axis=1)))
