import numpy as np
from testing.luke_imec1_dots_raw_lighthouse_check import geometry_and_mapping, patches, score_events

def test_geometry_excludes_only_ap191_and_translates():
    geom, keep=geometry_and_mapping(); bases, channel_sets=patches(geom)
    assert geom.shape==(383,2); assert 191 not in keep; assert channel_sets.shape[1]==16
    assert not np.any((bases >= 1840) & (bases <= 1960))
    relative=geom[channel_sets]-np.c_[np.zeros(len(bases)),bases][:,None,:]
    assert np.array_equal(relative, np.repeat(relative[:1],len(relative),axis=0))

def test_real_template_beats_decoys():
    rng=np.random.default_rng(0); templates=rng.normal(size=(2,49,16)).astype('float32'); templates/=np.linalg.norm(templates,axis=(1,2),keepdims=True)
    winner,score,margin,gain=score_events(templates[:1],templates)
    assert winner[0]==0; assert score[0]>.999; assert margin[0]>.1
