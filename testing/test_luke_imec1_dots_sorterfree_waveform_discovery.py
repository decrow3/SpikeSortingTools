import numpy as np
from testing.luke_imec1_dots_sorterfree_waveform_discovery import normalize_waveforms,score_block

def test_normalization_removes_amplitude_scale():
    rng=np.random.default_rng(1); w=rng.normal(size=(2,49,16)).astype('float32'); w[1]=3*w[0]
    n=normalize_waveforms(w); assert np.allclose(n[0],n[1],atol=1e-6); assert np.allclose(np.linalg.norm(n,axis=1),1)

def test_real_waveform_beats_decoys_and_rival():
    rng=np.random.default_rng(2); t=rng.normal(size=(2,49,16)).astype('float32'); t/=np.linalg.norm(t,axis=(1,2),keepdims=True)
    winner,score,margin=score_block(t[:1],t); assert winner[0]==0; assert score[0]>.999; assert margin[0]>.1
