import numpy as np
import pandas as pd

from motionqc.reference import canonical_mask, matched_null, per_block, shift_test, unit_common_mode, write_canonical_csv


def synthetic_peaks(seed=0, step_um=40.0):
    rng=np.random.default_rng(seed);n=5000
    base=rng.choice(np.arange(200,1800,40),n)+rng.normal(0,2,n);x=rng.choice([0,32],n)+rng.normal(0,1,n)
    return pd.DataFrame({"time_s":np.r_[rng.uniform(0,3,n),rng.uniform(5,8,n)],
                         "depth_um":np.r_[base,base+step_um],"x_um":np.r_[x,x]})


def test_shift_recovers_known_rigid_step_and_refinement():
    peaks=synthetic_peaks()
    result=shift_test(peaks,(5,8),(0,3))
    refined=shift_test(peaks,(5,8),(0,3),refine_um=2)
    assert result["accepted"] and result["best_shift_um"]==40
    assert abs(refined["best_shift_um"]-40)<=2


def test_brief_episode_recovers_known_shift():
    rng=np.random.default_rng(12);centres=np.repeat(np.array([210,347,529,812,1194,1661]),100);x=np.tile(np.repeat([0.,32.],50),6)
    rest=pd.DataFrame({"time_s":rng.uniform(0,3,len(centres)),"depth_um":centres+rng.normal(0,1,len(centres)),"x_um":x})
    episode=pd.DataFrame({"time_s":rng.uniform(4,4.5,len(centres)),"depth_um":centres-80+rng.normal(0,1,len(centres)),"x_um":x})
    result=shift_test(pd.concat([rest,episode],ignore_index=True),(4,4.5),(0,3))
    assert result["accepted"] and result["best_shift_um"]==-80


def test_per_block_recovers_uniform_step():
    result=per_block(synthetic_peaks(),(5,8),(0,3),blocks=((0,950),(950,1900)),margin_um=0,min_peaks=100)
    assert set(result.best_shift_um)=={40.0}


def test_matched_null_passes_stationary_raster():
    rng=np.random.default_rng(3);times=[];depth=[];x=[]
    centres=np.array([233,401,618,944,1271,1693]);weights=np.array([17,31,11,43,23,37])
    pattern=np.repeat(centres,weights);xp=np.concatenate([np.full(n,0 if i%2 else 32) for i,n in enumerate(weights)])
    for second in range(120):
        times.extend(second+rng.uniform(0,1,len(pattern)));depth.extend(pattern+rng.normal(0,.5,len(pattern)));x.extend(xp)
    peaks=pd.DataFrame({"time_s":times,"depth_um":depth,"x_um":x})
    frame,gate=matched_null(peaks,[1.0],[],(0,120),n=20,seed=2,min_peaks=100)
    assert len(frame)>=20 and gate["pass"]


def test_canonical_mask_source_order_and_csv(tmp_path):
    t=np.arange(0,100,.25);d=np.zeros_like(t);d[(t>=20)&(t<23)]=-100
    cat=pd.DataFrame([{"start_s":20,"end_s":21,"status":"accepted"},{"start_s":21,"end_s":22,"status":"unresolved"}])
    frame,mask,sources,_=canonical_mask(t,d,cat)
    assert "AUE" in set(frame.source)
    path=tmp_path/"mask.csv";digest=write_canonical_csv(frame,path)
    assert b"\r" not in path.read_bytes() and digest


def test_unit_common_mode_refuses_non_subrow_motion():
    spikes=pd.DataFrame({"time_s":[0.,1.],"unit":[0,1],"depth_um":[100.,200.]})
    with np.testing.assert_raises_regex(ValueError,"sub-row"):
        unit_common_mode(spikes,np.ones(20,bool),displacement=np.array([0.,21.]))
