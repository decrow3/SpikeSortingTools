import numpy as np
import pandas as pd

from motionqc.crossprobe import compare_episodes, fit_clock_map, map_intervals
from motionqc.reference import matched_null


def test_clock_map_recovers_affine_mapping_and_intervals():
    source=np.arange(1000,dtype=float);destination=1.000002*source-.003
    mapping=fit_clock_map(source,destination,source="a",destination="b")
    assert abs(mapping.slope-1.000002)<1e-12
    assert abs(mapping.intercept_s+.003)<1e-12
    assert mapping.residual_max_abs_s<1e-10
    frame=pd.DataFrame({"start_s":[1.],"end_s":[2.]})
    mapped=map_intervals(frame,mapping)
    assert np.allclose(mapped[["start_s","end_s"]],[[.997002,1.997004]])


def test_matched_null_pairs_duration_and_peak_target():
    rng=np.random.default_rng(4)
    peaks=pd.DataFrame({"time_s":np.linspace(0,200,20_001),
                        "depth_um":rng.uniform(0,1000,20_001),
                        "x_um":rng.uniform(-20,20,20_001)})
    frame,_=matched_null(peaks,[1.,2.],[],(0,200),n=4,seed=5,
                         peak_targets=[50,125],min_peaks=20,min_gain=0,
                         depth_range_um=(0,1000),x_range_um=(-24,24))
    assert len(frame)==4
    for row in frame.itertuples(index=False):
        expected=50 if np.isclose(row.end_s-row.start_s,1.) else 125
        assert row.target_peaks==expected
        assert row.episode_peaks==expected
        assert row.rest_peaks==expected


def test_compare_episodes_global_depth_range_does_not_collide_with_blocks():
    rng=np.random.default_rng(8);centres=np.repeat([250.,700.],400);x=np.tile(np.repeat([0.,32.],200),2)
    rest=pd.DataFrame({"time_s":rng.uniform(0,3,len(centres)),"depth_um":centres+rng.normal(0,1,len(centres)),"x_um":x})
    episode=pd.DataFrame({"time_s":rng.uniform(4,5,len(centres)),"depth_um":centres-80+rng.normal(0,1,len(centres)),"x_um":x})
    episodes=pd.DataFrame([{"start_s":4.,"end_s":5.,"measured_shift_um":-80.}])
    whole,blocks=compare_episodes(pd.concat([rest,episode],ignore_index=True),episodes,
                                  blocks=((0,950),),depth_block_margin_um=150,
                                  depth_range_um=(0,3840),x_range_um=(-32,80),min_peaks=100)
    assert whole.loc[0,"best_shift_um"]==-80
    assert blocks.loc[0,"best_shift_um"]==-80
