import numpy as np
import pandas as pd

from motionqc.crossprobe import fit_clock_map, map_intervals
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
