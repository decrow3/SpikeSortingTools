import numpy as np
import pandas as pd

from motionqc.field import MotionField
from motionqc.report import score_fields


def test_identical_fields_score_identically_and_samples_are_matched():
    t=np.arange(0,40,.25);y=np.sin(t/10)[:,None]
    a=MotionField(t,[0],y,source="a");b=MotionField(t[4:-4],[0],y[4:-4],source="b")
    episodes=pd.DataFrame([{"start_s":10,"end_s":12,"status":"accepted","measured_shift_um":0.0}])
    mask=pd.DataFrame([{"start_s":10,"end_s":12,"source":"E"}])
    scores,_,support=score_fields([a,b],episodes,mask)
    assert scores.quiet_samples.nunique()==1 and scores.quiet_pairs.nunique()==1
    assert np.isclose(scores.iloc[0].quiet_inc,scores.iloc[1].quiet_inc)
    assert support["shared_start_s"]>=b.time_s[0]


def test_exact_identical_fields_all_metrics_equal():
    t=np.arange(0,30,.25);y=np.zeros((len(t),1));fields=[MotionField(t,[0],y,source=x) for x in ("x","y")]
    episodes=pd.DataFrame([{"start_s":10,"end_s":11,"status":"accepted","measured_shift_um":-40.0}])
    mask=pd.DataFrame([{"start_s":10,"end_s":11,"source":"E"}])
    scores,_,_=score_fields(fields,episodes,mask)
    for column in ("episode_err","quiet_abs","quiet_inc","false_motion_frac"):
        assert scores[column].nunique(dropna=False)==1
