"""Scientific invariants: gaps/modes and regional independent-family voting."""
import numpy as np
import pandas as pd
from testing.luke_population_consensus_v1 import adaptive_groups, consensus


def test_no_gap_or_mode_averaging():
    t=np.r_[np.arange(25)*.1,10+np.arange(25)*.1]
    shifts=np.r_[np.zeros(12),np.full(13,40),np.full(25,40)]
    groups=list(adaptive_groups(t,shifts))
    assert np.array_equal(np.concatenate(groups),np.arange(50))
    for g in groups:
        assert len(np.unique(shifts[g]))==1
        assert len(g)<=20 and np.ptp(t[g])<=5
        assert len(g)<2 or np.diff(t[g]).max()<=2


def test_opposite_depths_and_no_duplicate_inflation():
    rows=[]
    for region,sign in [(0,1),(1,-1)]:
        for family,value in [('a',10),('b',11),('c',12)]:
            rows.append(dict(region=region,grid=10,family_id=family,value=sign*value,dredge=sign*9,span=1.))
    # Repeating family a cannot turn the positive consensus toward 10.
    rows+= [rows[0].copy() for _ in range(30)]
    table=consensus(pd.DataFrame(rows))
    assert table.consensus_um.tolist()==[11,-11]
    assert table.families.tolist()==[3,3]
    # No interpolation to absent grid cells or inference from two families.
    sparse=pd.DataFrame(rows).query("family_id != 'c'")
    missing=consensus(sparse)
    assert missing.consensus_um.isna().all()
    assert set(table.grid)=={10}


if __name__=='__main__':
    test_no_gap_or_mode_averaging();test_opposite_depths_and_no_duplicate_inflation();print('scientific invariants passed')
