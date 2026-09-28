import numpy as np
import pandas as pd
from testing.dd_lattice_remap import coordinate_map, remap_chunk, round_half_away, sample_q


def geom():
    return np.array([[0,0],[20,0],[0,40],[20,40],[0,80],[20,80]],float)


def test_round_half_away():
    assert np.array_equal(round_half_away(np.array([-1.5,-.5,.5,1.5])),[-2,-1,1,2])


def test_q0_bytes_and_no_time_shift():
    x=np.arange(30,dtype=np.int16).reshape(5,6); y,r=remap_chunk(x,np.zeros(5,int),geom())
    assert y.tobytes()==x.tobytes() and r["zero_filled_values"]==0


def test_positive_negative_40_known_peak_and_sign():
    x=np.zeros((1,6),np.int16); x[0,4]=-99
    up,_=remap_chunk(x,np.array([40]),geom()); down,_=remap_chunk(x,np.array([-40]),geom())
    assert up[0,2]==-99 and down[0,4]==0


def test_boundary_and_seam_samples():
    knots=pd.DataFrame({"time_s":[0.,.25,.5,.75],"q_um":[0,40,-40,0]})
    q=sample_q(7,0,8.,knots)
    assert np.array_equal(q,[0,40,40,-40,-40,0,0])
    x=np.arange(42,dtype=np.int16).reshape(7,6); y,_=remap_chunk(x,q,geom())
    assert y.shape==x.shape


def test_actual_coordinate_uniqueness_shape():
    g=np.array([[16,0],[48,0],[0,20],[32,20],[16,40],[48,40]],float)
    m=coordinate_map(g,40); assert len(m)==len(g) and np.unique(m[m>=0]).size==(m>=0).sum()
