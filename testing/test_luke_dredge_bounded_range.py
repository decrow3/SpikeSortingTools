import numpy as np
from testing.luke_dredge_bounded_range import bounded_maximum

def test_large_positive_and_negative_matches_require_wider_domain():
    lags=np.arange(-300,301);curve=np.zeros((2,2,len(lags)))
    curve[0,1,lags==200]=1;curve[1,0,lags==-200]=1
    curve[0,1,lags==40]=.4;curve[1,0,lags==-40]=.4
    curve[0,0,lags==0]=1;curve[1,1,lags==0]=1
    narrow,_=bounded_maximum(curve,lags,80);wide,_=bounded_maximum(curve,lags,250)
    assert narrow[1,0]==40 and narrow[0,1]==-40
    assert wide[1,0]==200 and wide[0,1]==-200

def test_outside_peaks_and_inclusive_bound():
    lags=np.arange(-300,301);curve=np.zeros((1,1,len(lags)))
    curve[0,0,lags==300]=2;curve[0,0,lags==250]=1
    d,c=bounded_maximum(curve,lags,250)
    assert d.item()==250 and c.item()==1
