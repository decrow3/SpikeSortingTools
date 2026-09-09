"""Explicit DREDGE correlation-lag bounds in micrometers (1um raster only)."""
import numpy as np
import torch
from spikeinterface.sortingcomponents.motion import estimate_motion
from spikeinterface.sortingcomponents.motion.dredge import make_2d_motion_histogram,get_window_domains,normxcorr1d,dredge_ap

def bounded_maximum(curve,lags,bound_um):
    keep=np.abs(lags)<=bound_um
    if not keep.any():raise ValueError('No candidate lags within bound')
    v=curve[:,:,keep]
    return lags[keep][v.argmax(axis=2)].T,v.max(axis=2).T

def estimate_bounded_range(record,peaks,locations,config):
    cfg=config.copy();cfg['extra_outputs']=True
    bound=float(cfg['max_disp_um'])
    if cfg['bin_um']!=1 or not bound.is_integer() or bound<=0:raise ValueError('Positive integer um bound and1um bins required')
    bound=int(bound);old,extra=estimate_motion(record,peaks,locations,**cfg)
    h,_,_=make_2d_motion_histogram(record,peaks,locations,weight_with_amplitude=True,avg_in_bin=False,direction=cfg['direction'],bin_s=cfg['bin_s'],bin_um=cfg['bin_um'],hist_margin_um=0.,spatial_bin_edges=None,depth_smooth_um=cfg['histogram_depth_smooth_um'],time_smooth_s=cfg['histogram_time_smooth_s'])
    raster=h.T;windows=extra['windows'];D=np.empty_like(extra['D']);C=np.empty_like(extra['C'])
    for b,sl in enumerate(get_window_domains(windows)):
        lo=max(0,sl.start-bound);hi=min(len(raster),sl.stop+bound);pad=max(lo-(sl.start-bound),sl.stop+bound-hi);lags=-np.arange(-(pad+sl.start-lo),pad+hi-sl.stop+1)
        curve=normxcorr1d(torch.tensor(raster[sl].T,dtype=torch.float32),torch.tensor(raster[lo:hi].T,dtype=torch.float32),weights=torch.tensor(windows[b,sl],dtype=torch.float32),padding=pad,normalized=True,centered=True).numpy()
        assert np.allclose(lags[curve.argmax(axis=2)].T,extra['D'][b]);assert np.allclose(curve.max(axis=2).T,extra['C'][b],atol=1e-5)
        D[b],C[b]=bounded_maximum(curve,lags,bound)
    for key in ['method','verbose']:cfg.pop(key,None)
    motion,ext=dredge_ap(record,peaks,locations,precomputed_D_C_maxdisp=(D,C,float(bound)),**cfg)
    assert np.isfinite(motion.displacement[0]).all() and np.max(abs(D))<=bound
    return motion,dict(D=D,C=C,U=ext['U'],legacy_displacement_um=old.displacement[0],changed_constraints=int((D!=extra['D']).sum()))
