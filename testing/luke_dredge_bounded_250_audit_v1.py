"""Diagnostic DREDGE replay with exact requested displacement-domain enforcement."""
import numpy as np
import torch
import os
from pathlib import Path
from spikeinterface.sortingcomponents.motion import estimate_motion
from spikeinterface.sortingcomponents.motion.dredge import make_2d_motion_histogram,get_window_domains,normxcorr1d,dredge_ap

def estimate_bounded(record,peaks,locations,config):
    cfg=config.copy();cfg['extra_outputs']=True
    old,extra=estimate_motion(record,peaks,locations,**cfg)
    assert cfg['bin_um']==1 and cfg['max_disp_um']==250
    h,_,_=make_2d_motion_histogram(record,peaks,locations,weight_with_amplitude=True,avg_in_bin=False,direction=cfg['direction'],bin_s=cfg['bin_s'],bin_um=cfg['bin_um'],hist_margin_um=0.,spatial_bin_edges=None,depth_smooth_um=cfg['histogram_depth_smooth_um'],time_smooth_s=cfg['histogram_time_smooth_s'])
    raster=h.T;windows=extra['windows'];D=np.empty_like(extra['D']);C=np.empty_like(extra['C'])
    for b,sl in enumerate(get_window_domains(windows)):
        lo=max(0,sl.start-250);hi=min(len(raster),sl.stop+250);pad=max(lo-(sl.start-250),sl.stop+250-hi);lags=-np.arange(-(pad+sl.start-lo),pad+hi-sl.stop+1)
        curve=normxcorr1d(torch.tensor(raster[sl].T,dtype=torch.float32),torch.tensor(raster[lo:hi].T,dtype=torch.float32),weights=torch.tensor(windows[b,sl],dtype=torch.float32),padding=pad,normalized=True,centered=True).numpy()
        if not np.allclose(lags[curve.argmax(axis=2)].T,extra['D'][b]):
            dest=Path(os.environ['DREDGE_AUDIT_DIR']);dest.mkdir(exist_ok=True)
            ii,jj=np.where(lags[curve.argmax(axis=2)].T!=extra['D'][b])
            np.savez_compressed(dest/f'window_{b}_mismatch.npz',i=ii,j=jj,lags=lags,curves=curve[jj,ii],old_D=extra['D'][b][ii,jj],old_C=extra['C'][b][ii,jj])
        assert np.allclose(lags[curve.argmax(axis=2)].T,extra['D'][b]),'Mismatch evidence saved in DREDGE_AUDIT_DIR'
        assert np.allclose(curve.max(axis=2).T,extra['C'][b],atol=1e-5)
        keep=abs(lags)<=250;v=curve[:,:,keep];D[b]=lags[keep][v.argmax(axis=2)].T;C[b]=v.max(axis=2).T
    for key in ['method','verbose']:cfg.pop(key,None)
    result,newextra=dredge_ap(record,peaks,locations,precomputed_D_C_maxdisp=(D,C,250.),**cfg)
    assert np.isfinite(result.displacement[0]).all() and abs(D).max()<=250
    return result,dict(D=D,C=C,U=newextra['U'],legacy_displacement_um=old.displacement[0],changed_constraints=np.asarray((D!=extra['D']).sum()))
