"""Estimator-independent motion references.

Behaviour is extracted from the repository's M/R/O reference scripts and the
hub T8/T13 implementations; this module contains no dataset paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, signal

from .field import MotionField, canonical_text_hash


def _columns(peaks):
    if isinstance(peaks, pd.DataFrame):
        return tuple(peaks[name].to_numpy() for name in ("time_s","depth_um","x_um"))
    return tuple(np.asarray(peaks[name]) for name in ("time_s","depth_um","x_um"))


def _selector(time_s, value):
    value=np.asarray(value)
    if value.dtype == bool:
        if value.shape != time_s.shape:raise ValueError("boolean selector has wrong shape")
        return value
    if value.size != 2:raise ValueError("selector must be a boolean mask or [start,end)")
    return (time_s>=float(value[0]))&(time_s<float(value[1]))


def _map(depth,x,depth_edges,x_edges):
    return np.log1p(np.histogram2d(depth,x,bins=(depth_edges,x_edges))[0])


def _refine(value,native_um,refine_um):
    if refine_um is None or np.isclose(refine_um,native_um):return value,native_um
    if refine_um<=0 or native_um%refine_um>1e-9:raise ValueError("refine_um must divide depth_bin_um")
    old=np.arange(value.shape[0])*native_um
    new=np.arange(0,old[-1]+1e-9,refine_um)
    return np.column_stack([np.interp(new,old,value[:,j]) for j in range(value.shape[1])]),refine_um


def _correlation(episode,rest,shift_um,step_um):
    offset=-int(round(shift_um/step_um))
    if offset<0:a,b=episode[-offset:],rest[:offset]
    elif offset>0:a,b=episode[:-offset],rest[offset:]
    else:a,b=episode,rest
    a,b=a.ravel(),b.ravel()
    if not len(a) or not len(b):return np.nan
    a=a-a.mean();b=b-b.mean();den=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return float(np.dot(a,b)/den) if den else np.nan


def shift_test(peaks, episode, rest, *, depth_bin_um=10.0, x_bin_um=8.0,
               shift_min_um=-320.0, shift_max_um=120.0, refine_um=None,
               min_peaks=150, min_gain=0.05, depth_range_um=None, x_range_um=None) -> dict:
    """Measure one episode-minus-rest depth shift using label-free peak maps."""
    t,z,x=_columns(peaks);em=_selector(t,episode);rm=_selector(t,rest)
    if depth_range_um is not None:
        in_depth=(z>=depth_range_um[0])&(z<depth_range_um[1]);em&=in_depth;rm&=in_depth
    ei,ri=np.flatnonzero(em),np.flatnonzero(rm)
    result={"episode_peaks":len(ei),"rest_peaks":len(ri),"best_shift_um":np.nan,
            "corr_zero":np.nan,"corr_best":np.nan,"gain":np.nan,"accepted":False}
    if len(ei)<min_peaks or len(ri)<min_peaks:return result
    used=np.r_[ei,ri];zz=z[used];xx=x[used]
    dlo,dhi=(np.floor(zz.min()/depth_bin_um)*depth_bin_um,np.ceil(zz.max()/depth_bin_um)*depth_bin_um) if depth_range_um is None else tuple(map(float,depth_range_um))
    xlo,xhi=(np.floor(xx.min()/x_bin_um)*x_bin_um,np.ceil(xx.max()/x_bin_um)*x_bin_um) if x_range_um is None else tuple(map(float,x_range_um))
    if dhi<=dlo:dhi=dlo+depth_bin_um
    if xhi<=xlo:xhi=xlo+x_bin_um
    de=np.arange(dlo,dhi+depth_bin_um,depth_bin_um);xe=np.arange(xlo,xhi+x_bin_um,x_bin_um)
    a=_map(z[ei],x[ei],de,xe);b=_map(z[ri],x[ri],de,xe);a,step=_refine(a,depth_bin_um,refine_um);b,_=_refine(b,depth_bin_um,refine_um)
    shifts=np.arange(shift_min_um,shift_max_um+step/2,step);corr=np.asarray([_correlation(a,b,s,step) for s in shifts])
    if not np.isfinite(corr).any():return result
    best=int(np.nanargmax(corr));zero=int(np.argmin(np.abs(shifts)))
    result.update(best_shift_um=float(shifts[best]),corr_zero=float(corr[zero]),corr_best=float(corr[best]),
                  gain=float(corr[best]-corr[zero]),accepted=bool(corr[best]-corr[zero]>=min_gain),
                  shifts_um=shifts,correlations=corr)
    return result


def matched_null(peaks, durations_s, flagged_intervals, window, *, n=20, seed=0,
                 peak_targets=None, placement_margin_s=2.0, quantisation_um=10.0, **shift_kwargs):
    """Duration/count-matched pseudo episodes placed outside flagged time."""
    t,_,_=_columns(peaks);rng=np.random.default_rng(seed);durations=np.asarray(durations_s,float)
    targets=None if peak_targets is None else np.asarray(peak_targets,int)
    if targets is not None and len(targets)!=len(durations):
        raise ValueError("peak_targets must pair one-to-one with durations_s")
    if np.any(np.diff(t)<0):
        order=np.argsort(t,kind="stable")
        peaks=peaks.iloc[order].reset_index(drop=True) if isinstance(peaks,pd.DataFrame) else {k:np.asarray(peaks[k])[order] for k in peaks}
        t,_,_=_columns(peaks)
    flagged=np.asarray(flagged_intervals,float).reshape(-1,2) if len(flagged_intervals) else np.empty((0,2))
    if len(flagged):flagged=flagged[np.argsort(flagged[:,0])]
    rows=[];tries=0
    while len(rows)<n and tries<max(5000,n*500):
        tries+=1;draw=int(rng.integers(len(durations)));duration=float(durations[draw]);a=float(rng.uniform(window[0]+4,window[1]-duration));b=a+duration
        if len(flagged) and np.any((flagged[:,0]-placement_margin_s<b)&(flagged[:,1]+placement_margin_s>a-4)):continue
        lo,hi=np.searchsorted(t,[a-4,b]);local=peaks.iloc[lo:hi].reset_index(drop=True) if isinstance(peaks,pd.DataFrame) else {k:np.asarray(peaks[k])[lo:hi] for k in peaks}
        lt,_,_=_columns(local);ep=(lt>=a)&(lt<b);rest=(lt>=a-4)&(lt<a-1)
        target=int(targets[draw]) if targets is not None else int(ep.sum())
        ei=np.flatnonzero(ep);ri=np.flatnonzero(rest)
        # A count-matched null must actually contain the requested number of
        # peaks in both maps.  Silently retaining fewer peaks changes the null
        # SNR and is not count matching.
        if len(ei)<target or len(ri)<target:continue
        if len(ei)>target:ei=rng.choice(ei,target,replace=False)
        if len(ri)>target:ri=rng.choice(ri,target,replace=False)
        em=np.zeros(len(lt),bool);rm=em.copy();em[ei]=True;rm[ri]=True
        value=shift_test(local,em,rm,**shift_kwargs)
        if np.isfinite(value["best_shift_um"]):rows.append({"start_s":a,"end_s":b,"target_peaks":target,**{k:v for k,v in value.items() if k not in ("shifts_um","correlations")}})
    frame=pd.DataFrame(rows);finite=frame.best_shift_um.dropna() if len(frame) else pd.Series(dtype=float)
    mode=float(finite.mode().iloc[0]) if len(finite) else np.nan;median_abs=float(np.median(np.abs(finite))) if len(finite) else np.nan
    gate={"resolved":len(finite),"requested":n,"mode_um":mode,"median_abs_um":median_abs,
          "pass":bool(len(finite)>=min(n,20) and mode==0 and median_abs<=quantisation_um)}
    return frame,gate


def per_block(peaks, episode, rest, blocks=((0,950),(950,1900),(1900,2850),(2850,3840)), margin_um=150, **kwargs):
    rows=[]
    for index,(lo,hi) in enumerate(blocks):
        value=shift_test(peaks,episode,rest,depth_range_um=(lo-margin_um,hi+margin_um),**kwargs)
        rows.append({"block":index,"start_um":lo,"end_um":hi,"centre_um":(lo+hi)/2,
                     **{k:v for k,v in value.items() if k not in ("shifts_um","correlations")}})
    return pd.DataFrame(rows)


def slow_drift_reference(peaks, mask_time_s, mask, *, block_s=10, lags_s=(60,300), refine_um=2, **kwargs):
    t,_,_=_columns(peaks);starts=np.arange(np.floor(t.min()/block_s)*block_s,np.ceil(t.max()/block_s)*block_s,block_s);rows=[]
    quiet={a:not np.asarray(mask)[(mask_time_s>=a)&(mask_time_s<a+block_s)].any() for a in starts}
    for lag in lags_s:
        for a in starts:
            if a-lag not in quiet or not quiet[a] or not quiet[a-lag]:continue
            value=shift_test(peaks,(a,a+block_s),(a-lag,a-lag+block_s),refine_um=refine_um,min_gain=0,**kwargs)
            rows.append({"time_s":a+block_s/2,"previous_time_s":a-lag+block_s/2,"lag_s":lag,
                         **{k:v for k,v in value.items() if k not in ("shifts_um","correlations")}})
    return pd.DataFrame(rows)


def episode_candidates(time_s, *, fields=(), population_counts=None, rate_fraction=0.5, gaze_intervals=(),
                       field_threshold_um=-40, running_median_samples=241, join_gap_s=1.0):
    time_s=np.asarray(time_s,float);mask=np.zeros(len(time_s),bool);sources={}
    for i,field in enumerate(fields):
        d=field.at(time_s,np.median(field.depth_um));base=pd.Series(d).rolling(running_median_samples,center=True,min_periods=1).median().to_numpy()
        sources[f"field{i}"]=d-base<=field_threshold_um;mask|=sources[f"field{i}"]
    if population_counts is not None:
        sources["rate"]=np.asarray(population_counts)<rate_fraction*np.median(population_counts);mask|=sources["rate"]
    gaze=np.zeros(len(time_s),bool)
    for a,b in gaze_intervals:gaze|=(time_s>=a)&(time_s<b)
    sources["gaze"]=gaze;mask|=gaze
    dt=float(np.median(np.diff(time_s)));mask=ndimage.binary_closing(mask,structure=np.ones(round(join_gap_s/dt)+1))
    return mask,sources


def canonical_mask(time_s, displacement_um, catalogue, *, median_samples=241, accepted_label="accepted",
                   unresolved_threshold_um=60, excursion_threshold_um=50, dilation_bins=2):
    """AE canonical mask; source labels concatenate as A, U, E."""
    t=np.asarray(time_s,float);d=np.asarray(displacement_um,float)
    if d.ndim==2:d=d[:,0]
    base=pd.Series(d).rolling(median_samples,center=True,min_periods=1).median().to_numpy();centred=d-base
    source={name:np.zeros(len(t),bool) for name in "AUE"}
    for row in pd.DataFrame(catalogue).itertuples(index=False):
        inside=(t>=float(row.start_s))&(t<float(row.end_s))
        if row.status==accepted_label:source["A"]|=inside
        elif inside.any() and np.max(np.abs(centred[inside]))>unresolved_threshold_um:source["U"]|=inside
    source["E"]=np.abs(centred)>excursion_threshold_um
    union=ndimage.binary_dilation(source["A"]|source["U"]|source["E"],structure=np.ones(2*dilation_bins+1))
    labels,n=ndimage.label(union);rows=[];dt=float(np.median(np.diff(t)))
    for value in range(1,n+1):
        ix=np.flatnonzero(labels==value);code="".join(name for name in "AUE" if source[name][ix].any())
        rows.append((t[ix[0]],t[ix[-1]]+dt,code))
    return pd.DataFrame(rows,columns=("start_s","end_s","source")),union,source,centred


def write_canonical_csv(frame, path, float_format="%.3f"):
    buffer=io.StringIO();frame.to_csv(buffer,index=False,float_format=float_format,lineterminator="\n")
    Path(path).write_text(buffer.getvalue(),encoding="utf-8",newline="")
    return canonical_text_hash(path)


def unit_common_mode(spikes, quiet_mask, *, dt_s=0.05, max_displacement_um=20, displacement=None,
                     bands=((.05,.5),(.5,2),(2,5),(5,10)), null_draws=20, seed=0):
    """Upper/lower shared spectrum; refuses motion outside the sub-row regime."""
    if displacement is not None and np.nanmax(np.abs(displacement))>max_displacement_um:
        raise ValueError("unit_common_mode is valid only for sub-row displacement <=20 um")
    frame=pd.DataFrame(spikes);required={"time_s","unit","depth_um"}
    if not required<=set(frame):raise ValueError(f"missing columns {sorted(required-set(frame))}")
    start=float(np.floor(frame.time_s.min()/dt_s)*dt_s);stop=float(np.ceil(frame.time_s.max()/dt_s)*dt_s);edges=np.arange(start,stop+dt_s/2,dt_s)
    home=frame.groupby("unit").depth_um.median();split=float(home.median());frame=frame.assign(dev=frame.depth_um-frame.unit.map(home),half=frame.unit.map(home)>=split)
    traces=[]
    for upper in (False,True):
        part=frame[frame.half.eq(upper)];idx=np.clip(np.searchsorted(edges,part.time_s,side="right")-1,0,len(edges)-2)
        total=np.bincount(idx,weights=part.dev,minlength=len(edges)-1);count=np.bincount(idx,minlength=len(edges)-1);traces.append(np.divide(total,count,out=np.full_like(total,np.nan,float),where=count>0))
    quiet=np.asarray(quiet_mask,bool)
    if len(quiet)!=len(traces[0]):raise ValueError("quiet_mask must be on the dt_s grid")
    valid=quiet&np.isfinite(traces[0])&np.isfinite(traces[1]);a,b=traces[0][valid],traces[1][valid]
    if len(a)<256:raise ValueError("insufficient quiet support")
    f,pab=signal.csd(a-a.mean(),b-b.mean(),fs=1/dt_s,nperseg=min(256,len(a)));rng=np.random.default_rng(seed);null=[]
    for _ in range(null_draws):
        k=int(rng.integers(1,len(b)));_,pn=signal.csd(a-a.mean(),np.roll(b,k)-b.mean(),fs=1/dt_s,nperseg=min(256,len(a)));null.append(pn)
    df=f[1]-f[0];rows=[]
    for lo,hi in bands:
        keep=(f>=lo)&(f<hi);real=float(np.sqrt(max(np.real(pab[keep]).sum()*df,0)));nr=[np.sqrt(max(np.real(p[keep]).sum()*df,0)) for p in null]
        rows.append({"band_lo_hz":lo,"band_hi_hz":hi,"shared_rms_um":real,"null_p95_um":float(np.percentile(nr,95))})
    return pd.DataFrame(rows)
