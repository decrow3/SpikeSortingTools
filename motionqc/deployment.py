"""Reusable sliding-window stitching and two-layer field composition."""
from __future__ import annotations

import numpy as np

from .field import MotionField


def sliding_windows(duration_s: float, *, window_s: float = 120.0, step_s: float = 60.0):
    """Cover ``[0, duration_s)`` and include a terminal edge-aligned window."""
    if duration_s <= 0 or window_s <= 0 or step_s <= 0 or step_s > window_s:
        raise ValueError("invalid duration/window/step")
    if duration_s <= window_s:
        starts = np.array([0.0])
    else:
        starts = np.arange(0.0, duration_s-window_s+1e-9, step_s)
        terminal = duration_s-window_s
        if not np.isclose(starts[-1], terminal):
            starts = np.r_[starts, terminal]
    return [{"id":f"w{i:03d}","start_s":float(a),"stop_s":float(min(duration_s,a+window_s))}
            for i,a in enumerate(starts)]


def stitch_fields(fields: list[MotionField], windows: list[dict], *, grid_s: float = 0.25,
                  duration_s: float | None = None, source: str = "stitched"):
    """Median-centre, chain overlap offsets, and triangular-blend fields."""
    if len(fields) != len(windows) or not fields:
        raise ValueError("fields and windows must be nonempty and equal length")
    order=np.argsort([float(w["start_s"]) for w in windows]);fields=[fields[i] for i in order];windows=[windows[i] for i in order]
    depth=fields[0].depth_um
    if any(len(f.depth_um)!=len(depth) or not np.allclose(f.depth_um,depth) for f in fields):
        raise ValueError("all fields must share depth coordinates")
    aligned=[];offsets=[];seams=[]
    for i,(field,spec) in enumerate(zip(fields,windows)):
        t=field.time_s;y=field.displacement_um-np.nanmedian(field.displacement_um,axis=0,keepdims=True)
        if i==0:
            offset=np.zeros(len(depth))
        else:
            pt,py=aligned[-1];lo=max(float(t.min()),float(pt.min()));hi=min(float(t.max()),float(pt.max()))
            if hi<lo:raise ValueError(f"adjacent fields do not overlap: {windows[i-1]['id']} / {spec['id']}")
            overlap=np.arange(lo,hi+grid_s/2,grid_s)
            cur=np.column_stack([np.interp(overlap,t,y[:,k]) for k in range(len(depth))])
            prev=np.column_stack([np.interp(overlap,pt,py[:,k]) for k in range(len(depth))])
            offset=np.nanmedian(prev-cur,axis=0);residual=prev-(cur+offset)
            seams.append({"left":windows[i-1]["id"],"right":spec["id"],"overlap_samples":len(overlap),
                          "median_abs_um":float(np.nanmedian(np.abs(residual))),
                          "p95_abs_um":float(np.nanpercentile(np.abs(residual),95)),
                          "max_abs_um":float(np.nanmax(np.abs(residual)))})
        aligned.append((t,y+offset));offsets.append(offset)
    stop=float(duration_s if duration_s is not None else max(w["stop_s"] for w in windows));grid=np.arange(0.0,stop,grid_s)
    numerator=np.zeros((len(grid),len(depth)));denominator=np.zeros_like(numerator)
    for spec,(t,y) in zip(windows,aligned):
        centre=(float(spec["start_s"])+float(spec["stop_s"]))/2;half=(float(spec["stop_s"])-float(spec["start_s"]))/2
        weight=np.maximum(0.0,1.0-np.abs(grid-centre)/half);inside=(grid>=spec["start_s"])&(grid<=spec["stop_s"])
        weight=np.where(inside,np.maximum(weight,1e-6),0.0)
        for k in range(len(depth)):
            values=np.interp(grid,t,y[:,k],left=np.nan,right=np.nan);good=np.isfinite(values)&(weight>0)
            numerator[good,k]+=weight[good]*values[good];denominator[good,k]+=weight[good]
    value=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0);fills=[]
    for k in range(len(depth)):
        finite=np.flatnonzero(np.isfinite(value[:,k]))
        if not len(finite) or np.isnan(value[finite[0]:finite[-1]+1,k]).any():raise ValueError("stitched support has a gap")
        value[:finite[0],k]=value[finite[0],k];value[finite[-1]+1:,k]=value[finite[-1],k]
        fills.append({"depth_index":k,"leading_samples":int(finite[0]),"trailing_samples":int(len(grid)-finite[-1]-1)})
    value-=np.median(value,axis=0,keepdims=True)
    diagnostics={"median_abs_seam_discontinuity_um":float(np.median([s["median_abs_um"] for s in seams])) if seams else 0.0,
                 "seams":seams,"offsets_um":[x.tolist() for x in offsets],"boundary_fills":fills}
    return MotionField(grid,depth,value,source=source,config={"window_s":windows[0]["stop_s"]-windows[0]["start_s"],"grid_s":grid_s}),diagnostics


def compose_two_layer(slow: MotionField, fast: MotionField, time_s, mask, *, crossfade_s: float = 1.0,
                      source: str = "two_layer"):
    """Use slow everywhere and fast inside mask, ramping only inside each run."""
    time_s=np.asarray(time_s,float);mask=np.asarray(mask,bool)
    if len(time_s)!=len(mask):raise ValueError("mask must match time grid")
    depth=slow.depth_um;slow_value=np.column_stack([slow.at(time_s,np.full(len(time_s),z)) for z in depth])
    fast_rigid=fast.at(time_s,np.full(len(time_s),np.median(fast.depth_um)));fast_value=np.repeat(fast_rigid[:,None],len(depth),axis=1)
    if not mask.any():
        return MotionField(time_s,depth,slow_value,source=source,config={"crossfade_s":crossfade_s,"no_episodes":True}),np.zeros(len(time_s))
    dt=float(np.median(np.diff(time_s)));n_ramp=max(1,int(round(crossfade_s/dt)));weight=np.zeros(len(time_s));weight[mask]=1
    starts=np.flatnonzero(mask&np.r_[True,~mask[:-1]]);stops=np.flatnonzero(mask&np.r_[~mask[1:],True])+1
    for a,b in zip(starts,stops):
        n=min(n_ramp,b-a);weight[a:a+n]=np.minimum(weight[a:a+n],np.arange(1,n+1)/n_ramp)
        weight[b-n:b]=np.minimum(weight[b-n:b],np.arange(n,0,-1)/n_ramp)
    value=slow_value*(1-weight[:,None])+fast_value*weight[:,None];value-=np.median(value,axis=0,keepdims=True)
    return MotionField(time_s,depth,value,source=source,config={"crossfade_s":crossfade_s,"no_episodes":False}),weight

