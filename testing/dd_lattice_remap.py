"""Exact-coordinate, sample-clock-preserving W2 lattice remap primitives."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd


def round_half_away(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,float)
    return np.sign(x)*np.floor(np.abs(x)+0.5)


def inside_mask(t: np.ndarray, spans: np.ndarray) -> np.ndarray:
    out=np.zeros(len(t),bool)
    for a,b in spans: out|=(t>=a)&(t<b)
    return out


def build_knot_table(field_path: Path, mask_path: Path, start: int, end: int, fs: float) -> tuple[pd.DataFrame,float]:
    with np.load(field_path,allow_pickle=False) as z:
        t=np.asarray(z["time_s"],float); d=np.asarray(z["displacement_um"],float)[:,0]
    spans=pd.read_csv(mask_path)[["start_s","end_s"]].to_numpy(float)
    left,right=start/fs,end/fs
    reference=(t>=left)&(t<right)&~inside_mask(t,spans)
    if not reference.any(): raise ValueError("no exact-window outside-mask reference knots")
    r=float(np.median(d[reference]))
    # Include centers in-window plus cells intersecting either exact edge.
    keep=(t+0.125>left)&(t-0.125<right)
    ix=np.flatnonzero(keep); masked=inside_mask(t[ix],spans)
    q=np.zeros(ix.size,np.int64); q[masked]=(40*round_half_away((d[ix][masked]-r)/40)).astype(np.int64)
    frame=pd.DataFrame({"knot_index":ix,"time_s":t[ix],"displacement_um":d[ix],"inside_mask":masked,"q_um":q})
    return frame,r


def sample_q(n: int, start: int, fs: float, knots: pd.DataFrame) -> np.ndarray:
    # Half-open knot cells [t-.125,t+.125); midpoint ties go to the later cell.
    sec=(start+np.arange(n,dtype=np.int64))/fs
    centers=knots.time_s.to_numpy(float); right=centers+0.125
    pos=np.searchsorted(right,sec,side="right")
    if np.any(pos>=len(knots)): raise ValueError("knot support does not cover sample")
    return knots.q_um.to_numpy(np.int64)[pos]


def coordinate_map(geom: np.ndarray, q_um: int) -> np.ndarray:
    geom=np.asarray(geom,float); lookup={(float(x),float(y)):i for i,(x,y) in enumerate(geom)}
    if len(lookup)!=len(geom): raise ValueError("nonunique recorded geometry")
    out=np.full(len(geom),-1,np.int64)
    for target,(x,y) in enumerate(geom): out[target]=lookup.get((float(x),float(y+q_um)),-1)
    valid=out>=0
    if np.unique(out[valid]).size!=valid.sum(): raise ValueError("nonunique mapped source")
    return out


def remap_chunk(source: np.ndarray, q_by_sample: np.ndarray, geom: np.ndarray) -> tuple[np.ndarray,dict]:
    source=np.asarray(source); q_by_sample=np.asarray(q_by_sample)
    if source.shape[0]!=q_by_sample.size: raise ValueError("sample/q mismatch")
    out=np.zeros_like(source); off=0; mappings={}
    for q in np.unique(q_by_sample):
        mapping=coordinate_map(geom,int(q)); mappings[int(q)]=mapping
        rows=np.flatnonzero(q_by_sample==q); valid=mapping>=0; off+=int(rows.size*(~valid).sum())
        out[np.ix_(rows,np.flatnonzero(valid))]=source[np.ix_(rows,mapping[valid])]
    return out,{"zero_filled_values":off,"mappings":mappings}


def pair_hash(knots: pd.DataFrame) -> str:
    text="".join(f"{int(r.knot_index)},{int(r.q_um)}\n" for r in knots.itertuples())
    return hashlib.sha256(text.encode()).hexdigest()
