"""Clock mapping and label-free cross-probe episode comparisons."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .reference import per_block, shift_test


@dataclass(frozen=True)
class ClockMap:
    """Affine destination-time mapping fitted from paired synchronization edges."""

    slope: float
    intercept_s: float
    source: str
    destination: str
    edge_count: int
    residual_rms_s: float
    residual_p95_abs_s: float
    residual_max_abs_s: float

    def map(self, time_s):
        return self.slope * np.asarray(time_s, dtype=float) + self.intercept_s

    def to_dict(self):
        return asdict(self)


def fit_clock_map(source_edges_s, destination_edges_s, *, source="source", destination="destination") -> ClockMap:
    source_edges_s=np.asarray(source_edges_s,float);destination_edges_s=np.asarray(destination_edges_s,float)
    if source_edges_s.ndim!=1 or destination_edges_s.ndim!=1 or len(source_edges_s)!=len(destination_edges_s):
        raise ValueError("sync edges must be paired one-dimensional arrays")
    if len(source_edges_s)<2 or np.any(np.diff(source_edges_s)<=0) or np.any(np.diff(destination_edges_s)<=0):
        raise ValueError("sync edges must contain at least two strictly increasing pairs")
    slope,intercept=np.linalg.lstsq(np.c_[source_edges_s,np.ones(len(source_edges_s))],destination_edges_s,rcond=None)[0]
    residual=destination_edges_s-(slope*source_edges_s+intercept);absolute=np.abs(residual)
    return ClockMap(float(slope),float(intercept),source,destination,len(residual),
                    float(np.sqrt(np.mean(residual**2))),float(np.quantile(absolute,.95)),float(absolute.max()))


def map_intervals(frame: pd.DataFrame, clock: ClockMap, *, start="start_s", end="end_s") -> pd.DataFrame:
    result=frame.copy();result[start]=clock.map(result[start].to_numpy());result[end]=clock.map(result[end].to_numpy())
    return result


def compare_episodes(peaks, episodes: pd.DataFrame, *, blocks=((0,950),(950,1900),(1900,2850),(2850,3840)),
                     depth_block_margin_um=150, **shift_kwargs):
    """Run one frozen episode/rest test and per-depth tests for every row."""
    rows=[];block_rows=[]
    time=np.asarray(peaks.time_s if isinstance(peaks,pd.DataFrame) else peaks["time_s"],float)
    if np.any(np.diff(time)<0):
        order=np.argsort(time,kind="stable")
        peaks=peaks.iloc[order].reset_index(drop=True) if isinstance(peaks,pd.DataFrame) else {k:np.asarray(peaks[k])[order] for k in peaks}
        time=np.asarray(peaks.time_s if isinstance(peaks,pd.DataFrame) else peaks["time_s"],float)
    measured="measured_shift_um" if "measured_shift_um" in episodes else "best_shift_um"
    end="end_s" if "end_s" in episodes else "stop_s"
    for index,row in enumerate(episodes.itertuples(index=False)):
        start_s=float(row.start_s);end_s=float(getattr(row,end));lo,hi=np.searchsorted(time,[start_s-4,end_s])
        local=peaks.iloc[lo:hi].reset_index(drop=True) if isinstance(peaks,pd.DataFrame) else {k:np.asarray(peaks[k])[lo:hi] for k in peaks}
        episode=(start_s,end_s);rest=(start_s-4,start_s-1)
        value=shift_test(local,episode,rest,**shift_kwargs)
        rows.append({"episode_index":index,"start_s":start_s,"end_s":end_s,
                     "source_shift_um":float(getattr(row,measured)),
                     **{k:v for k,v in value.items() if k not in ("shifts_um","correlations")}})
        # The whole-probe call may carry an explicit depth range. Per-block
        # evaluation defines its own range (block +/- margin), so forwarding
        # the global range would both conflict at the Python call boundary and
        # defeat the block restriction.
        block_kwargs={k:v for k,v in shift_kwargs.items() if k!="depth_range_um"}
        table=per_block(local,episode,rest,blocks=blocks,margin_um=depth_block_margin_um,**block_kwargs)
        table.insert(0,"episode_index",index);block_rows.append(table)
    return pd.DataFrame(rows),pd.concat(block_rows,ignore_index=True) if block_rows else pd.DataFrame()
