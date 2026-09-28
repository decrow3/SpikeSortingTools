#!/usr/bin/env python
"""Exact-event 10-second views of gentle imec1 waveform candidates."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from testing.luke_imec1_sorterfree_top20_motion_comparison import (
    DISCOVERY, ESTIMATORS, ORIGIN_S, SEED_INTERVAL, load_field,
)

ROOT=Path(__file__).resolve().parents[1]
GENTLE=ROOT/"testing/outputs/luke_imec1_sorterfree_gentle_match_v1"
OUT=ROOT/"testing/outputs/luke_imec1_sorterfree_gentle_10s_snippets_v2"
WINDOWS=[(70,80),(110,120),(420,430),(560,570),(820,830),(850,860)]
CANDIDATES=["p09_f014","p08_f038","p06_f000","p06_f015","p09_f042","p07_f022"]


def depth_trace(time,depth,motion,target):
    return np.asarray([np.interp(target,depth,row) for row in motion])


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    events=pd.read_csv(GENTLE/"gentle_events.csv")
    families=pd.read_csv(DISCOVERY/"families_after_depth_reveal.csv").set_index("family_id")
    fields={name:load_field(folder)[:3] for name,folder in ESTIMATORS.items()}
    status_style={"strict":("black","o",22,1.0),"lower_score":("#e58b2a","x",20,.8),"identity_ambiguous":("#999999",".",16,.65)}
    field_style={"DREDGE":"#007f73","KS sidecar":"#2265ac","Decentralized":"#aa7600","MEDiCINe":"#6e59a5"}
    rows=[]
    with PdfPages(OUT/"all_gentle_10s_snippets.pdf") as pdf:
        for index,(start,stop) in enumerate(WINDOWS):
            fig,axes=plt.subplots(len(CANDIDATES),1,figsize=(14,13),sharex=True,sharey=True,layout="constrained")
            for ax,fid in zip(axes,CANDIDATES):
                family=families.loc[fid]; center=float(family.seed_depth_median_um)
                q=events[(events.family_id.eq(fid))&(events.time_s>=start)&(events.time_s<stop)]
                counts=q.status.value_counts()
                for status,(color,marker,size,alpha) in status_style.items():
                    z=q[q.status.eq(status)]
                    if len(z): ax.scatter(z.time_s,z.waveform_centroid_um-center,c=color,marker=marker,s=size,alpha=alpha,label=status,zorder=4)
                for estimator,(time,depth,motion) in fields.items():
                    k=(time>=start)&(time<stop)
                    seed_k=(time>=SEED_INTERVAL[0])&(time<SEED_INTERVAL[1])
                    baseline=float(np.median(depth_trace(time[seed_k],depth,motion[seed_k],center)))
                    ax.plot(time[k],depth_trace(time[k],depth,motion[k],center)-baseline,color=field_style[estimator],marker=".",ms=4,lw=1,label=estimator,zorder=2)
                ax.axhline(0,color="#777777",lw=.5)
                ax.set_ylabel(f"{fid}\nrelative µm")
                ax.set_title(f"strict {int(counts.get('strict',0))} · lower {int(counts.get('lower_score',0))} · ambiguous {int(counts.get('identity_ambiguous',0))}",loc="left",fontsize=9)
                ax.set_xlim(start,stop); ax.set_ylim(-65,45)
                ax.set_xticks(np.arange(start,stop+.01,1.0)); ax.set_xticks(np.arange(start,stop+.01,.2),minor=True)
                ax.grid(which="major",axis="x",color="#cccccc",lw=.5); ax.grid(which="minor",axis="x",color="#eeeeee",lw=.35)
                rows.append({"start_s":start,"stop_s":stop,"family_id":fid,"strict":int(counts.get('strict',0)),"lower_score":int(counts.get('lower_score',0)),"identity_ambiguous":int(counts.get('identity_ambiguous',0)),"displayed_events":len(q)})
            handles,labels=axes[0].get_legend_handles_labels(); unique=dict(zip(labels,handles))
            axes[0].legend(unique.values(),unique.keys(),loc="upper right",ncol=4,fontsize=7)
            fig.supxlabel("Recording-relative time (s); faint vertical grid = 200 ms (5 Hz)")
            fig.suptitle(f"Frozen waveform candidates, exact event times: {start}–{stop} s\nMotion marks are native saved samples, not upsampled",fontsize=14)
            name=f"{index+1:02d}_{start:04d}_{stop:04d}_exact_events.png"
            fig.savefig(OUT/name,dpi=170); pdf.savefig(fig); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/"snippet_event_counts.csv",index=False)


if __name__=="__main__": main()
