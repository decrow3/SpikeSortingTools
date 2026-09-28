#!/usr/bin/env python3
"""DE common exact-domain scorecard for saved DARTsort and Kilosort outputs."""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DOMAINS = ("negative_excursion", "outside_mask_flat", "catalogue_outside_remainder")


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    p=Path(path); q=p.with_suffix(p.suffix+".partial")
    q.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n"); os.replace(q,p)


def spans_mask(t, spans):
    out=np.zeros(t.size,bool)
    for a,b in spans: out|=(t>=a)&(t<b)
    return out


def catalogue_state(t, cat):
    out=np.full(t.size,"outside_catalogue",object)
    for row in cat.itertuples():
        status="catalogue_accepted" if str(row.status)=="accepted" else "catalogue_unresolved"
        out[(t>=float(row.start_s))&(t<float(row.end_s))]=status
    return out


def merge(parts):
    out=[]
    for a,b,label in parts:
        if b<=a: continue
        if out and out[-1][2]==label and np.isclose(a,out[-1][1],rtol=0,atol=1e-12): out[-1][1]=b
        else: out.append([a,b,label])
    return [(float(a),float(b),str(c)) for a,b,c in out]


def intervals(t, dev, mask, left, right):
    cuts=[left,right,*t[(t>left)&(t<right)].tolist()]
    for threshold in (-120.,-20.,20.):
        ix=np.flatnonzero((dev[:-1]-threshold)*(dev[1:]-threshold)<0)
        roots=t[ix]+(threshold-dev[ix])*(t[ix+1]-t[ix])/(dev[ix+1]-dev[ix])
        cuts.extend(roots[(roots>left)&(roots<right)].tolist())
    for a,b in mask:
        if left<a<right: cuts.append(float(a))
        if left<b<right: cuts.append(float(b))
    cuts=np.unique(np.asarray(cuts,float)); mid=(cuts[:-1]+cuts[1:])/2
    val=np.interp(mid,t,dev); masked=spans_mask(mid,mask)
    label=np.full(mid.size,DOMAINS[2],object)
    label[(~masked)&(np.abs(val)<20)]=DOMAINS[1]; label[val<-120]=DOMAINS[0]
    return merge([(cuts[i],cuts[i+1],label[i]) for i in range(mid.size)])


def assign_domain(t, ft, dev, mask):
    val=np.interp(t,ft,dev); masked=spans_mask(t,mask)
    out=np.full(t.size,DOMAINS[2],object)
    out[(~masked)&(np.abs(val)<20)]=DOMAINS[1]; out[val<-120]=DOMAINS[0]
    return out


def segment_id(t, ivals, label):
    out=np.full(t.size,-1,np.int32); n=0
    for a,b,x in ivals:
        if x==label: out[(t>=a)&(t<b)]=n; n+=1
    return out


def load_arm(spec, start, end, fs):
    path=Path(spec["path"]); kind=spec["format"]
    if kind=="dartsort_npz":
        with np.load(path,allow_pickle=False) as z:
            local=np.asarray(z["times_samples"],np.int64); labels=np.asarray(z["labels"],np.int64)
            ch=np.asarray(z["channels"],np.int64); geom=np.asarray(z["geom"],float)
            if not np.isclose(float(z["sampling_frequency"]),fs,rtol=0,atol=1e-9): raise ValueError("FS mismatch")
        if np.any((local<0)|(local>=end-start)): raise ValueError("local event outside support")
        times=local+start; depth=geom[ch,1]; assigned=labels>=0; good=None
        noise_available=True; clock_type="window_local"
    elif kind=="kilosort_folder":
        times_all=np.load(path/"spike_times.npy",mmap_mode="r").reshape(-1); clock_type=spec.get("clock")
        if clock_type=="full_source": take=(times_all>=start)&(times_all<end); times=np.asarray(times_all[take],np.int64)
        elif clock_type=="window_local": take=(times_all>=0)&(times_all<end-start); times=np.asarray(times_all[take],np.int64)+start
        else: raise ValueError("Kilosort spec requires clock=full_source or window_local")
        labels=np.asarray(np.load(path/"spike_clusters.npy",mmap_mode="r").reshape(-1)[take],np.int64)
        pos=np.asarray(np.load(path/"spike_positions.npy",mmap_mode="r")[take]); depth=pos[:,1]
        assigned=np.ones(labels.size,bool); noise_available=False
        tab=pd.read_csv(path/"cluster_KSLabel.tsv",sep="\t")
        idcol="cluster_id" if "cluster_id" in tab else tab.columns[0]; labcol="KSLabel" if "KSLabel" in tab else tab.columns[-1]
        good_ids=set(tab.loc[tab[labcol].astype(str).str.lower()=="good",idcol].astype(int)); good=np.array([x in good_ids for x in labels])
    else: raise ValueError(kind)
    return times,labels,depth,assigned,good,noise_available,clock_type


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    cfg=json.loads(a.manifest.read_text()); out=a.output.resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out/"source").mkdir(); shutil.copy2(Path(__file__).resolve(),out/"source"/Path(__file__).name); shutil.copy2(a.manifest,out/"source"/a.manifest.name)
    started=time.perf_counter(); fs=float(cfg["sampling_frequency_hz"])
    required={Path(k):v for k,v in cfg["expected_sha256"].items()}; checks={str(p):sha256(p)==v for p,v in required.items()}
    if not all(checks.values()): raise ValueError({k:v for k,v in checks.items() if not v})
    with np.load(cfg["field"],allow_pickle=False) as z:
        ft=np.asarray(z["time_s"],float); disp=np.asarray(z["displacement_um"],float)
    scalar=disp[:,0]; baseline=pd.Series(scalar).rolling(480,center=True,min_periods=1).median().to_numpy(); dev=scalar-baseline
    max_column_difference=float(np.max(np.abs(disp-disp[:,[0]]))) if disp.ndim==2 else 0.
    mask=pd.read_csv(cfg["mask"])[["start_s","end_s"]].to_numpy(float); cat=pd.read_csv(cfg["catalogue"])
    count_rows=[]; isi_rows=[]; unit_rows=[]; rho_rows=[]; interval_rows=[]
    for w in cfg["windows"]:
        name=w["name"]; start,end=map(int,w["frames_half_open"]); left,right=start/fs,end/fs
        ivals=intervals(ft,dev,mask,left,right)
        iv=pd.DataFrame(ivals,columns=["start_s","end_s","domain"]); iv["duration_s"]=iv.end_s-iv.start_s; iv["window"]=name; interval_rows.extend(iv.to_dict("records"))
        exposure=iv.groupby("domain").duration_s.sum().reindex(DOMAINS)
        if not np.isclose(exposure.sum(),right-left,atol=1e-9): raise ValueError(f"{name} exposure closure")
        for arm,spec in w["arms"].items():
            times,labels,depth,assigned,good,noise_available,clock_type=load_arm(spec,start,end,fs); sec=times/fs
            domain=assign_domain(sec,ft,dev,mask); cstate=catalogue_state(sec,cat)
            for d in DOMAINS:
                inside=domain==d; seg=segment_id(sec,ivals,d); ix=np.flatnonzero(inside&assigned)
                if np.any(seg[inside] < 0) or np.any(seg[~inside] >= 0):
                    raise AssertionError(f"{name}/{arm}/{d}: domain/segment membership mismatch")
                order=ix[np.lexsort((times[ix],labels[ix]))]; same=(labels[order[1:]]==labels[order[:-1]])&(seg[order[1:]]==seg[order[:-1]]) if order.size>1 else np.zeros(0,bool); delta=np.diff(times[order])[same]
                subsets=[("all_units",np.ones(times.size,bool))]
                if good is not None: subsets.append(("good_units",good))
                for subset,subkeep in subsets:
                    den=inside&subkeep
                    count_rows.append({"window":name,"arm":arm,"subset":subset,"domain":d,"domain_exposure_s":exposure[d],"subset_exposure_s":exposure[d],"all_events":int(den.sum()),"assigned_events":int((den&assigned).sum()),"noise_events":int((den&~assigned).sum()) if noise_available else np.nan,"assigned_rate_hz":float((den&assigned).sum()/exposure[d]),"units":int(np.unique(labels[den&assigned]).size),"noise_representation_available":noise_available,"good_label_availability":"available" if good is not None else "unavailable","clock_type":clock_type})
                isi_rows.append({"window":name,"arm":arm,"domain":d,"adjacent_denominator":delta.size,"lag8":int((delta==8).sum()),"lag9_29":int(((delta>=9)&(delta<=29)).sum()),"lag30":int((delta==30).sum()),"fraction9_29":float(((delta>=9)&(delta<=29)).mean()) if delta.size else np.nan})
                for cs in ("catalogue_accepted","catalogue_unresolved","outside_catalogue"):
                    x=inside&(cstate==cs); count_rows.append({"window":name,"arm":arm,"subset":cs,"domain":d,"domain_exposure_s":exposure[d],"subset_exposure_s":np.nan,"all_events":int(x.sum()),"assigned_events":int((x&assigned).sum()),"noise_events":int((x&~assigned).sum()) if noise_available else np.nan,"assigned_rate_hz":np.nan,"units":int(np.unique(labels[x&assigned]).size),"noise_representation_available":noise_available,"good_label_availability":"available" if good is not None else "unavailable","clock_type":clock_type})
            rows=[]
            for u in np.unique(labels[assigned]):
                uu=assigned&(labels==u); flat=uu&(domain==DOMAINS[1]); row={"window":name,"arm":arm,"unit_id":int(u),"median_flat_event_depth_um":float(np.median(depth[flat])) if flat.any() else np.nan}
                for d in DOMAINS:
                    n=int(np.sum(uu&(domain==d))); row[f"count_{d}"]=n; row[f"log_rate_{d}"]=float(np.log((n+.5)/exposure[d]))
                row["negative_to_flat_rate_ratio"]=(row[f"count_{DOMAINS[0]}"]/exposure[DOMAINS[0]])/(row[f"count_{DOMAINS[1]}"]/exposure[DOMAINS[1]]) if row[f"count_{DOMAINS[1]}"] else np.nan
                rows.append(row)
            uf=pd.DataFrame(rows); uf["eligible_flat_count_ge_100"]=uf.count_outside_mask_flat>=100; uf["exclusion_reason"]=np.where(uf.eligible_flat_count_ge_100,"","flat_count_lt_100"); eligible=uf[uf.eligible_flat_count_ge_100].copy(); controls=[]
            for row in eligible.itertuples():
                near=eligible[(np.abs(eligible.median_flat_event_depth_um-row.median_flat_event_depth_um)<=10)&(eligible.unit_id!=row.unit_id)]
                controls.append(float(near.log_rate_outside_mask_flat.mean()) if len(near) else np.nan)
            eligible["same_row_other_unit_flat_log_rate"]=controls; uf=uf.merge(eligible[["unit_id","same_row_other_unit_flat_log_rate"]],on="unit_id",how="left"); unit_rows.extend(uf.to_dict("records"))
            r=spearmanr(eligible.log_rate_outside_mask_flat,eligible.log_rate_negative_excursion) if len(eligible)>=3 else (np.nan,np.nan)
            ok=np.isfinite(eligible.same_row_other_unit_flat_log_rate); rc=spearmanr(eligible.loc[ok,"same_row_other_unit_flat_log_rate"],eligible.loc[ok,"log_rate_negative_excursion"]) if ok.sum()>=3 else (np.nan,np.nan)
            rho_rows.append({"window":name,"arm":arm,"eligible_units":len(eligible),"excluded_units":len(uf)-len(eligible),"rho_negative_vs_flat":float(r.statistic if hasattr(r,"statistic") else r[0]),"rho_negative_vs_same_row_other_flat":float(rc.statistic if hasattr(rc,"statistic") else rc[0]),"same_row_control_units":int(ok.sum()),"identity_interpretation":False})
    pd.DataFrame(interval_rows).to_csv(out/"DOMAIN_INTERVALS.csv",index=False); pd.DataFrame(count_rows).to_csv(out/"PRIMARY_COUNTS_RATES.csv",index=False); pd.DataFrame(isi_rows).to_csv(out/"SEGMENT_SAFE_ISI.csv",index=False); pd.DataFrame(unit_rows).to_csv(out/"UNIT_RATE_INPUTS.csv",index=False); pd.DataFrame(rho_rows).to_csv(out/"UNIT_RATE_ASSOCIATIONS.csv",index=False)
    write_json(out/"VALIDATION.json",{"status":"pass","hash_checks":checks,"domains":list(DOMAINS),"catalogue_outside_remainder_is_true_rest":False,"rolling_baseline_knots":480,"field_depth_column":0,"field_max_abs_column_difference_um":max_column_difference,"exact_window_frame_bounds":{w["name"]:w["frames_half_open"] for w in cfg["windows"]},"no_cross_arm_unit_identity":True,"ks_noise_limit":"saved Kilosort event arrays have no DARTsort-style negative noise labels","elapsed_s":time.perf_counter()-started})
    (out/"README.md").write_text("# DE common saved-output scorecard\n\nExact field-domain accounting. `catalogue_outside_remainder` is operational and is not true rest. Unit IDs are arm-local.\n")
    products=[{"path":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in sorted(out.rglob("*")) if p.is_file() and p.name not in {"MANIFEST.json","COMPLETE.json"}]; write_json(out/"MANIFEST.json",{"products":products}); write_json(out/"COMPLETE.json",{"status":"complete","manifest_sha256":sha256(out/"MANIFEST.json"),"written_last_utc":datetime.now(timezone.utc).isoformat()})


if __name__=="__main__": main()
