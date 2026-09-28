#!/usr/bin/env python
"""Sorter-free raw-waveform lighthouse discovery for Luke0804 imec1 dots-RF.

No sorter times, labels, templates, locations, or motion estimates are read.
Independent raw detections in the frozen seed interval are clustered using only
normalized waveform samples on a canonical relative NP1 patch. Candidate rank
excludes absolute depth. Depth is revealed after identities are frozen, and the
same template bank then competes for detections in prespecified held-out windows.
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from testing.luke_imec1_dots_lighthouse_direct_check import MOTION_WINDOWS, ROOT, SEED_INTERVAL_S, atomic_json, sha256
from testing.luke_imec1_dots_raw_lighthouse_check import MANIFEST, OFFSETS, geometry_and_mapping, patches, preprocess, raw_sha_contract

SCHEMA="luke0804-imec1-dots-sorterfree-waveform-discovery-v2"
DEFAULT_OUTPUT=ROOT/"testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v2"
DETECTION_SIGMA=5.0
MIN_DETECTION_UV=30.0
N_CLUSTERS_PER_PHASE=48
PCA_COMPONENTS=20
MIN_FAMILY_EVENTS=30
MIN_HALF_EVENTS=10
MIN_REPEATABILITY=0.90
MIN_MEMBER_COSINE=0.80
MAX_RIVAL_COSINE=0.95
MIN_MEDIAN_SNR=6.0
N_SELECTED=50
STRICT_COSINE=0.86
DISPLAY_COSINE=0.80
IDENTITY_MARGIN=0.025
RANDOM_SEED=20260912


def phase_id(channel:int, bi:int, patch_channels:np.ndarray)->int:
    return int(np.flatnonzero(patch_channels[bi]==channel)[0])


def detect_waveforms(x:np.ndarray, fs:float, geom:np.ndarray, bases:np.ndarray, patch_channels:np.ndarray):
    """Both-sign local-maximum detector; returns canonical waves and metadata."""
    noise=np.median(np.abs(x[::10]-np.median(x[::10],axis=0)),axis=0)/0.67448975
    found=[]
    for channel in range(len(geom)):
        base=40.0*np.floor(geom[channel,1]/40.0); hit=np.flatnonzero(bases==base)
        if not len(hit): continue
        threshold=max(MIN_DETECTION_UV,DETECTION_SIGMA*noise[channel])
        ev=find_peaks(np.abs(x[:,channel]),height=threshold,distance=round(.0008*fs))[0]
        ev=ev[(ev>30)&(ev+30<len(x))]
        near=np.flatnonzero(np.abs(geom[:,1]-geom[channel,1])<=40.0)
        ev=ev[np.argmax(np.abs(x[ev[:,None],near]),axis=1)==np.flatnonzero(near==channel)[0]]
        bi=int(hit[0]); phase=phase_id(channel,bi,patch_channels)
        found.extend((int(e),channel,bi,phase,float(abs(x[e,channel])/noise[channel])) for e in ev)
    found=np.asarray(sorted(found),dtype=float).reshape(-1,5)
    if not len(found): return found,np.empty((0,len(OFFSETS),16),np.float32)
    waves=np.asarray([x[int(e)+OFFSETS][:,patch_channels[int(bi)]] for e,_,bi,_,_ in found],dtype=np.float32)
    return found,waves


def normalize_waveforms(waves:np.ndarray)->np.ndarray:
    flat=waves.reshape(len(waves),-1).astype(np.float32)
    return flat/np.maximum(np.linalg.norm(flat,axis=1,keepdims=True),1e-12)


def build_families(found:np.ndarray,waves:np.ndarray)->tuple[pd.DataFrame,pd.DataFrame,np.ndarray]:
    normalized=normalize_waveforms(waves); event_rows=[]; family_rows=[]; templates=[]
    for phase in sorted(np.unique(found[:,3]).astype(int)):
        ix=np.flatnonzero(found[:,3].astype(int)==phase)
        if len(ix)<2*N_CLUSTERS_PER_PHASE: continue
        x=normalized[ix]; components=min(PCA_COMPONENTS,len(x)-1,x.shape[1])
        pc=PCA(n_components=components,svd_solver="randomized",random_state=RANDOM_SEED+phase).fit_transform(x)
        n_clusters=min(N_CLUSTERS_PER_PHASE,max(2,len(ix)//50))
        labels=MiniBatchKMeans(n_clusters=n_clusters,n_init=3,batch_size=1024,random_state=RANDOM_SEED+phase).fit_predict(pc)
        phase_templates=[]; phase_records=[]
        for cluster in range(n_clusters):
            local=np.flatnonzero(labels==cluster); members=ix[local]
            if len(members)<MIN_FAMILY_EVENTS: continue
            first=members[::2]; second=members[1::2]
            if min(len(first),len(second))<MIN_HALF_EVENTS: continue
            a=np.median(normalized[first],axis=0); b=np.median(normalized[second],axis=0)
            a/=max(np.linalg.norm(a),1e-12); b/=max(np.linalg.norm(b),1e-12)
            template=a+b; template/=max(np.linalg.norm(template),1e-12)
            member_score=normalized[members]@template
            family_id=f"p{phase:02d}_f{cluster:03d}"
            phase_templates.append(template); templates.append(template.reshape(len(OFFSETS),16))
            phase_records.append({"family_id":family_id,"phase_id":phase,"seed_events":len(members),"first_half_events":len(first),"second_half_events":len(second),"split_half_cosine":float(a@b),"median_member_cosine":float(np.median(member_score)),"median_peak_snr":float(np.median(found[members,4]))})
            for row,score in zip(members,member_score): event_rows.append({"event_index":int(row),"family_id":family_id,"phase_id":phase,"member_cosine":float(score)})
        if phase_records:
            phase_flat=np.asarray(phase_templates); sim=phase_flat@phase_flat.T; np.fill_diagonal(sim,-np.inf)
            for record,rival in zip(phase_records,np.max(sim,axis=1)): record["nearest_rival_cosine"]=float(rival)
            family_rows.extend(phase_records)
    families=pd.DataFrame(family_rows)
    if families.empty: return families,pd.DataFrame(event_rows),np.empty((0,len(OFFSETS),16),np.float32)
    families["shape_eligible"]=(families.seed_events>=MIN_FAMILY_EVENTS)&(families.split_half_cosine>=MIN_REPEATABILITY)&(families.median_member_cosine>=MIN_MEMBER_COSINE)&(families.nearest_rival_cosine<MAX_RIVAL_COSINE)&(families.median_peak_snr>=MIN_MEDIAN_SNR)
    families["rank_score"]=(families.split_half_cosine*families.median_member_cosine*(1-families.nearest_rival_cosine.clip(upper=1))*np.log1p(families.seed_events)*np.log1p(families.median_peak_snr))
    families=families.sort_values(["shape_eligible","rank_score","family_id"],ascending=[False,False,True]).reset_index(drop=True)
    # Freeze a review panel, rather than calling only threshold passers candidates.
    # Sorting is still entirely depth blind: strict-gate families lead, followed
    # by the best lower-confidence waveform families.
    families["selected_depth_blind"]=False
    families.loc[families.index[:min(N_SELECTED,len(families))],"selected_depth_blind"]=True
    families["candidate_tier"]="not_selected"
    families.loc[families.selected_depth_blind,"candidate_tier"]="lower_confidence"
    families.loc[families.selected_depth_blind & families.shape_eligible,"candidate_tier"]="strict_shape_gate"
    families["candidate_rank_depth_blind"]=np.nan
    families.loc[families.selected_depth_blind,"candidate_rank_depth_blind"]=np.arange(1,int(families.selected_depth_blind.sum())+1)
    # Reorder templates to match the sorted family table.
    original_ids=[r["family_id"] for r in family_rows]; pos={fid:i for i,fid in enumerate(original_ids)}
    ordered=np.asarray([templates[pos[fid]] for fid in families.family_id],dtype=np.float32)
    return families,pd.DataFrame(event_rows),ordered


def add_seed_depth_after_freeze(families:pd.DataFrame,membership:pd.DataFrame,found:np.ndarray,waves:np.ndarray,geom:np.ndarray,patch_channels:np.ndarray)->tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]
    for row in membership.itertuples(index=False):
        event=int(row.event_index); bi=int(found[event,2]); energy=np.square(waves[event].astype(np.float64)).sum(axis=0)
        centroid=float(energy@geom[patch_channels[bi],1]/energy.sum())
        rows.append({**row._asdict(),"seed_frame_relative":int(found[event,0]),"peak_channel":int(found[event,1]),"peak_snr":float(found[event,4]),"waveform_centroid_um":centroid})
    event_table=pd.DataFrame(rows); out=families.copy(); stats=[]
    for fid,g in event_table.groupby("family_id"):
        stats.append({"family_id":fid,"seed_depth_median_um":float(g.waveform_centroid_um.median()),"seed_depth_p05_um":float(g.waveform_centroid_um.quantile(.05)),"seed_depth_p95_um":float(g.waveform_centroid_um.quantile(.95)),"seed_depth_p90_span_um":float(g.waveform_centroid_um.quantile(.95)-g.waveform_centroid_um.quantile(.05))})
    out=out.merge(pd.DataFrame(stats),on="family_id",how="left",validate="one_to_one")
    out["postmatch_seed_spatially_plausible"]=out.seed_depth_p90_span_um<=120.0
    return out,event_table


def score_block(waves:np.ndarray,templates:np.ndarray):
    """Cosine against real/time-reversed/space-reversed families."""
    flat=normalize_waveforms(waves); real=templates.reshape(len(templates),-1); real/=np.maximum(np.linalg.norm(real,axis=1,keepdims=True),1e-12)
    bank=np.concatenate([real,templates[:,::-1].reshape(len(templates),-1),templates[:,:,::-1].reshape(len(templates),-1)])
    bank/=np.maximum(np.linalg.norm(bank,axis=1,keepdims=True),1e-12)
    best=np.full((len(waves),len(bank)),-np.inf,np.float32)
    shaped=flat.reshape(len(flat),len(OFFSETS),16)
    for lag in range(-3,4):
        shifted=np.zeros_like(shaped)
        if lag<0: shifted[:,:lag]=shaped[:,-lag:]
        elif lag>0: shifted[:,lag:]=shaped[:,:-lag]
        else: shifted=shaped
        best=np.maximum(best,shifted.reshape(len(waves),-1)@bank.T)
    order=np.argsort(best,axis=1,kind="stable"); ar=np.arange(len(waves)); win=order[:,-1]
    return win,best[ar,win],best[ar,win]-best[ar,order[:,-2]]


def match_interval(x,found,waves,absolute_start,fs,geom,patch_channels,families,templates):
    selected=set(families.loc[families.selected_depth_blind,"family_id"]); rows=[]; counts={"detections":len(found),"decoy_winners":0,"below_display":0,"winner_not_selected":0}
    phase_values=families.phase_id.to_numpy(int)
    for phase in sorted(np.unique(found[:,3]).astype(int)):
        events=np.flatnonzero(found[:,3].astype(int)==phase); ti=np.flatnonzero(phase_values==phase)
        if len(ti)<2: continue
        for start in range(0,len(events),512):
            block=events[start:start+512]; winner,score,margin=score_block(waves[block],templates[ti]); nt=len(ti)
            counts["decoy_winners"]+=int((winner>=nt).sum())
            for event,win,sc,mar in zip(block,winner,score,margin):
                if win>=nt: continue
                family=families.iloc[ti[win]]
                if sc<DISPLAY_COSINE: counts["below_display"]+=1; continue
                if family.family_id not in selected: counts["winner_not_selected"]+=1; continue
                bi=int(found[event,2]); energy=np.square(waves[event].astype(np.float64)).sum(axis=0); centroid=float(energy@geom[patch_channels[bi],1]/energy.sum())
                status="strict" if sc>=STRICT_COSINE and mar>=IDENTITY_MARGIN else ("lower_score" if mar>=IDENTITY_MARGIN else "identity_ambiguous")
                rows.append({"family_id":family.family_id,"time_s":absolute_start+found[event,0]/fs,"phase_id":phase,"peak_channel":int(found[event,1]),"peak_snr":float(found[event,4]),"score":float(sc),"margin":float(mar),"status":status,"waveform_centroid_um":centroid})
    return pd.DataFrame(rows),counts


def plot_candidate_review(output:Path,families:pd.DataFrame,templates:np.ndarray,seed_events:pd.DataFrame,events:pd.DataFrame,fs:float,seed_interval_s=SEED_INTERVAL_S)->None:
    """Write one inspectable page per frozen candidate plus compact atlases."""
    page_dir=output/"candidate_pages"; page_dir.mkdir()
    selected=families[families.selected_depth_blind].copy()
    colors={"strict":"#1769aa","lower_score":"#f28e2b","identity_ambiguous":"#999999"}
    with PdfPages(output/"02_candidate_review_pages.pdf") as pdf:
        for family in selected.itertuples():
            rank=int(family.candidate_rank_depth_blind); fid=family.family_id
            template=templates[family.Index]
            seed=seed_events[seed_events.family_id.eq(fid)].copy()
            held=events[events.family_id.eq(fid)].copy() if len(events) else events
            fig,axs=plt.subplots(2,2,figsize=(12,8.5),layout="constrained")
            scale=max(float(np.max(np.abs(template))),1e-6)
            for channel in range(template.shape[1]):
                axs[0,0].plot(OFFSETS,template[:,channel]/scale+channel,color="black",lw=.65)
            axs[0,0].set(xlabel="Samples from detected peak",ylabel="Relative channel (normalized offset)",title="Frozen relative multichannel template")
            if len(seed):
                seed_time=seed_interval_s[0]+seed.seed_frame_relative/fs
                im=axs[0,1].scatter(seed_time,seed.waveform_centroid_um,c=seed.member_cosine,s=8,cmap="viridis",vmin=.6,vmax=1)
                fig.colorbar(im,ax=axs[0,1],label="Member cosine")
            axs[0,1].set(xlabel="Absolute recording time (s)",ylabel="Depth revealed after freeze (µm)",title=f"Seed members (n={len(seed)})")
            for status in ("identity_ambiguous","lower_score","strict"):
                q=held[held.status.eq(status)] if len(held) else held
                if len(q): axs[1,0].scatter(q.time_s,q.waveform_centroid_um,s=12,alpha=.75,c=colors[status],label=status)
            for start,stop in MOTION_WINDOWS: axs[1,0].axvspan(start,stop,color="#eeeeee",zorder=-2)
            axs[1,0].set(xlabel="Absolute recording time (s)",ylabel="Matched waveform centroid (µm)",title="Held-out identity evidence")
            if len(held): axs[1,0].legend(fontsize=7)
            status_counts=held.status.value_counts() if len(held) else pd.Series(dtype=int)
            axs[1,1].axis("off")
            axs[1,1].text(0,1,"\n".join([
                f"Depth-blind rank: {rank}/{len(selected)}",
                f"Tier: {family.candidate_tier}",
                f"Seed events: {family.seed_events}",
                f"Split-half cosine: {family.split_half_cosine:.3f}",
                f"Median member cosine: {family.median_member_cosine:.3f}",
                f"Nearest rival cosine: {family.nearest_rival_cosine:.3f}",
                f"Median peak SNR: {family.median_peak_snr:.1f}",
                f"Seed P90 depth span: {family.seed_depth_p90_span_um:.1f} µm",
                f"Strict held-out: {int(status_counts.get('strict',0))}",
                f"Lower-score held-out: {int(status_counts.get('lower_score',0))}",
                f"Identity-ambiguous: {int(status_counts.get('identity_ambiguous',0))}",
                "Absolute depth was unavailable to ranking.",
            ]),va="top",family="monospace",fontsize=10)
            fig.suptitle(f"imec1 sorter-free waveform candidate {rank:02d}: {fid}",fontsize=14)
            pdf.savefig(fig)
            fig.savefig(page_dir/f"{rank:02d}_{fid}.png",dpi=150)
            plt.close(fig)
    with PdfPages(output/"03_candidate_track_atlas.pdf") as pdf:
        for first in range(0,len(selected),25):
            block=selected.iloc[first:first+25]
            fig,axs=plt.subplots(5,5,figsize=(15,11),sharex=True,layout="constrained")
            for ax,(_,family) in zip(axs.flat,block.iterrows()):
                held=events[events.family_id.eq(family.family_id)] if len(events) else events
                for status in ("identity_ambiguous","lower_score","strict"):
                    q=held[held.status.eq(status)] if len(held) else held
                    if len(q): ax.scatter(q.time_s,q.waveform_centroid_um-family.seed_depth_median_um,s=5,c=colors[status],alpha=.7)
                ax.axhline(0,color="black",lw=.4)
                ax.set_title(f"{int(family.candidate_rank_depth_blind):02d} {family.family_id} [{family.candidate_tier[0]}]",fontsize=7)
                ax.tick_params(labelsize=6)
            for ax in axs.flat[len(block):]: ax.axis("off")
            fig.supxlabel("Absolute recording time (s)"); fig.supylabel("Depth relative to seed median (µm)")
            fig.suptitle(f"Frozen waveform candidates {first+1}–{first+len(block)}: strict blue, lower orange, ambiguous gray")
            pdf.savefig(fig); plt.close(fig)


def run(config:dict)->None:
    output=Path(config["output"]); output.mkdir(parents=True,exist_ok=False); atomic_json(output/"settings.json",config)
    if config.get("dummy"):
        print("dummy started",flush=True); time.sleep(20); atomic_json(output/"summary.json",{"schema":SCHEMA,"status":"complete","dummy":True}); print("dummy completed",flush=True); return
    manifest=json.loads(MANIFEST.read_text()); binary=Path(manifest["binary_path"]); fs=float(manifest["sampling_frequency_hz"]); before=raw_sha_contract(binary)
    raw=np.memmap(binary,dtype="<i2",mode="r",shape=(manifest["num_samples"],manifest["saved_channels_in_binary"])); geom,kept_raw=geometry_and_mapping(); bases,patch_channels=patches(geom)
    seed_start,seed_stop=SEED_INTERVAL_S; x=preprocess(raw,round(seed_start*fs),round(seed_stop*fs),kept_raw,fs); found,waves=detect_waveforms(x,fs,geom,bases,patch_channels)
    seed_detection_count=len(found); print("seed detections",seed_detection_count,flush=True); families,membership,templates=build_families(found,waves)
    frozen=families[[c for c in families.columns if not c.startswith("seed_depth") and not c.startswith("postmatch")]].copy(); frozen.to_csv(output/"families_depth_blind_frozen.csv",index=False)
    np.savez_compressed(output/"family_templates.npz",waveforms=templates,relative_geometry=geom[patch_channels[0]]-np.array([0.,bases[0]]))
    families,seed_events=add_seed_depth_after_freeze(families,membership,found,waves,geom,patch_channels); families.to_csv(output/"families_after_depth_reveal.csv",index=False); seed_events.to_csv(output/"seed_cluster_events.csv",index=False)
    print("families",len(families),"eligible",int(families.shape_eligible.sum()),"selected",int(families.selected_depth_blind.sum()),flush=True)
    del x,waves,membership
    parts=[]; count_rows=[]
    for index,(start,stop) in enumerate(MOTION_WINDOWS):
        x=preprocess(raw,round(start*fs),round(stop*fs),kept_raw,fs); detected,wave=detect_waveforms(x,fs,geom,bases,patch_channels)
        matched,counts=match_interval(x,detected,wave,start,fs,geom,patch_channels,families,templates); matched.to_csv(output/f"heldout_{index:02d}_events.csv",index=False); atomic_json(output/f"heldout_{index:02d}.complete.json",{"sha256":sha256(output/f"heldout_{index:02d}_events.csv"),**counts}); parts.append(matched); count_rows.append({"start_s":start,"stop_s":stop,**counts}); print(index,start,stop,counts,flush=True)
    events=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(); events.to_csv(output/"heldout_events.csv",index=False); pd.DataFrame(count_rows).to_csv(output/"heldout_counts.csv",index=False)
    strict=events[events.status.eq("strict")].copy() if len(events) else events; seed_center=families.set_index("family_id").seed_depth_median_um
    if len(strict): strict["relative_um"]=strict.waveform_centroid_um-strict.family_id.map(seed_center)
    rows=[]
    for family in families[families.selected_depth_blind].itertuples():
        q=strict[strict.family_id.eq(family.family_id)] if len(strict) else strict
        rows.append({"family_id":family.family_id,"seed_shape_eligible":bool(family.shape_eligible),"seed_spatially_plausible_postmatch":bool(family.postmatch_seed_spatially_plausible),"strict_heldout_matches":len(q),"heldout_windows":int(sum(((q.time_s>=a)&(q.time_s<b)).any() for a,b in MOTION_WINDOWS)) if len(q) else 0,"heldout_depth_span_um":float(q.waveform_centroid_um.max()-q.waveform_centroid_um.min()) if len(q) else np.nan,"provisional_lighthouse_candidate":bool(family.postmatch_seed_spatially_plausible and len(q)>=20 and sum(((q.time_s>=a)&(q.time_s<b)).any() for a,b in MOTION_WINDOWS)>=3 and q.waveform_centroid_um.max()-q.waveform_centroid_um.min()<=500) if len(q) else False})
    tracks=pd.DataFrame(rows); tracks.to_csv(output/"candidate_track_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(13,7),layout="constrained")
    for fid,g in strict.groupby("family_id") if len(strict) else []: ax.scatter(g.time_s,g.waveform_centroid_um,s=9,alpha=.6,label=fid)
    ax.set(xlabel="Absolute recording time (s)",ylabel="Waveform energy centroid after matching (µm)",title="Sorter-free waveform families: strict held-out matches\nDiscovery and ranking used no sorter, absolute depth, or motion estimate")
    if len(strict) and strict.family_id.nunique()<=12: ax.legend(fontsize=7,ncol=2)
    fig.savefig(output/"01_strict_heldout_depth_atlas.png",dpi=170); fig.savefig(output/"01_strict_heldout_depth_atlas.pdf"); plt.close(fig)
    plot_candidate_review(output,families,templates,seed_events,events,fs)
    if raw_sha_contract(binary)!=before: raise RuntimeError("raw source changed")
    summary={"schema":SCHEMA,"status":"complete","sorter_inputs_read":False,"motion_estimate_read":False,"seed_interval_s":list(SEED_INTERVAL_S),"seed_detections":seed_detection_count,"waveform_families":len(families),"shape_eligible_families":int(families.shape_eligible.sum()),"depth_blind_selected_families":int(families.selected_depth_blind.sum()),"strict_tier_candidates":int((families.candidate_tier=="strict_shape_gate").sum()),"lower_confidence_tier_candidates":int((families.candidate_tier=="lower_confidence").sum()),"candidate_review_pages":int(families.selected_depth_blind.sum()),"strict_heldout_events":len(strict),"provisional_lighthouse_candidates":tracks.loc[tracks.provisional_lighthouse_candidate,"family_id"].tolist(),"interpretation":"Candidate discovery only. The 50-family panel includes lower-confidence waveform candidates for review; selection count is not a claim of 50 cells or lighthouses.","limitations":["Exact 40-um translation support has alternate-phase dropout.","MiniBatchKMeans partitions waveform morphology and may merge unrelated lookalikes or split one variable identity.","Thresholds and candidate count are prespecified exploratory choices, not calibrated biological identity guarantees."]}
    atomic_json(output/"summary.json",summary); (output/"README.md").write_text("# Sorter-free imec1 waveform discovery\n\nNo spike sorter or motion estimate is read. Raw seed detections are clustered on normalized relative multichannel waveforms; the candidate table is frozen before absolute depth is attached. All families remain held-out rivals, with decoy competitors.\n"); print(json.dumps(summary,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--dummy",action="store_true"); args=ap.parse_args(); config=json.loads(args.config.read_text()); config["dummy"]=args.dummy; run(config)
if __name__=="__main__": main()
