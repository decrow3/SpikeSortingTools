#!/usr/bin/env python
"""Bounded broad-vs-compact proposal comparison on the 300--320 s seed bank.

Existing cluster assignments are held fixed. Templates use only 300--310 s;
310--320 s is qualification. Both arms face the same external s036 rival bank.
No depth or motion enters template construction, scoring, or qualification.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd

from testing.luke_imec1_dots_raw_lighthouse_check import MANIFEST, geometry_and_mapping, patches, preprocess, raw_sha_contract
from testing.luke_imec1_dots_sorterfree_waveform_discovery import detect_waveforms, normalize_waveforms
from testing.luke_imec1_dots_sorterfree_waveform_discovery_v3 import classify_scores, global_rivals, score_global
from testing.luke_imec1_dots_lighthouse_direct_check import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300'
EXTERNAL=ROOT/'testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3'
DEFAULT_OUTPUT=ROOT/'testing/outputs/luke_imec1_compact_core_pilot_v1'
SCHEMA='luke0804-imec1-compact-core-pilot-v1'
TRAIN=(300.0,310.0); QUALIFY=(310.0,320.0)
CORE_COSINE=.86; MIN_HALF_CORE=5; MIN_QUALIFIED=5; MIN_QUALIFIED_FRACTION=.20


def lagged_scores(waves:np.ndarray,template:np.ndarray)->np.ndarray:
    target=template.reshape(-1);target=target/max(np.linalg.norm(target),1e-20)
    out=np.full(len(waves),-np.inf)
    for lag in range(-3,4):
        shifted=np.zeros_like(waves)
        if lag<0: shifted[:,:lag]=waves[:,-lag:]
        elif lag>0: shifted[:,lag:]=waves[:,:-lag]
        else: shifted=waves
        flat=normalize_waveforms(shifted)
        out=np.maximum(out,flat@target)
    return out


def make_template(waves:np.ndarray)->tuple[np.ndarray,float]:
    flat=normalize_waveforms(waves)
    shape=np.median(flat,axis=0);shape/=max(np.linalg.norm(shape),1e-20)
    scale=float(np.median(np.linalg.norm(waves.reshape(len(waves),-1),axis=1)))
    return shape.reshape(waves.shape[1:]).astype(np.float32),scale


def build_arms(found,waves,membership,families):
    train_stop=(TRAIN[1]-TRAIN[0])*29999.835983263598
    rows=[]; broad=[];compact=[];bscale=[];cscale=[]
    for family in families.itertuples():
        ids=membership.loc[membership.family_id.eq(family.family_id),'event_index'].to_numpy(int)
        ids=ids[found[ids,0]<train_stop]
        if len(ids)<20:continue
        a,b=ids[::2],ids[1::2]
        if min(len(a),len(b))<5:continue
        ta,_=make_template(waves[a]);tb,_=make_template(waves[b])
        core_a=a[lagged_scores(waves[a],tb)>=CORE_COSINE]
        core_b=b[lagged_scores(waves[b],ta)>=CORE_COSINE]
        broad_t,broad_s=make_template(waves[ids])
        if min(len(core_a),len(core_b))>=MIN_HALF_CORE:
            compact_t,compact_s=make_template(waves[np.r_[core_a,core_b]])
            compact_repeat=float(lagged_scores(compact_t[None],broad_t)[0])
        else:
            compact_t,compact_s,compact_repeat=broad_t,broad_s,np.nan
        rows.append({'family_id':family.family_id,'phase_id':int(family.phase_id),'train_members':len(ids),'core_a':len(core_a),'core_b':len(core_b),'core_events':len(core_a)+len(core_b),'core_fraction':(len(core_a)+len(core_b))/len(ids),'has_compact_core':min(len(core_a),len(core_b))>=MIN_HALF_CORE,'compact_vs_broad_cosine':compact_repeat})
        broad.append(broad_t);compact.append(compact_t);bscale.append(broad_s);cscale.append(compact_s)
    return pd.DataFrame(rows),np.asarray(broad),np.asarray(compact),np.asarray(bscale),np.asarray(cscale)


def qualify_arm(name,proposal,templates,scales,external_families,external_templates,external_scales,qual_found,qual_waves,relgeom):
    own=proposal.copy();own['bank']='s300';ext=external_families.copy();ext['bank']='s036'
    combined=pd.concat([own,ext],ignore_index=True,sort=False)
    all_templates=np.concatenate([templates,external_templates]);all_scales=np.r_[scales,external_scales]
    scored=score_global(qual_waves,qual_found[:,3].astype(int),all_templates,combined.phase_id.to_numpy(int),all_scales,relgeom)
    status=classify_scores(scored);winner=scored['winner_template'];rows=[]
    for i,r in own.iterrows():
        ix=np.flatnonzero((winner==i)&(status=='strict'))
        depth=[]
        for event in ix:
            bi=int(qual_found[event,2]);energy=np.square(qual_waves[event].astype(float)).sum(axis=0)
            depth.append(float(energy@GLOBAL_GEOM[GLOBAL_PATCHES[bi],1]/energy.sum()))
        rows.append({'arm':name,'family_id':r.family_id,'phase_id':int(r.phase_id),'train_members':int(r.train_members),'has_compact_core':bool(r.has_compact_core),'qualification_strict':len(ix),'qualification_fraction_of_train':len(ix)/r.train_members,'qualifies_depth_blind':len(ix)>=MIN_QUALIFIED and len(ix)/r.train_members>=MIN_QUALIFIED_FRACTION,'qualification_depth_p90_span_um':float(np.quantile(depth,.95)-np.quantile(depth,.05)) if len(depth) else np.nan})
    return pd.DataFrame(rows),{'arm':name,'qualification_detections':len(qual_waves),'strict_all_combined':int((status=='strict').sum()),'decoy_winners':int((status=='decoy_winner').sum()),'external_winners':int(((winner>=len(own))&(status=='strict')).sum())}


GLOBAL_GEOM=None;GLOBAL_PATCHES=None
def run(config):
    global GLOBAL_GEOM,GLOBAL_PATCHES
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=False);atomic_json(output/'settings.json',config)
    if config.get('dummy'):
        time.sleep(20);atomic_json(output/'summary.json',{'schema':SCHEMA,'status':'complete','dummy':True});return
    manifest=json.loads(MANIFEST.read_text());binary=Path(manifest['binary_path']);fs=float(manifest['sampling_frequency_hz']);before=raw_sha_contract(binary)
    raw=np.memmap(binary,dtype='<i2',mode='r',shape=(manifest['num_samples'],manifest['saved_channels_in_binary']))
    geom,kept=geometry_and_mapping();bases,patch_channels=patches(geom);GLOBAL_GEOM,GLOBAL_PATCHES=geom,patch_channels
    x=preprocess(raw,round(TRAIN[0]*fs),round(QUALIFY[1]*fs),kept,fs);found,waves=detect_waveforms(x,fs,geom,bases,patch_channels)
    source=Path(config.get('source_bank',SOURCE));external=Path(config.get('external_bank',EXTERNAL))
    expected=json.loads((source/'summary.json').read_text())['seed_detections']
    if len(found)!=expected:raise RuntimeError(f'detection replay mismatch {len(found)} != {expected}')
    membership=pd.read_csv(source/'seed_cluster_events.csv')[['event_index','family_id']]
    families=pd.read_csv(source/'families_after_depth_reveal.csv')
    proposal,broad,compact,bscale,cscale=build_arms(found,waves,membership,families)
    proposal.to_csv(output/'proposal_core_build.csv',index=False)
    # Freeze proposal/core construction before the qualification half is scored.
    np.savez_compressed(output/'frozen_train_templates.npz',family_id=proposal.family_id.to_numpy(),phase_id=proposal.phase_id.to_numpy(),broad=broad,compact=compact,broad_scale=bscale,compact_scale=cscale)
    external_families=pd.read_csv(external/'families_after_depth_reveal.csv')[['family_id','phase_id']].copy();external_families['family_id']='s036:'+external_families.family_id
    with np.load(external/'family_templates.npz') as z:
        external_templates=z['waveforms'];external_scales=z['template_scale_uv'];relgeom=z['relative_geometry']
    qual=found[:,0]>=round((QUALIFY[0]-TRAIN[0])*fs);qfound=found[qual].copy();qfound[:,0]-=round((QUALIFY[0]-TRAIN[0])*fs);qwaves=waves[qual]
    arms=[];counters=[]
    for name,t,s in [('broad',broad,bscale),('compact',compact,cscale)]:
        table,count=qualify_arm(name,proposal,t,s,external_families,external_templates,external_scales,qfound,qwaves,relgeom);arms.append(table);counters.append(count)
    result=pd.concat(arms,ignore_index=True);result.to_csv(output/'qualification_comparison.csv',index=False);pd.DataFrame(counters).to_csv(output/'arm_counters.csv',index=False)
    paired=result.pivot(index='family_id',columns='arm',values=['qualification_strict','qualifies_depth_blind','qualification_depth_p90_span_um']).reset_index();paired.to_csv(output/'paired_proposal_outcomes.csv',index=False)
    if raw_sha_contract(binary)!=before:raise RuntimeError('raw source changed')
    summary={'schema':SCHEMA,'status':'complete','raw_voltage_read':True,'motion_estimate_read':False,'sorter_input_read':False,'train_s':TRAIN,'qualification_s':QUALIFY,'existing_cluster_assignments_saw_full_20s':True,'proposal_families_with_train_support':len(proposal),'families_with_compact_core':int(proposal.has_compact_core.sum()),'broad_depth_blind_qualified':int(result[(result.arm=='broad')].qualifies_depth_blind.sum()),'compact_depth_blind_qualified':int(result[(result.arm=='compact')].qualifies_depth_blind.sum()),'compact_new_qualifiers_vs_broad':sorted(set(result[(result.arm=='compact')&result.qualifies_depth_blind].family_id)-set(result[(result.arm=='broad')&result.qualifies_depth_blind].family_id)),'interpretation':'Bounded trimming/alignment test only. Existing all-20-s cluster assignments limit qualification independence; promotion requires a fresh proposal split.'}
    atomic_json(output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--dummy',action='store_true');args=ap.parse_args();config=json.loads(args.config.read_text());config['dummy']=args.dummy;run(config)
if __name__=='__main__':main()
