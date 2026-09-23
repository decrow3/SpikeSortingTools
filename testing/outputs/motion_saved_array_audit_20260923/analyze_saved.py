"""Saved-array arithmetic and export only. No voltage reads, estimators, or sorters."""
from pathlib import Path
import json, hashlib, shutil, sys, gzip
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
ROOT=Path('/home/huklab/Documents/RyanSorting/SpikeSortingTools')
OUT=Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
DATA=OUT/'export'; DATA.mkdir(exist_ok=True)
manifest=[]
def cp(src,rel):
 src=Path(src); dest=DATA/rel; dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 a,b=sha(src),sha(dest);assert a==b
 manifest.append(dict(source=str(src),export=str(dest.relative_to(OUT)),bytes=dest.stat().st_size,sha256=b))
 return dest
def directory(src,rel,extensions={'.npy','.npz','.json'}):
 for p in Path(src).iterdir():
  if p.is_file() and p.suffix in extensions:cp(p,Path(rel)/p.name)
def write(name,rows):pd.DataFrame(rows).to_csv(OUT/name,index=False)
def span(x,axis=0):return np.diff(np.percentile(x,[5,95],axis=axis),axis=0)[0]
PILOT=ROOT/'testing/outputs/cross_dataset_fast_motion_v2'
for probe in ['imec0','imec1']:
 for frac in [10,50,90]:
  p=PILOT/f'luke_{probe}_p{frac}/fit/field.npz';cp(p,f'pilot/luke_{probe}_p{frac}/field.npz')
cp(PILOT/'config.json','pilot/config.json')
SHARED=Path('/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-full-medicine-shared-recovery-20260913-223213')
MERGE=Path('/home/huklab/DARTsort_runs/luke0804-imec1-full-medicine-merge-bias-nvme-v1-20260920')
for label,p in [('shared_recovery',SHARED),('merge_bias',MERGE)]:
 directory(p/'motion',f'full_session/{label}/motion')
 if (p/'motion/raw').exists():directory(p/'motion/raw',f'full_session/{label}/motion/raw')
 for name in ['config.json','medicine-config.json','input-manifest.json']:
  if (p/name).exists():cp(p/name,f'full_session/{label}/{name}')
OLD=Path('/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec1')
directory(OLD/'motion/dredge-motion','original_dredge')
OLDMED=OLD/'motion_scale_sweep/runs/imec1/medicine_pipeline_default/full_5ddcdaf9c1bcecad'
# Only compact fields and provenance, not peak caches or voltage.
for name in ['motion.npy','time_bins.npy','depth_bins.npy','manifest.json']:
 if (OLDMED/name).exists():cp(OLDMED/name,'older_medicine/'+name)
cp(ROOT/'testing/outputs/luke_motion_candidate_results/internal_rigid_trace_comparison.csv','ks_internal/internal_rigid_trace_comparison.csv')
cp(ROOT/'testing/outputs/luke_upstream_stage_ablation/imec1/manifest.json','original_dredge/ablation_manifest.json')
cp(ROOT/'testing/outputs/luke_upstream_stage_ablation/imec1/motion_window_summary.csv','original_dredge/motion_window_summary.csv')
MATRIX=Path('/media/huklab/Data/luke_motion_348ch_matrix_v1_job/config.json');cfg=json.loads(MATRIX.read_text())
cp(MATRIX,'matrix_imec0/config.json')
for key in ['ap_resolved_fields','lfp_resolved_fields','dartsort_motion_export','dartsort_motion_audit']:
 cp(cfg[key],f'matrix_imec0/{key}{Path(cfg[key]).suffix}')
for p in (ROOT/'testing/outputs/luke_lfp_lighthouse_validation_v1/lfp_package').glob('*motion.npz'):
 cp(p,'quiet_lfp_package/'+p.name)
for name in ['candidate_increments.csv','candidate_summary.csv','binned_family_tracks.csv']:
 cp(ROOT/'testing/outputs/luke_motion_candidate_lighthouse_comparison_v1'/name,'quiet_lighthouse_imec0/'+name)
# Preserve all discovered imec1 scored event tables separately, including obsolete controls.
# Avoid pooling revisions or interpreting alternative hypotheses as independent events.
event_inventory=[]
for p in sorted((ROOT/'testing/outputs').glob('luke_imec1*/*.csv')):
 cols=pd.read_csv(p,nrows=0).columns
 if not ('time_s' in cols and 'score' in cols and ('waveform_centroid_um' in cols or 'depth_um' in cols)):continue
 d=pd.read_csv(p); rel=f'imec1_events/original/{p.parent.name}/{p.name}.gz'
 dest=DATA/rel;dest.parent.mkdir(parents=True,exist_ok=True)
 with p.open('rb') as src,gzip.open(dest,'wb',compresslevel=6) as dst:shutil.copyfileobj(src,dst)
 source_hash=hashlib.sha256(p.read_bytes()).hexdigest()
 with gzip.open(dest,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==source_hash
 manifest.append(dict(source=str(p),export=str(dest.relative_to(OUT)),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),uncompressed_sha256=source_hash))
 depth='waveform_centroid_um' if 'waveform_centroid_um' in d else 'depth_um'
 candidate=next((k for k in ['winner_family','family_id','unit_id'] if k in d),None)
 if candidate and p.name in ['event_predictions.csv','heldout_events.csv','events.csv','waveform_matches.csv','strict_primary_events.csv','primary_uncertain_events.csv','qualification_events_compact.csv']:
  # Preserve all original columns including status, event IDs, bank, rivals, margins and windows.
  if 'candidate' not in d:d['candidate']=d[candidate].astype(str)
  if 'depth_um' not in d:d['depth_um']=d[depth]
  norm=DATA/f'imec1_events/normalized/{p.parent.name}/{p.stem}.csv.gz';norm.parent.mkdir(parents=True,exist_ok=True);d.to_csv(norm,index=False,compression='gzip')
 t=d.time_s
 event_inventory.append(dict(source=str(p),rows=len(d),time_min_s=t.min(),time_max_s=t.max(),p10_rows=((t>=987.3553929363488)&(t<1107.3553896029887)).sum(),late_rows=((t>=8160)&(t<8280)).sum(),candidate_column=candidate,depth_column=depth))
write('event_inventory.csv',event_inventory)
for p in ['testing/outputs/luke_imec1_compact_alignment_pilot_v2/frozen_training_templates.npz','testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/family_templates.npz','testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/families_after_depth_reveal.csv','testing/outputs/luke_imec1_lighthouse_motion_comparison_v7/shared_movement_pairwise.csv','testing/imec1_lighthouse_motion_manifest_v1.json']:
 cp(ROOT/p,'identity/'+Path(p).name)
fields={}
z=np.load(PILOT/'luke_imec1_p10/fit/field.npz');fields['pilot_p10']=(z['session_time_s'],z['depth_um'],z['displacement_um'].astype(float))
for name,p in [('shared_recovery',SHARED),('merge_bias',MERGE)]:
 z=np.load(p/'motion/fields.npz');fields[name]=(z['time_bin_centers_s'],z['spatial_bin_centers_um'],z['displacement'].T.astype(float))
windows={'p10':(987.3553929363488,1107.3553896029887),'8160_8280':(8160.,8280.)}
excursions=[];lagrows=[];fastrows=[];spreads=[];step_rows=[]
for name,(t,z,m) in fields.items():
 for win,(a,b) in windows.items():
  keep=(t>=a)&(t<b);tt=t[keep];x=m[keep]
  if len(x)<2:continue
  assert np.allclose(np.diff(tt),.25,atol=1e-6)
  # Rigid = median across native saved depth traces; also export centered-depth variant.
  components={'rigid':np.median(x,axis=1),'rigid_after_depth_centering':np.median(x-np.median(x,axis=0),axis=1)}
  components.update({f'depth_{i}':x[:,i] for i in range(x.shape[1])})
  for comp,v in components.items():
   p5,p95=np.percentile(v,[5,95]);j=int(np.argmax(abs(np.diff(v))))
   excursions.append(dict(field=name,window=win,component=comp,depth_um=float(z[int(comp[6:])]) if comp.startswith('depth_') else np.nan,n=len(v),p5_um=p5,p95_um=p95,excursion_um=p95-p5))
   step_rows.append(dict(field=name,window=win,component=comp,max_abs_step_um=abs(v[j+1]-v[j]),signed_step_um=v[j+1]-v[j],from_s=tt[j],to_s=tt[j+1]))
   for lag in [.25,1,2,5,20]:
    k=round(lag/.25);d=v[k:]-v[:-k]
    lagrows.append(dict(field=name,window=win,component=comp,lag_s=lag,n_pairs=len(d),mean_increment_um=d.mean(),sd_increment_um=np.std(d,ddof=0)))
   # Descriptive high-frequency sensitivity, not an additive excursion decomposition.
   high=sosfiltfilt(butter(4,.5,btype='highpass',fs=4,output='sos'),v)
   vv=v[8:-8];hh=high[8:-8]
   fastrows.append(dict(field=name,window=win,component=comp,trim_s=2,cutoff_hz=.5,full_excursion_same_interior_um=span(vv),highpass_excursion_um=span(hh),excursion_ratio=span(hh)/span(vv),highpass_sd_um=np.std(hh),variance_ratio=np.var(hh)/np.var(vv)))
  for centered in [False,True]:
   xx=x-np.median(x,axis=0) if centered else x
   ss=np.percentile(xx,95,axis=1)-np.percentile(xx,5,axis=1)
   spreads.append(dict(field=name,window=win,depth_centered=centered,median_spread_um=np.median(ss),p95_spread_um=np.percentile(ss,95),temporal_depth_medians_um=json.dumps(np.median(x,axis=0).tolist())))
# Original DREDGE offset audited from the saved ablation manifest.
t=np.load(OLD/'motion/dredge-motion/time_bins.npy')-3057.6775463558583
m=np.load(OLD/'motion/dredge-motion/motion.npy');x=m[(t>=8160)&(t<8280)]
for centered in [False,True]:
 xx=x-np.median(x,axis=0) if centered else x;ss=np.percentile(xx,95,axis=1)-np.percentile(xx,5,axis=1)
 spreads.append(dict(field='original_dredge',window='8160_8280',depth_centered=centered,median_spread_um=np.median(ss),p95_spread_um=np.percentile(ss,95),temporal_depth_medians_um=json.dumps(np.median(x,axis=0).tolist())))
for name,rows in [('excursions.csv',excursions),('increment_sd.csv',lagrows),('fast_component.csv',fastrows),('depth_spread.csv',spreads),('max_steps.csv',step_rows)]:write(name,rows)
# Matrix fields: imec0, not imec1. LFP fields are rigid, one trace for every depth.
ap=np.load(cfg['ap_resolved_fields']);lfp=np.load(cfg['lfp_resolved_fields']);native=np.load(cfg['dartsort_motion_export'])
mat={n:(lfp[n+'_time_s'],np.array([np.nan]),lfp[n+'_displacement_um'][:,None]) for n in ['lfp_native','lfp_savgol']}
mat['fast_medicine_nonrigid']=(ap['ap_time_s'],ap['ap_depth_um'],ap['ap_nonrigid_displacement_um'])
mat['fast_medicine_applied_rigid']=(ap['ap_time_s'],np.array([np.nan]),ap['ap_rigid_displacement_um'][:,None])
mat['native_ap']=(native['time_s'],native['depth_um'],native['displacement_um'])
rows=[]
for name,(t,z,m) in mat.items():
 x=m[(t>=930)&(t<1030)]
 for comp,v,d in [('rigid',np.median(x,axis=1),np.nan)]+[(f'depth_{i}',x[:,i],z[i]) for i in range(x.shape[1])]:
  p5,p95=np.percentile(v,[5,95]);rows.append(dict(field=name,component=comp,depth_um=d,n=len(v),p5_um=p5,p95_um=p95,excursion_um=p95-p5))
write('matrix_excursions.csv',rows)
# Actual candidate motion RMS, and error RMS vs lighthouse, each family weighted equally.
d=pd.read_csv(ROOT/'testing/outputs/luke_motion_candidate_lighthouse_comparison_v1/candidate_increments.csv')
d=d[(d.sensitivity=='plausibility_lattice_node_5um')&(d.regime=='quiet')]
rows=[]
for (win,cand),g in d.groupby(['window','candidate']):
 eligible=g.groupby('family_id').size();eligible=eligible[eligible>=1].index;g=g[g.family_id.isin(eligible)]
 rows.append(dict(window=win,candidate=cand,families=g.family_id.nunique(),increments=len(g),candidate_increment_rms_um=np.sqrt(g.assign(v=g.candidate_delta_um**2).groupby('family_id').v.mean().mean()),lighthouse_increment_rms_um=np.sqrt(g.assign(v=g.lighthouse_delta_um**2).groupby('family_id').v.mean().mean()),residual_rmse_um=np.sqrt(g.assign(v=g.residual_um**2).groupby('family_id').v.mean().mean())))
write('quiet_motion.csv',rows)
# Duplicate evidence on final strict assignments: mutually exclusive labels can partition one cell.
d=pd.read_csv(ROOT/'testing/outputs/luke_imec1_lighthouse_motion_comparison_v7/event_predictions.csv')
a=d[d.winner_family=='p06_f010'].sort_values('time_s');b=d[d.winner_family=='p08_f044'].sort_values('time_s')
aa=a.time_s.to_numpy();bb=b.time_s.to_numpy();pairs=[];i=j=0
while i<len(aa) and j<len(bb):
 if abs(aa[i]-bb[j])<=.0005:pairs.append((int(a.index[i]),int(b.index[j]),float(bb[j]-aa[i])));i+=1;j+=1
 elif aa[i]<bb[j]:i+=1
 else:j+=1
# Only geometry-aware waveform cosine: phases differ, so flat-index cosine is not comparable.
sys.path.insert(0,str(ROOT))
from testing.luke_imec1_sorterfree_phase_audit_v1 import lagged_overlap_cosine
z=np.load(ROOT/'testing/outputs/luke_imec1_compact_alignment_pilot_v2/frozen_training_templates.npz',allow_pickle=True)
ge=np.load(ROOT/'testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/family_templates.npz')['relative_geometry']
ii=list(z['family_id']).index('p06_f010');jj=list(z['family_id']).index('p08_f044')
co,coverage,ncommon=lagged_overlap_cosine(z['compact'][ii],int(z['phase_id'][ii]),z['compact'][jj],int(z['phase_id'][jj]),ge)
identity=dict(candidate_a='p06_f010',candidate_b='p08_f044',n_a=len(a),n_b=len(b),pairs_within_0p5ms=len(pairs),fraction_a=len(pairs)/len(a),fraction_b=len(pairs)/len(b),compact_geometry_aware_best_lag_cosine=co,minimum_energy_coverage=coverage,common_channels=ncommon,matched_pairs=pairs)
(OUT/'identity_overlap.json').write_text(json.dumps(identity,indent=2)+'\n')
# Irregular lighthouse observations: 0.25s bin medians; endpoint pairs only, no interpolation.
# Historical records are kept separate. Short acquisition windows cannot support 20s increments.
canon=['luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/heldout_events.csv','luke_imec1_dots_sorterfree_waveform_discovery_v3/heldout_events.csv','luke_imec1_dots_raw_lighthouse_v2/events.csv','luke_imec1_dots_waveform_lighthouse_check_v3/waveform_matches.csv','luke_imec1_candidate_1690_report_v1/candidate_events_with_postmatch_field.csv']
observations=[(998,1002),(1014,1025),(1071,1075),(1097,1112)]
rows=[]
for rel in canon:
 p=ROOT/'testing/outputs'/rel
 if not p.exists():continue
 d=pd.read_csv(p);key=next(k for k in ['family_id','unit_id','winner_family'] if k in d)
 d=d[(d.time_s>=windows['p10'][0])&(d.time_s<windows['p10'][1])]
 if 'status' in d:d=d[d.status.eq('strict')]
 for ident,g in d.groupby(key):
  diffs={lag:[] for lag in [.25,1,2,5,20]}
  for a,b in observations:
   h=g[(g.time_s>=a)&(g.time_s<b)].copy();h['bin']=np.floor((h.time_s-a)/.25).astype(int)
   s=h.groupby('bin').waveform_centroid_um.median()
   for lag in diffs:
    k=round(lag/.25)
    diffs[lag].extend(float(s.loc[i+k]-v) for i,v in s.items() if i+k in s.index)
  for lag,values in diffs.items():
   rows.append(dict(source=rel,candidate=ident,window='p10',lag_s=lag,n_strict_events=len(g),n_pairs=len(values),mean_increment_um=np.mean(values) if values else np.nan,sd_increment_um=np.std(values,ddof=0) if len(values)>=2 else np.nan))
write('lighthouse_increment_sd.csv',rows)
write('source_manifest.csv',manifest)
print(json.dumps(identity,indent=2))
print('finished',len(manifest),'original files copied')
