"""Full-probe footprint and alignment audit of existing rejected cohorts."""
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
from testing.luke_epoch_corroboration import ROOT,BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_candidate_footprint_audit_v1';SRC=ROOT/'testing/outputs/luke_independent_transfer_review_v1'
def main():
 OUT.mkdir(exist_ok=False);m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];loc=np.asarray(m['channel_locations_um']);d=pd.read_csv(SRC/'candidate_screen.csv');events=np.load(SRC/'candidate_events.npz');off=np.arange(-45,46);rows=[];saved={}
 (OUT/'settings.json').write_text(json.dumps(dict(scope='Diagnostic of unchanged rejected cohorts, no promotion or exclusions',alignment_samples=[-3,3],repeatable_energy='Dot product of independent half median waveforms; signed cross-energy, diagnostic not an acceptance threshold',random_background='One deterministic independently sampled null cohort per half with equal event count; descriptive reference only'),indent=2))
 rng=np.random.default_rng(20260907)
 for start in [960,9510]:
  first=round(start*fs);n=round(10*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*m['gain_uv_per_count'];x=sosfiltfilt(butter(3,[300,6000],fs=fs,btype='bandpass',output='sos'),x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+n];del buf
  for r in d[d.start_s==start].itertuples():
   all_ev=events[r.candidate]-first;ev=[all_ev[(all_ev>=h*5*fs)&(all_ev<(h+1)*5*fs)] for h in [0,1]];ev=[e[np.linspace(0,len(e)-1,min(100,len(e)),dtype=int)] for e in ev];med=[np.median(x[e[:,None]+off],axis=0) for e in ev];null=[]
   for h,e in enumerate(ev):
    ne=rng.integers(round(h*5*fs)+50,round((h+1)*5*fs)-50,len(e));null.append(np.median(x[ne[:,None]+off],axis=0))
   local=abs(loc[:,1]-r.depth_um)<=60;far=abs(loc[:,1]-r.depth_um)>200;energy=(med[0]**2).sum(axis=0);cross=(med[0]*med[1]).sum(axis=0);nullenergy=(null[0]**2).sum(axis=0);far_ch=np.flatnonzero(far)[np.max(abs(med[0][:,far]),axis=0).argmax()];v=med[0][:,r.channel];best=np.full(len(ev[1]),-1.)
   for lag in range(-3,4):
    w=x[ev[1][:,None]+off+lag,r.channel];co=w@v/(np.linalg.norm(w,axis=1)*np.linalg.norm(v));best=np.maximum(best,co)
   rows.append(dict(candidate=r.candidate,depth_um=r.depth_um,local_energy_fraction=float(energy[local].sum()/energy.sum()),signed_cross_local_fraction=float(cross[local].sum()/cross.sum()),far_energy=float(energy[far].sum()),far_null_energy=float(nullenergy[far].sum()),far_cross_energy=float(cross[far].sum()),strongest_far_channel=int(far_ch),strongest_far_depth_um=float(loc[far_ch,1]),far_peak_uv=float(abs(med[0][:,far_ch]).max()),far_half_cosine=float(med[0][:,far_ch]@med[1][:,far_ch]/(np.linalg.norm(med[0][:,far_ch])*np.linalg.norm(med[1][:,far_ch]))),unaligned_event_fraction=r.event_shape_fraction,aligned_event_fraction=float((best>=.8).mean())))
   for h in [0,1]:saved[r.candidate+f'_h{h}']=med[h];saved[r.candidate+f'_null_h{h}']=null[h]
  print('Completed footprints',start,flush=True);del x
 summary=pd.DataFrame(rows);summary.to_csv(OUT/'summary.csv',index=False);np.savez_compressed(OUT/'full_probe_waveforms.npz',**saved)
 for page in range(3):
  group=d.iloc[page*4:page*4+4];fig,axes=plt.subplots(len(group),2,figsize=(12,3*len(group)),squeeze=False,layout='constrained')
  for ax,r in zip(axes,group.itertuples()):
   q=summary[summary.candidate==r.candidate].iloc[0];tt=off/fs*1000
   for h in [0,1]:
    w=saved[r.candidate+f'_h{h}'];ax[0].plot(np.max(abs(w),axis=0),loc[:,1],label=f'Half{h+1}');ax[1].plot(tt,w[:,int(q.strongest_far_channel)],label=f'Far half{h+1}')
   ax[0].axhspan(r.depth_um-60,r.depth_um+60,color='gray',alpha=.15);ax[0].set(title=r.candidate,xlabel='Median peak (µV)',ylabel='Depth (µm)',ylim=(0,3820));ax[0].legend(fontsize=7);ax[1].set(title=f'Strongest far channel {int(q.strongest_far_channel)} / {q.strongest_far_depth_um:.0f} µm',xlabel='Time (ms)',ylabel='µV');ax[1].legend(fontsize=7)
  fig.suptitle('Rejected cohorts: full-probe footprints and distant waveform repeatability',fontsize=12)
  for ext in ['png','pdf']:fig.savefig(OUT/f'01_footprints_{page+1}.{ext}',dpi=150)
 (OUT/'complete.json').write_text(json.dumps(dict(status='complete',cohorts=len(d)),indent=2));print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
