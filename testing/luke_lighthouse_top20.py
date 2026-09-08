"""Bounded lighthouse signal screen. No motion agreement enters selection."""
from testing.luke_epoch_corroboration import BASE,ROOT
import json
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_lighthouse_top20_v1'
def main():
 OUT.mkdir(exist_ok=False)
 settings=dict(intervals_s=[[4080,4090],[4100,4110]],band_hz=[300,6000],filter_order=3,padding_s=.05,reference='global median after bandpass',min_peak_uv=150,min_peak_noise_ratio=10,min_events_per_interval=15,min_average_cosine=.9,min_local_energy_fraction=.7,min_event_cosine=.8,min_event_fraction=.8,scope='Signal screening only; no validated identity or displacement claim. Previously inspected quiet intervals, not a fresh identity holdout.')
 (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
 rec=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=rec['sampling_frequency_hz'];loc=np.asarray(rec['channel_locations_um']);sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos')
 cur=BASE/'cur/cur_output';st=np.load(cur/'spike_times.npy',mmap_mode='r').ravel();cl=np.load(cur/'spike_clusters.npy',mmap_mode='r').ravel();labels=pd.read_csv(cur/'cluster_KSLabel.tsv',sep='\t').set_index('cluster_id').KSLabel.to_dict()
 waves={};noise=[];offs=np.arange(-45,46);rng=np.random.default_rng(20260907);background=[]
 for k,(start,end) in enumerate(settings['intervals_s']):
  first=round(start*fs);n=round((end-start)*fs);pad=round(.05*fs)
  with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:
   f.seek((first-pad)*384*2);buf=f.read((n+2*pad)*384*2)
  assert len(buf)==(n+2*pad)*384*2
  x=np.frombuffer(buf,dtype='<i2').reshape(-1,384).astype('float32')*rec['gain_uv_per_count'];x=sosfiltfilt(sos,x,axis=0).astype('float32');x-=np.median(x,axis=1,keepdims=True);x=x[pad:pad+n]
  noise.append(np.median(np.abs(x-np.median(x,axis=0)),axis=0)/.67448975)
  idx=np.flatnonzero((st>=first+46)&(st<first+n-46));times=np.asarray(st[idx])-first;ids=np.asarray(cl[idx]);background.append(x[rng.integers(46,n-46,30)[:,None]+offs].copy())
  for cid in np.unique(ids):
   ev=times[ids==cid];total=len(ev);ev=ev[np.linspace(0,total-1,min(total,150),dtype=int)]
   waves[(int(cid),k)]=(x[ev[:,None]+offs].copy(),total)
  print(f'Loaded conditioned voltage and events {start}–{end}',flush=True)
  del x,buf
 rows=[];payload={}
 for cid in sorted(set(c for c,k in waves)):
  if any((cid,k) not in waves for k in [0,1]):continue
  a,na=waves[cid,0];b,nb=waves[cid,1]
  if min(na,nb)<15:continue
  wa=np.median(a,axis=0);wb=np.median(b,axis=0);peak=int(np.max(np.abs(wa),axis=0).argmax());depth=loc[peak,1];channels=np.flatnonzero(abs(loc[:,1]-depth)<=60);v=wa[:,channels];v2=wb[:,channels];amp=float(abs(wa[:,peak]).max());ratio=amp/noise[0][peak];cos=float(np.sum(v*v2)/(np.linalg.norm(v)*np.linalg.norm(v2)+1e-20));compact=float(np.sum(v*v)/np.sum(wa*wa));energy=np.sum(wa*wa,axis=0);far=(abs(loc[:,1]-depth)>200);farmax=float(np.max(abs(wa[:,far]))/amp) if far.any() else 0
  # Single-event shape on the strongest channel: deliberately simple, visible in figures.
  ref=wa[:,peak];ec=b[:,:,peak]@ref/(np.linalg.norm(b[:,:,peak],axis=1)*np.linalg.norm(ref)+1e-20);event_fraction=float(np.mean(ec>=.8));event_snr=np.max(abs(b[:,:,peak]),axis=1)/noise[1][peak]
  reasons=[]
  for ok,name in [(amp>=150,'amplitude'),(ratio>=10,'SNR'),(cos>=.9,'repeatability'),(compact>=.7,'localization'),(event_fraction>=.8,'event_shape')]:
   if not ok:reasons.append(name)
  row=dict(unit_id=cid,KSLabel=labels.get(cid,'unknown'),depth_um=float(depth),peak_channel=peak,peak_uv=amp,noise_uv=float(noise[0][peak]),peak_noise_ratio=float(ratio),events_screen=na,events_repeat=nb,average_cosine=cos,local_energy_fraction=compact,far_peak_ratio=farmax,repeat_event_shape_fraction=event_fraction,repeat_median_event_snr=float(np.median(event_snr)),screen_pass=not reasons,reasons=';'.join(reasons))
  rows.append(row);payload[cid]=(a,b,wa,wb,peak,channels,ec)
 df=pd.DataFrame(rows);df.to_csv(OUT/'candidate_screen.csv',index=False)
 # Exclude first/last 120 µm; preserve strict screen and distinguish near misses.
 pool=df[(df.depth_um>=120)&(df.depth_um<=3700)&(df.average_cosine>=.9)&(df.repeat_event_shape_fraction>=.8)&(df.peak_noise_ratio>=10)]
 ranked=pool.sort_values(['screen_pass','peak_uv'],ascending=[False,False]).head(20).copy()
 assert len(ranked)==20
 ranked['rank']=np.arange(1,21)
 ranked.to_csv(OUT/'top20_ranked.csv',index=False)
 selected=ranked.unit_id.astype(int).tolist()
 (OUT/'ranking.json').write_text(json.dumps(dict(edge_margin_um=120,eligibility='At least 15 events each interval, local SNR >=10, average cosine >=0.9, repeat event shape fraction >=0.8',order='Original full-screen passes first; amplitude descending within pass/near-miss groups',caveat='Not 20 validated cells. Original amplitude and localization failures retained and marked; possible duplicate or mixed clusters remain.'),indent=2))
 df['contact_sheet']=df.unit_id.isin(selected);df.to_csv(OUT/'candidate_screen.csv',index=False)
 def save(fig,name):
  fig.savefig(OUT/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(10,7),sharey=True)
 for passed,col,label in [(False,'#aaaaaa','Below signal screen'),(True,'#167a9a','Pass signal screen')]:
  g=df[df.screen_pass==passed];ax[0].scatter(g.peak_uv,g.depth_um,c=col,s=22,label=f'{label} (n={len(g)})');ax[1].scatter(g.peak_noise_ratio,g.depth_um,c=col,s=22)
 for cid in selected:
  r=df.set_index('unit_id').loc[cid];ax[0].annotate(str(cid),(r.peak_uv,r.depth_um),xytext=(4,2),textcoords='offset points',fontsize=8)
 ax[0].axvline(150,c='k',ls=':',lw=1);ax[1].axvline(10,c='k',ls=':',lw=1);ax[0].set_xlabel('Median waveform peak (µV)');ax[1].set_xlabel('Peak / local robust noise σ');ax[0].set_ylabel('Probe depth (µm)');ax[0].set_ylim(0,3840);ax[0].legend(fontsize=8);fig.suptitle(f'Lighthouse signal screen: {df.screen_pass.sum()} / {len(df)} eligible clusters pass\n4080–4090 s; repeatability at 4100–4110 s; morphology still requires review');fig.tight_layout();save(fig,'01_probe_screen')
 for page in range((len(selected)+2)//3):
  subset=selected[page*3:page*3+3];fig,axes=plt.subplots(len(subset),3,figsize=(14,3.6*len(subset)),squeeze=False)
  for axs,cid in zip(axes,subset):
   a,b,wa,wb,peak,channels,ec=payload[cid];r=df.set_index('unit_id').loc[cid];tt=offs/fs*1000
   for w in b[np.linspace(0,len(b)-1,min(30,len(b)),dtype=int)]:axs[0].plot(tt,w[:,peak],c='#7899ae',alpha=.2,lw=.6)
   axs[0].fill_between(tt,-noise[1][peak]*3,noise[1][peak]*3,color='gray',alpha=.25,label='±3 noise σ');axs[0].plot(tt,wa[:,peak],c='k',label='Screen median');axs[0].plot(tt,wb[:,peak],c='#b65b24',ls='--',label='Repeat median');axs[0].set_title(f'Unit {cid} | {r.depth_um:.0f} µm | {r.peak_uv:.0f} µV\nSNR {r.peak_noise_ratio:.1f}; {r.events_screen}/{r.events_repeat} events');axs[0].set_xlabel('Time from sorted event (ms)');axs[0].set_ylabel('Filtered voltage (µV)');axs[0].legend(fontsize=7)
   im=axs[1].imshow(wb[:,channels].T,aspect='auto',origin='lower',extent=[tt[0],tt[-1],0,len(channels)],cmap='RdBu_r',vmin=-r.peak_uv,vmax=r.peak_uv);axs[1].set_yticks(np.arange(len(channels))+.5);axs[1].set_yticklabels([f'{c}: {loc[c,1]:.0f}' for c in channels],fontsize=6);axs[1].set_ylabel('Channel: depth (µm)');axs[1].set_xlabel('Time (ms)');axs[1].set_title(f'Repeat spatial waveform\nLocal energy {r.local_energy_fraction:.0%}');fig.colorbar(im,ax=axs[1],label='µV',fraction=.035)
   axs[2].plot(np.max(abs(wa),axis=0),loc[:,1],c='k',label='Screen');axs[2].plot(np.max(abs(wb),axis=0),loc[:,1],c='#b65b24',ls='--',label='Repeat');axs[2].set_ylim(0,3840);axs[2].set_xlabel('Median waveform peak (µV)');axs[2].set_ylabel('Probe depth (µm)');axs[2].set_title(f'Full-probe footprint | {"PASS screen" if r.screen_pass else "Below screen"}\nRepeat events with shape cosine ≥0.8: {r.repeat_event_shape_fraction:.0%}');axs[2].legend(fontsize=7)
  fig.suptitle('Candidate neural waveform review — signal screen is not identity validation\n300–6000 Hz + median reference; individual repeat events shown without shape selection',fontsize=13);fig.tight_layout(rect=[0,0,1,.94]);save(fig,f'02_contact_sheet_{page+1}')
 # Compact waveform gallery, same y scale throughout for honest amplitude comparison.
 from matplotlib.backends.backend_pdf import PdfPages
 with PdfPages(OUT/'top20_waveform_gallery.pdf') as pdf:
  for page in range(2):
   fig,axes=plt.subplots(2,5,figsize=(18,7),sharex=True,sharey=True)
   for ax,cid in zip(axes.ravel(),selected[page*10:page*10+10]):
    a,b,wa,wb,peak,channels,ec=payload[cid];r=ranked.set_index('unit_id').loc[cid];tt=offs/fs*1000
    for w in b[np.linspace(0,len(b)-1,min(25,len(b)),dtype=int)]:ax.plot(tt,w[:,peak],color='#7899ae',alpha=.22,lw=.6)
    ax.fill_between(tt,-3*noise[1][peak],3*noise[1][peak],color='gray',alpha=.2)
    ax.plot(tt,wa[:,peak],color='k',lw=1.3);ax.plot(tt,wb[:,peak],color='#b65b24',ls='--',lw=1.3)
    status='Screen pass' if r.screen_pass else 'Below: '+r.reasons
    ax.set_title(f'#{int(r["rank"])} · Unit {cid} · {r.KSLabel}\n{r.depth_um:.0f} µm | {r.peak_uv:.0f} µV | SNR {r.peak_noise_ratio:.1f}\n{status}',fontsize=9)
    ax.set_ylim(-360,360);ax.set_xlim(-1.5,1.5);ax.axhline(0,c='gray',lw=.4);ax.set_xlabel('Time (ms)')
   axes[0,0].set_ylabel('Filtered voltage (µV)');axes[1,0].set_ylabel('Filtered voltage (µV)')
   fig.suptitle(f'Top 20 lighthouse candidates — ranks {page*10+1}–{page*10+10}\nBlack: screen median; orange dashed: repeat median; faint: individual spikes; gray: ±3 noise σ\n120 µm edge exclusion; full-screen passes first, then amplitude. Candidates are not validated identities.',fontsize=12)
   fig.tight_layout(rect=[0,0,1,.87]);pdf.savefig(fig,bbox_inches='tight');save(fig,f'03_top20_gallery_{page+1}')
 summary=dict(status='screen_complete_morphology_review_required',eligible_clusters=len(df),signal_screen_pass=int(df.screen_pass.sum()),pass_unit_ids=df.loc[df.screen_pass,'unit_id'].tolist(),contact_sheet_ids=selected,raw_seconds_read=20,limitations=['Existing sorted identities supply event times; this does not establish independent detection precision.','Quiet intervals previously inspected; not a fresh holdout.','Signal thresholds are explicit screening choices, not biological proof.','No motion trajectory or amplitude completeness measured.'])
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
