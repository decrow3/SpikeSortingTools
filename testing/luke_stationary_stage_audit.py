"""Replay recorded conditioning provenance for the same stationary events."""
from testing.luke_epoch_corroboration import ROOT,BASE
import json,copy
import numpy as np,pandas as pd
import spikeinterface as si
from scipy.signal import butter,sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'testing/outputs/luke_stationary_stage_audit_v1'
def main():
 OUT.mkdir(exist_ok=False);manifest=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());prov=json.loads((BASE/'recording/provenance.json').read_text());fs=manifest['sampling_frequency_hz'];gain=manifest['gain_uv_per_count'];first=round(4180*fs);n=round(20*fs);pad=round(.1*fs)
 def absolute(d):
  if d['class'].endswith('SpikeGLXRecordingExtractor'):d['kwargs']['folder_path']=manifest['source_folder']
  if 'recording' in d['kwargs']:absolute(d['kwargs']['recording'])
 absolute(prov);chain=[prov]
 while 'recording' in chain[-1]['kwargs']:chain.append(chain[-1]['kwargs']['recording'])
 stages=list(zip(['acquisition','phase','blanked','interpolated'],reversed(chain)));(OUT/'resolved_provenance.json').write_text(json.dumps(prov,indent=2));events=pd.read_csv(ROOT/'testing/outputs/luke_stationary_voltage_audit_v1/matched_events.csv',dtype={'source':str});e=np.sort(events[events.source=='251_252'].frame.to_numpy())-first;e=e[np.linspace(0,len(e)-1,300,dtype=int)];off=np.arange(-90,91);tt=off/fs*1000;sos=butter(3,[300,6000],fs=fs,btype='bandpass',output='sos');payload={};rows=[];phase_snapshot=None;blank_snapshot=None
 # Cache voltage is the ground reference; reconstruct stages with recorded fill_value=9 counts and interpolation weights.
 with (BASE/'recording/traces_cached_seg0.raw').open('rb') as f:f.seek((first-pad)*768);buf=f.read((n+2*pad)*768)
 cache=np.frombuffer(buf,dtype='<i2').reshape(-1,384).copy();del buf
 for name,d in stages+[('cached',None)]:
  if d is None:a=cache
  else:
   r=si.load_extractor(d);assert r.get_num_channels()==384 and r.get_sampling_frequency()==fs;a=r.get_traces(start_frame=first-pad,end_frame=first+n+pad)
  stats=dict(stage=name,events=300,samples_above500uv_fraction=float(np.mean(abs(a.astype('float32')*gain)>500)))
  if name=='phase':phase_snapshot=a.copy()
  if name=='blanked':
   bad=abs(phase_snapshot.astype('float32'))>500/gain;stats.update(blanked_sample_fraction=float(bad.mean()),blanked_event_sample_fraction=float(bad[e[:,None]+off+pad].mean()));blank_snapshot=a.copy();del phase_snapshot
  if name=='interpolated':
   changed=np.any(a!=blank_snapshot,axis=0);stats['channels_changed_by_interpolation']=np.flatnonzero(changed).tolist();delta=a.astype('int32')-cache.astype('int32');stats['cache_fraction_different']=float(np.mean(delta!=0));stats['cache_abs_count_difference_p99']=float(np.quantile(abs(delta),.99));stats['cache_max_count_difference']=int(abs(delta).max());del delta,blank_snapshot
  voltage=a.astype('float32')*gain;snippet=voltage[e[:,None]+off+pad];medraw=np.median(snippet,axis=0);medraw-=np.median(medraw[:20],axis=0);payload[f'{name}_unfiltered']=medraw;del snippet
  filtered=sosfiltfilt(sos,voltage,axis=0).astype('float32')[pad:pad+n];del voltage;ref=np.median(filtered,axis=1);med=np.median(filtered[e[:,None]+off],axis=0);post=filtered-ref[:,None];medpost=np.median(post[e[:,None]+off],axis=0);refwave=np.median(ref[e[:,None]+off],axis=0);payload[f'{name}_filtered']=med;payload[f'{name}_postref']=medpost;payload[f'{name}_reference']=refwave;cos=(med.T@refwave)/(np.linalg.norm(med,axis=0)*np.linalg.norm(refwave)+1e-20);stats.update(reference_peak_uv=float(abs(refwave).max()),coherent_channels=int((cos>.9).sum()),ch252_pre_peak_uv=float(abs(med[:,252]).max()),ch252_post_peak_uv=float(abs(medpost[:,252]).max()),ch238_post_peak_uv=float(abs(medpost[:,238]).max()));rows.append(stats);print(json.dumps(stats),flush=True);del filtered,post
 np.savez_compressed(OUT/'stage_waveforms.npz',time_ms=tt,**payload);pd.DataFrame(rows).to_csv(OUT/'stage_metrics.csv',index=False)
 fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True);names=['acquisition','phase','blanked','interpolated','cached']
 for name in names:
  axes[0,0].plot(tt,np.median(payload[f'{name}_unfiltered'],axis=1),label=name);axes[0,1].plot(tt,payload[f'{name}_reference'],label=name);axes[0,2].plot(tt,payload[f'{name}_filtered'][:,252],label=name);axes[1,0].plot(tt,payload[f'{name}_unfiltered'][:,252],label=name);axes[1,1].plot(tt,payload[f'{name}_postref'][:,252],label=name);axes[1,2].plot(tt,payload[f'{name}_postref'][:,238],label=name)
 titles=['Unfiltered median across channels (baseline removed)','Global reference after common 300–6000 Hz filter','Ch252 filtered, before global reference','Ch252 unfiltered (baseline removed)','Ch252 after global reference','Ch238 after global reference']
 for ax,title in zip(axes.ravel(),titles):ax.set(title=title,xlabel='Time from aligned event (ms)',ylabel='µV');ax.legend(fontsize=7)
 fig.suptitle('Where does the stationary residual appear?\nSame 300 event times; recorded phase/blank/interpolation settings; filtering/reference applied identically per stage',fontsize=12);fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(OUT/'01_stage_comparison.png',dpi=160);fig.savefig(OUT/'01_stage_comparison.pdf');result=dict(status='complete',metrics=rows,limitations=['Acquisition AP stream already reflects acquisition hardware filtering/reference.','Conditioning replay chunk boundaries may differ from cached materialization; differences are measured explicitly.','Event-triggered templates are conditional averages, not proof of physical artifact origin.']);(OUT/'summary.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
