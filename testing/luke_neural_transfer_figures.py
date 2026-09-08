"""Review median waveform morphology for every passing transfer candidate."""
import numpy as np,pandas as pd,json
from testing.luke_neural_transfer_validation import OUT
from testing.luke_epoch_corroboration import BASE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];d=pd.read_csv(OUT/'candidate_screen.csv');d=d[d.screen_pass];z=np.load(OUT/'selected_waveforms.npz');fig,axes=plt.subplots(len(d),2,figsize=(11,3*len(d)),squeeze=False,layout='constrained')
for ax,r in zip(axes,d.itertuples()):
 for half in [0,1]:
  key=f's{r.start_s}_u{r.unit_id}_h{half}';ch=z[key+'_channels'];a=z[key+'_before'];b=z[key+'_after'];peak=np.flatnonzero(ch==r.peak_channel)[0];tt=np.arange(-45,46)/fs*1000
  ax[0].plot(tt,a[:,peak],label=f'Original half{half+1}');ax[0].plot(tt,b[:,peak],ls='--',label=f'Compensated half{half+1}')
 ax[0].set(title=f'Unit {r.unit_id}, {r.depth_um:.0f} µm · {r.start_s} s',xlabel='Time (ms)',ylabel='µV');ax[0].legend(fontsize=7)
 im=ax[1].imshow(a.T,origin='lower',aspect='auto',extent=[tt[0],tt[-1],0,len(ch)],cmap='RdBu_r',vmin=-r.peak_uv,vmax=r.peak_uv);ax[1].set(title='Original second-half spatial waveform',xlabel='Time (ms)',ylabel='Local channel index');fig.colorbar(im,ax=ax[1])
fig.suptitle('All four passing transfer candidates: waveform morphology\nCandidates from original-voltage screen; identity remains provisional',fontsize=11)
for ext in ['png','pdf']:fig.savefig(OUT/f'03_waveforms.{ext}',dpi=150)
