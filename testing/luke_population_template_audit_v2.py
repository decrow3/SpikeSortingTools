"""Cached template locality audit; flags suggest review, not cell certification."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_population_depth_v2 import OUT,BASE

def main():
 m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());geo=np.array(m['channel_locations_um']);inv=pd.read_csv(OUT/'candidates.csv');z=np.load(OUT/'templates.npz');rows=[];fig,axes=plt.subplots(10,3,figsize=(16,24),sharex=True)
 for ax,r in zip(axes.flat,inv[inv.selected].sort_values('depth_um').itertuples()):
  w=z[f'unit_{r.unit_id}_full'].astype(float);en=(w*w).sum(axis=0);local=abs(geo[:,1]-r.depth_um)<=60;far=abs(geo[:,1]-r.depth_um)>200;fraction=float(en[local].sum()/en.sum());ratio=float(abs(w[:,far]).max()/abs(w).max());common=np.median(w,axis=1);coherence=float(np.sum(common**2)*w.shape[1]/np.sum(w*w));suspect=fraction<.35 or ratio>.8
  rows.append(dict(unit_id=r.unit_id,local_energy_fraction=fraction,far_peak_ratio=ratio,median_channel_energy_fraction=coherence,broad_shared_suspect=suspect,review_rule='local energy <0.35 OR far peak >0.8; heuristic excludes consensus until reviewed'))
  ax.imshow(w.T,origin='lower',aspect='auto',extent=[-30/m['sampling_frequency_hz']*1000,30/m['sampling_frequency_hz']*1000,geo[:,1].min(),geo[:,1].max()],cmap='RdBu_r',vmin=-abs(w).max(),vmax=abs(w).max());ax.axhline(r.depth_um,color='lime',lw=.6);ax.set_title(f'ID {r.unit_id} · {r.depth_um:.0f} µm · local {fraction:.2f} · far {ratio:.2f}'+(' · REVIEW' if suspect else ''),fontsize=8);ax.set_ylabel('Depth µm',fontsize=8)
 for ax in axes[-1]:ax.set_xlabel('Time from seed spike (ms)')
 fig.suptitle('Whole-probe seed waveform audit · 930–940 s templates\nEach panel normalized separately; green = template peak depth. Heuristic flags do not establish identity.',fontsize=13);fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(OUT/'01_template_full_probe_contact.png',dpi=150);fig.savefig(OUT/'01_template_full_probe_contact.pdf');plt.close(fig);pd.DataFrame(rows).to_csv(OUT/'template_quality_audit.csv',index=False)
 print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
