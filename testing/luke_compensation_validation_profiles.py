"""Supplement frozen-model validation with population alignment and event-level recovery."""
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from testing.luke_epoch_corroboration import BASE
from testing.luke_compensation_validation import OUT, START
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz']
    edges=np.arange(0,3841); depth=edges[:-1]+.5; shifts=np.arange(-40,41)
    rows=[]; fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for name in ['Original','Compensated']:
        peaks=np.load(OUT/f'{name.lower()}_peaks.npy'); y=np.load(OUT/f'{name.lower()}_locations.npy')['y']
        profiles=[]
        for half in [0,1]:
            k=(peaks['sample_index']>=half*10*fs)&(peaks['sample_index']<(half+1)*10*fs)
            profiles.append(gaussian_filter1d(np.histogram(y[k],edges,weights=abs(peaks['amplitude'][k]))[0],1.))
        for ax,(lo,hi) in zip(axes.flat,[(0,900),(400,1000),(1960,2560),(2260,2860)]):
            k=(depth>=lo)&(depth<hi)
            vals=np.array([np.corrcoef(profiles[0][k],np.interp(depth[k]+s,depth,profiles[1]))[0,1] for s in shifts])
            rows.append(dict(arm=name,depth_lo_um=lo,depth_hi_um=hi,best_shift_um=int(shifts[np.argmax(vals)]),peak_correlation=float(vals.max()),zero_correlation=float(vals[40])))
            ax.plot(shifts,vals,label=name,ls='--' if name=='Original' else '-')
            ax.axvline(0,c='gray',lw=.6); ax.set(title=f'Depth {lo}–{hi} µm',xlabel='Second-half − first-half shift (µm)',ylabel='Amplitude-weighted profile correlation'); ax.legend()
    fig.suptitle('Population alignment after frozen-model compensation, 4240–4260 s\nSame 1 µm bins and smoothing; diagnostic complements the DREDGE replay',fontsize=11)
    for suffix in ['png','pdf']: fig.savefig(OUT/f'03_population_alignment.{suffix}',dpi=160)
    pd.DataFrame(rows).to_csv(OUT/'alignment_objective.csv',index=False)
    rec=pd.read_csv(OUT/'event_recovery.csv').pivot(index=['unit_id','frame'],columns='arm',values='coincident')
    transitions=[]
    for cid,g in rec.groupby('unit_id'):
        transitions.append(dict(unit_id=int(cid),reference_events=len(g),both=int((g.Original&g.Compensated).sum()),original_only=int((g.Original&~g.Compensated).sum()),compensated_only=int((~g.Original&g.Compensated).sum()),neither=int((~g.Original&~g.Compensated).sum())))
    pd.DataFrame(transitions).to_csv(OUT/'event_transitions_by_unit.csv',index=False)
    print(json.dumps(dict(alignment=rows,event_transitions=transitions),indent=2))

if __name__=='__main__': main()
