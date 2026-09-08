"""Render all existing median lighthouse templates; no voltage access."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from testing.luke_lowpass_waveform_preservation_v2 import OUT, BASE


def main():
    d = pd.read_csv(OUT/'per_unit_bin_metrics.csv')
    d = d[d.support == 'original'].sort_values(['depth_um','time_s'])
    saved = np.load(OUT/'median_templates.npz')
    fs = json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz']
    units, times = d.unit_id.unique(), sorted(d.time_s.unique())
    fig, axes = plt.subplots(len(units),len(times),figsize=(14,18),sharex=True,sharey='row',layout='constrained')
    audit=[]
    for i,u in enumerate(units):
        broad=[saved[f'unit{u}_t{int(t)}_original_broad_uv'] for t in times]
        lowpass=[saved[f'unit{u}_t{int(t)}_original_lowpass_uv'] for t in times]
        channels=saved[f'unit{u}_t{int(times[0])}_original_channels']
        # One broad strongest channel per unit, fixed across four bins and filters.
        ch=int(np.argmax(np.max(np.abs(np.stack(broad)),axis=(0,1))))
        limit=max(np.max(np.abs(w[:,ch])) for w in broad+lowpass)*1.12
        for j,(t,wb,wl) in enumerate(zip(times,broad,lowpass)):
            ax=axes[i,j];row=d[(d.unit_id==u)&(d.time_s==t)].iloc[0]
            ax.plot(np.arange(-30,31)/fs*1000,wb[:,ch],color='#2468a2',lw=1.6,label='Compensated broadband')
            ax.plot(np.arange(-30,31)/fs*1000,wl[:,ch],color='#cc7722',ls='--',lw=1.6,label='+3 kHz low-pass')
            ax.axhline(0,color='#cccccc',lw=.5);ax.set_ylim(-limit,limit)
            ax.text(.02,.96,f'n={int(row.sampled_events)}   A={row.template_amplitude_ratio:.2f}   cos={row.template_cosine:.2f}',va='top',fontsize=8,transform=ax.transAxes)
            if i==0:ax.set_title(f'{int(t-5)}–{int(t+5)} s',fontsize=11)
            if j==0:ax.set_ylabel(f'Unit {u} · ch {channels[ch]}\n{row.depth_um:.0f} µm · voltage (µV)',fontsize=9)
            if i==len(units)-1:ax.set_xlabel('Time from accepted frame (ms)')
            ax.spines[['top','right']].set_visible(False)
            audit.append(dict(unit_id=int(u),time_s=float(t),plotted_channel=int(channels[ch]),amplitude_ratio=row.template_amplitude_ratio,cosine=row.template_cosine,sampled_events=int(row.sampled_events)))
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('Median waveforms at one fixed strongest broadband channel per lighthouse\nSame accepted frames; shared voltage scale within each row. A and cosine summarize all original support channels.',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(OUT/f'02_all_unit_waveform_contact_sheet.{ext}',dpi=150)
    pd.DataFrame(audit).to_csv(OUT/'contact_sheet_selection.csv',index=False)
    ranges=d.groupby('unit_id').template_centroid_shift_um.agg(lambda s:s.max()-s.min())
    print('Original within-unit maximum pairwise change bias (um)',ranges.to_dict())


if __name__=='__main__':main()
