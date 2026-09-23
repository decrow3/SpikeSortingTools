"""Build a descriptive review of sealed cross-dataset pilot results."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def review(root):
    config=json.loads((root/'config.json').read_text())
    motion=[]
    noise=[]
    for w in config['windows']:
        stage=root/w['id']
        if (stage/'fit/complete.json').exists():
            row=json.loads((stage/'fit/summary.json').read_text())
            channels=json.loads((stage/'extraction/channels.json').read_text())
            row['retained_channels']=len(channels['channel_ids'])
            motion.append(row)
        # Noise can be reviewed before extraction/fitting finishes; label it explicitly.
        path=stage/'extraction/noise_channel_samples.csv'
        if path.exists():
            frame=pd.read_csv(path)
            if len(frame['sample'].unique()) != 12:
                continue
            perchannel=frame.groupby(['reference','channel']).sigma_uv.median().reset_index()
            for ref, group in perchannel.groupby('reference'):
                noise.append(dict(window=w['id'],dataset=w['dataset'],probe=w['probe'],fraction=w['fraction'],
                    reference=ref,median_sigma_uv=group.sigma_uv.median(),p10_channel_sigma_uv=group.sigma_uv.quantile(.1),
                    p90_channel_sigma_uv=group.sigma_uv.quantile(.9)))
    m=pd.DataFrame(motion)
    n=pd.DataFrame(noise)
    if len(n):
        n.to_csv(root/'review_noise_windows.csv',index=False)
        groups=[(s['dataset'],s['probe']) for s in config['records']]
        fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
        for ax,ref in zip(axes,['none','shank_median','local_100um_exclude_self']):
            for i,(dataset,probe) in enumerate(groups):
                subset=n[(n.dataset==dataset)&(n.probe==probe)&(n.reference==ref)]
                ax.scatter(i+(subset.fraction-.5)*.3,subset.median_sigma_uv,s=35)
            ax.set_xticks(range(6),[a+'\n'+b for a,b in groups],fontsize=8)
            ax.set(title=ref.replace('_',' '),ylabel='Median channel robust σ (µV)',ylim=(0,None))
            ax.grid(axis='y',alpha=.25)
        fig.suptitle(f'AP voltage variability: {len(noise)//3}/18 windows; 300–6000 Hz, same samples across references')
        fig.savefig(root/'noise_comparison.png',dpi=160)
        plt.close(fig)
    text=['# Fast MEDiCINe: Bacon, Luke and Allen', '',
          f'{len(motion)}/{len(config["windows"])} motion fits are sealed. Noise is available from {len(noise)//3} windows.' + (' Partial results must not be treated as a balanced comparison.' if len(motion)<len(config['windows']) else ''), '',
          'The question is whether these sessions differ in motion magnitude, speed, depth dependence and noise under a consistent fast estimator. Luke having the largest/fastest motion is a hypothesis, not an input to selection.', '',
          'Each recording has three prespecified 120 s windows centered at 10%, 50% and 90% of its duration. The six probe/shank streams belong to three sessions; they are not six independent preparations. Relative session position does not match behavior.', '']
    if len(m):
        m.to_csv(root/'review_motion_windows.csv',index=False)
        text += ['| Dataset/probe | Completed windows | Median depth excursion (µm) | P99 local speed (µm/s) | P95 dynamic depth spread (µm) |', '|---|---:|---:|---:|---:|']
        for (dataset,probe),g in m.groupby(['dataset','probe'],sort=False):
            vals=[g[c].median() for c in ['median_depth_excursion_p95_p5_um','p99_local_speed_um_s','p95_nonrigid_spread_um']]
            text.append(f'| {dataset} {probe} | {len(g)} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} |')
        text += ['', 'Entries are medians of the available window-level measurements. Inspect individual points, peak support and retained channels before using these summaries. No session-level significance or biological causality is inferred.', '']
    if len(n):
        text += ['| Dataset/probe | No reference (µV) | Shank median (µV) | Local ≤100 µm, excluding self (µV) |', '|---|---:|---:|---:|']
        for (dataset,probe),g in n.groupby(['dataset','probe'],sort=False):
            a=g.groupby('reference').median_sigma_uv.median()
            text.append(f'| {dataset} {probe} | {a["none"]:.2f} | {a["shank_median"]:.2f} | {a["local_100um_exclude_self"]:.2f} |')
        text += ['', 'Noise entries summarize per-channel MAD/0.67449 across twelve 1 s samples in each window, then channels and available windows. Spikes remain in this voltage-variability measure. Local neighborhoods contain different channel counts on NP and Nandy probes. Allen shanks are referenced separately.', '']
    text += ['## Reading the evidence', '',
        '- MEDiCINe: 0.25 s bins, 1 s triangular kernel, four depth bins, bound 500 µm, 10000 Adam steps and seed 0; DARTsort native initial detection only, with per-window denoiser fitting.',
        '- Frontend recorded for this run: '+config['frontend']+'.',
        '- Speed is the finite difference of a smoothed model field at 250 ms spacing. Four samples per second is not 4 Hz physical-motion bandwidth.',
        '- Excursion is P95−P5 through time. Dynamic depth spread is the across-depth P95−P5 after removing constant offsets independently at each depth. Opposing movement can be large even if the rigid median is small.',
        '- The NP and Nandy depth spans differ. Their whole-shank differential spread does not measure deformation over equal physical distances.',
        '- Greater estimated motion alone does not prove greater accuracy. Peak-depth rasters, per-depth field traces, localization support, channel exclusions and loss traces remain the direct checks. These fits have no new independent cell-identity validation.',
        '- Windows do not measure full-session drift and can miss rare large events. All windows were selected before inspecting the new estimates.',
        '- Raw voltage is not motion-corrected and no spike sorting is performed. Original full-session estimates remain separate historical evidence.', '',
        '## Files', '', '[Motion comparison](comparison.png) · [Noise comparison](noise_comparison.png) · [Motion rows](review_motion_windows.csv) · [Noise rows](review_noise_windows.csv) · [Settings](config.json)', '']
    (root/'REVIEW.md').write_text('\n'.join(text))
    print(root/'REVIEW.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    review(p.parse_args().output.resolve())
