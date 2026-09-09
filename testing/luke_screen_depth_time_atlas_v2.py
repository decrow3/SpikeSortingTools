"""Legacy-view depth/time atlas:0.25s×10µm,magma,pooled99.5thpercentile; cached arrays only."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from testing.luke_screen_all_lighthouse_overlays_v1 import TITLES
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'testing/outputs';SWEEP=SRC/'luke_screen_sweep_v1';OUT=SRC/'luke_screen_depth_time_atlas_v2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 pp=SRC/'luke_long_lighthouse_motion_v1/compensated_peaks.npy';yp=pp.with_name('compensated_locations.npy');mp=SWEEP/'manifest.csv';manifest=pd.read_csv(mp)
 from testing.luke_epoch_corroboration import BASE
 meta=BASE/'recording/rescue_recording_manifest.json';fs=json.loads(meta.read_text())['sampling_frequency_hz']
 peaks=np.load(pp);loc=np.load(yp);assert len(peaks)==len(loc)==216721
 t=4160+peaks['sample_index']/fs;z=loc['y'];amp=abs(peaks['amplitude']);te=np.arange(4160,4260+.25,.25);de=np.arange(0,3841,10.)
 inside=(t>=te[0])&(t<=te[-1])&(z>=de[0])&(z<=de[-1])
 def histogram(k):
  return np.histogram2d(t[k],z[k],bins=[te,de])[0].T,np.histogram2d(t[k],z[k],bins=[te,de],weights=amp[k])[0].T
 base=histogram(np.ones(len(peaks),bool));cache={};pooled=[[np.log1p(base[col]).ravel()] for col in range(2)]
 for r in manifest.itertuples():
  keep=np.load(SWEEP/'masks'/f'{r.name}.npy');retained=histogram(keep);removed=histogram(~keep);cache[r.name]=(retained,removed)
  for col in range(2):pooled[col].extend([np.log1p(retained[col]).ravel(),np.log1p(removed[col]).ravel()])
 vmax=[float(np.quantile(np.concatenate(values),.995)) for values in pooled]
 del pooled
 rows=[];sources=[pp,yp,mp,meta,Path(__file__).resolve()];index=['# Screening peak depth/time atlas','','[Complete 21-page PDF](all_21_depth_time.pdf)','','Every page shows the same baseline, retained peaks, and removed peaks: count at left, absolute detected-amplitude sum at right. Localized depth; 0.25 s × 10 µm bins. Magma colors show log(1 + bin total), with identical limits across rows and variants for each metric. Colors saturate at the pooled 99.5th percentile across the baseline and all retained/removed histograms; one frozen scale per metric across all pages. No per-arm normalization. Off-probe localizations are counted explicitly in the caption. These are detected populations, not sorted units.','','| Page | Variant | Retained | Depth/time PNG | PDF | Nine-cell overlay |','|---|---|---:|---|---|---|']
 with PdfPages(OUT/'all_21_depth_time.pdf') as pdf:
  for page,r in enumerate(manifest.itertuples(),1):
   path=SWEEP/'masks'/f'{r.name}.npy';sources.append(path);keep=np.load(path);assert keep.dtype==bool and len(keep)==len(peaks) and keep.sum()==r.peaks
   assert hashlib.sha256(keep.tobytes()).hexdigest()==r.mask_sha256
   retained,removed=cache[r.name]
   for col in range(2):assert np.allclose(base[col],retained[col]+removed[col],rtol=1e-10,atol=1e-7)
   assert retained[0].sum()==np.count_nonzero(keep&inside)
   stem=f'{page:02d}_{r.name}'
   np.savez_compressed(OUT/f'{stem}_histograms.npz',time_edges=te,depth_edges=de,baseline_count=base[0],retained_count=retained[0],removed_count=removed[0],baseline_amplitude_mass=base[1],retained_amplitude_mass=retained[1],removed_amplitude_mass=removed[1])
   fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True,sharey=True,layout='constrained')
   for ri,(title,data) in enumerate([('Unscreened compensated 5σ baseline',base),('Retained',retained),('Removed',removed)]):
    for col in range(2):
     ax=axes[ri,col];im=ax.imshow(np.log1p(data[col]),origin='lower',aspect='auto',extent=[te[0],te[-1],de[0],de[-1]],interpolation='nearest',cmap='magma',vmin=0,vmax=vmax[col]);ax.set(title=title+' · '+('peak count' if col==0 else 'amplitude sum'),ylabel='Localized depth (µm)',xlabel='Recording time (s)');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
     fig.colorbar(im,ax=ax,label='log(1 + '+('count)' if col==0 else 'summed |peak amplitude| in µV)'))
   outside=int(np.count_nonzero(keep&~inside));rows.append(dict(name=r.name,retained=int(keep.sum()),plotted_retained=int(retained[0].sum()),outside_plot=outside))
   fig.suptitle(f'{page:02d}/21 · {TITLES[r.name]}\n{r.peaks:,} / {len(peaks):,} peaks retained ({r.retained_fraction:.1%}) · 4,160–4,260 s',fontsize=15)
   fig.supxlabel(f'0.25 s × 10 µm bins; shared colors saturate at pooled 99.5th percentile. {outside:,} retained events outside plotted time/depth bounds.\nColor represents detected population density or amplitude mass, not biological identity or motion accuracy.',fontsize=10)
   for ext in ['png','pdf']:fig.savefig(OUT/f'{stem}.{ext}',dpi=150)
   pdf.savefig(fig);plt.close(fig)
   index.append(f'| {page} | {TITLES[r.name]} | {r.retained_fraction:.1%} | [View]({stem}.png) | [PDF]({stem}.pdf) | [Motion](../luke_screen_all_lighthouse_overlays_v1/{stem}.png) |')
 pd.DataFrame(rows).to_csv(OUT/'coverage_audit.csv',index=False)
 (OUT/'README.md').write_text('\n'.join(index)+'\n')
 (OUT/'provenance.json').write_text(json.dumps(dict(source_sha256={str(p):sha(p) for p in sources},render_only=True,bin_s=.25,bin_um=10,color='magma',color_percentile=99.5,pooling='Baseline once plus each of21retained/removed pairs; common limits across pages',shared_log1p_vmax=vmax,partition_checks_pass=True),indent=2))
 print('Saved all21 histogram pages; partition/count/mask checks passed.',flush=True)
if __name__=='__main__':main()
