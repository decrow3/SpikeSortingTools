"""Exploratory pair selected after the all-cell direct scatter; cached data only."""
from testing.luke_lighthouse_direct_check_v1 import OUT
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 e=pd.read_csv(OUT/'accepted_measurements.csv');q=pd.read_csv(OUT/'matched_event_pairs.csv');q=q[q.pair=='388_398']
 fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained',sharex=True)
 for u,color,marker in [(388,'#0072B2','o'),(398,'#D55E00','x')]:
  z=e[(e.unit_id==u)&(e.time_s>=940)];axes[0].scatter(z.time_s,z.relative_um,s=20,c=color,marker=marker,alpha=.65,label=f'Candidate {u}')
 axes[0].legend();axes[0].set(ylabel='Relative centroid (µm)',title='All accepted events, with each cell’s930–940s template as its reference')
 axes[1].scatter(q.time_s,q.difference_um,s=20,c='#333333');axes[1].set(ylabel='388 − 398 (µm)',xlabel='Recording time (s)',title=f'{len(q)} reciprocal nearest event pairs, ≤0.25s apart · median |difference| {q.difference_um.abs().median():.1f} µm')
 for ax in axes:ax.axhline(0,color='gray',lw=.6);ax.set(xlim=(940,1030),ylim=(-150,100));ax.grid(alpha=.15);ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 fig.suptitle('Exploratory example: do candidates near2000/2040µm move together?',fontsize=14)
 fig.supxlabel('Selected after inspecting the all-cell scatter; not independent confirmation. No smoothing, binning, DREDGE, or fitted offsets.\nIndividual-event centroid and identity biases remain; these points do not establish calibrated physical motion.',fontsize=9)
 for ext in ['png','pdf']:fig.savefig(OUT/f'03_central_pair_example.{ext}',dpi=160)
 plt.close(fig)
if __name__=='__main__':main()
