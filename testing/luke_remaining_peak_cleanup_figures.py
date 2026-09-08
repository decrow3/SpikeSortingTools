"""Readable contact sheets for the remaining-peak cohort audit."""
from testing.luke_remaining_peak_cleanup import OUT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 d=pd.read_csv(OUT/'waveform_review.csv');w=np.load(OUT/'review_waveforms.npz')
 for page in range(3):
  fig,axs=plt.subplots(4,2,figsize=(12,10),layout='constrained')
  for ax,(_,r) in zip(axs,d.iloc[page*4:(page+1)*4].iterrows()):
   ch=int(r.channel);x=w[f'ch{ch}_waveforms'];v=x[:,:,ch];tt=np.arange(-30,31)/30
   ax[0].plot(tt,v.T,color='gray',alpha=.25,lw=.8);ax[0].plot(tt,np.median(v,axis=0),color='black',lw=2)
   ax[0].set(title=f'ch{ch} · {r.depth_um:.0f}µm · 24 individual events',ylabel='µV',xlabel='Time (ms)')
   ax[1].plot(tt,np.median(v[:12],axis=0),label='Earlier 12');ax[1].plot(tt,np.median(v[12:],axis=0),label='Later 12');ax[1].legend(fontsize=8)
   ax[1].set(title=f'Local footprint repeat cosine {r.local_half_repeat_cosine:.2f}\nMedian detector amplitude {r.median_peak_snr:.1f}σ',ylabel='µV',xlabel='Time (ms)')
  fig.suptitle('Remaining compensated peak populations · 930–1030s\nReview only: mixed shapes can contain multiple neurons; stationarity alone does not establish artifact')
  for ext in ['png','pdf']:fig.savefig(OUT/f'02_waveform_page{page+1}.{ext}',dpi=140)
if __name__=='__main__':main()
