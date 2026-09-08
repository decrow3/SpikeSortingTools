"""Presentation QA: undefined fractions are missing, not 100 percent."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]/'testing/outputs/luke_lighthouse_gentle_v1'
def main():
 d=pd.read_csv(OUT/'identity_validation.csv');fig,axes=plt.subplots(1,2,figsize=(11,7),sharey=True);yy=np.arange(len(d));col=np.where(d.validation_pass,'#167a9a','#aaaaaa');recall=d.validation_recall.where(d.calibration_pass,np.nan);extra=d.validation_unassigned_fraction.where(d.validation_accepted>0,np.nan)
 axes[0].barh(yy,recall,color=col);axes[1].barh(yy,extra,color=col)
 for i,r in enumerate(d.itertuples()):
  if not r.calibration_pass:
   axes[0].text(.02,i,'No qualifying calibration threshold',fontsize=7,va='center');axes[1].text(.02,i,'No accepted events; fraction undefined',fontsize=7,va='center')
 axes[0].set_yticks(yy,[f'{r.unit_id} ({r.depth_um:.0f} µm)' for r in d.itertuples()]);axes[0].invert_yaxis();axes[0].axvline(.5,c='k',ls=':');axes[1].axvline(.1,c='k',ls=':');axes[0].set(xlim=(0,1),xlabel='Recall of provisional reference events');axes[1].set(xlim=(0,1),xlabel='Accepted events without reference match');fig.suptitle(f'Local identity controls: {d.validation_pass.sum()}/20 pass separate quiet validation\n4130–4150 s; thresholds frozen from 4110–4130 s; blue = pass both gates');fig.tight_layout();fig.savefig(OUT/'01_identity_controls.png',dpi=160,bbox_inches='tight');fig.savefig(OUT/'01_identity_controls.pdf',bbox_inches='tight')
 # Machine-readable undefined fractions should also remain explicit.
 d['validation_unassigned_fraction']=extra;d['validation_recall']=recall;d.to_csv(OUT/'identity_validation_reviewed.csv',index=False)
if __name__=='__main__':main()
