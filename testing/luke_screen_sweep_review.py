"""Integrity checks and concise mechanism figures for the screen sweep."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_screen_sweep import OUT,LONG,SRC,mask
from testing.luke_epoch_corroboration import BASE

def main():
 f=np.load(OUT/'features.npz');p=np.load(LONG/'compensated_peaks.npy');m=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text());fs=m['sampling_frequency_hz'];geo=np.asarray(m['channel_locations_um']);manifest=pd.read_csv(OUT/'manifest.csv');full=np.load(OUT/'masks/full_screen.npy');assert np.array_equal(mask(f),full)
 for _,r in manifest.iterrows():
  k=np.load(OUT/'masks'/f'{r["name"]}.npy');a=np.load(OUT/'fields'/f'{r["name"]}.npz');assert len(k)==len(p) and k.dtype==bool and k.sum()==r.peaks;assert np.isfinite(a['displacement_um']).all() and abs(a['D']).max()<=80
 group=(p['sample_index']/fs).astype(int)*40+(geo[p['channel_index'],1]//100).astype(int);strat=np.load(OUT/'masks/random_depth_time.npy');assert np.array_equal(np.bincount(group[full],minlength=4000),np.bincount(group[strat],minlength=4000))
 for seed in [14,29]:assert np.load(OUT/'masks'/f'random_count_seed{seed}.npy').sum()==full.sum()
 scores=pd.read_csv(OUT/'scores.csv').set_index('name');names=['compensated','full_screen','without_snr','without_neighbor','center_only','random_count_seed14','random_count_seed29','random_depth_time'];labels=['Compensation only','Full screen','Screen without amplitude cut','Screen without neighbor rule','Off-center rule only','Random same count (seed14)','Random same count (seed29)','Random same depth/time counts'];colors=['#2878b5','#b13775','#57a17b','#57a17b','#57a17b','gray','gray','gray']
 fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
 for ax,col,title in zip(axs,['overall_mae_um','movement_mae_um'],['Overall lighthouse difference','Drop + recovery difference']):
  ax.barh(np.arange(len(names)),scores.loc[names,col],color=colors);ax.set(yticks=np.arange(len(names)),yticklabels=labels,xlabel='Mean absolute difference (µm)',title=title);ax.invert_yaxis()
 fig.suptitle('Why the screen degrades registration: amplitude/neighbor selection and population thinning\nSame100s and DREDGE settings; differences from provisional lighthouse centroids, not ground-truth errors')
 for ext in ['png','pdf']:fig.savefig(OUT/f'02_key_controls.{ext}',dpi=150)
 constraints=[];fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained');bins=np.arange(0,85,5)
 for name,color in [('compensated','#2878b5'),('full_screen','#b13775'),('without_snr','#57a17b'),('random_depth_time','gray')]:
  a=np.load(OUT/'fields'/f'{name}.npz');t=a['time_s'];b=np.argmin(abs(a['depth_um']-2260));pairs=(t[:,None]>=4225)&(t[:,None]<4240)&(t[None,:]<4220);D=a['D'][b][pairs];C=a['C'][b][pairs];U=a['U'][b][pairs];active=U>0
  r=dict(name=name,depth_um=float(a['depth_um'][b]),active_pairs=int(active.sum()),large_displacement_weight_fraction=float(U[abs(D)>15].sum()/U.sum()),median_correlation=float(np.median(C[active])),median_abs_displacement_um=float(np.median(abs(D[active]))));constraints.append(r)
  hist=np.histogram(abs(D),bins,weights=U)[0]/U.sum();axs[0].stairs(hist,bins,label=name,color=color);axs[1].hist(C[active],bins=np.linspace(0,1,21),density=True,histtype='step',label=name,color=color)
 axs[0].set(xlabel='Absolute pairwise shift (µm)',ylabel='Fraction of constraint weight',title='Weight shifts toward large displacements');axs[1].set(xlabel='Pairwise correlation',ylabel='Density',title='Correlation strength does not separate the problem');axs[0].legend(fontsize=8);axs[1].legend(fontsize=8)
 fig.suptitle('Central window2210µm · pairs linking4225–4240s to times before4220s\nThis region was selected to diagnose the already-observed screened excursion, not to score the sweep')
 for ext in ['png','pdf']:fig.savefig(OUT/f'03_pairwise_mechanism.{ext}',dpi=150)
 pd.DataFrame(constraints).to_csv(OUT/'pairwise_mechanism.csv',index=False)
 d=pd.read_csv(OUT/'event_matched_predictions.csv');fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
 for ax,u in zip(axs,[445,463]):
  for name,color in [('compensated','#2878b5'),('full_screen','#b13775'),('without_snr','#57a17b'),('center_only','#d49a00')]:
   g=d[(d.name==name)&(d.unit_id==u)];ax.plot(g.time_s,g.predicted_um,'.-',label=name,color=color)
  g=d[(d.name=='compensated')&(d.unit_id==u)&(d.accepted_events>=10)];ax.plot(g.time_s,g.centroid_um,'ko',label='Lighthouse');ax.set(title=f'Unit{u}',xlabel='Recording time (s)',ylabel='Relative motion (µm)')
 axs[0].legend(fontsize=8);fig.suptitle('Selected central trajectories after screen ablation\nMotion summarized at the same accepted event times as the lighthouse centroids')
 for ext in ['png','pdf']:fig.savefig(OUT/f'04_central_trajectories.{ext}',dpi=150)
 result=dict(status='passed',checks=['All mask counts/index lengths and finite bounded fields','Full-mask feature reconstruction exact','Random controls preserve requested counts','Depth/time random control matches every1s×100um stratum'],pairwise_diagnostics=constraints);(OUT/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
