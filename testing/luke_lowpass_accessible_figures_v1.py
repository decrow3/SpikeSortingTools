"""Re-render saved low-pass comparisons with redundant color/style encodings."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'testing/outputs'
OUT=SRC/'luke_lowpass_accessible_figures_v1'
ARMS=['broad_3sigma','broad_screen','lowpass_fixed','lowpass_fixed_screen','lowpass_adjusted','lowpass_adjusted_screen']
LABELS=['Broadband 3σ','Broadband 3σ + screen','3 kHz LP · fixed threshold','3 kHz LP · fixed + screen','3 kHz LP · noise-adjusted','3 kHz LP · adjusted + screen']
COLORS=['#0072B2','#0072B2','#D55E00','#D55E00','#882255','#882255']
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)
def main():
 OUT.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'semibold'})
 paths=[SRC/'luke_3sigma_lowpass_screen_v2/scores.csv',SRC/'luke_3sigma_lowpass_screen_v2/event_matched_predictions.csv',SRC/'luke_lighthouse_gentle_v1/gentle_tracks.csv']
 scores=pd.read_csv(paths[0]).set_index('name').loc[ARMS]
 fig,ax=plt.subplots(figsize=(10.5,5.6),layout='constrained');y=np.arange(6)
 for offset,col,color,hatch,label in [(-.19,'overall_mae_um','#0072B2',None,'Overall'),(.19,'movement_mae_um','#E69F00','///','Drop/recovery')]:
  bars=ax.barh(y+offset,scores[col],.35,color=color,hatch=hatch,edgecolor='#333333',linewidth=.5,label=label)
  ax.bar_label(bars,fmt='%.3f',padding=5,fontsize=10)
 ax.set(yticks=y,yticklabels=LABELS,xlim=(0,2.7),xlabel='Mean absolute disagreement with provisional lighthouses (µm)')
 ax.invert_yaxis();ax.legend(loc='upper right',frameon=False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 fig.suptitle('Panel A · Six conditioning arms\n4,160–4,260 s · Nine provisional lighthouse references',fontsize=14)
 fig.supxlabel('Disagreement is not physical motion error; reference identity and displacement sensitivity remain uncertain.',fontsize=10)
 save(fig,'01_panel_a_colorblind')
 predictions=pd.read_csv(paths[1]);tracks=pd.read_csv(paths[2])
 fig,axes=plt.subplots(3,3,figsize=(16,12),sharex=True);handles=[]
 for ax,(unit,g) in zip(axes.ravel(),tracks.groupby('unit_id')):
  g=g.sort_values('time_s');base=float(g.iloc[0].median_waveform_centroid_um)
  for i,(arm,label,color) in enumerate(zip(ARMS,LABELS,COLORS)):
   q=predictions[(predictions.name==arm)&(predictions.unit_id==unit)].sort_values('time_s')
   line,=ax.plot(q.time_s,q.predicted_um,color=color,ls='--' if i%2 else '-',marker='s' if i%2 else 'o',mfc='white' if i%2 else color,ms=3.5,lw=1.7,label=label)
   if len(handles)<6:handles.append(line)
  good=g.accepted_events>=10
  ref=ax.errorbar(g.time_s[good],g.median_waveform_centroid_um[good]-base,yerr=np.array([g.median_waveform_centroid_um[good]-g.bootstrap_low_um[good],g.bootstrap_high_um[good]-g.median_waveform_centroid_um[good]]),fmt='D',color='black',mfc='white',mew=1.3,ms=5,capsize=3,lw=1.3,zorder=10,label='Provisional lighthouse')
  ax.set(title=f'Unit {unit} · {g.depth_um.iloc[0]:.0f} µm',xlabel='Recording time (s)',ylabel='Relative displacement (µm)',xticks=[4160,4180,4200,4220,4240,4260]);ax.ticklabel_format(axis='x',style='plain',useOffset=False);ax.grid(alpha=.15)
 fig.suptitle('Six-arm lighthouse comparison · 4,160–4,260 s\nSame saved predictions, evaluated at accepted spike times',fontsize=16)
 fig.subplots_adjust(left=.065,right=.985,bottom=.19,top=.89,hspace=.42,wspace=.24)
 fig.legend(handles+[ref],LABELS+['Provisional lighthouse'],loc='lower center',bbox_to_anchor=(.5,.05),ncols=3,frameon=False,fontsize=11)
 fig.text(.5,.014,'Color: input family. Solid + filled circles: no screen. Dashed + open squares: screened.\nBlack diamonds: original reference estimates and bootstrap bars; not ground truth. Each panel has its own y scale.',fontsize=10,ha='center')
 save(fig,'02_lighthouse_overlay_colorblind')
 (OUT/'provenance.json').write_text(json.dumps({'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'changes':'Rendering only: explicit colors, line/marker redundancy, neutral titles. Original data and reference bars unchanged.','palette':{'broadband':'#0072B2','lowpass_fixed':'#D55E00','lowpass_adjusted':'#882255','reference':'#000000'}},indent=2))
if __name__=='__main__':main()
