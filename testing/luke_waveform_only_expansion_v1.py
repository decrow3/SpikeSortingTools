"""Training-only cached cohort expansion; preserve original strict decisions."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from testing.luke_waveform_only_global_v1 import OUT as SOURCE,SRC as SEEDS,BASE,ROOT
from testing.luke_waveform_only_global_review_v1 import near
from testing.luke_waveform_only_peak_overlay_v1 import segments
OUT=ROOT/'testing/outputs/luke_waveform_only_expansion_v1'

def main():
 OUT.mkdir(exist_ok=True);mf=BASE/'recording/rescue_recording_manifest.json';fs=json.loads(mf.read_text())['sampling_frequency_hz'];d=pd.read_csv(SOURCE/'seed_distinctness.csv');training=pd.concat([pd.read_csv(SOURCE/f'chunk_{t}.csv') for t in [930,935]]);bank=np.load(SEEDS/'templates.npz');prior=pd.read_csv(SOURCE/'training_qualified_candidates.csv');old=set(prior[prior.selected].unit_id);rows=[]
 for c in d.itertuples():
  seed=bank[f'unit_{c.unit_id}_seed_frames']/fs
  q=training[(training.unit_id==c.unit_id)&(training.score>=.8)&training.gain.between(.35,3)]
  confident=q[q.margin>=.025];n=int(near(seed,q.time_s.to_numpy()).sum());nc=int(near(seed,confident.time_s.to_numpy()).sum());chosen=bool(c.eligible and n>=5 and n/len(seed)>=.2)
  tier='original_strict' if c.unit_id in old else ('expanded_margin_pass' if nc>=5 and nc/len(seed)>=.2 else 'ambiguous_recovery')
  rows.append(dict(unit_id=c.unit_id,selected=chosen,cohort=tier,seed_events=len(seed),recovered_including_ambiguous=n,recovered_margin_pass=nc,recovery_fraction=n/len(seed),margin_pass_recovery_fraction=nc/len(seed),shape_eligible=c.eligible,nearest_rival_id=c.nearest_rival_id,nearest_rival_cosine=c.nearest_rival_cosine,seed_centroid_um=c.seed_centroid_um))
 candidates=pd.DataFrame(rows);candidates.to_csv(OUT/'candidate_audit.csv',index=False);selected=candidates[candidates.selected].sort_values('unit_id');ids=selected.unit_id.tolist();assert old.issubset(ids)
 config=dict(selection_time_s=[930,940],evaluation_time_s=[940,1030],selection='Retain prior seed shape eligibility (repeatability>=.85,SNR>=5,local energy fraction>=.35,far ratio<=.8,global rival cosine<.95); require>=5 and>=20% recovered seed times at cosine>=.80,gain.35–3,including ambiguity; temporal tolerance0.3ms. No depth quotas or motion/background agreement.',original_strict='Original status accepted: cosine>=.86,margin>=.025,gain.35–3. Preserved.',expanded='Score .80–.86 with margin>=.025: exploratory lower-score support, not promoted to strict accepted.',ambiguous='Cosine>=.80 but margin<.025; retained as uncertain, not certified identity.',cohort_sizes=selected.cohort.value_counts().to_dict(),selection_uses_heldout=False,extraction=False,limitation='17 candidate templates are not17 verified independent biological cells. Same original sorted template inventory, exact40um translated support; no DREDGE/depth prior/smoothing.')
 (OUT/'settings.json').write_text(json.dumps(config,indent=2));print('Frozen training-selected candidates:',ids,flush=True)
 # Held-out outputs are read only after the training selection is saved.
 e=pd.read_csv(SOURCE/'events.csv');e=e[e.unit_id.isin(ids)&(e.score>=.8)&e.gain.between(.35,3)].copy();e['evidence']=np.where(e.margin<.025,'identity_ambiguous',np.where(e.status=='accepted','strict_accepted','lower_score'));e.to_csv(OUT/'overlay_events.csv',index=False)
 bg={};paths=[mf,SOURCE/'events.csv',SOURCE/'seed_distinctness.csv',SOURCE/'training_qualified_candidates.csv',SEEDS/'templates.npz',Path(__file__)]+[SOURCE/f'chunk_{t}.csv' for t in [930,935]]
 for arm in ['original','compensated']:
  pf=ROOT/f'testing/outputs/luke_long_context_validation_v1/{arm}_peaks.npy';lf=pf.with_name(arm+'_locations.npy');p=np.load(pf);l=np.load(lf);t=930+p['sample_index']/fs;y=l['y'];assert ((t>=930)&(t<1030)).all();bg[arm]=(t,y);paths.extend([pf,lf])
 (OUT/'chart_contract.json').write_text(json.dumps(dict(question='Which additional depth-independent waveform candidates have usable temporal support?',surface='Standalone PDF/PNG, Matplotlib scatter with short strict trace segments',grain='All localized peaks and selected hypothesis events930–1030s',palette='Gray peaks,blue strict filled circles,blue open lower-score circles,orange ambiguous crosses',scales='Absolute depth, all displayed alternatives retained; per-unit y limits, identical across input arms',uncertainty='Tier stated on each page; no interpolation across depth-patch changes or gaps>2s'),indent=2))
 handles=[Line2D([],[],color='#0072B2',marker='o',lw=1,ms=4,label='Strict accepted (≥0.86)'),Line2D([],[],color='#0072B2',marker='o',mfc='none',lw=0,ms=4,label='Lower score (0.80–0.86)'),Line2D([],[],color='#D55E00',marker='x',lw=0,ms=4,label='Identity ambiguous (≥0.80)')]
 def draw(ax,arm,q,seed,small=False):
  vals=np.r_[q.centroid_um,seed];lo=max(0,vals.min()-80);hi=min(3840,vals.max()+80)
  if hi-lo<300:mid=(lo+hi)/2;lo=max(0,mid-150);hi=min(3840,mid+150)
  t,y=bg[arm];ix=np.isfinite(y)&(y>=lo)&(y<=hi);ax.scatter(t[ix],y[ix],s=.45 if small else .6,c='#666666',alpha=.2,linewidths=0,rasterized=True);ax.axvspan(930,940,color='gray',alpha=.09);ax.axhline(seed,color='#0072B2',ls=':',lw=.65,alpha=.6)
  a=q[q.evidence=='strict_accepted'];b=q[q.evidence=='lower_score'];u=q[q.evidence=='identity_ambiguous'];size=6 if small else 15
  ax.scatter(u.time_s,u.centroid_um,s=size,c='#D55E00',marker='x',linewidths=.45,alpha=.6,rasterized=True,zorder=3);ax.scatter(b.time_s,b.centroid_um,s=size,facecolors='none',edgecolors='#0072B2',linewidths=.6,rasterized=True,zorder=4);ax.add_collection(LineCollection(segments(a),colors='#0072B2',linewidths=.8,zorder=4));ax.scatter(a.time_s,a.centroid_um,s=size,c='#0072B2',linewidths=0,rasterized=True,zorder=5)
  ax.set(xlim=(930,1030),ylim=(lo,hi),xlabel='Time (s)',ylabel='Depth (µm)');ax.ticklabel_format(axis='x',style='plain',useOffset=False)
 summaries=[]
 with PdfPages(OUT/'02_all_17_candidate_overlays.pdf') as pdf:
  for c in selected.itertuples():
   q=e[e.unit_id==c.unit_id];counts=q[q.time_s>=940].evidence.value_counts();summaries.append(dict(unit_id=c.unit_id,cohort=c.cohort,heldout_strict=int(counts.get('strict_accepted',0)),heldout_lower_score=int(counts.get('lower_score',0)),heldout_ambiguous=int(counts.get('identity_ambiguous',0))))
   fig,axs=plt.subplots(1,2,figsize=(17,8),sharey=True,layout='constrained')
   for ax,arm in zip(axs,bg):draw(ax,arm,q,c.seed_centroid_um);ax.set_title(f'{arm.capitalize()} peak background')
   axs[0].legend(handles=handles,fontsize=8,loc='best');fig.suptitle(f'Unit {c.unit_id} · {c.cohort.replace("_"," ")} · 930–1,030 s\nTraining recovery {c.recovered_including_ambiguous}/{c.seed_events} including ambiguity; {c.recovered_margin_pass}/{c.seed_events} with identity margin',fontsize=15)
   fig.supxlabel(f'Held out: {counts.get("strict_accepted",0)} strict · {counts.get("lower_score",0)} lower-score · {counts.get("identity_ambiguous",0)} ambiguous. Same original-waveform matches on both backgrounds.\nShading:training. Strict trace segments only within one patch and gaps≤2s. Candidate identities remain provisional; lower-score evidence is not strict acceptance.',fontsize=10)
   pdf.savefig(fig);fig.savefig(OUT/f'unit_{c.unit_id}.png',dpi=140);plt.close(fig)
  print('Rendered individual pages',flush=True)
 # Six candidates/page overview; avoid17 overplotted colors.
 with PdfPages(OUT/'01_candidate_overview.pdf') as pdf:
  for k in range(0,len(selected),6):
   batch=selected.iloc[k:k+6];fig,axs=plt.subplots(3,2,figsize=(15,12),layout='constrained')
   for ax,c in zip(axs.flat,batch.itertuples()):draw(ax,'original',e[e.unit_id==c.unit_id],c.seed_centroid_um,True);ax.set_title(f'Unit {c.unit_id} · {c.cohort.replace("_"," ")}',fontsize=11)
   for ax in list(axs.flat)[len(batch):]:ax.axis('off')
   fig.suptitle(f'Expanded waveform-only candidates · original peak scatter · page {k//6+1}/3',fontsize=14);fig.supxlabel('Individual depth scales. Filled blue:strict; open blue:lower score; orange:ambiguous. Shaded930–940s was used for selection.\nAll247 cached identities competed without absolute-depth or continuity priors. This is17 candidate templates, not17 verified independent cells.',fontsize=10);pdf.savefig(fig);fig.savefig(OUT/f'overview_{k//6+1}.png',dpi=150);plt.close(fig)
 summary=pd.DataFrame(summaries);summary.to_csv(OUT/'heldout_counts.csv',index=False)
 (OUT/'README.md').write_text('''# Expanded waveform-only lighthouse candidates\n\n17 candidate templates (12 additions):5 original strict,8 added with margin-passing training recovery at cosine0.80,4 added only when identity-ambiguous training recovery is included. Candidate selection used only930–940s and unchanged seed waveform quality/locality/global-distinctness gates; no absolute-depth preference, motion estimate, peak-background agreement or held-out coverage entered selection.\n\nOpen01_candidate_overview.pdf (3 pages) or02_all_17_candidate_overlays.pdf (17 pages, original/compensated peak backgrounds). Blue filled circles preserve original strict acceptance (cosine≥0.86, identity margin≥0.025); blue open circles are exploratory lower-score evidence (0.80–0.86, margin≥0.025); orange crosses are identity ambiguous (cosine≥0.80, margin<0.025). Gain0.35–3 applies throughout. Original decisions and reports remain unchanged.\n\nThe expanded cohort requires≥5 and≥20% recovered cached seed spikes including ambiguity. Eight additions also meet that recovery rule without ambiguity; four require ambiguous matches. This expansion relaxes the evidence needed to enter the candidate plot, not the definition of a strict match.17 templates are not certified independent biological cells. Unknown waveform lookalikes, overlapping spikes, inherited sort labels, incomplete edge support and exact40µm sampling remain limitations. Positive appearance on the background does not independently establish identity.\n\nAll rendered peak events are cached localized peaks, without subsampling. Identical original-waveform centroids appear on both background inputs. Dotted line:seed centroid. Lines connect strict observations only within one patch with gaps≤2s; no smoothed or interpolated path. Each unit has an individual depth scale including its ambiguous alternatives. Candidate audit and heldout counts retain empty/low-support outcomes. No recording extraction was needed.\n''')
 (OUT/'manifest.json').write_text(json.dumps(dict(status='complete',candidate_count=len(ids),new_candidates=len(set(ids)-old),tier_counts=selected.cohort.value_counts().to_dict(),heldout_totals=summary[['heldout_strict','heldout_lower_score','heldout_ambiguous']].sum().to_dict(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}),indent=2));print(summary.to_string(index=False),flush=True)
if __name__=='__main__':main()
