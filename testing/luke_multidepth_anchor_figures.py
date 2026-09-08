"""Export control failures explicitly; do not label rejected tracks as motion."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from testing.luke_multidepth_anchors_v2 import OUT

def main():
    local=pd.read_csv(OUT/'anchor_validation.csv');full=pd.read_csv(OUT/'review/full_search_validation.csv');g=local.merge(full,on=['unit_id','depth_um']);fig,axes=plt.subplots(1,2,figsize=(12,4.5));xx=np.arange(len(g));labels=[f'{r.unit_id}\n{r.depth_um:.0f} µm' for r in g.itertuples()]
    axes[0].bar(xx-.18,g.validation_unassigned_fraction*100,.36,label='Original channel only');axes[0].bar(xx+.18,g.full_search_unassigned_fraction*100,.36,label='Full ±280 µm search');axes[0].axhline(10,color='red',ls='--',label='Qualification limit');axes[0].set(xticks=xx,xticklabels=labels,ylabel='Accepted events unmatched to reference labels (%)',ylim=(0,100));axes[0].legend(fontsize=8)
    axes[1].bar(xx-.18,g.validation_recall*100,.36,label='Original channel only');axes[1].bar(xx+.18,g.full_search_recall*100,.36,label='Full ±280 µm search');axes[1].axhline(50,color='red',ls='--',label='Minimum recall');axes[1].set(xticks=xx,xticklabels=labels,ylabel='Reference-event recovery (%)',ylim=(0,105));axes[1].legend(fontsize=8)
    fig.suptitle('Four candidates recover reference events, but fail expanded-search specificity\nSeparate 4100–4110 s control; unmatched is an ambiguity flag, not proven false-positive truth');fig.tight_layout(rect=[0,0,1,.88]);fig.savefig(OUT/'review/02_search_specificity.png',dpi=180);fig.savefig(OUT/'review/02_search_specificity.pdf');plt.close(fig)
    matches=pd.read_csv(OUT/'matches.csv');fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True)
    for ax,row in zip(axes,g.itertuples()):
        d=matches[(matches.unit_id==row.unit_id)&~matches.ambiguous];ax.scatter(d.time_s+(d.frame/29999.835983263598-d.time_s),d.centroid_um,s=2,alpha=.35);ax.axhline(row.depth_um,color='black',lw=.7,ls='--');ax.set_ylabel('Candidate depth (µm)');ax.set_title(f'Cluster {row.unit_id} template — rejected as a reliable motion anchor');ax.set_ylim(row.depth_um-320,row.depth_um+320)
    axes[-1].set_xlabel('Seconds from recording frame zero');fig.suptitle('Candidate event matches occupy multiple depths\nThese are rejected candidate assignments, not validated neuron trajectories');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(OUT/'review/03_rejected_candidate_rasters.png',dpi=180);fig.savefig(OUT/'review/03_rejected_candidate_rasters.pdf');plt.close(fig)
    (OUT/'review/INTERPRETATION.json').write_text(json.dumps(dict(status='no_qualified_multidepth_motion_anchors',supersedes='Parent summary completed_with_qualified_anchors refers to original-channel controls only. Expanded-search validation rejects all four.',recalibration='No threshold in the saved full-search calibration candidates retained >=50% reference recall, >=10 detections, and <=5% unmatched/spatially incorrect assignments. Therefore no fresh-validation or further target run was launched.',next_requirement='Improve identity discrimination using richer multichannel waveform features and a competing-unit dictionary, or qualify a new local stable-period sorting cohort. Do not infer stationary or moving tissue from rejected candidates.'),indent=2)+'\n')
if __name__=='__main__':main()
