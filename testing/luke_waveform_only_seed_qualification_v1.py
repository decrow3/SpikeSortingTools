"""Choose observable distinctive seeds from training data only; freeze before review."""
import json
import numpy as np
import pandas as pd
from testing.luke_waveform_only_global_v1 import OUT,SRC,BASE
from testing.luke_waveform_only_global_review_v1 import near

def main():
    e=pd.concat([pd.read_csv(OUT/f'chunk_{s}.csv') for s in [930,935]],ignore_index=True);d=pd.read_csv(OUT/'seed_distinctness.csv');z=np.load(SRC/'templates.npz');fs=json.loads((BASE/'recording/rescue_recording_manifest.json').read_text())['sampling_frequency_hz'];rows=[]
    for c in d.itertuples():
        seed=z[f'unit_{c.unit_id}_seed_frames']/fs;q=e[(e.unit_id==c.unit_id)&(e.status=='accepted')];n=int(near(seed,q.time_s.to_numpy()).sum());frac=n/len(seed)
        rows.append(dict(unit_id=c.unit_id,seed_events=len(seed),seed_recalled=n,recovery_fraction=frac,qualifies=bool(c.eligible and n>=5 and frac>=.2),rank_score=c.rank_score*np.sqrt(frac)))
    q=pd.DataFrame(rows);chosen=q[q.qualifies].sort_values('rank_score',ascending=False).head(20).unit_id;q['selected']=q.unit_id.isin(chosen);q.to_csv(OUT/'training_qualified_candidates.csv',index=False)
    (OUT/'training_qualification_settings.json').write_text(json.dumps(dict(source_chunks=[930,935],threshold=.86,margin=.025,gain=[.35,3],minimum_recovered_seed_events=5,minimum_recovery_fraction=.2,maximum_candidates=20,selection='Existing depth-hidden shape/locality/repeatability eligibility plus seed-time recovery; rank global distinctness times sqrt(recovery); heldout events not read',limitations='Training recovery is a sensitivity control, not heldout identity validation; sorted seed timestamps are provisional'),indent=2));print(q[q.selected].to_string(index=False))
if __name__=='__main__':main()
