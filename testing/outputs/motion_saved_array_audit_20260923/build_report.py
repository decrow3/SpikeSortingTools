from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
P=Path(__file__).resolve().parent
read=lambda n:pd.read_csv(P/(n+'.csv'))
def table(d):return d.to_markdown(index=False,floatfmt='.3f')
exc=read('excursions');inc=read('increment_sd');fast=read('fast_component');sp=read('depth_spread');mat=read('matrix_excursions');quiet=read('quiet_motion');lh=read('lighthouse_increment_sd')
ids=['p06_f010','p06_f045','p08_f025','p08_f029','p08_f033','p08_f044','p09_f020','p09_f024']
ls=lh[(lh.source=='luke_imec1_dots_sorterfree_waveform_discovery_v3_s300/heldout_events.csv')&lh.candidate.isin(ids)]
rigid='rigid_after_depth_centering'
text='''# Saved motion-array audit and export — 23 September 2026

Only existing documents, field arrays, template arrays, and event assignments were read. No estimation, raw-voltage extraction, sorting, or injection was performed. `analyze_saved.py` contains the arithmetic and export procedure; `build_report.py` renders this report. Source artifacts were not modified.

## Definitions and important corrections

- All times here are seconds from AP recording frame zero. Intervals are half-open. The old DREDGE absolute clock is converted by subtracting 3057.6775463558583 s.
- Displacement convention: corrected depth = observed depth − displacement. All motion magnitudes are µm.
- **Window-centred rigid** means subtract each depth trace's temporal median within the selected window, then take the median over depths at each time. This reproduces the pilot's published rigid definition. The columns named `rigid` in the machine-readable files instead preserve the other convention used in the preceding answer: median over the saved, uncentred depth traces. `rigid_after_depth_centering` identifies the pilot-consistent convention explicitly. The two are not interchangeable because the cross-depth median is nonlinear.
- Excursion means P95 minus P5 over time, not full range. Per-depth excursions are unaffected by subtracting constant offsets. Reported per-depth P5/P95 endpoints preserve saved offsets; their zero is arbitrary.
- Increment SD is population SD (`ddof=0`) of d(t+lag)−d(t), without removing a fitted trend. For uniform 0.25 s fields, lags use exact index differences 1,4,8,20,80 and pair counts 479,476,472,460,400 in each 480-sample window. It measures variability of changes, not average speed or physical ground truth.
- There is no pilot field covering 8160–8280 s. Both full-session fields cover both requested windows. No pilot field was extrapolated.
- The 930–1030 s 348-channel matrix is **imec0**. The p10 and late-window analyses are **imec1**.

## 1. imec1 p10 alone

The pilot interval is [987.3553929363488,1107.3553896029887) s. Actual field samples run from 987.356092949996 to 1107.106092949996 s.

'''
text+=table(exc[exc.field=='pilot_p10'][['component','depth_um','p5_um','p95_um','excursion_um']])+'\n\n'
text+='Largest absolute 0.25 s steps:\n\n'+table(read('max_steps').query("field == 'pilot_p10'")[['component','max_abs_step_um','signed_step_um','from_s','to_s']])+'\n\n'
text+='''The largest local step is a model-output change, not a validated tissue displacement. The pilot and full-session fits differ greatly in the same p10 interval; the two full-session fits agree much more closely with one another. This prevents treating the pilot's magnitude as an established recording property.

## 2. Increment time scales

Window-centred rigid increment SD:

'''
text+=table(inc[inc.component==rigid].pivot(index=['field','window'],columns='lag_s',values='sd_increment_um').reset_index())+'\n\n'
text+='Window-centred rigid excursions:\n\n'+table(exc[exc.component==rigid][['field','window','p5_um','p95_um','excursion_um']])+'\n\n'
text+='''### Faster-than-two-second sensitivity

A percentile range has no unique additive frequency decomposition. As an explicit descriptive convention, high-pass the selected rigid trace with a fourth-order Butterworth at 0.5 Hz using forward/backward SOS filtering; omit 2 s at each end and compare P95−P5 of the high-pass component with P95−P5 of the original on that same interior. This is a smooth cutoff (with the forward/backward squared response), not an exact division at 2 s. The ratio is **not a fraction of total excursion explained**, and the field's 1 s estimation kernel already limits fast-motion representation. No field was refit.

'''
text+=table(fast[fast.component==rigid][['field','window','full_excursion_same_interior_um','highpass_excursion_um','excursion_ratio','variance_ratio']])+'\n\n'
text+='''### Saved lighthouse observations

The eight-proposal compact expansion has **no events in p10 or 8160–8280 s**. Historical imec1 tables do cover short pieces of p10. A scan of 76 scored event tables found no events in the late interval; see `event_inventory.csv`. These are candidate measurements, not verified independent cells. Revisions and hypotheses are preserved separately and never pooled.

For irregular event depths, use medians in 0.25 s bins separately for each candidate and source table. Difference only occupied endpoints within the same recorded acquisition interval: 998–1002, 1014–1025, 1071–1075, or 1097–1112 s, clipped to p10. No gap interpolation, filling, or cross-window differences. Thus lags are bin-grid lags (individual event separations have sub-bin jitter), unlike exact field sample lags. SD requires at least two pairs. Missing statistics are unavailable, not zero. No interval supports a 20 s lag.

The following are the historical **broad-template s300** assignments for IDs that later entered the eight-proposal expansion; they are not the expansion's final compact-template tracks:

'''
wide=ls.pivot(index='candidate',columns='lag_s',values='sd_increment_um').reindex(ids)
text+=table(wide.reset_index())+'\n\nPair counts for the same rows:\n\n'+table(ls.pivot(index='candidate',columns='lag_s',values='n_pairs').reindex(ids).reset_index())+'\n\n'
text+='''The very sparse, candidate-specific and sometimes enormous changes do not establish how fast the tissue moved. A high-frequency excursion decomposition is not warranted for these gapped, unverified tracks. `lighthouse_increment_sd.csv` also reports the other historical sources separately.

## 3. Across-depth spread and offsets

The previously quoted 24.59 µm DREDGE and 20.69 µm shared-recovery MEDiCINe spreads were computed from **uncentred saved depths**. They include constant depth-dependent offsets. In contrast, the pilot's published spread used per-depth temporal centring. For 8160–8280 s:

'''
text+=table(sp[sp.window=='8160_8280'].drop(columns='temporal_depth_medians_um'))+'\n\n'
text+='''The temporal depth-median offsets have full depth ranges of 21.47 µm (original DREDGE), 9.31 µm (shared recovery), and 11.80 µm (merge bias). Subtracting them changes the median spread but does not change each depth's own temporal excursion. A depth-varying static offset in a field applied directly to voltage causes a permanent depth deformation; a common offset merely translates. The original upstream interpolation passed the saved DREDGE array directly into `Motion`, so these offsets were not removed there. This analysis does not establish that the offsets are biologically real or spurious. `depth_spread.csv` retains all offsets.

## 4. Are p06_f010 and p08_f044 duplicates?

The final strict-event table contains 115 p06_f010 events and 23 p08_f044 events. There are **0 one-to-one temporal matches within ±0.5 ms**: 0/115 and 0/23. These are all qualification and expansion windows combined, not just common-support increments. The global winner assignment is exclusive, so lack of coincident assignments does not rule out one cell being partitioned into two labels.

The compact-template geometry-aware cosine is **0.926689**, maximizing temporal lag from −3 to +3 waveform samples over ten exactly overlapping relative channels; minimum bidirectional energy coverage is **0.730265**. Different peak-channel phases make a naive flattened-array cosine inappropriate. This is the cosine between this pair, not either candidate's similarity to its nearest rival. Both the frozen compact templates and geometry are exported.

These results do not prove duplication, but they also do not validate independence. Their correlated trajectories should not be counted as independent cellular replication until that identity question is resolved. Moreover, the 2.9 µm separation is between broad seed-depth summaries, not independently verified instantaneous cell depths.

## 5. imec0 930–1030 s matrix fields

Both LFP candidates are rigid traces. Their per-depth excursion is therefore the same at every channel; neither has four or nine independent depth estimates. SG25 is the saved native-250-Hz trace after the documented Savitzky–Golay operation.

The saved matrix AP-rigid arm is the equal-depth rigid projection and has a 61.341 µm excursion. Taking a cross-depth median of the saved nonrigid AP field instead gives 63.291 µm. The applied arm and this diagnostic median are explicitly separated below.

'''
text+=table(mat)+'\n\n'
text+='''### The actual 10 September quiet-period result

This is a **different comparison**: its LFP candidate was the frozen `band_0p5_8` rigid trace, not the matrix's native-250-Hz/SG25 pair. Do not assign these errors to SG25 or native AP. The AP candidates are the full-session MEDiCINe field and its rigid projection.

Reference = consecutive 5 s increments of frozen, strict depth-aware lighthouse family medians, on common support, excluding predeclared concern units and retaining the 5 µm lattice-node sensitivity. Quiet = absolute observed lighthouse increment below 20 µm. Each family has equal weight; quiet strata permit at least one increment per family. The table distinguishes RMS of the field's predicted increments from RMSE of those increments against the lighthouse reference.

'''
text+=table(quiet)+'\n\n'
text+='''In particular, LFP predicts 48.766 µm RMS quiet increments against 1.437 µm RMS reference increments in 930–1030 s, producing 48.863 µm residual RMSE. In 1150–1200 s those values are 21.666, 1.426, and 21.855 µm. These are 5 s change statistics, not full-window excursions or direct proof of physical estimator error.

## 6. Export inventory

`export/` contains all requested field files, source configurations/receipts where available, original KS trace, frozen identity templates, the original 10 September comparison arrays/tables, and 76 original scored imec1 event tables. Event CSVs are losslessly gzip-compressed; selected canonical tables also have normalized copies adding `candidate` and `depth_um` while retaining original time, score, status, bank, event IDs, and per-event runner-up fields where present. No rival score was invented where absent. Older variants remain explicitly named and are not independent datasets.

- `source_manifest.csv`: source absolute path → exported path, byte count, exported SHA256, and decompressed original SHA256 for gzip CSVs. Copied source bytes were verified against originals.
- `SHA256SUMS`: checksums of every delivered file except the checksum file itself.
- `increment_sd.csv`: **all four depths**, raw-median rigid, and window-centred rigid at every requested lag.
- `matrix_excursions.csv`: exact P5/P95 endpoints, depths and excursions for all matrix fields.
- `fast_component.csv`, `depth_spread.csv`, `max_steps.csv`, `identity_overlap.json`, `lighthouse_increment_sd.csv`: complete numerical results.

The export is a directory rather than a duplicate archive because this filesystem has limited free space. No source files were deleted. All source paths remain in the manifest.

## Appendix: per-depth increment SDs

'''
for (name,win),g in inc[inc.component.str.startswith('depth_')].groupby(['field','window']):
 text+=f'### {name}: {win}\n\n'+table(g.pivot(index='component',columns='lag_s',values='sd_increment_um').reset_index())+'\n\n'
(P/'README.md').write_text(text)
# Check all quantitative results against basic invariants; no estimator or sorter tests.
assert len(inc)==150
assert set(inc.n_pairs)=={479,476,472,460,400}
assert len(mat)==21
assert int(json.loads((P/'identity_overlap.json').read_text())['pairs_within_0p5ms'])==0
for p in (P/'export').rglob('*.npz'):
 with np.load(p,allow_pickle=True) as z:assert len(z.files)>0
# Checksums cover originals, normalized tables, calculations, report, and analysis scripts.
lines=[]
for p in sorted(P.rglob('*')):
 if p.is_file() and p.name!='SHA256SUMS':lines.append(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(P)))
(P/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
print('Report and',len(lines),'checksums written')
