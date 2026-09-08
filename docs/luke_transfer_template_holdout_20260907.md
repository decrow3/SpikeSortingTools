# Early/late candidate templates persist in separate temporal holdouts

Used frozen first-half candidate templates from 960–965 and 9510–9515 s to search independently detected peaks at 970–990 and 9520–9540 s. No sort identities or DREDGE estimates entered matching. Templates use the central 61 samples and channels within ±60 µm, weighted by template energy relative to fixed noise. Matching criteria were fixed before evaluation: cosine ≥0.9, gain 0.4–2.5, timing adjustment ±3 samples, and a 0.03 margin over available nearby competitor templates. Same-template duplicates within 1 ms were suppressed.

## Results

| Candidate | Holdout | Accepted events | Valid five-second bins |
|---|---|---:|---:|
| Early ch293 / 2920 µm | 970–990 s | 180 | 3/4 |
| Early ch338 / 3380 µm | 970–990 s | 377 | 4/4 |
| Late ch290 / 2900 µm | 9520–9540 s | 381 | 4/4 |
| Late ch330 / 3300 µm | 9520–9540 s | 98 | 2/4 |

13/16 bins contain at least ten accepted events. Their median waveform/template cosine ranges from 0.935 to 0.984. These medians are conditional on passing the event template score and therefore are not an independent specificity test. Centroid variation within each candidate's valid bins is small (approximately 0.7–1.7 µm range), but no physical-displacement calibration or confidence interval is supplied by this experiment.

The late ch330 candidate has two nearby competitor templates (ch322 and ch334) and retains matches with the fixed margin. The other three targets have no nearby competitor in the present sparse library. Their matches demonstrate temporal waveform support but do not establish identity specificity. Different waveform cohorts at different depths must not be assumed to represent the same motion without further comparison.

Low-count bins remain gaps: early ch293 has one accepted event at 977.5 s; late ch330 has nine events in each of the first two bins. These bins were not plotted as valid motion estimates or filled by interpolation. Small count differences around the ten-event display threshold are not biological state changes.

## Consequence

These candidates provide usable provisional waveform observations in new early/late intervals. The next comparison can test whether original versus compensated DREDGE fields are consistent with these observations, while retaining sparse-identity and depth-coverage limitations. A DREDGE disagreement must trigger investigation, not tuning toward these centroids. Shallow coverage and difficult-motion identity remain outstanding.

## Artifacts and execution

Script: `testing/luke_transfer_template_holdout.py`. Outputs: `testing/outputs/luke_transfer_template_holdout_v1/`, including settings, freshly detected peak arrays, accepted event times/scores/gains, bin-level counts and centroids, saved median waveforms and PNG/PDF figure. The figure was inspected and its time axes were adjusted to show all bins, including low-count gaps; no analysis was rerun for that presentation fix.

Independent service `luke-transfer-template-holdout-v1` was verified running after launcher exit and later completed with zero exit status. Launch command, logs and receipt are persisted. No production motion correction or spike sort was launched. Full goal completion remains unproven.
