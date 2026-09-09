# Luke saved native motion audit — September 7, 2026

**Sharing completed after explicit user approval:** 53 audit files (15.83 MiB)
were copied to `/mnt/NPX/Luke/20250804/shared_analysis/luke_motion_audit_20260907_v1/`
and every destination SHA-256 was verified. `TRANSFER_COMPLETE.json` records
completion. Earlier blocked-transfer statements below are historical.
Verification used the shared mount on huklaban5; access from huklaban1 was not
independently tested.

**Nomination: rescue 12/9 with motion off is the operational winner among the
completed, tested configurations. Do not take either saved native rigid or
native nonrigid configuration unchanged to the full probe.** Confidence is
moderate in this operational exclusion, low in motion-off's biological
superiority. Motion correction remains scientifically relevant; this comparison
used an unstable native motion estimate and cannot settle its potential benefit.
No sort was launched. The existing full-session rigid hold remains in force.

This resolves the missing-file blocker in the previous method-selection report.
The saved outputs are available on huklaban5. The recommendation now rests on
an observed estimator problem, rather than the invalid overlap-asymmetry score.

## Evidence and scope

These are completed 10,473.554-second sorts of the same 100-contact strip,
1400–2380 µm, with the scoring interior 1600–2180 µm. All use thresholds 12/9.
The audit validates the recording request digest, completed sort manifests,
29,999.835983 Hz sampling rate, channel geometry, settings, and sorter version.
Sort and curated motion arrays agree exactly. Kilosort 4.0.27 was recorded in
all three logs and is installed locally.

| Check | Native rigid | Native nonrigid |
|---|---:|---:|
| Configured blocks / saved overlapping fields | 1 / 1 | 6 / 11 |
| Two-second batches | 5,237 | 5,237 |
| Physical displacement range, µm, before median removal | −309.5 to +315.5 | −325.5 to +329.5 |
| Median absolute adjacent field step, µm | 2.5 | 3.5 |
| 95th percentile absolute adjacent field step, µm | 146.125 | 149.0 |
| Batch transitions with any field step >100 µm | 560 / 5,236 (10.7%) | 610 / 5,236 (11.7%) |
| Batches with any median-centered displacement >100 µm in magnitude | 384 / 5,237 (7.3%) | 405 / 5,237 (7.7%) |
| Batches with any scoring-interior sampling coordinate outside the processing strip | 76 | 75 |

The nonrigid step quantiles pool all 11 fields; batch counts use “any field” and
are not directly equivalent to single-field counts. Its median trajectory has
correlation **0.9972** with rigid. Both share large, brief excursions; increasing
spatial flexibility did not eliminate them. Sampling outside the strip affects
only 0.087% / 0.097% of interior batch-channel coordinates, respectively; it is
an additional diagnostic, not an explanation for all yield differences.

![Saved motion estimates](../testing/outputs/luke_saved_motion_field_audit_v1/motion_fields.png)

The top panel shows the full native excursions without smoothing or clipping.
The lower panel uses a different, explicitly labeled vertical scale to show the
smaller independent estimates at rigid's 1889 µm reference depth. Both native
curves share the large excursions, so they often overlap visually.

## Sign, clock, and application checks

Kilosort's saved `dshift` is a correction shift: `get_drift_matrix` samples the
input at `y - dshift`. SpikeInterface samples at `y + motion`. Therefore the
physical displacement used here is **−dshift**, fixed before examining
correlations. This is not a sign chosen to maximize agreement.

An analytic +20 µm spatial translation on the saved probe geometry confirms
the application sign: relative waveform-vector error is 0.0346 with native
`dshift=-20`, 0.5423 with zero, and 0.9625 with `+20`. This validates matrix
orientation/sign on a smooth test profile; it does not validate real-waveform
fidelity or the estimated motion itself.

Native batches are placed at `(batch_index + 0.5) × 60000 / fs`. Independent
time bins are acquisition-clock values, from which the recording's attested
3057.677050340359-second origin is subtracted. Comparisons interpolate linearly
in time and depth on common support, remove only each series' median, and do
not optimize lag, scale, or sign. DREDGE covers 5,237 native centers; the other
two estimators cover 5,236. The small end mismatch is excluded, not extrapolated.

The installed 4.0.27 application path multiplies `(M @ Wrot) @ X` using each
batch's `dshift`; these are consequential voltage-interpolation coordinates.
The off arm's effective `nblocks=0` and `dshift=None` are verified. Its nested
settings still say `nblocks=1`, demonstrating why effective state must be checked.

## Comparison with independent estimates

| Estimate at 1889 µm | RMS after median removal, µm | Pearson correlation with native rigid | RMS difference, µm |
|---|---:|---:|---:|
| DREDGE | 3.390 | 0.213 | 45.163 |
| Decentralized | 0.370 | 0.149 | 45.639 |
| KS motion sidecar | 3.545 | 0.271 | 44.866 |

Native rigid's RMS is approximately **45.7 µm**. Per-depth comparisons for all
11 nonrigid fields are saved in `field_agreement.csv`. Independent estimates
are not ground truth, share detections, and differ in preprocessing and spatial
coverage from native sorting. In particular, decentralized is almost flat at
this depth, so agreement among three methods should not be overstated.
Nevertheless, the very large native excursions are unsupported by these
same-time, same-depth estimates. A 100-channel strip is not automatically
invalid; the actual saved field is the reason for concern.

Kilosort's [official parameter guidance](https://kilosort.readthedocs.io/en/latest/parameters.html)
notes that drift estimates depend on channel count, spacing, and available
spikes, and explains rigid versus nonrigid settings. It does not establish a
universal failure threshold for this strip. A plausible explanation is unstable
coarse histogram alignment: 4.0.27 accumulates shifts across coarse iterations
before the local refinement, and both arms share near-identical excursions.
This is an inference from the implementation and fields, not a proven root cause.
The exact native pre-registration detection raster was not saved here.

## Corroborating detection-profile diagnostic

Using the existing, uncorrected localized detections, select batches whose
native rigid shift is >100 µm from its median, then choose the nearest batch
within 10 seconds with shift ≤25 µm from its median. Compare 5 µm depth
histograms in the scoring interior on common interpolation support; require
at least 100 high-batch strip detections, 100 quiet-interior detections, and
80 supported depth bins. Quiet batches may be reused.

There are **378 pairs and 360 distinct quiet batches**. Median spatial-profile
correlation is **0.401 without shifting** versus **0.308 after the native relative
shift**. Native shifting lowers correlation in **58.5%** of pairs. This is
modest corroboration, not a decisive independent validation: the screen was
chosen after inspecting the native field, profiles mix neurons, firing changes
confound agreement, and the same detections underlie independent motion fields.
It is not raw-voltage evidence, a neuron-family recovery measure, or a promotion
gate. No significance test is claimed.

## Updated method selection

- **12/9 motion off — nominated operational choice.** It avoids the demonstrated
  native field instability and retains the established reference operating point.
  Motion-related dropout and its biological completeness remain unresolved.
- **12/10 off — conservative threshold alternative.** Fewer coincidences, but
  substantially fewer events; their validity remains unknown. This audit does
  not add evidence that changes the prior threshold ranking.
- **10/10, 10/9, 9/9 off — no new reason to prefer them.** Prior comparisons found
  limited additional yield or less favorable interior/guardrail tradeoffs.
- **10/8 and 9/8 off — higher yield, unqualified recovery.** The roughly one-third
  event increase still carries a coincidence concern.
- **12/9 native rigid — reject unchanged for full-probe promotion.** Known-truth
  motion benefit remains relevant, but does not rescue this unstable real-data
  estimate. Its lower cluster/event totals are not measured neuron losses.
- **12/9 native nonrigid — reject unchanged.** It largely repeats the rigid
  excursions, with added depth dependence and no demonstrated aggregate rescue.

The prior 4/141 amplitude-completeness coverage and invalid directional overlap
score remain limitations. This audit does not replace them with a proven
biological winner. It also does not validate cross-arm neuron families, review
raw waveforms of candidate-only events, or qualify pooling of the correspondence
graph. Those remain necessary to measure biological recovery reliably.

If the priority is a defensible operational run using an already tested setting,
choose **12/9 off**. If the priority is resolving motion before that commitment,
the next useful computation is **full-probe registration-only estimation**, using
the intended preprocessing and independent field comparison before any template
learning/full sort. Qualify its field before reconsidering native correction;
do not blindly replay either strip configuration. No such computation was
launched by this audit, and no full-sort hold was removed.

## Reproducibility and sharing

Primary computation: `testing/luke_saved_motion_field_audit.py`. Companions:
`testing/luke_saved_motion_raster_check.py` and
`testing/luke_saved_motion_operator_check.py`. Results are in
`testing/outputs/luke_saved_motion_field_audit_v1/`, including input SHA-256s,
field exports that do not require pickle, exact per-depth metrics, batch-pair
rows, a detection histogram, the figure, and an executed audit notebook.

The intended additive shared handoff (transfer blocked; not yet created) is
`/mnt/NPX/Luke/20250804/shared_analysis/luke_motion_audit_20260907_v1/`.
The locally staged package README documents included files, shared-source dependencies, reproduction,
and inventory verification. It preserves the original handoff and failed-run
records. It includes native ops/settings for all three arms, independent motion arrays,
code, aggregate diagnostics, reports, and the hold. Curated spike trains,
per-unit templates, original comparison tables, large PCA arrays and recording
binaries are omitted; their source locations remain documented. A broader
curated-array transfer was rejected by automatic approval review; the transfer
was narrowed to the files required for this audit. Automatic review also rejected
that 15.83 MiB package, requiring explicit approval to share motion fields,
ops/settings and diagnostics to the network drive. Nothing was transferred.
The concrete package is staged locally at
`testing/outputs/luke_motion_handoff_pending_20260907_v1/`, with a verified inventory. The exact native detection
raster cannot be supplied because it was not saved.

**Validation assessment: share with caveats.** Recording/settings, field
calculations, sign behavior and saved detection-profile calculations were checked.
The figure was visually inspected, four notebook code cells executed successfully,
and seven focused comparison/screen tests passed. Field implausibility is supported;
biological superiority and precise failure mechanism are not established.
