# Luke 2025-08-04: leading failure hypotheses and decisive tests

## 2026-08-29 raw-voltage qualification

The original six-sigma unmatched-event evidence below remains useful for
following individual Luke events through the pipeline, but it is **not a fair
cross-session estimate of biological spike density**. A new matched raw audit
shows that Luke has more unreferenced high-frequency extrema and substantially
more shared local voltage than Yates. After the same 100 µm local median
reference, a six-sigma threshold corresponds to approximately 42--45 µV on
imec0, 38--40 µV on imec1, and 30 µV on Yates, so six sigma is not a common
voltage threshold across recordings.

At a fixed 75 µV threshold after that reference, imec0's negative-event density
is much closer to Yates, while imec1 retains fewer negative events and a six-
to ten-fold positive-event excess. The leading upstream question is therefore
no longer simply whether Luke lacks detectable voltage. It is whether shared
high-frequency signal, reference choice, polarity or waveform morphology—most
strongly on imec1—prevents compact neural events from reaching the sorter in a
usable form. Lowering a global detection threshold is not currently supported.

Spatial-footprint results from the first raw audit are not evidence: their
normalization and temporal search window were found to be unfair across probe
types. The corrected footprint calculation and shank-median reference control
remain to be run. Full details are in
[`luke_yates_raw_voltage_audit_notes.md`](luke_yates_raw_voltage_audit_notes.md).

## Current answer

The strongest working model is not a single acquisition failure. Luke appears
to contain real shared motion plus a large population of raw high-frequency
events, but their neural content is reference- and polarity-dependent. Some
compact neural events are still likely rejected, misassigned, or damaged
during conditioning and registration. The five hypotheses below are ordered
by the value of testing them, not by certainty.

The immediate goal is to distinguish three event-level outcomes before doing
more full sorts:

1. the raw candidate is an artifact and should be rejected;
2. it is a real spike detected under another local cluster;
3. it is a real spike with no local sorted event.

## 1. The apparent misses include artifacts or non-neural transients

**Why it leads:** The current deliberately simple detector finds negative raw
peaks above six robust noise standard deviations. That proves a large event is
present, but not that it is a neuronal spike. The example panels include both
spike-like biphasic events and less convincing sharp or noisy events.

**Prediction:** Events rejected by Kilosort will disproportionately have broad
spatial footprints, common-mode timing, implausibly narrow or long waveforms,
filter ringing, or no refractory-compatible recurrence.

**Minimal test:** For every unmatched raw event, measure waveform duration,
peak/trough ratio, 3--5-channel spatial footprint, spatial center, common-mode
coincidence, and proximity to saturation blanking. Hand-label a blinded sample
of 100 matched and 100 unmatched events. Use the labels to select a conservative
spike-like gate, then recompute the unmatched fraction.

**Decision rule:** If fewer than about 20% of unmatched six-sigma events pass
the spike-like gate, the raw-event result is mostly a detector artifact. If a
large, visually convincing population remains, advance hypotheses 2--5.

## 2. Kilosort thresholds or template matching reject genuine large spikes

**Why it leads:** On imec1, 49--67% of six-sigma raw peaks in the template
window and 51--73% in the registration-outlier window had no sorted event
within 0.5 ms and 100 um. These are neighborhood event counts, not unit
identity. The sorter is also highly configuration-sensitive.

**Prediction:** Conservative spike-like raw candidates will appear in
Kilosort's preprocessed signal but fail at detection, template assignment, or
the learned-template threshold. Lowering the relevant threshold should recover
the same events without a disproportionate artifact or duplicate burden.

**Minimal test:** On one quiet and one motion window, trace each raw candidate
through Kilosort's detection and template-matching intermediates. Run a small
threshold sweep with all preprocessing and correction fixed. Report recall of
the reviewed raw event set, local duplicate rate, refractory violations, and
false detections.

**Decision rule:** Support this hypothesis if a modest threshold change
recovers at least 25% more reviewed events while contamination and duplicate
burden remain approximately stable.

## 3. External nonrigid correction damages or relocates otherwise sortable spikes

**Why it leads:** Luke's nonrigid displacement magnitude depends strongly on
the estimator, while the current correction applies a depth warp with
`border_mode='force_zeros'` and casts the result to `int16`. Only a small
fraction of units match across correction variants.

**Prediction:** Reviewed raw spikes will lose amplitude, spatial coherence, or
waveform similarity after nonrigid interpolation, particularly at large
displacements, steep spatial gradients, and probe boundaries. Rigid or no
external correction will preserve more events.

**Minimal test:** Apply no correction, rigid correction, and current nonrigid
correction to the same short windows without sorting. Propagate the exact same
raw-event sample indices through each branch. Measure retained amplitude,
waveform cosine, peak-channel/depth displacement, zero-filled samples, and
events that fall below a fixed detection threshold.

**Decision rule:** Support this hypothesis if nonrigid correction uniquely
causes a material loss (pilot target: at least 10%) in reviewed-event
detectability or waveform similarity relative to both no correction and rigid
correction.

## 4. Real tissue motion causes waveform drift and unit fragmentation

**Why it leads:** The two probes share a strong rigid trajectory, independent
motion estimators broadly agree on rigid displacement, and spike yield falls
during large nonrigid-spread periods. imec1 is less temporally stable than
imec0.

**Prediction:** The same raw waveform family will move coherently across
channels/depth and be assigned to different clusters before and after motion
events. Combining those clusters should restore a continuous firing history
and a plausible refractory period.

**Minimal test:** Choose 10 high raw-SNR waveform families spanning depth.
Track amplitude-normalized multichannel waveforms and peak depth through the
shared event near 7,275 s and the imec1 outlier near 8,220 s. Search nearby
clusters for complementary time support and waveform continuity. Compare the
observed depth trajectory with both probes' motion estimates.

**Decision rule:** Support this hypothesis if multiple waveform families show
continuous raw trajectories but mutually exclusive cluster assignments whose
merge restores temporal continuity without creating refractory violations.

## 5. Conditioning and repeated Kilosort preprocessing distort the signal

**Why it leads:** The Luke path includes saturation blanking, bad-channel
interpolation, a 12th-order zero-phase bandpass, and local median reference.
The saved Kilosort configuration also enables its preprocessing and CAR. Their
combined effect has not been isolated against the raw event set.

**Prediction:** One conditioning step, or the combination of local reference
with Kilosort CAR/high-pass, will reduce reviewed-event amplitude or alter
spatial footprints. The effect will be reproducible before motion correction.

**Minimal test:** On matched raw windows, compare raw high-pass only; current
conditioning; current conditioning without blanking; global instead of local
reference; and exactly one versus repeated filtering/CAR. Keep the raw event
indices fixed and measure event retention, amplitude, waveform correlation,
noise, ringing, and common-mode rejection.

**Decision rule:** Support this hypothesis if one step causes a reproducible
event-retention or waveform penalty that is not explained by better artifact
rejection.

## Recommended execution order

1. Hand-label and spatially validate the six-sigma unmatched events.
2. Trace the validated events through conditioning and motion interpolation.
3. Trace the survivors through Kilosort detection and assignment.
4. Run short threshold and correction sweeps only after the failure stage is
   known.
5. Run at most two full-session finalists.

This order prevents a sorter sweep from compensating for artifacts or an
upstream waveform-damage problem.

## Implemented first-stage validation

The blinded multichannel review is implemented in
`testing/luke_multichannel_event_validation.py`. It draws 100 locally matched
and 100 unmatched candidates, extracts five nearby AP-band waveforms directly
from the original binary, and writes contact sheets, a blank label form,
quantitative metrics, and a separate key. Run it from the repository root with:

```bash
python testing/luke_multichannel_event_validation.py
```

Review each event as `neural`, `artifact`, or `uncertain` in
`review_labels.csv` without opening `review_key.csv`. After all labels are
complete, unblind and score the result with:

```bash
python testing/luke_score_multichannel_review.py
```

The automatic neural-like flag is an advisory screen only. The prespecified
primary endpoint is the manually labeled neural fraction among unmatched
events, excluding uncertain labels.

Reviewed events can then be traced through the uncurated and curated sort
outputs with:

```bash
python testing/luke_trace_reviewed_events.py
```

This distinguishes candidates present in Kilosort's raw output but lost during
curation from candidates already absent before curation.

The reviewed events can be compared against the existing full-session sort
variants, with a local time-jitter null to control for different spike
densities, using:

```bash
python testing/luke_compare_reviewed_sort_variants.py
```

### Current result

The first blinded visual pass labeled 62 of 100 originally unmatched events as
neural and 32 as artifact, with 6 uncertain. That five-channel review
overcalled some probe-wide events as neural: events lost during curation had a
median of 137 active channels and only 9% local spatial energy. The full spatial
screen should therefore remain part of the definition of a credible neural
event.

More decisively, the patched and legacy DREDge sorts identify the same saved
preprocessed binary and have no recorded Kilosort-setting difference other than
the patched `cross_peel_claim_ms=0.25` and `cross_peel_claim_um=75` parameters.
The patched curated output contains 21.0 million spikes versus 53.9 million in
the legacy DREDge output. Existing non-patched full-session variants recover
61--82% of visually neural unmatched events, compared with 16% in the patched
sort; time-jitter null recovery is only 8--11% and 3%, respectively. This makes
the claim-mask intervention the leading explanation for the apparent Luke
yield deficit in the patched output. Event recovery alone cannot establish that
the denser legacy outputs are better, because they may contain duplicate peels;
refractory, contamination, duplicate-pair, and unit-continuity metrics remain
required.

The initial quality guardrails confirm a tradeoff rather than a simple winner.
The patched and legacy DREDge outputs contain 410 and 407 units, respectively,
but the patch increases KS-good units from 105 to 151 and reduces median
Kilosort contamination from 44.9% to 30.25%. At the same time, total spikes
fall from 53.9 million to 21.0 million and median unit rate falls from 3.47 Hz
to 1.30 Hz. The leading working hypothesis is therefore **over-suppression by
an otherwise useful duplicate-control intervention**.

The next Luke-specific short-window sweep should hold the saved preprocessed
input and all other Kilosort settings fixed while comparing: claim disabled;
0.10 ms at 25 and 50 µm; 0.25 ms at 25 and 50 µm; and the current 0.25 ms at
75 µm. Score each condition on reviewed-event recovery above jitter null,
near-zero-lag duplicate pairs, refractory violations, contamination, and unit
continuity through the registration-outlier window. Do not select on spike or
unit count alone.

### Implemented claim-mask sweep

`testing/luke_claimmask_window_sweep.py` implements that prespecified sweep.
Because the original cached preprocessed binary was deleted, it reconstructs
the same lazy graph from the raw imec1 stream, cached bad-channel metrics, and
saved full-session DREDGE motion. It saves two full-probe windows once: a
240-second window spanning the adjacent `pre_shared` and `template` review
epochs, and the 120-second `registration_outlier` epoch. Together these cover
all 200 reviewed events, including all 62 visually neural unmatched candidates
and all 12 candidates passing the conservative automatic neural-like gate.

Inspect the exact paths, samples, storage estimate, and 12 planned jobs without
writing data:

```bash
python testing/luke_claimmask_window_sweep.py --plan-only
```

On a machine with the patched Kilosort environment and a CUDA GPU, prepare,
run, and score with:

```bash
NUMBA_CACHE_DIR=/tmp/luke-claimmask-numba-cache \
python testing/luke_claimmask_window_sweep.py --prepare --run --score
```

The scorer reports recovery above a 20--500 ms time-jitter null, Kilosort good
unit count and contamination, nearby cross-unit near-coincident spike fraction,
median per-unit refractory violation fraction, and 10-second-bin activity
coverage. Existing complete jobs are reused; ambiguous partial jobs fail rather
than being silently overwritten.

### Claim-mask sweep result

All 12 short-window Kilosort jobs completed. The strongest compromise is
`cross_peel_claim_ms=0.10`, `cross_peel_claim_um=50`. In the shared/template
window it retained all 12/12 conservative neural-like events while reducing
final spikes from 1.576 million to 0.839 million, nearby cross-unit coincidence
fraction from 50.3% to 27.7%, median contamination from 46.0% to 34.35%, and
median refractory violation fraction from 2.21% to 0.92%. KS-good units rose
from 93 to 121. Its conservative recovery above jitter null was 0.932.

The original 0.25 ms/75 µm mask reduced the shared-window coincidence fraction
further to 12.1% and contamination to 28.55%, but recovered only 10/12
conservative events and 15/35 visually neural unmatched events. The 0.10/50
setting recovered 12/12 and 22/35, respectively. This directly supports
over-suppression by the original mask rather than simple removal of harmless
duplicates.

The registration-outlier epoch is qualitatively different. Claim-off produced
1.30 million learned-template detections in 120 seconds and a 77.0% nearby
cross-unit coincidence fraction. Even 0.10/25 removed 65% of learned detections.
At 0.10/50, final coincidence fraction was 32.7%, median contamination 43.9%,
and 82 units were KS-good, versus 77.0%, 51.55%, and 74 with claim disabled.
No reviewed unmatched events in this epoch passed the conservative automatic
neural gate, so its visual recovery differences should not drive parameter
selection.

The saved aggregate result is
`claimmask_window_sweep/claimmask_window_sweep_scores.csv` under the surviving
Luke DREDGE pipeline directory. Within an otherwise fixed DREDGE pipeline,
0.10 ms/50 µm is the preferred mask compromise. The upstream ablation below
shows that a no-external-motion control must now be tested before selecting any
full-session finalist. Do not interpret the mask setting as the primary remedy
or return directly to 0.25 ms/75 µm.

## Interpretation boundaries

- Kilosort `cluster_Amplitude` is not a raw-voltage or raw-SNR ranking.
- A raw event coincident with a different cluster is evidence of possible
  splitting or misassignment, not an outright miss.
- Yates is a useful known-good reference but has different geometry, channel
  coverage, acquisition scaling, and preparation. It should not define Luke's
  expected spikes per recorded channel.

## Upstream ablation result

The claim-mask sweep is now superseded as a root-cause diagnostic by a fixed
2 × 2 conditioning/motion ablation in the registration-outlier window. With
claim masking disabled in all conditions, removing external DREDGE correction
reduced learned detections from 1.298 million to 0.562 million, reduced
cross-unit near-coincident burden from 0.770 to 0.365, increased KS-good units
from 74 to 95, and improved visually neural unmatched-event recovery from
70.4% to 85.2%. Preserving float precision without removing motion changed the
learned count by less than 1% and did not improve recovery.

The evidence therefore promotes **motion interpolation damages the sorting
branch** to the leading verified upstream mechanism for this window. It does
not yet distinguish a poorly constrained DREDGE field from interpolation of an
otherwise valid field. Full results, definitions, limitations, and next tests
are in `docs/luke_20250804_upstream_ablation_report.md`.
