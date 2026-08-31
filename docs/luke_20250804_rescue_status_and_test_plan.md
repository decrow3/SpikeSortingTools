# Luke 2025-08-04 rescue status and test plan

**Evidence snapshot:** 2026-08-31

**Scope:** both Luke probes unless a result is explicitly probe-specific

**Status:** frozen rescue graph replicated at full-session scale across both
probes; bounded pinned-AIND challenger completed; preprocessing conditionally
frozen; motion estimation is the active unfinished stage.

## 2026-08-31 strategic update

This update supersedes the older prioritization language below while retaining
the detailed experiment history. Rescue and pinned AIND are substantially
different preprocessing architectures, yet they tie aggregate sealed-event
recovery at 470/720 and have similar pooled continuity and firing-rate
distributions. AIND improves refractory/coincidence diagnostics, while rescue
retains better yield and normalized similar-template burden. Together with the
successful full-session rescue on both probes, this places preprocessing in
diminishing-returns territory.

The frozen rescue graph is now the production/downstream reference. Pinned
AIND remains an independent competent comparator, not a full-session finalist.
Broad preprocessing searches are paused; reopen them only for a demonstrated,
stage-local failure. The active sequence is motion-estimator validation,
coordinate-only motion application, bounded sorter experiments, and only then
voltage warping or suppressive mechanisms.

The governing rule is **stage-local validation**: each stage must pass an
observable the next stage cannot rescue or manufacture. A better final sort
does not prove that every upstream choice was correct. The complete observable
map, advancement gates and reopening criteria are in
[`luke_pipeline_stage_local_validation_strategy.md`](luke_pipeline_stage_local_validation_strategy.md).

## Executive answer

Luke is not presently explained by a simple absence of extracellular signal or
by one failed channel. The most useful working model is a combination of:

1. distributed high-amplitude, positive-polarity and shared high-frequency
   voltage that interacts strongly with saturation handling;
2. a real bad-channel problem at physical channel 191 whose treatment affects
   whitening and localization;
3. excessive loss or duplication introduced by particular downstream choices,
   especially the tested claim mask and current external nonrigid voltage warp;
4. genuine long-timescale tissue motion that will probably require correction
   or motion-aware unit tracking after the unwarped baseline is established;
5. incomplete separation, in the current metrics, of neural recovery from
   artifact recovery, template duplication and collision-related over-peeling.

The data therefore still look potentially rescuable by pipeline changes. The
evidence does **not** yet support claiming that Luke can match Yates in usable
unit yield or biological event density. It does support testing that possibility:
Luke has abundant large raw voltage, comparable broad-band noise in the matched
audits, and substantially better reviewed-neural recovery when the conditioning
and sorter interventions are chosen carefully.

## Frozen preprocessing reference

The current baseline fixes one upstream chain before motion correction is
reintroduced:

1. apply the Neuropixels phase correction;
2. apply samplewise bilateral blanking at 500 microvolts;
3. interpolate physical channel 191 and include it in Kilosort;
4. allow one internal Kilosort high-pass/CAR/whitening stage;
5. disable external motion correction, Kilosort internal motion correction,
   the cross-peel claim mask and Kilosort's batch artifact threshold;
6. save raw samples exceeding 500 microvolts as a separate artifact sidecar and
   exclude nearby detections from artifact-sensitive claims.

This is the frozen engineering reference, not a biological optimum. In
particular, point blanking creates local filter ringing and false peaks around
saturation. It is retained because removing it caused a much larger loss of
reviewed-neural recovery at the sorter output in both harder test windows.

## What the matched tests establish

### Channel 191

Interpolating channel 191 and including the resulting trace in Kilosort is the
better tested choice. In the 120-second good window, blanking plus interpolation
produced 197 Kilosort-good units versus 173 when channel 191 was excluded, with
lower median contamination (17.3% versus 21.45%). The same direction held
without blanking: 178 versus 165 Kilosort-good units and 19.8% versus 23.5%
median contamination.

The caveat is important. Floating-point interpolation makes the channel exactly
dependent on its neighbors and can make covariance singular; production int16
quantization regularizes this relationship. Channel 191 must remain excluded
from localization, covariance and claim metrics even when its synthesized trace
is supplied to the sorter.

### Saturation blanking

The good window could not distinguish blanking policies: all four conditions
recovered all 14 reviewed neural events. The neutral and pathological windows
did distinguish them:

| Window | 500 microvolt blanking | No blanking | Absolute recovery change |
|---|---:|---:|---:|
| Neutral | 95.2% of 21 | 42.9% of 21 | +52.4 percentage points |
| Pathological | 81.5% of 27 | 48.1% of 27 | +33.3 percentage points |

Removing blanking reduced near-zero-lag coincidence excess and slightly
improved pathological-window contamination, but those gains were accompanied
by severe loss of reviewed-neural recovery. This makes the blanker necessary
for the current baseline while also identifying it as a high-priority target
for a sidecar-based artifact exclusion policy.

### Kilosort batch artifact rejection

Kilosort's native artifact threshold zeroes an entire batch when any sample
exceeds the threshold. That mechanism is too coarse here. At 300 microvolts it
rejected 14 of 60 neutral-window batches and erased 9 reviewed neural events;
at 400 microvolts it became mostly ineffective; and at 500 microvolts it did
nothing in either harder window. It stays disabled.

### Claim mask

Every nonzero claim-mask setting tested reduced reviewed-event recovery. The
mask remains disabled. It should not be reconsidered until the upstream signal
and motion choices are fixed, and then only on an explicit recovery-versus-
duplicate frontier.

### Motion correction

The current external DREDGE warp is harmful in the pathological 120-second
window. Compared with no motion correction, it produced more learned detections
(1,298,050 versus 562,417) and more final spikes (905,605 versus 544,553), but
fewer Kilosort-good units (74 versus 95), higher median contamination (51.55%
versus 38.65%), much higher cross-unit coincidence excess (0.770 versus 0.365),
and lower reviewed-neural recovery (70.4% versus 85.2%).

This rejects the **tested implementation and parameterization**, not voltage
registration in general. The next motion experiments must validate the motion
field and the resampling operator separately before another nonrigid sort.

A discovery-only known-drift residual screen now supplies that first operator
test at eight previously used two-second motion states. Across five exact donor
templates and three assumptions about the continuous waveform between contacts,
full nonrigid kriging minimized scaled residual but introduced enough additional
amplitude error to fail the joint gate; full nonrigid IDW was worse. The
existing 0.25-strength rigid field passed, and sigma-20 kriging variants adding
either 0.10 or 0.25 of the residual nonrigid field also passed. These two
nonrigid variants advance only to paired injected-versus-control raw-snippet
testing. They are not eligible for confirmatory sorting yet. The parallel
provisional-unit residual route retained only four coherent events from one
unit and cannot rank candidates. See `docs/luke_validation_scaffolds.md` for
the protocol, outputs and stopping rules.

A separate direct motion-scale audit tests whether the successful fractional
gain simply reflects overestimated displacement. Explicit amplitude-depth
fingerprint matching agrees with DREDGE's direction in nearly every qualified
large-shift pair, but robust observed/DREDGE slopes range from 0.51 to 0.64
across the viable amplitude-resolved definitions; a depth-only control gives
0.81. The primary 95% interval is 0.31--0.90. This is compatible with moderate
scale inflation but not a clean fourfold coordinate error. Because the audit
and DREDGE share the same pre-registration peaks, an LFP, independently
qualified waveform-family, or external mechanical comparison is still needed
before assigning physical meaning to the 0.25 engineering gain. See
`docs/luke_20250804_direct_motion_scale_audit.md`.

The production path was subsequently verified to use two terminal AP branches:
DREDGE peaks and locations were computed from 300--3000 Hz voltage, and the
resulting field was applied to 300--6000 Hz voltage. A saved-peak amplitude
fingerprint confirms the historical estimator branch directly. A paired
60-second ablation then re-estimated otherwise identical nonrigid DREDGE fields
from 300--3000 and 300--6000 Hz using sample-time-grouped estimator/evaluation
halves, so simultaneous multichannel transients could not leak across the
split. The cutoff substantially changed detection count (244,805 versus
198,019); the resulting fields nevertheless had rigid correlation 0.85 and
residual correlation 0.90.

At 0.25 gain both fields preserved the same 160 temporally separated wideband
events closely (median waveform correlations 0.9954 and 0.9963). At full gain
both damaged voltage, but the wider estimator was less damaging: median
waveform correlation was 0.9047 versus 0.8848 and retained peak amplitude was
0.929 versus 0.893. Paired event-bootstrap intervals supported both differences.
The seven qualified held-out raster pairs gave mixed scale and correlation
results, so this does not show that the wider field is geometrically correct.
Treat estimator bandwidth as secondary: retain no motion as the baseline, and
carry 300--6000 Hz at 0.25 gain only as the leading estimator-band challenger
for a second discovery-window replication. Do not spend a full sort on this
axis without that replication and an independent motion anchor. The
implementation and durable outputs are
`testing/luke_motion_estimator_band_ablation.py` and the imec1 pipeline's
`motion_estimator_band_ablation` directory.

### Luke versus Yates raw voltage

The matched raw audit does not show a simple lack of large voltage in Luke.
Luke has more shared/local high-frequency voltage, an imec1 negative-event
deficit at fixed amplitude, and a large distributed positive-polarity excess.
Broad noise is of similar order (about 11 microvolts in the current matched
summary). Because reference and polarity affect event counts strongly, current
Luke--Yates density differences cannot yet be interpreted biologically.

### Cross-session recurrence check (2025-08-05)

A discovery-only cross-session check now tests whether imec1's positive-polarity
excess and the channel-191 outlier are specific to 2025-08-04 or recur on the
next recording session on the same probes and rig. `testing/luke_20250805_polarity_recurrence_audit.py`
reprocesses ten evenly spaced 2 s samples across the full session for both
probes on both 2025-08-04 and 2025-08-05, using the same 300--6000 Hz
bandpass, 100 um local median reference, and fixed 75 uV threshold used
throughout the raw-voltage audit. The 2025-08-04 recompute closely reproduces
the previously reported session-wide numbers (imec1 approximately 108/464
negative/positive events per mm per s here versus 102/471 in
`luke_yates_raw_voltage_audit_notes.md`), which supports treating the new
2025-08-05 numbers as comparable.

Both headline imec1 anomalies recur. The positive:negative event-rate ratio at
fixed 75 uV after local referencing is 4.31 for imec1 versus 1.15 for imec0 on
2025-08-04, and 2.80 for imec1 versus 1.30 for imec0 on 2025-08-05: imec0
stays close to parity on both days while imec1 stays sharply positive-dominant
on both days, computed independently each time. The physical-channel outlier
in `conditioning/channel_metrics.npy` at channel 191 also recurs on both
probes on 2025-08-05, matching 2025-08-04 and the SpikeGLX-metadata-declared
disconnected site near channel 191 at 1900 um noted in
`luke_validation_scaffolds.md`. Channel 191's own polarity ratio is stable and
negative-dominant in all four recordings (log-ratio -0.35 to -0.45), so it is
not the source of the positive excess, consistent with the existing
row-216-not-the-driver conclusion from `testing/luke_imec1_event_localization_audit.py`.

The per-channel polarity profile (log((positive+0.5)/(negative+0.5)) by
physical channel) correlates moderately across days for imec1 (Spearman
r=0.42, p=7e-18, n=384 channels) and weakly, with the opposite sign, for imec0
(r=-0.28, p=2e-8). The two probes do not track each other consistently within
a day either (r=-0.43 on 2025-08-04, r=0.08 and not significant on
2025-08-05). This pattern -- a real but moderate same-probe cross-day
recurrence specific to imec1, without a comparable imec0 recurrence or a
stable within-day cross-probe relationship -- favors a stream-fixed cause tied
to the imec1 probe, headstage, cable, or reference path over a
2025-08-04-specific biological or pipeline-tuning explanation. Moderate (not
near 1) recurrence means this is a real signal layered on session-specific
variation, not a fully deterministic hardware fingerprint, and it does not
identify which hardware element is responsible.

This raises the priority of the acquisition/electrical-bank audit (previously
P1, single-probe and confounded with depth) relative to further claim-mask or
motion-gain tuning, since a stream-fixed cause would be shared across every
session recorded on this imec1 stream rather than being fixed by adjusting
sorter parameters session by session. It does not by itself change the
provisional conditioning baseline, and it has not yet been run against a third
session, a second probe pair, or another rig. Outputs are in
`testing/outputs/luke_20250805_polarity_recurrence_audit/`
(`channel_event_summary.csv`, `channel_polarity_wide.csv`,
`session_probe_summary.csv`, `channel_191_check.csv`, `decision.json`). No
dedicated test file exists yet for this script.

### Deferred depth-resolved Luke--Yates comparison

The shallow cortical portions of the Luke and Yates recordings are the closest
anatomical comparison currently available. Luke's longer Neuropixels probes
were also lowered farther into cortex to sample deeper V1 banks representing
more peripheral visual field locations, so Yates should not be treated as a
whole-probe biological-yield standard. Conversely, Luke's shallow contacts may
have sampled cortex affected by accumulated damage from repeated penetrations.
That is a plausible biological explanation for a persistent shallow deficit,
but it is not yet separable from the current pipeline effects.

A future comparison should report raw-voltage, compact-event, unit-quality and
unit-continuity metrics by anatomically anchored depth. It should stratify each
depth by quiet versus high-motion epochs and include local rigid displacement,
nonrigid residual or displacement gradient, and motion-estimator support.
Possible interpretations should be kept distinct:

- a depth-stationary shallow deficit that persists in low-motion periods would
  be consistent with accumulated penetration damage, although not proof of it;
- a deficit that changes with local displacement, gradient or waveform drift
  would favor depth-dependent mechanical stress or motion fragmentation; and
- polarity, saturation or common-mode structure aligned to electrical banks or
  channel geometry rather than anatomy would favor an acquisition or
  conditioning mechanism.

This analysis is deliberately deferred until conditioning, preprocessing,
artifact handling and the final motion strategy are locked. Until then, depth
must not become another pipeline-optimization target. Preserve the full raw
depth coverage, channel/electrical-bank mapping, time origins and anatomical or
receptive-field anchors needed to perform the comparison later.

## Methodological risks and validation guardrails

The existing event cohort and short windows are useful discovery and regression
tests, but they are not independent ground truth. The same 200 reviewed events,
selected partly by their relationship to the patched sort, have been reused to
choose among claim-mask, conditioning, interpolation and motion variants. The
good, neutral and pathological windows have likewise influenced multiple
pipeline decisions. Repeated improvement on these endpoints could therefore
reflect specialization to the discovery cohort rather than recovery of more
stable neurons.

The remaining validation must address the following risks:

1. **Holdout leakage.** Freeze the existing events and named windows as the
   discovery set. Before further finalist tuning, draw and seal untouched
   session-wide validation windows across imec0 and imec1, quiet and high-motion
   epochs, depth, polarity and amplitude. Do not use the holdout to select
   thresholds or revise gates.
2. **Negative/high-amplitude selection.** The original six-sigma detector
   preferentially samples large negative peaks despite imec1's positive-event
   excess. Add polarity-balanced and amplitude-stratified candidates so that
   low-amplitude, positive-dominant and broader waveforms are not invisible to
   the rescue endpoint.
3. **Coincidence is not identity.** A sorted spike within 0.5 ms and 100 um can
   count as recovered by chance, especially in dense or duplicate-heavy sorts.
   Keep the jitter control, but require compatible local multichannel waveform
   or template shape and reduction of the event-centered residual for the
   strongest neural-recovery claims.
4. **Review labels are descriptive.** The five-channel review overcalled some
   probe-wide events, and no unmatched event in the pathological window passed
   the conservative automatic neural gate. Do not treat the visual labels or
   Kilosort labels as biological ground truth.
5. **Blanking can inflate apparent recovery.** The 500 microvolt point blanker
   creates ringing and false peaks. Stratify recovery by temporal and spatial
   distance from the raw saturation sidecar, and verify waveform identity before
   crediting blanker-proximal detections as rescued spikes.
6. **No end-to-end injected truth set exists.** Inject realistic multichannel
   Luke templates into genuine raw background before conditioning, spanning
   amplitude, polarity, depth, collision timing, saturation proximity and
   controlled drift. Measure recall, localization error, duplicate count and
   residual reduction through each finalist branch.
7. **The positive-polarity bands may reflect acquisition architecture.** Map
   positive-event and saturation burden against ADC or electrical bank,
   multiplexing phase, physical and electrical channel order, reference path,
   gain metadata and headstage/probe configuration before assigning the pattern
   to biology or tissue condition.
8. **Original-stream integrity remains separate from cache integrity.** Audit
   timestamps and sample continuity, dropped or repeated blocks, sign and gain
   metadata, synchronization, channel maps and time-origin conventions. The
   previously corrected recording-start offset demonstrates that coordinate
   errors can silently select the wrong data interval.
9. **Motion evidence needs an independent anchor.** DREDGE and Kilosort both use
   signals affected by the same conditioning and shared transients. Where
   available, compare inferred motion with LFP or anatomical landmarks,
   behavior/video or mechanical records, and stable unit families measured
   without voltage resampling. Cross-probe agreement alone does not distinguish
   tissue movement from a shared electrical artifact.
10. **Sorter-derived QC is not independent validation.** KS-good labels,
    contamination and near-zero-lag coincidence remain useful guardrails, but a
    finalist should also pass blinded unit review, waveform and amplitude
    stability, presence and isolation measures, residual analysis, and pairwise
    template/CCG checks. Near-synchronous firing should not automatically be
    classified as duplicate peeling.
11. **Luke--Yates matching remains limited.** In addition to the deferred depth
    analysis, match behavior or state, insertion location, probe and reference
    configuration, rig, duration and quality criteria before making a biological
    yield claim.

The three highest-value additions after the currently running materialization
and segment jobs are: (1) seal the untouched polarity-balanced validation
cohort; (2) build the injected-template end-to-end benchmark; and (3) audit raw
integrity and positive-event structure against electrical-bank, channel-map,
reference and saturation metadata. These should gate promotion of a finalist,
not interrupt or retroactively redefine the current data-writing jobs.

The plan-only, synthetic and metadata-first scaffolds for these three additions
are now implemented. Their safety boundaries, commands and initial acquisition
metadata results are recorded in
[`luke_validation_scaffolds.md`](luke_validation_scaffolds.md). The prospective
holdout event draw and discovery-only CPU injection pilot have now completed;
bulk hashing and any holdout sorter runs remain deferred.

## Known, unknown and likely fixability

| Question | Current evidence | Interpretation |
|---|---|---|
| Is Luke simply low signal? | No; large raw events are abundant. | A pipeline rescue remains plausible. |
| Is channel 191 the whole problem? | No; interpolation helps but distributed abnormalities remain. | Locally fixable, not sufficient. |
| Is 500 microvolt blanking harmless? | No; it creates ringing. | Retain provisionally and mask artifacts downstream. |
| Can blanking simply be removed? | No under current tests; reviewed recovery collapses. | Requires a better replacement, not deletion. |
| Is Kilosort batch rejection suitable? | No tested threshold is both safe and effective. | Disable it. |
| Is the claim mask suitable? | No nonzero tested setting passed recovery gates. | Disable it. |
| Is current external DREDGE correction suitable? | No; it expands detections and worsens quality. | Fixable implementation/parameter question remains. |
| Is motion real? | Yes; estimates and cross-probe behavior support it. | Motion estimation remains unfinished; voltage warping is not yet authorized. |
| Is preprocessing still the leading search axis? | No; rescue and pinned AIND converge on broadly similar bounded KS4 outcomes, and rescue generalized across both probes. | Freeze rescue; retain AIND as a comparator and reopen only for a specific demonstrated failure. |
| Does Luke approach Yates after repair? | Unknown. | Requires a stable full-duration result and fair matched claims. |

## Baseline materialization complete

The active baseline is a correctly sourced, full-duration 96-channel depth
strip spanning physical channels 176--271 and approximately 10,473.6 seconds.
It was regenerated from the original raw recording with the provisional
conditioning chain. The binary size is 60,327,186,048 bytes
(approximately 56.2 GiB).

An earlier same-size strip was quarantined because its provenance pointed to an
incomplete, zero-filled full-session cache. File size alone is therefore not an
acceptance criterion. The new strip passed the initial source-provenance,
exact-size and random-chunk population checks. Seven one-second samples spanning
0--99% of the duration were 99.04--99.12% nonzero. Across those samples,
per-channel standard deviations were 36.97--38.78 stored counts, no channel had
a standard deviation below one count, and interpolated channel 191 was populated
normally. The reproducible receipt at
`testing/outputs/luke_depth_strip_integrity_audit/receipt.json` passed every
prespecified check.

The 96-channel strip is a longitudinal diagnostic, not a substitute for a
full-probe finalist. Its internal Kilosort CAR differs from a 384-channel run.
The upstream
conditioning order has now been measured: across matched 10-second good,
neutral and pathological windows, conditioning the full 384 channels before
slicing was sample-for-sample identical to conditioning channels 176--271
after slicing, including interpolated channel 191. The existing strip therefore
does not need rebuilding for this order-of-operations concern.

A bounded 600-second depth-strip calibration spanning 7,800--8,400 seconds
completed in 147 seconds. It produced 142,878 spikes and 68 units, with stable
60-second spike rates (CV 0.096) and no detection explosion. It recovered 18 of
21 discovery-cohort neural unmatched events, but that recovery must not be
treated as independent validation under the holdout guardrail above. Linear
scaling estimates approximately 43 minutes for the full strip, with the caveat
that Kilosort has fixed and data-dependent stages.

The full 10,473.6-second no-motion strip then completed in 1,398.73 seconds
(23.3 minutes), faster than the calibration projection. It produced 3,269,181
final spikes, 125 units and 32 Kilosort-good units. Median contamination was
14.4%, median unit refractory violations were 0.60%, and discovery-cohort
in-depth neural-unmatched recovery was 90.9% versus 7.8% under time jitter.
Median unit activity covered 97.1% of 300-second bins, 118 of 125 units were
active in more than 75% of bins, and the median apparent lifetime was 10,414.7
seconds. These are encouraging continuity indicators, not proof of unit
identity: near-coincidence excess remained 0.094, 9.98% of spikes localized
within 40 micrometers of a strip boundary, and spike-rate CV across time bins
was 0.246. Those effects must be localized in time, depth, waveform and residual
space before motion handling or biological claims are promoted.

The follow-up localization changes the interpretation of those aggregate
burdens. Five units account for 98.3% of all boundary-localized spikes, and only
nine of 125 units have more than half their spikes within 40 micrometers of a
strip edge. This is a concentrated boundary/template problem rather than a
probe-wide failure of the baseline conditioning chain. Median unit depth
excursion was 32.2 micrometers (90th percentile 53.7 micrometers). Median
first-to-last PC-feature cosine was 0.953, and seven units fell below 0.8 and
require targeted waveform-family review. This corrected calculation uses
Kilosort's `spike_detection_templates.npy` assignments to select the dominant
detection-template subset within each cluster; `spike_templates.npy` contains
cluster IDs in this Kilosort 4 export and is not a detection-template record.
These are continuity diagnostics, not unit-identity proof.

Ninety-four cross-unit pairs passed a deliberately broad coincidence screen.
Only two survived direct CCG, template-similarity and merged-refractory checks
as strong duplicate hypotheses. Reconstructed Kilosort-conditioned waveforms,
transformed into the exported-template space using the saved inverse whitening
matrix, did not provide sufficient residual support to merge either pair. Both
remain manual-review candidates; no unit was merged or deleted. This materially
narrows the duplicate concern but does not explain all excess coincidence.

Across 300-second bins, rigid-motion excursion correlated positively with spike
rate (Spearman 0.438) and negatively with median amplitude (-0.694), consistent
with a real motion interaction worth testing. In contrast, boundary fraction
was essentially unrelated to rigid excursion (-0.051), and coincidence excess
was lower rather than higher in high-excursion bins (-0.784). These descriptive
correlations do not validate a motion field, but they show that motion alone is
not a sufficient explanation for the current boundary or coincidence burdens.

Six prospective 120-second windows per probe are now sealed across imec0 and
imec1, with paired high-motion and relative-quiet epochs in each session third.
The version-2 manifest SHA-256 is
`01643baa20fd9ee4905a9bd6e9282ab25e1365dd4899bd69244ea63a4a7fcc9b`.
Selection used only pre-existing motion coordinates, excluded all discovery
intervals by at least 300 seconds, and did not inspect sorter output. The
sealed raw-voltage protocol subsequently selected four SHA-ranked events in
every probe/window by depth-third, polarity and fixed amplitude cell. All 216
cells met quota, yielding 864 events with zero deficit and no borrowing. The
CUDA local-reference implementation was accepted only after a sealed six-chunk
NumPy comparison produced 0.0 microvolt differences and identical candidate
strata and selected identities. Reviewers receive opaque candidate IDs only;
strata and coordinates remain in separate internal files. Because the
no-motion imec1 baseline already spans the session, this remains prospective
validation for future artifact and motion branches, not pristine retrospective
validation of that existing baseline.

One deterministic event from every sealed cell (216 total) was then used for a
sorter-free motion pilot, leaving 648 events untouched for confirmation. The
historical p=1, zero-border warp again attenuated voltage: median amplitude
ratio was 0.885 and 31.5% of events lost more than 20%. Official p=2
extrapolation improved the nonrigid result to a 0.984 median amplitude ratio
and 0.965 median waveform correlation, but retained a 7.4% low-correlation
tail. Full rigid p=2 preserved voltage better but did not pass sorter-quality
tradeoffs in the legacy pathological diagnostic.

A 0.25-times rigid p=2/extrapolating field was effectively neutral on the
balanced voltage cohort: median and tenth-percentile amplitude ratios were
1.0, tenth-percentile waveform correlation was 0.9986, and no event crossed
the 20% amplitude, 0.8-correlation or 20-um depth-error guardrails. In the
corrected single-pass shared-window sort, however, this branch did not dominate
no motion. Neural-unmatched recovery was 97.1% for both. The rigid branch
slightly improved all-reviewed recovery, contamination and coincidence, but
produced 161 versus 168 Kilosort-good units and slightly higher refractory
violations. No motion therefore remains the baseline; 0.25-times rigid p=2 is
the sole motion candidate eligible for a longer confirmatory depth-strip pilot.

That longer 96-channel, 10,473.6-second pilot is now complete, and it changes
the advancement decision. The saved strip reset its clock to zero while the
DREDGE bins retained acquisition-absolute seconds, so the implementation first
rebased the field to 0.5--10,473.5 seconds; the earlier 120-second snippet
source retained its correct absolute clock and is unaffected. Materialization
took 6.2 minutes, passed exact length/geometry/dtype and sampled-voltage checks,
and the identical claim-off, internal-motion-disabled sort took 22.6 minutes.

Across the whole strip, rigid-0.25 superficially improved recovery (90.9% to
93.2%), Kilosort-good count (32 to 33), median contamination (14.4% to 12.5%),
coincidence excess (0.0940 to 0.0860) and median refractory violations (0.00602
to 0.00569). It nevertheless failed the prespecified boundary gate: spikes
within 40 um of a retained edge doubled from 10.0% to 20.1%, with three very
large Kilosort-good units anchored at artificial strip boundaries. Edge burden
was unrelated to time-varying rigid excursion (Spearman 0.009), implicating
spatial boundary support rather than motion magnitude.

An 80-um interior sensitivity analysis removed the apparent advantage.
Recovery was unchanged at 90.9%, while contamination increased from 17.5% to
26.4%, coincidence excess from 0.108 to 0.117 and refractory violations from
0.00615 to 0.00956. Because boundary units can affect global template learning,
this sensitivity analysis cannot repair the candidate after the fact. The
already-cropped/no-halo rigid branch is rejected. This does **not** reject
voltage resampling in general: a fair strip pilot must condition and resample
with real-voltage channels beyond both retained edges, then crop. Three bounded
attempts at a 16-channel-per-side halo exposed a 35--60-minute conditioning
bottleneck and were stopped; partial sparse files were quarantined. No motion
therefore remains the long-duration baseline, and the halo-supported question
is deferred rather than promoted to a full sort.

The cheaper inverse construction has now resolved that question for the
current Kilosort 4 pipeline. Both cached 96-channel voltages were cropped
inward to the identical 80 channels *before* sorting, leaving 80 um (four
kriging scales) of real-voltage support on each side of every retained channel
during the earlier correction. Exact byte counts and sample-for-sample source
equality passed at five session positions. The paired full-duration sorts used
claim-off, internal-motion-disabled KS4 with identical settings.

The dramatic artificial-edge excess disappeared: edge fractions were 10.0%
without motion and 10.3% with rigid-0.25. Recovery was identical at 90.9%, but
rigid-0.25 produced 100 versus 110 units, 25 versus 26 Kilosort-good units,
23.25% versus 16.7% median contamination and 0.01130 versus 0.00702 median
refractory violations. Coincidence excess improved slightly (0.1153 to 0.1130)
and depth excursion contracted modestly, but median first/last PC cosine fell
from 0.954 to 0.928 and low-cosine units increased from two to four. The rigid
branch passed only two of six prespecified gates. It is therefore rejected for
the current KS4 pipeline. This conclusion is sorter-specific and does not
invalidate the sorter-free finding that p=2 rigid interpolation preserves
isolated voltage waveforms.

A discovery-only CPU injection pilot has also completed. Ten raw templates
were injected into paired 0.5-second quiet and pathological backgrounds and
traced through the current conditioning stages without sorting or motion
estimation. The current CAR/high-pass/materialized stage retained median peak
amplitude 0.699 and median waveform cosine 0.674 relative to the
phase-corrected injected delta; median absolute peak-channel displacement was
four indices. These are stage-distortion diagnostics, not recovery rates. More
decisively, independent events carrying the same discovery cluster/window
label had median raw multichannel cosine only 0.047, with zero of ten at or
above 0.8. The existing manual/Kilosort assignments therefore cannot serve as
unit-identity ground truth for an end-to-end injection benchmark. A new donor
set needs independent waveform-family qualification and positive/morphology
coverage before paired sorter runs are justified.

The artifact-sidecar generator and a real neutral-window pilot are complete.
A direct full-session phase-domain scan benchmarked at roughly five hours, so
the full sidecar is deferred until it is needed for post-sort artifact-sensitive
claims. This does not change the sorter voltage and must not be interpreted as
permission to credit blanker-proximal detections without the sidecar.

## Historical prioritized corrections and tests (superseded)

This table records the sequence that produced the current evidence. Its open
P0/P1 labels no longer define the active work order; the 2026-08-31 strategic
update and stage-local strategy document do.

| Priority | Test | Purpose | Advancement gate |
|---|---|---|---|
| P0 complete | Validate materialized strip provenance and completeness | Prevent another preallocated, partial or zero-filled input from entering a sort | Passed: exact size, correct source/preprocessing graph, populated representative chunks and no stuck channels |
| P0 complete | Compare full-384-then-slice with slice-then-condition | Quantify phase, interpolation and reference order effects | Passed: sample-for-sample identical in all three matched windows, including channel 191 |
| P0 | Build raw saturation/artifact sidecar | Separate voltage conditioning from claim exclusion | Segment implementation passed; full scan deferred after a measured approximately five-hour cost. Reproducible intervals/components and a documented temporal margin remain required for finalists |
| P0 | Sweep downstream artifact exclusion margins | Remove ringing-driven claims without suppressing neural recovery | Reduce artifact-proximal false events while preserving reviewed-neural recovery |
| P0 complete | Freeze and draw prospective holdout | Prevent further tuning on the reused reviewed-event cohort | Version-2 imec0/imec1 windows and method sealed before raw access; 864 opaque-ID events fill all 216 depth/polarity/amplitude cells with zero deficit; NumPy/CUDA equivalence passed |
| P0 complete | Sort the no-motion full-duration strip | Establish the baseline that motion candidates must beat | Completed in 23.3 minutes with strong apparent continuity and recovery; boundary and coincidence burdens require localization |
| P1 active | Audit longitudinal unit-family continuity | Distinguish motion fragmentation from loss of detectable voltage | Aggregate PC continuity measured; review the seven low-cosine units and motion-crossing waveform families without merge-induced refractory failures |
| P1 | Qualify motion estimates without resampling | Test field support and reproducibility independently | Supported time-depth bins, cross-method agreement where signal exists, controlled edges |
| P1 complete for implementation | Validate voltage-resampling implementation | Test interpolation order, geometry, dtype, gain and border policy | Historical p=1/zero-border rejected; p=2/extrapolation validated; 0.25-times rigid best preserves the sealed voltage cohort |
| P1 complete for rigid-0.25/KS4 | Gate motion candidates on matched snippets and depth strip | Add one motion factor at a time | A supported 80-channel inward crop removed the artificial-edge confound but rigid-0.25 still passed only 2/6 gates; reject it for current KS4 and retain no motion |
| P1 active | Audit templates, collisions and residuals | Identify artifact-seeded templates, duplicate peeling and genuine misses | Broad screen reduced to two manual-review pairs; residual evidence did not authorize merges. Continue event-centered residual and collision tests on targeted units |
| P1 active | Inject known templates before conditioning | Obtain end-to-end recall, localization and duplication ground truth | CPU stage adapter ran, exposing approximately 30% median peak attenuation and weak donor identity labels; independently qualify polarity/morphology-balanced donors before any paired sorter benchmark |
| P1 | Audit original-stream and electrical-bank integrity | Test acquisition continuity and localize the positive-polarity bands | Consistent timestamps, samples, gain/sign, channel map and reference metadata; burden localized or excluded by acquisition-architecture controls |
| P1 | Revisit distributed channel artifacts | Separate channel 191 from row/common-mode or acquisition effects | Stable results with synthetic channel excluded from metrics and geometry-aware controls |
| P1 partial | Test positive-polarity excess across sessions and rigs | Determine whether the excess is session-, probe- or setup-specific | 2025-08-04 vs 2025-08-05 recurrence (both probes) supports an imec1-stream-fixed cause over a session-specific one; still needs a third session, another probe pair and another rig |
| P2 | Depth-resolved Luke--Yates comparison after pipeline lock | Separate shallow penetration damage, anatomical sampling and depth-dependent motion | Anatomically anchored depth profiles stratified by motion state, using Yates only over the matched shallow interval |
| P2 | Revisit claim mask only after upstream lock | Explore duplicate control at the correct stage | A nondominated recovery/duplicate setting; otherwise remain off |
| P2 | Run at most two full-probe finalists | Estimate attainable full-session quality | All snippet and depth-strip gates passed first |

## Motion reintroduction ladder

The no-motion strip has now been scored and localized, the holdout drawn, and
the resampler audited both without sorting and in a bounded long-duration sort.
The historical external warp, Kilosort internal rigid correction, full-gain
rigid correction and rigid-0.25 applied after depth cropping remain rejected.
The supported inward-crop replication additionally rejects rigid-0.25 for the
current KS4 pipeline even when retained voltages have real spatial support. The
sequence is now:

1. no external motion and no internal motion;
2. Kilosort internal rigid correction (completed and rejected here);
3. conservative rigid external registration with validated p=2 interpolation,
   0.25 displacement gain and real-voltage boundary support (completed and
   rejected for current KS4; never resample an already cropped finalist);
4. nonrigid registration only in time-depth regions with adequate estimator
   support;
5. an unwarped alternative that links temporally fragmented clusters into unit
   families or allows templates to follow the tissue.

Each candidate should first pass the good, neutral and pathological snippets,
then the full-duration depth strip. Yield alone is not a success criterion.
Required endpoints include reviewed-neural recovery, Kilosort-good units,
contamination, refractory violations, near-zero-lag cross-unit coincidence,
residual energy, boundary accumulation and longitudinal continuity.

## Principal contamination hypotheses

The hypotheses now worth carrying forward are:

1. **Distributed saturation/common-mode interaction.** A recurring population
   of large positive and shared events forces blanking in nearly every batch,
   and filtering the resulting discontinuities creates false peaks.
2. **Bad-channel/whitening interaction.** Channel 191 is locally repairable,
   but exact interpolation or including it in metrics can destabilize covariance,
   localization and template learning.
3. **Resampling implementation, clock, boundary and scale pathology.** The historical p=1,
   zero-border warp measurably attenuates sealed raw events. Official p=2 with
   extrapolation fixes most implementation loss, but full-gain and nonrigid
   fields still harm the sorter or retain a distortion tail. Saved recordings
   can also reset time to zero while motion bins remain acquisition-absolute,
   and resampling after depth cropping creates unsupported spatial boundaries.
   The evidence now implicates displacement scale, nonrigid field structure,
   clock alignment and boundary support in addition to interpolation defaults.
4. **Motion-driven fragmentation.** Genuine tissue motion may split otherwise
   continuous waveform families over a 2.9-hour recording even when short
   windows sort well. The amplitude and rate correlations with rigid excursion
   strengthen this interaction hypothesis, but the localized edge burden and
   inverse coincidence correlation show that it is not a complete explanation.
5. **Template/collision over-peeling.** Shared transients, ringing and collisions
   may seed redundant templates or repeated assignment, producing high
   coincidence and spike counts without better unit quality.
6. **Overaggressive duplicate control.** The tested cross-peel claim mask removes
   real reviewed events along with duplicates.
7. **Reference- and polarity-dependent acquisition artifact.** The imec1
   positive excess may reflect the session, probe stream or recording setup and
   must be tested across sessions before any biological density claim. A first
   cross-session check (2025-08-04 versus 2025-08-05, see the cross-session
   recurrence subsection above) supports the probe-stream explanation over a
   session-specific one, but is only two sessions on one rig.

## Definition of a credible rescue

A rescued pipeline must preserve reviewed neural events through conditioning,
avoid artifact-proximal and near-synchronous duplicate inflation, maintain
plausible refractory behavior and residuals, and keep unit families continuous
through motion. Only after those gates pass should Luke be compared with Yates
on matched depth, reference, voltage threshold, duration, behavior and quality
criteria.

## Full-probe rescue result (2026-08-30)

The frozen full-probe rescue completed and is a strong provisional success.
It produced 43,669,711 final spikes, 583 units and 216 KS-good units over
10,473.6 s. This is 65 more KS-good units than the best prior full-probe result
(+43.0%), and 70 more than `pipeline_an5` (+47.9%) despite 6.7% fewer assigned
spikes. Median KS-good contamination is 3.55%, median 1.5 ms refractory
violation fraction is 0.125%, median 300 s presence is 100%, and median
lifetime is 10,430.6 s. The median holdout-window cross-unit coincidence excess
is 0.077, below the prior no-motion strip's 0.094.

The sealed automatic imec1 raw-event holdout recovers 74.5% overall versus a
22.1% jitter-null mean, while the reused reviewed-neural cohort recovers 93.5%.
The important unresolved localization is the middle depth third, where sealed
automatic recovery is only 47.2%. Eleven nearby similar KS-good pairs also
remain for targeted duplicate review. These checks can be done on the accepted
sort and do not justify another full run.

The result establishes a materially better Luke operating regime, but not
Yates parity. Although depth-normalized KS-good yield is higher than the
available sampled Yates comparator, anatomy, duration, preprocessing and the
quality of that Yates sort are not matched. The complete interpretation and
tables are in `docs/luke_20250804_full_probe_rescue_result.md` and
`testing/outputs/luke_full_probe_rescue_diagnostics/`.

The updated near-term claim is therefore: **the rescue pipeline materially
improves full-session Luke yield and internal quality, and is ready for targeted
curation and multi-session replication; matched Yates quality remains
unproven.** The next decisive evidence is review of the localized holdout and
similar-template exceptions, followed by replication rather than another
parameter sweep. A different motion candidate or sorter should return only
under a separately prespecified comparison, and must improve these endpoints
without boundary accumulation, detection expansion or refractory-quality loss.

## Evidence inventory

- `docs/luke_pipeline_stage_local_validation_strategy.md`
- `docs/luke_20250804_aind_downstream_bounded_result.md`
- `testing/outputs/luke_aind_downstream_bounded_endpoint_review/README.md`
- `docs/luke_20250804_imec0_rescue_control_plan.md`
- `testing/outputs/luke_full_probe_rescue_diagnostics_imec0_legacy/summary.json`
- `testing/outputs/luke_full_probe_rescue_diagnostics_imec0_legacy/acceptance_criteria.json`
- `testing/outputs/luke_full_probe_rescue_diagnostics/summary.json`
- `docs/luke_20250804_full_probe_rescue_result.md`
- `testing/outputs/luke_conditioning_final_decision/decision.json`
- `testing/outputs/luke_conditioning_stage_audit/`
- `testing/outputs/luke_saturation_policy_audit/`
- `testing/outputs/luke_kilosort_artifact_threshold_audit/threshold_summary.csv`
- `testing/outputs/luke_conditioning_order_audit/decision.json`
- `testing/outputs/luke_depth_strip_integrity_audit/receipt.json`
- `testing/outputs/luke_full_strip_diagnostic_audit/summary.json`
- `testing/outputs/luke_full_strip_pair_ccg_audit/summary.json`
- `testing/outputs/luke_full_strip_pair_residual_audit/summary.json`
- `testing/outputs/luke_prospective_holdout/seal_v2.json`
- `testing/outputs/luke_prospective_holdout/event_draw_summary_v2.json`
- `testing/outputs/luke_prospective_holdout/holdout_output_roles_v2.json`
- `testing/outputs/luke_prospective_holdout/backend_equivalence_result_v2.json`
- `testing/outputs/luke_targeted_unit_audit/summary.json`
- `testing/outputs/luke_holdout_resampling_audit/decision.json`
- `testing/outputs/luke_holdout_resampling_audit_rigid025/decision.json`
- `testing/outputs/luke_motion_reintroduction_decision/decision.json`
- `testing/outputs/luke_rigid025_depth_strip/decision.json`
- `testing/outputs/luke_rigid025_depth_strip/comparison.json`
- `testing/outputs/luke_rigid025_depth_strip/interior_sensitivity.json`
- `testing/outputs/luke_inward_crop_pair/decision.json`
- `testing/outputs/luke_inward_crop_pair/comparison.json`
- `testing/outputs/luke_20250805_polarity_recurrence_audit/decision.json`
- `docs/luke_20250804_upstream_ablation_report.md`
- `docs/luke_yates_raw_voltage_audit_notes.md`
- `docs/luke_20250804_leading_hypotheses.md`
- `docs/luke_20250804_presort_motion_handoff.md`
- `docs/luke_validation_scaffolds.md`
