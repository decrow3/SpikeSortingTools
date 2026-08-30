# Luke--Yates matched raw-voltage audit: paused handoff

## Current status

The first complete raw audit sampled ten 2 s batches from each of seven
recording/window cohorts:

- Luke imec0: pathological, shared-template, and session-wide;
- Luke imec1: pathological, shared-template, and session-wide;
- Yates: session-wide.

Both recordings were converted to microvolts and passed through the same
in-memory 300--6000 Hz Butterworth filter. Events were physically deduplicated
within 0.5 ms and 100 µm. The audit reports both per-channel robust-sigma
thresholds and fixed 50/75/100 µV thresholds, with event counts normalized by
sampled physical depth.

The completed output predates two code corrections to the exploratory spatial
footprint calculation. Therefore the event-rate and channel-noise outputs are
usable, but the saved footprint summaries and current PNG must not be used as
evidence. A shank-median reference sensitivity run was started after those
corrections and intentionally interrupted at the user's request; it writes
only on successful completion, so there are no partial shank-median results.

## What is established

### Luke does not have a simple raw-amplitude event shortage

Before referencing, Luke contains substantially more large high-frequency
extrema than Yates. In the pathological windows, the median negative-event
rates after the common bandpass were:

| Condition | 6σ negative events/mm/s | 75 µV negative events/mm/s | Median bandpass σ |
|---|---:|---:|---:|
| Luke imec0 pathological | 608 | 2,538 | 37.8 µV |
| Luke imec1 pathological | 1,096 | 1,303 | 31.0 µV |
| Yates session-wide | 159 | 647 | 17.4 µV |

This direction is the opposite of the previously observed Kilosort-input
negative-event deficit. That deficit therefore does not pre-exist as a simple
absence of large raw AP extrema.

### Luke has a large shared high-frequency component

A 100 µm local median reference reduces median bandpass noise from
approximately 38 to 7.2 µV on Luke imec0 and from 31 to 6.3 µV on imec1.
Yates falls from approximately 17.4 to 5.0 µV. The much larger reduction on
Luke shows that a substantial fraction of its raw high-frequency voltage is
shared across nearby contacts rather than independent channel noise.

This makes event density strongly reference-dependent and warns against
interpreting either unreferenced extrema or a single fixed-sigma detector as
neural spike counts.

### After local referencing, the residual is probe- and polarity-specific

At a fixed 75 µV threshold after the 100 µm local median reference:

| Cohort | Negative events/mm/s | Positive events/mm/s |
|---|---:|---:|
| Luke imec0 pathological | 242 | 246 |
| Luke imec0, session-wide median | 205 | 242 |
| Luke imec1 pathological | 110 | 445 |
| Luke imec1, session-wide median | 102 | 471 |
| Yates session-wide | 293 | 69 |

Across the independent shared window, imec0 has 252 negative and 365 positive
events/mm/s; imec1 has 132 negative and 699 positive events/mm/s. Thus imec0
approaches Yates's fixed-amplitude negative-event density, whereas imec1 has a
persistent negative deficit paired with a six- to ten-fold positive excess.

This argues against one global Luke sensitivity adjustment. The unresolved
imec1 problem is more consistent with a polarity/reference/morphology or
shared-signal issue than with a simple lack of extracellular events.

### Cross-session recurrence (2025-08-05)

A same-method rerun on 2025-08-05, the next recording session on the same
probes, reproduces this asymmetry independently: the fixed-75 µV,
local-referenced positive:negative event-rate ratio is 2.80 for imec1 versus
1.30 for imec0, compared with 4.31 versus 1.15 on 2025-08-04. The per-channel
polarity profile correlates moderately across days for imec1 (Spearman
r=0.42, p=7e-18, n=384 channels) but not for imec0 (r=-0.28), and imec0/imec1
profiles do not track each other consistently within a day either. This favors
a stream-fixed acquisition cause (imec1 probe, headstage, cable or reference
path) over a 2025-08-04-specific biological explanation, though the moderate
correlation means it is a real signal on top of session-specific variation,
not a deterministic fingerprint. See the cross-session recurrence subsection
of `luke_20250804_rescue_status_and_test_plan.md` and
`testing/luke_20250805_polarity_recurrence_audit.py`.

### A fixed-sigma comparison is not voltage-equivalent

After the 100 µm local reference, median channel noise remains approximately
7.0--7.5 µV on imec0, 6.3--6.6 µV on imec1, and 5.0 µV on Yates. A 6σ
threshold is therefore roughly 42--45 µV on imec0, 38--40 µV on imec1,
and 30 µV on Yates. This difference materially inflates Yates's 6σ event
count. Both sigma-normalized and fixed-microvolt results are required.

## What is not yet established

- The 100 µm local reference is not probe-neutral: its neighborhood contains
  many more contacts on Luke's dense Neuropixels layout than on a Yates shank.
- The interrupted shank-median control has no completed output. It is needed to
  bracket the reference dependence.
- The saved spatial-footprint summary used a full-shank energy denominator and
  an overly broad temporal maximum. Those choices unfairly penalize Luke's
  384-contact shank. The code now uses a ±500 µm denominator and a ±3-sample
  temporal window, but those corrected metrics have not been rerun.
- Normalized shank depth is not matched cortical layer. Histology or another
  layer anchor is still needed for a biological density comparison.
- The shallow cortical portions are the closest Luke--Yates anatomical match.
  Luke's longer probes extended into deeper V1 banks representing more
  peripheral visual-field locations, while its shallow cortex may also have
  accumulated damage from repeated penetrations. A depth-resolved biological
  comparison must therefore be restricted to anatomically matched support and
  must stratify by depth-dependent rigid/nonrigid motion. This comparison is
  deferred until conditioning, preprocessing and motion handling are fixed.
- The metadata describe Yates `recording.dat` as the raw 64-channel int16
  recording in geometry order, but a direct sample-level reconciliation to the
  original Open Ephys `.continuous` files remains a useful provenance check.

## Rescue implications at the pause point

1. Do not lower Luke's global detection threshold. Its unreferenced AP voltage
   already contains an excess of large extrema, so this would mainly increase
   correlated/artifactual detections.
2. Prioritize separation of compact neural voltage from Luke's shared
   high-frequency component.
3. Treat imec0 and imec1 separately. imec1's large positive excess and negative
   deficit are the sharper rescue target.
4. Preserve the current no-external-motion baseline while testing upstream
   reference/common-mode models; nonrigid voltage interpolation remains a
   verified causal failure.

## Resume sequence

1. Complete the shank-median reference sensitivity run.
2. Add an equal-neighbor-count reference alongside the fixed-radius and
   shank-wide controls.
3. Rerun the corrected event-footprint metrics and classify compact versus
   diffuse events within matched physical support.
4. Map noise, polarity, event burden, and compact-event yield by channel and
   normalized depth, with special attention to rows 191 and 216.
5. Test candidate shared-component removals on the fixed reviewed-event cohorts
   before sorting: robust local regression, leave-one-out common-mode
   subtraction, and low-rank common-component subtraction are the leading
   candidates.
6. Sort only candidates that improve compact-event SNR without attenuating the
   reviewed raw events; score them after identical merging.

## Reproducibility

- Script: `testing/luke_yates_raw_voltage_audit.py`
- Tests: `testing/test_luke_yates_raw_voltage_audit.py` (5 passing)
- Completed event output:
  `testing/outputs/luke_motion_candidate_results/raw_voltage_audit/raw_event_summary.csv`
- Completed channel output:
  `testing/outputs/luke_motion_candidate_results/raw_voltage_audit/raw_channel_summary.csv`
- The current PNG and footprint CSV are explicitly provisional and should not
  be cited until the corrected rerun completes.
- Cross-session recurrence script: `testing/luke_20250805_polarity_recurrence_audit.py`
  (no dedicated test file yet); outputs in
  `testing/outputs/luke_20250805_polarity_recurrence_audit/`.
