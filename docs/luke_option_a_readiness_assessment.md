# Option A readiness — can external registration support one bounded experiment?

Date: 2026-09-06. Status: **assessed on real artifacts. Answer: no.**
Three prerequisites block it; the smallest one that could be completed now has
been completed, and it is the measurement that establishes the other two.

Receipt: `/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_option_a_prerequisites/option_a_field_prerequisites.json`
Implementation: `testing/luke_option_a_field_prerequisites.py` (+ 13 fixtures).
Governing spec: [motion build instructions §5](luke_two_motion_pipeline_build_instructions.md).

## 1. Answer

**Option A cannot support a bounded candidate-versus-control run on Luke
2025-08-04 imec0 today.** The blocking prerequisite is the motion field itself,
not the runner and not the operator.

| prerequisite (§5.3 / §5.4) | status | measured |
|---|---|---|
| estimator implementation | ✅ available | `estimate_full_session_motion` + 4 accepted imec0 fields on disk |
| application operator | ✅ available | SpikeInterface 0.102.1 `InterpolateMotionRecording`, `load_motion_info` (in the `spikeinterface` env) |
| displacement in calibrated range (≤ 60 µm) | ✅ passes | 6.6 – 29.5 µm full session |
| acquisition ↔ recording time mapping | ✅ **now established** | **3057.677050340359 s**, two independent sources agreeing exactly |
| supported time/depth fraction (≥ 0.95) | ❌ **fails** | **0.917**; only **1.5%** of time bins are supported at every depth |
| estimated gain error (≤ 0.30) | ❌ **fails** | **≥ 0.80** implied by accepted-estimator disagreement |
| split-half reproducibility (≥ 0.80) | ❌ **never measured** | requires re-estimating on independent halves |
| `qualify_field` receipt | ❌ fails closed | no accepted estimator passes |
| `testing/luke_external_warp_pipeline.py` | ❌ missing | — |

`qualify_field` was already written to fail closed on missing evidence, and
`materialize_qualified_correction` refuses an unqualified field. Both behave
correctly. What did not exist was any *measurement* saying which limb blocks, so
"Option A is blocked" was readable only from the code. It is now a receipt.

## 2. The smallest required implementation task, completed

**The §5.3 field-qualification receipt.** It is required before any
materialization regardless of when Option A runs, it is the only thing that
turns three assumptions into numbers, and it is a prerequisite for the other
limbs — a per-window support fraction or a nominated interval cannot even be
computed without first establishing which clock the field is on.

Building the runner first would have been building a front end for a gate that
cannot pass. Attempting the gain calibration would have been the broad motion
investigation that is explicitly out of scope.

### 2.1 The time origin, and a live mis-declaration

Every accepted estimator's time axis runs **3058.7 → 13530.7 s** on a 10,473.55 s
recording. That is not a bug: SpikeInterface writes motion time bins in
*acquisition* time, while the peaks they were estimated from are indexed in
recording frames from zero (measured span 0.0 → 10473.55 s). The origin is
recoverable from two independent sources, which agree to the digit:

| source | value |
|---|---|
| SpikeGLX meta, `firstSample` 91729810 ÷ `imSampRate` 29999.835983263598 | 3057.677050340359 s |
| SpikeInterface `binary.json`, `t_starts[0]` | 3057.677050340359 s |

Prior window-metric work applied this correctly — recomputing
`candidate_window_motion_metrics_v2.csv` correlates 0.79 with the offset applied
and 0.08 without — so no earlier selection is invalidated. But nothing recorded
it, and one artifact on disk actively contradicts it:

> `rescue_pipeline_results_.../motion/dredge-rigid-sidecar/estimate.npz`
> declares `time_reference: "selected_recording_start"` while carrying times
> 3058.18 → 13531.18 s. **MISDECLARED.**

`pipeline.motion_coordinates.load_qualified_motion_field` validated that string
and trusted it. It now accepts an optional `recording_duration_s` and refuses a
field whose times fall outside the recording, so the declaration is checkable
rather than believed. The v2 runner passes the duration.

### 2.2 Support: the field covers more than it measures

Support is the fraction of the field's own (time, depth) domain with at least 5
detected peaks behind it, over 28,719,735 peaks on the dredge grid (10,474 × 40):

| | |
|---|---|
| support fraction | **0.917** (gate 0.95) |
| median peaks per cell | 42 |
| depth bins never supported | 0 |
| **boundary depth support** | **0.434** |
| **time bins supported at every depth** | **0.015** |

The last two are the substantive finding. The field is defined across the whole
probe, but its spatial boundaries have real data behind them less than half the
time, and only 1.5% of time bins are supported everywhere. `build_spikeinterface_motion`
already refuses a partially supported field rather than extrapolating one, so
this field would be refused at the point of application even if the other limbs
passed.

### 2.3 Scale: the gain is not known to within the gate

The four accepted estimators, over the same development windows on the same
recording clock:

| estimator | median 120 s rigid excursion |
|---|---|
| ks-motion | 4.06 µm |
| dredge-motion | 10.53 µm |
| medicine | 11.73 µm |
| decentralized-motion | 20.48 µm |

Max/min ratio **5.05**, implying a gain error of at least **0.80** against a
gate of 0.30. This is a lower bound, not a calibration: it says the gain is not
known to better than this, not which estimator is right. Decision 0013 already
carries the disagreement as an open quantification problem; this expresses it in
the units `qualify_field` gates on.

Closing it needs calibration against known motion. The D2b tolerance envelope is
retracted, so that is new calibration work — the broad investigation this task
was told not to open.

## 3. Where Option A would be tested, once unblocked

Nominated on motion coordinates only. No sorter output, no candidate result, and
no reference to cluster 37 — which was the previous target and is **not** where
the motion is.

Ranked by the **minimum** rigid excursion across all four accepted estimators,
so a window wins only if every estimator agrees there is motion there. 55
non-overlapping 120 s windows inside the contract's development windows
qualified.

| | interval | ks | dredge | medicine | decentralized | min |
|---|---|---:|---:|---:|---:|---:|
| **nominated** | **[7200, 7320] s** | 16.1 | 24.6 | 18.3 | 41.7 | **16.1 µm** |
| quietest considered | [5460, 5580] s | 2.0 | 4.4 | 11.1 | 6.9 | 2.0 µm |

The nominated window carries an order of magnitude more agreed motion than the
quietest, and sits inside development window [5760, 8700] s. For contrast, the
v2 case interval [6590, 6810] s was selected by amplitude dropout, not motion.
Nominating [7200, 7320] s rather than reusing it is the point: registration
should be tested where there is displacement to remove.

This nomination is recorded now so that it precedes any Option A result. It is
not a commitment to run — the field cannot qualify.

## 4. What is not being done, and why

- **No Option A candidate-versus-control run is prepared.** Its prerequisites
  are missing; preparing a contract for an experiment that cannot execute would
  repeat the v1 mistake of declaring an execution nobody had written.
- **No gain calibration is started.** It is the binding constraint and it is a
  broad motion investigation.
- **No split-half estimate is run.** It is a real bounded job (re-estimating on
  two independent halves of the saved peaks) but it cannot unblock Option A on
  its own while support and gain both fail.
- **The v2 replay is not promoted, and its 18 unvalidated merges are not carried
  forward.** They are preserved in
  [luke_v2_unvalidated_merge_candidates.md](luke_v2_unvalidated_merge_candidates.md).

## 5. If Option A is to be revived

In dependency order, all three needed:

1. **Restrict the field's domain to its support**, or accept a support gate that
   matches what the probe actually measures. A field applied where nothing was
   detected is extrapolation presented as correction.
2. **Split-half reproducibility** from the saved peaks — bounded, no voltage.
3. **Gain calibration** against known motion. This is the binding constraint and
   the only one that is not small. Until it exists, no field on this session can
   pass `qualify_field`, and that is the gate working as designed.

Any future Option A experiment must still carry amplitude completeness as its
endpoint with **adequate time coverage** (the v2 coverage requirement: below the
declared fraction of the eligible population it is `inconclusive`, never `pass`)
and **identity and contamination support** alongside it.

## Related records

- [Motion build instructions §5](luke_two_motion_pipeline_build_instructions.md)
- [Decision 0013 — imec0 has appreciable rigid motion](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md)
- [Decision 0002 — motion is estimated, never applied](decisions/0002-motion-is-estimated-never-applied.md)
- [Direct motion-scale audit](luke_20250804_direct_motion_scale_audit.md)
- [v2 result — the closed replay branch](luke_first_pipeline_candidate_v2_result.md)
