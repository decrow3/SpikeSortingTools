# Option A readiness — can external registration support one bounded experiment?

Date: 2026-09-06. Status: **assessed on real artifacts, then revised.**
**Answer: not on the whole probe — but yes on a frozen local domain, for a
development-only operational comparison.**

**Revision note.** The first version of this document concluded "no" outright
and reported an "implied gain error ≥ 0.80". Both are corrected below. Gain is
**unmeasured**, not measured-and-failing (§2.3). And the support criterion that
appeared to block was a *whole-probe, whole-session* threshold; assessed inside
the nominated interval it passes over a 1.9 mm interior (§3). The measurements
are unchanged and preserved — the conclusions drawn from them were wrong.

Receipt: `/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_option_a_prerequisites/option_a_field_prerequisites.json`
Implementation: `testing/luke_option_a_field_prerequisites.py` (+ 13 fixtures).
Governing spec: [motion build instructions §5](luke_two_motion_pipeline_build_instructions.md).

## 1. Answer

**A bounded, development-only Option A comparison can proceed on a frozen local
domain.** It cannot proceed as a *qualified* application on the whole probe, and
`qualify_field` still fails closed and is not waived.

The requirements split into two kinds, and they fail for different reasons:

- **Indispensable implementation integrity** — clock mapping, supported
  application, polarity, operator behaviour. If one of these fails the operation
  is wrong however accurate the estimate is. **All four are now verified.**
- **Provisional scientific criteria** — the numeric gates in `FieldGate`,
  inherited from the retracted D2b-1 envelope. Two are unmet or unevaluable and
  their replacement does not exist.

Proceeding on the first kind inside a frozen domain, with the second kind
recorded as unmet rather than waived, is an explicit versioned departure. See
the [development comparison prespec](luke_option_a_development_comparison_prespec.md).

| prerequisite (§5.3 / §5.4) | status | measured |
|---|---|---|
| **indispensable** — acquisition ↔ recording clock mapping | ✅ verified | 3057.677050340359 s, two independent sources agreeing exactly |
| **indispensable** — supported application domain | ✅ verified locally | frozen band draws only from bins with ≥0.95 local support |
| **indispensable** — displacement polarity | ✅ verified | zero-motion identity; +60 µm → −60.0000 µm shift, symmetric |
| **indispensable** — operator behaviour | ✅ verified | SI 0.102.1 `InterpolateMotionRecording`, exact to 3e-7 |
| estimator implementation | ✅ available | `estimate_full_session_motion` + 4 accepted imec0 fields on disk |
| application operator | ✅ available | SpikeInterface 0.102.1 `InterpolateMotionRecording`, `load_motion_info` (in the `spikeinterface` env) |
| displacement in calibrated range (≤ 60 µm) | ✅ passes | 6.6 – 29.5 µm full session |
| *provisional* — displacement ≤ 60 µm | ✅ passes locally | 39.0 µm for the selected field in the window |
| *provisional* — supported fraction ≥ 0.95, whole probe | ❌ fails globally | **0.917**; **passes locally** — see §3 |
| *provisional* — estimated gain error ≤ 0.30 | ⚠️ **not evaluable** | gain is **unmeasured**; estimators *disagree* by 5.05× |
| *provisional* — split-half ≥ 0.80 | ⚠️ **not measurable** | the accepted fields' configurations are not recorded — see §3.1 |
| `qualify_field` receipt | ❌ fails closed, not waived | no accepted estimator passes |
| `testing/luke_external_warp_pipeline.py` | ❌ missing | needed to execute, not to decide |

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

### 2.3 Scale: the gain is unmeasured, and the estimators disagree

**Corrected.** An earlier version of this document reported "implied gain error
≥ 0.80". That was wrong and is withdrawn. Estimator disagreement is not an error
measurement: no field on this session has been compared against known motion, so
every field's **absolute gain is unmeasured**. The spread below shows the four
cannot all be right; it does not say by how much any one of them is wrong, and
it must not be quoted as a measured error. The measurements themselves are
unchanged and preserved.

The four accepted estimators, over the same development windows on the same
recording clock:

| estimator | median 120 s rigid excursion |
|---|---|
| ks-motion | 4.06 µm |
| dredge-motion | 10.53 µm |
| medicine | 11.73 µm |
| decentralized-motion | 20.48 µm |

Max/min ratio **5.05**. `qualify_field`'s `estimated_gain_error_fraction` is
therefore **not evaluable**, which is a different state from "evaluated and
failing": there is no number to compare against the 0.30 gate, because the
calibration that would produce one is retracted. Decision 0013 already carries
this disagreement as an open quantification problem.

An operational comparison can still be run under an explicit statement that the
absolute gain is unknown — it just cannot claim to have corrected motion by any
particular amount. See §4.

## 3. The local domain: where it *can* be tested

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

This nomination is recorded before any Option A result.

### 3.1 Support inside the nominated window

The whole-probe 0.917 was dominated by times and depths a bounded experiment
never touches. Inside [7200, 7320] s, on a common 100 µm grid over 555,661
peaks, only **two of 40** depth bins fall below 0.95 local support: 1660 µm
(0.942) and 3860 µm (0.125).

The longest contiguous supported run is **[1710, 3810] µm**. Correcting a
channel at depth *y* by displacement *d* reads source data near *y + d*, smeared
by the interpolation kernel, so that run is eroded at both ends by
max |d| + 3σ. For the selected field: 39.0 + 60 = 99 µm, giving a usable
interior of **[1810, 3710] µm — 1.9 mm, 190 of 384 channels.**

| estimator | max \|d\| | margin | usable interior | channels |
|---|---:|---:|---|---:|
| ks-motion | 39.0 | 99.0 | [1809, 3711] | 190 |
| dredge-motion | 46.0 | 106.0 | [1816, 3704] | 190 |
| medicine | 26.1 | 86.1 | [1796, 3724] | 194 |
| decentralized-motion | 64.1 | 124.1 | [1834, 3686] | 186 |

The domain is frozen now, before any sort. Correction is applied on the full
geometry and the crop happens after; `border_mode` is `remove_channels`, not
`force_extrapolate`, so nothing is invented off the probe.

### 3.2 Split-half: what it found, and the concrete failure it names

Run on the frozen window — 555,661 peaks, deterministic interleaved 2 s blocks,
re-estimated with `dredge_ap`. **Both halves returned identically zero
displacement**, twice: once with library defaults, once with the only DREDGE
configuration recorded on disk. The correlation is undefined, not low.

That configuration is the *withdrawn* rescue sidecar's, carrying
`"fallback": "identity"` — the degeneracy decision 0013 documents. So the zero
is consistent with reproducing a known failure, and is not a measurement of the
accepted field.

The finding is upstream of the number:

> **The accepted estimates' generating configurations are not recorded.** Their
> directories hold `motion.npy`, `time_bins.npy` and `depth_bins.npy` and no
> request or config, so the accepted fields cannot be re-estimated and their
> reproducibility cannot be measured.

This bounds the conclusion rather than blocking the question: the experiment
applies a field identified by **content digest**, not by a recipe. An
unreproducible field cannot support a claim that generalises past this artifact,
and the contract says so.

## 4. The bounded comparison that follows

Specified and frozen in
[`configs/option_a_development_comparison.v1.json`](../configs/option_a_development_comparison.v1.json)
and its [prespec](luke_option_a_development_comparison_prespec.md): one
development-only, operational comparison of one digest-identified field
(`ks-motion`) against an uncorrected control, on the frozen interval and depth
band, with amplitude completeness as the primary endpoint under a 50% coverage
floor and identity/contamination as required support.

Its absolute motion gain remains uncertain and the contract says so in its own
terms: it can answer "did applying this array help here", never "the motion was
corrected by N µm".

What is still **not** being done:

- **No gain calibration.** It is the binding constraint on *qualification* and it
  is a research programme, not a step. The comparison above does not need it,
  and does not pretend to substitute for it.
- **No production change and no promotion.** `FieldGate` and `qualify_field` are
  unmodified and still fail closed.
- **The v2 replay is not promoted, and its 18 unvalidated merges are not carried
  forward.** They are preserved in
  [luke_v2_unvalidated_merge_candidates.md](luke_v2_unvalidated_merge_candidates.md).

## 5. What full qualification would still require

The bounded comparison is not a qualification and does not become one by
succeeding. For a field to pass `qualify_field` on this session:

1. **Record the estimates' generating configurations**, so the accepted fields
   can be re-estimated at all. Nothing else in this list is reachable without
   it — including split-half.
2. **Split-half reproducibility** from the saved peaks, once (1) makes the
   accepted field reproducible. Bounded, no voltage. Measures precision only.
3. **Gain calibration** against known motion. The binding constraint, and the
   only one that is a research programme. Until it exists no field on this
   session can pass, and that is the gate working as designed.

Any Option A experiment, bounded or not, carries amplitude completeness as its
endpoint with **adequate time coverage** (below the declared fraction of the
eligible population it is `inconclusive`, never `pass`) and **identity and
contamination support** alongside it.

## Related records

- [Motion build instructions §5](luke_two_motion_pipeline_build_instructions.md)
- [Decision 0013 — imec0 has appreciable rigid motion](decisions/0013-luke-imec0-has-appreciable-rigid-motion.md)
- [Decision 0002 — motion is estimated, never applied](decisions/0002-motion-is-estimated-never-applied.md)
- [Direct motion-scale audit](luke_20250804_direct_motion_scale_audit.md)
- [v2 result — the closed replay branch](luke_first_pipeline_candidate_v2_result.md)
