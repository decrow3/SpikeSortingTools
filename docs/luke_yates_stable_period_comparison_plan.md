# Prespec: the Luke–Yates stable-period comparison

**Status:** proposed 2026-09-03. Runs **in parallel with C2 v3** (§9 of
[`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)); it is not gated
by it and does not gate it.

> **INCREMENT 1 RAN 2026-09-03 — GATE FAILED.** Zero of 87 Luke imec0 120 s
> windows reach the Yates-Q75 overlap box under any common estimator; Luke's
> quietest 2-minute window still drifts more (rigid P95−P5 ≥ 4.2 µm, primary
> estimator) than the Yates 75th percentile (3.6 µm). Non-rigid gradient is
> *not* the problem — rigid translation magnitude/rate is. The motion-matched
> design (increments 2–5) is **not viable as written**; there is no
> genuinely-quiet Luke subset to match. See
> [`luke_yates_stable_window_overlap_result.md`](luke_yates_stable_window_overlap_result.md)
> for the numbers, the estimator-disagreement caveat, the "1.28 µm" discrepancy,
> and the three salvage options. Increments 2–5 below are on hold pending that
> decision.

**Supersedes as the active form of:** the depth-resolved biological comparison
deferred in
[`luke_yates_raw_voltage_audit_notes.md`](luke_yates_raw_voltage_audit_notes.md)
("A depth-resolved biological comparison must therefore be restricted to
anatomically matched support … This comparison is deferred until conditioning,
preprocessing and motion handling are fixed").

**Why un-defer it now.** That deferral assumed the comparison had to wait for a
settled motion correction. It does not, *if* windows are selected to be
genuinely motion-quiet in both recordings — then motion handling is not on the
critical path for those windows. The un-deferral is conditional on the
**overlap gate** (increment 1) confirming a non-empty motion-matched region.

---

## 1. The question, as a falsifiable claim

> When motion is genuinely small and depth is anatomically matched, Luke_20250804
> is a basically healthy extracellular recording: its motion-quiet windows reach
> Yates's motion-quiet windows on input-signal quality, on units/mm of cortex
> after identical sorting, and on per-unit waveform/refractory quality — while
> Luke's high-motion windows at the same depths do not.

The decisive structure is an **interaction**: `dataset × motion regime`, depth
controlled. Three pre-committed readings:

| Result | Reading |
|---|---|
| Luke-stable ≈ Yates-stable on compact-event/mm, units/mm, waveform/refractory quality; Luke-high-motion collapses | Recording is fundamentally viable; motion/sorting is the dominant recoverable problem — the C2 v3 / D2 line is aimed correctly. |
| Luke-stable deficient versus Yates-stable **even motion-matched at matched depth** | Stop attributing the deficit to motion. Investigate penetration history, anatomy, acquisition/reference — in particular the stream-fixed imec1 positive-excess / negative-deficit finding ([`luke_yates_raw_voltage_audit_notes.md`](luke_yates_raw_voltage_audit_notes.md), cross-session recurrence). |
| A known injected donor recovers equally well in Luke-stable and Yates-stable backgrounds, but real units/mm is still lower in Luke-stable | Signal-content deficit, not sorting tractability — the background is hospitable, there are simply fewer neurons in the sampled tissue. |

**Not endpoints** (carried over from `pipeline_improvement_plan.md` §5): KS-good
count, total spikes, `KSLabel == good` alone, population medians of any per-unit
metric across the two sorts.

---

## 2. Increments, and their gates

| # | Increment | Gate to proceed |
|---|---|---|
| **1** | **Motion-overlap feasibility screen.** Signature every candidate 120 s window in both recordings from estimator arrays only; apply the prespecified overlap box. This is a *feasibility / calibration* screen, **not** the matched-window selection — it only asks whether enough Luke windows reach Yates's own quiet region for a matched comparison to be built. These datasets and metrics were already examined in `luke_yates_motion_comparison.py`, so increment 1 is calibrated/descriptive, not a fresh confirmatory test; freezing it protects the downstream sorting inference. | ≥ 6 Luke imec0 windows **and** ≥ 6 unique Yates session time-intervals quiet on **all** shanks, in the overlap region under the primary estimator. **If it fails, the motion-matched design is not viable as written** — report and stop; do not silently widen the box. |
| **2** | **Depth/anatomy anchor.** Fix the shallow-V1 overlap depth range in each recording from the available anchors (Yates `laminar.npz`; Luke — TBD, see §5). Express everything per physical mm. | A stated, defensible overlap depth range exists, or the comparison is explicitly labelled normalized-depth-only with that caveat. |
| **3** | **Common sorting subgraph.** Define the preprocessing + KS4 config that can run on *both* probe geometries (see §4). Build cached snippets for the selected windows. | The subgraph is frozen and its Neuropixels-specific omissions for Luke are recorded (conservative: Luke loses its NP-specific conditioning, it is not favoured). |
| **4** | **Three-layer comparison + injected-donor control** on the selected windows. | — |
| **5** | **Blinded manual scoring** of a stratified unit sample. | — |

This document freezes increments 1–2. Increments 3–5 get their own frozen
parameters appended here **before any sort runs**.

---

## 3. Increment 1 — frozen parameters

**Inputs (read-only):** motion-estimator arrays only —
`motion.npy` / `time_bins.npy` / `depth_bins.npy`. No sorter labels, no voltage.

- Luke_20250804: `/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_{imec0,imec1}/motion/<estimator>/`
- Yates: `/media/huklab/Data/Yates_session_copy/processed/Allen_2022-02-16/{shank1,shank2}-motion/<estimator>/`

**Common estimators:** `medicine`, `ks-motion`, `decentralized-motion`.
Primary = `medicine` (MEDiCINe; the most robust across the existing
`luke_yates_motion_comparison.py` work). `dredge-motion` exists for Luke only and
is reported, never used for the gate.

**Windows:** non-overlapping, `WINDOW_S = 120`, `STRIDE_S = 120`, enumerated from
`t.min()` of each source's own time base. A window is emitted only if it holds
≥ 20 motion time-bins whose timestamps span ≥ `0.9 × WINDOW_S` (a partial
trailing window that clears both thresholds is kept; the coverage rule replaces a
literal `t.max() − 120` bound so irregular real bin spacing does not drop good
windows). Each window records **both** a native-clock start
(`window_start_native_s`, e.g. Luke's ~3058 s origin) and a recording-relative
start (`window_start_recording_s = start − t.min()`); shank matching and any
later voltage extraction use the recording-relative clock. The exact
native→frame mapping for snippet extraction is frozen and tested in increment 3.

**Per-window signature** (rigid = mean of motion over the depth axis):

| Field | Definition |
|---|---|
| `rigid_excursion_um` | P95 − P5 of `rigid` over the window (µm; comparable across probes directly) |
| `nonrigid_grad_um_per_mm` | median over time of `(max − min of motion across depth) ÷ (depth-bin span in mm)`. **Depth-span normalised** so Luke's ~4 mm span and Yates's ~1.2 mm span are comparable; `0` for a one-depth-column estimator |
| `p95_nonrigid_grad_um_per_mm` | P95 over time of the same normalised gradient |
| `rigid_speed_um_s` | P95 of `|d(rigid)/dt|` |
| `finite_fraction` | fraction of in-window bins with all-finite motion |
| `depth_span_um`, `n_depth_bins`, `dt_median_s`, `max_time_gap_s` | recorded for QC |

The estimators do not share a depth resolution — `medicine` gives 2 depth
columns on both recordings, `ks-motion` / `decentralized-motion` give 18 on Luke
vs 4 on Yates — hence the per-mm normalisation rather than a raw across-depth
range.

**Overlap box (prespecified, anchored to Yates's own quiet regime).** For each
estimator, pool all Yates windows (shank1 + shank2). On each of the three axes
`{rigid_excursion_um, nonrigid_grad_um_per_mm, rigid_speed_um_s}` set the upper
edge `Q_axis = quantile(Yates windows, 0.75)`. A window (either dataset) is **in
the overlap region** iff all three axis values `≤ Q_axis`, `finite_fraction ≥
0.90`, and `max_time_gap_s ≤ 3 × dt_median_s` (no large internal hole). Box
construction aborts if any Yates axis value is non-finite.

Rationale and limits of the box: it is defined without reference to Luke or to
any sorting result. But it is only a **marginal** screen — a Luke window merely
has to fall below three independent Yates-Q75 ceilings, and two distributions can
clear that and still differ. So the increment-1 verdict is "are enough
candidate windows available?", not "are these windows matched". Q75 (not the
min) keeps the Yates side populated so a Luke shortfall is informative rather
than an artifact of an over-tight box. The **actual matched-window selection**
(balanced nearest-neighbour / coarsened-exact matching on the normalised
signatures, with balance diagnostics) is frozen separately in increment 3, and a
frozen sensitivity table over the box quantile is reported without being allowed
to change the primary verdict.

**Gate.** For the primary estimator: PASS iff
`n(Luke imec0 windows in overlap) ≥ 6` **and**
`n(unique Yates session time-intervals in overlap on every shank) ≥ 6`. The
Yates side counts **distinct session times** (a `time_interval_id` on the shared
recording-relative clock), each required to be in-overlap on *all* shanks
present — so quiet-on-one-shank / moving-on-the-other periods do not count, and
simultaneous shank windows are not double-counted. Reported for every estimator
and for imec1; the overall verdict follows the primary estimator on imec0. All
required sources (2 probes × 3 common estimators for Luke, 2 shanks × 3 for
Yates) must load and pass shape/monotonic-time/depth-dimension checks, or the run
aborts without a verdict.

**Luke high-motion controls (for increment 4).** Luke imec0 windows with
`rigid_excursion_um ≥ quantile(Luke imec0 windows, 0.90)` for the primary
estimator. Enumerated here; depth-matched selection happens in increment 3.

**Outputs** — `testing/outputs/luke_yates_stable_window_overlap/`:
`window_signatures.csv`, `overlap_gate.json`, `overlap_scatter.png`.

---

## 4. Increment 3 preview — the common subgraph (parameters frozen later)

The `ladder_sorter.RESCUE` graph is Neuropixels-specific: NP phase-shift
correction, 500 µV bilateral blanking, physical-channel-191 interpolation, and a
384-channel CAR. Yates is a **Nandy64** probe — 2 shanks × 32 sites, 35 µm
pitch, ~1085 µm span. "The same graph on both" is therefore only the intersection:

- bandpass 300–6000 Hz
- common reference (per-shank median for Yates; whole-probe / per-column for Luke — **TBD, must be stated**)
- KS4 internal high-pass + whitening, `do_correction` / `nblocks = 0` (no external and no internal motion correction — windows are motion-quiet by construction)
- identical detection thresholds and `score_sort` config

Luke's NP-specific conditioning (phase shift, blanking, ch-191) is **dropped** in
this arm. That handicaps Luke, which is the conservative direction for a
"is Luke viable" test.

---

## 5. Open items to resolve before increments 2–3

- **Luke depth anchor.** Yates has `laminar.npz` (CSD/laminar). What is the Luke
  equivalent — RF eccentricity, CSD, LFP landmark? If none, increment 2 proceeds
  normalized-depth-only with that caveat stated in every density figure.
- **Common-reference choice for Luke** in the §4 subgraph.
- **Injected-donor control scope.** Reuse the 14 compact D2b-2 donors
  ([`d2b2_donor_cohort`](luke_20250804_d2b2_donor_cohort.md)); inject the
  identical train + waveform into motion-matched Luke-stable and Yates-stable
  backgrounds, sort with the common subgraph, compare recovery.

---

## 6. Reproducibility

- Increment 1 script: `testing/luke_yates_stable_window_overlap.py`
- Tests: `testing/test_luke_yates_stable_window_overlap.py`
