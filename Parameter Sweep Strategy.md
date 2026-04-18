# Parameter Sweep Strategy — Kilosort4 Reliability Optimisation
## April 2026 | imec1, Luke03162026_V2V1_RH_g0

---

## Goal

Find a parameter regime that produces the most **reliable** sorting output using the existing KS4 pipeline — without rewriting the sorter. "Reliable" is defined operationally as:

| Metric | Current baseline (default) | Target direction |
|---|---|---|
| Efficiency (well-detected / total) | ~0.147 (5/34 units) | ↑ |
| Median mean-missing % | ~29.6% | ↓ |
| n_well_detected (mpct<20%, presence>0.5) | ~5 | ↑ |
| Flagged split candidates (split_diagnostics) | 0 at default | stay 0 or ↓ |

The efficiency and missing-% numbers indicate that **detection is marginal for the majority of units** — most spikes are close to the detection floor. Solving this is the central problem. Split artifacts are a secondary but related concern: they tend to emerge precisely at parameters that push detection into the marginal regime.

---

## What the first sweep already told us

### ccg_threshold is not the main lever
Sweeping ccg_threshold (0.40 → 0.60 → 0.75 → 0.90) had almost no effect on efficiency or missing %, and produced zero flagged split candidates against the default reference. The merge step is not where the pipeline is failing. **Drop ccg from future priority sweeps.**

### Split artifacts are threshold- and geometry-sensitive
Flagged handoff candidates appear in Th_lo, Th_hi, Thl_lo, Thl_hi, and chans_fewer — not in ccg sweeps. The strongest cases (Th_lo unit 6: conservation ~0.98, anticorr ~-0.22) show clear spike-conservation + temporal tradeoff, consistent with the sorter repartitioning one neuron's spikes across two templates in a time-dependent way. This is upstream of the merge step.

### The dominant hypothesis is detection/representation instability
The pattern is most consistent with **Mechanism 2 (learned-template threshold dropout)** and **Mechanism 3 (discrete template lattice)** from the splitting analysis — the sorter loses or misassigns spikes at particular times because waveforms fall between template boundaries or below detection, not because neurons change biologically.

> **Note on drift / nblocks:** KS4's internal `nblocks` correction operates on the raw recording. Since we feed KS4 an already DREDGE-corrected binary, re-enabling `do_correction` would mean applying a second motion correction pass on a signal that has already been aligned — potentially reintroducing artefacts rather than removing them. The drift hypothesis is therefore addressed by the quality of the upstream DREDGE step, not by sweeping `nblocks` here.

---

## Phase 1 — Fine threshold sweep (Th_learned priority)
**Status: partially run (coarse). Run after Phase 1.**

### Rationale
The existing analysis suggests that `Th_learned` is a more direct lever than `Th_universal`:
- `Th_learned` controls assignment of spikes to already-learned templates (second pass). Lowering it should recover marginal spikes that currently drop below threshold, directly reducing missing %.
- `Th_universal` controls first-pass detection. Changes here affect which neurons are seeded into the template-learning step at all — a more global effect with a less predictable interaction with split artifacts.

The current sweep (Th_learned at 6 vs 12) is coarse and may jump over the sweet spot. The key question is: **is there a Th_learned value between 6 and 9 where we recover spikes without creating split artifacts?**

### What to run

**Th_learned sweep (Th_universal fixed at 12):**

| run_name | Th_learned | Expected direction |
|---|---|---|
| `Thl_8` | 8 | Modest recovery, likely no splits |
| `Thl_7` | 7 | Stronger recovery, watch for splits |
| `Thl_6` | 6 | Already run — creates 1 flagged split |

**Th_universal sweep (Th_learned fixed at 9):**

| run_name | Th_universal | Expected direction |
|---|---|---|
| `Thu_10` | 10 | Test whether 9→10 eliminates the Thu_lo split artifacts |
| `Thu_11` | 11 | Intermediate |
| `Thu_9` | 9 | Already run |

### What to look at
1. Plot **n_well_detected** and **median_mpct** vs Th_learned — is there an elbow where quality improves before splits appear?
2. Check **split_diagnostics**: does lowering Th_learned from 9 → 8 → 7 add flagged candidates, or does the low-mpct regime exist without splits?
3. In Fig 3, check whether the same units that were flagged at Thl_lo (Th_learned=6) are already split at Thl=7 or only at 6.

### Success criterion
Identify the highest Th_learned value (lowest threshold) that does not introduce flagged split candidates while measurably improving median missing % relative to default. That value becomes the recommended Th_learned.

---

## Phase 3 — Spatial geometry (only if Phases 1-2 insufficient)
**Status: not yet run. Lower priority.**

### Rationale
`chans_fewer` (nearest_chans=12, max_channel_distance=48) produced 1 flagged split candidate, but this is a compound change. The key geometry parameters to test independently are:

- `nearest_chans`: controls the channel neighbourhood for template fitting. Too few channels → templates are spatially coarse → nearby neurons can be confused.
- `min_template_size`: controls the spatial scale of detection templates (not currently swept). Small templates may be sensitive to sub-channel shifts.

### What to run (if needed)

| run_name | nearest_chans | min_template_size | Notes |
|---|---|---|---|
| `nc_16` | 16 | default | One step down from 20 |
| `nc_24` | 24 | default | One step up |
| `tsize_lo` | 20 | reduce by ~20% | Test spatial scale sensitivity |

### Success criterion
Geometry changes should only be pursued if Phases 1 and 2 do not produce an acceptable parameter set. A geometry change that helps efficiency without adding split candidates is useful, but this is unlikely to be the primary fix.

---

## Measurement protocol (applies to all phases)

For each new sweep run, the comparison script already computes everything needed. The key outputs to check in order:

1. **`sweep_summary.csv`** — scan efficiency and median_mpct columns first. Any run with efficiency meaningfully higher than the 0.147 baseline is a candidate.
2. **`split_diagnostics.csv`** — check flagged=True rows. Zero flagged is the bar to clear.
3. **`fig2_param_sweep.pdf`** — visual confirmation that the improvement is monotonic in the swept parameter (not an artefact of one outlier run).
4. **`fig3_per_unit.pdf`** — for the best candidate run, check the per-unit pages for the units that currently have high missing %. Are the mpct traces lower and more stable, or just shifted?

---

## What "optimal" looks like

We are looking for a **Pareto improvement** over the current default:

> Lower median missing % AND zero flagged split candidates AND efficiency ≥ 0.20

A secondary target: if no single parameter combination achieves this, identify which combination minimises the sum of (flagged splits × 10 + median_mpct). Weighted this way, avoiding split artefacts is treated as worth ~10 percentage points of missing % — i.e., we would accept slightly worse detection to keep sorting stable.

If no parameter combination reaches the target, the conclusion would be that reliable sorting at this SNR requires a pipeline-level change (e.g. different preprocessing, multi-pass detection, or post-hoc merge curation) rather than KS4 parameter tuning alone.

---

## Execution order

```
1. Phase 1a: Thl_8, Thl_7          ← fine Th_learned sweep, highest priority
   └── find the threshold elbow (spikes recovered vs splits introduced)

1b: Thu_10, Thu_11                  ← only if 1a is insufficient
    └── test Th_universal independently

2. Phase 2: spatial geometry        ← only if Phase 1 doesn't reach target

3. Stimulus overlay                 ← parallel to all phases
   └── overlay known task epochs on flagged split pages
   └── distinguishes stimulus-locked from slow-drift splits
```

---

## Caveats

- The shallow crop (±175 µm around the dense zone) means we are optimising for a specific depth range. The best parameters here may not generalise to the full probe or to deeper units. Any parameters chosen here should be validated on at least one other depth region before being adopted pipeline-wide.
- The baseline efficiency of ~15% may be a property of this recording's SNR, not just the parameter choices. If no sweep achieves efficiency ≥ 0.20 at zero splits, it may indicate that these units are genuinely marginal and the downstream analysis needs to account for variable unit quality rather than expecting a clean sorting solution.
- All sweeps share the same DREDGE-corrected recording binary. The threshold and geometry results are therefore valid only under the quality of the current DREDGE correction — if DREDGE itself is imperfect (e.g. fast event-locked motion not captured by the model), that is a separate upstream problem that parameter tuning here cannot fix.
