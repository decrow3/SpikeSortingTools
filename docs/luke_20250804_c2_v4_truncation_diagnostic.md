# C2 v4 staircase amplitude-truncation diagnostic

**Date:** 2026-09-04  
**Status:** complete, post-hoc diagnostic; **not production QC and not a promotion endpoint**  
**Code:** [`testing/luke_c2_v4_truncation_diagnostic.py`](../testing/luke_c2_v4_truncation_diagnostic.py)  
**Output:** `testing/outputs/luke_rescue_c2_drift_challenge_v4/truncation_diagnostic/`  
**Tests:** 26 pass across the new diagnostic and the existing fitter audit.

## Answer

The truncation statistic supports a specific interpretation of the staircase:
**uncorrected rescue primarily loses continuity of identity across positions; it
does not generally lose the low-amplitude tail within each position.**

For 13/14 donors, uncorrected rescue split the injected neuron across the 0 and
40 µm phases. The best single output identity consequently missed a median
**50.51%** of the full train. Within the phase owned by each fragment, however,
the median true loss was only **1.02%** and the median amplitude-truncation
estimate was **0.62%**. Ten of those 13 split donors had estimates below 5% in
both phases.

That is the prediction pre-registered in the main plan for **temporal
fragmentation**: each fragment retains its local amplitude distribution, so its
truncation estimate stays low even though no single cluster follows the neuron
through the whole recording. Truncation QC is therefore not a replacement for
injected-truth recall or identity-continuity scoring.

## Why this is a diagnostic rather than production QC

Production truncation QC requires 1,000 spikes in a continuous block. C2's
staircase has only 687 admitted spikes, and moving fragments commonly contain
about 340. Applying production defaults would yield almost no estimates.

This analysis instead freezes the following exploratory policy:

- 250 spikes per amplitude-CDF window;
- pool the two plateaus at the same displacement level, then construct centred,
  non-overlapping count windows;
- analyze 0 and 40 µm separately;
- analyze each capturing output cluster separately — never pool amplitude
  values across separately normalised clusters;
- fit exactly 250 amplitudes per window, correcting the production helper's
  historical inclusive-stop/exclusive-slice discrepancy;
- compare every estimate with known injected-truth missingness;
- treat 50% fits as censored and exclude them; retain a separate fallback flag.

The sorter was **not rerun**. The diagnostic reads the retained C2 L1 curation
caches and uses the same `full_st[:, 2]` amplitude field as production QC.

## Small-window calibration

The 250-spike version was tested with 100 deterministic simulations at each
known missing fraction under the truncated-logistic model assumed by the
fitter.

| true missing | median estimate | median absolute error | 5th–95th percentile |
|---:|---:|---:|---:|
| 0.5% | 0.93% | 0.43 pp | 0.35–2.03% |
| 1% | 1.36% | 0.42 pp | 0.66–2.73% |
| 2% | 2.38% | 0.60 pp | 1.30–4.41% |
| 5% | 5.50% | 1.01 pp | 3.29–8.13% |
| 10% | 9.72% | 1.68 pp | 5.65–15.73% |
| 20% | 21.99% | 3.66 pp | 13.56–30.50% |
| 30% | 31.62% | 5.06 pp | 18.19–44.54% |
| 40% | 38.90% | 7.65 pp | 25.39–50.00% |

Median bias is at most 1.99 percentage points through 20%, but individual
windows become imprecise as missingness rises. At 40% true missing, 23% of
simulations hit the fitter's 50% censoring boundary. These intervals assume the
model is correct; real multimodal or contamination-heavy clusters can behave
worse.

## Coverage and fit guards

| item | result |
|---|---:|
| frozen staircase cells read | 98 |
| capturing output clusters | 119 |
| cluster × phase combinations | 238 |
| eligible at 250 spikes | 197 |
| fitted windows | 210 |
| usable, uncensored windows | 194 |
| 50%-censored windows | 16 |
| fallback-signature windows | 0 |
| donor-level primary phase rows | 196 |
| primary phase rows with a usable estimate | 189 |

The donor-level analysis selects the cluster with the most true captures in
each phase. This ensures each donor contributes once per cell and phase; donors
with extra fragments do not receive extra weight.

## Main paired comparisons

The most important contrast is the change from each configuration's own static
arm. Values below are medians of matched donors.

| configuration | phase | Δ true loss within phase | Δ truncation estimate | Δ whole-train loss of best identity |
|---|---:|---:|---:|---:|
| rescue, uncorrected | 0 µm | +0.29 pp | **−0.12 pp** | **+49.64 pp** |
| rescue, uncorrected | 40 µm | +0.15 pp | **−0.25 pp** | **+49.64 pp** |
| rescue, rigid correction | 0 µm | +0.15 pp | +0.08 pp | +0.29 pp |
| rescue, rigid correction | 40 µm | +0.15 pp | ~0.00 pp | +0.29 pp |
| legacy-style | 0 µm | 0.00 pp | −0.08 pp | −0.07 pp |
| legacy-style | 40 µm | 0.00 pp | +0.27 pp | −0.07 pp |
| exact corrected rescue | 0/40 µm | 0.00 pp | **0.00 pp** | 0.00 pp |

The exact-corrected and static rescue estimates were identical across all 28
donor × phase pairs. This independently recovers the staircase machinery's
bit-for-bit sorting control.

The median whole-train missing percentage of the best identity was:

- static rescue: **0.87%**;
- moved rescue, uncorrected: **50.51%**;
- moved rescue with rigid correction: **0.73%**;
- moved legacy-style: **0.95%**;
- exact-corrected rescue: **0.87%**.

## Exceptions matter

The statistic was not uniformly benign, which is useful rather than a reason to
discard it:

- D10's 0 µm fragment truly missed 34.9% of that phase and estimated 37.2% — a
  genuine within-phase loss that the statistic detected.
- D11's 0 µm fragment truly missed 13.7% and estimated 20.0%.
- D08 estimated 36.4% despite only 4.4% truth loss; its phase cluster was only
  61.7% injected spikes. This is a contamination/non-model failure and shows
  why the truth comparison and precision column must accompany the estimate.
- Sixteen of 210 windows saturated at 50%; they are recorded as censored and
  excluded rather than averaged as measurements.

## Interpretation

The staircase gives a clean separation:

1. **Uncorrected rescue:** large whole-train identity loss, generally low
   phase-local truncation — predominantly temporal/positional fragmentation.
2. **Rigid and legacy-style correction:** both whole-train continuity and
   phase-local truncation remain near their own static baselines.
3. **A few donors:** real within-phase amplitude loss or contaminated amplitude
   distributions occur and are visible as exceptions.

This strengthens the case that the staircase failure of uncorrected rescue is
not simply "half the spikes fell below threshold." The sorter generally finds
the spikes at each lattice position but assigns the two positions to different
identities. It does **not** establish how truncation behaves at Luke's
fractional 5/11/22 µm ramps, whose forward-model attenuation remains a separate
confound.

## Files

- `summary.json` — frozen policy, calibration, coverage, paired contrasts and
  mechanistic headline;
- `calibration.csv` — all synthetic calibration summaries;
- `primary_phase.csv` — one selected capturing cluster per donor/cell/phase;
- `cluster_phase.csv` — every capturing cluster × phase, including ineligible
  combinations;
- `windows.csv` — every fitted window, both missingness estimates, truth error,
  censoring and fallback flags.
