# EM — reproduce the lattice remap in SpikeInterface, then compare it with standard interpolation methods

**Status:** DRAFT / UNSENT. Drafted by Claude (hourly review) on 2026-09-28.
- **User request, relayed:** "We should ask our coordinator to do this test… It should probably live in SpikeSortingTools." That applies to this test only. The user's broader hold on new work otherwise stands. Please confirm scope with the user if your policy needs it in your own context.
- **Home:** SpikeSortingTools, not the DARTsort repo. Specifically the huklaban5 **`/home/huklaban5/Documents/SpikeSortingTools/NPX_preprocessing`** git repo (HEAD `06d9c70`, venv with SpikeInterface **0.104.7**), because the W2/W3 recordings and the DD materialized remap are on huklaban5.
  - Sorting reuses the existing DARTsort static worker and scorer, consuming the SpikeInterface outputs.
  - huklaban1's `RyanSorting/SpikeSortingTools` (motionqc) does an independent source review only. No voltage leaves huklaban5.
- **Next unused letter per the ledger:** EM. Please rename if it's already allocated.

## Why

A literature check ([literature-lattice-remap-20260928.md](literature-lattice-remap-20260928.md)) found that raw-data registration before a static sorter is established (Kilosort 2.5 datashift; SpikeInterface `interpolate_motion` with kriging/IDW; Garcia et al. 2024). But no published method restricts the correction to **exact whole-period permutations**.

SpikeInterface's `spatial_interpolation_method="nearest"` ("snapping") picks the nearest electrode in 2D, with no period constraint. It failed in Garcia et al. for **sub-period** drifts (12–40 µm, with a 44 µm period). For a **rigid, lattice-rounded** field, every target lands exactly on a real electrode, so snapping should **equal** our DD lattice remap. If it does:
- the method gets a standard, one-call implementation;
- the natural methods comparison (lattice vs kriging vs IDW vs plain snapping vs none) becomes cheap.

## EM.1 Equivalence (CPU; source-first; no sort)

1. **Source and config.** From installed SpikeInterface 0.104.7, cite file and line for:
   - `interpolate_motion` target construction (`channel_locations_moved[:, dim] += displacement`);
   - the time-bin selection (nearest motion bin vs interpolation);
   - `get_spatial_interpolation_kernel(method="nearest")`: argmin of the 2D distance, weight 1;
   - border handling and `force_extrapolate`.

   From the executed DD source (h5 `dd_lattice_inputs`/materializer), cite:
   - the lattice period (NP1.0, 40 µm; x alternates {0, 32}/{16, 48});
   - the reference offset r (W2 −4.69558824159 µm) and rounding rule;
   - the sign convention (`corrected = observed − displacement`);
   - the time grid and knot selection;
   - zero-fill;
   - preprocessing order (383 supported channels after filter, phase, two global references and scale, then remap, then 182-site crop; AP191 excluded).
2. **Build the SpikeInterface version:**
   - a **rigid** `Motion` (one spatial bin) whose displacement is the DD lattice-rounded shift `40·round((d − r)/40)` on DD's exact time grid, with the sign set so that SpikeInterface's `location + displacement` sampling reproduces DD;
   - `border_mode` set to reproduce DD's zero-fill (for example `force_zeros`), not `remove_channels`;
   - applied at the same preprocessing stage and channel set as DD, then the same crop.
3. **Sign fixture first:** one synthetic channel at a constant +40 µm. Assert that SpikeInterface output equals the expected channel permutation.
4. **Equivalence check** against the DD materialized W2 remap (`/home/huklaban5/DARTsort_experiment_scratch/dd_lattice_w2_20260928/…`).
   - Frozen sample: 3 s from low-deviation rest, 3 s from an episode core, and 3 s spanning an episode onset/offset transition, chosen deterministically from the DC state table before reading voltage.
   - **Pass:** maximum absolute difference ≤ 1e-6 (float32), identical zero-fill positions, and identical per-time-bin channel mappings.
   - **Fail:** report the first differing sample and channel and its cause (sign, reference, time bin, border, order). Don't tune until it passes. A documented, explained mismatch is acceptable if the cause is a DD convention SpikeInterface can't express.
5. Commit the helper, for example `npx_preprocessing/motion/lattice_remap_si.py`, with its tests in SpikeSortingTools. **Nothing goes into the DARTsort repo.**

## EM.2 Methods comparison (conditional on EM.1 passing; GPU)

- **Inputs, W2 first:**

  | Arm | Method | Field |
  |---|---|---|
  | none | S0, existing | — |
  | lattice | DD SL, existing; **or** the equivalent SpikeInterface nearest-on-lattice-rounded output (not both) | lattice-rounded |
  | kriging | SpikeInterface kriging, default parameters | unrounded two-layer field, rigid |
  | idw | SpikeInterface IDW, default parameters | unrounded two-layer field, rigid |
  | snap | SpikeInterface nearest | **unrounded** field (plain snapping) |

  That's three new sorts per window.
- **Sorter:** the existing static DARTsort S config and seed, unchanged; the same single-linkage/complete-linkage choice as the EF contract.
- **Scoring:** the frozen EF scorecard. Identity rate-rank ρ, yield ratio, and short-interval fractions by field-defined state, with 2000-draw common-block bootstrap CIs. **No RF**, per the user's stop.
- **Pre-registered reading, before outcomes:**
  - **Lattice advantage:** lattice beats kriging and IDW on ρ (Δ ≥ 0.05, CI lower bound > 0), with yield and purity within the EF tolerances.
  - **Interpolation equivalent:** |Δρ| < 0.05 with the CI including 0. Kriging/IDW is then an acceptable standard substitute.
  - **Snapping check:** plain snapping is expected to trail lattice. If it doesn't, report that plainly.
  - **Anything else is mixed.** Report it without tuning.
- **Transfer:** W3 only if W2 is informative, with the same frozen reading.
- **Optional:** the same five inputs on the DK hybrid (exact-lattice injection, so it favours lattice and snapping), labelled as an upper-bound check.

## Resources and limits

- Set by the coordinator from measured costs. The expected envelope:
  - EM.1: ≤ 20 min CPU, reading ≤ 1 GB of voltage;
  - EM.2 on W2: three sorts of about 20 min each plus scoring.
- Lazy SpikeInterface recordings are preferred; materialize only if the worker needs it, on local scratch within the existing caps.
- No voltage export, no production change, no RF, and no outer-holdout access.

## Report

- Verdict first: EM.1 pass/fail, then the EM.2 reading.
- One table: arm × {ρ, yield, short fractions} with CIs.
- One figure: ρ by arm with CIs, Okabe–Ito colours plus markers.
- The implementation checks performed.
- The SpikeSortingTools commit hash for the helper and tests.
