# Phase D candidate 2: a non-rigid motion representation — the lever is real, the estimate is the problem

**Date:** 2026-09-02
**Advances:** Phase D of [`pipeline_improvement_plan.md`](pipeline_improvement_plan.md)
**Modules:** `testing/ladder_motion.py` (oracle correction), `ladder_sorter.NONRIGID`
**Evaluation:** `testing/luke_rescue_c2_nonrigid_eval.py` — on the cached C2
injected-truth recordings, drift penalty scored the identical way.

## Verdict

**A non-rigid motion representation is a real lever** — the first Phase D
candidate that is. Handed the **exact** injected trajectory and told to correct
the recording before the frozen rescue sort, the severe rigid-drift penalty on
the two clean donors closes completely: T04 and T06 at 40 µm rigid drift go from
accuracy ≈ 0.40 back to ≈ 0.99, i.e. all the way to their static baseline. That
is the causal confirmation C2 pointed to — remove the motion and the
fragmentation goes with it.

**But KS4's own non-rigid datashift (`nblocks=6`) is not the vehicle** — on a
120 s snippet it is *worse than no correction* (median penalty −0.48 vs −0.35),
because it has too few units over too short a window to estimate drift and then
interpolates by the noise. And **the correction itself has a cost**:
interpolation blurs the waveform, so it does nothing for sub-channel (15 µm)
drift and actively hurts the sharpest, highest-SNR donor (T01).

The path forward is an **external non-rigid estimate computed where it has the
data to be accurate — the full session** — then interpolation, then the rescue
sort. Tested the same decisive way stitching was: injected-truth drift penalty
*and* full-session reconstitution of the 127.

## The three arms

| arm | what it is |
|---|---|
| `rescue` | frozen rescue config, no motion representation (the C2 baseline) |
| `nonrigid` | KS4's own non-rigid datashift, `do_correction=True nblocks=6`, rescue detection thresholds unchanged — the *estimated* case |
| `oracle` | `InterpolateMotionRecording` with the **exact** injected trajectory (`ladder_motion.oracle_corrected_recording`), then the frozen rescue sort — the *ceiling* |

`oracle` correction leaves the static arms intact (T01 0.941→0.941, T04
0.948→0.948) — the sign convention and the zero-motion interpolation are
faithful; the effects below are motion handling, not artefact.

## Drift penalty (moving − static accuracy on the injected unit)

| donor | trajectory | rescue | nonrigid | oracle |
|---|---|---:|---:|---:|
| T01 | rigid 15 µm | −0.35 | −0.32 | −0.35 |
| T01 | rigid 40 µm | −0.54 | −0.75 | −0.65 |
| T01 | osc 20 µm/40 s | −0.70 | −0.72 | −0.43 |
| T04 | rigid 15 µm | −0.30 | −0.58 | −0.28 |
| **T04** | **rigid 40 µm** | **−0.53** | −0.49 | **+0.04** |
| T04 | osc 20 µm/40 s | −0.38 | −0.48 | −0.17 |
| T06 | rigid 15 µm | +0.19 | −0.10 | +0.18 |
| **T06** | **rigid 40 µm** | **−0.12** | +0.21 | **+0.48** |
| T06 | osc 20 µm/40 s | +0.02 | −0.13 | +0.49 |
| | **median** | **−0.35** | **−0.48** | **−0.17** |

(T06's static baseline is 0.505 — below the C2 sanity bar — so its penalties
are noisy and its positive oracle deltas partly reflect interpolation sharpening
a marginal recovery.)

Absolute moving-arm accuracy, the clearest cut:

| donor | trajectory | rescue | oracle |
|---|---|---:|---:|
| T04 | rigid 40 µm | 0.42 | **0.99** |
| T06 | rigid 40 µm | 0.38 | **0.99** |
| T06 | osc 20 µm/40 s | 0.53 | **0.99** |
| T04 | osc 20 µm/40 s | 0.57 | 0.78 |
| T01 | rigid 40 µm | 0.40 | 0.29 |
| T01 | rigid 15 µm | 0.59 | 0.59 |

## What this means

1. **Motion representation works** where family stitching did not. The
   fragmentation is caused by the moving footprint; correcting the recording
   with the true motion removes it, on the clean donors, at the drift
   magnitudes that hurt most.
2. **The bottleneck is the estimate.** KS4's internal non-rigid datashift on a
   snippet is worse than nothing. The estimate has to come from somewhere with
   more data.
3. **Interpolation is not free.** It does not help sub-channel drift (15 µm ≈
   0.75 channel — the interpolation noise swamps the signal), and it degrades
   T01 — the sharpest waveform, most sensitive to spatial blur. A real candidate
   must be judged per-SNR-tertile, not on a median.
4. **T01 is a warning.** Even a perfect motion vector can lose to no correction
   when the waveform is sharp and the drift is small. The rescue pipeline's
   whole thesis is "preserve the voltage"; interpolation spends some of that.

## Next

- **External non-rigid estimate, full session.** `estimate_motion` (dredge /
  decentralized, non-rigid, multiple spatial windows) on the whole imec0
  recording — where a drift estimate has 10 000 s and hundreds of units to work
  with — then `InterpolateMotionRecording`, then the rescue sort. Score against
  the 127 (full-session reconstitution, the test that killed stitching) and
  against C2 injected truth (drift penalty, per SNR tertile).
- If the external estimate on the full session reconstitutes a large share of
  the 127 without a similar-pair or edge-spike regression, it is the Phase D
  winner and goes to L2 on the frozen panel.
- If it does not — if the estimate is still too coarse or the interpolation cost
  still dominates — then the rescue-vs-legacy question resolves as: the two
  pipelines trade the same errors and neither is clearly better on this
  recording, and the improvement lever is elsewhere (detection, clustering).

## Limits

- Diagnostic: C2 discovery-cohort donors (T01/T04/T06), one background window,
  negative-compact polarity, 120 s.
- The oracle arm imposes rigid motion on a spatially-uniform strip; a true
  non-rigid (depth-varying) trajectory is not yet tested.
- `nblocks=6` is one datashift setting; the finding is that snippet-scale
  estimation fails, not that this exact number is wrong.
