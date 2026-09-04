# Noise floor in the quietest windows — Luke imec0 / imec1 vs Yates

**Date:** 2026-09-04
**Script:** `scratchpad/quiet_noise_compare.py` (one-off; reads raw `.ap.bin` /
`recording.dat` directly). Outputs `scratchpad/quiet_noise_compare.csv`.
**Status:** descriptive. Not gated by, and does not gate, anything. A small
concrete piece of the Luke↔Yates comparison that survives the abandoned
motion-matched design.

## What was measured

Per-channel robust σ (MAD / 0.6745, in µV) of the 300–6000 Hz bandpassed voltage,
over the **four quietest 120 s windows** of each recording. "Quietest" = lowest
consensus rigid-excursion rank from
`luke_yates_stable_window_overlap/window_signatures.csv` (concordant estimators:
Luke ks/dredge/decentralized; Yates ks/decentralized). 12 × 1 s chunks per
window. Three reference schemes: none / global median CAR / local (≤ 100 µm)
median reference. Channel 191 excluded from Luke summaries (no material effect —
the shared component dominates).

## Result

Per-channel σ (µV), median across the 4 quiet windows / [p10–p90 across channels]:

| reference | Luke imec0 | Luke imec1 | Yates |
|---|---:|---:|---:|
| **none** (bandpass only) | **41.0**  [39.7–43.2] | **32.9**  [31.9–34.6] | **18.5**  [15.1–25.3] |
| global median CAR | 9.5  [7.4–12.4] | 8.8  [7.0–10.6] | 11.6  [8.1–20.0] |
| local ≤100 µm median | **7.2**  [5.6–9.7] | **6.5**  [5.1–8.3] | **5.4**  [4.3–8.3] |

Window-to-window spread is tiny (imec0 unreferenced 38.5–41.3; Yates local
5.2–5.5) — see the CSV.

## Reading

1. **Unreferenced, Luke is ~2× noisier** than Yates (imec0 2.2×, imec1 1.8×).
2. **Almost all of that excess is a shared / common-mode high-frequency
   component.** Global CAR takes Luke from ~40 → ~9 µV; it takes Yates only
   18.5 → 11.6. After CAR Luke is *below* Yates. Yates's noise is much more
   channel-independent (and global CAR across its two 200-µm-separated shanks
   even adds a little — p90 stays ~20 µV; a per-shank reference is the fair
   number there, and the local-100 µm scheme already does within-shank only).
3. **The genuinely local noise floor — what limits single-channel spike
   detectability — is ~7 µV (imec0) / ~6.5 µV (imec1) vs ~5.4 µV (Yates): a
   ~1.3× gap, not 2×.** Real but modest.
4. **Motion-quiet ≠ voltage-quiet.** These numbers are essentially identical to
   the earlier raw-voltage audit's *"pathological"* windows (38 / 31 / 17.4 µV
   unreferenced; 7.2 / 6.3 / 5.0 µV local). Luke's excess voltage is a stationary
   property of the recording, not a transient concentrated in bad epochs.

## Caveats

- Different probes: NP1.0 (25 µm rows, 4 dense columns) vs Nandy64 (35 µm pitch,
  2 shanks). A ≤100 µm neighbourhood is ~12–14 channels on NP vs ~5–6 on Nandy —
  local referencing is more aggressive on Luke, which flatters the last row for
  Luke somewhat.
- σ is over the full bandpassed trace (spikes included), so it is "voltage
  variability", marginally above a spike-free noise estimate; ordering unaffected.
- Luke's ADC step is 2.34 µV/count (Yates 0.195); quantisation floor
  ≈ 0.68 µV — not limiting.
- Luke's quietest windows still carry 4–6 µm rigid excursion (ks-motion
  1.7–2.2 µm); Yates's carry ~0.1–0.5 µm. Not motion-matched — but this metric
  is a per-channel voltage statistic, largely motion-insensitive at these
  amplitudes.
