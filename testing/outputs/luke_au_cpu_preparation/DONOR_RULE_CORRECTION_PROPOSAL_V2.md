# AW donor construction proposal, revision 2

Status: **prospective design for review; not applied.** This document supersedes
the design advice in `DONOR_RULE_CORRECTION_PROPOSAL.md` only if it is approved.
It does not change the original frozen blocked result, select donors, authorize
voltage extraction, launch a benchmark, or alter the deployed 182-channel
sorter. The earlier proposal is retained as an audit record.

## Corrections to the earlier proposal

The supplemental QC has no `quality` or `isolation_score`. Those unavailable
fields are retired without substitutes. The cohort is described as
**low-refractory static final-label templates**, not verified well-isolated
neurons. `presence_ratio`, SpikeInterface ISI metrics and refractory-period
contamination remain descriptive because none is an isolation score.

A final-label donor may be constructed directly from an explicitly chosen
final S label's saved spike times. It does not require a one-to-one mapping to
the 654-unit pre-matching template bank. Merge, split and representative-template
lineage should still be reported as a diagnostic and biological-interpretation
caveat. One-to-one lineage is required only for the separate question of
reproducing a historical matching template; it is not an eligibility gate for
a newly constructed final-label donor.

The earlier proposal incorrectly reused centred-cosine >=0.99 and PTP ratio
within 2% as biological split-half stability thresholds. Those tolerances test
deterministic exact-copy/remap fidelity. They are not justified for two noisy
200-spike estimates. Split-half stability is therefore diagnostic in this
revision; no biological stability cutoff is invented.

## Frozen inputs and exact domains

- W2 is 900--1240 s at the recorded sampling frequency
  **29,999.759166666667 Hz**, not an assumed 30 kHz.
- Final S labels and QC are the matching run's 182-channel shallow domain,
  AP202--AP383. QC rows and spike counts match final labels 0--537 exactly.
- Waveform measurement uses all 384 physical AP channels after the exact saved
  reference-before-crop `ibllikecmr` preprocessing. This is measurement only.
  Any injected arm remains in the historical 182-channel deployed sorter
  domain; there is no switch to full-probe sorting.
- The mask used by the prescreen is
  `/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab/censor_mask_v1.csv`,
  9,151 bytes, SHA-256
  `31b39a494ede16e3f5919caf1cf53a6222dd5e737f027d59155e96a9ce9179ed`.
  It has the same 422 `[start_s,end_s)` intervals and normalized A/U/E source
  labels as the 8,750-byte hub canonical CSV, SHA-256
  `86425e8a6627a10ac8932ba01b799cb7be7141f731ebb4ecc01e303326b9da55`.
  The byte difference is source-label serialization (`A+E` versus `AE`), not
  mask membership. A future worker must verify numerical equality before use.

## Prospective cohort construction

The saved-array pre-screen is unchanged and is not selection. For each final S
label, report total spikes, the exact 1 ms
`raw_adjacent_isi_lt_refractory_fraction`, rest fraction outside the canonical
AE mask, valid rest-waveform count, PTP and depth. The current proposed scalar
thresholds are >=800 total spikes, adjacent-ISI fraction <=0.005, rest fraction
>=0.80 and >=400 valid rest waveforms. Ninety of 538 labels pass the first three;
the valid-waveform count is not known. These thresholds themselves remain part
of the proposal and must be approved before application.

For each approved pre-screen label, construct a new full-probe template directly
from that final label's valid rest spike times using the recorded 121-sample
DARTsort `TemplateData.from_config`/`peelreduce` median path, trough offset 42,
saved denoising/SVD settings and at most 500 spikes. Report deterministic
temporal split-half cosine and PTP ratios as diagnostics with confidence or
empirical distributions; do not reject on the old 0.99/2% operator tolerances.

Lineage to the 654-unit saved bank is reported separately as one-to-one,
many-to-one, split, representative-only, ambiguous or absent where it can be
derived. Crop projection of the new template versus a saved historical template
is likewise a reproduction diagnostic, not a prerequisite for defining the
new final-label donor.

## Mandatory support and operator-fidelity checks

For every occupied W2 state `{0, -40, -80, -120, -160, -200, -240}` um:

1. translate the newly measured 384-channel donor by the exact same-column
   physical mapping and then project it into AP202--AP383, the actual injected
   and matching domain;
2. require >99% of the donor energy needed for that translated injected
   waveform to be supported by real measured channels, without padding,
   extrapolation or many-to-one channel mappings;
3. exact inverse remapping of the same deterministic template must reproduce it
   with centred cosine >=0.99 and PTP ratio in `[0.98,1.02]`. This is explicitly
   an operator-fidelity test, not biological stability;
4. require the translated footprint and immutable injection samples to remain
   inside the 182-channel recording and W2 time support; and
5. exclude placements that reproduce an occupied translation of the donor's
   native peak or a native template clone under the same operator-fidelity
   tolerances.

The 440 um interior margin (240 um maximum occupied state plus the matcher's
200 um interpolation neighborhood) is retained only as a conservative
diagnostic screen. It is not equivalent to the mandatory actual-state support
test and is not a hard donor gate unless separately justified as necessary to
prevent a specific padding or extrapolation path. The old 7/654 result remains
a shallow-crop diagnostic, not a full-probe eligibility count.

## Ranking and cohort-size decision

Before inspecting any injected-arm outcome, divide the full-probe depth span
into six fixed equal strata and rank candidates by calibrated full-probe PTP,
then lower exact 1 ms adjacent-ISI fraction, then larger valid-rest count, then
final label ID. Aim for approximately 30 donors with depth coverage.

Thirty is a design target, not an evidence-backed exact stopping threshold. If
the approved gates yield a different count, report the eligible count, depth
coverage and expected paired-comparison precision using scorer-only simulation
or resampling that does not inspect S/D2L arm outcomes. Pause for an explicit
cohort-size decision; do not silently reduce the target, relax gates or select
on arm performance.

## Smallest coherent no-sort CPU recipe

1. Freeze the approved final-label pre-screen and mask equality receipt.
2. Read W2 AP voltage once, read-only: about 7.83 GB.
3. Apply the exact 384-channel preprocessing graph in RAM (about 15.67 GB).
4. For approved final labels, sample valid rest spikes deterministically and
   build full-probe templates under the saved DARTsort builder.
5. Compute mandatory actual-state support/operator checks, diagnostic
   split-half stability, and optional lineage/crop-reproduction diagnostics in
   the same pass.
6. Persist only templates, per-unit measurements and receipts (under 0.2 GB),
   then make the separate cohort-size/precision decision.

The provisional CPU estimate remains 20--35 minutes, with a crop-overlap dry
run first to confirm builder/config fidelity. This recipe performs no sort,
does not modify voltage and does not change the 182-channel production domain.
