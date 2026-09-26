# AW donor-rule correction proposal

Status: **prospective design correction for review only.** This does not replace
the original frozen rule, authorize extraction, select donors, reduce the donor
target, or launch a hybrid benchmark. The original blocked result remains in
`preparation_manifest.json` and `handoff_validation.json`.

## Why correction is needed

The user and coordinator requested high PTP, low refractory burden,
mostly-rest spikes, depth coverage and full-probe waveform support. During AW
preparation, two additional gates were introduced before checking whether they
exist: `quality == good` and `isolation_score >= 0.80`. They are not user
requirements, and the exact saved QC products contain neither field. Their
absence is therefore an inapplicable preparation rule, not evidence that the
static units are poorly isolated.

The supplemental handoff is byte-verified:

- manifest: `483d7233d31d5b5d95efb6d464d82f4b986c5524a0bb9741a37a625d3689ea13`;
- completion marker: `70b8febef3a3d44c2f8921a2f632154d29264e458f3872f5b7a01017194457c5`;
- 538-row QC CSV: `e0ff5f1a61b2b07a8632dfcaf59dca282fdc213b3983557ee5d67fb1f0aaf7dc`.

The QC table defines `raw_adjacent_isi_lt_refractory_fraction` as the fraction
of adjacent ISIs shorter than the configured 1 ms interval. It also contains
SpikeInterface ISI-violation outputs and a model-based refractory-contamination
estimate. These are not interchangeable and none is an isolation score.

## Proposed rule v2

This entire section is one prospective specification. It must be approved and
versioned before application. If any gate leaves fewer than 30 donors, stop and
report; do not relax it after seeing counts or injected-arm outcomes.

### 1. Identity and provenance gate

Each candidate must have an explicit one-to-one lineage from the saved S/W2
matching-template unit to the final static sorting label and QC row. Numeric ID
overlap alone is insufficient: the template bank contains IDs 0--653, whereas
the final sorting and QC contain the same 538 IDs 0--537. Require saved label
transforms or spike-event lineage proving one parent and one child, with no
merge, split or ambiguous representative. Ambiguous and unmapped units fail.

Bind the candidate to the exact hashes already recorded for the S bank, final
sorting, preprocessing/config, D2L field, AE mask and QC supplement. No motion
arm output participates in identity or ranking.

### 2. Saved scalar eligibility gate

For the mapped final static label, require all of:

1. `spike_count >= 800`. This reuses the previously frozen C2/D2b-2 minimum,
   where 800 total spikes supported a 400-spike STA and avoided interpreting a
   zero-violation handful of events as strong evidence.
2. `raw_adjacent_isi_lt_refractory_fraction <= 0.005`, using the QC table's
   explicit 1 ms adjacent-ISI definition. Do not substitute
   `isi_violations_ratio` or `rp_contamination_estimate`.
3. `rest_spike_fraction >= 0.80`, where the numerator is the final label's
   spikes outside the canonical AE mask and the denominator is all its W2
   spikes. Use half-open mask intervals and the exact W2 sample-to-session-time
   mapping.
4. At least 400 valid rest spikes with full 121-sample waveform support after
   excluding recording edges. This carries forward C2's 400-waveform support.

The current read-only scalar audit finds 90/538 final labels passing the first
three criteria. This is a pre-screen count only, not an eligible or selected
cohort because lineage, waveform stability and full-probe support are untested.

Report `presence_ratio`, `isi_violations_ratio` and
`rp_contamination_estimate` descriptively. Do not gate or rank on them unless a
later instruction explicitly amends this proposal.

### 3. Full-probe construction and stability gate

For each scalar-pre-screened, lineage-resolved candidate:

1. Recreate W2 900--1240 s with the exact 384-channel `ibllikecmr` float32
   preprocessing graph, including reference-before-crop and the saved channel,
   time and gain contracts. AP voltage remains read-only.
2. Rebuild the final-label template with DARTsort's recorded 121-sample
   `TemplateData.from_config`/`peelreduce` path: trough offset 42, median
   reduction, registered templates, prewhiten/postapply strategy, saved SVD
   settings and a 500-spike cap. Do not replace this with a hand-written mean.
3. From the candidate's valid rest spikes, deterministically take 400 samples
   evenly across W2 time, then alternate them into two 200-spike subsets. Build
   a median template from each subset under the same preprocessing/config.
4. Require each 200-spike template versus the 400-spike template to have
   centred cosine >=0.99 and peak-to-peak ratio in [0.98, 1.02]. These reuse the
   already frozen faithful-waveform tolerances; they are not tuned to reach 30.
5. Project the new full-probe template onto AP202--AP383 and require the same
   cosine/PTP tolerances against its lineage-matched saved crop template. This
   is an extraction-reproduction gate, separate from biological stability.

If preprocessing output cannot be certified in physical microvolts, preserve
its recorded unit and stop calling PTP `ptp_uv`. No amplitude threshold is
introduced: high PTP is implemented by ranking after eligibility.

### 4. Two distinct support gates

Both gates are required and must be reported separately.

1. **Conservative interior-support gate:** more than 99% of full-probe template
   energy must lie at least 440 µm from each probe-depth boundary. The margin is
   the 240 µm maximum occupied injection state plus the actual drifty matcher's
   200 µm interpolation neighborhood. This is a stringent support screen, not
   the actual translation measurement.
2. **Actual remap gate:** at every occupied W2 lattice state
   `{0, -40, -80, -120, -160, -200, -240}` µm, exact same-column translation
   must retain >99% energy; exact inverse translation must give PTP ratio in
   [0.98, 1.02] and centred cosine >=0.99. Missing, ambiguous or many-to-one
   channel maps fail.

The earlier 7/654 crop result is only a shallow-crop geometry/support result.
It neither measures full-probe energy nor changes under the QC correction.
AP0--AP201 remain unknown until extraction.

### 5. Placement gate

Freeze injection placement before any arm outcome. For each donor and every
occupied state, require the translated footprint to remain within the verified
full-probe support. Reject a placement if its target peak equals any occupied
translation of the donor's native peak, or if a saved native full-probe
template at that placement matches the proposed injected waveform with centred
cosine >=0.99 and PTP ratio in [0.98, 1.02]. This prevents scoring a native
waveform clone as injected recovery. Preserve exact channel IDs, 30 kHz sample
clock, W2 time origin, preprocessing hash and immutable injected waveforms.

### 6. Ranking and selection

Divide the registered full-probe depth span into six fixed equal strata before
examining eligibility. Within each stratum rank eligible candidates by:

1. larger calibrated static full-probe PTP;
2. smaller 1 ms adjacent-ISI fraction;
3. larger valid-rest spike count;
4. smaller final unit ID.

Take five per stratum. If a stratum is short, fill remaining slots globally by
the same ordering while recording the shortfall. Require exactly 30 unique
donors; otherwise stop. Candidate ranking never uses D2L/S recovery, RF, motion
arm scores or injected outcomes.

The production truth population remains independent per-unit seeded renewal
trains, mean 5 Hz and minimum 3 ms refractory. The synchronized regular 5 Hz
train remains scorer-fixture-only.

## What is already saved versus still requires extraction

Saved arrays suffice for hashes, final-label spike counts, the 1 ms adjacent-ISI
fraction, rest fractions, the 90-label scalar pre-screen, D2L occupied states,
and detection of the unresolved 654-template to 538-final-label lineage. A
small saved label-lineage artifact may resolve identity without voltage.

Voltage extraction is required for certified full-probe PTP, the 400/200-spike
stability templates, crop-overlap reproduction, full-probe energy/support, and
native-clone placement checks. The concrete no-sort remedy remains one read of
the 900--1240 s AP span (about 7.83 GB), an exact 384-channel float32 temporary
recording in RAM (about 15.67 GB), and under 0.2 GB persistent templates and
receipts. The 20--35 minute CPU estimate remains provisional until a dry-run
confirms `TemplateData.from_config` reproduces the saved crop template. No
extraction is authorized by this document.

AV continuity observations remain exploratory. The chronological split, fixed
left/right builder self-review and unpaired representative-template handling
are accepted corrections; T6 supplies no quality inference.
