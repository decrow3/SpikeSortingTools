# Review candidate: H5-local A four-unit waveform QC

This packet freezes a non-executable channel-level waveform-QC contract for the completed A capture. It follows the independent post-execution recommendation bound by manifest `fbf9aa3a79b0580477b7d75c497df45715436868351e5f4eb2da814f2a9abe00`.

No retained outcome array was opened while preparing or testing this packet. The preflight returned `outcome_archives_opened=0`, the planned output remains absent, and all six tests use synthetic known answers only. No recording, service, sort, training, detection, RF, or holdout was touched.

## Frozen decision

The only decision is whether these selected signals show technically coherent original-versus-corrected waveform/stage behavior worthy of informing the first trained comparison. The result may not be interpreted as biological identity, purity, motion-estimator correctness, or sorter improvement.

The contract binds all 20 slot identities and exact archive hashes. Arrays are channel-by-time int16 ADC counts. Single-event index 60 is the event time; pair endpoints are indices 60 and `60+lag`. Signal and baseline windows, exact channels, geometry, remap sign, raw/centered/CAR stages, channel metrics, robust noise normalization, controls, thresholds, tie-breaking, exclusions, output schema, and resource bounds are frozen in the config.

For each state, the corrected target at `(x,y)` must equal the original source at `(x,y+state_um)`, with unsupported targets exactly zero. All four q0 members must be whole-array byte-identical. Wrong-sign and no-shift mappings are outcome negative controls and must stay below 0.99 exact equality for each of the eight displaced ordinary slots. These comparisons are fixed before outcome inspection.

Visibility is assessed only on the 12 single-event slots. A slot is visible when corrected-A CAR-centered local peak-to-peak SNR is at least 6 and its global peak is within 80 um of the frozen corrected anchor. A unit is usable with at least two visible single-event slots. PASS requires every integrity/remap/control check plus at least three usable units; CAUTION requires exactly two; otherwise FAIL. Pair slots are retained for exact-remap and morphology reporting but cannot count toward usability.

Native high-pass filtering is deliberately excluded from channel-level recomputation: the retained arrays lack the original 512-sample padding, so applying Kilosort's FFT filter to 121–150 samples would change boundary semantics. The already-reviewed aggregate capture table remains prior execution context, not a channel-level outcome for this QC.

## Resources and outputs

The eventual reviewed execution may open each archive twice (hash plus load), retain at most 3,912,192 numeric input bytes, use at most 512 MiB RAM, two CPU threads, 120 seconds, and 32 MiB persistent output. Recording-read allowance is exactly zero. It must stop on any gate/hash/key/shape/dtype/resource/fresh-output mismatch and write `COMPLETE.json` last.

No voltage will be published. Any eventual compact result packet may publish tables, decisions, plots, and hashes only; the two local NPZ archives remain H5-local.

## Review request

Independent H1 review is requested through `REVIEW_REQUEST.json`. Review must inspect only this packet and synthetic fixtures, without opening real outcome arrays. A later execution requires a new hash-bound enabled config; this candidate must remain immutable.

## Implementation checks

- Done: post-execution recommendation and capture identities bound to exact manifests.
- Done: unit/state anchors derived from the hash-bound geometry and exact remap convention before outcome inspection.
- Done: selection/window/axis/scale, duplicate semantics, per-stage/channel metrics, robust summaries, exact comparisons, controls, exclusions, thresholds, outputs, resource bounds, and completion conditions frozen.
- Done: six synthetic known-answer fixtures passed; disabled preflight opened zero outcome archives.
- Not done: real archive hash/key/member validation, waveform metrics, plots, or decision; execution is intentionally disabled pending independent H1 review.
- Can establish: the candidate is a bounded, auditable analysis plan with executable fail-closed implementation and outcome-independent rules.
- Cannot establish: any real waveform result, biological identity/purity, motion benefit, or trained-sorter effect.
