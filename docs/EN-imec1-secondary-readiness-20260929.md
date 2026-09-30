# EN optional imec1 secondary-arm readiness — 2026-09-29

## Verdict

Do not launch the optional EN imec1 rounded-field arm under the frozen EN
design.  A complete imec1 motion-off Kilosort 12/9 raw sort exists, but an
imec1 reference completed through the same curation and QC chain does not.
The design's condition for the optional secondary comparison is therefore not
met.

## Evidence

The available reference root is
`/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec1`.
Its `kilosort4/rescue_sort_manifest.json` reports a complete Kilosort 4 sort
with `Th_universal=12`, `Th_learned=9`, `do_correction=false`, effective
`nblocks=0`, and `do_CAR=true`.  It contains 583 raw units, 216 KS-good units,
and 43,669,711 assigned spikes.

The reference root has no `rescue_sort_identity.json`, no
`cur/curation_receipt.json`, no `qc/qc_receipt.json`, and no
`qc/standard/standard_qc_receipt.json`.  A full file inventory through four
levels contains only the recording and sort manifests among files named for
curation, QC, receipts, or manifests.  Thus the raw sort is usable as existing
descriptive evidence, but it does not satisfy EN's requirement for an imec1
motion-off reference with the same curation and QC chain.

No sort was launched, no voltage was read or written, and no RF or outer
holdout data were accessed for this readiness check.
