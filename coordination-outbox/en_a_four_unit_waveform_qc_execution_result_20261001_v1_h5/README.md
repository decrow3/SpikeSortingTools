# A four-unit saved-array waveform QC: one-shot result

Exactly one managed execution was invoked under H1 GO manifest `1adde718c916412d1422ecd7e52e2df626311ff436da08e13f4eae1a3aaff2d5`; no retry occurred. The enabled config differs from the reviewed candidate at exactly the four authorized gate/status paths. The reviewed source remained byte-identical.

The process completed and wrote the exact ten declared outputs with `COMPLETE.json` last. It opened the two local retained archives under their fail-closed hash/key/shape/dtype checks, processed exactly 3,912,192 numeric bytes, made zero recording reads, and exported no NPY, NPZ, RAW, BIN, or other waveform payload. The local result tree is 7,110,888 apparent bytes, below the 33,554,432-byte cap. Standard output and error are empty. The transient unit was successfully collected and no process remains.

## Frozen result: FAIL

The result is FAIL under the precommitted technical rules. Integrity controls did not all pass: exact remap passed 16/20 slots and failed four non-q0 unit-278 slots. Their equality fractions were 0.730122, 0.618614, 0.751055, and 0.635029. All four q0 identity controls passed; all eight wrong-sign and eight no-shift negative controls passed; all 20 descriptive CAR correlations were defined.

None of the four units met the frozen usability rule. All 12 corrected-A single-event slots exceeded the local SNR threshold of 6, but every global peak lay more than 80 um from its anchor, so all 12 visibility flags were false. Exact values are preserved in `VERDICT_SUMMARY.json` and the source tables.

The unexpected unit-278 remap failures are an implementation/provenance discrepancy under a construction-integrity control. The consumed contract permits no tuning, retry, or post-hoc rule change. This packet preserves the failure; it does not infer its cause. Any diagnosis would require a separately frozen saved-array audit with a new access budget.

The result does not establish biological identity, purity, motion-estimator correctness, or sorter benefit.

## Implementation checks

- Done: H1/candidate manifests, source, metadata inputs, and exact gate-only enabled-config delta verified before invocation.
- Done: archive hashes, keys, shapes, axes and dtype passed in-process fail-closed validation, evidenced by durable completion after the reviewed validation path.
- Done: exact output set, hashes, COMPLETE-last ordering, zero recording reads, no waveform export, compact size, empty stderr, terminal collection, and no remaining process verified.
- Done: every frozen control count, failed remap slot, single-event SNR/distance/visibility metric, unit usability flag, and final verdict preserved without tuning.
- Not done: memory peak measurement; the transient unit was collected before post-run `MemoryPeak` could be queried, although the accepted managed command records `MemoryMax=536870912`.
- Not done: cause of the unit-278 remap discrepancy; no additional archive opens or post-hoc computation were authorized.
- Can establish: the exact reviewed technical QC completed once and returned FAIL under its frozen rules.
- Cannot establish: biological identity/purity, motion correctness, sorter benefit, or the cause of the remap discrepancy.
