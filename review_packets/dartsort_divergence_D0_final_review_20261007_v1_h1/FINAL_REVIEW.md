# H1 focused independent review — DARTsort divergence D0 inventory

## Verdict

**ACCEPT_COMPARISON_1_WITH_REQUIRED_CAPABILITY_CORRECTIONS.**

The D0 packet is sealed correctly and comparison 1 is the least duplicative
saved-output investigation among the two inventoried choices. Comparison 2 has
already progressed through an input-equivalence audit, one harmonized W3
control, saved-output scoring, and independent review. Its authoritative result
is `VALIDATED_INCONCLUSIVE_NO_PROGRESSION`, with automatic follow-up false.
Therefore another generic full-versus-window comparison is not the cheapest
next test.

The proposed comparison-1 test is appropriately bounded as a pipeline-level,
saved-output diagnostic. It can compare REF--A, REF--DARTsort, and
A--DARTsort event correspondence on the shared AP-frame-zero clock while
preserving unmatched and ambiguous events. It cannot isolate a sorter effect:
the three recordings differ in preprocessing, channel construction, remap,
scaling, and internal preprocessing.

Two statements must be corrected before the next contract is frozen:

1. `matching1.h5` labels are the saved membership entering the downstream
   PCMerge/TMM/agglomeration chain, not final DARTsort labels. The parent HDF5
   row is stable into the final NPZ, and bounded prior probes found corresponding
   times/channels, but all 62 probed matching labels differed from final labels.
   Thus “row-aligned labels” must not be read as label equality or a final-label
   checkpoint. The next contract must bind parent row identity and keep
   matching-stage and final-stage labels as distinct fields.
2. Replay capability is stage-specific. A matching-stage replay may be
   technically possible but remains unqualified. Exact comparable final-label
   downstream replay is unavailable from saved artifacts alone: PCMerge and TMM
   memberships/models were not saved, and post-TMM template estimation plus
   agglomeration requires recording voltage. Mark the latter unavailable unless
   a separately frozen voltage-backed replay is authorized and qualified.

The Kilosort gap is also stated correctly as a binding gap, not an absence of
source code. Kilosort 4.0.27 producer semantics are inspectable in the pinned
project environment/source, including `full_st`, `full_clu`,
`kept_spikes`, `spike_detection_templates`, and `spike_templates`. What D0 does
not yet establish is that the exact executable package bytes/source that made
REF and A are bound to those run receipts. The follow-up must bind the executed
Kilosort package/source and the curator's exact transformation, then validate
the mapping on known-answer fixtures before treating an intermediate as a
detection or matching checkpoint.

## Exact bindings independently checked

- All four D0 payload members match `MANIFEST.json`; the manifest SHA-256 is
  `d3ac54f8e1fa56536405dde9db138a9bee69655c2835ac749189880c7db67c64`
  and `COMPLETE.json` SHA-256 is
  `b7364341a3494c42536a91334c8bed364305b9db9258983df412ddbe64456376`.
- The selected DARTsort final NPZ hash `3b324bbc...ade660`, clock
  29999.759166666667 Hz, global `[0,314204094)`, and frozen source/config
  identities agree with prior sealed EM audits and the reviewed downstream
  feasibility trace.
- REF identity `7bc7ce4c...50a80d`, release commit `32d04e7d...8156`, the
  314204094-frame clock, and the accepted recording identity agree with the
  prior final review. A's recording content identity `f4eaa1c4...b8899` and
  exact clock agree with the reviewed Part-2b recording manifest. The exact
  A curated payload itself is H5-local and was not independently traversed on
  H1.
- Comparison 2's final status, point estimate, interval, failed yield
  guardrail, `progression_supported=false`, and
  `automatic_followup_authorized=false` were read directly from its sealed
  `FINAL_VERDICT.json`.

## Required next-contract changes

- Replace any generic DARTsort “row-aligned labels” wording with separate
  `parent_row_id`, `matching_label`, and `final_label` semantics. Add positive
  and negative fixtures for duplicate times, time inversions, row reorder,
  feature-only reorder, and stage-label substitution.
- Split replay status into matching-stage and downstream-final status. Do not
  imply that saved matching models are a complete final-label recovery
  checkpoint.
- Bind REF and A independently to exact Kilosort executable/package source,
  SpikeInterface wrapper, compatibility patch, effective settings, and
  curator source. A version string or repository commit alone is insufficient.
- Keep the proposed 3--6 neighborhoods a diagnostic sample. Pre-freeze their
  selection from the existing panel, document its mostly within-Kilosort
  origin, and do not claim cross-sorter representativeness.

## Implementation checks

- Done: packet integrity -> all four payload hashes/sizes, MANIFEST, and
  COMPLETE independently verified.
- Done: exact clocks and source/input identities -> compared D0 bindings with
  prior sealed EM, REF, A-recording, and W3-control evidence.
- Done: matching/final stage semantics -> reviewed executed DARTsort recovery
  trace and the 62-row known-answer/bounded real-row probe; matching labels are
  not final labels.
- Done: replay semantics -> reviewed the executed downstream trace showing
  missing PCMerge/TMM state and voltage-backed post-TMM
  template/agglomeration dependencies.
- Done: comparison-2 reuse -> authoritative final verdict read directly;
  generic repetition is duplicative and unsupported.
- Done: Kilosort semantics availability -> inspected pinned Kilosort 4.0.27
  export source and project loader; source exists, while exact REF/A executable
  binding remains absent from D0.
- Not done: H5-private REF/A or DARTsort event traversal -> prohibited by this
  focused review and unnecessary for reviewing D0's inventory decision.
- Not done: exact REF/A installed-package byte binding and curator consumer
  trace -> required before detection/matching-stage attribution.
- Can establish: comparison 1 is the cheapest nonduplicative saved-output
  diagnostic of the two inventoried choices, subject to the corrections above.
- Cannot establish: biological identity, purity, sorter-only causality,
  cross-sorter checkpoint equivalence, or feasibility of exact final-label
  replay from saved artifacts alone.

