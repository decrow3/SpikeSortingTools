# First medium trained comparison: native path and required saves

Status: **implementation note only; execution disabled**. No recording or saved
voltage was opened, and no sort, training, detection, RF, or holdout operation
was launched while preparing this note.

## Milestone and decision

- Milestone: define the actual Kilosort 4.0.27 trained path and the minimum
  evidence needed for a representative 600-second comparison.
- Decision this note can change: whether a later frozen launch contract is
  implementation-complete enough for independent review.
- Cheapest adequate design: reuse the hash-bound completed `REF384` and `B384`
  outputs, run one fresh repaired `B384` end to end, and run one fresh exact
  `REF384` repeat as a repeatability diagnostic. If the repeat materially
  disagrees with the historical REF384 under precommitted repeatability rules,
  the comparison is inconclusive and the comparators should be rerun rather
  than interpreting the repaired arm.
- Completion condition: the future contract binds all four arm identities,
  adds a reviewed non-smoke trained entry point, freezes completion and
  repeatability rules, and saves the artifacts below. This note is not that
  contract and is not launch authorization.

## Arm identities

| Arm | Input and role | Execution requirement |
|---|---|---|
| `REF384` | Original/reference 384-row binary; completed 600-s arm | Reuse only by exact config, receipt, ops, input-manifest, crop, runtime and artifact hashes. |
| `existing_corrected_B384` | Existing rounded-remap B binary; completed 600-s arm | Reuse on the same identity conditions. It is the current corrected comparator, with no Kilosort drift matrix. |
| `repaired_B384` | The same B binary, crop, 384-row geometry and native settings, with the reviewed pointwise support policy inserted after temporal filtering/CAR and before whitening | Fresh end-to-end run. Only this arm enables the support policy. |
| `REF384_repeat` | Exact REF384 input/configuration repeated in a fresh namespace | Fresh end-to-end diagnostic. It estimates run-to-run implementation variability; it is not a fourth efficacy condition. |

The reusable comparator base is
`configs/en_common_support_crop_screen.v1.json`, SHA-256
`bb6dd8f21539e5ee57f4216702b7c9d4c36d1a0feeefe3859ba25449af52ad17`.
It freezes frames `[208498882,226498783)`, 29,999.835983263598 Hz, 384 rows,
Kilosort internal `nblocks=0`, CAR enabled, 300-Hz high pass, and the native
training settings. The completed REF384 and B384 receipts are respectively
`e3798289f79e51545a2da5fefaee99684c7a667cf863d9ec7cc687f6d1164ac9`
and `8a89ea4330871dce820329ca753890e12877b2695158ce9beddc77e9f190de3b`;
their saved `ops.npy` hashes are respectively
`5b3b365a83babe1dc5842fb92ff5e56f35a93aefe2557a1cd31ca42b46d72f58`
and `7a8f5d219de1b98776134abb00f637777f85ba94113758ae47a1e4b26b2a44f0`.

## Actual native trained path

The installed `run_kilosort` performs the following single native pipeline:

1. Initialize effective ops, then compute the high-pass filter and covariance-
   derived whitening matrix (`run_kilosort.py:219-228,465-522`).
2. Seed NumPy and Torch, build the filtered/whitened `BinaryFiltered`, and run
   the native drift stage (`run_kilosort.py:229-235,534-608`). With the frozen
   settings, `nblocks=0` and the completed arms record `dshift=None`.
3. Learn `wPCA` and `wTEMP` from sampled filtered batches because
   `templates_from_data=true` (`spikedetect.py:49-85,197-204`).
4. Run universal-template detection and features over every batch
   (`spikedetect.py:243-264`), then first clustering and construction of the
   learned matching bank `Wall3` (`run_kilosort.py:653-688`).
5. Run learned-template extraction using `Wall3`, `wPCA`, `iU`, `iCC`, and
   `iCC_mask` (`template_matching.py:56-125`; `run_kilosort.py:683-694`).
6. Run final clustering and merging, then native Phy-format saving
   (`run_kilosort.py:702-770,773-815`; `io.py:340-450`).

The support implementation subclasses `BinaryFiltered` and changes
`padded_batch_to_torch`: it obtains the native temporally filtered/CAR array
with W temporarily disabled, zeroes unsupported channel/time values, and then
applies W (`kilosort_support_mask_candidate.py:530-575`). Because every listed
native consumer reads through this method, a trained repaired run applies the
same support policy to covariance, PCA/template learning, both detection
passes, learned matching, and any preprocessing export.

The currently reviewed runtime entry point is **not** a trained-run entry
point. It accepts only schema `en-minimum-training-support-mask-v4-smoke` and
uses `stop_after_whitening` (`kilosort_support_mask_candidate.py:792-832`). A
future launch therefore requires a fresh, independently reviewed trained-mode
schema/entry point that preserves the same native call chain, removes the
deliberate post-W stop, validates normal native completion, and retains the
atomic fresh-namespace gate. Merely flipping the smoke boolean is not an
adequate reviewed contract.

## What must be independently refit

The support policy changes the samples entering covariance and every later
learning consumer. The following are arm-specific adaptive objects and must be
fit from scratch inside each fresh run:

- covariance `C` and native local whitening `Wrot`;
- learned `wPCA` and `wTEMP`;
- universal-template detections and PC features;
- first-clustering `Wall` and the postprocessed learned bank `Wall3`;
- `iU`, `iCC`, `iCC_mask`, learned-template detections/features;
- final clusters/merges, final `Wall`, exported templates, similarity matrix,
  refractory/contamination outputs, and duplicate-removal mask.

Do not inject the passing smoke C/W into the repaired trained arm. The smoke
arrays prove the frozen boundary only; the trained arm must recompute and bind
its own C/W on its actual crop and schedule. Likewise, do not share REF384 or
existing-B384 PCA/templates with repaired B384. Such sharing would turn the
comparison into a different fixed-basis attribution experiment.

The high-pass filter coefficients may be identical because their inputs are
only fs/cutoff, and geometry-derived neighbor tables can be identical, but
they must still be recreated or verified within each run and saved in effective
ops. Existing REF384/B384 do not retain their pre-W covariance, so reuse of
those completed outputs supports an end-to-end output comparison, not a
mechanistic covariance attribution.

## Required saves for the two fresh runs

Persist these in never-used namespaces; hash the final inventory and write
`COMPLETE.json` last:

1. Contract/provenance: exact command, enabled config and recursive diff,
   approval identity, Python/package/source hashes, resolved recording manifest
   and binary identity, probe/field identity, crop frames and clock, GPU/driver,
   random seeds and determinism flags, unit definition/properties, journal,
   stdout/stderr, terminal exit/result and resource accounting.
2. Preprocessing evidence: actual covariance array plus validation, W array
   plus validation, sampled covariance batch/frame schedule, high-pass/CAR
   settings, mask direction/field binding, and bounded per-consumer support
   summaries. Current audit rows do not identify the pipeline phase when a
   batch index is reused; trained-mode review must add explicit phase labels or
   immutable boundary summaries for covariance, PCA learning, universal
   detection, and learned extraction.
3. Exact adaptive bank before learned extraction: `Wall3.npy`, `wPCA.npy`,
   `wTEMP.npy`, `chanMap.npy`, `iU.npy`, `iCC.npy`, `iCC_mask.npy`, and
   `iU_physical_binary_channel.npy`, with a receipt binding Wall3 axis 0 to
   `spike_detection_templates.npy`.
4. Native outputs: `ops.npy`, channel map/positions/shanks, spike times,
   clusters, final and detection template IDs, positions, amplitudes,
   `kept_spikes.npy`, templates/template indices, similarity matrix, whitening
   matrices, PC features/indices, cluster tables, `params.py`, and
   `kilosort4.log`.
5. With `save_extra_vars=true`: `tF.npy`, `Wall.npy`, `full_st.npy`,
   `full_clu.npy`, and `full_amp.npy`. Keep `save_preprocessed_copy=false`; no
   `temp_wh.dat` or voltage is needed.
6. Validation: effective settings/axes/time origin, global half-open spike
   bounds, finite arrays, aligned event-array lengths, physical template-channel
   ancestry, declared file set, file sizes/hashes, failure markers and exact
   completion state.

For the reused REF384/B384 arms, copy no numeric arrays into the compact packet.
Bind their immutable file inventories/hashes in place and verify the previously
saved pre-extraction banks and native outputs. The baseline repeat should use
the improved evidence schema above. Freeze a repeatability rule before opening
its outcomes; otherwise it cannot serve as a diagnostic gate.

## Source identities inspected

- Installed `kilosort/io.py`:
  `767b76a07647122b45048ba12cdaf49185f87f3fa33334081d4af099825414bd`
- Installed `kilosort/preprocessing.py`:
  `2329b288ae068361a937632a53461e6cceeb27e30191337e9b9a0fa7404d763d`
- Installed `kilosort/spikedetect.py`:
  `db1e8f0357319d9db3530a700bd1002c830337c50fde58b3e74e83e7af140892`
- Installed `kilosort/template_matching.py`:
  `39235cc98428dbb279706718f74a1c3d84568ef31a4f9d1d997a60f7002b1a66`
- Installed `kilosort/run_kilosort.py`:
  `5c1baee40edef14fa7566e441997da5d2ecc4f7940b8385cf83de40c8bd0f945`
- Support-mask candidate:
  `705b4af444699a6e40ddd0e8a189232b619c4e1bbff53f8150b448f5ad160391`
- Support-mask geometry helper:
  `9c981047f94d91a930e17bd51e634fbc96faa5e956c85d0882f3a603f7f42e0f`
- Historical crop runner:
  `800b190b1ad7f0bb832f310c2b86bf4d6802b63ae5990169138bb86ce5d6c897`

Repository commit was
`32d04e7d5d8f15e9bd895ff3a4e633f75088b156`; the inspected project sources
are untracked working-tree files, so the hashes above, not the commit, are the
review identities.

## Interpretation boundary

This design can estimate an end-to-end difference among the frozen original,
existing-corrected and repaired-corrected pipelines while exposing whether a
baseline rerun is stable enough to interpret. It cannot isolate a mask-only
mechanism because C/W/PCA/templates are intentionally refit. It cannot establish
biological identity, purity, motion correctness, long-window generalization, or
RF performance.

## Implementation checks

- Done: actual installed native call chain inspected -> covariance/W precede
  learned PCA/templates, two detection stages, final clustering and saving
  (`run_kilosort.py:219-259,465-522,614-770`).
- Done: support consumer boundary inspected -> the subclass masks pre-W and is
  consumed by covariance, PCA learning, detection and learned extraction
  (`kilosort_support_mask_candidate.py:530-575`; installed consumer call sites).
- Done: existing comparator provenance inspected -> frozen config, REF384/B384
  receipts, effective ops and pre-extraction snapshot identities are present.
- Done: adaptive dependency question -> changing support requires independent
  C/W, wPCA/wTEMP, Wall3, detection/features and final clustering/templates.
- Done: current entry-point suitability -> smoke-only schema deliberately stops
  after W and cannot launch a trained run without a fresh reviewed delta.
- Not done: fresh arm execution, recording access, trained output validation or
  scorecard evaluation -> prohibited by this preparation-only assignment.
- Not done: independent review of a future trained entry point and repeatability
  rule -> prerequisites for launch.
- Can establish: the native dependency graph, reusable comparator identities,
  required refits, and minimum provenance/artifact saves for contract design.
- Cannot establish: successful trained execution, comparator repeatability,
  sorter benefit, biological identity/purity, or promotion to long/full data.

