# Luke0804 next stage: two-machine coordination

User authorization in huklaban1 task `01a07da8-01c2-7c71-aee2-c6a3443f0f8b`:
"Let's do that, can you communication and coordinate with the codex agent on
the huklaban5 machine to achieve this?" This accepts the preceding two-machine
plan: reuse/validate the full-probe off reference on huklaban1 and perform one
full-probe rigid registration-only diagnostic on huklaban5, followed by review.

## Ownership

- **huklaban1 — Validate motion correction metrics**
  (`01a07da8-01c2-7c71-aee2-c6a3443f0f8b`): validate the existing reference;
  prepare the fixed-time waveform/completeness evaluation; review motion outputs.
- **huklaban5 — Assess motion correction metrics**
  (`01a07df9-7a1d-7282-811d-52c4745f9869`): implement, verify and execute the
  bounded registration diagnostic; export its source-bound fields and report.
- Shared exchange directory:
  `/mnt/NPX/Luke/20250804/shared_analysis/luke_next_stage_coordination_20260907_v1/`.
  Each machine writes only its own status/results; do not overwrite the other's
  messages or existing handoffs. Use atomic replacement for status JSONs.

The parent can read/monitor the remote Codex task, but the available app tools
do not expose send-message/start-turn. **A shared assignment is not an
acknowledgement or a running job.** Remote work is pending until that agent
reads this file and writes an acknowledgement or starts a verified task turn.

## huklaban1 assignment and execution

Reference:
`/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/`.
Already confirmed: complete 384-channel sort, 314,204,894 samples,
29,999.835983263598 Hz, 12/9 thresholds, effective nblocks=0, dshift=None,
and current RESCUE parameter equality. Existing curation reports 29,227,829
spikes, 710 clusters and 301 KS-good clusters; these are not neuron counts.

Run the read-only `testing.luke_full_reference_reuse_audit` under service
`luke-full-reference-reuse-audit-v1.service`. It verifies pinned source-sort
hashes, downstream request/receipt identity and settings, required QC files,
every retained event's time/amplitude lineage, and the full accepted binary
checksum. It never sorts, re-curates, repairs caches or modifies original data.
Local logs and exact launch/job receipts are under `testing/outputs/`.

While its full-file checksum is running, reserve bulk shared-drive reads for
huklaban1. huklaban5 may develop/test on fixtures and inspect small metadata.
Publish the final source identities and content verification as
`huklaban1_reference_summary.json`, then mark bulk reads released in
`huklaban1_status.json`. Failure stops reuse and preserves all evidence.

After reuse qualification, freeze a manageable diagnostic panel before viewing
a new corrected sort: physical-time quiet/high-motion periods, suspected
dropout and representative interior units, with zero-event bins and explicit
amplitude-fit missingness. Family comparisons require spatial/raw-waveform
support and globally exclusive within-family events; never pool the existing
giant correspondence component. This panel is downstream work, not claimed
complete by the reuse audit.

## huklaban5 assignment — one full-probe rigid registration-only diagnostic

1. Acknowledge in `huklaban5_status.json`, recording task ID, host, local output
   root and current process/job state. Inspect current hardware/workload and
   repository instructions. Read the existing motion audit package at
   `/mnt/NPX/Luke/20250804/shared_analysis/luke_motion_audit_20260907_v1/`.
2. Use the accepted full-probe reference recording above: all 384 channels and
   all 10,473.553728 seconds. Reuse a verified local copy if already present;
   otherwise coordinate bulk reads with huklaban1. Do not allocate an additional
   241 GB binary on /mnt. Source identity and actual sampling rate must match.
3. Hold conditioning, Kilosort 4.0.27, thresholds 12/9 and other RESCUE settings
   fixed. Enable rigid estimation (effective nblocks=1) for this diagnostic.
   Save exact resolved settings and implementation hashes. Confirm 384-channel
   geometry and source recording provenance before GPU work.
4. Inspect the pinned Kilosort API, then implement a runner that executes only
   initialization, preprocessing and `compute_drift_correction`. Stop and close
   the returned binary handle before `detect_spikes`, template learning,
   clustering, curation or full-sort export. Drift estimation has its own event
   detection; retaining those events is intended. Test the boundary so the full
   sorting pipeline cannot be called inadvertently.
5. Use the already verified independent job manager (or verify launcher
   disconnection on a cheap dummy if changing methods). Persist launch command,
   settings, identifier, stdout/stderr and exit code locally. One GPU job at a
   time. No automatic retries; inspect failures. State explicitly what must be
   recomputed after interruption; do not call stage reuse within-sort resume.
6. Save native pre-registration detections/raster support, native `dshift`,
   `yblk`, batch centers, relevant ops/preprocessing, probe geometry and resource
   usage. Export a compact, non-pickled field package with per-file hashes.
   Keep large detection arrays local; share sufficient compact diagnostics and
   source locations, plus the exact replay code.
7. Compare the new field with the accepted strip field and independent
   same-depth/same-time estimates. Use physical displacement=-dshift from the
   verified application rule; subtract acquisition origin 3057.677050340359 s
   from independent time bins; exclude unsupported endpoints rather than
   extrapolating. Report raw and median-centered traces, range/P5–P95, RMS,
   adjacent steps, >50/>100 µm screens, spatial support, and correlations without
   freely optimizing sign/lag/scale. Investigate raster support at excursions.
8. Publish a report classifying the full-probe field as supported, unsupported,
   or unresolved, with continuous metrics and visual evidence. The >100 µm
   screen is descriptive, not an automatically sufficient scientific pass gate.
   Report whether expanding probe support eliminates the strip's pathology.

**Stop after the registration report.** No nonrigid sweep and no full sort are
part of this dispatched job. Preserve `configs/luke_full_session_rigid.HOLD.json`
and failed-run artifacts. A credible field informs the next full-sort decision;
it does not by itself establish biological recovery or remove a cancellation hold.

## Exchange and acceptance

- `huklaban1_status.json`: parent-owned; live verified state, local job identifiers,
  bulk-read ownership and reference qualification state.
- `huklaban5_status.json`: remote-owned; acknowledgement, concrete job/process
  handles, phase, output paths and terminal exit status. A status file alone is
  not proof of liveness; also verify systemd/process state.
- Shared outputs: additive compact manifests, field arrays, metrics, plots,
  code and hashes needed for this authorized collaboration. No raw recording
  duplication or unnecessary curated/PCA array transfers.
- Parent monitors remote task through the app and checks published artifacts;
  dispatch is complete only after acknowledgement. Registration is complete only
  after a successful real job and reviewable verified outputs exist.
