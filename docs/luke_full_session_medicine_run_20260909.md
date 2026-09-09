# One shared full-session MEDiCINe estimate on huklaban1

User instruction: "we should only do the motion estimation on one machine,
since that is shared. Let's do that here using the found best parameters for
MEDiCINe." This authorizes full-session estimation here; huklaban5 consumes
the output for its nonrigid correction arm. No sort is launched by the estimator.

## Frozen first full-session configuration

- Luke0804 imec0 accepted RESCUE recording: all 314,204,894 samples,
  384 channels, 29,999.835983263598 Hz; approximately 10,473.55 s.
- Documented primary candidate: 5σ negative locally-exclusive detection,
  50 µm radius; relaxed screen (SNR≥6, central energy≥0.5,
  qualified-neighbor cosine≥0.6) with unchanged width/broad/shared gates.
- Existing third-order 300–6000 Hz filter, 50 ms context, frozen 31-tap
  compensation and original noise vectors. These condition estimation input;
  source/sorting voltage is unchanged. Reflect only at physical file endpoints.
- MEDiCINe: one continuous field, motion_bound=500 (internal pre-centering
  ±250 µm), 0.25 s bins, 1 s triangular kernel, four depth bins, network
  (256,256), Adam lr=0.0005, batch=4096, 10,000 steps, seed=0, float32 CUDA,
  initial motion noise=0.1 annealed over 2,000 steps, epsilon=0.001,
  no extra amplitude-quantile rejection. No estimator sweep or stitched windows.

"Best" means the documented primary shortlisted configuration, not universal
superiority. In the newly available 1030–1230 s extension, strict per-candidate
median discrepancies aggregate to 8.24 µm overall / 65.10 µm for large excursions
with this setting; 30,000-step 5σ/relaxed gives 7.80 / 50.19 µm and 6σ/full gives
6.40 / 49.95 µm (14 overall and six excursion contributors). Thus the initial
100 s ranking does not settle every later interval. Preserve those comparisons;
do not describe a completed full-session fit as validated motion correction.

## Implementation and preparation checks

`testing/luke_full_session_medicine.py` adds disjoint sample-indexed chunks,
partial first/last chunks, streamed complete-binary SHA-256 verification,
explicit empty retained populations, bounded feature and peak-array copying,
and one fit plus compact field/support reports. Compatible 930–1230 s chunks
are reused only after hash/provenance checks; one-sample boundary mismatches
are recomputed. Detector edge exclusions remain explicit at every chunk.

The control first refits saved 930–1030 s inputs and must reproduce the cached
primary field within 0.1 µm on identical grids. This is software reproduction,
not a scientific error threshold. Final arrays preserve native displacement;
separate candidate arrays use the frozen 930–940 s seed reference and contain
the nonrigid field plus its equal-depth rigid projection. That projection's
grid denotes model support, not verified cell coverage or a qualified voltage
application domain.

Four targeted tests cover full-sample partitioning, reflected endpoint reads
and streamed checksum identity, peak time concatenation with empty chunks,
and refusal of interrupted/corrupt stages. Existing feature-batching and
localization-independence tests are run as well. Real-voltage smoke checks
compare a cached 20 s extraction and exercise the first/last second.

## Job contract

Dedicated launcher: `testing/launch_luke_full_session_medicine.py`. It freezes
the needed source files into a private job bundle, pins input hashes and all
resolved parameters, and requires a successful dummy/disconnection proof.
The systemd user manager has lingering enabled so logout does not own the job.
One service owns extraction, fit and report; no detached shell children,
sorter, curation or automatic correction is launched.

Proposed actual launch (consult receipts below for execution status):

```bash
environments/rescue-production/.venv/bin/python -m testing.launch_luke_full_session_medicine \
  --unit luke-full-session-medicine-20260909-v1 \
  --job-dir /media/huklab/Data/luke_full_session_medicine_20260909_v1_job \
  --output /media/huklab/Data/luke_full_session_medicine_20260909_v1 \
  --proof /home/huklab/Documents/RyanSorting/SpikeSortingTools/testing/outputs/luke_full_medicine_dummy_20260909_v1_job
```

Runtime state is `progress.json` plus actual systemd/process inspection.
The sibling `_job` directory contains launch commands, frozen code/config,
stdout/stderr, managed `receipt.json`, and systemd `service_result.json`.
Completed sealed chunks/fits are reusable only on explicit audited relaunch.
There is no within-chunk or optimizer checkpoint; interrupted work in that
stage must restart after investigation. Old native-rigid sort holds remain.

Completed compact artifacts are published under
`/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/estimation_huklaban1_v1/`
with a root `field_manifest.json`. The field is marked `requires_review` and
`correction_ready=false`. Native/seed-referenced fields, rigid projection,
coverage, figures and provenance are shared; peak caches and voltage stay local.
No automatic sorting follows artifact publication.

## Status

Launched as `luke-full-session-medicine-20260909-v1.service` on
2026-09-09 at approximately 04:52 UTC. Actual service inspection after launcher
exit showed active/running with MainPID=2195155. All six tests passed. The
fresh 930 s extraction reproduced all 19,616 cached retained peaks exactly;
localizations agreed to 1e-6 tolerance. First/last-second checks also passed.
The new fit wrapper reproduced the saved 100 s primary field with **zero maximum
difference** before full-session extraction began.

This records launch-time evidence, not completion. Inspect current service and
process state, `progress.json`, and terminal receipts for current status. The
run is not complete until the fit/report artifacts and successful independent
process/service exit receipts exist. Source runs from its frozen job bundle,
so subsequent repository pulls/commits do not alter the executing code.
