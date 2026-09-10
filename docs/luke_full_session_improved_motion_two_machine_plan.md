# Luke0804 imec0: full-session rigid and nonrigid correction

Status: user-requested plan and coordination package, 2026-09-09 UTC.
No estimation, correction materialization, or sorting launched by this plan.

Subsequent instruction: the user explicitly requested pushing the repository
handoff and running motion estimation **once, here on huklaban1**, with the
selected MEDiCINe parameters. See [the estimation run record](luke_full_session_medicine_run_20260909.md)
for implementation and actual launch status. huklaban5 consumes this shared
estimate; it must not launch a second estimation run.

The user subsequently requested automatic rigid correction and Kilosort after
estimation. [The persistent rigid queue](luke_medicine_rigid_queue_20260909.md)
implements that authorization with producer-exit, integrity, domain and runtime
checks. A second manual launch approval is no longer part of this rigid arm's
sequence. Scientific qualification labels and old cancelled-run holds remain.

## Requested experiment

Use the improved waveform-screened motion estimate for two new Kilosort sorts
over the complete Luke0804 imec0 recording: 314,204,894 samples at
29,999.835983263598 Hz, approximately 10,473.55 seconds, with the complete
384-channel source geometry. Run the correction arms on separate machines in
parallel when both have passed preparation and resource checks.

| Role | Host | Motion application |
| --- | --- | --- |
| Existing primary comparator | Reuse verified RESCUE outputs | No correction |
| New rigid arm | huklaban1 | One displacement at each time, derived from improved field |
| New nonrigid arm | huklaban5 | Depth-dependent displacement from the same improved field |
| Historical context | Reuse legacy outputs | Older correction and preprocessing |

The existing full-session baseline must not be rerun merely to create this
comparison. Its recording content SHA-256 is
`2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372`;
sort identity is `22ded4d503b6de8edf4851a08797ae4e594fe41118b913451365727ffbd616ac`.
The prior reuse audit verified 241,309,358,592 recording bytes. Recheck receipt
compatibility and immutability before use. Source:
`/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/`.
Legacy results are contextual because their preprocessing differs.

## One field, two application models

Nominate screened MEDiCINe 5σ/relaxed as the estimator configuration. Reuse
compatible sealed peak evidence and the frozen Luke conditioning/noise inputs.
Keep this estimator-only conditioning out of the accepted sorting voltage.
The same frozen nonrigid estimate supplies both arms, avoiding an estimator
change confounded with rigid versus nonrigid application.

Proposed rigid reduction: the equal-weight spatial mean of the field evaluated
at a fixed, uniformly spaced supported depth grid, at every saved time. Freeze
the depth grid before sorting; no time-varying peak-count weighting or changing
depth membership. This is the least-squares depth-constant approximation on
that grid. Use the same seed-time reference/gauge for both arms. If a fixed
supported depth grid cannot be established, report the gap and resolve it before
creating a rigid field; do not silently average whichever bins survive.
The nonrigid arm retains the complete depth-dependent field. Both use gain 1,
identical correction operator settings where applicable, and KS4 internal
motion estimation/correction disabled. Enabling native `nblocks=1` would estimate
a different field and is not this rigid arm.

## Full-session field preparation is required

The current `testing/luke_screened_medicine.py` requires start=930 and durations
divisible by 20 s. It is not a whole-session launcher. The existing 930–1230 s
benchmark and lighthouse extension are reusable calibration evidence, not a
field for the rest of the recording.

1. Extend extraction to sample-indexed whole-session bounds, partial final
   chunks, safe first/last filtering support, and explicit empty/low-support
   outcomes. Retain hashes, provenance and sealed chunk reuse. Cache reuse
   requires equivalent conditioning, chunk context, timing, and event identity.
2. Inspect cached waveform/peak evidence for shared movement across several
   identities, then inspect representative early/middle/late session support
   and artifacts. Whole-probe waveform identity matching remains independent
   of the field. Preserve strict/lower-score/ambiguous/unmatched observations.
3. Start from one continuous fit if measured memory permits. Freeze a training
   budget after a scaling check: neither 10,000 steps nor the five-minute runtime
   is automatically adequate for a 2.9-hour field. Inspect displacement-bound
   saturation, coverage and residual disagreement. The current internal
   pre-centering bound is ±250 µm, not validated recording-wide capacity.
4. If one continuous fit is infeasible, specify and validate overlapping-window
   offset reconciliation before using segmented fitting. Never concatenate
   independently centered fields or fill gaps by an unvalidated motion prior.
5. Save one immutable full-session field package: native time/depth grids,
   physical units, sign, gauge, validity/support, extraction/fit settings, source
   hashes, and deterministic rigid reduction with its own digest.

Cheaper preparation: reuse the five-minute arrays and test the adapter on short
voltage patches, including zero shift, signed exact geometry translations,
fractional shifts, large excursions and edges. These checks establish mapping
and operator behavior; they cannot answer full-session completeness or identity
continuity and do not substitute for the requested sorts.

## Whole-duration application and boundaries

Pin one external correction implementation/version and apply it once on full
geometry, using the same accepted RESCUE voltage and sorter settings (including
12/9 thresholds) for both new arms. Validate field time mapping from metadata;
do not copy the historical acquisition offset into a recording-relative field.
Test sign, interpolation, nonrigid coordinate mapping/invertibility, amplitude
preservation and final dtype conversion over the actual proposed range.

Unlike the superseded snippet proposal, retain every source sample in time.
Define a versioned boundary policy before launch: use the nearest supported
field value only for short first/last estimator-bin margins, record the exact
duration as extrapolated, and exclude those margins from field-validation
claims. Internal unsupported gaps are not terminal margins and require review.
No silently dropped time or claims of whole-duration validation of extrapolated
samples. Numerical margin limits must be fixed in the executable contract.

Derive a common valid interior output channel set across both application
models and all times, accounting for interpolation support. Prefer retaining
all 384 output channels only if supported; do not fabricate off-probe voltage
to preserve a channel count. Freeze channel policy before sorting. If channel
removal is necessary, both arms use the same channel set, and the unchanged
full-probe baseline becomes a contextually different comparator: report that
limitation and restrict matched metrics to common physical support. Do not
quietly claim a perfectly controlled correction-only comparison in that case.

## Two-machine coordination and storage

Use the new shared directory
`/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/`.
Never overwrite earlier shared assignments or source data.

User-selected delivery: save instructions in this repository and pull them on
huklaban5; no direct SSH dispatch is needed. Start remote preparation with
[the huklaban5 handoff](luke_improved_motion_huklaban5_handoff.md). The shared
directory is the subsequent artifact exchange, not an automatic task launcher.

- huklaban1 owns field preparation, the rigid arm, and final comparison.
- huklaban5 owns remote preflight, the nonrigid arm, and its curation/QC exports.
  It can prepare its environment and operator checks while field preparation
  proceeds on huklaban1. Sorting is parallel after the common field is frozen.
- Share compact code bundles, configs, field arrays, receipts, metrics and
  plots. Publish content hashes and completion markers after writes finish.
  Each host owns its status file; an assignment file is not remote acceptance.
- Keep corrected binaries and sorter/PCA scratch local. Proposed local roots:
  `/media/huklab/Data/luke_improved_motion_20260909_v1/` on huklaban1 and
  `/media/huklaban5/Data/luke_improved_motion_20260909_v1/` on huklaban5, subject
  to actual capacity checks. Prefer existing hash-verified local source copies.
  Schedule shared-drive staging/materialization reads sequentially; once local
  inputs are ready, run one GPU sort per host concurrently.
- Freeze a source bundle with hashes, including required uncommitted files;
  repository HEAD alone is insufficient in the current dirty workspace.
  Verify Python, KS4, SpikeInterface, CUDA, geometry, dtype and resolved settings
  on both hosts. Reuse existing environment checks and a cheap deterministic
  fixture to detect incompatible execution before full runs.
- Require explicit remote acknowledgement identifying hostname, output root,
  free disk/RAM/GPU, existing jobs, source availability and job-manager proof.
  Preserve other active work; no workload termination to claim a GPU.

Preflight observations, not reservations: huklaban1 A5000 reported 24,564 MiB
VRAM, 1,564 MiB used and 0% utilization. Local Data had about 834 GB available;
the shared volume about 489 GB available. Budget source copies, corrected
voltage (~241 GB per full-probe int16 arm), sorter intermediates and exports
before allocating. Do not put two corrected binaries on the nearly full server.
Direct SSH to `huklaban5` failed name resolution even outside the sandbox;
remote live capacity remains unverified. The app can read existing remote tasks
but exposes no send/start tool here. Publishing the handoff does not start work.

## Job lifetime and holds

Use independent systemd user services on both hosts for sorting AND downstream
curation/QC/comparison. Verify launcher-disconnection survival and retained final
exit status on a cheap dummy for each actual launch method. Confirm user-manager
lifetime as well as process ownership. Persist commands, settings, source/input
hashes, service IDs, stdout/stderr and terminal exit receipts. Report liveness
from service/process queries, not old logs or status JSON alone.

Installed KS4 4.0.27 sorting has no validated within-sort checkpoint. Plan and
budget whole-arm restart after interruption; completed field/input stages can
be reused after hash checks. No automatic restart loop. Preserve failed-run
evidence and investigate before relaunching.

The user's new request establishes the scope of this improved rigid/nonrigid
experiment. It does not request restarting the cancelled native-rigid v1 job.
Keep `configs/luke_full_session_rigid.HOLD.json` and its historical outputs
intact, create new run-specific IDs/configs, and document the new authorization
without bypassing unrelated holds or altering production decision 0002.

## Comparison and completion

Compare both new whole-session sorts against the reused no-correction baseline,
and directly against each other. Carry legacy results as a separate historical
comparison. Primary interpretation is descriptive development evidence.

Use exclusive event correspondence, per-unit common-time amplitude-completeness
trajectories, measured-fit coverage and gaps, waveform/event recovery during
movement, and identity continuity. Retain gained/lost/split/merged/ambiguous
populations, contamination, refractory burden and duplicate assignments.
Population spike/unit counts alone cannot decide the result. Use original
voltage and frozen whole-probe lighthouse evidence for waveform checks without
motion-guided identity acceptance. Review severe individual failures alongside
aggregate changes. No retrospective numerical production-promotion gate.

Deliver two terminal sort/curation/QC/export receipts, immutable field/config
packages, per-unit and per-time comparisons with all missing support visible,
figures, and resource measurements. No additional full baseline sort, estimator
sweep or third candidate is part of this request.
