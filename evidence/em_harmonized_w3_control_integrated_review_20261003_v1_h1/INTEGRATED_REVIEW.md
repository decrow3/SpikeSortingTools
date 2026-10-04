# H1 integrated review — input-harmonized W3 control

## Decision

**NO_GO_REPAIR_REQUIRED. Do not launch.**

The input side passes review: the package binds the historical full-session
cache to full sort SHA-256 `3b324bbc...de660`, exposes exactly global frames
`[239998073,250197991)` as 10,199,918 local frames, preserves 182-channel
float32 geometry/order, and applies no refiltering, scaling, reference, motion
fit, or second nontrivial crop. An independent row-major little-endian reader
and the SpikeInterface view matched exactly on 769 frames, including beginning,
end, seven seams, and +/-1 around all 248 reported state transitions. The
preflight read 1,119,664 saved-cache bytes and zero raw-recording bytes.

The scientific contract also has the requested structure: H-F is primary with
four frozen categories at +/-0.05; H-L is secondary and non-decompositional;
F/L/H share one joint common-block resampling matrix; L-based yield and all
three segment-safe 9-29-sample interval guardrails and thresholds are explicit;
undefined results fail closed.

Launch is blocked by two load-bearing implementation gaps and one release-
preflight inconsistency.

## Required repairs

### 1. Freeze the scorecard's executable dependency closure

`source/harmonized_scorecard.py` imports `intervals` from the live checkout's
`testing.de_common_outcome_scorecard` and `DOMAINS`, `arm_inputs`,
`block_domain_exposure`, and `bootstrap_rhos` from the live checkout's
`testing.em2b_w2_scorecard` (lines 15-20). Neither dependency is included in
the packet manifest or `EXTERNAL_BINDINGS.json`, and the scorecard does not
verify their hashes before use. The reviewed script and tests could therefore
execute different measurement/bootstrap semantics after checkout drift.

Repair by vendoring the exact dependency sources into the new packet or adding
their exact hashes to the frozen contract and enforcing those hashes before
import. The focused tests must run against that frozen dependency closure.

### 2. Provide a managed terminal evaluation/sealing path

The contract's completion condition requires an immutable scorecard/guardrail
receipt after a successful sort, but `launch/COMMAND_TEMPLATE.json` only starts
the historical dataset-pipeline sort (lines 6-13). No execution command or
independent job-manager unit exists for `harmonized_scorecard.py`, output
manifest generation, or terminal sealing.

Repair with an execution-disabled managed post-sort unit/chain that:

- starts only after the H sort has a successful terminal marker;
- verifies the frozen H/F/L and dependency hashes;
- runs the frozen scorecard exactly once with no retry;
- writes logs and an immutable result manifest/terminal receipt;
- preserves scorecard failure separately and never promotes or cleans output;
- survives coordinator/chat disconnection.

The repair may be an `OnSuccess=` service or a separately released managed job;
it must not rely on an interactive agent returning after the sort.

### 3. Make the release resource checks internally consistent

The contract stops below 25 GiB free, while the executable pipeline config
enforces only a 20-GiB reserve. The >=160-GiB `MemAvailable` condition and
single-device requirement are documented but are not represented in the
launch template. Freeze a release-preflight receipt that rechecks packet hashes,
output absence, cache identity, >=25 GiB free, >=160 GiB available RAM, the
intended single GPU/device, and the <=60-minute managed wall immediately before
installation/start. Align the executable disk threshold with the 25-GiB rule.

The absent historical peak RSS is acceptable as a disclosed limitation under
the user's documented-resource-condition authorization; it does not need to be
invented or retrospectively estimated.

## Reviewed controls

- One fresh H sort; seed 0; no learned-model reuse; output must not exist.
- Exact historical effective/internal sorter configuration hashes are bound.
- One GPU (`cuda:0`), four CPU jobs, <=340 seconds input, <=60 minutes managed
  wall, no retry, bounded saved-cache preflight, stage logs, and no automatic
  cleanup are frozen.
- F and L sorting hashes and domain-input hashes are frozen; H must be tied to
  its terminal marker.
- Stable time sorting moves labels with samples, and every arm must remain in
  local `[0,10199918)` before one global-frame mapping.
- Practical equivalence uses inclusive `[-0.05,+0.05]`; advantage/worse use
  strict outer bounds; all other or undefined outcomes are non-advancing.
- RF, outer holdout, target A, long/full work, retries, searches, and production
  promotion remain excluded.

H1's earlier `em_full_vs_w3_equivalence_h1_final_review_20261003_v1_h1`
proposal remains superseded historical evidence. This review does not modify
or overwrite it.

## Implementation checks

- Done: packet integrity -> all 18 H5 manifest members verified; H5 manifest
  SHA-256 `b4686ace...0f9c`, COMPLETE SHA-256 `a5e2a079...54af`.
- Done: what would run -> exact cache view, historical runner/config diff,
  launch command, scoring script, tests, failure receipts, and disabled service
  state inspected.
- Done: axes/clock/geometry -> authoritative interval, local/global mapping,
  sample rate, dtype, channel order, and geometry are bound and independently
  sampled.
- Done: scoring/guardrails -> H-F categories, H-L scope, joint bootstrap,
  stable ordering, L anchors, yield rule, and segment-safe interval rules traced
  through the executed source.
- Done: current state -> output absent, prospective service not installed,
  zero sorts/GPU/scoring runs, and zero raw-recording bytes.
- Not done: frozen scorecard dependency closure -> missing from the packet and
  not hash-enforced at runtime.
- Not done: durable post-sort scoring/sealing -> no managed command/unit exists.
- Not done: release resource receipt -> 25-GiB disk and 160-GiB RAM checks are
  not yet aligned/enforced in the launch package.
- Can establish: the read-only H input view and scientific decision contract are
  technically coherent and need only bounded operational/provenance repairs.
- Cannot establish: launch readiness, H's outcome, identity/purity, causal
  decomposition of H-L, or suitability for long/full/production use.
