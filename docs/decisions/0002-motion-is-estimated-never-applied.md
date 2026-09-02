# 0002 — Motion is estimated and recorded, never applied to voltage

**Status:** Adopted; initial production milestone implemented 2026-08-31
**Implementation:** `pipeline/motion_sidecar.py`, tests in `testing/test_motion_sidecar.py`

## Decision

> Estimate rigid motion with DREDGE, save the estimate and its quality-control
> evidence, and sort the exact accepted recording with KS4 motion correction
> disabled.

This holds **regardless of whether DREDGE succeeds, fails, or returns an
unqualified estimate**. The motion field is an observational sidecar and a
diagnostic, never an input to voltage.

The governing invariant: **spatial resampling of voltage for motion correction
is forbidden.**

## Why

The historical failure path ran from a nonrigid motion estimate to automatic
voltage warping, and that warp was implicated in loss and relocation of
otherwise sortable spikes. Routing motion into an artifact rather than into the
signal makes severe-motion sessions diagnosable without reopening that path.

Identity is implemented by **routing, not by a zero-motion operator** — the
recording handed to KS4 is the accepted recording itself, not a
motion-corrected recording that happens to have zero displacement. This matters:
a zero-motion operator would still resample.

## Explicitly out of scope

Selective rigid voltage correction is **unimplemented and unauthorized**. It may
be implemented only after a separately versioned crossover policy and its
operator have passed the required validation. Do not add it because a motion
estimate looks large.

## Contract

- Accepted recording contract: `rescue-recording-manifest-v2`, which includes and
  verifies full binary SHA-256 receipts before estimation and sorting.
- Motion artifacts are accepted only after every required core array, figure, and
  requested split-half terminal artifact has been hashed into the final manifest.

## Reopening conditions

A future selective rigid correction branch requires: a versioned crossover
policy, a validated operator, and advancement metrics agreed in advance. It does
not inherit authorization from this record.

## Evidence pointers

- `docs/conservative_dredge_sidecar_ks4_implementation_plan.md` (§2 architecture,
  §11 identity guarantee, §18 final production policy)
