# Full-session development comparison proposal

Date: 2026-09-06. Status: **prospective proposal; not executed.**

**Subsequent authorization, 2026-09-06:** the user selected full Luke0804 imec0
with native rigid KS4 motion correction versus the accepted rescue baseline.
Execution is recorded in [the active run record](luke_full_session_rigid_v1.md).
The open candidate slot and lack of authorization described below are the
proposal's earlier state. This is a descriptive development comparison; no
production pass margin or promotion is asserted from it.

## Decision context

The 120-second Option A comparison completed both arms but measured the primary
completeness endpoint for only 2 of 53 eligible units. The baseline feasibility
screen found no permitted extension reaching the 50% coverage floor, and
250/500-spike fits did not provide enough accuracy or coverage to replace the
1,000-spike endpoint. More four-window case reviews are diagnostic, not the
next experimental unit.

The next unit should be **one complete development session**. Individual cases
remain diagnostics inside that session; they do not determine whether the
pipeline is useful over the recording.

## Candidate and data policy

The development session must be explicitly named before execution. Any Luke
session or interval used for this comparison is development data; a previously
reserved portion loses held-out status prospectively and must be recorded as
burned. A different session remains sealed for later validation.

The reference is the pipeline currently used operationally: the accepted rescue
pipeline at its frozen production configuration. The candidate must be one
existing implementation/configuration that could plausibly be adopted, with no
parameter sweep or new architecture. The candidate slot is currently **open but
unfilled**: **no candidate is currently justified for execution**. Option A is
not yet supported by a feasible primary endpoint or qualified full-session
field; Option B is closed on its prespecified test; and threshold candidates
are closed by Stage 2.

Candidate selection must produce a versioned candidate ID, resolved settings,
source identity, output namespace, runtime cap, and prospective decision rule
before results are inspected. This proposal does not authorize a new candidate
or a run.

## Smallest executable comparison after candidate selection

Run the complete reference and candidate pipelines over the same full recording,
using the same accepted input, geometry, channel order, sorting settings where
the candidate does not explicitly change them, curation, QC, and standard
exports. Do not substitute the closed 120-second Option A comparison or a
retained-sort snippet replay for the full-session comparison.

The run must emit, for both arms:

- input, environment, settings, source, and output-content receipts;
- full-session spike and unit inventories;
- per-unit, time-resolved 1,000-spike completeness trajectories using
  `full_st[kept_spikes][:, 2]` and historical production indexing;
- valid-fit coverage, no-fit gaps, insufficient-spike intervals, nonfinite fits,
  and boundary-pinned fits, without filling or silently dropping difficult time;
- matched-unit completeness only on exclusive corresponding units and common
  physical-time intervals;
- separate gained, lost, split, merged, unmatched, and ambiguous populations;
- waveform preservation, contamination/refractory distributions, healthy
  interval preservation, condition-dependent recovery where already specified,
  and runtime per unit data.

Completeness is central but is not required to describe every low-rate unit.
The unmeasurable population remains visible in the denominator and coverage
report. A cleaner fit obtained by removing difficult spikes is a failure of the
measurement contract, not an improvement.

## Decision rule

After the full-session run, make exactly one decision: adopt provisionally for
broader testing, reject, or identify one specific unresolved issue that prevents
a decision. Use short cases afterward to explain full-session differences; do
not use cases to select a candidate after seeing the full-session result.

## Gates before execution

Execution-critical work only:

- choose and freeze one existing candidate, or record that no candidate is
  currently justified;
- verify the full-session source and output namespace are disjoint and writable;
- verify reference and candidate settings, clocks, geometry, masks, curation,
  QC, and exports are comparable;
- verify the full-session runtime/storage budget;
- confirm which Luke portions are development data and burn any previously
  reserved portions used by this run;
- reserve a different session for validation;
- write the complete experiment contract and decision margins before execution.

No new sorter framework, estimator study, parameter search, cluster-37 work,
or smaller-window endpoint is part of this proposal. Full-session context does
not solve model mismatch or establish neuronal identity by itself; it supplies
the temporal context and spike counts repeatedly missing from the short
comparison.
