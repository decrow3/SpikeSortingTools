# 0016 — Longitudinal, evaluator-first pipeline development

Date: 2026-09-06. Status: accepted development policy; no production pipeline
change is implied.

## Decision

Pipeline development will use long continuous recordings, preferentially a
depth-reduced processing band with a physical halo and an interior scoring band.
The evaluator must pass named negative controls before it ranks real pipelines.
The initial search is restricted to mature preprocessing, native motion and
bounded threshold variants with frozen curation.

Candidates advance on practically meaningful longitudinal completeness and
identity improvements while satisfying contamination, duplication, split/merge,
waveform and boundary guardrails. Unit yield remains descriptive. Passing arms
are compared as a Pareto set rather than collapsed into a composite score.
Custom methods require a repeatedly demonstrated residual failure that names
their target.

The policy is enforced by `testing/development_ladder.py`, the
`python -m testing.run_development_ladder` command and versioned experiment contracts based on
`configs/example.development_comparison.v1.json`.

## Evidence and scope

The rationale and the investigation failures this corrects are recorded in the
[hindsight-first prescription](../spikesorting_hindsight_development_prescription.md).
This decision changes development sequencing and evidence requirements. The
accepted rescue graph remains the reference, not a declared biological winner.
Existing production decisions remain in force until a candidate completes the
full validation sequence and receives a separate adoption decision.
