# First-release decision proposal — no candidate currently justified

Date: 2026-09-06. Status: **proposal; no new execution authorized by this record.**

## Decision

**No candidate is currently justified for first-release testing or promotion.**

The current Option A experiment is closed as an engineering completion with an
inconclusive scientific result. Its primary endpoint was measurable for only 2
of 53 eligible units, or 3.77%, below the frozen 50% coverage floor. The
baseline-only feasibility screen found that even the longest interval inside
the nominated development window measured 38 of 124 baseline-defined units
(30.6%). Extending the interval also failed the measured support policy.

This is an evidence limitation, not a candidate failure. It does not justify
calling external registration ineffective, and it does not justify launching a
longer comparison without a new feasible contract.

## Candidate disposition

| candidate or branch | existing evidence | first-release disposition |
|---|---|---|
| Option A, `ks-motion` external registration | The 120 s arms completed; support passed locally, but completeness coverage was 2/53. The baseline feasibility screen found no permitted duration reaching 50%, and the 2,880 s extension failed support. | **Not justified now.** No v2 contract or rerun. |
| Option B, unwarped identity v2 | Safe partition of original clusters; 0 splits and 18 conservative unvalidated merges. The nominated cluster-37 completeness change was 0.0 pp against a 20.0 pp margin; candidate closed. | **Closed.** Do not reopen from this record. |
| Threshold candidates 8/8 and 9/9 | Frozen Stage 2 donor-bootstrap intervals crossed zero; decision was no threshold change. | **Closed.** No further threshold work. |
| A future targeted curation repair | A real amplitude-completeness audit established that recognizable dropout cases exist, but the available evidence does not nominate a specific repair that preserves identity, contamination, and healthy intervals. | **Candidate slot remains open, not selected.** A repair becomes testable only after its exact intervention and supporting evidence are frozen. |

Thus the answer to the first-release candidate question is deliberately:
**no candidate currently justified**.

## Two denominators, kept separate

### Baseline-measurable completeness cohort

This denominator is used only for the amplitude-completeness primary endpoint.
It must be defined from the baseline before candidate results are inspected and
must use the production source `full_st[kept_spikes][:, 2]`, historical window
indexing, 1,000-spike windows, `max_isi_s=10`, and at least two finite-interior
fits. Units with too few spikes, long gaps, insufficient finite fits, boundary
pinning, or nonfinite fits remain in the denominator and are reported by reason.

For the closed Option A feasibility screen:

- current 120 s run: 53 eligible units, 30 exclusive reciprocal matches, 2
  measurable paired units;
- maximum permitted 2,880 s nominated interval: 124 baseline-defined units,
  38 measurable units (30.6%);
- required coverage: 50% of the fixed baseline denominator.

Completeness is therefore currently an **inconclusive endpoint**, not a yield
endpoint and not a license to remove hard-to-measure units.

### Whole-population preservation safeguards

These safeguards are evaluated on the full fixed baseline-defined population,
not only on units with a successful completeness fit:

- exclusive identity correspondence and explicit ambiguous/unmatched counts;
- identity coverage and no candidate-created split or merge relationship that
  is not validated;
- contamination/refractory change for every matched unit with a valid QC value;
- waveform cosine and peak-retention distributions by unit, including lower
  quantiles and relevant amplitude/depth strata;
- healthy-control interval preservation;
- condition-dependent recovery and artifact-adjacent behavior where already
  specified by the frozen contract;
- runtime and output-integrity receipts.

Unmatched or ambiguous units are not improvements. A candidate cannot pass by
showing a favorable completeness median over the small measurable subset while
discarding the rest of the population.

## Smallest executable comparison if a candidate becomes justified

No comparison is launched by this proposal. The smallest future comparison is
one candidate versus the current rescue control, using the existing first-
pipeline contract and one frozen development failure plus its reserved healthy
controls:

1. Freeze the candidate intervention, exact settings, source identity, output
   namespace, endpoint amendment, and candidate count before inspecting output.
2. Run the current rescue control and candidate through the same accepted input,
   sorting or retained-sort replay, curation, QC, waveform extraction, and
   export path.
3. Define the baseline-measurable completeness cohort before candidate results;
   report the whole denominator and every exclusion reason.
4. Match units exclusively using waveform, spatial, and event criteria; retain
   unmatched and ambiguous units in the population safeguard denominator.
5. Apply completeness only to supported reciprocal pairs with two finite-
   interior fits in both arms and require the declared coverage floor.
6. Evaluate whole-population identity, contamination, waveform preservation,
   healthy-interval preservation, and runtime alongside the primary endpoint.
7. Return exactly one of `pass`, `fail`, or `inconclusive`. A coverage failure
   is inconclusive; it is never converted into a pass by changing duration,
   fit requirements, or the denominator after seeing results.

The existing Option A result supplies the orchestration and export path for
this template, but not a feasible completeness domain. The next candidate must
therefore bring its own pre-execution feasibility evidence or remain unrun.

## Release recommendation

Retain the current rescue pipeline as the operational reference. Do not promote
Option A, Option B, 8/8, or 9/9. Do not reopen the closed Option B cluster-37
branch, launch a full-session comparison, substitute unit yield for completeness,
or build another framework.

The next milestone is a candidate with a measurable primary endpoint and a
whole-population safeguard plan. Until that exists, the correct first-release
decision is **no candidate currently justified**.