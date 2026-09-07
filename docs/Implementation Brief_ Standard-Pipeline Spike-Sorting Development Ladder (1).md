# Implementation brief: build the hindsight-first spike-sorting development ladder inside the existing repository

## Objective

Implement a **simple, reusable development framework for comparing mature spike-sorting pipeline configurations on long recordings**, using the substantial infrastructure that already exists in this repository.

Do **not** build a new spike-sorting architecture.

Do **not** refactor the repository broadly.

Do **not** resume the old peeler, stitcher, DARTsort, KIASORT, custom DREDGE-warp, claim-mask, or synthetic-motion programs as part of this task.

The purpose of this implementation is to answer:

> Which practical, mostly standard sorting pipeline best preserves spikes and neuronal identity through long recordings, without unacceptable contamination, duplication, fragmentation, merging, or waveform instability?

The immediate development platform is Luke0804, but the implementation should be sufficiently generic that another accepted recording can be substituted later without rewriting the evaluator.

---

# 1. General implementation philosophy

The repository contains much more experimental code than is needed for this task. Treat existing code as three classes:

### A. Production-grade infrastructure: reuse directly

Prefer composition over reimplementation.

Relevant infrastructure includes:

- accepted/materialized recording manifests and content validation;
- locked runtime/environment validation;
- content-bound sorter requests and manifests;
- parameterized Kilosort configurations;
- sort identity pinning;
- restartable curation;
- restartable waveform/amplitude/refractory QC;
- standard exports;
- current exclusive full-session correspondence logic.

These pieces should form the spine of the new framework.

### B. Research/evaluation code containing useful functions: extract or call narrowly

Examples include:

- full-session exclusive correspondence;
- common-physical-time amplitude comparison;
- long-strip time/depth diagnostics;
- chance-aware coincidence machinery;
- amplitude truncation/missingness fitting;
- similar-template and CCG audits;
- long-term waveform/PC-feature continuity.

Reuse tested functions where possible. If code is heavily tied to one historical hard-coded experiment, extract only the generic calculation into a small new module rather than making the new framework depend on the whole historical script.

### C. Historical or currently irrelevant branches: leave untouched

Do not integrate these into the first implementation:

- `pipelineold` as a new dependency;
- `pipeline.bakeoff` challenger architectures;
- KS4-seeded static/motion-aware peelers;
- post-sort stitchers/linkers;
- DARTsort;
- KIASORT;
- external DREDGE voltage registration;
- selective motion correction;
- claim-mask patching;
- old injected-motion challenge ladders;
- synthetic motion-residual laboratories;
- old short-window candidate panels;
- historical analyses whose conclusions have been retracted.

They remain useful evidence and regression material, but they are not the current development path.

---

# 2. Preserve repository boundaries

Do not turn experimental development code into production code prematurely.

Use:

```text
pipeline/
```

only for genuinely reusable infrastructure that is already part of the accepted production graph or is clearly generic enough to belong there.

Put the new development ladder and evaluation orchestration under:

```text
testing/
```

with machine-readable experiment specifications under an appropriate `configs/` or `testing/configs/` location.

Outputs should go under:

```text
testing/outputs/<experiment_name>/
```

or another explicitly configured non-production development location.

Never overwrite an accepted recording or accepted production sort.

Never silently reuse an output whose request digest, recording digest, content digest, or sorter configuration differs.

---

# 3. Start by inventorying, not rewriting

Before changing code, identify the exact reusable interfaces.

At minimum inspect:

```text
pipeline/preprocess.py
pipeline/sorting.py
pipeline/downstream.py
pipeline/runtime.py

testing/ladder_sorter.py
testing/luke_full_session_rigid.py
testing/luke_full_session_compare.py
testing/luke_full_strip_diagnostic_audit.py
testing/luke_full_probe_rescue_diagnostics.py
```

Also inspect the current amplitude/truncation implementation and its corrected audit/tests before reusing it.

The first coding deliverable should be a short implementation map documenting:

```text
required capability -> existing implementation -> reuse/adapt/new
```

Do not spend significant time reorganizing code unless duplication actually prevents implementation.

---

# 4. Build one generic experiment specification

Create one machine-readable experiment specification rather than another collection of hard-coded Luke scripts.

It should describe:

## Recording

- accepted recording path;
- expected recording request digest;
- expected content digest where available;
- probe/stream;
- total duration;
- channel geometry identity.

## Development region

Represent the long depth-reduced development region in **physical coordinates**, not merely channel indices.

Specify:

- processing depth range;
- interior scoring depth range;
- halo above;
- halo below;
- minimum edge exclusion for units.

The implementation should derive channel IDs from the actual probe geometry.

Do not assume a fixed number of contacts corresponds to a fixed physical halo.

## Duration

Support:

- full recording;
- contiguous partial duration.

The preferred Luke development configuration should use the full duration unless storage/runtime constraints make that unreasonable.

## Candidate sorter configurations

Reference configurations by a named `SorterConfig` or equivalent content-digested object.

## Evaluation configuration

Include:

- spike correspondence tolerance;
- minimum overlap required to retain a correspondence edge;
- primary-match requirements;
- nominal amplitude-window size;
- time bins used for longitudinal metrics;
- interior unit rule;
- refractory parameters;
- coincidence parameters;
- any metric availability requirements.

Every experiment should write the resolved specification and digest before running expensive work.

---

# 5. Generalize the long-depth development recording

The old full-duration 96-channel strip demonstrated that this is a useful development strategy. Implement it properly now.

## Required behavior

Given an accepted full-probe recording and a physical depth specification:

1. validate the accepted source recording;
2. identify all channels needed for the **processing strip**, including halo;
3. preserve original channel IDs and geometry;
4. produce or expose a recording containing the full requested duration;
5. write a content-bound strip manifest;
6. verify:
   - sample count;
   - sampling frequency;
   - dtype;
   - channel IDs;
   - coordinates;
   - source recording digest;
   - requested physical support;
   - expected bytes if materialized;
   - representative data population across early/middle/late chunks.

Do not rely on file size alone. The repository already encountered a full-sized but partially zero-filled recording.

## Halo/interior contract

The strip must have two regions:

```text
PROCESSING REGION
| halo |----------- interior scoring region -----------| halo |
```

Sort using the entire processing region.

Evaluate the principal unit metrics only for units whose prespecified spatial coordinate lies inside the interior scoring region.

Save metrics for edge/halo units separately as diagnostics.

The comparison must report:

- fraction of spikes near processing boundaries;
- number of units near processing boundaries;
- whether candidate effects are concentrated near those boundaries.

A candidate cannot be promoted because of improvements produced predominantly by artificial strip edges.

---

# 6. Reuse and extend `testing/ladder_sorter.py`

Do not create separate bespoke scripts for every KS4 configuration.

Use the existing `SorterConfig` abstraction and its digest/effective-setting validation.

The repository already distinguishes:

- rescue/no motion;
- rescue/native rigid;
- legacy-style;
- native nonrigid.

Preserve the important test that requested settings are not enough: verify the **effective saved Kilosort settings** after the run.

Add named standard configurations only when required.

---

# 7. Initial standard-pipeline search

The first search should remain deliberately small.

Do not run a factorial search across every setting.

## A. Motion axis

On identical preprocessing and thresholds:

1. native motion off;
2. native KS4 rigid correction;
3. native KS4 nonrigid correction, if the installed KS4 implementation exposes this as a normal supported configuration and it can be run without custom patching.

This is a standard-sorter comparison. No external voltage warp.

## B. Threshold axis

With native motion initially off, run a bounded conventional threshold set.

Suggested starting points:

```text
12/9    current reference
10/9
9/9
9/8
```

An additional 8/8 arm may be included if there is a clear reason, but do not build a dense threshold grid.

The earlier donor-level null does not establish that 12/9 is globally optimal because the evaluation target was different.

## C. Do not cross the axes initially

Do not immediately run:

```text
4 thresholds × 3 motion settings
```

First identify whether either axis contains a meaningful improvement.

Only test one interaction if the first-stage results create a specific reason to do so.

---

# 8. Preprocessing is frozen for the first ladder

The current accepted rescue input is the shared starting point.

Do not restart a broad preprocessing search in this implementation.

In particular, do not add new:

- common-reference algorithms;
- blanking policies;
- filtering schemes;
- bad-channel rules;
- AIND variants;

unless the standard-sorter ladder exposes a specific remaining failure that implicates preprocessing.

Historical AIND/legacy products may be reported as contextual comparators if their identities are trustworthy, but they should not expand the first experimental matrix.

---

# 9. Curation must be identical across candidate arms

Every candidate must pass through the same curation implementation with the same settings.

Use the existing identity-bound downstream stage.

Report both:

```text
pre-curation
post-curation
```

where feasible.

A candidate should not appear better merely because curation removed different amounts of problematic output.

Do not enable the claim mask.

Do not tune curation during the sorter search.

---

# 10. Build a generic long-sort comparator

The recent full-session rigid comparison contains the correct starting logic.

Do not rewrite the exclusive matcher.

Generalize it.

At minimum preserve these properties:

- candidate pair enumeration happens per cluster pair;
- matching is exclusive;
- one event cannot be reused;
- unrelated clusters do not compete for events;
- reciprocal primary correspondence is distinguished from ambiguous split/merge edges;
- all labels can participate, not only `good` clusters;
- ambiguous ties remain ambiguous;
- correspondence is explicitly described as spike-train correspondence, not proof of biological identity.

Create a generic interface such as:

```python
compare_sorts(
    baseline_sort,
    candidate_sort,
    baseline_qc,
    candidate_qc,
    config,
    spatial_region=None,
)
```

Do not bind it to Luke-specific paths.

---

# 11. Required comparison outputs

Every pairwise comparison should produce a stable schema.

At minimum:

```text
summary.json
candidate_manifest.json
correspondence_edges.csv
primary_matches.csv
split_merge_summary.json
amplitude_windows.csv
amplitude_completeness_pairs.csv
unit_metrics_baseline.csv
unit_metrics_candidate.csv
guardrail_summary.csv
coverage_summary.json
decision.json
```

The full edge table is important. Do not save only primary matches; split/merge structure is one of the phenomena being diagnosed.

---

# 12. Primary efficacy metrics

Do not create one composite quality score.

For matched/corresponding units report:

## A. Spike-train preservation

- baseline → candidate retained-event fraction;
- candidate → baseline retained-event fraction;
- Jaccard overlap;
- number of exclusive matched events;
- unmatched events in each direction.

Summarize the distribution across primary matched units.

## B. Amplitude completeness

Use the current corrected amplitude/missingness implementation.

Rules:

- nominal primary window remains 1,000 spikes unless deliberately changed in a separate methodological experiment;
- compare only windows with compatible physical-time support;
- never pool unrelated cluster amplitude distributions;
- retain fit status/failure/censoring information;
- report measurement coverage explicitly;
- do not convert poor coverage into an efficacy conclusion.

For each candidate report:

```text
number of primary matches
number measurable in baseline
number measurable in candidate
number measurable in both on common time
fraction adequately measured
paired completeness difference
```

If only a tiny subset is measurable, the result is:

```text
endpoint infeasible / insufficient coverage
```

not candidate success or failure.

## C. Longitudinal identity/stability

For each unit, compute standardized time-resolved quantities such as:

- presence by time bin;
- firing rate by time bin;
- amplitude trajectory;
- depth trajectory;
- early/middle/late waveform or feature similarity;
- amplitude CV over time;
- apparent active lifetime.

Reuse the useful calculations in the full-strip diagnostic audit, but make them generic.

---

# 13. Hard guardrails

Every candidate must also report:

## Refractory / contamination

- existing refractory metrics;
- sliding-RP contamination if available in the installed analysis stack.

## Duplicate / over-splitting burden

- chance-aware near-coincident cross-unit burden;
- similar-template pair burden;
- CCG-supported duplicate hypotheses;
- split/merge degree in the correspondence graph.

## Waveform/cluster quality

Where supported:

- NN isolation;
- NN miss rate;
- SD ratio;
- noise cutoff;
- amplitude CV range.

These are useful additions, but **do not upgrade or destabilize the locked production environment merely to obtain them**.

Feature-detect availability.

If the pinned SpikeInterface version does not expose a metric safely, either:

1. implement it in a separate analysis environment using saved outputs; or
2. mark it unavailable for this iteration.

Do not change the production lock as a side effect of adding QC.

## Spatial guardrails

For depth-strip runs:

- edge spike fraction;
- edge unit fraction;
- candidate-specific accumulation at boundaries;
- interior-only sensitivity summary.

---

# 14. Decision logic

Do not produce an automatic scalar rank.

Create a Pareto-style comparison table.

For every candidate show:

### Primary efficacy

- paired completeness change;
- spike-train retention change;
- longitudinal stability change.

### Guardrails

- refractory contamination;
- duplicate burden;
- split/merge burden;
- waveform heterogeneity/isolation;
- boundary burden;
- measurement coverage.

### Descriptive only

- KS-good units;
- total units;
- total spikes;
- median rate;
- conventional contamination labels.

A candidate advances only when:

1. it produces a practically meaningful improvement on at least one primary efficacy axis;
2. the improvement is measured on adequate support;
3. it does not create a clear guardrail failure;
4. the result is not primarily an edge/boundary artifact.

Do not select a candidate because it has the largest unit count.

---

# 15. Successive-halving execution strategy

The intended workflow is:

## Level 0 — implementation smoke

Use a short recording only to establish:

- code runs;
- settings are actually applied;
- outputs are valid;
- geometry/time/dtype are correct.

No scientific selection at this level.

## Level 1 — long depth strip

Run every initial standard candidate on the same long/full-duration processing strip.

This is the **main development screen**.

Discard clearly dominated candidates.

## Level 2 — full probe, full session

Advance at most **two** candidate configurations beyond the reference.

Prefer one if there is a clear winner.

Use the existing accepted full 384-channel recording directly where possible rather than copying another huge binary.

## Level 3 — second session

If a candidate beats the reference on Luke0804 full-session/full-probe data, freeze it.

Replicate the frozen comparison on one independent session before adopting it generally.

No further parameter tuning on the replication session.

---

# 16. Short snippets are now diagnostic only

After the long comparison reveals a specific discrepancy, short windows may be used to localize it.

Examples:

- candidate loses events specifically during high motion;
- candidate develops increased duplicate burden;
- amplitude completeness worsens in one epoch;
- units split after a particular transition.

Then use a short interval to trace:

```text
raw / accepted voltage
→ Kilosort input
→ detection
→ template assignment
→ final cluster
→ curation
```

This is fault isolation.

Do not use the short interval to declare the candidate globally better.

---

# 17. Do not implement bespoke correction yet

This task explicitly excludes:

- stitchers;
- cluster-family linkers;
- motion-aware peelers;
- selective motion correction;
- external nonrigid voltage correction;
- custom template tracking.

Those become authorized only if the standard-pipeline ladder demonstrates the corresponding residual failure.

Examples:

### Authorize a stitcher only if

multiple units show:

- compatible waveform families;
- complementary temporal support;
- strong correspondence evidence;
- a merge restores continuity;
- the merge does not create refractory/heterogeneity failure.

### Authorize a motion-aware matcher only if

the evidence shows:

- spikes remain present in the voltage;
- failures increase with motion;
- static template matching is the failure stage;
- ordinary native Kilosort correction does not solve it safely.

Do not build solutions before the failure mode is demonstrated.

---

# 18. Sparse injected truth is a later validation layer

Do not restart the large synthetic program now.

Preserve the existing injection infrastructure.

After the standard pipeline search has identified one or two credible candidates, build a small Allen-style hybrid truth test:

- small number of independently qualified donor units;
- real Luke background;
- sparse injection so background statistics remain realistic;
- randomized repeats;
- true precision, recall and accuracy;
- stratification by SNR/amplitude and motion state.

The historical donor qualification failures mean current Kilosort cluster membership cannot simply be treated as donor identity.

This benchmark should validate finalists, not choose among dozens of configurations.

---

# 19. Historical evidence is regression evidence, not a work queue

The repository contains many retracted or superseded analyses.

Do not attempt to “complete” every branch.

Use historical pathologies as regression tests where valuable.

For example, the evaluator should be able to demonstrate that:

- event reuse cannot inflate correspondence;
- a tied split is not declared a primary identity match;
- amplitude populations are compared on common physical time;
- missing coverage returns an explicit infeasible state;
- a fake improvement caused by dropping difficult spikes is not rewarded;
- changed sorter settings invalidate cache reuse;
- wrong recording clocks are rejected.

The existing tests around the full-session matcher should remain and be extended rather than replaced.

---

# 20. Testing requirements

Every new generic calculation needs small synthetic/unit tests.

At minimum add tests for:

### Recording/strip contract

- physical-depth channel selection;
- halo selection;
- interior selection;
- geometry preservation;
- invalid/out-of-range strip requests;
- manifest/digest sensitivity;
- detection of changed source identity.

### Sorter configuration

- threshold changes alter digest;
- motion changes alter digest;
- effective settings match requested scientific factor;
- one-factor candidates do not accidentally alter unrelated settings.

### Correspondence

Retain existing tests for:

- exclusive event matching;
- duplicate event non-reuse;
- reciprocal symmetry;
- ambiguous split handling;
- common-time interval handling.

Add:

- clear one-to-many split;
- clear many-to-one merge;
- good→MUA correspondence preserved;
- unmatched unit;
- candidate with much higher background spike density.

### Amplitude/common-time evaluation

- incompatible time support rejected;
- overlapping windows rejected;
- insufficient nominal spike count flagged;
- fit failure propagated;
- coverage denominator correct;
- swapping baseline/candidate reverses signed difference appropriately.

### Interior scoring

- edge units excluded from primary summary;
- interior units retained;
- edge diagnostics still saved;
- same physical boundary rule applied across candidates.

---

# 21. Reproducibility requirements

Every expensive run must save:

- git commit;
- dirty-tree status or diff hash if practical;
- environment receipt;
- recording identity/digest;
- strip specification;
- sorter config and digest;
- effective saved Kilosort settings;
- curation request digest;
- QC request digest;
- evaluation config and digest;
- completion status.

A run should fail closed rather than silently reuse incompatible output.

---

# 22. First concrete implementation milestone

Do not try to implement the entire future roadmap before producing data.

The first milestone is complete when the repository can do this:

```text
accepted Luke0804 imec0 recording
        |
        +--> full-duration halo-supported depth strip
                |
                +--> rescue 12/9, motion off
                |
                +--> rescue 12/9, native rigid
                |
                +--> bounded no-motion threshold variants
                        |
                        v
              identical curation + QC
                        |
                        v
            generic exclusive comparator
                        |
                        v
              compact Pareto report
```

The baseline should be reusable rather than rerun if its accepted identity and configuration match.

The already running/completed full-session rigid experiment should also be consumable by the same comparator; do not create a parallel evaluation implementation for it.

---

# 23. Recommended file-level deliverables

Names may change if equivalent generic modules already exist, but a reasonable layout is:

```text
testing/
    standard_pipeline_ladder.py
    long_depth_strip.py
    sort_comparison.py
    longitudinal_metrics.py
    standard_quality_metrics.py

testing/configs/
    luke0804_standard_pipeline_ladder.json

testing/test_standard_pipeline_ladder.py
testing/test_long_depth_strip.py
testing/test_sort_comparison.py
testing/test_longitudinal_metrics.py
```

However, if `testing/luke_full_session_compare.py` can be made generic cleanly, prefer renaming/generalizing it over creating a second matcher.

Likewise, reuse `testing/ladder_sorter.py` rather than introducing another sorter-configuration abstraction.

Avoid unnecessary module proliferation.

---

# 24. Explicit non-goals

This implementation is **not** complete when:

- all historical pipeline experiments have been ported;
- every sorter is supported;
- all motion algorithms are wrapped;
- all SpikeInterface quality metrics exist;
- every old document has been reconciled;
- a sophisticated dashboard exists.

It is complete when a scientist can specify a handful of standard pipeline configurations, run them reproducibly on a long recording, and receive a trustworthy comparison of spike completeness, correspondence, longitudinal stability, and guardrails.

---

# 25. Decision discipline for the agent

When you encounter a possible extension, ask:

> Does this change the answer to the current standard-pipeline comparison?

If no, defer it.

If an existing implementation is ugly but correct and tested, reuse it.

If a metric is unavailable but nonessential, report it unavailable.

If an experiment has inadequate support, report infeasibility rather than inventing another estimator.

If a candidate loses, do not “fix” it unless the result identifies one specific implementation defect that invalidated the comparison.

If all standard candidates fail, stop and summarize the residual failure mode before proposing custom algorithms.

The goal is to make the next scientific decision quickly and correctly, not to make the repository maximally elegant.

---

# Definition of done

The task is done when:

1. A long, halo-supported development strip can be generated reproducibly from an accepted recording.
2. A small named set of standard KS4 configurations can be run through the existing sorter infrastructure.
3. Exact effective settings are verified for every arm.
4. Curation and QC are identical and identity-bound.
5. The corrected exclusive correspondence machinery is generic and reused.
6. Amplitude completeness is compared only on common physical time with explicit coverage.
7. Longitudinal unit stability and split/merge structure are reported.
8. Core refractory, duplicate, waveform and boundary guardrails are reported.
9. Outputs use a stable machine-readable schema.
10. The system produces a compact Pareto-style comparison rather than a single quality score.
11. Clearly dominated candidates can be stopped after the long-strip stage.
12. At most two finalists can be promoted to full-probe/full-session comparison.
13. No peeler, stitcher, custom voltage warp, claim-mask modification, or challenger sorter is added as part of this task.
14. The portable unit/contract test suite passes.
15. The resulting framework can be pointed at a second accepted recording without rewriting its core evaluation logic.