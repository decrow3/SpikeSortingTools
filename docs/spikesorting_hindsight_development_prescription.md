# A hindsight-first prescription for developing the Luke spike-sorting pipeline

## Purpose

This document describes the development strategy that would have been preferable from the beginning, given what the Luke investigation has since taught us. It is also the recommended strategy for the work from this point forward.

The central correction is strategic rather than algorithmic:

> **Build a trustworthy diagnostic battery early, use it first to compare mature and mostly standard sorting pipelines on long recordings, and introduce custom algorithms only after a residual failure mode has been demonstrated clearly enough that the custom method is targeted to it.**

The project should optimize for **decision value per unit of human effort**, not for the smallest possible compute job or the most complete mechanistic explanation.

The operational goal is not to build the most sophisticated spike-sorting system. It is to obtain spike trains that are sufficiently complete, stable and correctly partitioned over long recordings to support the lab's scientific analyses.

---

## 1. Define the actual problem before optimizing the pipeline

The primary failure motivating the Luke work is longitudinal: neurons that appear healthy for part of a long recording may lose spikes, change apparent amplitude/depth, fragment into different clusters, or disappear from the sort. Therefore the relevant experimental unit is fundamentally **a neuron followed over a substantial fraction of the recording**, not a 60–120 s snippet.

The primary development question should have been:

> **Which practical sorting pipeline most faithfully preserves detectable neurons and their spikes through physical time, without creating unacceptable contamination, duplication, splitting or merging?**

This immediately implies several consequences.

1. **A full session, or at least a long continuous fraction of one, is the natural efficacy dataset.** Short snippets are diagnostic tools.
2. **Unit count is not the primary endpoint.** More clusters can mean better recovery, over-splitting, artifact seeding, or duplicate peeling.
3. **Conventional quality metrics are necessary guardrails but are not sufficient.** Low contamination and clean refractory periods can coexist with missing spikes or fragmented identities.
4. **The principal comparisons should be between plausible complete pipelines**, not between increasingly elaborate local interventions whose effect on the whole recording is unknown.

The first milestone should therefore have been something concrete such as:

> Identify a standard or minimally modified pipeline that improves longitudinal spike completeness/identity on a long Luke development dataset, while preserving refractory, duplicate and waveform-quality guardrails; confirm the result on the full probe and then on a second session.

---

## 2. Establish one safe reference graph, but do not mistake it for a winner

A reproducible reference pipeline is necessary so every experiment is anchored to identical voltage, geometry, time coordinates, software versions and downstream evaluation. The rescue work ultimately achieved this and that achievement should be retained.

The reference graph should be intentionally boring:

- verified acquisition stream and channel map;
- Neuropixels phase correction;
- conservative saturation handling that is known not to catastrophically destroy reviewed events;
- explicit handling of known bad channels;
- no experimental claim mask;
- no external voltage motion warp unless already validated;
- one clearly defined filtering/reference/whitening path;
- content-bound caches and immutable manifests;
- identical curation and QC for all compared arms.

Critically, calling this the **reference** must mean only that it is the stable comparator. It must not imply that it is biologically superior or that higher yield establishes better recovery.

The Luke rescue results demonstrated exactly why this distinction matters: favorable KS-good yield, contamination and refractory metrics did not prove improved spike completeness, and several early cross-sort completeness conclusions had to be withdrawn when matching/evaluation defects were found.

---

## 3. Validate the evaluator before doing a broad pipeline search

The most durable product of the investigation is the diagnostic framework. That should have been built and stress-tested early, because a parameter search is only as good as the score used to rank candidates.

### 3.1 The evaluator must answer two different questions

**Question A: Is this sort better?**

This requires end-to-end, neuron-centered outcomes.

**Question B: If it is worse, where did the failure enter?**

This requires stage-local diagnostics.

These should not be collapsed into one scalar score.

### 3.2 Primary efficacy endpoints

For long development comparisons, the primary endpoints should be:

1. **Time-resolved amplitude completeness / missingness** on units with adequate measurement support, always evaluated over common physical time.
2. **Exclusive spike-train correspondence between pipelines**, including the fraction of corresponding events and the amount of one-to-many / many-to-one structure.
3. **Longitudinal identity continuity**, including whether waveform/amplitude/depth trajectories remain consistent through early, middle and late recording periods and through motion epochs.
4. **Injected-ground-truth recall/precision/accuracy** when a suitably qualified sparse injection benchmark is available.

No endpoint should be interpreted when its measurement coverage is inadequate. A comparison in which completeness is measurable for 2 of 53 eligible units is a feasibility failure, not an efficacy result.

### 3.3 Hard quality guardrails

Every candidate should also report:

- refractory/ISI contamination, preferably including a sliding-RP estimate;
- chance-aware near-coincident cross-unit burden;
- similar-template/CCG duplicate structure;
- split/merge burden in the cross-sort correspondence graph;
- presence and firing stability over time;
- waveform stability and spatial footprint;
- edge/boundary burden for reduced-depth experiments;
- artifact proximity when saturation/ringing is relevant.

### 3.4 Standardized unit metrics worth adding

The following SpikeInterface-style metrics would add useful orthogonal information to the existing battery:

- **NN miss rate** and **NN isolation**: feature-space evidence for missed/competing populations and poor cluster separation;
- **SD ratio**: useful as a waveform-heterogeneity/overmerge guardrail, especially when an apparently clean merge combines different epochs;
- **noise cutoff**: complementary evidence for low-amplitude truncation without depending entirely on a Gaussian amplitude model;
- **amplitude CV range**: a standardized time-resolved amplitude-instability measure;
- **sliding refractory-period contamination**: a more robust scalar refractory estimate;
- optionally exact-sample synchrony and spatial template metrics as cheap artifact alarms.

These are supporting metrics, not a new optimization target.

### 3.5 Evaluator failure modes that must remain explicitly prohibited

The investigation identified several ways a reasonable-looking evaluator can produce false conclusions. The new framework should make these impossible by construction:

- do not compare unmatched populations and call the difference completeness;
- do not allow one event to be reused by multiple matches;
- do not let unrelated clusters compete freely for injected spikes;
- use exclusive correspondence where identity is being claimed;
- estimate chance coincidence/coverage rather than treating temporal proximity as identity;
- evaluate amplitude distributions over the same physical time support;
- do not mix incompatible amplitude populations and interpret the resulting fit as dropout;
- content-bind caches to the recording and configuration;
- verify acquisition versus selected-recording clocks;
- preserve true probe geometry in synthetic motion/injection experiments;
- do not treat Kilosort labels, reviewed-event labels, or same-cluster membership as biological ground truth unless independently qualified.

Before using the evaluator to choose a pipeline, test it against **known bad manipulations**. It should flag the pathological claim mask and harmful external nonrigid warp as worse, and it should reject a fake completeness improvement produced by dropping the difficult part of a train. If it cannot do that, the metric is not ready to rank real pipelines.

---

## 4. Use long depth-reduced recordings as the development workhorse

The biggest data-design correction is to reduce **depth before time** when the scientific failure is longitudinal.

A 60–120 s full-probe snippet preserves spatial context but destroys the main phenomenon we need to measure. A 1.5–3 h depth strip preserves the time axis and therefore supports:

- amplitude distributions with enough spikes to estimate missingness;
- early/middle/late waveform comparison;
- motion-conditioned analyses;
- natural quiet and high-motion intervals;
- population-level fragmentation/splitting statistics;
- complementary temporal support between candidate clusters;
- unit lifetimes and presence;
- repeated opportunities for the same neuron to encounter different recording conditions.

The completed full-duration 96-channel Luke strip later demonstrated the value of this approach and was cheap enough that the earlier emphasis on very short snippets was not justified by compute alone.

### 4.1 Do not use a naked narrow strip

Depth reduction changes the sorting problem, so it needs a spatial-support contract.

Recommended design:

- choose an **interior evaluation band** large enough to contain a useful population of units;
- load/process a **wider halo** above and below that band so filtering, referencing, whitening and any motion operator have real neighboring voltage;
- choose the halo in physical micrometers based on the largest relevant waveform/reference/interpolation support, not an arbitrary channel count;
- sort the wider strip if required by the algorithm, but score only units safely inside the prespecified interior region;
- explicitly measure boundary accumulation and reject conditions that create edge units.

For Luke, something on the order of an 80–120 channel processing strip with a smaller prespecified interior scoring region is preferable to an isolated 60-channel strip with unsupported edges. The exact channel count should follow physical support and GPU/memory constraints rather than become a biological parameter.

### 4.2 Duration

For development, prefer:

- the **full recording duration** when a depth strip is cheap enough; otherwise
- at least **~1–1.5 h continuous duration** spanning substantial motion and quiet periods.

Short 60–240 s windows remain useful for implementation checks and causal ablations after a full/long comparison reveals a specific difference.

---

## 5. Search the mature standard-pipeline space before inventing new algorithms

This should have been the main development phase.

The mistake was not parameter search itself. The mistake was searching with incomplete scores and impoverished data, while simultaneously escalating to custom architectures before ordinary pipelines had been given a fair test.

### 5.1 Candidate families

Start with a compact set of mature, practical pipelines. For example:

1. **Stable reference / rescue preprocessing + KS4, motion off.**
2. **Same input + native KS4 rigid motion correction.**
3. **Minimal/single-pass preprocessing + KS4, motion off**, allowing KS4 to perform its normal high-pass/CAR/whitening once.
4. **Single-pass preprocessing + native KS4 motion**, if the motion-off version is viable.
5. **One well-motivated alternative common-reference preprocessing** (for example the already explored AIND-style CMR), initially with motion off.
6. **A mature independent sorter baseline** such as KS2.5 if practical and useful as an architecture control.

Do not start with DARTsort, KIASORT, a custom peeler, or a stitcher unless the standard space fails and the failure mode points specifically toward one of those approaches.

### 5.2 Threshold search

Detection thresholds deserve a real but bounded search. The fact that 8/8 and 9/9 failed to separate from 12/9 under one frozen analysis does **not** prove that search is intrinsically misguided or that 12/9 is optimal.

A better approach is:

- choose a modest grid around the standard/default and current lab settings;
- evaluate it on the long depth strip with the validated longitudinal metrics;
- use successive halving: discard clearly dominated settings early and spend the full-duration/full-probe budget only on the survivors;
- do not tune thresholds simultaneously with preprocessing, motion and curation unless an initial one-factor screen shows a strong interaction worth testing.

The goal is not to exhaust every Kilosort parameter. It is to establish whether the required improvement already exists within a conventional operating regime.

### 5.3 Freeze curation during the search

Run the same conservative curation strategy for every condition. Report both pre-curation and post-curation results so a curation step cannot hide a sorter failure or create a false recovery advantage.

Custom claim suppression stays off unless duplicate burden remains a demonstrated limiting failure after the upstream pipeline is selected.

---

## 6. Rank candidates by a Pareto-style decision, not one composite score

A single quality score would conceal the tradeoffs we repeatedly encountered.

For each candidate, construct a compact decision table with:

### Primary benefit

- amplitude completeness/missingness change on adequately supported matched units;
- longitudinal correspondence/identity preservation;
- injected-truth recall if available.

### Guardrails

- refractory/sliding-RP contamination;
- chance-aware near-coincident duplicate burden;
- NN isolation/miss;
- SD ratio / waveform heterogeneity;
- amplitude CV range;
- noise cutoff;
- split/merge structure;
- boundary and artifact burden.

### Secondary descriptive outputs

- KS-good units;
- total units;
- assigned spikes;
- mean/median rate;
- conventional contamination labels.

A candidate advances when it provides a **practically meaningful primary improvement without paying for it through a clear guardrail failure**. The minimum meaningful effect should be fixed before the comparison based on measurement repeatability, not chosen after seeing the result.

A small increase in good-unit count or a 1–2% improvement in a proxy metric is not enough to justify a new production pipeline if completeness and identity are unchanged.

---

## 7. Use short snippets to explain an observed effect, not to establish global efficacy

Once a long-strip comparison shows that candidate B is better or worse than A, short windows become extremely useful.

Choose windows that maximize the contrast and ask causal questions such as:

- Did the raw event survive preprocessing?
- Was it present at the sorter input?
- Was it detected but assigned to another cluster?
- Did a motion operator attenuate amplitude or alter the spatial footprint?
- Did nonrigid gradients create duplicate peeling?
- Did curation remove a genuine event?

The diagnostic chain should be:

**raw voltage → accepted/preprocessed voltage → sorter detection → cluster assignment → curation → final unit train**.

This is where 30–240 s cases excel: debugging, mechanism, visualization and regression tests.

They should not be asked to answer whether a pipeline preserves a neuron for 2.9 h.

---

## 8. Fault-isolation framework: use the metrics to implicate the failing stage

The diagnostic battery should support a simple decision tree.

### A. Raw voltage already lacks a plausible compact event

Likely domain: acquisition/biology/reference comparison rather than the sorter.

Check:

- raw amplitude/SNR;
- spatial compactness;
- polarity;
- common-mode structure;
- acquisition continuity and channel metadata.

### B. Raw event is healthy but degraded before sorting

Likely domain: preprocessing.

Check stagewise:

- amplitude retention;
- waveform cosine;
- peak-depth/channel displacement;
- ringing near saturation handling;
- common-mode/reference effect;
- bad-channel interpolation effect.

### C. Sorter input is healthy but event is absent from detections

Likely domain: detection threshold/template detection.

This justifies a threshold/detection search.

### D. Event is detected but assigned inconsistently or disappears from the final unit

Likely domain: template matching, clustering, duplicate control or curation.

Use:

- exclusive event correspondence;
- feature-space NN metrics;
- local competing clusters;
- CCG/template similarity;
- pre- versus post-curation tracing.

### E. The waveform family persists in voltage but moves between complementary clusters over time

Likely domain: identity fragmentation.

Only here does a stitcher become a well-motivated intervention.

### F. The static template matcher itself fails while a stable waveform family moves through depth

Only here does a motion-aware peeler/template-tracking architecture become a well-motivated intervention.

### G. Multiple units repeatedly claim the same physical events

Likely domain: over-peeling/artifact seeding/duplicate handling.

Only after the upstream signal and motion pipeline are fixed should a claim-mask-like intervention be revisited.

---

## 9. Escalation ladder for custom methods

Bespoke methods should require evidence, not merely plausibility.

### Level 0 — standard pipeline variation

Preprocessing, native motion handling, thresholds, mature sorter versions.

### Level 1 — targeted parameterization

A small search around the best standard pipeline because a specific standard component is clearly limiting.

### Level 2 — targeted custom intervention

Allowed only if the residual failure is demonstrated repeatedly.

Examples:

- **stitcher** only after showing continuous raw waveform families divided into complementary cluster epochs;
- **motion-aware peeler** only after showing that spikes remain present but static matching fails systematically with motion;
- **custom voltage resampling** only after native/unwarped approaches fail and the resampling operator itself passes waveform integrity tests;
- **claim mask** only after true duplicate over-peeling remains a major problem in the otherwise accepted pipeline.

### Level 3 — new sorter architecture

DARTsort/KIASORT/other challengers become worthwhile only if the mature pipeline space leaves a reproducible, scientifically important deficit that their architecture specifically addresses.

A new architecture is not the next experiment merely because it is interesting.

---

## 10. Sparse injected truth should be a benchmark, not a parallel research program

The injection idea remains valuable, but it should be simpler and more conservative than our first attempts.

The goal is not to synthesize the entire Luke problem. It is to embed a small amount of trustworthy ground truth in real Luke background.

Recommended design:

1. Independently qualify a small set of waveform donors; do not use cluster labels alone as identity truth.
2. Include polarity/morphology/SNR diversity and relevant depth support.
3. Inject only a small number of units into each real background so the background statistics remain realistic.
4. Use paired injected/uninjected controls.
5. Initially inject static units; add controlled drift only after the static benchmark is valid.
6. Run the same standard pipeline candidates used in the real-data comparison.
7. Report per-injected-unit precision, recall, accuracy, localization and duplication, stratified by amplitude/SNR and motion condition.
8. Repeat across several randomized injection instances rather than building one enormous synthetic benchmark.

Injection is primarily a **calibration and falsification tool** for the real-data metrics. A pipeline should not be selected solely because it wins on synthetic data.

---

## 11. Validation sequence

Once a standard candidate clearly wins the development strip, stop tuning it.

### Stage 1 — long depth-strip development

Use the same continuous long interval for all candidates. Rank and eliminate.

### Stage 2 — full-duration depth strip

If Stage 1 did not already use the full session, run the survivor(s) over the full duration with the same interior/halo contract.

### Stage 3 — full-probe, full-session comparison

Run only the reference and at most one or two survivors over all channels. No retuning after seeing this result.

Evaluate:

- common-time amplitude completeness;
- exclusive train correspondence;
- split/merge populations;
- longitudinal waveform identity;
- refractory/duplicate/NN/SD/noise-cutoff guardrails;
- unmatched populations as descriptive outputs rather than evidence of completeness.

### Stage 4 — second-session replication

Run the frozen candidate on a different session selected without reference to whether it is expected to win.

The second session should ideally contain a different motion regime so the result establishes a useful domain rather than merely repeating Luke0804.

### Stage 5 — scientific consequence

Only after the sorting improvement is independently supported should the lab ask whether it changes scientific endpoints. Freeze the downstream analysis and change only the sorting input.

A change in tuning/decoding/effect size does not itself validate the sort; it measures the consequence of a separately validated pipeline change.

---

## 12. Stopping rules to prevent another time sink

Every experiment must begin with a written sentence of the form:

> **If result X occurs, we will make decision Y.**

If no plausible outcome can change a near-term pipeline decision, do not run the experiment yet.

Additional rules:

1. **One mechanism-directed revision after a failed candidate**, unless the run was technically invalid.
2. **A feasibility failure closes the endpoint**, rather than triggering repeated attempts to salvage it by progressively weakening measurement requirements.
3. **No custom architecture without a demonstrated target failure mode.**
4. **No mechanism explanation is required to reject a harmful pipeline branch.** If an external warp is clearly worse, stop using it; explaining whether the dominant cause is estimator scale, kernel width or field regularization can be deferred.
5. **Do not optimize engineering elegance before a candidate shows scientific value.** Tests, provenance and cache correctness are mandatory; generalized frameworks, dashboards and reusable abstractions are optional until repeated use justifies them.
6. **Compute is cheap relative to researcher time.** Prefer a 20–60 minute sort that answers the real longitudinal question over days spent constructing a clever 120 s proxy.
7. **Null is a valid stopping result.** Do not convert every null into a sweep.
8. **Keep a deferred-question list with return triggers.** Interesting unresolved mechanisms do not automatically become blockers.

---

## 13. Recommended concrete search now

The current full-session native-rigid KS4 comparison is already aligned with this prescription and should complete before adding new custom work.

After that result, the next phase should be deliberately conventional.

### If native rigid clearly improves longitudinal completeness/identity

1. Verify that refractory, duplicate, NN isolation/miss, SD ratio and amplitude-stability guardrails do not worsen materially.
2. Freeze the settings.
3. Replicate on another Luke session.
4. If replicated, use the pipeline; explain individual rescued cases afterward rather than before adoption.

### If native rigid is null or worse

Do not immediately return to external warps, peelers or stitching.

Run a bounded long-depth standard matrix centered on:

- rescue/reference preprocessing, motion off;
- single-pass/minimal preprocessing, motion off;
- the same promising preprocessing with native rigid if justified;
- one alternative CMR preprocessing arm;
- a modest KS4 threshold grid on the best preprocessing/motion configuration;
- optionally a mature independent Kilosort generation as an architecture control.

Use successive halving on the long strip, then take at most two configurations to the full probe.

### Only if that matrix fails

Use the diagnostic tree to identify the dominant residual failure. Then select **one** custom intervention specifically matched to it.

---

## 14. What should remain from the work already done

Although the development path was inefficient, much of the resulting infrastructure is valuable and should be retained:

- the provenance-safe rescue/accepted-recording contract;
- cache and sort identity receipts;
- stagewise voltage integrity measurements;
- artifact sidecar concept;
- longitudinal amplitude/presence/waveform diagnostics;
- corrected exclusive matching/correspondence machinery;
- split/merge population analysis;
- chance-aware coincidence controls;
- synthetic/injected-truth scaffolding, once donors are independently qualified;
- motion-field support and coordinate auditing;
- evidence that boundary/clock/geometry issues can invalidate apparently successful corrections;
- the discipline that lower estimated missingness is not improvement if it is achieved by dropping the difficult spikes.

These tools should now serve a much simpler experimental strategy rather than generating more branches of investigation.

---

## 15. The intended end state

The project does not need to prove that one sorter is universally optimal or explain every anomalous Luke cluster.

A successful endpoint is:

1. a simple, reproducible pipeline built predominantly from mature components;
2. demonstrably better longitudinal spike recovery/identity than the current reference on a long development recording;
3. no material degradation in refractory, duplicate, waveform/feature-space isolation or artifact guardrails;
4. full-probe confirmation;
5. replication on another session;
6. clear documentation of the domain in which the pipeline has been validated;
7. a diagnostic suite capable of identifying when a future session falls outside that domain and which stage is most likely responsible.

That last point may ultimately be the most important product of the Luke investigation. The lab should not need one perfect universal pipeline. It needs a **good standard pipeline plus enough instrumentation to recognize when and why that pipeline is failing**.

---

## Condensed workflow

**1. Verify raw/provenance → 2. Freeze simple reference → 3. Validate evaluator on known failures → 4. Build long depth-strip + halo → 5. Compare standard preprocessing/motion/threshold variants → 6. Rank by completeness + identity with contamination/duplicate guardrails → 7. Full-probe confirmation → 8. Second-session replication → 9. Only then diagnose residual failures and consider a stitcher, peeler, custom motion correction or new sorter.**

The governing principle is simple:

> **Use sophisticated metrics to evaluate simple pipelines before using sophisticated algorithms to solve problems that may already have simple solutions.**

