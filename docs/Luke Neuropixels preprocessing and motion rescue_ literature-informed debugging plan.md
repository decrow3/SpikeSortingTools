# Luke Neuropixels preprocessing and motion rescue: literature-informed debugging plan

## Executive working model

The Luke recordings contain substantial recoverable neural voltage and genuine electrode–tissue motion. The central unresolved problem is increasingly upstream of spike sorting:

> **Large spatially shared disturbances appear to survive or interact poorly with the current conditioning chain, while some conditioning operations also measurably alter compact spike waveforms. The resulting event population is therefore a poor input to motion estimation and, subsequently, spike sorting.**

Luke shows a recurring shared-conditioning problem on both probes, with quantitatively different manifestations between imec0 and imec1, most notably the imec1 polarity imbalance. This does not yet establish a qualitatively separate imec1 pathology. All benchmarks must therefore remain probe-stratified, and pooled Luke results cannot determine advancement.

The objective of the next pilot is therefore **not to maximize Kilosort unit yield**. It is to construct a preprocessing chain for which we can demonstrate, stage by stage, that:

1. known compact neural-like signals are preserved;
2. large shared/diffuse nuisance components are reduced;
3. preprocessing does not manufacture spike-like residuals;
4. the peak population supplied to motion estimation becomes more stable and physically interpretable; and
5. destructive operations are used only where the data show that they are needed.

Final sorting is a downstream validation step.

---

# 1. What the Luke data currently establish

Luke does not have a simple shortage of extracellular voltage. Before referencing, Luke contains substantially more large AP-band extrema than the known-good Yates recording. A large fraction of Luke's AP-band voltage is spatially shared: with the current 100-µm-scale local median reference, median 300–6000 Hz channel noise falls from approximately 38 to 7.2 µV on imec0 and 31 to 6.3 µV on imec1, versus approximately 17.4 to 5.0 µV in Yates.

That is compatible with substantial common-mode or spatially smooth contamination, but it does **not** by itself distinguish electrical artifact, mechanical artifact, genuine spatially extended biological voltage, or processing-generated structure.

The existing conditioning chain also measurably alters injected signals. In the first discovery-only injection experiment, the CAR/high-pass/materialized stage retained only about 0.699 median peak amplitude and 0.674 median multichannel waveform cosine relative to the phase-corrected injected delta. Those values are operator-distortion measurements rather than neural-recovery measurements, because the original donor identities were not independently qualified.

Motion is independently supported. Multiple motion estimators indicate more movement in Luke than in Yates, simultaneous Luke probes share a substantial rigid component, and direct amplitude-by-depth fingerprint matching agrees with DREDGE on displacement direction in nearly all qualified large-shift comparisons. The direct tracker generally estimates smaller displacement, with useful observed/DREDGE scale estimates centered roughly around 0.5–0.8 rather than 0.25.

The failure of motion correction is therefore not evidence that motion is absent.

Instead, the existing experiments establish that **voltage resampling is currently unsafe**. Full nonrigid fields from both DREDGE and an independent estimator produce similar overdecomposition and event-loss phenotypes. Even much gentler rigid correction has not reliably improved the full-duration KS4 result. 
This is consistent with the broader motion-correction literature. Modular simulations with known ground-truth drift show that motion-estimation error and interpolation error are separable, and that even interpolation using ground-truth motion can degrade spike-sorting performance relative to truly static recordings. The authors explicitly question whether resampling the raw traces is the best long-term solution to drift.

Therefore:

> **Motion estimation and voltage motion correction should remain separate experimental questions.**

---

# 2. Likely nature of the shared disturbance

The phrase "shared high-frequency noise" should be used cautiously.

A large slow or abrupt disturbance is broadband. High-pass or band-pass filtering can transform it into sharp AP-band extrema and ringing. Possible contributors include mechanical electrode/tissue movement, movement of cables or connectors, common reference/ground disturbances, brief electrical insults, and genuine spatially broad extracellular activity.

SpikeGLX explicitly describes large multichannel artifacts produced by events such as licking, chewing, head movements, or stimulation. Ordinary CAR works well when the disturbance is genuinely similar across channels and unsaturated, but can fail for large nonuniform events.

Thus the most useful current hypothesis is:

**large shared disturbance**
→ **temporal/spatial preprocessing**
→ **residual or generated AP-band structure**
→ **contaminated peak/localization population**
→ **unstable or biased motion estimate**
→ **poor sorting input**

This sequence is a hypothesis to test, not an established causal chain.

A high-priority diagnostic is therefore to inspect the common component at every preprocessing boundary, beginning as early in the acquisition representation as practical.

---

# 3. The benchmarking philosophy

The denoising problem has two objectives:

\[
\text{remove nuisance} \qquad\text{and}\qquad \text{preserve neural signal}.
\]

Neither is meaningful alone.

Reducing RMS noise, first-PC variance, or threshold-crossing count cannot establish improvement because genuine extracellular signals can also be spatially correlated.

Likewise, perfect preservation of injected spikes is insufficient if a method leaves the large shared artifact population untouched.

Every candidate preprocessing chain should therefore be evaluated on a **Pareto frontier of nuisance suppression versus signal distortion**. Signal preservation and nuisance behavior should remain separate scorecards; they should not be collapsed into a weighted quality score.

The initial benchmark should stop before spike sorting.

## Identity calibration before method comparison

The first run should measure the numerical floor and ceiling of the passport itself before any candidate method is judged. Repeated extraction of identical events, explicit no-op/identity transformations, and equivalent materialization paths should establish the repeatability of amplitude, cosine, localization and artifact-creation metrics.

Engineering tolerances should be frozen only after this calibration and before the four-way method comparison is opened. This avoids both demanding precision that the measurement system cannot deliver and accepting distortion far above its numerical floor.

Preservation denominators must also be explicit. When comparing spatial-reference branches, the primary comparison should be against the paired output of the shared minimal temporal-preprocessing branch. Raw or phase-corrected injected waveforms can be retained as secondary absolute references, but unavoidable transformations shared by every branch should not be attributed to the spatial reference.

---

# 4. A stage-level "spike passport"

Each selected or injected event should retain the same identity through the processing graph:

**raw**
→ **temporal alignment / phase correction**
→ **temporal filtering**
→ **bad-channel handling**
→ **continuous common-mode removal**
→ **artifact handling**
→ **motion-event detection/localization**
→ **motion estimation**
→ optional **motion interpolation**
→ **sorter preprocessing**

The exact order must be recorded from the implementation rather than assumed from documentation. This is particularly important when reproducing an external pipeline: the current AIND paper and parameter documentation describe the same operations but are not completely consistent in their prose ordering, so the instantiated SpikeInterface graph, package versions, parameters and hashes should be treated as authoritative.

For every `event × stage`, retain a small multichannel waveform patch and calculate:

- local peak-amplitude retention;
- matched-filter amplitude;
- full multichannel waveform cosine;
- peak timing;
- peak-channel and physical-depth displacement;
- spatial centroid;
- local energy within fixed physical radii;
- spatial footprint width;
- polarity;
- temporal extrema count;
- spatial peak count;
- energy appearing outside the original footprint;
- opposite-polarity "ghost" energy; and
- fixed-threshold detection margin.

Median performance alone is insufficient. Lower-tail behavior must also be reported because a method that preserves most spikes perfectly but destroys 5–10% of waveform classes can still be unacceptable.

---

# 5. Paired injection as an operator assay

For real Luke background \(X\), a known injected waveform \(W\), and processing chain \(F\), compute

\[
\Delta_F(W;X)=F(X+W)-F(X).
\]

This measures the incremental transformation of the known injected component in its actual background.

It naturally captures interactions with nonlinear operations such as median referencing and artifact detection.

However, injection has important limitations.

If too many artificial events are injected, they can alter the common reference, low-rank basis, noise estimate or motion raster being evaluated. The Allen hybrid framework therefore injects only a small number of units into each real recording and uses repeated independent iterations rather than densely contaminating one recording.

Our stage benchmark should follow the same principle: sparse injections, repeated across backgrounds.

Two complementary donor sets should be used.

## External template probes

For Neuropixels 1.0, the Allen hybrid framework uses templates derived from the IBL Brain-Wide Map and scales them over a broad amplitude range, typically 50–200 µV. This gives us an external waveform population that is independent of the Luke sorter.

These templates are especially useful for evaluating the operators themselves because their identity does not depend on our current pipeline.

They are not assumed to perfectly reproduce marmoset V1 waveform statistics.

External templates may be admitted without spatial interpolation only after a hard compatibility assertion verifies that source and target probe type, physical site coordinates, enabled-site pattern and channel-map convention match exactly. For matched NP1-to-NP1 templates, no interpolation or relocation should be introduced. If compatibility fails, the coordinate transform, interpolation, support channels and resulting truth footprint must be explicit benchmark inputs rather than hidden preprocessing.

Analytic compact, broad and depth-gradient challenge functions generated directly on the Luke geometry should be retained alongside biological templates. These test operator behavior without depending on the biological representativeness or geometry of an external donor bank.

## Luke-derived raw waveform families

A second donor set should be derived from recurrent waveform families identified directly in minimally processed Luke voltage.

Sorter membership should not define donor identity. The first injection pilot showed why: events assigned to the same discovery unit/window had very poor raw waveform agreement.

Luke-derived donors should instead require independently demonstrated recurrence, spatial compactness and waveform coherence.

The combined external + Luke donor strategy avoids relying entirely on either synthetic representativeness or sorter-selected "easy" neurons. Hybrid-ground-truth literature has explicitly noted that selecting well-isolated donor units introduces an unavoidable easy-unit bias.

Injection strata should eventually include:

- positive and negative polarity;
- compact and spatially extended waveforms;
- narrow and broad temporal waveforms;
- weak, intermediate and strong amplitude;
- different probe depths;
- quiet versus high-common-mode backgrounds;
- isolated spikes and controlled collisions.

Collision tests should be considered a later dedicated benchmark because collisions are a known independent failure mode of modern spike sorters.

---

# 6. Continuous common-mode removal: established methods first

The first benchmark should emphasize established, interpretable methods before more flexible models.

## A. Allen/AIND global common median reference

This should be the primary external baseline.

The current AIND pipeline performs phase/time correction, high-pass filtering, bad-channel detection/removal, and then either global CMR or IBL spatial destriping. CMR is the default because spatial destriping can introduce waveform artifacts. DREDGE motion is estimated, but voltage interpolation is disabled by default. The pipeline currently contains no general artifact-removal stage analogous to CatGT `gfix`.

The current documented default is a 300-Hz high-pass, coherence+PSD bad-channel detection, removal of bad channels, global median reference, and DREDGE motion estimation with `apply=false`.

This is an excellent clean-room baseline precisely because it is simple and externally developed.

It is **not** expected to solve catastrophic Luke artifacts by itself.

## B. Annular/local reference

Local referencing is an established option when background signal is not uniform over a long shank.

Power Pixels, for example, uses an annular reference that excludes the nearest contacts in order to avoid directly subtracting the same neuron's footprint. Its representative default annulus is approximately 50–200 µm.

SpikeGLX provides the same basic strategy but warns that spike footprints can be larger than expected, so local CAR can substantially attenuate true spikes. It suggests using a fairly large outer support region when background varies by anatomical depth.

Therefore the current Luke 40–140 µm reference is **not obviously wrong based on literature geometry alone**. It must compete empirically against wider annuli and global CMR on the injection benchmark.

Parameter selection should be based on physical micrometers, not channel count.

The initial wide comparator should be fixed in advance rather than selected from a radius sweep. A 60–480 µm annulus is a useful deliberately wider control. Its contributing-channel count will vary near shank ends and excluded channels, so every target channel must retain its actual reference support in the manifest. The comparison should require a prespecified minimum support or be restricted to a common valid depth region.

## C. IBL destriping / spatial high-pass filtering

IBL preprocessing treats some contamination as low spatial frequency along the probe rather than exactly common across all channels.

SpikeInterface's implementation uses spatial high-pass filtering and can remove "stripes" whose amplitude changes smoothly with depth.

DREDGE itself has been successfully evaluated on recordings preprocessed using the IBL chain of temporal filtering, demultiplexing phase correction, bad/noisy-channel handling and spatial destriping.

The important caveat is that Allen currently defaults to CMR because destriping can distort spike waveforms.

Destriping should therefore be an important candidate, but never judged by reduction in striping or RMS alone.

## D. Demultiplex-aware CAR

Demux-CAR should be retained primarily as a **mechanistic diagnostic**, not a default contender.

It groups Neuropixels channels according to the channels physically digitized at the same instant and computes the common reference within those groups.

This can outperform `phase shift + global CAR` for exceptionally rapid artifacts whose waveform changes appreciably over the multiplex cycle. SpikeGLX notes that its principal advantage concerns very fast components approaching or exceeding what can be accurately reconstructed at the normal 30-kHz sample rate. It also warns that the smaller reference groups are noisier and can generate inverted spike-like overcorrection artifacts.

Consequently:

- test demux-CAR on the least temporally restricted signal available;
- organize the largest shared artifacts by multiplex phase;
- ask whether phase-group-specific residuals are actually present;
- do not promote demux-CAR merely because it reduces individual dramatic events.

If a 6-kHz low-pass has already removed the very fast structure, much of the diagnostic advantage of demux grouping may already have been erased.

---

# 7. More flexible common-mode models: second tier

Only if the established methods leave substantial structured contamination while failing the preservation/suppression Pareto test should more flexible models be introduced.

## Robust per-channel regression

Ordinary global CAR assumes effectively identical nuisance amplitude on every channel:

\[
x_c(t)=s_c(t)+r(t).
\]

A simple extension is

\[
x_c(t)=s_c(t)+\beta_c r(t).
\]

The reference waveform \(r(t)\) is estimated robustly, while each channel receives its own robustly fitted coefficient.

Adaptive virtual-reference approaches have previously used multichannel reference signals with channel-specific adaptive filtering to recover extracellular spikes in severe common-mode interference.

The literature example was not a Neuropixels pipeline and involved a much smaller MEA, so this should be described as an adaptation of an established principle rather than a validated Neuropixels method.

Advantages:

- highly interpretable;
- only slightly more flexible than CAR;
- explicitly accommodates different artifact amplitudes along the shank.

Risks:

- coefficients can absorb genuine shared neural activity;
- fitting during spike-rich or artifact-rich samples can bias the nuisance model;
- time-varying coefficients can easily become over-flexible.

Initial fits should therefore be robust, slowly varying or fixed within reasonably long blocks, and trained with local spike neighborhoods excluded where possible.

## Leave-neighborhood-out low-rank regression

A richer model is

\[
X=L+S+E,
\]

where the shared nuisance \(L\) is approximated with a small number of latent components.

For each target neighborhood \(R\), estimate nuisance components from channels outside \(R\), then predict their expression inside \(R\). The target neuron's local footprint therefore cannot directly define the nuisance basis used to remove it.

This borrows from multichannel artifact-regression and PCA approaches but is **not an established standard Neuropixels preprocessing method**.

Its flexibility is both its advantage and its greatest danger.

It should not be optimized to maximize variance removed.

The number of components and spatial exclusion zone should be constrained using held-out prediction **and** injected-signal preservation.

Robust PCA or more elaborate source-separation models should remain later exploratory options rather than initial candidates.

---

# 8. Catastrophic artifacts must be separated from continuous denoising

This is an important revision to the earlier plan.

Large nonlinear/saturating events are not necessarily appropriate targets for CAR, destriping or low-rank subtraction.

SpikeGLX explicitly distinguishes ordinary common background from large fast multichannel "insults." `gfix` detects events based on amplitude, slope and the fraction of channels participating, then edits their affected interval. `gfix` and common referencing are intended to solve different problems and can be used together.

Critically, the destructive edit is performed after filtering. Current CatGT versions can line-fill the affected interval rather than simply leaving raw zeros.

This matters directly for Luke because the current pointwise blanker precedes later filtering and demonstrably creates ringing.

The first implementation should therefore separate three concepts:

**artifact detection sidecar**
→ identify event start/end, spatial extent, amplitude and slope without modifying the canonical voltage;

**artifact-aware motion estimation**
→ exclude detections/localizations associated with identified artifact windows before they reach DREDGE;

**optional post-filter editing for sorting**
→ only after the detector has been validated, compare leaving the interval untouched, masking/excluding it downstream, line filling, or another bounded replacement.

The preprocessing chain should not create a sharp discontinuity and then pass that discontinuity through a high-order temporal filter unless an explicit benchmark shows this is safe.

The amount of data removed or modified must be a first-class metric.

A method that achieves clean traces by deleting a substantial fraction of the session has not necessarily improved the recording.

The sidecar is an annotation system, not ground truth. It should be audited manually on discovery data for both false positives and false negatives, and its event boundaries should be compared in minimally processed and post-filter representations. Filtering may reveal the relevant insult, but it can also create the AP-band morphology used to detect it.

The annotation sidecar must exist before the first denoiser comparison so ordinary background, artifact shoulders and catastrophic cores can be reported separately. Destructive editing remains a later experiment.

---

# 9. Benchmarks for nuisance suppression

Nuisance suppression must be measured independently of Kilosort.

Useful metrics include:

### Long-range spatial covariance

Compute covariance/correlation between channels as a function of physical separation after excluding known injected events and major artifact intervals.

The objective is to reduce anomalously broad correlation while preserving expected local correlation.

### Common-component spectrum

Use robust PCA/SVD or another descriptive decomposition—not necessarily the same algorithm used for cleaning—to quantify how variance in the first few spatially broad components changes.

Do not optimize these components to zero.

### Diffuse-event burden

For threshold-crossing events, measure:

- number of participating contacts;
- depth span;
- local-energy fraction;
- spatial smoothness;
- temporal synchrony across distant contacts;
- positive/negative structure.

### Event-triggered common component

Around large shared events, inspect the common and depth-dependent components before and after each processing stage.

This should establish whether the AP-band event existed in the original signal or was sharpened/generated by processing.

### Cross-probe correspondence

Simultaneous probe-wide events are useful diagnostic observations.

Reduction of cross-probe synchrony is not itself a success criterion because genuine behaviorally or mechanically coupled signals can occur on both probes.

## Lightweight acquisition-cause lane

Computational rescue and causal diagnosis should proceed in parallel but retain separate success criteria. A preprocessing branch can be superior without identifying the original insult, and a hardware association does not establish how best to rescue an existing recording.

The initial causal-diagnosis sidecar should remain lightweight and include:

- event-triggered AP and LF views;
- exact raw-stream clipping and saturation statistics;
- acquisition reference and ground configuration;
- coincidence with available sync, behavior, reward and stimulation signals;
- multiplex phase and electrical grouping; and
- historical probe, headstage, cable or reference-path swaps that provide natural interventions.

These observations may motivate later mechanistic tests but should not determine which reference branch advances unless they are incorporated into a frozen event-level comparison.

---

# 10. Explicit artifact-creation challenges

Each preprocessing candidate should also be asked what artifacts it creates from known inputs.

Challenge signals should include:

- an isolated compact spike;
- a spatially broad spike;
- a pure common-mode impulse;
- a smooth depth-gradient transient;
- a sharp step-like common disturbance;
- a saturation-sized transient;
- a multiplex-phase-varying fast transient;
- and an empty/no-injection control.

Measure whether the operation creates:

- remote threshold crossings;
- inverted spike replicas;
- ringing;
- multiple extrema from one event;
- alternating spatial bands;
- apparent compact peaks from an originally diffuse event.

This is particularly important for spatial filters and local references, for which visually improved background can coexist with waveform distortion. Both Allen and SpikeGLX explicitly caution about such failure modes.

---

# 11. Bad-channel handling should remain a separate axis

Allen's current baseline removes detected bad/noisy channels before CMR, whereas the present Luke engineering baseline interpolates channel 191 and presents the synthesized trace to Kilosort.

Those choices should not be conflated with the common-mode comparison.

For the upstream benchmark:

- exclude known bad channels from nuisance estimation;
- exclude synthesized channels from covariance, localization and benchmark metrics;
- separately compare removal versus interpolation only for operations that require a complete geometry.

Injection probes near and far from channel 191 can determine the spatial region over which each policy affects waveform/localization estimates.

---

# 12. Motion estimation becomes a downstream internal-consistency analysis

Only denoising candidates that survive the voltage-level tests should be used to estimate motion.

For each surviving chain, run identical peak detection/localization and motion estimation and evaluate:

- peak count per second;
- compact/diffuse event fraction;
- depth and amplitude distribution of peaks;
- support per time-depth bin;
- split-half motion reproducibility;
- nearby-parameter stability;
- rigid versus nonrigid structure;
- agreement with the independent direct amplitude-depth raster tracker;
- cross-probe rigid correspondence;
- relationship between extreme nonrigid gradients and estimator support.

These are consistency and support metrics, not physical-accuracy metrics. Several reuse overlapping AP event populations, so apparent agreement between them is correlated evidence rather than independent validation. Cross-probe agreement is especially ambiguous: removing shared electrical contamination can reduce it, whereas preserving genuine shared mechanical motion can increase it.

This analysis tells us whether a frozen denoising method gives the motion estimator a healthier input population. Motion endpoints should be secondary descriptive outputs and must not determine advancement in the first reference experiment.

It does **not** independently prove the motion field is physically correct.

An independent anchor—LFP landmarks, stable waveform families, video/mechanical information, or another observable—remains desirable before assigning precise physical scale to the field.

No voltage interpolation is required for this stage.

---

# 13. Allen/AIND should be implemented as an independent clean-room baseline

The legacy pipeline should remain frozen as a historical comparator.

A new baseline should reproduce the current AIND/SpikeInterface preprocessing implementation as exactly as practical, including:

- software versions;
- temporal filter parameters;
- phase-shift parameters;
- bad-channel detection;
- channel removal;
- global CMR;
- DREDGE configuration;
- `motion.apply = false`;
- exact operator order;
- full manifest and processing hashes.

Because literature and production documentation change, every version-sensitive claim used to define this baseline should cite a frozen paper version, documentation revision or source commit. The manifest should record the SpikeInterface version, AIND pipeline version/commit, resolved preprocessing graph, all arguments and relevant source hashes rather than relying on the label "Allen pipeline."

Before execution, the plan's provenance appendix should freeze at least:

- the Allen/AIND paper version and hybrid-pipeline implementation commit;
- the SpikeInterface release and resolved preprocessing implementation;
- the CatGT version/documentation governing median CAR, local-CAR geometry and `gfix` behavior;
- the external template-bank version, probe type, coordinate table and file hashes; and
- the Luke probe metadata, enabled-site map and channel-map convention used by the compatibility assertion.

The Allen pipeline is attractive not because it is known to solve Luke but because it is simple, externally developed and extensively used. As of the AIND report, the production pipeline had processed more than a thousand sessions and thousands of individual probes. It also provides a dedicated hybrid evaluation framework explicitly intended to compare processing changes.

There is an important limitation: **the current Allen preprocessing pipeline explicitly lacks general artifact detection/removal**, and its authors identify this as a planned addition.

Therefore an Allen failure on Luke would not imply that standard Neuropixels preprocessing is inadequate in general. It may simply establish that Luke requires an explicit artifact stage.

---

# 14. Proposed experimental ladder

## Phase 0: passport identity calibration

Do not change the legacy production defaults.

Implement the stage-passport harness and manifests, then measure identity/no-op repeatability and equivalent materialization paths. Use these results to freeze engineering tolerances and the exact decision procedure before opening the method comparison.

## Phase 1a: characterize the disturbance

On minimally processed voltage, quantify:

- raw/common component;
- depth-dependent common structure;
- event rise times;
- saturation and clipping;
- multiplex-phase dependence;
- cross-probe timing;
- AP/LF and available behavioral correspondence; and
- transformation of the same event across phase correction and temporal filtering.

This phase should determine whether demux-CAR remains mechanistically plausible and whether the major artifact population is fundamentally continuous/common-mode or episodic/catastrophic. Demux-CAR remains a separate targeted diagnostic rather than a branch in the first reference comparison.

## Phase 1b: non-destructive artifact annotation

Build a permissive artifact sidecar on dedicated discovery snippets and manually audit its false-positive, false-negative and boundary behavior. Do not edit the canonical voltage.

Before applying the finalized annotator to select benchmark windows, freeze the window-selection rules. Then freeze six windows: relative quiet, ordinary/typical and catastrophic-artifact windows on each probe. Window selection must not be optimized after inspecting sidecar summaries or branch outcomes. Use existing prespecified discovery windows where practical.

Every later voltage metric should be reported separately for:

- ordinary background;
- pre/post-artifact shoulders;
- catastrophic-event cores; and
- the aggregate window.

Do not open the remaining sealed confirmatory cohort for annotator tuning, window selection or parameter tuning.

## Phase 2: compact four-way reference experiment

The first comparison is a fixed 2 probes × 3 windows × 4 branches = 24 processed chunks experiment:

1. minimal temporal preprocessing without spatial reference;
2. exact Allen/AIND global CMR baseline;
3. current Luke 40–140 µm local reference; and
4. one prespecified 60–480 µm wide annular reference.

Within each of the 24 processed chunks, use the frozen artifact sidecar, sparse paired injected/uninjected challenges and real-event nuisance measurements. Approximately 20–30 sparse challenges per chunk may be used to span amplitude, polarity and footprint, but they are repeated measurements rather than independent recordings.

Maintain two scorecards:

**Signal preservation**

- amplitude retention;
- waveform cosine;
- timing and localization;
- remote/opposite-polarity ghost energy;
- polarity flips; and
- lower-tail failures.

**Nuisance behavior**

- diffuse-event burden;
- long-range covariance;
- broad-component variance;
- catastrophic residual amplitude; and
- generated threshold crossings.

Do not run destriping, flexible regression, spike sorting, motion interpolation or a weighted overall score in this experiment.

## Phase 3: select, or fail to select, a reference strategy

Apply the frozen voltage-level decision rule. A valid outcome is that no branch advances. Results must be probe-stratified; pooled Luke results cannot override a failure on either probe.

Only after the method comparison is closed should the sealed confirmatory cohort be opened to test the selected branch and frozen parameters.

## Phase 4: later denoising escalation, only if needed

If the four established branches fail, test IBL destriping next using the same passport and decision framework.

Only if established approaches still leave substantial removable shared structure should the following be introduced:

1. robust per-channel common-reference regression;
2. leave-neighborhood-out low-rank regression.

Complexity is justified only if it moves the signal-preservation/nuisance-suppression Pareto frontier.

## Phase 5: artifact-aware motion estimation

With preprocessing frozen, determine whether excluding artifact-associated detections/localizations improves the peak raster and motion estimate **without modifying voltage**.

Estimate, but do not apply, motion. Report estimator support, split-half and nearby-parameter stability, direct-raster agreement and cross-probe correspondence as secondary internal-consistency metrics.

## Phase 6: independent motion validation

Use LFP landmarks, independently qualified stable waveform families, video/mechanical information or another genuinely separate observable before assigning physical accuracy or scale to a motion field.

## Phase 7: destructive editing and downstream sorting

Only if sorting requires it, compare leaving catastrophic intervals untouched, masking/excluding them downstream, post-filter line filling or another bounded replacement. Report all modified intervals and duty cycle.

Only after voltage preprocessing and artifact policy are frozen should one or two finalists proceed to short sorting comparisons. Voltage motion interpolation remains a separate, later question.

---

# 15. Advancement criteria

The decision procedure, not biologically unmotivated magic numbers, should be preregistered. It should combine identity-calibrated hard engineering failures, preservation noninferiority, a minimum useful nuisance improvement, paired uncertainty intervals and Pareto dominance. Particular emphasis should be placed on lower-tail waveform performance rather than only means or medians.

The following values are proposed discovery gates, not yet frozen tolerances. They must be checked against Phase 0 identity/repeatability measurements and then frozen before the branch comparison:

| Endpoint | Proposed discovery gate |
|---|---:|
| Median injected amplitude retention | ≥0.98 |
| 10th-percentile amplitude retention | ≥0.90 |
| Median waveform cosine | ≥0.98 |
| 10th-percentile waveform cosine | ≥0.90 |
| 95th-percentile localization error | ≤20 µm |
| Polarity flips | <1% of compact challenge events |
| Remote threshold crossings generated | <0.05 per event |
| Catastrophic-artifact edited duty cycle | report; >1% triggers explicit review rather than automatic rejection |
| Shared/diffuse nuisance burden | ≥25% improvement over minimal baseline, or a prespecified practically clear Pareto improvement |

For each spatial-reference branch, preservation should be evaluated primarily against the paired minimal-temporal output on the same background and injection. A candidate advances only if it passes the frozen preservation noninferiority and engineering gates and improves at least one important nuisance endpoint without materially worsening another. No weighted composite score should resolve tradeoffs.

For paired preservation comparisons, the hierarchical interval should exclude the frozen practically important loss margin; merely failing to detect a difference is not evidence of noninferiority.

Uncertainty must respect the experimental hierarchy. Injections within a chunk are repeated measurements, so intervals should resample windows and waveform families, with event-level resampling nested within them where appropriate. Event rows must not be treated as independent recordings. With only six windows, statistical clarity cannot substitute for a prespecified practically important effect size.

A candidate should be rejected or modified if it:

- creates remote or inverted spike-like residuals;
- substantially changes localization of compact injected events;
- attenuates a distinct waveform class;
- exceeds the frozen editing/duty-cycle review policy;
- or gains apparently cleaner voltage primarily by removing local correlated neural structure.

No method should be selected on:

- noise RMS alone;
- number of threshold crossings alone;
- amount of PCA variance removed;
- DREDGE motion smoothness alone;
- Kilosort unit count;
- KS-good count;
- or contamination estimate.

The final choice should lie on a defensible probe-stratified Pareto frontier. A valid result is failure to select any candidate.

---

# 16. Main strengths and limitations of this plan

The major strength is **causal localization**.

Instead of asking Kilosort whether preprocessing worked, known signals and known artifact challenges are followed through individual transformations. This follows the same modular philosophy that has made motion-correction benchmarking interpretable and the same hybrid-ground-truth philosophy used by Allen, SpikeForest and earlier sorter-validation studies.

A second strength is that established methods are used as anchors before novel algorithms are introduced.

The largest limitations are:

**Hybrid representativeness.**  
Injected spikes cannot span every real waveform, collision or tissue configuration. External and Luke-derived donors should therefore be combined.

**Nonlinearity and estimator interaction.**  
Median references, artifact detectors and low-rank estimates can themselves change when a spike is injected. Sparse injection is necessary, and this interaction should be regarded as part of the operator being tested rather than ignored.

**Genuine shared neural activity.**  
A method can appear to suppress "common noise" by removing real synchronized or spatially extended biological voltage. Common-component suppression is therefore never sufficient evidence of improvement.

**Discovery-set overfitting.**  
The same Luke windows and reviewed events have already influenced many decisions. Candidate selection should use dedicated discovery data and synthetic/external injections; the existing sealed remainder should remain confirmatory.

**No independent tissue-motion ground truth.**  
A preprocessing branch that produces a cleaner, more reproducible DREDGE field may still be wrong in physical scale. Motion validation remains a separate problem.

**Model flexibility.**  
Low-rank and adaptive-regression methods can always be made to produce cosmetically clean traces. Their flexibility makes strict signal-preservation tests more important, not less.

---

# Current recommended starting point

The immediate implementation should therefore be:

**passport identity calibration → non-destructive artifact sidecar → frozen six-window/challenge-bank design → four-way reference experiment.**

The four branches are minimal temporal preprocessing, exact Allen/AIND global CMR, the current Luke 40–140 µm local reference and a prespecified 60–480 µm wider annular reference. All outcomes remain probe-stratified and are reported on separate signal-preservation and nuisance-behavior scorecards.

The first question is not yet whether sophisticated regression beats CAR.

It is:

> **Where does the large shared Luke disturbance originate, how does each established preprocessing operation transform it, and can a standard modern conditioning chain remove enough of it while demonstrably preserving known spike waveforms?**

If the initial references fail, IBL destriping becomes the next established candidate. Only if it also leaves a large structured residual without an acceptable preservation/suppression tradeoff do robust per-channel regression and leave-neighborhood-out low-rank subtraction become well-motivated experiments rather than speculative preprocessing complexity.

The objective is to reach a point where motion estimation and spike sorting receive a population of compact extracellular events whose transformation through every earlier stage is understood and quantitatively bounded.
