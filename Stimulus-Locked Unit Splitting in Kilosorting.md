Stimulus-Locked Unit Splitting in Kilosort4: Mechanistic Hypotheses and Tests
Framing the problem

We observe discontinuous unit identity changes aligned to stimulus epochs in Neuropixels data processed with Kilosort4 (KS4). A common intuition is that neurons have stereotyped extracellular waveforms, invariant up to scaling and small shifts. Under that assumption, a well-behaved spike sorter should not produce condition-dependent splits.

However, KS4 does not explicitly implement a continuous generative model of waveform identity. Instead, it approximates this through a sequence of discrete and partially decoupled steps. This creates a gap between:

Biophysical invariance (true waveform stability), and
Algorithmic invariance (what KS4 can represent and recover)

The goal here is to identify mechanisms within KS4 that can produce stimulus-locked splitting despite stationary underlying neurons, and to propose testable hypotheses grounded in the actual implementation.

KS4 pipeline: what is actually implemented

From the documentation and code, KS4 follows roughly:

Preprocessing
High-pass filtering
Common average referencing (CAR)
Channel whitening (local neighborhoods)
Drift estimation and correction
Performed before spike inference
Parameterized by nblocks (rigid vs non-rigid)
Universal spike detection
Uses a discrete lattice of template centers
Finite spatial scales (template_sizes)
Local channel neighborhoods (nearest_chans)
Initial clustering
Graph-based clustering in feature space (tF)
Learned template extraction
Final clustering
Graph clustering again on extracted features
Merge stage
Based on:
waveform correlation (r_thresh)
cross-correlogram criteria (ccg_threshold, acg_threshold)

Crucially:

KS4 does not jointly infer waveform identity and drift in a continuous parameter space.
Instead, it relies on discrete approximations + sequential corrections.

Key implication

Even if a neuron is perfectly stationary in raw voltage space:

It may not appear stationary in KS4 feature space.

This opens the door to condition-dependent clustering behavior.

Mechanistic hypotheses
1. Fast drift or mis-registration (highest priority)

Mechanism

Drift correction is estimated first, not jointly inferred
Fast or event-locked motion (e.g. licking, posture, cable movement) can:
violate the assumed timescale of drift
produce residual spatial misalignment

The KS4 docs explicitly warn that fast drift can produce behavior-dependent apparent firing changes.

Consequence

Same neuron appears at slightly different depths or channel profiles
At low rate → looks like noise
At high rate (stimulus) → becomes separable cluster

Prediction

Splits align with behavioral/stimulus events that induce motion
Sorting outcome depends on drift parameters

Test

Run with:

nblocks = 0 (no drift correction)
nblocks = 1 (rigid)
nblocks > 1 (non-rigid)

Evaluate:

Does the split timing change?
Does spike conservation improve (sum of split units)?
Do amplitude vs depth trajectories become smoother?
2. Learned-template threshold dropout

Mechanism

After initial clustering, KS4 learns templates and re-detects spikes
Detection depends on Th_learned

Docs explicitly note:

neurons may “disappear and reappear over time” depending on this threshold

Consequence

During certain epochs (e.g. stimulus):
spikes fall below detection threshold
or are reassigned to neighboring templates

This creates:

apparent unit dropout
or “daughter” clusters absorbing spikes

Prediction

Splits coincide with marginal detection regimes
Lowering threshold restores continuity

Test

Rerun with:

Th_learned reduced by 1–2

Check:

Does the parent unit recover spikes?
Does the daughter unit shrink or vanish?
Does total spike count become smoother?
3. Discrete template lattice (representation mismatch)

Mechanism

Spike detection uses:

discrete spatial positions (dmin, dminx)
limited channel neighborhoods (nearest_chans)
Gaussian spatial envelopes (min_template_size)

This approximates spatial invariance, but only coarsely.

Consequence

Small shifts in observed waveform (even if biologically trivial) can:

move spikes to a different template center
change feature representation

At high firing rate:

enough samples accumulate to form a separate cluster

Prediction

Splits depend on spatial discretization
Boundaries shift when detection geometry changes

Test

Perturb:

dmin, dminx
nearest_chans
min_template_size

Check:

Does the split persist?
Does its timing shift?
4. Graph clustering oversplit + failed merge

Mechanism

Clustering operates on PC features (tF)
Oversplitting is expected and later corrected by merging
Merge requires:
high waveform correlation
acceptable cross-correlogram structure

If condition-dependent effects distort:

feature distributions
temporal structure

then merge may fail

Consequence

One neuron becomes two clusters
Splits persist because merge criteria are not satisfied

Prediction

Split units:
have similar waveforms
but fail merge due to CCG structure
Spike counts trade off between clusters

Test

For candidate pairs:

compute waveform correlation
compute cross-correlogram
compute summed firing rate

Key diagnostic:

If sum is smooth but clusters anti-correlate → oversplit

Also:

test sensitivity to ccg_threshold
5. Preprocessing / whitening-induced feature shifts

Mechanism

Whitening uses local covariance estimates (whitening_range)
Background statistics may change across stimulus epochs

Thus:

identical raw waveforms can map to different feature vectors

Consequence

feature space becomes state-dependent
clustering separates conditions

Prediction

Raw waveforms remain unimodal
Preprocessed features become bimodal

Test

Compare:

raw snippets vs preprocessed snippets
PCA projections across conditions

If:

raw space = continuous manifold
feature space = split clusters

→ failure originates upstream of clustering

Diagnostic strategy

For a suspected split pair (A, B):

1. Spike conservation
Compute: rate(A + B) over time
If smooth → strong evidence for artificial split
2. Temporal tradeoff
Check anti-correlation of A and B across bins
Especially around stimulus events
3. Waveform consistency
Align raw snippets across conditions
Test for unimodality vs bimodality
4. Drift sensitivity
Rerun with different nblocks
Inspect depth and amplitude trajectories
5. Threshold sensitivity
Lower Th_learned
Check for re-merging or recovery
6. Representation sensitivity
Perturb spatial parameters (dminx, nearest_chans)
Look for instability of split

Operationalizing these tests in this repo (what to run + what to look for)

The shallow sweep workflow is designed to make these hypotheses falsifiable with minimal reruns.

1) Run a small parameter ablation (mechanism probe)

- Template threshold dropout hypothesis:
	Sweep Th_learned (e.g. 9 → 6 → 12). Expect condition-locked “daughter” units to shrink/disappear when Th_learned is lowered if dropout/assignment thresholding is causal.

- Oversplit + failed merge hypothesis:
	Sweep ccg_threshold (e.g. 0.90 → 0.60 → 0.40). If the split is mostly graph oversplitting, more aggressive merging (lower threshold) should reduce the number of split cases without changing the combined firing of the neuron.

- First-pass detection sensitivity:
	Sweep Th_universal (e.g. 9/12/15). If the issue is marginal detection, changing this should strongly alter apparent “unit dropouts” (loss of spikes) rather than merely re-partitioning spikes between two children.

- Drift / mis-registration hypothesis:
	Requires adding an nblocks sweep (0/1/>1) at the sorter stage. If event-locked motion is causal, split signatures should be highly sensitive to nblocks and coincide with depth/amplitude trajectory kinks.

2) Use the split diagnostics (evidence of algorithmic identity swapping)

The script compare_shallow_sweeps.py now writes two extra outputs:

- split_diagnostics.csv
	For each reference unit and each other run, it finds the top-2 coincident matches (child1/child2) and computes:
	- split_score = child1_frac + child2_frac
		High means the reference spikes are “explained” by two units.
	- segregation = mean |c1 - c2| / (c1 + c2) across time bins
		High means the two children dominate in different time bins (i.e. a handoff).
	- anticorr = corr(c1, c2) across bins
		Negative supports a tradeoff between children.
	- conservation_mean = mean (c1 + c2) / ref_counts across bins (where ref_counts>0)
		High supports spike conservation (suggesting a split rather than true spike loss).

- fig_split_diagnostics.pdf
	One page per flagged candidate showing time-binned counts and a “child1 fraction” trace.
	A stimulus-locked split typically looks like step-like changes in child1 fraction across time.

Interpretation shortcuts

- “Clean split artifact” pattern:
	split_score high + segregation high + anticorr negative + conservation_mean high.
	This supports oversplit / merging issues more than true neuron nonstationarity.

- “True dropout / detection loss” pattern:
	conservation_mean low (combined children do not recover reference spikes), especially if sensitive to Th_learned or Th_universal.

- “Geometry / discretization instability” pattern:
	split candidates appear/disappear or move around when changing nearest_chans / spatial parameters, without a consistent conservation signature.

Notes / caveats

- These diagnostics don’t require explicit stimulus timestamps; they look for time-localized handoffs. If you do have stimulus epochs, overlaying them should make the interpretation much sharper.
- The coincidence window (COINC_TOLERANCE) is a real knob: if it’s too tight, conservation will look artificially low.
Conceptual takeaway

The key distinction is:

Biological identity ≠ cluster identity in KS4

Even if:

the neuron is stationary
the waveform is stereotyped

KS4 operates on:

discretized spatial templates
approximate drift correction
locally whitened feature space
heuristic clustering + merging

So the effective model is:

identity = cluster in a condition-dependent feature space

not:

identity = invariant waveform under continuous transforms

Working hypothesis

The most plausible unified explanation is:

Stimulus epochs change population activity and/or motion, which alters the effective feature geometry and template competition, revealing latent nonstationarity or pushing the data across clustering boundaries.

This does not require large biological waveform changes.

Priority ranking of mechanisms
Drift / mis-registration
Learned-template threshold dropout
Discrete template lattice effects
Graph oversplit + failed merge
Whitening / preprocessing mismatch
Next step

A small, controlled ablation on a short time window:

nblocks ∈ {0, 1, 5}
Th_learned ∈ {8, 7, 6}
one spatial perturbation

combined with spike conservation and waveform diagnostics

should be enough to distinguish:

true waveform nonstationarity
vs
sorting instability

If you want, we can formalize this into a reusable analysis script that automatically flags likely temporal splits and classifies them by mechanism.