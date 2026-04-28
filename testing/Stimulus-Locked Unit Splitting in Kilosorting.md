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
Discrete lattice of template centers
Finite spatial scales (template_sizes)
Local neighborhoods (nearest_chans)
Initial clustering
Graph-based clustering in feature space (tF)
Learned template extraction
Final clustering
Merge stage
Waveform correlation threshold
Cross-correlogram (CCG) criteria

Crucial point:

KS4 does not jointly infer waveform identity and drift in a continuous parameter space.
It relies on discrete approximations + sequential corrections.

Key implication

Even if a neuron is perfectly stationary in raw voltage space:

It may not appear stationary in KS4 feature space.

Mechanistic hypotheses
1. Fast drift or mis-registration

(unchanged, still high priority)

2. Duplicate fitting from imperfect peeling (CONFIRMED as major mechanism)

This is now the best-supported KS4-specific mechanism in this dataset.

Mechanism

KS4 performs iterative template matching ("peeling"):

Detect spikes from template responses
Subtract fitted templates from the residual
Repeat on the residual for multiple passes (max_peels)

If subtraction is imperfect:

A spike is detected once
Residual structure remains
Residual exceeds detection threshold
Same event is detected again on a later peel

Critically:

The second detection may be assigned to a different template
This creates duplicate spikes assigned to different clusters
Merge failure

KS4 merge logic requires:

waveform similarity
acceptable cross-correlogram (CCG)

Duplicate fits violate refractoriness:

many near-zero-lag coincidences
CCG appears "refractory-violating"

So:

The clusters cannot merge, even if they represent the same neuron.

Consequence

You get:

two clusters representing one neuron
spikes split across them
strong near-coincident activity
persistent failure to merge
Why this can be stimulus-locked

Peeling errors are state-dependent:

high-rate stimulus epochs -> more overlap
more overlap -> worse subtraction
worse subtraction -> more duplicate detections

So the mechanism is always present, but:

It becomes visible only during high-activity periods.

Predictions
Two clusters with:
very similar waveforms
nearby depths
Strong near-zero-lag coincidence
High spike conservation across the pair
Negative or structured correlation at coarse timescales
Insensitivity to merge threshold (ccg_threshold)
Sensitivity to peeling parameters
Tests
Reduce max_peels
e.g. max_peels = 1
Expect daughter unit to shrink/disappear
Increase Th_learned
Residual fits should disappear first
Examine fine-timescale cross-correlogram
Look for excess near-zero-lag coincidences
Check timing offsets (dt)
Near-identical templates with small lag differences
Compare across stimulus vs baseline
duplication should increase in high-rate periods
3. Learned-template threshold dropout

(unchanged)

4. Discrete template lattice effects

(unchanged)

5. Graph clustering oversplit + failed merge

This remains relevant but is now secondary to duplicate-fit failures.

6. Preprocessing / whitening mismatch

(unchanged)

Diagnostic strategy (updated)

For a suspected split pair (A, B):

1. Spike conservation
rate(A + B) smooth -> artifact
2. Temporal tradeoff
anti-correlation across bins
3. Fine-timescale coincidence (new key test)
near-zero-lag peak -> supports duplication
4. Waveform consistency
raw snippets unimodal vs bimodal
5. Peeling sensitivity (new key test)
rerun with max_peels = 1
6. Threshold sensitivity
vary Th_learned
Interpretation of current shallow sweep results (updated)

Cross-run split diagnostic (temporal handoff test):

- Flagged split-handoff candidates appear in threshold / geometry-type runs, not in the dedicated ccg_threshold sweeps. This rules out merge-threshold failure as the primary cause.
- Flagged counts vs the default run:
	Th_lo: 2,  Th_hi: 2,  Thl_lo: 1,  Thl_hi: 1,  Thu_hi: 1,  chans_fewer: 1
	Thu_lo, chans_more, dist_lo, dist_hi, ccg_lo, ccg_mid, ccg_hi: 0
- Strongest examples:
	Th_lo, ref unit 6: split_score ~1.03, anticorr ~-0.22, conservation_mean ~0.98
	Th_lo, ref unit 29: split_score ~1.00, anticorr ~-0.56, conservation_mean ~0.96
	chans_fewer, ref unit 6: split_score ~0.98, anticorr ~-0.84, conservation_mean ~0.97

Within-run near-zero-lag CCG screen (peeling duplication test) — max_peels dose-response:

Dataset 0316 (marginal SNR, median_mpct ~30% at default):

| run       | max_peels | nearby pairs | median nzf | flagged pairs |
|-----------|-----------|-------------|------------|---------------|
| default   | 100       | 273         | 0.157      | 236           |
| peel3     | 3         | 195         | 0.091      | 53            |
| peel2     | 2         | 132         | 0.075      | 21            |
| peel1     | 1         | 14          | 0.000      | 0             |

Dataset 0302 (better SNR, median_mpct ~7% at default) — independent replication:

| run       | max_peels | nearby pairs | median nzf | flagged pairs |
|-----------|-----------|-------------|------------|---------------|
| default   | 100       | 1443        | 0.194      | 971           |
| peel3     | 3         | 616         | 0.095      | 260           |
| peel2     | 2         | 235         | 0.071      | 57            |
| peel1     | 1         | 0           | —          | 0             |

The monotonic collapse replicates across both datasets. The mechanism is not recording-specific. The within-run duplicate-peel signal is present at scale in any default-settings KS4 run on these Neuropixels recordings.

Practical quality — 0316 (marginal SNR):

| run     | n_units | n_good | n_well | efficiency | median_mpct |
|---------|---------|--------|--------|------------|-------------|
| peel1   | 22      | 22     | 6      | **0.273**  | **20.3%**   |
| peel2   | 21      | 4      | 6      | 0.286*     | 25.4%       |
| peel3   | 27      | 5      | 5      | 0.185      | 24.9%       |
| default | 34      | 8      | 5      | 0.147      | 29.6%       |

*peel2's efficiency is inflated by near-duplicate units; only 4/21 pass KS4's refractory criterion.

Practical quality — 0302 (better SNR):

| run     | n_units | n_good | n_well | efficiency | median_mpct |
|---------|---------|--------|--------|------------|-------------|
| peel3   | 46      | 27     | 11     | **0.239**  | **8.5%**    |
| peel2   | 30      | 19     | 7      | 0.233      | 10.5%       |
| peel1   | 26      | 26     | 5      | 0.192      | 10.3%       |
| default | 64      | 30     | 12     | 0.188      | 6.8%        |

Claim-mask follow-up — 0302 code-level intervention:

To test whether the same mechanism can be suppressed without collapsing overlap recovery as aggressively as `max_peels = 1`, we patched KS4 with a minimal cross-peel claim rule:

- `claim_tonly`: suppress later-peel candidates within 0.25 ms of an already accepted event
- `claim_spatial`: same time rule plus a 75 um spatial radius

Using the same within-run nearby-pair screen and the same duplicate thresholds (`near_zero_frac >= 0.05`, `zero_peak_ratio >= 1.25`), the 0302 claim-mask branch gives:

| run           | n_units | n_good | n_well | efficiency | median_mpct | nearby pairs | median nzf | flagged pairs |
|---------------|---------|--------|--------|------------|-------------|--------------|------------|---------------|
| default       | 65      | 33     | 12     | 0.185      | 5.62%       | 1335         | 0.189      | 912           |
| claim_tonly   | 48      | 32     | 12     | 0.250      | 6.18%       | 726          | 0.043      | 115           |
| claim_spatial | 51      | 32     | 14     | **0.275**  | 3.56%       | 778          | 0.044      | 124           |
| peel3         | 40      | 26     | 13     | **0.325**  | **3.48%**   | 355          | 0.101      | 181           |

Interpretation:

- Both claim-mask variants sharply suppress the duplicate-peel signature relative to default: flagged nearby pairs fall from 912 to 115-124, and median near-zero fraction drops from 0.189 to 0.043-0.044.
- The burden reduction is stronger than `peel3` on the duplicate screen itself (`115-124` flagged pairs versus `181` for `peel3`), while preserving more units (`48-51` versus `40`).
- `claim_spatial` is the best-balanced practical intervention so far on 0302: it preserves almost all good units (32 versus 33 at default), increases well-isolated units (14 versus 12), and raises efficiency from 0.185 to 0.275.

This is the first result in this project where a targeted code-level intervention looks better than a pure `max_peels` reduction for the good-SNR regime. The mechanistic interpretation remains the same: later peels are re-claiming event cores that should already be owned by an earlier match, and preventing that re-claiming removes a large fraction of the split burden without forcing KS4 into a near-single-peel operating mode.

Key contrast: in 0316, peel1 dramatically improves efficiency (0.147 → 0.273) because the recording is marginal and duplicate artifacts dominate unit quality. In 0302, the units are well above threshold (median_mpct 6.8%) and peel1 barely moves the needle (0.188 → 0.192) — here peel3 is the better tradeoff, recovering more genuine units while the within-run duplicate fraction remains tolerable.

The optimal max_peels is therefore SNR-regime dependent:
- Marginal SNR (most units near detection floor): peel1 — duplicate suppression outweighs sensitivity loss
- Good SNR (units well above floor): peel3 — extra sensitivity recovers real units without dominant duplicate contamination

The claim-mask result refines that conclusion. For good-SNR recordings like 0302, `peel3` is no longer the best available tradeoff once cross-peel re-claiming is explicitly blocked. The current best practical setting is the spatial claim-mask variant, because it preserves default-like yield while removing most of the within-run duplicate burden.

Pre-curation versus post-curation interpretation

An important distinction in this repo is that the comparison can now be run either:

- after curation/merging, using cur/cur_sorter_output, or
- before curation, using kilosort4/sorter_output.

The concern is not that curation creates the problem. The concern is that curation may partially repair it by applying additional merges, thereby compressing the observed failure mode and making the underlying mechanism harder to diagnose.

Current evidence supports that concern:

- The split / duplicate-like signatures are already visible in the pre-curation comparison, so the phenomenon is not introduced by the curation stage.
- The post-curation outputs can still be useful for judging practical downstream quality, but they are not the cleanest view of the raw KS4 failure mode.
- Therefore, mechanistic interpretation should prioritize the pre-curation analysis, while post-curation analysis should be treated as a secondary view of what survives after partial repair.

Operationally, this means:

- If a candidate appears before curation and weakens or disappears after curation, that is evidence that curation is masking part of the sorter error, not evidence that the raw error was absent.
- If a candidate survives both before and after curation, that is stronger evidence for a robust underlying sorter-stage failure.

Reverse-direction peel1 test (post- and pre-curation)

The crucial follow-up test was to reverse the comparison direction and use peel1 as the reference run.

Why this matters:

- The original split diagnostic asks whether another run appears to split the reference units.
- If the hypothesis is that default contains extra duplicate or fragment units that peel1 removes, then peel1 should be the reference and default should light up as splitting peel1 units.

What we observed:

- Post-curation, with peel1 as the reference, default had:
	n_flagged_splits_vs_ref = 0
	n_duplicate_fit_candidates_vs_ref = 0
- Pre-curation, with peel1 as the reference, default again had:
	n_flagged_splits_vs_ref = 0
	n_duplicate_fit_candidates_vs_ref = 0

This is the key result for interpreting the peeling hypothesis.

It means the reverse-direction test did not show default's extra units fragmenting peel1 reference units in the specific way the current diagnostic is designed to detect, and that conclusion holds both before and after curation.

So:

- Curation is not the reason the reverse-direction default-versus-peel1 result is quiet.
- The absence of a strong reverse-direction signal is present in the raw pre-curation sorter output.
- This is a negative result for one specific cross-run prediction, not a falsification of the peeling mechanism.

The cross-run split diagnostic tests temporal handoffs between runs. It is not designed to detect within-run duplicate detections. The within-run CCG screen is the correct test, and its result is unambiguous (see results table above).

Why peeling duplication is now the leading explanation:

- Merge threshold changes do not help → merge-threshold failure ruled out
- Strong conservation suggests spikes are not lost → repartitioning, not dropout
- Within-run near-zero-lag CCG signal collapses monotonically with max_peels → direct causal evidence
- peel1 has 22/22 good KS4 labels → KS4's own refractory criterion confirms the duplicate-peel units are the bad ones
- Residual: peel1 still has 1 flagged cross-run split with zero near-zero CCG signal → a secondary mechanism (threshold / representation instability) contributes at a lower level

Important caveat:

The cross-run split diagnostic (temporal handoff between a reference unit and two child units in another run) is not the right test for within-run peeling duplication. The peeling hypothesis predicts that extra peeling passes create near-zero-lag duplicate detections of the same event, assigned to different templates within the same run. That shows up as a strong near-zero-lag peak in the cross-CCG between those two templates — not as a cross-run temporal handoff.

The reverse-direction peel1 test is therefore a negative result for one specific prediction (cross-run fragmentation of peel1 units by default), not a falsification of the broader peeling story.

The correct test is now implemented: within_run_duplicate_screen in compare_shallow_sweeps.py computes cross-CCGs between all nearby unit pairs (< 100 µm) in the raw pre-curation KS4 output for default vs peel1. If default has a heavier right tail of near-zero-lag fraction than peel1, that is direct evidence for within-run duplicate detections from extra peeling passes.

The other important caveat is that post-curation summaries may understate the magnitude of the raw problem if curation merges away some of the offending units. For diagnosing mechanism, the pre-curation outputs are the more faithful readout.

Revised priority ranking (updated after within-run CCG results)
1. Duplicate fitting from imperfect peeling (CONFIRMED, dominant mechanism)
2. Threshold / assignment instability (residual — explains splits that survive peel1)
3. Fast drift or mis-registration
4. Discrete template lattice effects
5. Graph oversplit + failed merge
6. Whitening / preprocessing mismatch
Critical falsification experiment (COMPLETE)

max_peels ablation: done. peel1 (max_peels=1) has 22/22 good-labeled units vs 8/34 for default (max_peels=100). Operationally decisive.

Within-run near-zero-lag CCG screen: done. The dose-response across default → peel3 → peel2 → peel1 is monotonic and unambiguous (see results table in the interpretation section). This is the direct mechanistic confirmation.

Conclusion: duplicate fitting from later peeling passes is a major causal source of within-run unit contamination in this dataset. Reducing max_peels to 1 eliminates that artifact while improving downstream QC. Some residual cross-run repartitioning remains at peel1, so peeling duplication is not the only mechanism — threshold and representation instability contribute at a lower level. The elbow is sharp: peel2 already reintroduces 21 flagged within-run pairs and collapses KS4's good-unit count from 22 to 4.

Conceptual takeaway (refined)

The key issue is not just clustering:

KS4 can create multiple representations of the same physical event.

And once that happens:

downstream merge logic may be structurally unable to repair it.

So the effective failure mode is:

detection-level duplication -> assignment divergence -> merge blockade

Working hypothesis (confirmed across two datasets)

Later peeling passes are a major causal source of within-run unit contamination in KS4. The within-run near-zero-lag CCG signal collapses monotonically with max_peels in both the 0316 (marginal SNR) and 0302 (good SNR) datasets. The dominant failure mode under default settings is:

detection-level duplication (extra peeling passes) → near-zero-lag spikes across nearby templates → KS4 refractory violations → merge blockade → proliferation of mua-labeled units

The optimal max_peels setting is SNR-dependent. In marginal-SNR recordings peel1 is recommended; in good-SNR recordings peel3 recovers more genuine units at an acceptable duplicate cost. A practical heuristic: choose the lowest max_peels value where n_well and efficiency still clearly exceed the default, as indicated by the within-run CCG screen showing a low flagged-pair count.

Next steps

1. ~~Within-run CCG screen (fig4)~~ — done. Peeling duplication confirmed.
2. ~~max_peels dose-response (peel2, peel3)~~ — done across two datasets.
3. ~~Independent dataset replication (0302)~~ — done. Mechanism replicates; optimal max_peels is SNR-dependent.
4. Evaluate claim_tonly and claim_spatial runs (0302) — tests whether a cross-peel duplicate suppression mask can recover peel1-like duplicate control without discarding later peels entirely.
5. Add stimulus-vs-baseline conditioning to the top within-run pairs — confirm the duplicate rate increases during high-activity epochs, completing the stimulus-locked story.
6. Finer Th_learned sweep (Thl_7, Thl_8) to map the residual threshold-instability contribution.
7. Overlay stimulus timing on the flagged cross-run split pages once epoch timestamps are available.

Implementation order

1. Run and evaluate claim_tonly / claim_spatial on 0302 — if the claim mask suppresses within-run duplicates while keeping peel3/peel4 sensitivity, that is the best practical parameter recommendation.
2. Stimulus epoch overlay on fig4 top-pair CCGs.
3. Finer Th_learned sweep if threshold instability still matters after claim-mask evaluation.

Final note

The interpretation has shifted from:

"units changing identity"

to:

"events being multiply represented across templates due to residual-pass detections, then permanently mis-partitioned because of CCG refractory violations"

This is directly grounded in KS4's implementation, explains both the unit proliferation and the failure to merge, and now has quantitative experimental support from the within-run CCG dose-response.




Note for companion figure. fig_peeling_dose_response.png
Caption
Figure X. Later peeling passes create a monotonic, dose-dependent burden of duplicate-like within-run unit pairs.
Reducing Kilosort4 max_peels from the default setting (100) to 3, 2, and 1 produced a monotonic collapse in the within-run near-zero-lag cross-correlogram signature expected from duplicate detections of the same event. Nearby unit pairs were defined as pairs within 100 µm in the raw pre-curation KS4 output. Duplicate-like pairs were identified by excess near-zero-lag coincidence, quantified as a near-zero-lag fraction of at least 0.05 together with a zero-lag peak-to-baseline ratio of at least 1.25. Under default settings, 236/273 nearby pairs met this criterion, compared with 53/192 for peel3, 21/131 for peel2, and 0/12 for peel1. This monotonic dose-response links later residual passes directly to refractory-violating duplicate detections. The decline in duplicate-like pair burden was accompanied by cleaner downstream sorting, with peel1 yielding the strongest overall practical operating point (22/22 KS4-good units; efficiency 0.273; median missing percentage 20.3%). peel2 showed a slightly higher efficiency (0.286) but retained substantial duplicate-like structure and only 4/21 KS4-good units, indicating that its apparent gain is partly inflated by residual near-duplicate partitioning. Together, these results support imperfect peeling as a major causal source of unit contamination in this dataset, while leaving open a smaller residual contribution from non-peeling mechanisms.

Panel Legend
(A) Empirical cumulative distributions of near-zero-lag fraction across all nearby within-run unit pairs for default, peel3, peel2, and peel1. Leftward compression of the distribution with decreasing max_peels indicates a progressive loss of duplicate-like pair structure. The dashed vertical line marks the near-zero-lag fraction threshold used in the duplicate screen.

(B) Summary dose-response across peeling conditions. Filled circles show the fraction of nearby pairs classified as duplicate-like; open circles show the median near-zero-lag fraction. Both metrics decline monotonically as max_peels is reduced.

(C) Relationship between duplicate burden and downstream sorting quality. Each point is one peeling condition, positioned by duplicate-like pair fraction and KS4-good unit count; point size encodes the efficiency metric (n_well / n_units). Lower duplicate burden is associated with cleaner downstream output, with peel1 providing the best overall balance.

(D) Representative fine-timescale cross-correlograms for nearby unit pairs from each peeling condition. The large near-zero-lag peak under default settings weakens in peel3, weakens further in peel2, and is absent in peel1, illustrating the raw signal-level collapse underlying the summary statistics in panels A and B.

Short Methods-Style Legend Addendum
Cross-correlograms were computed from spike times in the raw pre-curation KS4 sorter output using a ±5 ms window and 0.2 ms bins. Near-zero-lag fraction was defined as the proportion of pair counts falling within ±0.5 ms. Nearby pairs were restricted to units separated by at most 100 µm, with emphasis on pairs involving at least one non-good unit in the screening stage.