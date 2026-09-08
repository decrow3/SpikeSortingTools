# Direct-detection early/late candidate review

Detected large peaks independently of sort labels at 960–970 and 9510–9520 s. Detection used both signs, locally exclusive radius 50 µm, and per-channel absolute threshold max(150 µV, 10 × original voltage MAD). Cohorts group detection channel and sign, with at least 15 events in each five-second half; no template-score filtering was used to make second-half events more consistent.

The two epochs produced 1,922 and 4,063 detections, respectively, and 12 eligible channel/sign cohorts. None passed every fixed original-voltage quality gate. This does not prove that the cohorts are noise or that no usable neurons exist.

## What the rejected-cohort review shows

All 12 cohort median waveforms and local footprints were exported and visually inspected. Several have clear, repeatable trough/recovery or positive-dominant waveforms. For example, the early channel-322 negative cohort is approximately 363 µV and 42.6 original noise units, with half-waveform cosine 0.992 and local energy fraction 0.930, but only 74% of unselected second-half events exceed the fixed 0.8 shape-cosine threshold.

Other repeatable positive-dominant cohorts fail the full-probe compactness fraction despite localized-looking local plots. Local plots do not show all far-field energy, so they cannot overturn that gate. Conversely, integrated full-probe energy can include residual background and common components; a failed fraction is not itself biological proof of non-neural origin. Full-probe footprint and background contribution need explicit review before accepting or rejecting those candidates.

Positive and negative channel-322 cohorts may describe different phases of overlapping spike populations; they must not be counted as two independent cells. Late channel-322 waveforms also show substantial differences between halves, consistent with mixed or changing event composition. Probe-edge channels remain unsuitable as reliable position guides.

The additional prereference common-coherence gate is provisional: common signal at event times does not alone prove the local neural component is artifact. No events were removed from any motion input on the basis of this candidate screen.

## Compensation preservation

For descriptive review, the frozen model was applied at the same events for all 12 rejected cohorts. All 24 median-waveform preservation comparisons passed the previously fixed cosine/amplitude-ratio gates. These are preservation observations on unqualified cohorts, not 24 certified neural checks. They do not fill the missing early/late lighthouse coverage or validate motion estimates there.

## Next step

Inspect full-probe coherent footprint versus background energy and individual-event identity consistency for the strongest repeatable cohorts. This should establish whether the screening metrics are rejecting useful local signals before seeking more candidates or relaxing thresholds. Preserve independent lighthouse qualification; do not promote a cohort because it agrees with DREDGE or survives compensation.

## Artifacts and execution

Script: `testing/luke_independent_transfer_candidates.py`. Default output `testing/outputs/luke_independent_transfer_candidates_v1/` preserves initial detection/screen results. `--review-all` produces `testing/outputs/luke_independent_transfer_review_v1/`, with all 12 cohort waveforms, event times, descriptive preservation table, and three PNG/PDF contact sheets. All three contact sheets were visually inspected.

Both independently managed services completed with zero exit status and terminal process state was checked. Launch commands, logs and receipts are persisted alongside output directories. No production configuration, spike sort or motion application was changed. The full goal remains active and incomplete.
