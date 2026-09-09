# Whole-probe waveform-only lighthouse test

User requested the930–1030s interval with absolute depth excluded from identity matching, retaining relative multichannel waveform shape.

Implementation: `testing/luke_waveform_only_global_v1.py`. Outputs: `testing/outputs/luke_waveform_only_global_v1/`. Independent service: `luke-waveform-only-global-v1`; persistent launch command, logs and exit receipts: `testing/outputs/luke_waveform_only_global_v1_job/`.

Cached first check compared247 complete seed templates after exact spatial recentering and temporal extremum alignment, allowing ±3 sample timing. Selected12 repeatable, localized seeds by global rival distinctness without depth strata. All247 templates remain rivals in the full recording search, including weak/nonselected seeds. The cached seed labels inherit prior sorting; this is not a wholly unsupervised discovery of cells.

Every full-support patch across the probe is eligible. Relative lateral and vertical geometry are exact under40µm translations; there is no prior favoring original depth, no restricted displacement range, no nearby-only competitor list, no DREDGE and no trajectory continuity constraint. A local strongest-channel detector chooses event patches; geometry is still used to assemble relative support. Identity scores are ordinary normalized multichannel waveform cosine. This coarse lattice can lose matches when sub-grid movements alter waveform sampling; incomplete edge patches are excluded.

All detected events retain best identity, runner-up, scores, gain and unmatched/ambiguous status. Per-chunk NPZs retain full identity score/gain/timing alternatives. Waveform centroids use the observed patch; physical displacement is revealed only after identity comparison. Provisional matches are not identity proof, particularly with whole-probe multiple comparisons and unknown/overlapping spikes. Look for multiple cells with supported movement and inspect waveform examples before claiming shared motion.

No sorting is launched. The extraction uses the previously dummy-tested independent systemd launcher; separate post-launch state inspection verified active/running MainPID2003565. Completed5s chunks are hash checked and reused. Interrupted chunks have no internal checkpoint and require evidence review before restart; unsealed output causes a hard stop. Source/settings changes require a new version. Report generation runs within the same independently managed process after all20 chunks complete.

## Training sensitivity check

Initial12 seeds recovered only20/693 cached seed spikes at the fixed0.86 cosine gate, so they cannot support a stationary-cell conclusion from dropout. Added a training-only qualification step (`testing/luke_waveform_only_seed_qualification_v1.py`): retain original waveform-only eligibility, require at least5 recovered cached seed events and20% recovery, then rank by global distinctness times square-root recovery, maximum20. This reads only930–940s chunks and does not use depth or held-out results. Five identities qualify:161,283,527,555,673. Original12 outputs remain controls. `testing/luke_waveform_only_global_review_v1.py` produces threshold sensitivity, qualified-cohort motion, same-patch event support and simultaneous spatially separated identity conflict audits after completion.

## Completion

The service finished successfully at2026-09-08T19:08:12Z (exit0); actual subsequent systemd state was inactive/dead, MainPID0. All20 chunks completed, totaling1,254,063 detections. Qualified five-cell cohort:519 held-out accepted matches,84 beyond±120µm. Repeated waveform groups around−443µm (161) and+159µm (555) deserve inspection, but multiple separated depth bands raise lookalike-identity concerns. No shared-motion conclusion is established. Recommended PDFs:04_training_qualified_motion.pdf and05_heldout_waveform_examples.pdf in the output directory. All initial selection outputs and uncertainty evidence remain preserved.
