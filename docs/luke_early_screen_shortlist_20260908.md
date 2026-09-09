# Brief screening transfer check:930–1030s

Compare five prespecified arms: compensated5σ baseline, center-energy only, relaxed combination, full screen without amplitude gate, and full screen. Reuse all188,514cached detections and locations exactly. Compute each20s chunk's waveform features once for all masks. All five motion fields, including baseline, use the same strict±80µm bounded helper; original cached field remains historical evidence. No redetection, relocalization, parameter sweep, low-pass change or sorting.

Independent service `luke-early-screen-shortlist-v1` runs core computation then cached-data report. Launch command/logs/receipts live in `testing/outputs/luke_early_screen_shortlist_v1_job/`. Launcher-disconnection survival verified. Partial files persist but this brief version refuses an existing output directory and has no within-stage or automatic resume.

The report will show paired population rasters using the requested0.25s×10µm/magma/99.5th-percentile display and event-matched curves for the two cached deep candidates over970–990s. Neither candidate is identity-qualified; there are no nine-cell references in this epoch. Disagreement is descriptive and cannot establish a physically accurate winner, especially if an arm simply flattens motion. Stop after this five-arm comparison.


Completed successfully in approximately130seconds including figures; both managed receipts and final service result exited0. Five cached-input reconstructions passed amplitude and array checks. At2920µm, all five arms have approximately9.7µm mean disagreement with the provisional20sfootprint observations. At3380µm, full screen and without-amplitude-gate give8.46/8.55µm versus11.16baseline,12.67center-only and11.84relaxed; the change is chiefly one977.5sbin. This is a local change in agreement, not an established physical accuracy gain. No overall winner or additional experiment is inferred.

[Reference overlay](../testing/outputs/luke_early_screen_report_v1/01_cached_reference_overlay.png) · [Depth/time figures](../testing/outputs/luke_early_screen_report_v1/02_depth_time.png). Black open diamonds in the overlay are the provisional fixed-support footprint observations; unsupported bins remain gaps.
