# Nonrigid experimental release — 2026-09-09

The user explicitly authorizes starting the full-session nonrigid comparison now, using the same experimental authorization standard as rigid. No further scientific validation or approval is a prerequisite. Preserve `scientific_status: requires_review`; this experiment does not establish estimator accuracy.

Shared directory: `/mnt/NPX/Luke/20250804/shared_analysis/luke_improved_motion_two_machine_20260909_v1/`.

The producer has supplied `nonrigid_run_contract_authorized_v1.json`, bound to the unchanged field manifest, using the rigid arm's exact 346 channels and interpolation settings. Consume this as the frozen run contract. Accept its explicit development authorization instead of requiring scientific promotion of the estimator manifest.

The initial huklaban5 runner visible in its task history needs these implementation corrections before consuming this contract:

1. In `_corrected_recording`, construct Motion temporal bins with `field['time_s'] + source.sample_index_to_time(0)`. The source acquisition origin is 3057.677050340359 seconds; the field is recording-relative. Keep endpoint checks recording-relative.
2. Interpolate with full source geometry, then `select_channels(arm['output_channel_ids'])`, in the exact contract order. Automatic border removal alone does not implement the common interior domain.
3. Use checked int16 conversion: round nearest and fail on nonfinite/out-of-range values, consistent with the rigid arm. Reuse the rigid runner implementation where possible.
4. Preserve the existing GPU self-PID exclusion. Preserve full sample count, thresholds 12/9, and internal correction off.

These are implementation alignment fixes, not additional biological review gates. Reuse existing fixture evidence, with a focused check of nonzero acquisition origin and exact output channel order for the changed code.

On huklaban5 inspect the actual current runner first (history may be older). Replace only the waiting controller if required; preserve its receipts and do not interrupt an arm already correcting/sorting. Launch the corrected controller through the already-proven persistent systemd method, with a new job receipt, frozen source/settings, no automatic restart, and the same accepted source and full-session scope. Publish live transition status to the shared directory. No re-estimation is needed. Kilosort interruption requires restarting the entire sort after investigation.
