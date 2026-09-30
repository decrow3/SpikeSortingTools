# EM.2g postvalidation event-order recovery

The full EM.2f run completed successfully at 09:22 PDT on 2026-09-29. The
first fresh postvalidation dispatch (`em2g-26f20315862e55ac.service`) stopped in
the initial cached-sort slicing step before any scorecard, event-overlap, or
whole-session result was computed. Its evidence is retained under
`testing/outputs/em2g_full_validation_job_v2_20260929` and
`/home/huklaban5/DARTsort_experiment_scratch/em2g_full_validation_20260929/v2`.

The failure exposed an incorrect adapter assumption: final DARTsort event rows
were expected to be globally time-sorted. The 21,822,563-row output contains
287,811 adjacent inversions, all bounded by at most two samples. DARTsort's
per-event time shifts can create this local ordering without breaking the
alignment of times, labels, and channels.

The repaired adapter applies one stable ordering by `times_samples` to every
aligned event array before half-open window slicing or time-dependent
whole-session metrics. It records whether normalization occurred, the inversion
count, and the largest backward step in each result. Equal-time rows retain
their original order. This is a mechanical representation repair; it does not
change event times, labels, channels, validation windows, endpoints, thresholds,
or decision rules. No RF or voltage data is read.

The corrected v3 run completed both W2/W3 scorecards and event-overlap analyses,
then stopped before the whole-session descriptive calculation because that
script's direct-file invocation did not add the project root to Python's import
path. The completed window results and failure logs remain intact. The import
bootstrap was repaired without changing any scientific calculation or contract;
the complete suite is rerun at v4 so it can produce one terminal receipt.
