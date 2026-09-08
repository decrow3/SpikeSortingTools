# Bounded overnight shallow-reference experiment

The experiment tests whether a spatial weighting change can recover useful shallow references without relaxing identity gates. Unit80 previously failed injection sensitivity; nearby distinctive candidates may supply evidence near the disputed 410 µm region. This is reference qualification and bounded tracking, not an estimator sweep or a full-session scan.

## Frozen comparison and gates

Reuse the 4080–4100 s training banks, including all available nearby labeled cohorts and independently detected unknown waveform families. Consider unit80 and at most one candidate from each depth band 300–380, 380–440, and 440–520 µm. Additional candidates are selected solely from training-half repeatability and peak/noise strength, with half-template cosine ≥0.9. At most four units proceed to qualification; unavailable geometric hypotheses produce a recorded failure.

Compare the original union-of-shifts weights with **shift-conditioned common target weights** on exactly the same development events and injections. Under each spatial context, every rival identity/shift uses the same weights as the target. The highest target score wins before the unchanged acceptance rules are applied. Comparing normalized scores across contexts is a method change, not an established invariance; its qualification controls are essential.

The 4110–4120 s interval is a **reused development qualification interval**, previously included in gentle-matcher calibration. It differs from the recent v3 test interval but is not independent validation of the overall method search. No threshold is tuned on it. Match gates remain cosine ≥0.8, gain 0.4–2.5, identity margin ≥0.03 and unique-shift margin ≥0.03. Qualification requires ≥15 reference events, independent-detection recall ≥50%, ≤10% unassigned accepted events, ≥90% injected identity recovery, ≥90% unique correct-shift recovery, and ≤1% worst-rival false-target rate.

Inject independent-half target templates at five exact geometric shifts and gains 0.75/1/1.25 into real development backgrounds. All labeled rivals **and unknown waveform families** receive corresponding controls. Unknown families are conservative stress controls, not certified distinct cells; failing against a duplicated or mixed family does not establish biological identity failure. Known-center injection recovery is explicitly separate from independent detector recall.

Only contextually qualified units enter transition tracking. Process 4240–4260 s first, then 4180–4200, 4160–4180, 4200–4220 and 4220–4240 s. Each unit/chunk is a separate checkpoint. Matching uses no DREDGE fields. Five-second location summaries require ≥10 events in the dominant unique-shift population, ≥80% dominance and no boundary-shift events; sparse, mixed or edge bins remain gaps. Save discrete hypotheses, observed median waveforms and conditional centroid measurement intervals. The intervals use 200 event bootstrap resamples plus an interleaved-half repeatability check. They do not establish physical accuracy or identity certainty.

## Independent execution and checkpoint semantics

Run through the already verified independent job manager:

```text
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
MPLCONFIGDIR=<persisted writable cache directory>
environments/rescue-production/.venv/bin/python -m testing.luke_shallow_reference_overnight_v1
```

Optional `--output <absolute-directory>` selects a separately versioned run. Default output is `testing/outputs/luke_shallow_reference_overnight_v1/`. The launcher must persist its command, environment, job ID, stdout/stderr and exit status outside chat.

Stages are preparation, quiet voltage caching, one qualification stage per unit, one voltage-cache stage per 20 s chunk, one tracking stage per qualified unit/chunk, and a final report. A process-wide exclusive `flock` prevents simultaneous restart processes. Stage work goes to a `.partial` directory; after output hashing and a receipt write, an atomic directory rename commits the complete stage. Restart validates exact output filenames and hashes and reuses only completed stages. Interrupted partial directories are renamed and preserved before restarting the entire incomplete stage. **There is no within-stage score-matrix checkpoint.**

Settings bind source-code hashes, package/Python versions, source bank and label/time hashes, recording metadata, raw file identity/stat, and hashes of the exact bounded raw read windows. Completed stages with missing/corrupt outputs are refused rather than silently rebuilt. Source/input and bounded raw hashes are checked again before the final report. Changes require investigation and a new run version. No failed evidence is overwritten.

Small validation passed: both weighting modes on synthetic templates, empty detection handling, completed-stage reuse, settings mismatch and corruption rejection, and an actual subprocess SIGKILL followed by launcher restart. The killed process released its lock; the completed stage was reused and partial evidence preserved before rerunning the interrupted stage. Test evidence directories: `/tmp/shallow-kill-test-repjbeo1` and `/tmp/shallow-receipt-test-wcjsgic5`.

## Resources and handoff

Use one numerical worker with all numerical thread counts set to one. Existing reconstruction creates temporary full-probe arrays; allow approximately 6–8 GB RAM. Bounded voltage caches occupy approximately 5.1 GB, plus score matrices and figures; reserve 8 GB disk. Runtime is not yet benchmarked for the contextual matcher; budget roughly 15–90 minutes, depending on detections and how many units pass. A no-pass outcome ends after qualification and is a useful completed result.

Each qualification stage persists both methods' decisions, full score matrices, rival counts/rates, outcomes and a comparison figure. Tracking stages persist all scores/events, accepted frames, per-bin median templates, support/uncertainty tables and figures. A separate postprocessor may compare retained DREDGE against saved accepted frames only after the tracking stages commit; such comparison must not change matching or promote failed references. The core experiment itself reads no motion field.
