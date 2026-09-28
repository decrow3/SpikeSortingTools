# CS H1 independent review scope

Status: frozen before H5 CS outcome review.

H1 will not duplicate the H5 stage replay. It will review the exact published
source and one saved ledger sample, then verify:

1. signed timing identity `L_post = L_pre - (u_b - u_a)` using the actual
   convention `t' = t - u`;
2. the real millisecond-to-sample radius conversion and inclusive boundary;
3. reference selection, score source, reassignment, stable sorting, dedup
   survivor/drop lineage, and preservation of unknown multi-event partners;
4. candidate-envelope completeness for every final 9--29-sample event;
5. separation of all-near-pair and consecutive-ISI denominators;
6. identical nonwrapping common support for the fixed +/-0.5, 1, 2 and 4 s
   state-segment nulls;
7. route/state denominators and the distinction between deterministic lag
   agreement and physical duplicate evidence;
8. exact cache compatibility before accepting waveform evidence; and
9. if an alignment-only counterfactual is published, unchanged grouping,
   original dedup radius, pre-outcome frozen offsets, event-level changes and
   score/assignment closure.

CR evidence will be integrated only after the duplicate ledger is understood:
eligibility and RF scores are not duplicate-aware neuronal-yield measures.

Limits: at most 900 additional all-work CPU seconds, two threads, one reader,
20 GB RAM, 500 MB final output, no GPU, raw-voltage read, sort, calibration,
prefix, matching, template construction, field fit, or production patch.
Prior cumulative charge is 16,253.22 seconds; the existing 19,300-second
ceiling remains sufficient.
