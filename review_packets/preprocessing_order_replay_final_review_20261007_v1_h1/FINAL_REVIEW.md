# Independent review: saved-stage preprocessing-order replay

Verdict: `GO_SCOPED_RESULT_ACCEPTED_TRADEOFF_NO_M3_ACTIVATION`.

The compact packet is internally complete and its load-bearing result independently recomputes. All 15 manifested files open and hash correctly. The 60 window rows, 48 pair rows and 40 switch rows have unique keys, finite required metrics and exact support-count closure.

The five views match the frozen design: saved `W_REF F(X)`, saved `W_A F(PX)`, `W_A S F(PX)`, `W_REF P F(X)`, and `P W_REF F(X)`. The executed source uses time-by-channel arrays, Kilosort's saved whitening orientation and target-to-source partial permutations without wraparound. The noncommuting fixture covers zero and both shift signs, changes mapping inside an interval, and distinguishes W/P order. The new post-W path is not the October 3 mask-before-W hook.

The repaired decision denominators close:

- Unsupported output is exactly zero in all 12 post-W windows.
- The five frozen full-window q0 identities are exact. The corrected code excludes `imec1_03`, whose q0-to-+40 transition invalidated the preserved predecessor.
- At the single frozen boundary in each canonical switch window, common support is 380 channels. Current-A to post-W jump RMS falls from 4.996486 to 1.347740 on imec0 (73.026%) and from 5.486496 to 1.381534 on imec1 (74.819%).
- All four stable-displaced event neighborhoods fail every intended representation guardrail: cosine 0.4792–0.6327 versus >=0.95, gain 0.1640–0.2763 versus 0.75–1.25, and normalized residual 0.7744–0.8777 versus <=0.35. Each denominator contains 2,013 common-support values.

The figures are consistent with the compact tables and use a shared per-window scale in whitened internal units. They do not relabel whitened amplitude as physical voltage or historical events as new detections.

The scoped TRADEOFF verdict is therefore defensible and conservative. The switch-local improvement is not enough to justify M3 because the frozen displaced representation guardrail fails in all four selected neighborhoods. Do not prepare or activate M3 from this candidate; return the scientific fork to HubPlanner.

Implementation checks

- Done: verified compact MANIFEST/COMPLETE and all 15 member hashes.
- Done: traced the executed source and frozen contract through P/S/W ordering, array and whitening orientation, mapping sign, support, q0 equality, switch denominators, stable-event denominators, figures and decision logic.
- Done: independently recomputed all four decision criteria and both switch reductions from the published CSVs; no use was made of H5's local review as independent evidence.
- Done: inspected both five-view figures and confirmed frame/unit labels, common per-window scales, state-boundary markers and visible view distinctions.
- Not done: independent opening or rehash of H5-local `VIEWS.h5`; the compact packet binds its reported SHA256 but does not republish the arrays.
- Not done: independent opening of the H5-local failed predecessor; its path and seal hashes are recorded, and the corrected denominator is independently visible in source and outputs.
- Can establish: a selected-window representation tradeoff and the frozen decision `TRADEOFF_NO_M3_ACTIVATION`.
- Cannot establish: biological loss, identity, purity, spike recovery, sorting harm/benefit, mechanism, causality, prevalence or generalization.
