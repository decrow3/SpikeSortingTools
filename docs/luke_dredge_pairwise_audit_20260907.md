# DREDGE pairwise audit: shallow large shifts precede the final solve

Replayed original and compensated 4240–4260 s cached inputs with unchanged settings and `extra_outputs=True`. Both motion fields reproduced saved outputs within 1e-5 µm. The independently managed job completed with exit status zero.

At the 410 µm window, 45% of the 100 first-half/second-half pairs request an absolute displacement above 20 µm in both arms. They account for 41.1% of solver weight originally and 40.7% after compensation. Median correlation for these large matches is 0.643/0.639, well above the configured 0.1 cutoff. Pairwise shifts reach 76/77 µm.

The shallow matches also contain substantial inconsistency: the 90th percentile of absolute three-time cycle closure error is 62 µm in both arms (median 2 µm). Closure is computed as D(i,j)+D(j,k)-D(i,k) over all ordered triples, including repeated indices. A consistent displacement trajectory would close these cycles. This establishes that the solver receives mutually inconsistent large constraints; it does not establish which particular profile features are neural or artifact.

At 2410 µm, original constraints are all zero. After compensation, the median closure error remains zero and the 90th percentile is 2 µm. One cross-half pair has a 130 µm displacement with correlation 0.526, but carries only 0.70% of cross-half weight. The local implementation's search padding can be asymmetric near histogram boundaries; the saved requested `max_disp_um=80` is therefore not sufficient evidence that every returned displacement lies within ±80. This isolated case requires explicit search-domain review before production adoption.

## Interpretation and next action

The shallow +26–27 µm result is not solely introduced by final smoothing: large, moderately correlated and inconsistent pairwise matches are present upstream. Compensation fixes the demonstrated central stationary contamination but barely affects these shallow constraints. Raising a global correlation threshold without inspecting these matches would be premature.

Next inspect representative shallow pairwise correlation-versus-lag curves and the associated depth profiles to distinguish broad/background-driven matches, changing neural population composition, and discrete competing peaks. Also check effective lag bounds in the installed implementation. Do not tune against lighthouse displacements.

## Artifacts

`testing/luke_dredge_pairwise_audit.py` reproduces constraint extraction and figures. `testing/outputs/luke_dredge_pairwise_audit_v1/` contains both saved constraint archives, per-depth summaries, cycle-consistency summary and PNG/PDF figures. `compensated_constraints.png` was visually inspected. The displacement heatmaps use independent color scales, so compare numeric legends when inspecting shallow versus central panels.

No production correction or sort was launched. Full-recording model stability, difficult-epoch motion reliability and downstream sorting/amplitude-completeness benefit remain unverified.
