# Fixed spatial-scale decomposition does not solve the shallow mismatch

Using unchanged compensated peak rasters from 4240–4260 s, decomposed each depth profile into a Gaussian-smoothed broad component (sigma 40 µm) and signed fine remainder. The scale was fixed before computation, corresponding to two 20 µm contact rows; no scale sweep or lighthouse-based tuning was performed. All correlation searches were restricted to the requested ±80 µm.

At the 410 µm window, the representative original profile pair prefers −77 µm at correlation 0.588. The broad component prefers the −80 µm search edge at 0.891. Removing the broad component reduces the representative maximum to 0.206 but still prefers −77 µm. Large cross-half matches decrease from 45% to 35%, while 90th-percentile cycle inconsistency remains 61 µm (original 62 µm).

At 2410 µm, the strict-bound original profile prefers −1 µm at 0.509 and the fine remainder prefers −2 µm at 0.413. Its cycle inconsistency stays small (90th percentile 2 µm). The broad component alone instead prefers the +80 µm boundary at 0.873.

Broad spatial structure can yield high correlations at large, misleading lags. However, subtracting it does not restore reliable shallow registration: the fine features themselves still give inconsistent matches. This diagnostic is not a production preprocessing change. Its signed raster also changes the interpretation of subsequent correlation weights, so no full DREDGE solution was inferred from it.

Next work should retain uncertainty about shallow registration, investigate identity-preserving evidence across more time, and assess shared-model temporal transfer before broad motion application. Do not remove plausible neural events merely because their population profile changes.

Reproducible script: `testing/luke_profile_scale_diagnostic.py`. Outputs: `testing/outputs/luke_profile_scale_diagnostic_v1/`, including fixed settings, per-representation D/C arrays, summary CSV and PNG/PDF figure. Execution completed successfully and the exported figure was visually inspected. No library or production configuration was changed and no sort was launched.
