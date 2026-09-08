# Exact DREDGE match profiles and effective search bounds

Reconstructed the installed DREDGE correlation curves directly from cached original and compensated peaks for 4240–4260 s. For both inspected depth windows and both input arms, every reconstructed pairwise argmax and peak correlation reproduced the saved D/C matrices (correlation tolerance 1e-5). No parameters were tuned and no lighthouse measurements entered selection. Representative pairs were selected by largest absolute cross-half displacement at each inspected window.

## Shallow pair

At the 410 µm window, comparing 4246.5 and 4254.5 s:

- Original best internal displacement: −76 µm, correlation 0.5941. Best correlation within ±10 µm: 0.3387.
- Compensated best displacement: −77 µm, correlation 0.5877. Best within ±10 µm: 0.3311.

The correlation curve has a broad maximum around the large displacement. The profiles show substantial changes in the amplitude distribution, including much less shallow amplitude mass in the later bin. This supports a population-profile mismatch explanation, but does not distinguish changing neural activity from remaining non-neural events. It does not prove actual large movement. Internal D signs are not plotted physical-displacement signs.

The ±10 µm comparison is descriptive only. It is not a proposed motion cap, nor a value used to fit or select the estimator.

## Central isolated outlier and search-range discrepancy

At the compensated 2410 µm window, the largest pair (4243.5 versus 4255.5 s) favors +130 µm at correlation 0.5257. A competing maximum at −1 µm reaches 0.5088. These nearly competing peaks differ from the broad shallow failure.

Exact lag arrays produced by the installed implementation:

| Window center | Requested range | Effective range |
|---|---|---|
| 410 µm | ±80 µm | −160 to +80 µm |
| 2410 µm | ±80 µm | −80 to +160 µm |

In `spikeinterface/sortingcomponents/motion/dredge.py::xcorr_windows`, symmetric convolution padding compensates for a truncated search neighborhood at a histogram edge. Its `possible_displacement` array includes additional lags on the opposite side, and `calc_corr_decent_pair` takes the argmax over all of them. Gaussian windows have broad nonzero domains, so this affects more than the outermost window center.

This directly explains how a +130 µm constraint is permitted despite the requested 80 µm limit. It does not explain away the −77 µm shallow match, which lies inside the requested range. The next bounded implementation check should restrict argmax candidates to the already-requested range and replay cached inputs, preserving the existing outputs and avoiding any new biologically chosen motion limit.

## Next input diagnosis

The shallow profiles warrant examination of contributing event waveforms and their time-varying depth distribution. Large amplitude-mass changes should not automatically be labeled noise: population firing changes can also alter the registration objective. Establish morphology and shared-signal evidence before excluding events. Keep lighthouse measurements independent of any screening thresholds.

## Artifacts and scope

Script: `testing/luke_dredge_match_profiles.py`.
Outputs: `testing/outputs/luke_dredge_match_profiles_v1/`, including exact full pairwise curves for both inspected windows, smoothed population rasters, effective bounds, representative match CSV, and original/compensated PNG/PDF figures. `compensated_profiles.png` was visually reviewed. Script completed with exit code zero.

No library code, production settings, correction application, or sort was changed. The overall goal remains incomplete: full-recording stability, difficult-epoch corroboration and downstream sorting/amplitude-completeness benefit require further evidence.
