# v2 unvalidated merge candidates — preserved, not audited

Date: 2026-09-06. Status: **preserved for later. Not validated, not promoted,
not carried into any next candidate by default.**

Produced by the closed [v2 replay](luke_first_pipeline_candidate_v2_result.md)
on the Luke 2025-08-04 imec0 rescue sort. The cluster-37 replay branch is closed
and these are **not** being audited now; this file exists so the candidates
survive the branch closing.

## What they are

Groups of original clusters that the v2 linker joined into one family. Every
accepted link passed, on at least one adjacent-epoch pair: depth separation
within 30 um, amplitude ratio within 2.0, de-whitened waveform cosine at least
0.9, and a refractory-violation increase within 0.01 both on the pair and on the
exported train. All are labelled `unvalidated` in the export; none is labelled
`good`.

## What they are not

- **Not validated.** No downstream train has been checked, no QC has been run on
  them, and no endpoint has been evaluated for them.
- **Not evidence about cluster 37.** Cluster 37 is not among them.
- **Not a promotion path.** Combining them into a future candidate by default is
  explicitly out of scope. If they are ever taken up it must be as their own
  bounded evaluation with its own prospectively fixed endpoints.
- **Marginal by construction.** The accepted merges sit close to the cosine
  gate (median 0.914 across 133 accepted links in the case arm), which is where
  a threshold admits its weakest cases.

## The candidates

`arms` records whether the family formed in the case interval [6350, 7050] s,
the healthy interval [2880, 3480] s, or both. Forming in **both** means the same
clusters were independently joined ~3,500 s apart; that is a reproducibility
observation, not a correctness one.

Metrics are the worst across that family's accepted links: lowest cosine,
largest depth separation, largest amplitude ratio, largest refractory increase.
A negative increase means the merged train was *cleaner* than its worst
contributor alone.

| clusters | arms | min cosine | max depth sep (um) | max amp ratio | max refractory increase |
|---|---|---:|---:|---:|---:|
| 87 + 89 | both | 0.910 | 6.8 | 1.12 | +0.0005 |
| 90 + 92 | case | 0.981 | 2.2 | 1.02 | +0.0000 |
| 97 + 105 | healthy | 0.926 | 12.5 | 1.88 | +0.0018 |
| 109 + 110 | both | 0.904 | 12.1 | 1.61 | +0.0000 |
| 121 + 126 | case | 0.913 | 23.4 | 1.67 | +0.0019 |
| 128 + 134 | healthy | 0.928 | 0.7 | 1.59 | -0.0273 |
| 137 + 138 | case | 0.936 | 12.2 | 1.51 | +0.0023 |
| 137 + 138 + 141 | healthy | 0.901 | 14.8 | 1.80 | +0.0014 |
| 140 + 142 | both | 0.947 | 7.8 | 1.78 | -0.0126 |
| 158 + 160 | case | 0.911 | 1.5 | 1.90 | +0.0000 |
| 178 + 181 | case | 0.968 | 3.0 | 1.22 | +0.0036 |
| 182 + 186 | healthy | 0.908 | 7.5 | 1.81 | +0.0002 |
| 243 + 244 | case | 0.922 | 9.4 | 1.47 | +0.0000 |
| 287 + 298 | both | 0.931 | 16.4 | 1.08 | +0.0000 |
| 297 + 300 | both | 0.921 | 11.1 | 1.22 | +0.0070 |
| 303 + 311 | both | 0.908 | 9.6 | 1.77 | -0.0005 |
| 312 + 314 + 316 | healthy | 0.910 | 27.7 | 1.32 | +0.0075 |
| 314 + 316 | case | 0.929 | 6.0 | 1.39 | +0.0000 |
| 337 + 339 | both | 0.904 | 10.0 | 1.84 | +0.0000 |
| 343 + 344 | both | 0.902 | 3.2 | 1.34 | +0.0041 |
| 363 + 373 | both | 0.934 | 12.8 | 1.28 | +0.0000 |
| 383 + 386 | both | 0.909 | 2.8 | 1.17 | +0.0059 |
| 398 + 401 | both | 0.912 | 8.2 | 1.48 | +0.0001 |
| 551 + 556 | healthy | 0.938 | 4.6 | 1.80 | +0.0030 |
| 574 + 708 | healthy | 0.959 | 8.0 | 1.42 | +0.0000 |
| 666 + 667 | healthy | 0.959 | 3.9 | 1.68 | +0.0010 |

18 families formed in the case interval, 19 in the healthy
interval, 11 in both.

Full per-run detail, including every rejected candidate link and the gate that
refused it, is in the run's own `candidate_links.csv` and `unit_provenance.csv`
under `/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_first_pipeline_candidate_v2`.

## Related records

- [v2 result](luke_first_pipeline_candidate_v2_result.md)
- [v2 prespec](luke_first_pipeline_candidate_v2_prespec.md)
