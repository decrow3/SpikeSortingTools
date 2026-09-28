# Luke0804 imec1 lighthouse motion comparison

Completed 13 September 2026. This is a waveform-only candidate analysis, not a
certification that every accepted event belongs to one neuron and not a motion
ground-truth declaration.

## Outcome

The bounded expansion produced enough nominal events to score the eight frozen
primary proposals, but the first premise check failed: simultaneous waveform
depth changes are not consistently shared across candidates. Fine MEDiCINe
slightly reduces movement RMSE relative to native AP and zero motion on the
five-family native-supported subset, while native AP has slightly better
movement MAE and fewer large-tail exceedances. All estimators are dominated by
implausibly large, candidate-specific residuals. The current evidence therefore
does not justify declaring either AP field the better motion estimate.

The existing fine-MEDiCINe sort recovers three more reference events than the
native sort at 0.5 ms (513 versus 510), but has lower mean conditional
concentration (0.593 versus 0.664) and more clusters per family (14.375 versus
12.875). Native has more output labels that receive multiple candidate families
(18 versus 13). These are distinct fragmentation and mixing tradeoffs, not a
motion verdict.

The full-crop DREDGE-LFP package was received through the shared NPX mount after
direct SSH authentication failed. Its independently verified tar SHA256 is
`707721bcbf181c9d11dd687eb9b1922d33c29123301a7b3f169dc23634dcebeb`, and every
payload entry passes the producer's `SHA256SUMS`. The two unsharpened estimates
were scored; the derivative estimates remain diagnostics because they have zero
supported samples inside the crop.

## Waveform-only expansion

Six 30-second windows were frozen without reading motion, sorter, stimulus, or
RF outcomes: 120–150, 320–350, 484–514, 696–726, 941–971, and 1120–1150 s in
AP-recording time. The managed systemd job completed successfully in 594.7 s.
It evaluated 1,007,499 detections and retained 716 strict primary events plus
4,008 uncertain primary events. Together with the 167 qualification events,
the comparison contains 883 strict events across all eight proposals.

The 5–95% extra-window depth spans were:

| Proposal | Strict events | 5–95% span (µm) |
|---|---:|---:|
| p06_f010 | 89 | 355.2 |
| p06_f045 | 65 | 74.3 |
| p08_f025 | 53 | 538.2 |
| p08_f029 | 131 | 239.9 |
| p08_f033 | 202 | 161.2 |
| p08_f044 | 18 | 637.3 |
| p09_f020 | 58 | 203.6 |
| p09_f024 | 100 | 241.8 |

Thus a 250-µm search is physically plausible for five proposals, but p08_f025
and p08_f044 exceed even the 450-µm allowance robustly. The full min/max spans
are larger still (622 and 701 µm respectively). These wide tracks were kept in
the plots and evidence tables rather than silently rejected.

## Shared-movement premise

Five candidates contribute adjacent 5-second increments on the native AP common
support. There are 24 distinct time-bin edges, 17 with at least two candidates
and nine with at least three. Pairwise correlations are inconsistent: examples
include p06_f010 versus p08_f044 at 0.84, p06_f010 versus p08_f025 at -0.46,
and p06_f010 versus p09_f024 at -0.53. Other pairs range from -0.18 to 0.53.
Large simultaneous changes can also point in opposite directions. This is not
the shared displacement expected from clean lighthouse cells and makes
identity/lookalike switching the leading limitation.

## Motion-field scores

All values below use family-balanced adjacent 5-second increments. Quiet means
an observed lighthouse change below 20 µm. Direct native-versus-MEDiCINe scores
use exactly the same 55 supported increments from five families (36 movement,
19 quiet; three quiet families satisfy the three-increment gate).

| Field | Movement RMSE | Movement MAE | Movement P95 | >120 µm | Quiet RMSE |
|---|---:|---:|---:|---:|---:|
| Zero | 217.3 | 169.7 | 424.4 | 61.1% | 6.8 |
| Native AP | 213.1 | 139.1 | 486.8 | 41.7% | 12.2 |
| Fine MEDiCINe | 209.5 | 143.8 | 453.9 | 44.4% | 9.1 |
| MEDiCINe rigid diagnostic | 209.7 | 142.4 | 433.6 | 44.4% | 13.4 |
| DREDGE-LFP 80 µm | 219.1 | 165.9 | 409.3 | 52.8% | 75.1 |
| DREDGE-LFP 260 µm | 251.9 | 188.5 | 511.2 | 55.6% | 175.7 |

Adding LFP common support leaves 460/883 events and 54 increments from the same
five native-supported families. LFP80 supports 833 events overall (94.3%) and
LFP260 supports 843 (95.5%). Their large quiet-period errors make both worse
than the AP fields in this frozen comparison.

On its larger zero-versus-MEDiCINe support, MEDiCINe has 104 increments from
all eight families and improves movement RMSE from 190.0 to 172.6 µm, while
raising quiet RMSE from 5.2 to 13.1 µm. This is useful directional evidence but
does not repair the failed shared-movement premise.

## Existing-sort concentration

At the primary 0.5-ms/100-µm one-to-one match:

| Existing sort | Matched / 883 | Mean recovery | Overall concentration | Conditional concentration | Clusters/family | Mixed output labels |
|---|---:|---:|---:|---:|---:|---:|
| Native AP | 510 | 0.615 | 0.417 | 0.664 | 12.875 | 18 |
| Fine MEDiCINe | 513 | 0.630 | 0.373 | 0.593 | 14.375 | 13 |

The 0.2-ms sensitivity is effectively unchanged (510 native, 512 MEDiCINe).
The largest adverse MEDiCINe fragmentation is p06_f010: conditional
concentration falls from 0.586 to 0.224 while clusters rise from 21 to 31.
P06_f045 remains highly concentrated in both arms. Output-label mixing and
within-family fragmentation must remain separate diagnostics.

## Reproducibility and files

- Frozen contract: `testing/imec1_lighthouse_motion_manifest_v1.json`
- Waveform extraction receipt:
  `testing/outputs/luke_imec1_lighthouse_extra_windows_v3_job/receipt.json`
- Event and field comparison:
  `testing/outputs/luke_imec1_lighthouse_motion_comparison_v7/`
- Existing-sort comparison:
  `testing/outputs/luke_imec1_lighthouse_sort_concentration_v5/`
- Tests: 10 passed, including archived imec0 regression values, sign/axis,
  no-clamping, LFP gap support, family weighting, one-to-one assignment, and
  label-renumbering invariance.

The next cheapest useful step is to inspect event-level waveform rivals for
p08_f025 and p08_f044 and require repeated shared movement across separated-depth
families before promoting these tracks to lighthouse ground truth. The RF
holdout remains unopened.
