# Luke full-probe rigid registration audit — September 7, 2026

**Classification: unsupported for application as a single whole-probe rigid
correction field. Expanding from 100 to 384 contacts did not eliminate the
strip's large, abrupt excursions.** This is a field-qualification decision,
not proof that motion correction is unhelpful or that every native shift is
wrong. The physical cause of the excursions remains unresolved.

The one authorized registration diagnostic and its CPU audit completed with
exit code 0. No downstream sort, nonrigid sweep, curation, or full-sort export
was run. The full-sort hold is unchanged. Work stops after this report.

## What changed with full-probe support

Both estimates cover the same 10,473.553728-second conditioned recording at
29,999.835983263598 Hz, in 5,237 approximately two-second batches. The source
recording and RESCUE 12/9 settings were held fixed. The new diagnostic uses all
384 contacts, 0–3820 µm, with effective `nblocks=1`; the earlier rigid run used
100 contacts at 1400–2380 µm. The native reference centers are 1909 and 1889 µm,
respectively. Only registration was computed for the new full-probe arm.

| Field diagnostic | Earlier strip | Full probe |
|---|---:|---:|
| Raw physical displacement range, µm | −309.5 to +315.5 | −234.5 to +65.0 |
| Raw physical displacement median, µm | −0.5 | +16.5 |
| Median-centered displacement range, µm | −309.0 to +316.0 | −251.0 to +48.5 |
| P5–P95 displacement span, µm | 138.5 | 170.0 |
| Median-centered RMS, µm | 45.684 | 53.512 |
| Median absolute adjacent step, µm | 2.5 | 2.0 |
| 95th percentile absolute adjacent step, µm | 146.125 | 153.125 |
| Maximum absolute adjacent step, µm | 307.0 | 242.0 |
| Adjacent transitions >100 µm | 560 / 5,236 (10.7%) | 691 / 5,236 (13.2%) |
| Batches with centered magnitude >100 µm | 384 / 5,237 (7.3%) | 571 / 5,237 (10.9%) |

The full-probe raw RMS is 50.700 µm. Its centered magnitude exceeds 50 µm in
629 batches; adjacent steps exceed 50 µm in 876 transitions. Full and strip
centered trajectories correlate at 0.705. Although the single largest step is
smaller, RMS, robust span, and the frequency of large steps are worse. The
>50/>100 µm screens are descriptive, not an automatic scientific pass/fail rule.

![Full-probe field and native raster](../testing/outputs/luke_full_probe_rigid_registration_v1/audit/full_probe_motion_audit.png)

The top panel retains the full excursions in both native fields. The second
compares the full-probe native field with independent estimates at its 1909 µm
center, using a separately scaled axis. The bottom is the complete native
pre-registration detection raster, displayed as log(1 + count). It supports
inspection of population activity, not neuron identity or biological recovery.

## Independent estimates disagree with a global rigid interpretation

Sign was fixed from the verified application rule: **physical displacement =
−native dshift**. Native batch centers use the actual sampling rate. The
independent acquisition-clock time bins subtract 3057.677050340359 seconds.
Comparisons interpolate linearly at matched time/depth on common support and
remove each series' median. No sign, lag, or scale was optimized; unsupported
endpoints are excluded. DREDGE supplies 5,237 centers; the others supply 5,236.

| Independent estimate at 1909 µm | Centered RMS, µm | Correlation with full native | RMS difference, µm |
|---|---:|---:|---:|
| DREDGE | 2.608 | 0.165 | 53.226 |
| Decentralized | 0.326 | 0.122 | 53.484 |
| KS motion sidecar | 3.334 | 0.261 | 52.768 |

At the old strip center, 1889 µm, the full-probe correlations are similarly low:
0.172, 0.128 and 0.255. The discrepancy is not resolved by matching the old
center rather than the new one.

Agreement varies materially with depth, so center-only evidence must not be
generalized into “no independent support anywhere.” Across the frozen
400:200:3400 µm grid plus the two centers, the highest correlations are 0.577
for DREDGE, 0.802 for decentralized, and 0.420 for the KS sidecar. In particular,
decentralized motion at 1000 µm correlates at 0.802, but its RMS is 23.355 µm
versus approximately 53.5 µm native; their RMS difference remains 37.122 µm.
Even the smallest full/independent RMS difference anywhere on this grid is
36.806 µm. A single native rigid field therefore does not agree consistently
in magnitude and timing across depth.

These estimates are not ground truth and share detection inputs. Preprocessing
and localization differ from the native estimator. The almost-flat
decentralized estimate near 1909 µm is especially weak as an independent
motion yardstick there. The conclusion rests on the combined magnitude,
temporal, spatial, and raster evidence—not three purportedly independent votes.

## Raster support: mixed evidence, retained explicitly

The native event table contains **11,494,386 pre-registration detections**.
All are represented in the saved batch/depth histogram; there are no zero-event
batches. Counts range from 572 to 4,814 per batch, with median 2,171. These are
drift-estimation events, not sorted spikes or neuron counts, and cannot be
compared directly with the accepted reference's curated spike total.

The frozen profile check compares adjacent batches on common interpolation
support in the 400–3420 µm interior, requiring at least 100 detections in each
specified profile and 100 supported depth bins. Applying the relative native
shift improves the median spatial-profile correlation from **0.647 to 0.658**
over all 5,236 pairs; 69.9% improve. Among the 876 >50 µm transitions, median
correlation improves from **0.331 to 0.428**, with 67.1% improving.

This is real support for the estimator's histogram-alignment objective, and
must not be suppressed to make the negative recommendation cleaner. However,
the motion estimate was fitted using these same events. Spatial profiles mix
neurons and firing-state changes; improved agreement can arise from aligning
activity redistribution rather than physical tissue translation. The statistic
is not an independent motion-validation endpoint.

![Largest native steps and raster support](../testing/outputs/luke_full_probe_rigid_registration_v1/audit/largest_step_raster_support.png)

The five largest adjacent steps are 235–242 µm. Their native rasters show
changes in band strength and localized structure, alongside bands that remain
roughly horizontal. Visual review does not establish a coherent whole-probe
translation of the reported size. This qualitative assessment is limited by
aggregate raster resolution; no individual-neuron or raw-waveform adjudication
was performed. A plausible concern is histogram alignment responding to
changing activity composition or depth-dependent motion. That mechanism is
an inference, not an identified root cause.

The actual native sampling coordinates place 1.081% of all batch/contact
coordinates outside 0–3820 µm; almost any nonzero global shift affects a probe
edge, so this is not itself a failure gate. With 200 µm edge exclusion, only
0.00455% lie outside support, across 31 batches. Edge extrapolation alone does
not explain the central disagreement.

## Decision and stopping point

The full-probe field remains **unsupported for promotion to voltage correction**:
full spatial support did not remove the earlier pathology, and the apparent
large translations lack consistent independent support across depth. The
native raster objective's improvement leaves physical causation unresolved;
this report does not assert that all movement is artifactual.

Keep the accepted **12/9 motion-off reference** for the next comparison. This
new diagnostic does not establish motion-off's biological superiority and does
not settle amplitude completeness. It also does not qualify a corrected sort.
If further work is requested, the unresolved question is why native
histogram alignment favors these excursions, and whether independent
waveform/depth evidence supports them. No additional diagnostic, parameter
sweep or full sort is launched here.

## Provenance, execution, and handoff

huklaban1 verified the full 241,309,358,592-byte recording checksum and released
bulk reads before this job started. The checksum is
`2d6fac755db3182841bf121d822754f963ebada3bcb8cd9e96e0b63e6f9b7372`.
The actual 384-contact geometry exactly matches the recording manifest. Input
conditioning is the accepted phase correction, bilateral blanking and bad-channel
interpolation graph, with no external filtering, referencing, or voltage warp.
Source size/mtime remained unchanged; source metadata and implementation hashes
were verified. There was no new 241 GB recording copy on either shared or local
storage.

Kilosort 4.0.27 ran in the pinned production environment under
`luke-full-probe-rigid-registration-v1.service`. The manager survived a separate
launcher-disconnection dummy test and retained exit status. Eight boundary/sign
and endpoint tests passed. Runtime guards excluded the full-run entry point,
post-drift detection, learned-template extraction, clustering and sort export.
The drift estimator's own universal-template extraction was retained as required
by the unchanged `templates_from_data=True` setting. Its returned binary handle
was closed before saving registration artifacts.

Registration took 5,264.6 seconds (87.7 minutes). Peak CUDA allocation was
4,336,389,632 bytes (4.04 GiB), with 7.46 GiB reserved. Process peak RSS was
100,628,824 KiB (96.0 GiB), including mapped recording pages; sampled anonymous
memory was much smaller than file-backed residency. The service used a 96 GiB
soft / 128 GiB hard memory limit. Registration and CPU audit stage receipts,
stdout/stderr, exact launch command and resource samples are preserved.
There were no retries. Interruption during registration would require restarting
preprocessing and estimation after investigation; completed registration outputs
could support a separate audit replay, not within-estimation checkpointing.

Local output root:
`testing/outputs/luke_full_probe_rigid_registration_v1/`.
Exact replay code:
`testing/luke_full_probe_registration.py`,
`testing/luke_full_probe_registration_audit.py`,
`testing/luke_registration_controller.py`, and
`testing/luke_registration_status.py`.

The compact shared handoff is published beneath the coordination directory as
`huklaban5_registration_v1/`, with inventory and completion receipt. It includes
the non-pickled field, metrics, figures, source-bound plan/hashes, report and
code. Ops, the native raster matrix, detailed launch/test receipts and logs remain
local after automatic approval review rejected the broader publication payload. The approximately 552 MB native detection table stays
local, with its path, size and hash documented. The shared status records the
terminal process state, classification, output paths, and release of bulk reads.
The full-sort hold SHA-256 remains
`78c187c00ea54eb18e981a2aa0a4f866f525c17170011982accae7ab0be515dc`.
