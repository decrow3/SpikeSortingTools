# Detailed Kilosort–DARTsort snippet QC

Date: 2026-09-10  
Interval: Luke imec0, 930–1030 s, common AP26–AP373 depth range

## Conclusion

The detailed comparison strengthens three conclusions.

1. The unwarped, AP-rigid, and DARTsort-motion Kilosort arms recover broadly
   corresponding neurons. The DARTsort-motion Kilosort arm has the strongest
   family-union agreement with native DARTsort: median Jaccard 0.512, Kilosort
   coverage 0.683, and DARTsort coverage 0.817.
2. Native DARTsort is substantially more fragmented/selective. In the three
   credible Kilosort arms, 69–75 correspondence families contain one Kilosort
   unit and multiple DARTsort units. Unioning those fragments reduces, but does
   not eliminate, the DARTsort-versus-Kilosort rate deficit.
3. The two LFP-warped Kilosort outputs remain qualitatively different. They
   have low family Jaccard (~0.213), very large Kilosort depth excursions, and
   Kilosort-only median waveforms that are much less similar to shared-event
   waveforms in a deliberately selected arbitration sample.

None of this establishes biological ground truth. It does provide stronger
evidence that the LFP-warped Kilosort arms are pathological relative to both
native DARTsort and the three credible Kilosort arms, while the residual
Kilosort-versus-DARTsort disagreement in the credible arms is a mixture of
selectivity, fragmentation, and some waveform-inconsistent exclusive events.

## Identity and family agreement

| Kilosort arm | Strict pairs | KS-good pairs | Families | 1 KS : many DART families | Median KS coverage | Median DART coverage | Median Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| Unwarped | 105 | 28 | 141 | 69 | 0.640 | 0.765 | 0.480 |
| AP rigid | **109** | **29** | 139 | **75** | 0.612 | 0.777 | 0.442 |
| DARTsort-motion KS | 108 | **29** | 133 | **75** | **0.683** | **0.817** | **0.512** |
| LFP native | 39 | 14 | 59 | 9 | 0.302 | 0.494 | 0.213 |
| LFP SG25 | 39 | 19 | 59 | 10 | 0.398 | 0.435 | 0.215 |

The strict counts exactly reproduce the preceding frozen correspondence
analysis. Every strict assignment is one-to-one within an arm, family
identifiers are unique, and all coverage values passed range checks.

## Refractory evidence

Across all strict matches in the three credible arms, native DARTsort has a
lower observed fraction of ISIs below 1.5 ms:

| Arm | KS median | DART median | Paired median difference | Bootstrap 95% interval | DART lower |
|---|---:|---:|---:|---:|---:|
| Unwarped | 2.72% | 0.52% | -0.91 pp | [-1.92, -0.34] pp | 70.5% |
| AP rigid | 2.93% | 0.61% | -1.06 pp | [-1.81, -0.62] pp | 74.3% |
| DARTsort-motion KS | 2.79% | 0.61% | -1.16 pp | [-2.18, -0.57] pp | 72.2% |

This difference is not equally strong among the matched KS-good subset. For
the 28–29 KS-good pairs in each credible arm, both medians are approximately
zero and the bootstrap median-difference intervals include zero. The apparent
DARTsort refractory advantage is therefore concentrated in the broader
Kilosort population rather than clearly improving already-KS-good units.

The Hill/SpikeInterface contamination scalar is retained in the artifacts but
often reaches its 1.0 ceiling on dense 100 s family unions. The direct observed
ISI fraction is the primary refractory description here.

## Rate and fragmentation

For strict pairs, DARTsort reports substantially fewer events in the credible
arms. Median paired rate differences are -2.24 Hz unwarped, -5.07 Hz AP-rigid,
and -5.18 Hz for DARTsort-motion KS; all three bootstrap intervals exclude
zero. After unioning all qualified fragments in a family, the deficits shrink
to -0.80, -1.25, and -0.75 Hz, respectively, but remain below zero by the
bootstrap intervals.

This is direct evidence that fragmentation explains part, but not all, of the
rate asymmetry. The KS-good strict subset has much smaller median rate
differences whose intervals include zero in all three credible arms. As with
the refractory result, the largest disagreement lies outside the strongest
Kilosort-labeled cells.

DARTsort contains 1,020 units with at least 20 spikes in the common depth
range, but only 181 distinct DARTsort units participate in a strict match in
any arm. The eligible unmatched DARTsort units are smaller and spatially less
compact than matched ones: median 144 versus 210 spikes and median depth
excursion 69.8 versus 37.8 µm. Their median GMM maximum responsibility remains
near one, so that field does not separate the unmatched tail. These units are
unresolved yield, not evidence for hundreds of additional neurons.

## Spatial stability

The credible arms show different spatial relationships:

- Unwarped Kilosort is more spatially compact than DARTsort for strict pairs:
  paired depth excursion difference +9.61 µm for DARTsort, bootstrap interval
  [+5.09, +13.15] µm.
- AP-rigid and DARTsort-motion Kilosort show larger excursion than DARTsort:
  paired differences -15.55 and -10.95 µm, respectively. Ten-second binned
  median-depth ranges show the same direction.
- LFP-native and LFP-SG25 Kilosort units have median paired excursions of
  193–195 µm versus 27–28 µm for their DARTsort counterparts. Family union does
  not remove this: Kilosort remains at 206–211 µm versus DARTsort at 41–45 µm.

Because Kilosort and DARTsort use different localization machinery, a modest
excursion difference is not motion ground truth. The roughly 150 µm LFP gap,
its replication across matched units, and the low identity agreement are much
harder to explain as a small localization convention difference.

## Amplitude and temporal stability

Absolute Kilosort and DARTsort amplitudes were not compared. Kilosort stores a
template-feature coefficient while DARTsort supplies denoised PTP amplitude.

Within each sorter, exclusive events tend to be lower amplitude than shared
events. Median exclusive/shared ratios in the credible arms are 0.934–0.970
for Kilosort and 0.875–0.899 for DARTsort. The asymmetry is consistent with both
sorters disagreeing more often on marginal events, especially DARTsort-only
events. In the LFP arms the ratios are much closer to one, so their disagreement
is not explained mainly by faint threshold-edge spikes.

DARTsort has lower within-unit amplitude CV but generally higher variability
of the ten-second median amplitude and higher ten-second firing-rate CV in the
credible matched sets. A narrower overall amplitude distribution should
therefore not be called better longitudinal stability.

## Bounded raw-waveform arbitration

Five high-disagreement families containing at least one KS-good unit were
selected per arm. Up to 256 events per partition were read from the same
unwarped 348-channel voltage recording, over 61 samples and channels within
100 µm. Median partition waveforms were compared with the KS-shared median,
allowing only ±2 samples of alignment.

| Arm | KS-only cosine | DART-only cosine | Shared DART cosine |
|---|---:|---:|---:|
| Unwarped | 0.935 | 0.779 | 0.984 |
| AP rigid | 0.943 | 0.901 | 0.972 |
| DARTsort-motion KS | 0.857 | 0.925 | 0.970 |
| LFP native | 0.747 | 0.961 | 0.975 |
| LFP SG25 | 0.694 | 0.967 | 0.987 |

The high shared-DART cosines are an internal positive control for the timing,
channel, and coordinate alignment. The selected LFP families show a specific
asymmetry: DARTsort-only events preserve the shared waveform well, whereas
Kilosort-only events do not. In the selected unwarped families the direction
reverses. AP-rigid is relatively concordant on both sides, and the
DARTsort-motion Kilosort result is mixed.

This sample is intentionally enriched for disagreement and cannot estimate a
population prevalence. It is sufficient to reject the idea that every
exclusive event is simply the same waveform missed by the other sorter. It
also identifies concrete families for manual waveform/CCG review.

## Artifacts

Final cached detailed analysis:

- `testing/outputs/luke_kilosort_dartsort_detailed_qc_v5/summary.json`
- `unit_metrics.csv`
- `strict_matched_pair_qc.csv`
- `paired_metric_summary.csv`
- `paired_metric_summary_ks_good.csv`
- `correspondence_family_qc.csv`
- `family_metric_summary.csv`
- `event_partition_qc.csv`
- `event_partition_summary.csv`
- `validation.json`
- `01_detailed_qc_summary.png` and PDF

Final waveform arbitration:

- `testing/outputs/luke_kilosort_dartsort_waveform_arbitration_v2/summary.json`
- `waveform_arbitration_candidates.csv`
- `waveform_partition_qc.csv`
- `waveform_family_summary.csv`
- `01_waveform_arbitration.png` and PDF

Reproducible implementations:

- `testing/luke_kilosort_dartsort_detailed_qc.py`
- `testing/luke_kilosort_dartsort_waveform_arbitration.py`

Failed intermediate output directories were preserved rather than overwritten.
No spike sort was launched.
