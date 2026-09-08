# Luke lighthouse local controls and gentle-epoch comparison

Eleven of twenty candidates pass the separate quiet-period local matching control. Nine spatially spaced candidates were then followed through 4160–4260 s. Several footprints share a rise/drop/recovery pattern, while the highest-depth footprints change much less. This provides useful corroboration of coordinated waveform-footprint changes, but does not yet calibrate displacement or establish a single correct motion field.

## Findings

- Passing control IDs: 587, 554, 612, 445, 463, 317, 80, 154, 549, 510, 246.
- Tracked IDs after >=100 µm spacing: 80, 154, 246, 317, 445, 463, 510, 549, 587, spanning **220–3100 µm**. The earlier progress message's upper limit of 3180 µm described the passing pool; candidate 612 at 3180 µm was excluded by spacing.
- Candidates 666 and 667 have weighted template cosine 0.991 at the same 3540 µm depth. Both also resemble 665. This is unresolved identity redundancy, not proof of duplicate cells.
- Candidate 341 narrowly misses the preset 10% unassigned-event criterion (11.27%); 628 misses at 16.37%. No thresholds were relaxed after validation.
- Nine candidates give 84/90 usable 10 s bins (minimum ten accepted events). All six unusable bins also have fewer than ten events in the existing reference train; missing bins cannot be assigned a stationary position. This does not establish why the reference activity itself falls.
- Several candidates from 620 to 2380 µm share a rise near 4175–4185 s, a drop near 4195 s and subsequent recovery. This visible agreement is consistent with coordinated movement, but waveform changes or residual selection effects remain possible.
- Observed centroid ranges: 80 7.00 µm; 154 7.60; 246 7.20; 317 9.60; 445 7.24; 463 5.94; 510 1.76; 549 0.88; 587 0.38. These are **footprint-centroid ranges, not calibrated physical travel**. Different footprints can have different displacement sensitivity.
- Existing native rigid follows some shared temporal features. DREDGE/decentralized show larger excursions at shallow depths; at 2260/2380 µm they are nearly flat while the observed centroids change. The disagreements depend on depth and method, rather than supporting a blanket claim that all motion estimates fail.

## Methods and independence

Use current accepted conditioned imec0 reference and actual manifest gain, then 300–6000 Hz third-order forward/backward Butterworth with 50 ms margins and global median reference. No sort, external correction or motion estimator rerun.

Train median templates from 4080–4090 s for all reference clusters with >=10 events (up to 100 uniformly selected events). For each of the twenty approved candidates, score independent both-polarity extrema on channels within 20 µm of its peak. Matching uses fixed channels within 60 µm, template-energy/noise weights, and ±3 sample alignment. No spatial translations or field-guided positions enter selection. Nearby reference templates within 120 µm with cosine >=0.7 compete under the same anchor weights; require winner margin >=0.03. This screens available neighbors, not every possible unidentified source.

Calibrate minimum score on 4110–4130 s (grid 0.8–0.995, gain 0.4–2.5), requiring >=15 reference events, >=10 accepted events, recall >=50%, unassigned fraction <=10%. Freeze thresholds and repeat on separate 4130–4150 s data. Reference coincidence radius is 0.5 ms. Within-identity NMS is 0.8 ms; distant simultaneous events are allowed. These are consistency checks against provisional labels, not independent neuronal ground truth.

Track control passes in 4160–4260 s with ten-second bins. This interval was chosen before target voltage was inspected, from nearby native fields avoiding the subsequent large jumps. It is gentle according to native rigid, not unanimous estimator agreement. At least ten accepted events are required for a median multichannel waveform. Compute its energy centroid on fixed local support; bootstrap accepted events 100 times for a conditional 95% interval. Baseline is each series' first bin. The bootstrap holds the baseline estimate fixed and excludes baseline uncertainty and systematic tracking error.

Main comparison figure uses **10 s medians of each motion field**, matching the footprint time resolution. AP fields are interpolated in depth; native rigid is global. No sign, lag or scale optimization. The older dense-field comparison is supplementary; its point-sampled difference CSV is descriptive only and should not be used as an estimator ranking.

## Limits and next decision

Fixed local support and a frozen template can understate movement and preferentially retain events that still resemble the baseline. Passing quiet controls does not prove motion robustness. Sparse/spatially concentrated footprints can have very different centroid sensitivity. Before judging estimator magnitudes, calibrate that sensitivity and inspect the footprints through this already-read gentle pattern. Keep worst epochs deferred. No amplitude-completeness or motion-off efficacy conclusion follows.

## Figures and execution

Output: testing/outputs/luke_lighthouse_gentle_v1.

- 01_identity_controls PNG/PDF: held-out local consistency; zero-denominator fractions explicitly undefined. identity_validation_reviewed.csv is the presentation-safe table; the original execution table preserves sentinel failure values.
- 02_gentle_tracks PNG/PDF: all nine tracks alongside dense historical fields.
- 03_neighbor_similarity PNG/PDF: nearest training-template competitor.
- 04_matched_resolution PNG/PDF: six examples at common 10 s resolution, varying panel scales labeled.
- Full settings, calibration/validation/target detections, templates, source-reference count context, field medians and summary persisted.

Managed service luke-lighthouse-gentle-v1 completed with MainPID=0, ExecMainStatus=0, active/exited. Read 3,485,930,496 raw bytes including filter margins. Raw size and mtime unchanged. Launch command, log and final exit receipt saved in testing/outputs; full-sort hold preserved. Main comparison and corrected control figures visually inspected.
