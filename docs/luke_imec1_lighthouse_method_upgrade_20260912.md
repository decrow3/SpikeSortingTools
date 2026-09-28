# Luke0804 imec1 lighthouse method upgrade, 2026-09-12

## Result

The imec0 lessons were applied to a new, versioned imec1 sorter-free discovery
pass. The run completed successfully and preserved the old v2 results as a
control. It reduced 192 raw-waveform proposals to three depth-blind
seed-qualified candidates. After absolute depth was revealed, only one,
`p06_f000`, retained a localized seed identity (P90 span 40.8 um).

This is a useful negative/contracting result. The current imec1 panel still
lacks the two independent localized identities required to test shared motion.
No motion estimator should be ranked from these tracks yet.

## Cached phase audit before extraction

The v2 bank constructed and compared families separately within detector phase.
Exact peak-centered relative-geometry comparison across phases showed:

- 11/50 selected v2 candidates had an invisible cross-phase rival with cosine
  at least 0.95;
- median selected cross-phase rival cosine was 0.911;
- at complete-link thresholds 0.90, 0.95, and 0.97, the 50 selected templates
  occupied 47, 48, and 50 groups;
- after depth reveal, only 2, 3, and 3 selected groups respectively had seed
  P90 depth span no greater than 120 um.

This audit used no sorter product, motion estimate, or absolute depth in
waveform grouping. Depth was used only for the frozen post-hoc plausibility
readout.

Artifacts:

- `testing/luke_imec1_sorterfree_phase_audit_v1.py`
- `testing/outputs/luke_imec1_sorterfree_phase_audit_v1/summary.json`
- `testing/outputs/luke_imec1_sorterfree_phase_audit_v1/global_rival_audit.csv`
- `testing/outputs/luke_imec1_sorterfree_phase_audit_v1/complete_link_group_audit_after_depth_reveal.csv`

## Revised v3 method

The v3 method keeps the v2 raw both-sign detector and seed morphology proposals,
but changes qualification and matching as follows:

1. All 192 real hypotheses and their two decoy forms compete globally across
   all detector phases.
2. Cross-phase comparisons use exact channel coordinates after recentering each
   footprint on its detected peak channel. No absolute probe depth enters.
3. A cross-phase hypothesis is unsupported unless both waveforms place at least
   50% of their energy on the exact common coordinates.
4. The imec0 physical gain gate, 0.35--3, is restored. V2 normalized shapes but
   did not calculate or gate gain.
5. A strict candidate must recover at least five and 20% of its own seed-family
   events after global real/decoy competition.
6. Complete-link groupings are retained as dependence audits rather than
   automatically merging templates. This avoids the single-link chains found
   in the imec0 Kilosort family experiments.
7. Candidate ranking and seed recovery remain depth blind. Absolute depth is
   revealed afterward, and a seed P90 span no greater than 120 um is required
   for the post-match plausibility flag.
8. Motion estimates and sorter outputs are not read.

## Candidate contraction

| Candidate | Depth-blind seed recovery | Global rival | Seed P90 span | Status |
| --- | ---: | ---: | ---: | --- |
| `p06_f045` | 175/236 (74.2%) | 0.832 | 190.3 um | Seed-qualified but spatially diffuse |
| `p08_f038` | 280/535 (52.3%) | 0.860 | 780.5 um | Seed-qualified but spatially implausible |
| `p06_f000` | 80/221 (36.2%) | 0.929 | 40.8 um | Sole post-match plausible candidate |

Useful negative controls remain visible. For example, `p09_f003` is localized
in the seed interval (17.0 um span) and recovers 147/486 seed members, but its
global rival cosine is 0.958, so it correctly fails identity distinctness.
`p06_f015` remains localized (44.4 um) and recovers 33/83 seed members, but it
fails the frozen waveform-shape gate and remains lower-confidence.

## Held-out support

Across the six pre-existing motion-rich windows, the global bank produced 503
strict selected-family events from 31 displayed families. That number is not a
31-cell result: 30 lack the full seed qualification plus localization contract.

For `p06_f000`:

- 76 strict events in all six held-out windows;
- 168 lower-score and 174 identity-ambiguous events;
- strict observations occupy phases 6, 8, and 9, showing that the revised
  matcher can follow alternate detector phases;
- held-out centroid span is 273.4 um, which is large but not automatically
  implausible given the known motion; its localized seed identity and eventwise
  alternatives must remain visible;
- strict rate is 1.46 events/s with 20% occupancy of 200 ms bins;
- strict plus lower-score rate is 4.69 events/s but occupies only 50% of 200 ms
  bins. Lower-score support cannot be promoted merely to achieve >4 Hz.

Decoys won 53,271/265,416 held-out detections (20.1%), higher than the v2
aggregate. This is expected after global competition and confirms that decoy
outcomes must stay in the qualification contract.

## Scientific decision

- Preserve `p06_f000` as one provisional lighthouse lead.
- Preserve `p06_f015`, `p09_f003`, and diffuse families as explicit controls,
  not independent lighthouse votes.
- Do not rerun the top-20 estimator comparison or infer >4 Hz agreement.
- The next candidate-finding step should use one or more separately frozen seed
  intervals to propose additional localized identities under this unchanged v3
  scoring contract. Treat those intervals as development if they have already
  been viewed. Do not lower gates to obtain a target count.
- Once a second localized, seed-qualified identity exists, first plot exact
  event-time replication between the two. Only after shared movement is visible
  should motion fields be refit at at least 5 Hz and sampled at the events.
- A sorter-based secondary route may reuse the imec0 premerge checkpoint, but
  it must emit singleton clusters as well as linked families.

## Execution and verification

- Implementation: `testing/luke_imec1_dots_sorterfree_waveform_discovery_v3.py`
- Persistent launcher: `testing/launch_luke_imec1_dots_sorterfree_waveform_v3.py`
- Results: `testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3/`
- Corrected cached review pages showing global-rival scores:
  `testing/outputs/luke_imec1_dots_sorterfree_waveform_discovery_v3_review/`
- Job evidence: `testing/outputs/luke_imec1_dots_sorterfree_waveform_v3_job/`
- Service: `luke-imec1-sorterfree-waveform-v3.service`
- Final state: inactive/dead, `MainPID=0`, `Result=success`, exit status 0
- Runtime: 2026-09-13 02:28:29--02:32:29 UTC
- Seven focused tests passed, covering exact cross-phase coordinates, lagged
  overlap, complete-link chain prevention, gain scoring/gating, and inherited
  v2 scoring behavior.

The source bundle, input contract, command, launcher-disconnection state, logs,
interval seals, final summary, and managed-job receipt are persisted. No
production sorter, correction, motion estimate, or concurrent full-probe job
was modified.
