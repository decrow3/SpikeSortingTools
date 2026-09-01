# Static KS4-seeded TDC fidelity trace

## What this experiment actually tests

The static peeler is not “Kilosort4 with motion disabled.” It reuses accepted
KS4 unit IDs and seed events, re-estimates templates under a shared 300 Hz
high-pass plus global median reference, sparsifies them to 100 µm, and then
hands them to SpikeInterface 0.104.8's TriDesClous peeler. Static versus motion
therefore tests motion inside this TDC replay architecture. It does not isolate
a motion flag inside KS4's learned-template matching implementation.

The corrected one-to-one replay rate is 7.83% (69,480/887,910 eligible KS4
events), and the static peeler emits only 148,684 events. The basic event-count
deficit precedes label agreement.

## The dominant interface mismatch is peak polarity

TDC was run with its default `peak_sign="neg"`. In the shared re-estimated
template set, 707,990/887,910 eligible KS4 events (79.7%) belong to
positive-dominant templates. Static one-to-one replay is:

| Template polarity | Eligible KS4 events | Static replay |
|---|---:|---:|
| Negative-dominant | 179,920 | 29.54% |
| Positive-dominant | 707,990 | 2.31% |

This does not prove that every positive-dominant template is a neuron, but it
does prove that the benchmark handed a predominantly positive event population
to a negative-only first-pass detector.

## Gate trace

Four hundred accepted KS4 events were sampled deterministically: 200 with and
200 without one-to-one same-label static replay. Each event was traced through
the exact preprocessed static input and SI 0.104.8 first-pass detector. Fits are
isolated diagnostic fits and omit TDC's simultaneous-neighbor regression.

| Local gate metric | Missed controls | Replayed controls |
|---|---:|---:|
| Fast detector peak near event | 77.0% | 97.5% |
| Raw detector threshold in source-template channels | 22.5% | 83.5% |
| Correct template spatially eligible | 22.5% | 91.5% |
| Correct template selected as nearest | 0.5% | 76.5% |
| Median correct-template amplitude | 0.11× | 1.07× |
| Median isolated fit improvement | 2.3% | 26.8% |

The distinction between “some nearby peak” and “a peak in the source
template's neighborhood” is decisive. Most missed events have an unrelated
nearby detector peak but no 5-SD negative threshold crossing on the source
template's active channels.

The first observed gates among the 200 missed controls were 23.0% with no fast
peak, 54.5% with the correct template outside the detected peak's candidate
neighborhood, 22.0% losing nearest-template competition, and 0.5% below the
amplitude floor after the correct template was selected.

## Small counterfactuals

These are local gate traces on the same fixed events, not full peeler reruns.

| Trace configuration | Source threshold | Correct candidate | Correct nearest template |
|---|---:|---:|---:|
| Original: negative, threshold 5, radius 150 µm | 22.5% | 22.5% | 0.5% |
| Both polarities, threshold 5, radius 150 µm | 86.0% | 79.0% | 47.0% |
| Both polarities, threshold 4, radius 150 µm | **96.0%** | **84.5%** | **49.5%** |
| Both polarities, threshold 5, radius 250 µm | 86.0% | 84.5% | 42.5% |

Allowing both polarities fixes much of the threshold/candidate mismatch.
Lowering the threshold from 5 to 4 adds a smaller improvement. Widening the
candidate radius admits the correct template more often but worsens its chance
of winning nearest-template competition.

No tested local configuration approaches the required >90% correct-template
selection. The best counterfactual still loses 35% of missed controls to a
different nearest template, and another 16.5% passes the isolated local gates
while remaining absent from the original full peeler output.

SI 0.104.8's amplitude behavior is also asymmetric: fits below 0.7 are
rejected, while fits above 1.4 are retained with amplitude forced to 1 for a
subsequent peeling pass. The nominal upper bound is therefore not a rejection
gate in this implementation.

## Decision

The polarity mismatch is a real benchmark configuration error, but it is not a
complete explanation of the six-fold event deficit. Candidate construction,
nearest-template competition, short-window fitting, and full peeling context
remain substantial architectural differences from KS4.

Do not reinterpret the existing motion result as “motion-aware KS4.” It says
only that moving templates worsened the tested KS4-seeded TDC configuration.
Do not rerun the motion arms yet.

If preserving this branch is worth one more bounded execution, the only
justified run is a **static-only** TDC control with `peak_sign="both"` and
threshold 4. It must achieve >90% one-to-one any-event support and strong unit
correspondence before any motion arm is reconsidered. The local trace makes
that outcome unlikely; failure should close the branch and rename it a
KS4-seeded TDC architecture test.

Outputs are in `testing/outputs/luke_static_tdc_fidelity_trace*`. The primary
script is `testing/luke_static_tdc_fidelity_trace.py`.
