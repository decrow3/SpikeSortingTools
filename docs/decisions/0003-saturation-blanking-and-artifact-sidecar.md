# 0003 — Bilateral 500 µV blanking with a raw artifact sidecar

**Status:** Adopted, with a known and accepted cost

## Decision

1. Apply samplewise **bilateral** blanking at 500 µV before sorting.
2. Write every raw sample exceeding 500 µV to a separate artifact sidecar.
3. Exclude detections near sidecar claims from artifact-sensitive claims.
4. Leave Kilosort's own batch artifact threshold **disabled** — the sidecar
   replaces it, and the two together would double-handle the same events.

## Known cost, accepted deliberately

Point blanking creates local filter ringing and false peaks around saturation.
This is a real defect, not an oversight.

It is retained because removing it caused a **much larger** loss of
reviewed-neural recovery at the sorter output, in both harder test windows. The
ringing is the cheaper error. Anyone proposing to remove blanking must first
reproduce that comparison.

## Evidence

imec0 exact sidecar (validated 20-worker implementation, 2 h 55 min):

- 3,921,905 threshold points
- 397,839 claim-active samples
- AP191-only threshold events excluded

The sidecar was decisive in the imec0 duplicate-pair review: **every** spike from
all four questionable units (184, 191, 164, 165) falls within 0.5 ms of a sidecar
claim sample, so an outside-artifact counterfactual for them does not exist.
Their raw footprints were extreme — median maximum simultaneously-over-500 µV
channel counts of 15, 19, 269 and 75, all above the 92nd percentile of good
units.

## Interpretation limit

Artifact **proximity alone is not causal evidence**, because a unit may itself
contribute to the threshold crossings it sits near. The sidecar supports
classifying units as artifact-associated for sensitivity accounting; it does not
by itself authorize removing or merging them.

## Evidence pointers

- `docs/luke_20250804_imec0_rescue_result.md`
- `pipeline/artifacts.py`, `testing/luke_artifact_sidecar.py`
