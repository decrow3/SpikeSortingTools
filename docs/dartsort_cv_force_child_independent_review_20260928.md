# CV/CW force-child independent review

## Interim verdict on preserved CV v1

CV v1 is a **qualified pass for its actual 148-family, two-largest-child
subset**, not for the frozen all-family/all-child scope. Its observed
8--29-sample event fractions, saved manifests, native/final namespaces, and
subset pair-ledger closure survive finite independent checks. Its headline
denominator and null-support wording are superseded pending CW v2.

| Window | All actual split families | CV v1 selected | Omitted | Omitted parent rows | Omitted second-child rows |
|---|---:|---:|---:|---:|---:|
| W2 | 91 | 88 | 3 | 212 | 39 |
| W3 | 62 | 60 | 2 | 4,298 | 152 |

The hidden selection was second-child rows >=20 and assigned share >=0.10.
CV v1's five omitted families have zero observed 8--29-sample top-two partner
fraction in every evaluable state, so adding them does not reverse the narrow
top-two conclusion. That does not repair the scope: timing and null profiles
still omit all child pairs involving children ranked third or lower.

## Independent checks passed

- All packet-manifest hashes pass.
- The saved selection is reproduced exactly from the actual force/no-force
  label arrays.
- Eighteen deterministic family/state observed-fraction cells (smallest,
  median, and largest selected family per window) reproduce exactly.
- Every selected family's complete list of actual no-force children is present
  in `FINAL_CHILD_EVENT_PROFILE.csv`, but only the two largest feed timing.
- The v1 null contains 200 unique signed offsets per window and 200 rows per
  selected family/state cell; both signs, unavailable cells, and support counts
  are retained. This validates serialization, not the now-superseded target-
  support definition.
- The saved close-pair NPZ, curve CSV, and closure CSV agree exactly for the
  selected-family domain under both actual-no-force and common-row clocks.
  The explicit remainder is 2,931/2,932 pairs in W2 and 2,405/2,411 in W3.
- Native constituent rows are explicitly distinguished from final-child
  events; 644 cross-child native-pair rows are present. Physical observed
  support never exceeds 182 channels, and the padded 206 columns are not
  treated as all observed.

## Why CW v2 is required

The frozen scope requires all force parents/families and every other child,
plus a common-target-support signed-shift null and whole-force adjacency
closure. CV v1 instead thresholds families, uses top two children by count,
and translates the same query roots while comparing observed and null at
different target coordinates. Its pair ledger also closes only over selected
parents. CW v2 corrects these without changing any scientific threshold or
rerunning a sort.

## CW v2 final independent result

CW v2 passes the bounded independent audit on the completed scientific
artifacts.  It covers every assigned all-force parent and every actual
no-force child: W2 has 552 parents, 91 multi-child families, 1,509
parent/child relations, and 9,049 all-child pairs; W3 has 488, 62, 1,076,
and 5,258 respectively.  The separate amplitude-maximum primary subsets
contain 957 W2 and 588 W3 pairs.

Whole-branch adjacency counts reproduce directly from the saved label/time
arrays.  For W2, all-force versus no-force 9--29-sample endpoints are
6,250 versus 2,903; for W3 they are 5,685 versus 3,051.  The saved complete
ledger agrees exactly, including 8- and 30-sample boundary counts.  The
common-target null contains all 200 signed offsets per window and its
offset-summary table explicitly retains zero-query cells.  The much larger
segment ledger is intentionally sparse (positive-support rows only), which
must be stated when that file is reused.

The producer stopped after writing the analysis receipt and did not emit its
promised final `COMPLETE.json`.  No compute process remains.  This is a
closeout defect, not a failed numerical check; the independent result above
is recorded in `CW_V2_INDEPENDENT_AUDIT.json` with status `pass`.

## Saved-output readiness for the next bounded outcome check

The completed imec1 rescue Kilosort 12/9 no-motion sort directly covers the
same AP-frame clock and exact W2/W3 intervals. W2 contains 1,411,980 saved
spikes and W3 1,194,681; saved event labels, positions, templates, geometry and
KS labels are present. It can serve as a sorter-derived control without raw
voltage or a new sort. See `dartsort_cw_saved_rescue_readiness_20260928.md` for
hashes, frame endpoints, and interpretation limits.

## Artifacts

- `testing/outputs/cv_force_child_independent_review_v1/CV_INDEPENDENT_AUDIT.json`
- `testing/outputs/cv_force_child_independent_review_v1/EXCLUDED_SPLIT_FAMILIES.csv`
- `testing/outputs/cv_force_child_independent_review_v1/EXCLUDED_FAMILY_COINCIDENCE.csv`
- `testing/outputs/cv_force_child_independent_review_v1/SELECTED_FAMILY_OBSERVED_RECHECKS.csv`
- `testing/outputs/cv_force_child_independent_review_v1/PAIR_LEDGER_INDEPENDENT_CHECKS.csv`
- `testing/outputs/cv_force_child_independent_review_v1/CW_V2_INDEPENDENT_AUDIT.json`
