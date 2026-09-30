# EN rounded-field Tier-1 result — 2026-09-30

**Verdict: the validated rounded-field arms are not production candidates for
this Kilosort pipeline. Keep the pinned motion-off profile.** Arms A and B
increase yield and some continuity proxies, but both worsen the frozen
short-interval contamination proxy during nonzero motion states. This fails the
predeclared requirement that a production candidate be at least as good as REF
on contamination. Tier 2 was therefore not started.

This is a profile decision under the frozen EN rule, not a claim that the
short-interval metric measures biological purity. Rate correlation and
time-only correspondence remain continuity proxies, and the correspondence
graph is highly ambiguous.

## Tier-1 panel

| Arm | Raw units | KS-good | Assigned spikes | Edge-unit fraction | Median primary F1 to REF |
|---|---:|---:|---:|---:|---:|
| REF | 731 | 318 | 30,483,402 | 6.21% | — |
| XR, unrounded rigid | 480 | 108 | 28,176,859 | 6.04% | 0.729 |
| A, rounded AM.3 | 841 | 352 | 31,716,724 | 7.70% | 0.764 |
| B, rounded XR | 812 | 332 | 31,052,960 | 5.39% | 0.760 |

Relative to REF, A has 15.0% more raw units, 10.7% more KS-good units, and
4.0% more assigned spikes. B has 11.1% more raw units, 4.4% more KS-good units,
and 1.9% more assigned spikes. XR is substantially worse on yield. Median
presence is 1.0 in every arm, and no exact duplicate timestamps were found.

Chance-aware coincidence excess is 0.155 for REF, 0.195 for XR, 0.149 for A,
and 0.145 for B. Thus the rounded arms do not fail every contamination-related
summary. The state-specific short-interval result is nevertheless adverse and
directly aligned to the motion states the intervention changes.

## Decisive contamination proxy

For Arm A's field, A has a higher 1–29 sample short-interval fraction than REF
at every nonzero negative state. The separation is large and the state-level
95% intervals do not overlap:

| State (µm) | A | REF |
|---:|---:|---:|
| -280 | 19.95% | 7.49% |
| -240 | 12.53% | 7.01% |
| -200 | 11.03% | 6.64% |
| -160 | 9.79% | 6.44% |
| -120 | 11.18% | 6.70% |
| -80 | 11.53% | 7.24% |
| -40 | 7.99% | 6.10% |

The nonzero-state pooled descriptive fraction is 10.59% for A versus 6.60%
for REF over the same 1,907.5 s of state exposure. The rare +40 µm state also
points adversely (15.48% versus 6.50%) but has wider uncertainty.

For Arm B's field, B is also worse than REF at the practically supported
nonzero states: 8.73% versus 7.47% at -80 µm, 6.84% versus 5.81% at -40 µm,
and 7.44% versus 4.92% at +40 µm. The nonzero-state pooled descriptive fraction
is 7.20% for B versus 5.81% for REF over 2,286.0 s. The -120 µm bin contains
only 0.25 s and is not used as a stable standalone result.

At q=0, A is slightly better than REF (2.62% versus 3.00%), while B and REF are
similar (3.04% versus 2.90%). This localization argues against a global counting
artifact and places the adverse result in the motion-affected states.

## Continuity and correspondence

A's state-rate correlation is higher than REF from -280 through -80 µm, but is
lower at -40 µm. B improves correlation at -120, -80, and slightly at -40 µm,
but is worse at +40 µm. These gains do not override the contamination guardrail.

Time-only primary matches to REF number 260 for A, 263 for B, and 102 for XR.
Median primary F1 favors A and B over XR, but almost every candidate and
reference unit has more than one eligible edge. Lost-reference fractions remain
64.4% for A, 64.0% for B, and 86.0% for XR. These measurements describe event
continuity and do not establish cell identity.

## Execution and validation

Arm A completed successfully under
`en-rounded-ks129-arm-a-20260929-v1.service`:

- 241,309,358,592-byte full-session recording;
- exact AM.3 field SHA-256
  `4c769125020aad3263bf6eaa1889cdaa7eccbf19dcf0aa3a068b185221fadb9f`;
- Kilosort 4.0.27, thresholds 12/9, CAR on, internal correction off, effective
  `nblocks=0`;
- 4,174.2 s materialization and 12,348.1 s sort wrapper time;
- 841 raw units, 352 KS-good units, and 31,716,724 assigned spikes.

The frozen Tier-1 service completed in 297.1 s with no voltage reads, new sorts,
RF access, or outer-holdout access. Independent validation confirmed:

- all 28 output products match `MANIFEST.json`;
- `COMPLETE.json` binds manifest SHA-256
  `f139c122a11eddf8457943d16e92b73a9319d43798abb8a77ddae3515171d13c`;
- spike, unique-unit, and KS-good counts reproduce from every arm's raw arrays;
- all 33 saved state-level short-interval fractions equal their saved numerator
  divided by denominator.

Authoritative result directory:
`/media/huklaban5/Data/en_rounded_field_ks129_20260929_v1/tier1_v1`.

Compact published packet:
`/mnt/NPX/Luke/DARTsort_motion_experiments/en_tier1_rounded_field_result_20260930_v1`,
manifest SHA-256
`cb885a0243891ba52f408218ec7551c4e4487554bfb7a3d39618895e1b4ffa9c`.
The packet contains the result, state tables, figures, manifests, and execution
receipts; it contains no voltage or sorter arrays. All 17 products and the
last-written completion receipt were verified after publication.
