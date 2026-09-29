# External-nonrigid field-regime screen — 2026-09-28

## Result

High-correction periods do not rescue the rejected full-session external-
nonrigid candidate. The frozen selective-follow-up outcome is
`selective_followup_not_supported`. Do not pursue a high-field-only application
of this unchanged field and operator.

This is a cached, sorter-free analysis of the completed 12/9 motion-off and
external-nonrigid sorts. It does not estimate motion, alter spike assignments,
or launch a sort.

## Frozen screen

The full 10,473.55 s session was divided into 174 complete 60 s bins. Exposure
was defined before reading spike outcomes as the mean spatial RMS of the exact
four-depth nonrigid displacement field applied to the candidate. The lowest and
highest field-RMS quartiles define 44 quiet and 44 high bins:

- quiet: field RMS <= 22.345 um;
- high: field RMS >= 28.098 um;
- observed full-bin field RMS range: 16.56--44.85 um;
- observed mean spatial spread range: 28.28--72.89 um.

All 65 reciprocal primary pairs that are interior in both sorts have at least
100 events per arm in each regime. Events were matched exclusively within 0.5
ms. The continuation gate required all of:

1. at least 20 supported pairs;
2. median high-minus-quiet retention of baseline events >= 0.05;
3. median high-field candidate-minus-baseline refractory fraction <= 0.01;
4. median paired high/quiet candidate-to-baseline event-count ratio >= 0.95.

## Measurements

| Paired median across 65 units | Value | Gate | Result |
|---|---:|---:|---|
| High-minus-quiet baseline-event retention | -0.00231 | >= +0.05 | fail |
| High candidate-minus-baseline refractory fraction | +0.00134 | <= +0.01 | pass |
| High/quiet candidate-to-baseline count ratio | 0.93219 | >= 0.95 | fail |

Median baseline-event retention is 0.802 in quiet bins and 0.795 in high bins.
Only 18/65 pairs improve baseline retention by at least five points, and their
population median remains flat. Median candidate-event retention rises from
0.725 to 0.763, but that uses candidate events as the denominator and coincides
with a lower paired candidate/reference count ratio. It therefore does not show
recovery of reference events. Only 30/65 pairs pass the count-ratio gate.

The field defines application intensity, not physical motion truth. Primary
spike-train correspondences are not verified biological identities, and the
full-session candidate already failed global population guardrails. This screen
only asks whether those losses spare the field's strongest applied-correction
periods. They do not.

Artifacts are under
`testing/outputs/luke_improved_nonrigid_regime_screen_v1/`; the request digest is
`db2ddbd5b255aed4e17e5dca4c76f2ec8b6d9dc061251e766452dcdf4d93f043`.
