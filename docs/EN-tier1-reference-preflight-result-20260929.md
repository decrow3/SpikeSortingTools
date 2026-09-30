# EN Tier-1 full-scale reference preflight

**Verdict: pass.** The frozen Tier-1 implementation completed the full REF arm
on the rounded XR-field state axis in 47.97 s with 3,760,244 KiB peak RSS. It
read no voltage, launched no sort, and did not access RF data. This establishes
that the eventual four-arm panel is a minutes-scale saved-output analysis.

The input identity was the pinned REF sort
`22ded4d503b6de8edf4851a08797ae4e594fe41118b913451365727ffbd616ac`.
The field hash was
`547ff39d1c91d819b758246c701603d868bfd4d50bd6b791e3d5b7140c01c71d`,
with median reference 14.522145247697315 µm and rounded states −120, −80,
−40, 0, and 40 µm. All raw sort files were independently hashed before use.

## Reference measurements

- raw units: 731; KS-good units: 318;
- assigned spikes: 30,483,402;
- exact duplicate samples: 0;
- median presence ratio: 1.0 (95% unit-bootstrap CI 1.0–1.0);
- P10 presence ratio: 0.7429 (95% CI 0.6857–0.7714);
- processing-domain units: 644;
- edge-unit fraction: 0.0621 (95% CI 0.0435–0.0807);
- chance-aware marked-spike excess: 0.1554 (95% common-block CI
  0.1463–0.1644).

These are baseline implementation and scale checks. They do not rank a motion
arm, establish biological identity, or trigger Tier 2. The complete local
packet is `testing/outputs/en_tier1_reference_preflight_v1`; its manifest
SHA-256 is
`e3b3825e182df72fa38f93154dcec4efc5d798e4602c974023e605daf65c86d6`.
