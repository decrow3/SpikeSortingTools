# EM local actual-geometry mapping audit

**Verdict:** stock SpikeInterface 0.104.7 `nearest` with
`force_extrapolate=False` is not DD-equivalent on the actual W2 source and
target geometries. It matched every exactly supported target and correctly
zeroed the four outer-border targets, but substituted a nearest neighbor at
four targets whose requested source coordinate is excluded AP191. The exact
adapter preserves DD's zero-fill rule.

This was a geometry/kernel audit only. It read the SpikeGLX `.meta` file and
published DD knot table; it did not read voltage, launch a sort, or access the
outer holdout.

## Inputs

- Source: 383 AP channels, all raw AP sites except `imec1.ap#AP191`.
- Target: `imec1.ap#AP202` through `imec1.ap#AP383` (182 sites).
- Realized shifts: `{-240, -200, -160, -120, -80, -40, 0, 40}` µm.
- SpikeGLX metadata SHA-256:
  `89832b7afd62e96bce1e7441fc132d53c4da2eac812a88bde19effad9f1be23d`.
- DD W2 knot table SHA-256:
  `7176135d8c6a300d5465a24722a39f27c7707b67c5b3bb8fa3a6d07473d4a1d1`.
- Source geometry SHA-256:
  `c01d267823426cf3cd96609984105f9a5833890dec9801bbb15f5d14b54d8f71`.
- Target geometry SHA-256:
  `cb5a508f62f02ac6cd0774472e19cea99b56a3696ac6fc76c9610337437a8607`.

## Results

| Class | Count | Stock behavior |
|---|---:|---|
| Exact coordinate | 1,448 | All selected the DD source |
| AP191 interior hole | 4 | Incorrect nearest-neighbor substitution |
| Outer border | 4 | Correct zero-fill |

The AP191 mismatches were:

| `q` (µm) | Target `(x, y)` µm | Requested missing source |
|---:|---:|---:|
| -120 | (32, 2020) | (32, 1900), AP191 |
| -160 | (32, 2060) | (32, 1900), AP191 |
| -200 | (32, 2100) | (32, 1900), AP191 |
| -240 | (32, 2140) | (32, 1900), AP191 |

For each missing interior source, the unconstrained nearest distance was
25.612497 µm with a four-way tie. The installed kernel's `argmin` tie behavior
selected one recorded neighbor rather than zero. No exactly supported target
was incorrectly zeroed or remapped.

The valid/zero site counts by shift exactly reproduce the published DD
`SOURCE_SUPPORT.csv`: one zero for each of −240, −200, −160, and −120 µm; no
zeros for −80, −40, or 0 µm; and four zeros at +40 µm.

## SpikeInterface 0.104.7 source confirmation

The required 0.104.7 environment reproduced the four-mismatch result. Its
source shows:

- `get_spatial_interpolation_kernel` computes 2-D Euclidean distances and uses
  `np.argmin` for nearest, then zeroes only targets outside the coordinate-wise
  source bounding box (`preprocessing/preprocessing_tools.py`, lines 55–58 and
  96–107; SHA-256
  `a3e7e03c7580915d0b64317a623fe33f1d3fe3ec68968ccfbfa9af7491121fee`).
- `force_zeros` selects `force_extrapolate=False`
  (`sortingcomponents/motion/motion_interpolation.py`, lines 390–393).
- Frames are assigned with right-sided edge search and clipped to valid bins
  (lines 171–180); moved targets use `location + displacement` (lines 189–208).

The full 0.104.7 audit is under
`testing/outputs/em_lattice_mapping_audit_actual_si010407_20260928/`.

## Scope of the verdict

This establishes stock failure on the actual geometry under the required
SpikeInterface 0.104.7 implementation. It does not establish byte equivalence
of the exact adapter to DD's materialized voltage; that bounded huklaban5 check
remains pending.

Generated detailed evidence is under
`testing/outputs/em_lattice_mapping_audit_actual_v2_20260928/`; generated
`testing/outputs` content is ignored by default and should be force-added only
when intentionally curating the packet.
