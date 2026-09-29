# EM.2a SpikeInterface operator-screen result — 2026-09-29

**Verdict: EM.2a is complete.** Rounded-field SpikeInterface nearest and IDW
are exactly identical over all 49,139,454 compared values, so they are
redundant and should not both be sorted. Unrounded nearest, IDW, and kriging are
materially distinct. For EM.2b, retain the existing S0 and exact-DD controls,
use unrounded kriging as the primary standard comparator, retain unrounded IDW
and nearest as secondary and diagnostic comparators, and use rounded kriging as
the single field-rounding bridge. No sort or RF analysis ran, no outer holdout
was accessed, and no voltage was exported.

The final compact result is
`testing/outputs/em2a_operator_screen_v4_20260929/RESULT.json` (SHA-256
`6c4f16a7e63b49174c046eeb24f09d7ea1830a8b487d1b5e743323ecf6eeadb6`).
Its manifest SHA-256 is
`0bac48e6862b97d53a6d8f86534e85086a583e76c7c8601096c51ffa14170b3b`.
The run used Python 3.12.4 from
`/home/huklaban5/Documents/SpikeSortingTools/NPX_preprocessing/.venv/bin/python`
and SpikeInterface 0.104.7.

## What the screen establishes

All arms used the same three frozen EM.1 windows and the accepted 383-channel
parent. AP191 was reconstructed first with SpikeInterface's bad-channel
weights (`sigma=20 um`, `p=1.3`, threshold `0.005`; 13 nonzero weights; weight
hash `500aa938...`), then motion correction was applied and the output was
cropped to the 182 target sites. The operator settings are pinned in
`testing/em2a_operator_screen.py:59-106` and their effective kernels and hashes
are recorded in `KERNELS.json`.

| Candidate versus reference | RMS (uV) | Maximum (uV) | Interpretation |
|---|---:|---:|---|
| rounded nearest vs rounded IDW | 0 | 0 | exactly redundant |
| rounded nearest vs rounded kriging | `7.59e-7` | `2.10e-5` | only small float32/pseudoinverse numerical effects |
| unrounded nearest vs unrounded IDW | `0.45235` | `12.6055` | distinct operators |
| unrounded nearest vs unrounded kriging | `0.38212` | `11.5767` | distinct operators |
| unrounded IDW vs unrounded kriging | `0.14138` | `2.9181` | distinct operators |
| unrounded kriging vs rounded kriging | `0.40977` | `13.2650` | field quantization is material |

Rounded nearest and IDW differ from exact DD only at the AP191 hole: RMS
`0.02555` uV, maximum `7.59301` uV, and 127,498 values above `1e-6`. Rounded
kriging has the same AP191 difference plus its small numerical residual. This
separates the AP191 production-order effect from the broader unrounded-field
and kernel effects.

The 144 cached-event waveform screen used up to 16 deterministic, time-ordered
events in each window and each of three strata: the AP191 neighborhood,
interior, and top edge. Relative to exact DD, unrounded kriging had median
cosine `0.98669`, median absolute-peak ratio `0.89860`, and median RMS
`0.22001` uV. Unrounded IDW had `0.96122`, `0.81941`, and `0.35443` uV.
These measurements favor kriging as the primary standard comparator for the
next experiment, but they establish voltage behavior and nonredundancy only;
they do not establish sorting benefit, cell identity, or purity.

## Seams, unsupported sites, and conventions

The core and transition windows exercise the AP191 hole. Exact DD produced
127,498 structurally zero frame-channel values across seven target rows; the
production-order SI arms produced none there because AP191 was interpolated.
The frozen snippets did not contain a physical outer-border state, so the
runner separately audited every rounded W2 state and the unrounded W2 extrema
(`testing/em2a_operator_screen.py:454-521`). Each rounded operator zero-fills
four outer targets at `+40 um`; each unrounded operator zero-fills two outer
targets at the maximum W2 displacement, `+17.4244910542 um`.

The convention audit from EM.1 remains controlling and is cited in
`docs/EM-huklaban5-voltage-equivalence-result-20260928.md:58-85`: SI samples at
`location + displacement`, equivalent to DD's
`corrected = observed - displacement`; DD uses reference
`r = -4.69558824159 um` and 40-um half-away-from-zero rounding; time cells are
half-open with exact boundaries assigned to the later cell; motion is rigid
with one spatial bin; and processing order is preprocess, remap, then crop. The
EM.2a runner reasserts the field sign and exact saved-field interpolation before
reading cached events (`testing/em2a_operator_screen.py:395-429`) and implements
the same time-bin rule at `testing/em2a_operator_screen.py:129-144`.

The accepted input hashes were verified before voltage reads: unrounded source
field `85062a37...`, saved W2 D2L field `ee34e8a6...`, cached S_L peaks
`ca05b7a1...`, frozen selections `06ec3014...`, and parent provenance
`0ea45d85...`. The saved W2 field exactly equals interpolation of the source
field at every saved source time. The original DD limitation remains: its
actual-voltage gate failed 6/10 q0 and 9/10 remap-catalogue checks even though
the paired-input integrity check passed.

## Resources and preserved development history

The final run used 67.76 CPU seconds, 59.16 wall seconds, and 945,159,792
logical voltage bytes. The first development run failed after reading one
window (315,053,264 logical bytes) because waveform margins introduced field
states absent from the kernel cache; that failure is preserved without tuning.
Two complete development runs then used 945,159,792 bytes each: v2 exposed a
missing direct rounded-operator comparison, and v3 added it. Including the
final v4 run, development read 3,150,532,640 logical voltage bytes in total.
Every run was limited to the same frozen snippets and cached events.

Nineteen focused tests pass. The compact evidence contains only hashes,
metrics, event indices, zero-fill counts, and PNG/PDF waveform overlays; it
contains no voltage arrays. EM.2b still requires the exact local scorecard,
resource measurement, and separate authorization for every new sort.
