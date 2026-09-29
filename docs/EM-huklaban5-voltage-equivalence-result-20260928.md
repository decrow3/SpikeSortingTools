# EM.1 huklaban5 voltage-equivalence result — 2026-09-28

**Verdict: EM.1 passes.** Arm A, the exact adapter, satisfies the requested
`1e-6` voltage tolerance, identical-zero-mask, and identical-source-map rules.
Arm B, stock SpikeInterface 0.104.7 `nearest` with `force_zeros`, differs at
exactly four mapping pairs, all caused by the AP191 interior hole. Arm C,
interpolate AP191 and then apply stock `nearest`, matches DD at every other
value and is usable as the production remap implementation for imec1. No sort
or RF analysis ran, no outer holdout was accessed, and no voltage was exported.

The compact result is
`testing/outputs/em_voltage_equivalence_three_arm_v1_20260928/RESULT.json`
(SHA-256 `049e9a6a3a41998b581f63373ef5c5f250d6858ba64356ac8c9a7d2897c5f599`).
The run used Python 3.12.4 from
`/home/huklaban5/Documents/SpikeSortingTools/NPX_preprocessing/.venv/bin/python`
and SpikeInterface 0.104.7.

## Results

The frozen metadata-only selection was made before voltage was read. It contains
three 89,999-frame windows: rest at 1118.625 s, episode core at 1031.625 s,
and a transition at 938.375 s. The episode core exercises all four AP191-hole
shifts; the transition independently exercises three of them.

| Arm | Verdict | Result |
|---|---|---|
| A: exact adapter | pass | Maximum error 0 at rest, `1.49e-8` in the core, and `2.98e-8` at the transition; all zero masks and all frame/source assignments equal DD |
| B: stock nearest, 383 sources | expected mismatch | 4 of 1,456 mapping pairs differ, and every voltage difference above `1e-6` is at one of those pairs |
| C: interpolate AP191, then stock nearest | pass | All 1,456 mapping decisions match DD semantics (1,452 source mappings and 4 outer zeros); differences from DD occur only where DD zero-fills AP191 |

Arm A's small residuals are the already diagnosed one-to-two-float32-ULP
SpikeInterface 0.104.7 versus DD-producer 0.104.8 preprocessing differences.
They are well below the explicit `1e-6` criterion. The earlier report treated
byte equality as mandatory and therefore called this a failure; the present
verdict applies the acceptance rule in the task without tuning or changing any
sample, map, or interval.

Arm B selected AP188 in each four-way 25.612497-um tie:

| q (um) | Target | Frames checked | Values > `1e-6` | Mean abs delta (uV) | Max abs delta (uV) |
|---:|---|---:|---:|---:|---:|
| -120 | AP203 | 30,000 | 29,908 | 0.7502 | 4.4210 |
| -160 | AP207 | 22,499 | 22,435 | 0.7775 | 5.1024 |
| -200 | AP211 | 59,999 | 59,811 | 0.7699 | 9.8159 |
| -240 | AP215 | 15,000 | 14,941 | 0.7636 | 4.2581 |

Arm C is applicable because the imec1 baseline independently labels AP191 dead
(`NPX_preprocessing/manifests/baselines/luke-bad-channels.json:1572-1575`).
It used the SpikeInterface bad-channel interpolation convention: inferred
20-um spacing, Gaussian/kriging channel weights, and `p=1.3`
(`spikeinterface/preprocessing/interpolate_bad_channels.py:32-40,48-66`). Its
AP191 values differ from DD's deliberate zeros by mean 0.3740–0.4062 uV and
maximum 3.3410–7.5930 uV. No non-AP191 value differs by more than `1e-6`.
The imec0 operational profile records bad-channel interpolation as preprocessing
step 3 (`configs/luke_operational_pipeline.v1.json:17-23`); imec1 needs its own
profile/receipt carrying the same order before this is deployed.

## Conventions checked

- **Sign.** The one-output-channel constant `+40 um` fixture selected the source
  at `target_y + 40`, as required. SpikeInterface moves target locations by
  `location + displacement` before building the kernel
  (`spikeinterface/sortingcomponents/motion/motion_interpolation.py:191-207`).
  DD records `corrected = observed - displacement` and maps
  `target(x,y) <- source(x,y+q)`
  (`dd_lattice_inputs.py:60-64,90-97,121-131`).
- **Reference and lattice rounding.** DD takes the reference median over the W2
  out-of-mask interval and applies 40-um half-away-from-zero rounding to
  `displacement - r` (`dd_lattice_inputs.py:52-54,69-76`). The frozen receipt
  gives `r = -4.695588241594664 um`, consistent with the requested rounded
  `-4.69558824159 um`.
- **Time bins.** DD assigns frames with
  `floor((time + 0.25/2) / 0.25)` (`dd_lattice_inputs.py:181-183`). The adapter
  uses half-open cells and right-sided search, so an exact boundary enters the
  later cell (`npx_preprocessing/motion/lattice_remap_si.py:319-360`). All
  269,997 frozen frame assignments equal DD.
- **Rigid motion.** SpikeInterface defines a one-column displacement and one
  spatial bin as rigid (`spikeinterface/core/motion.py:15-28`). The check builds
  exactly that object and asserts one spatial bin
  (`testing/em_huklaban5_voltage_equivalence_arms.py:328-339`).
- **Operation order.** The accepted 383-channel parent is loaded first, the
  remap is built from it, and the result is cropped to AP202:383
  (`dd_lattice_inputs.py:84-115,174-183`). Arm C inserts the AP191 interpolation
  before stock remapping; target rendering/cropping follows
  (`testing/em_huklaban5_voltage_equivalence_arms.py:378-418`).

## Identity and resource checks

Before any `get_traces` call, the runner verified all 15 h1 packet members and
the local adapter, the 383-source and 182-target geometry hashes
(`c01d2678...`, `cb5a508f...`), the knot hash (`7176135d...`), the accepted
parent provenance, the DD input receipt, and the frozen selections/assignments.
It confirmed the parent has 383 channels with AP191 absent. DD's complete
materialization receipts supplied the full 7.43-GB binary hashes; the runner
rechecked status and file sizes, confirmed the recorded q0 byte identity, and
then validated the sampled S_L bytes directly. Rehashing both full recordings
would itself exceed the 1-GB read cap.

The completed comparison consumed 26.60 CPU seconds, 28.65 wall seconds, and
748,077,816 logical voltage bytes (0.748 GB). Twenty focused adapter and runner
tests pass. The compact output contains only hashes, counts, and aggregate
errors.

EM.2 remains out of scope and requires the user's separate go-ahead.
