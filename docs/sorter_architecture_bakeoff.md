# Sorter architecture bake-off

This comparison asks whether drift-aware neuron or template tracking improves
the accepted unwarped Luke recording. It is not another voltage-registration
sweep.

## Invariants

- Every candidate starts from the accepted rescue recording manifest.
- A bounded comparison uses one fingerprinted interval relative to that
  materialized recording. Receipts also retain original-source frame bounds.
- Spatial resampling of voltage for motion correction is forbidden.
- Sorter-native filtering, referencing, standardization, and whitening are
  allowed but must be recorded. This is a comparison of complete sorting
  architectures, not an assertion that their frontends are identical.
- A challenger cannot advance on unit count alone. Reviewed-event recovery,
  refractory violations, and near-coincident duplicate burden are guardrails
  before longitudinal continuity is interpreted.

## Candidates

### KS4 without motion

The existing accepted KS4 sort is the production reference. For a bounded
window, the bake-off extracts spikes in the half-open interval and rebases
their sample indices to the start of the window. This is explicitly recorded
as `accepted_full_sort_window_extraction`: it is not misrepresented as a fresh
KS4 run on a short recording.

### DARTsort native motion tracking

DARTsort receives the same accepted unwarped recording, runs its explicitly
recorded native frontend, estimates motion, and uses its native drift-aware
sorting pipeline. The repository describes DARTsort as work in progress and
states that it is not currently recommended for production spike sorting.
Accordingly, every DARTsort receipt is marked `experimental=True`; it is a
challenger, not a replacement for the KS4 reference.

The initial pinned challenger version is `dartsort==0.5.16`. Its native output
must contain `dartsort_sorting.npz` with `times_samples`, `channels`, and
`labels`, and the runner atomically accepts the result only after validating
those arrays.

### KIASORT through its upstream SpikeInterface wrapper

KIASORT's own repository includes `SpikeInterface_wrapper/` with a Python
`run_kiasort()` adapter around its MATLAB no-GUI entrypoint. This is separate
from SpikeInterface's built-in sorter registry. Set `KIASORT_PATH` to a pinned
KIASORT checkout (or pass `--kiasort-path`). The runner fingerprints both the
upstream Python wrapper and `run_kiasort_nogui.m`, invokes MATLAB, normalizes the
returned sorting to `spike_times.npy` and `spike_labels.npy`, and atomically
records the result.

KIASORT performs its own frontend processing. That frontend is part of the
candidate architecture and must not be described as identical to KS4's
frontend.

Machine audit on 2026-08-31: MATLAB R2022b is installed, and the Signal
Processing, Statistics and Machine Learning, Parallel Computing, Image
Processing, and Curve Fitting toolbox license checks all pass. The complete
KIASORT repository is installed at `/home/huklab/Documents/KIASORT` and pinned
to commit `4caecd25132da06485b6b2dde267fe1ecac5b895`; hashes are recorded in
`third_party/kiasort_pin.json`. Both MATLAB entrypoints and the Python wrapper
are required by the installation audit.

To test KIASORT graphically, open desktop MATLAB and run:

```matlab
addpath(genpath('/home/huklab/Documents/KIASORT'));
kiaSort
```

The graphical entrypoint resolves to the pinned checkout's `kiaSort.m`. The
pipeline separately uses the root `run_kiasort_nogui.m` through KIASORT's
upstream SpikeInterface wrapper. Current upstream also contains a mirrored
copy under `No_GUI/`; the adapter deliberately selects the root copy and
fingerprints it to avoid MATLAB path-order ambiguity.

### SpikeInterface motion-aware peeler prototype

The production environment remains pinned to SpikeInterface 0.102.1. The
isolated challenger environment pins 0.104.8, where the TDC peeler exposes
`motion_aware`, `motion`, `interpolation_time_bin_size_s`, and `motion_step_um`.
The adapter converts a `qualified-motion-field-v1` artifact into a
SpikeInterface `Motion` object only when every field bin passes the requested
support and confidence gates. It refuses to invent or extrapolate unsupported
motion.

This component is not yet an end-to-end sorter condition: template creation,
clustering, matching, and normalized output still need one bounded prototype
before it can become runnable in the bake-off.

## Isolated environment

Create a separate environment; do not upgrade the accepted production
environment in place:

```bash
conda env create -f environment-challengers.yml
conda activate spike-sort-challengers
```

The environment pins SpikeInterface 0.104.8, DARTsort 0.5.16, and the
CUDA-12.4 build of Torch 2.6.0. This avoids pip's current CUDA-13 default,
which cannot initialize against this host's NVIDIA 550 driver. On 2026-08-31,
the host reports an RTX A5000 with 24,564 MiB total memory; a successful Torch
device-allocation smoke test is still required before a DARTsort data run.

## Commands

```bash
python sorter_bakeoff.py --rescue-output-dir /path/to/results --plan
python sorter_bakeoff.py --rescue-output-dir /path/to/results --plan \
  --window-name rapid_motion --start-s 5910 --duration-s 120
python sorter_bakeoff.py --rescue-output-dir /path/to/results \
  --run ks4_no_motion --window-name rapid_motion --start-s 5910 --duration-s 120
python sorter_bakeoff.py --rescue-output-dir /path/to/results \
  --run dartsort_native --window-name rapid_motion --start-s 5910 --duration-s 120
python sorter_bakeoff.py --rescue-output-dir /path/to/results --run kiasort \
  --kiasort-path /home/huklab/Documents/KIASORT \
  --window-name rapid_motion --start-s 5910 --duration-s 120
```

Window outputs live under
`sorter_bakeoff/windows/<name>-<window-digest>/<candidate>`, so intervals with
the same human label cannot overwrite one another.

## First bounded execution

On 2026-08-31, the imec1 `rapid_motion` window (nominal start 5910 s,
duration 120 s; 3,599,971 samples) completed under DARTsort 0.5.16. The atomic
receipt reports 488,236 detected spikes, 478,251 assigned spikes, and 763 unit
labels. The normalized arrays contain 478,251 sorted, window-relative sample
indices ranging from 46 to 3,599,881, all inside the requested interval.
DARTsort's native timing report gives 1,531.6 s for its pipeline stages; total
wall time was longer because the accepted recording slice was first cached as
5.53 GB of float32 data.

This is an execution/integrity result, not evidence that DARTsort beats KS4.
The accepted full-session KS4 extraction for the same window contains 888,641
spikes and 514 labels, but raw spike or unit counts are not comparable quality
metrics across architectures. Event recovery, refractory/duplicate burden,
and longitudinal waveform-family continuity remain the advancement gates.

The run emitted a Numba warning that its installed TBB was too old for Numba's
TBB threading backend, so that backend was disabled. The run nevertheless
completed. DARTsort also reported 104 initially uncovered spikes during a
neighborhood-coverage check and expanded coverage to zero uncovered spikes.
