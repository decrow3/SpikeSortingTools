"""Bounded split-half reproducibility inside [7200, 7320] s.

Run with the `spikeinterface` conda env, which is where SpikeInterface lives:
    /home/huklab/anaconda3/envs/spikeinterface/bin/python -m testing.luke_option_a_split_half_probe

RESULT ON 2026-09-06: both halves returned identically zero displacement, with
library defaults and again with the only DREDGE configuration recorded on disk
(the withdrawn rescue sidecar's, which carries `"fallback": "identity"`). The
correlation is undefined, not low. See docs/luke_option_a_readiness_assessment.md
section 3.2: the finding is that the accepted estimates' generating
configurations are not recorded, so the field that would actually be applied
cannot be re-estimated and its reproducibility cannot be measured.


Deterministic interleaved 2 s blocks: both halves span the same 120 s and see
the same slow motion, so agreement means the estimate is stable rather than
that the halves are the same data.

Reproducibility (precision) ONLY. It says nothing about whether either half's
displacement is the true tissue motion; the absolute gain stays unmeasured.
"""

import json
import sys

import numpy as np
from spikeinterface.core import load
from spikeinterface.sortingcomponents.motion import estimate_motion

D = "/mnt/NPX/Luke/20250804/dredge_pipeline_results_Luke0804_V2V1_g0_imec0"
# The dredge pipeline's traces_cached_seg0.raw is deleted; the rescue recording
# carries geometry/duration/fs, which is all estimate_motion needs. Traces are
# never read here.
REC = "/mnt/NPX/Luke/20250804/rescue_pipeline_results_Luke0804_V2V1_g0_imec0/recording"
FS = 29999.835983263598
LO, HI = 7200.0, 7320.0
BLOCK_S = 2.0

full = load(REC)
binary = json.load(open(f"{D}/preprocessed_recording/binary.json"))["kwargs"]
assert full.get_num_channels() == binary["num_channels"]
assert abs(full.get_sampling_frequency() - binary["sampling_frequency"]) < 1e-6

F0, F1 = int(round(LO * FS)), int(round(HI * FS))
rec = full.frame_slice(start_frame=F0, end_frame=F1)
print(f"sliced recording: {rec.get_num_samples()} samples "
      f"({rec.get_num_samples()/FS:.1f} s), {rec.get_num_channels()} chans", file=sys.stderr)

peaks = np.load(f"{D}/motion/peaks.npy")
locs = np.load(f"{D}/motion/peak_locations.npy")
inside = (peaks["sample_index"] >= F0) & (peaks["sample_index"] < F1)
peaks, locs = peaks[inside].copy(), locs[inside].copy()
peaks["sample_index"] -= F0                     # relative to the slice
print(f"peaks in window: {peaks.size}", file=sys.stderr)

block = (peaks["sample_index"] / FS // BLOCK_S).astype(int)
results, traces = {}, {}
for name, mask in (("a", block % 2 == 0), ("b", block % 2 == 1)):
    print(f"estimating half {name}: {int(mask.sum())} peaks", file=sys.stderr)
    motion = estimate_motion(rec, peaks[mask], locs[mask], direction="y",
                             rigid=True, method="dredge_ap",
                             progress_bar=False, verbose=False)
    d = np.asarray(motion.displacement[0], dtype=float)
    t = np.asarray(motion.temporal_bins_s[0], dtype=float)
    trace = d.mean(axis=1) if d.ndim > 1 else d
    traces[name] = (t, trace)
    results[name] = {
        "n_peaks": int(mask.sum()),
        "n_time_bins": int(trace.size),
        "rigid_excursion_um": float(np.percentile(trace, 95) - np.percentile(trace, 5)),
        "max_abs_um": float(np.abs(trace - trace.mean()).max()),
    }
    print(f"  half {name}: {trace.size} bins, excursion "
          f"{results[name]['rigid_excursion_um']:.2f} um", file=sys.stderr)

(ta, va), (tb, vb) = traces["a"], traces["b"]
grid = np.arange(max(ta.min(), tb.min()), min(ta.max(), tb.max()), 1.0)
ca = np.interp(grid, ta, va); ca -= ca.mean()
cb = np.interp(grid, tb, vb); cb -= cb.mean()
denom = float(np.linalg.norm(ca) * np.linalg.norm(cb))

results.update({
    "split_half_correlation": float(np.dot(ca, cb) / denom) if denom > 0 else float("nan"),
    "rms_difference_um": float(np.sqrt(np.mean((ca - cb) ** 2))),
    "n_common_bins": int(grid.size),
    "block_s": BLOCK_S,
    "interval_s": [LO, HI],
    "method": "dredge_ap, rigid",
    "geometry_carrier": REC,
    "measures": "reproducibility (precision) only; NOT absolute accuracy",
})
print(json.dumps(results, indent=2))
