# EM.1 huklaban5 voltage-equivalence handoff

**Outcome:** this historical execution handoff is superseded. The completed
[EM.1 result](EM-huklaban5-voltage-equivalence-result-20260928.md) passes the
task's explicit `1e-6` rule: the exact adapter's maximum residual is
`2.98e-8` uV and its maps and zero masks are identical. The old byte-strict
qualification failed because the nonzero-motion snippets retain 1–2-ULP
upstream preprocessing-version differences.

EM.2a has since completed; see the
[operator-screen result](EM2a-operator-screen-result-20260929.md). This file
remains the provenance for the original EM.1 command and return packet. It does
not authorize an EM.2b sort.

The compact huklaban5 return packet is published under
`/mnt/NPX/Luke/DARTsort_motion_experiments/em_spikeinterface_lattice_20260928/huklaban5_result_v1`
with manifest SHA-256
`84281482db49255f3a5361570fc1930a53d39a44df736bb9071d6d7f4bf74071`.

The 0.104.7 source/kernel and actual-geometry audit is complete: stock nearest
fails only at four AP191 interior-hole mappings. Run the exact adapter against
the existing accepted DD parent and S_L materialization. Do not launch a sort.

## Preflight

Use the published packet's `MANIFEST.json` to verify every source hash. Confirm:

- hostname is huklaban5;
- Python reports SpikeInterface 0.104.7;
- the DD root has `lattice_W2_pair.csv`, `inputs/S_L/recording.json`, and
  `source/dd_lattice_inputs.py`;
- the accepted W2 parent has `provenance.json`;
- the output directory does not already exist.

## Command

```bash
PACKET=/mnt/NPX/Luke/DARTsort_motion_experiments/em_spikeinterface_lattice_20260928/huklaban1_v1
PYTHONPATH="$PACKET/source" \
  /home/huklaban5/Documents/SpikeSortingTools/NPX_preprocessing/.venv/bin/python \
  "$PACKET/source/testing/em_huklaban5_voltage_equivalence.py" \
  --output-dir /home/huklaban5/DARTsort_experiment_scratch/em_spikeinterface_lattice_20260928/voltage_equivalence_v1
```

The runner freezes `SELECTIONS.json` before reading voltage. It reads three
deterministically selected 3-second snippets (flat, episode core, transition),
applies the exact adapter lazily on the 383-channel accepted parent, selects
AP202:383, and compares with DD S_L. It writes hashes and differences only—no
voltage samples—and writes `COMPLETE.json` last only on full pass.

Expected selections from the published knot table are nominal starts 1118.625 s
(flat), 1031.625 s (core), and 938.375 s (transition). Treat any change as an
input/version mismatch and stop.

## Return packet

Publish the compact output directory plus executed source hashes to
`/mnt/NPX/Luke/DARTsort_motion_experiments/em_spikeinterface_lattice_20260928/huklaban5_result_v1`.
Verify hashes and write `COMPLETE.json` last. Do not publish parent traces,
snippet arrays, caches, or raw/materialized voltage. Preserve a failed local
result for diagnosis; do not tune or rerun before reporting the first mismatch.
