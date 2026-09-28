# EM.1 huklaban5 voltage-equivalence handoff

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
