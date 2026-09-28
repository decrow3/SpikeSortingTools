#!/bin/bash
set -euo pipefail

repo=/home/huklab/Documents/RyanSorting/SpikeSortingTools
out=/media/huklab/Data/NPX/Ryansorting/Luke/luke_imec1_medicine_reference_sweep_v2/stage4_ab
mkdir -p "$out/runtime_tmp" "$out/runtime_cache/torch" "$out/runtime_cache/mpl"
export TMPDIR="$out/runtime_tmp"
export TORCH_HOME="$out/runtime_cache/torch"
export MPLCONFIGDIR="$out/runtime_cache/mpl"
cd "$repo"
exec /home/huklab/Documents/DARTsort/environments/medicine-estimator/.venv/bin/python \
  testing/luke_imec1_slow_layer_ab_v1.py fit-all
