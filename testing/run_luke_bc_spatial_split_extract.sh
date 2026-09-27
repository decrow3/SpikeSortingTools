#!/usr/bin/env bash
set -euo pipefail

repo=/home/huklab/Documents/RyanSorting/SpikeSortingTools
attempt=${LUKE_AW_ATTEMPT:?LUKE_AW_ATTEMPT must be an explicit BC attempt id}
out="$repo/testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/$attempt"
mkdir -p "$out"
exec >>"$out/service.log" 2>&1

free_bytes=$(df -B1 --output=avail /media/huklab/Data | tail -n 1 | tr -d ' ')
if (( free_bytes < 30000000000 )); then
  echo "BC disk guard failed: ${free_bytes} bytes free"
  exit 75
fi

export CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/luke_aw_numba_cache
export TMPDIR=/dev/shm
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/home/huklab/Documents/DARTsort/src:/home/huklab/Documents/DARTsort
export LUKE_AW_ATTEMPT="$attempt"
export LUKE_AW_PRIOR_CHARGED_S=${LUKE_AW_PRIOR_CHARGED_S:-3272.6309895813465}
export LUKE_BC_SPATIAL_SPLITS=1

cd "$repo"
exec /usr/bin/timeout --signal=TERM --kill-after=120s 3500s \
  /home/huklab/Documents/DARTsort/.venv/bin/python \
  testing/luke_aw_full_probe_extract.py
