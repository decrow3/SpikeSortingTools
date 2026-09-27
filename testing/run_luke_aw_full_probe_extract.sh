#!/usr/bin/env bash
set -euo pipefail

repo=/home/huklab/Documents/RyanSorting/SpikeSortingTools
attempt=${LUKE_AW_ATTEMPT:-attempt3}
out="$repo/testing/outputs/luke_au_cpu_preparation/full_probe_extraction_v1/$attempt"
mkdir -p "$out"
exec >>"$out/service.log" 2>&1

export CUDA_VISIBLE_DEVICES=""
export NUMBA_CACHE_DIR=/tmp/luke_aw_numba_cache
export TMPDIR=/dev/shm
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONPATH=/home/huklab/Documents/DARTsort/src:/home/huklab/Documents/DARTsort
export LUKE_AW_ATTEMPT="$attempt"

cd "$repo"
exec /usr/bin/timeout --signal=TERM --kill-after=120s 3600s \
  /home/huklab/Documents/DARTsort/.venv/bin/python \
  testing/luke_aw_full_probe_extract.py
