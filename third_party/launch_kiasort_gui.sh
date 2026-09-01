#!/usr/bin/env bash
set -euo pipefail

kiasort_root="${KIASORT_PATH:-/home/huklab/Documents/KIASORT}"
kiasort_python="${KIASORT_PYTHON_EXECUTABLE:-/home/huklab/anaconda3/envs/kiasort-python/bin/python}"
python_root="$(dirname "$(dirname "${kiasort_python}")")"
export LD_LIBRARY_PATH="${python_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONNOUSERSITE=1
export NUMBA_NUM_THREADS="${KIASORT_NUMBA_THREADS:-2}"
export NUMBA_CACHE_DIR="${KIASORT_NUMBA_CACHE_DIR:-/tmp/kiasort-numba-cache}"

exec matlab -desktop -r "setenv('PYTHONNOUSERSITE','1'); pyenv('Version','${kiasort_python}','ExecutionMode','OutOfProcess'); addpath(genpath('${kiasort_root}')); kiaSort"
