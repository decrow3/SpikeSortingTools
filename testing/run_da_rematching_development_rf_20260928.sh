#!/usr/bin/env bash
set -euo pipefail

root=/home/huklab/Documents/RyanSorting/SpikeSortingTools
share=/mnt/NPX/Luke/DARTsort_motion_experiments/da_rematching_development_rf_20260928
out="$share/run"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

available_kb=$(df -Pk "$share" | awk 'NR==2 {print $4}')
if (( available_kb < 30 * 1024 * 1024 )); then
    echo "Refusing launch: fewer than 30 GiB free on output filesystem" >&2
    exit 73
fi

exec /usr/bin/time -v \
    /home/huklab/Documents/DataRowleyV1V2/DataRowleyV1V2/.venv/bin/python \
    "$root/testing/cp_w2_rf_evaluator.py" \
    --arm "D2L=/mnt/NPX/Luke/DARTsort_motion_experiments/luke0804-imec1-ao-d2l-v2/windows/W2/runs/D2L/sort/dartsort_sorting.npz" \
    --arm "REMATCH0=/mnt/NPX/Luke/DARTsort_motion_experiments/cy_fixed_bank_rematching_20260928/arms/REMATCH0_v3/dartsort_sorting.npz" \
    --arm "CD1_FULL=/mnt/NPX/Luke/DARTsort_motion_experiments/cy_fixed_bank_rematching_20260928/arms/CD1_FULL_v1/dartsort_sorting.npz" \
    --config "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/jobs/luke0804-imec1-full-session-dots-rf-v1/source/dots_rf_config.json" \
    --stimulus-cache "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-dots-rf/stimulus_cache_v1" \
    --gaze-csv "/media/huklab/Data/NPX/Ryansorting/Luke/DARTsort_motion_experiments/luke0804-imec1-dots-rf/gaze_calibration_development_v2/data_root/processed_declan/Luke_2025-08-04/dpi_calibration/right_eye/calibrated_dpi.csv" \
    --interval-metadata "$root/testing/inputs/da_rematching_development_rf_20260928/W2_INTERVAL_METADATA.json" \
    --output "$out" \
    --device cpu
