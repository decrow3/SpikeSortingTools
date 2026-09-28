#!/bin/bash
set -euo pipefail

output=${1:?stage3_q output path is required}
pilot_unit=luke-medicine-q-t-pilot-20260924.service
python=/home/huklab/anaconda3/envs/spikeinterface/bin/python
controller=/home/huklab/Documents/RyanSorting/SpikeSortingTools/testing/luke_imec1_medicine_stage3_q_v1.py

while [[ "$(systemctl --user is-active "$pilot_unit")" == active ]]; do
    sleep 30
done
test "$(systemctl --user show "$pilot_unit" --property=ExecMainStatus --value)" = 0
"$python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "pass"' "$output/pilot_gate.json"

"$python" "$controller" validate-block-concat-u --output "$output"
"$python" "$controller" extract-full-u --output "$output"
"$python" "$controller" sweep --scope full --output "$output"
"$python" "$controller" stitch-full --output "$output"
"$python" "$controller" evaluate-full --output "$output"
"$python" "$controller" catalogue --output "$output"
"$python" "$controller" handoff --output "$output"
