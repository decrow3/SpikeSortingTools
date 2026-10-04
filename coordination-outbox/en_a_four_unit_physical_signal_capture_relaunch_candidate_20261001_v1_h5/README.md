# A four-unit physical-signal capture fresh relaunch candidate

Status: **review-only; not installed, not authorized by a fresh H1 GO, and not
launched**.

## Delivery contract

- Global milestone: snippet-stage physical-signal validation of the four frozen
  A units before any medium-window progression.
- Decision changed: whether the reviewed 20 selections show usable original
  versus corrected-A physical waveform evidence under identical preprocessing.
- Cheapest adequate test: the unchanged bounded 41-read capture with duplicate
  positive control; no sort, training, matcher, RF, or holdout.
- Completion condition: exactly 41 reads, 32,492,544 logical input bytes and
  3,912,192 retained bytes; passing duplicate control; durable counters,
  summaries and completion/failure receipt within 300 seconds, 4 CPUs and 2 GiB.

The earlier v2 GO receipt and `approved_v1` output namespace are consumed
historical evidence. This candidate creates a new cryptographic chain and uses:

- future config `configs/en_a_four_unit_physical_signal_capture.approved.v3.json`;
- future validator `testing/en_a_four_unit_physical_signal_capture_launch_guard_v3.py`;
- future service `testing/en_a_four_unit_physical_signal_capture_approved_v3.service`;
- fresh output `testing/outputs/en_a_four_unit_physical_signal_capture_20261001_approved_v2`;
- fresh external receipt
  `testing/outputs/en_a_four_unit_physical_signal_capture_20261001_approved_v2_launch/H1_GO.v3.json`;
- fresh log `testing/outputs/en_a_four_unit_physical_signal_capture_20261001_approved_v2.launch.log`.

Scientific inputs, selections, both recordings, windows, preprocessing,
arithmetic, limits, stop condition, and exclusions are byte-for-byte or
semantically unchanged from approved v2. The capture source itself is
byte-identical (`e33bc834...b0c74fd`). Only consumed paths, receipt schema,
packet-member identities, and their hashes changed.

The v3 validator remains fail-closed and invokes one exact child only after a
future external H1 receipt binds this packet manifest plus installed config,
service, source, validator, governing contract, and child argv. `Restart=no` is
preserved; a refusal or failure consumes the future decision and requires a new
namespace.

## Required H1 review

Inspect the v2-to-v3 namespace/binding delta, run the included 13-test gate
suite, confirm config equality after removing `config_path`, `service`, and
`output`, and verify all fresh live paths remain absent. A fresh H1 GO receipt
must be created outside this packet only after the reviewed candidate files are
installed byte-for-byte and immediate H5 state/preflight checks pass.

Prior denial evidence:
`en_a_four_unit_physical_signal_capture_result_20261001_v1_h5`, manifest
`551bec4a850cfc9e2fd31ed480f102a653cb878a8f7ac26b0d9007a53b376122`.

## Implementation checks

- Done: v2/v3 normalized config comparison -> only config/service/output
  namespaces differ; capture source unchanged.
- Done: v3 gate suite -> 13/13 pass, including exact harmless one-child positive,
  validate-only zero-child path, and binding/mutation refusal cases.
- Done: resource and lifecycle lines -> 4 CPUs, 2 GiB, no swap, 300 seconds,
  one static start, `Restart=no`, fresh output/log/receipt paths.
- Done: prospective config/service/validator/output/log/receipt paths -> absent.
- Not done: installation, external H1 v3 GO receipt, voltage, capture, RF,
  holdout, sorting, or training -> deliberately pending independent review.
- Can establish: a collision-free, scientifically unchanged candidate chain is
  ready for H1 implementation review.
- Cannot establish: installed-chain correctness, launch authorization, capture
  completion, or physical-signal interpretation.
