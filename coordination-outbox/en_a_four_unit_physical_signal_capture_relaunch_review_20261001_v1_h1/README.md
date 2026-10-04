# Independent H1 review: A four-unit capture relaunch candidate

Review date: 2026-10-01.

## Verdict

**GO for byte-identical installation, no-read preflight, and exactly one normal-tool
start, conditional on the prelaunch checks in `GO_DECISION.json`.** This review
does not install, copy a live gate receipt, start a service, open a recording, or
read voltage.

The fresh candidate packet is complete and hash-valid:

- packet manifest: `f897ff3b18d89d7cd55ea7500be1dfc4019f3f7b117e0cd2abd1edc3ffb5b6f3`;
- packet COMPLETE: `edf3e8af0a56846052e5fac125ab8853355854038a837b73585d6d1dbad8b816`.

The v2-to-v3 delta is limited to the consumed config, service, validator,
receipt, output and log namespaces plus the hashes and manifest path required
to bind those names. After removing `config_path`, `service`, and `output`, the
v2 and v3 configs are equal. The capture source remains byte-identical at
`e33bc8346b2acda30cd2032c7b6a00a004f79f7416a1198c6bcd718f3b0c74fd`.
Thus the recording identities, 20 selections, windows, preprocessing,
duplicate positive control, arithmetic, resource limits and stop condition are
unchanged.

The v3 launch guard binds an external receipt to the candidate manifest,
installed config, service, capture source, executing validator, governing
contract manifest and exact child argv. It fails closed for absent, malformed,
stale, mutated or mismatched bindings. Its harmless positive fixture invokes
one test double exactly once; validate-only invokes zero children and reports
zero recording access.

The unmodified 13-test file initially produced 11 passes and two path errors on
H1 because two preservation fixtures intentionally reference H5's live v2
installation. Those three v2 inputs were then staged byte-for-byte from the
immutable v2 launch-gate packet in a temporary tree, `LIVE_ROOT` was redirected
in memory, and the same test module passed 13/13. No production child or
recording path was opened.

The prior v2 decision was consumed by a pre-execution normal-tool denial,
preserved under `en_a_four_unit_physical_signal_capture_result_20261001_v1_h5`
with manifest `551bec4a850cfc9e2fd31ed480f102a653cb878a8f7ac26b0d9007a53b376122`.
That packet records zero service starts, zero recording opens/reads/bytes and
no production output. The user's subsequent direct standing authorization
covers this comparable bounded capture; the prior approval-provenance denial
does not invalidate the fresh technical chain.

`INSTALL_PREFLIGHT_ONE_START_GO.json` is the fresh review authorization. The
validator-format `H1_GO.v3.json` must remain unreleased until H5 installs the
reviewed bytes, verifies all installed hashes, performs the two zero-read
preflights, and confirms the fresh output/log/receipt namespaces are absent.
It must then bind the exact values in `V3_GATE_BINDINGS.json`, be copied to the
listed external live receipt path, and be consumed by one start request only.
No automatic or manual retry is authorized; any refusal or failure consumes
this decision and requires another fresh namespace and review chain.

## Implementation checks

- Done: candidate manifest and COMPLETE hashes -> exact expected values; all
  nine manifest members verify.
- Done: v2/v3 config, service and validator diffs -> only required fresh
  namespace/schema/member/binding changes; capture source unchanged.
- Done: v3 gate fixtures -> 13/13 pass with immutable v2 H5 inputs staged on
  H1; mutation, stale-path, wrong-argv and receipt-location negatives refuse.
- Done: resource/lifecycle semantics -> 41 reads, 32,492,544 logical input
  bytes, 3,912,192 retained bytes, 4 CPUs, 2 GiB RAM, no swap, 300 seconds,
  `Restart=no`, one authorized start request.
- Done: prior denial linkage -> exact result manifest and zero-execution state.
- Not done: current H5 installed-path hashes, live collision check, external
  validator-format receipt release, voltage capture or physical-signal result;
  these are deliberately post-GO prelaunch/execution work.
- Can establish: the relaunch candidate is scientifically unchanged and its
  fresh fail-closed chain is ready for byte-identical install and conditional
  one-start dispatch.
- Cannot establish: current mutable H5 state, successful capture, duplicate
  control outcome, or physical waveform interpretation.
