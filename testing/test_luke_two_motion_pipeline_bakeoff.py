"""Contract-to-runner-to-export integration tests for the thin candidate runner.

The runner is the seam where three separately written pieces meet: the delivery
contract, the identity replay, and the export/QC endpoint. These tests own that
seam. They build a small synthetic curated sort with a known answer and a small
contract of the same shape as the shipped one, then check that

* the contract's resolved settings and declared inputs are what execute, and a
  command line disagreeing with any of them is refused;
* a run reads only its declared interval and permitted context, and never the
  sealed panel, its buffer, or another arm's reserved evaluation interval;
* the recording clock and the original spike-row identities survive the export;
* labels are not invented -- originals are preserved and new families are
  `unvalidated`, not `good`;
* the control arm produces the same artifacts the candidate arm does.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from testing.first_pipeline_candidate_contract import (
    DEFAULT_CONTRACT,
    ContractRefusal,
    canonical_digest,
    freeze_acceptance,
    load_contract,
    validate,
)
from testing.luke_two_motion_pipeline_bakeoff import (
    RunnerRefusal,
    resolve_settings,
    run_bakeoff,
)

FS = 30000.0
DURATION_S = 1000.0
N_CHANNELS = 8
N_TEMPLATES = 2


# --------------------------------------------------------------------------- #
# a synthetic curated sort with a known answer
# --------------------------------------------------------------------------- #
def _write_curated(root: Path) -> Path:
    """Two clusters that are one neuron in time, plus two that are not.

    Cluster 1 fires up to 315 s and cluster 2 takes over at 330 s, 15 um away
    on the same template: they never coexist in one epoch, so a chain of links
    should join them into one family. The handoff sits exactly on the epoch
    overlap, which is the only way two clusters can hand over without appearing
    together somewhere.

    Cluster 3 sits 400 um away and must stay separate on distance. Cluster 4 is
    a perfect waveform, depth and amplitude match for cluster 3 but *coexists*
    with it in every epoch, so nothing may link them: two neurons firing side by
    side are not one neuron moving.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    def train(start, stop, rate, cluster, depth, amplitude, template):
        n = int((stop - start) * rate)
        times = np.linspace(start, stop, n, endpoint=False)
        times = times + rng.uniform(-0.002, 0.002, n)  # jitter, no ISI violations
        return [
            (t, cluster, depth, amplitude, template)
            for t in times
        ]

    spikes = (
        train(250.0, 315.0, 30.0, 1, 100.0, 12.0, 0)
        + train(330.0, 470.0, 30.0, 2, 115.0, 12.0, 0)
        + train(250.0, 470.0, 30.0, 3, 500.0, 40.0, 1)
        + train(250.15, 470.0, 30.0, 4, 505.0, 40.0, 1)
    )
    spikes.sort(key=lambda s: s[0])

    samples = np.round(np.array([s[0] for s in spikes]) * FS).astype(np.int64)
    clusters = np.array([s[1] for s in spikes], dtype=np.int32)
    depths = np.array([s[2] for s in spikes], dtype=np.float32)
    amps = np.array([s[3] for s in spikes], dtype=np.float64)
    templates = np.array([s[4] for s in spikes], dtype=np.int32)

    full_st = np.stack([samples.astype(np.float64), np.zeros(samples.size), amps], axis=1)
    np.save(root / "spike_times.npy", samples)
    np.save(root / "spike_clusters.npy", clusters)
    np.save(root / "full_st.npy", full_st)
    np.save(root / "kept_spikes.npy", np.ones(samples.size, dtype=bool))
    np.save(root / "spike_positions.npy",
            np.stack([np.zeros(samples.size, dtype=np.float32), depths], axis=1))
    np.save(root / "spike_templates.npy", templates)
    # `amplitudes.npy` deliberately differs from the QC amplitudes: if anything
    # reads this file instead of full_st[kept][:, 2] the numbers change.
    np.save(root / "amplitudes.npy", np.full(samples.size, 999.0, dtype=np.float32))

    n_samples = 10
    trough = -np.exp(-0.5 * ((np.arange(n_samples) - 5.0) / 1.5) ** 2)
    bank = np.zeros((N_TEMPLATES, n_samples, N_CHANNELS), dtype=np.float32)
    bank[0] = np.outer(trough, [0.1, 0.5, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0])
    bank[1] = np.outer(trough, [0.0, 0.0, 0.0, 0.1, 0.5, 1.0, 0.5, 0.1])
    np.save(root / "templates.npy", bank)
    np.save(root / "channel_positions.npy",
            np.stack([np.zeros(N_CHANNELS), np.arange(N_CHANNELS) * 20.0], axis=1))

    for name, label in (("cluster_group.tsv", "good"), ("cluster_KSLabel.tsv", "good")):
        (root / name).write_text(
            "cluster_id\t" + ("group\n" if "group" in name else "KSLabel\n")
            + "".join(f"{cid}\t{label}\n" for cid in (1, 2, 3, 4))
        )
    return root


def _contract(tmp_path: Path, **overrides) -> Path:
    """The shipped contract's shape, pointed at the synthetic sort."""
    payload = copy.deepcopy(load_contract(DEFAULT_CONTRACT))
    curated = _write_curated(tmp_path / "curated")
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir(exist_ok=True)

    payload["recording"].update({"sampling_frequency_hz": FS, "duration_s": DURATION_S,
                                 "num_samples": int(DURATION_S * FS),
                                 "data_dir": str(tmp_path / "raw")})
    for role in ("legacy", "rescue_control"):
        payload["comparators"][role].update({
            "curated": str(curated if role == "rescue_control" else tmp_path / "legacy_curated"),
            "qc_dir": str(qc_dir),
            "source_recording": str(tmp_path / f"{role}_recording"),
        })
    payload["intervals"]["sealed_panel"].update(
        {"exclusion_buffer_s": 20.0, "windows_s": [[600.0, 640.0]]}
    )
    payload["intervals"]["healthy_control_intervals"]["windows"] = [
        {"name": "H_test", "start_s": 200.0, "stop_s": 260.0, "time_third": 0,
         "combined_motion_score": 0.5}
    ]
    payload["intervals"]["development_windows"]["windows_s"] = [
        [0.0, 200.0], [260.0, 580.0], [660.0, DURATION_S]
    ]
    payload["output_root"] = str(tmp_path / "out")

    settings = payload["candidate"]["settings"]["value"]
    resolved = settings["resolved_configuration"]
    resolved["identity_link"].update(
        {"epoch_duration_s": 60.0, "epoch_overlap_s": 15.0, "min_spikes_per_epoch": 10,
         "waveform_channel_neighbourhood_um": 60.0}
    )
    resolved["completeness_qc"].update(
        # the 15 s handoff gap is the epoch overlap, not a recording gap
        {"spikes_per_window": 200, "min_finite_interior_windows": 1, "max_isi_s": 20.0}
    )
    resolved["execution"]["arms"]["case"].update(
        {"name": "synthetic_case", "endpoint_interval_s": [300.0, 400.0],
         "processing_interval_s": [270.0, 450.0]}
    )
    resolved["execution"]["arms"]["healthy_control"].update(
        {"name": "H_test", "endpoint_interval_s": [200.0, 260.0],
         "processing_interval_s": [140.0, 320.0]}
    )
    payload["acceptance"]["practical_failure"]["value"].update(
        {"name": "synthetic_case", "sort_id": "rescue_luke0804_v2v1_g0_imec0",
         "cluster_id": 1, "interval_s": [300.0, 400.0]}
    )
    for dependency in payload["candidate"]["unresolved_implementation_dependencies"]:
        if dependency["id"] in payload["candidate"]["dependency_requirements_resolved"]["value"]:
            dependency["status"] = "resolved"

    for dotted, value in overrides.items():
        node = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value

    settings["configuration_digest"] = canonical_digest(resolved)
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


@pytest.fixture
def contract(tmp_path) -> Path:
    return _contract(tmp_path)


@pytest.fixture
def out_root(tmp_path) -> Path:
    return tmp_path / "out"


def _frozen(contract: Path, out_root: Path) -> Path:
    freeze_acceptance(contract, out_root)
    return contract


def _load(path: Path):
    return json.loads(Path(path).read_text())


# --------------------------------------------------------------------------- #
# the contract, not the command line, decides what runs
# --------------------------------------------------------------------------- #
def _resolve(contract: Path, **kwargs):
    defaults = dict(option="unwarped_identity", arm="case", snippet_dir=None,
                    motion_info_dir=None, truth_path=None, config_path=None)
    return resolve_settings(load_contract(contract), **{**defaults, **kwargs})


def test_settings_come_from_the_contract_not_from_flags(contract):
    resolved = _resolve(contract)
    assert resolved["execution_mode"] == "retained_sort_replay"
    assert resolved["identity_config"].epoch_duration_s == 60.0
    assert resolved["processing_interval_s"] == (270.0, 450.0)
    assert resolved["curated"].name == "curated"


def test_a_snippet_dir_that_is_not_the_declared_input_is_refused(contract, tmp_path):
    with pytest.raises(RunnerRefusal, match="refusing --snippet-dir"):
        _resolve(contract, snippet_dir=tmp_path / "somewhere_else")


def test_a_config_that_does_not_match_the_contract_is_refused(contract, tmp_path):
    rogue = tmp_path / "rogue.json"
    rogue.write_text(json.dumps({"max_spatial_distance_um": 500.0}))
    with pytest.raises(RunnerRefusal, match="refusing --config"):
        _resolve(contract, config_path=rogue)


def test_a_config_that_matches_the_contract_is_accepted(contract, tmp_path):
    same = tmp_path / "same.json"
    same.write_text(json.dumps(
        load_contract(contract)["candidate"]["settings"]["value"]["resolved_configuration"]
    ))
    assert _resolve(contract, config_path=same)["arm"] == "case"


def test_truth_is_refused_for_a_retained_sort_replay(contract, tmp_path):
    with pytest.raises(RunnerRefusal, match="refusing --truth"):
        _resolve(contract, truth_path=tmp_path / "truth.json")


def test_motion_info_is_refused_when_the_contract_declares_no_motion(contract, tmp_path):
    with pytest.raises(RunnerRefusal, match="refusing --motion-info-dir"):
        _resolve(contract, motion_info_dir=tmp_path / "motion")


def test_a_contract_declaring_a_resort_is_refused_by_a_replay_runner(tmp_path):
    path = _contract(tmp_path)
    payload = load_contract(path)
    payload["candidate"]["settings"]["value"]["execution_mode"] = "resort"
    payload["candidate"]["settings"]["value"]["inputs"] = {
        "source_recording": payload["comparators"]["rescue_control"]["source_recording"]
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(RunnerRefusal, match="implements retained_sort_replay"):
        _resolve(path)


def test_a_tampered_configuration_digest_is_refused(tmp_path):
    path = _contract(tmp_path)
    payload = load_contract(path)
    payload["candidate"]["settings"]["value"]["configuration_digest"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(RunnerRefusal, match="configuration_digest mismatch"):
        _resolve(path)


# --------------------------------------------------------------------------- #
# a run reads only its declared interval and permitted context
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "processing, expected",
    [
        ([560.0, 700.0], "intersects sealed window"),
        ([180.0, 300.0], "intersects reserved healthy evaluation interval"),
        ([270.0, 1200.0], "leaves the recording"),
        ([450.0, 270.0], "must have stop > start"),
    ],
)
def test_a_case_interval_outside_its_permitted_context_is_refused(
    tmp_path, processing, expected
):
    path = _contract(tmp_path)
    payload = load_contract(path)
    arm = payload["candidate"]["settings"]["value"]["resolved_configuration"]["execution"]["arms"]["case"]
    arm["processing_interval_s"] = processing
    arm["endpoint_interval_s"] = [max(processing) - 10.0, max(processing) - 5.0]
    resolved = payload["candidate"]["settings"]["value"]["resolved_configuration"]
    payload["candidate"]["settings"]["value"]["configuration_digest"] = canonical_digest(resolved)
    path.write_text(json.dumps(payload))
    with pytest.raises(RunnerRefusal, match=expected):
        _resolve(path)


def test_the_healthy_arm_may_use_its_own_reserved_interval_and_only_that_one(tmp_path):
    path = _contract(tmp_path)
    assert _resolve(path, arm="healthy_control")["endpoint_interval_s"] == (200.0, 260.0)

    payload = load_contract(path)
    resolved = payload["candidate"]["settings"]["value"]["resolved_configuration"]
    resolved["execution"]["arms"]["healthy_control"]["endpoint_interval_s"] = [700.0, 760.0]
    resolved["execution"]["arms"]["healthy_control"]["processing_interval_s"] = [690.0, 800.0]
    payload["candidate"]["settings"]["value"]["configuration_digest"] = canonical_digest(resolved)
    path.write_text(json.dumps(payload))
    with pytest.raises(RunnerRefusal, match="not one of the contract's reserved healthy"):
        _resolve(path, arm="healthy_control")


def test_an_endpoint_outside_its_processing_interval_is_refused(tmp_path):
    path = _contract(tmp_path)
    payload = load_contract(path)
    resolved = payload["candidate"]["settings"]["value"]["resolved_configuration"]
    resolved["execution"]["arms"]["case"]["endpoint_interval_s"] = [260.0, 300.0]
    payload["candidate"]["settings"]["value"]["configuration_digest"] = canonical_digest(resolved)
    path.write_text(json.dumps(payload))
    with pytest.raises(RunnerRefusal, match="not inside its processing interval"):
        _resolve(path)


def test_only_whole_grid_epochs_inside_the_interval_are_processed(contract, out_root):
    _frozen(contract, out_root)
    manifest = run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                           out_root=out_root, contract_path=contract)
    identity = manifest["stages"]["unwarped_identity"]
    lo, hi = identity["epoch_span_s"]
    assert 270.0 <= lo and hi <= 450.0
    assert identity["epoch_indices"] == [6, 7, 8]


# --------------------------------------------------------------------------- #
# unresolved dependencies block execution
# --------------------------------------------------------------------------- #
def test_execution_is_refused_while_a_required_dependency_is_unresolved(tmp_path, out_root):
    path = _contract(tmp_path)
    payload = load_contract(path)
    for dependency in payload["candidate"]["unresolved_implementation_dependencies"]:
        if dependency["id"] == "option_b_unwarped_identity":
            dependency["status"] = "unresolved"
    path.write_text(json.dumps(payload))
    freeze_acceptance(path, out_root)
    with pytest.raises(ContractRefusal, match="dependencies this candidate requires are not"):
        run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                    out_root=out_root, contract_path=path)


def test_every_resolved_dependency_names_the_check_that_resolved_it():
    """`resolved` is a claim about a check that ran, not a label anyone may set.

    These three were `unresolved` while the runner and the linker were being
    corrected, which is what kept the contract unexecutable; they moved only
    when the tests named here passed.
    """
    payload = load_contract(DEFAULT_CONTRACT)
    required = set(payload["candidate"]["dependency_requirements_resolved"]["value"])
    for dependency in payload["candidate"]["unresolved_implementation_dependencies"]:
        if dependency["id"] in required:
            assert dependency["status"] == "resolved", dependency["id"]
            assert dependency["resolved_by"].startswith(("testing/", "Same integration test"))
    assert not validate(DEFAULT_CONTRACT, mode="authoring").unresolved_required_dependencies


# --------------------------------------------------------------------------- #
# real depths and the production amplitude source are required inputs
# --------------------------------------------------------------------------- #
def test_missing_depths_are_a_refusal_not_a_zero_depth(tmp_path, out_root):
    path = _contract(tmp_path)
    (tmp_path / "curated" / "spike_positions.npy").unlink()
    freeze_acceptance(path, out_root)
    with pytest.raises(RunnerRefusal, match="never defaulted to zeros"):
        run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                    out_root=out_root, contract_path=path)


def test_verify_checks_the_inputs_execution_will_consume(contract, out_root):
    manifest = run_bakeoff(option="unwarped_identity", arm="case", mode="verify",
                           out_root=out_root, contract_path=contract)
    consumed = manifest["stages"]["verify"]["files_execution_will_consume"]
    assert "full_st.npy" in consumed and "spike_positions.npy" in consumed
    # the observable the prescription forbids is not among them
    assert "amplitudes.npy" not in consumed


def test_the_replay_uses_the_qc_amplitudes_not_amplitudes_npy(contract, out_root):
    _frozen(contract, out_root)
    run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                out_root=out_root, contract_path=contract)
    exported = np.load(out_root / "case__candidate_export" / "qc_amplitudes.npy")
    # the fixture's amplitudes.npy is a constant 999.0; the QC amplitudes are not
    assert not np.allclose(exported, 999.0)
    assert set(np.unique(exported)) <= {12.0, 40.0}


# --------------------------------------------------------------------------- #
# the export preserves the clock, the rows and the original labels
# --------------------------------------------------------------------------- #
def test_the_export_preserves_the_recording_clock_and_the_original_rows(contract, out_root):
    _frozen(contract, out_root)
    run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                out_root=out_root, contract_path=contract)

    curated = Path(load_contract(contract)["comparators"]["rescue_control"]["curated"])
    original_samples = np.load(curated / "spike_times.npy")
    for arm_dir in ("case__baseline_export", "case__candidate_export"):
        rows = np.load(out_root / arm_dir / "spike_row_id.npy")
        samples = np.load(out_root / arm_dir / "spike_times.npy")
        assert samples.dtype == original_samples.dtype
        # integer samples of the recording clock, taken from the original rows
        assert np.array_equal(samples, original_samples[rows])
        assert np.unique(rows).size == rows.size

    baseline_rows = np.load(out_root / "case__baseline_export" / "spike_row_id.npy")
    candidate_rows = np.load(out_root / "case__candidate_export" / "spike_row_id.npy")
    assert np.array_equal(baseline_rows, candidate_rows)


def test_new_families_are_unvalidated_and_original_labels_are_preserved(contract, out_root):
    _frozen(contract, out_root)
    manifest = run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                           out_root=out_root, contract_path=contract)
    export = manifest["stages"]["candidate_export"]
    assert export["n_unvalidated_families"] >= 1
    assert "cluster_group.original.tsv" in export["original_labels_preserved"]

    provenance = (out_root / "case__candidate_export" / "unit_provenance.csv").read_text()
    assert "unvalidated" in provenance
    # every unit labelled `good` inherited that label from one original cluster
    for line in provenance.splitlines()[1:]:
        unit, label, source, cids, n = line.split(",", 4)
        if label == "good":
            assert source.startswith("original label of cluster")
            assert int(n) == 1

    original = (out_root / "case__candidate_export" / "cluster_group.original.tsv").read_text()
    assert original == (Path(
        load_contract(contract)["comparators"]["rescue_control"]["curated"]
    ) / "cluster_group.tsv").read_text()


def test_the_two_clusters_that_are_one_neuron_are_joined_into_one_family(contract, out_root):
    """The known answer: clusters 1 and 2 link; 3 and 4 stay separate."""
    _frozen(contract, out_root)
    manifest = run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                           out_root=out_root, contract_path=contract)
    endpoints = _load(out_root / "case__endpoints.json")
    assert endpoints["carrier_contributing_clusters"] == [1, 2]

    families = _load(out_root / "case__unwarped_identity" / "unwarped_identity_manifest.json")
    assert families["num_families_built_from_a_link"] == 1


def test_two_clusters_that_coexist_in_an_epoch_are_never_merged(contract, out_root):
    """Clusters 3 and 4 match on depth, amplitude and waveform, and fire together.

    Exclusivity is what separates them: each already claims its own successor in
    the next epoch, so neither can also claim the other's. A linker enforcing
    exclusivity on one side only would merge two simultaneously active neurons
    into a single family whose train violates its own refractory period.
    """
    _frozen(contract, out_root)
    run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                out_root=out_root, contract_path=contract)
    provenance = (out_root / "case__candidate_export" / "unit_provenance.csv").read_text()
    for line in provenance.splitlines()[1:]:
        contributing = line.split(",")[3].split()
        assert not {"3", "4"} <= set(contributing)


# --------------------------------------------------------------------------- #
# the control arm produces comparable outputs
# --------------------------------------------------------------------------- #
def test_the_control_arm_produces_the_same_export_the_candidate_does(contract, out_root):
    _frozen(contract, out_root)
    control = run_bakeoff(option="control", arm="case", mode="smoke",
                          out_root=out_root, contract_path=contract)
    assert control["stages"]["baseline_export"]["n_rows"] > 0
    for name in ("spike_times.npy", "spike_clusters.npy", "spike_row_id.npy",
                 "qc_amplitudes.npy", "cluster_group.tsv", "unit_provenance.csv"):
        assert (out_root / "case__baseline_export" / name).exists()
    assert (out_root / "case__control__bakeoff_manifest.json").exists()


# --------------------------------------------------------------------------- #
# the endpoint that motivated the candidate
# --------------------------------------------------------------------------- #
def test_the_case_arm_reports_all_four_gates_against_the_frozen_margins(contract, out_root):
    _frozen(contract, out_root)
    run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                out_root=out_root, contract_path=contract)
    endpoints = _load(out_root / "case__endpoints.json")

    completeness = endpoints["completeness"]
    assert completeness["baseline"]["amplitude_source"] == "full_st[kept_spikes][:, 2]"
    assert completeness["gate"]["verdict"] in ("pass", "fail", "unevaluable")
    assert completeness["gate"]["margin_pp"] == 20.0
    assert "family_amplitude_scale_check" in completeness

    assert endpoints["identity"]["floor"] == 0.8
    assert 0.0 <= endpoints["identity"]["retained_fraction"] <= 1.0
    assert endpoints["contamination"]["max_tolerated_increase"] == 0.01
    assert "not an anchor" in endpoints["contamination"]["scored_on"]


def test_pooling_incompatible_amplitude_scales_is_unevaluable_not_a_pass(tmp_path, out_root):
    """A family whose contributors sit at different amplitude scales is refused.

    Cluster 2's amplitude is dropped to 6.0 against cluster 1's 12.0. The link
    still passes the 2x amplitude-ratio gate, so the family forms -- and the
    completeness endpoint must then decline to fit the pooled distribution
    rather than report whatever number a bimodal fit produces.
    """
    path = _contract(tmp_path)
    curated = Path(load_contract(path)["comparators"]["rescue_control"]["curated"])
    clusters = np.load(curated / "spike_clusters.npy")
    full_st = np.load(curated / "full_st.npy")
    full_st[clusters == 2, 2] = 6.0
    np.save(curated / "full_st.npy", full_st)

    freeze_acceptance(path, out_root)
    run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                out_root=out_root, contract_path=path)
    endpoints = _load(out_root / "case__endpoints.json")

    scale_check = endpoints["completeness"]["family_amplitude_scale_check"]
    assert not scale_check["compatible"]
    assert scale_check["ratio"] == pytest.approx(2.0)
    assert endpoints["completeness"]["gate"]["verdict"] == "unevaluable"
    assert "`unevaluable` is not a pass" in endpoints["completeness"]["gate"]["note"]


def test_the_healthy_arm_measures_preservation_on_its_reserved_interval(contract, out_root):
    _frozen(contract, out_root)
    run_bakeoff(option="unwarped_identity", arm="healthy_control", mode="smoke",
                out_root=out_root, contract_path=contract)
    endpoints = _load(out_root / "healthy_control__endpoints.json")
    assert endpoints["endpoint_interval_s"] == [200.0, 260.0]
    assert endpoints["max_tolerated_increase_pp"] == 2.0
    assert endpoints["verdict"] in ("pass", "fail", "unevaluable")


# --------------------------------------------------------------------------- #
# write location
# --------------------------------------------------------------------------- #
def test_an_output_root_under_mnt_is_refused(contract):
    with pytest.raises(ContractRefusal, match="/mnt"):
        run_bakeoff(option="control", arm="case", mode="verify",
                    out_root=Path("/mnt/nope"), contract_path=contract)


def test_a_manifest_is_written_even_when_a_stage_fails(tmp_path, out_root):
    path = _contract(tmp_path)
    (tmp_path / "curated" / "spike_positions.npy").unlink()
    freeze_acceptance(path, out_root)
    with pytest.raises(RunnerRefusal):
        run_bakeoff(option="unwarped_identity", arm="case", mode="smoke",
                    out_root=out_root, contract_path=path)
    manifest = _load(out_root / "case__unwarped_identity__bakeoff_manifest.json")
    assert manifest["status"] == "failed"
    assert "spike_positions.npy" in manifest["stages"]["failure"]["reason"]
