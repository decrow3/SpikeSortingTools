import sys
import tempfile
import unittest
from unittest import mock
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
KS = ROOT / "environments/rescue-production/.venv/lib/python3.12/site-packages"
sys.path.insert(0, str(KS))

from kilosort import io, preprocessing, spikedetect, template_matching
from npx_preprocessing.motion import kilosort_support_mask_candidate as candidate_module
from npx_preprocessing.motion.kilosort_support_mask_candidate import (
    EXACT_LATTICE_LINEAGE, CovarianceValidation, SupportMaskPolicy,
    ValidatedSupportMaskContract, execute_native_with_runtime_covariance,
    execute_with_validated_support_mask,
    installed_validated_whitening, make_support_masked_binary_class,
    metadata_covariance_support_audit,
    validate_covariance, validate_support_mask_contract,
)


def policy(enabled=True, audit=None):
    geometry = np.array([[0., 0.], [0., 20.], [0., 40.], [0., 60.]])
    centers = np.array([0.005, 0.015, 0.025, 0.035])
    shifts = np.array([0., 20., 0., 20.])
    contract = ValidatedSupportMaskContract(
        "synthetic", "synthetic-contract", True, geometry, centers, shifts,
        1000., 0, 50, 4, "manifest", "content", "field",
    ) if enabled else None
    return SupportMaskPolicy(
        enabled=enabled,
        channel_locations_um=geometry, temporal_centers_s=centers,
        shifts_um=shifts, contract=contract,
    )


def bfile(pol, data, whitening=None, nt=3, NT=20):
    cls = make_support_masked_binary_class(io, pol)
    return cls(filename="synthetic", n_chan_bin=4, fs=1000, NT=NT, nt=nt,
               nt0min=1, chan_map=np.arange(4), whiten_mat=whitening,
               do_CAR=False, device=torch.device("cpu"), file_object=data)


class TestKilosortSupportMaskCandidate(unittest.TestCase):

 def test_disabled_is_bitwise_native(self):
    rng = np.random.default_rng(4)
    data = rng.normal(size=(50, 4)).astype("float32")
    native = io.BinaryFiltered(filename="x", n_chan_bin=4, fs=1000, NT=20, nt=3,
        chan_map=np.arange(4), whiten_mat=torch.eye(4), do_CAR=False,
        device=torch.device("cpu"), file_object=data)
    candidate = bfile(policy(False), data, torch.eye(4))
    for j in range(int(native.n_batches)):
        self.assertTrue(torch.equal(native.padded_batch_to_torch(j), candidate.padded_batch_to_torch(j)))


 def test_single_pre_w_mask_and_post_w_mixing(self):
    data = np.zeros((50, 4), dtype=np.float32)
    data[:, 0] = np.arange(50, dtype=np.float32) * 2
    data[:, 3] = np.arange(50, dtype=np.float32)
    W = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                      [0., 0., 1., 0.], [1., 0., 0., 1.]])
    bf = bfile(policy(), data, W)
    out, inds = bf.padded_batch_to_torch(0, return_inds=True)
    raw, _ = io.BinaryRWFile.padded_batch_to_torch(bf, 0, return_inds=True)
    bf.whiten_mat = None
    pre = io.BinaryFiltered.filter(bf, raw)
    bf.whiten_mat = W
    # +20 support drops final physical row; W may still make its output index nonzero.
    local = 20  # global frame 17, inside the +20 state
    self.assertNotEqual(pre[3, local], 0)
    self.assertNotEqual(out[3, local], 0)
    self.assertEqual(out[3, local], pre[0, local])
    self.assertEqual(inds[0], -3)


 def test_actual_covariance_consumer_finite_full_rank_and_metadata_schedule(self):
    rng = np.random.default_rng(9)
    data = rng.normal(size=(80, 4)).astype("float32")
    pol = policy()
    bf = bfile(pol, data, whitening=None, NT=20)
    W = preprocessing.get_whitening_matrix(
        bf, np.zeros(4), np.arange(4) * 20., nskip=1, nrange=4
    )
    self.assertTrue(torch.isfinite(W).all())
    self.assertEqual(torch.linalg.matrix_rank(W), 4)
    report = metadata_covariance_support_audit(
        pol, fs=1000, imin=0, imax=80, batch_size=20, nt=3, nskip=1
    )
    self.assertEqual(report["sampled_batch_indices"], [0, 1, 2])
    self.assertTrue(report["all_rows_have_valid_samples"])
    self.assertEqual(report["action"], "retain_all_rows")


 def test_actual_detection_and_feature_primitives_receive_masked_batch(self):
    data = np.zeros((50, 4), dtype="float32")
    data[17, 3] = 30
    bf = bfile(policy(), data, torch.eye(4))
    X = bf.padded_batch_to_torch(0)
    clips = spikedetect.extract_snippets(
        X, nt=3, twav_min=1, Th_single_ch=2, loc_range=[0, 1],
        long_range=[0, 1], device=torch.device("cpu")
    )
    # Unsupported row's pulse cannot become a detection/feature snippet.
    self.assertEqual(clips.shape[0], 0)
    self.assertIs(bf.support_mask_policy.audit.batches[-1]["transition_samples_retained"], True)

 def test_actual_pca_and_universal_template_learning_consumer(self):
    data = np.random.default_rng(1).normal(0, .05, (240, 4)).astype("float32")
    for i in [10, 70, 130, 190]:
        data[i, 0] += 20
    bf = bfile(policy(), data, torch.eye(4), nt=3, NT=20)
    ops = {"settings": {"n_pcs": 1, "n_templates": 1}}
    wPCA, wTEMP = spikedetect.extract_wPCA_wTEMP(
        ops, bf, nt=3, twav_min=1, Th_single_ch=2, nskip=1,
        device=torch.device("cpu")
    )
    self.assertEqual(tuple(wPCA.shape), (1, 3))
    self.assertEqual(tuple(wTEMP.shape), (1, 3))
    self.assertTrue(torch.isfinite(wPCA).all() and torch.isfinite(wTEMP).all())

 def test_actual_learned_template_extraction_consumer(self):
    data = np.random.default_rng(2).normal(0, .1, (50, 4)).astype("float32")
    bf = bfile(policy(), data, torch.eye(4), nt=3, NT=20)
    ops = {"settings": {"nearest_chans": 2, "position_limit": 100., "n_pcs": 1},
           "xc": np.zeros(4), "yc": np.arange(4) * 20., "nt": 3,
           "nt0min": 1, "batch_size": 20, "Th_learned": 100.,
           "max_peels": 2, "wPCA": torch.tensor([[0., 1., 0.]])}
    U = torch.ones((1, 1, 4), dtype=torch.float32)
    st, tF, _ = template_matching.extract(
        ops, bf, U, device=torch.device("cpu")
    )
    self.assertEqual(st.shape, (0, 3))
    self.assertEqual(tuple(tF.shape), (0, 2, 1))
    self.assertEqual(len(bf.support_mask_policy.audit.batches), int(bf.n_batches))


 def test_actual_preprocessing_export_uses_same_hook(self):
    data = np.arange(200, dtype=np.float32).reshape(50, 4)
    bf = bfile(policy(), data, torch.eye(4))
    ops = {"Nbatches": int(bf.n_batches), "nt": 3, "batch_size": 20,
           "preprocessing": {"whiten_mat": torch.eye(4), "hp_filter": None},
           "dshift": None, "chanMap": np.arange(4), "data_dtype": "float32",
           "n_chan_bin": 4}
    before = len(bf.support_mask_policy.audit.batches)
    with tempfile.TemporaryDirectory() as td:
        target = Path(td) / "preprocessed.bin"
        io.save_preprocessing(target, ops, bfile=bf)
        after = len(bf.support_mask_policy.audit.batches)
        self.assertGreater(after, before)
        self.assertEqual(target.stat().st_size, 20 * int(bf.n_batches) * 4 * 2)


 def test_missing_support_fails_closed_without_channel_drop(self):
    geometry = np.array([[0., 0.], [0., 20.]])
    centers = np.array([0.005, 0.015]); shifts = np.array([100., 100.])
    contract = ValidatedSupportMaskContract(
        "s", "s", True, geometry, centers, shifts, 1000., 0, 50, 2,
        "m", "c", "f",
    )
    pol = SupportMaskPolicy(True, geometry, centers, shifts, contract)
    report = metadata_covariance_support_audit(
        pol, fs=1000, imin=0, imax=50, batch_size=20, nt=3, nskip=1
    )
    self.assertFalse(report["all_rows_have_valid_samples"])
    self.assertEqual(report["action"], "fail_closed_no_geometry_change")

 def test_covariance_validator_positive_and_negative_cases(self):
    good = validate_covariance(np.eye(4), expected_channels=4, contract_digest="x")
    self.assertTrue(good.passed)
    cases = [
        np.array([[1., np.nan], [0., 1.]]),
        np.array([[1., np.inf], [0., 1.]]),
        np.diag([1., 1., 0., 1.]),
        np.diag([1., 1., 1., 1e-9]),
        np.eye(3),
    ]
    expected = ["nonfinite", "nonfinite", "rank_deficient", "over_conditioned", "wrong_shape"]
    for value, reason in zip(cases, expected):
        result = validate_covariance(value, expected_channels=4, contract_digest="x")
        self.assertFalse(result.passed)
        self.assertTrue(any(r.startswith(reason) for r in result.reasons))

 def _synthetic_contract_files(self, td, *, execution_enabled=True):
    td = Path(td); geometry = np.c_[np.zeros(4), np.arange(4) * 20.]
    field_path = td / "field.npz"
    np.savez(field_path, time_s=np.array([.005, .015, .025, .035]),
             rigid_displacement_um=np.array([0., 20., 0., 20.]))
    ops_path = td / "ops.npy"
    np.save(ops_path, {"fs": 1000., "batch_size": 20, "nt": 3,
        "nt0min": 1, "do_CAR": False, "artifact_threshold": 100.,
        "invert_sign": False, "chanMap": np.arange(4),
        "settings": {"nblocks": 0, "Th_universal": 12., "Th_learned": 9.,
                     "nskip": 25, "whitening_range": 32}})
    marker = td / "source.py"; marker.write_text("frozen\n")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest_path = td / "manifest.json"
    manifest = {"complete": True, "adapter": EXACT_LATTICE_LINEAGE,
        "recording_content_sha256": "content", "num_samples": 50,
        "num_channels": 4, "sampling_frequency_hz": 1000., "dtype": "float32",
        "channel_locations_um": geometry.tolist(),
        "recording_binary_files": [{"name": "synthetic.raw", "sha256": "binary"}]}
    manifest_path.write_text(json.dumps(manifest))
    config = {"schema": "en-minimum-training-support-mask-v2",
        "status": "synthetic-base", "failure_reason": "synthetic-base-disabled",
        "execution_enabled": execution_enabled,
        "real_covariance_receipt": None,
        "covariance_gate": {"maximum_condition_number": 1e8},
        "prohibitions": ["recording read", "sort"],
        "arbitrary_nested": {"keep": {"value": 7}},
        "structural": {"empty_dict": {}, "empty_list": [],
            "nonempty_dict": {"value": 1}, "nonempty_list": [1],
            "nested": {"keep": {"value": 1}}},
        "source_bindings": [{"path": str(marker), "sha256": sha(marker)}],
        "recording": {"manifest_path": str(manifest_path),
            "manifest_sha256": sha(manifest_path), "required_adapter": EXACT_LATTICE_LINEAGE,
            "recording_content_sha256": "content", "num_samples": 50,
            "num_channels": 4, "sampling_frequency_hz": 1000., "dtype": "float32",
            "binary_path": str(td / "synthetic.raw"),
            "binary_name": "synthetic.raw", "manifest_recorded_binary_sha256": "binary"},
        "crop": {"imin": 0, "imax": 50},
        "field": {"path": str(field_path), "sha256": sha(field_path),
            "time_key": "time_s", "displacement_key": "rigid_displacement_um",
            "rounding_um": 20.},
        "ops": {"path": str(ops_path), "sha256": sha(ops_path)},
        "frozen_kilosort_settings": {"fs": 1000., "batch_size": 20,
            "nt": 3, "nt0min": 1, "do_CAR": False, "artifact_threshold": 100.,
            "invert_sign": False, "settings.nblocks": 0,
            "settings.Th_universal": 12., "settings.Th_learned": 9.,
            "settings.nskip": 25, "settings.whitening_range": 32},
        "native_invocation": {"filename": str(td / "synthetic.raw"),
            "results_dir": str(td / "results"), "data_dtype": "float32",
            "do_CAR": False, "invert_sign": False, "device": "cpu",
            "save_extra_vars": True, "clear_cache": True,
            "save_preprocessed_copy": False,
            "settings": {"n_chan_bin": 4, "fs": 1000., "batch_size": 20,
                "tmin": 0., "tmax": .05, "nt": 3, "nt0min": 1,
                "nblocks": 0, "Th_universal": 12., "Th_learned": 9.,
                "artifact_threshold": "Infinity", "nskip": 25,
                "whitening_range": 32}}}
    config_path = td / "contract.json"; config_path.write_text(json.dumps(config))
    return config_path, sha(config_path), config, manifest_path, ops_path

 def _smoke_contract_files(self, td, *, execution_enabled=True):
    path, _, config, manifest_path, ops_path = self._synthetic_contract_files(
        td, execution_enabled=execution_enabled
    )
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    base_path = Path(td) / "base.json"
    base_path.write_text(json.dumps(config))
    base_sha = sha(base_path)
    smoke = json.loads(json.dumps(config))
    smoke["schema"] = "en-minimum-training-support-mask-v4-smoke"
    smoke["status"] = "synthetic-smoke-ready"
    smoke["failure_reason"] = "synthetic-smoke-review-pending"
    smoke["base_binding"] = {"path": str(base_path), "sha256": base_sha}
    run_root = Path(td) / "run"
    smoke["smoke"] = {
        "mode": "stop_after_covariance_whitening",
        "stop_after_whitening": True,
        "run_root": str(run_root),
        "launch_evidence_dir": str(run_root / "launch_evidence"),
    }
    smoke["native_invocation"]["results_dir"] = str(run_root / "native_results")
    path.write_text(json.dumps(smoke))
    return path, sha(path), smoke, base_path, base_sha, manifest_path, ops_path

 def _patched_smoke_base(self, base_path, base_sha):
    return mock.patch.multiple(
        candidate_module,
        REVIEWED_BASE_CONFIG_PATH=Path(base_path),
        REVIEWED_BASE_CONFIG_SHA256=base_sha,
        SMOKE_RUN_ROOT=Path(base_path).parent / "run",
        SMOKE_STATUS="synthetic-smoke-ready",
        SMOKE_FAILURE_REASON="synthetic-smoke-review-pending",
        SMOKE_SOURCE_HASH_INDICES=(),
        SMOKE_ALLOWED_DELTAS=(
            (("key", "base_binding"), ("key", "path")),
            (("key", "base_binding"), ("key", "sha256")),
            (("key", "failure_reason"),),
            (("key", "native_invocation"), ("key", "results_dir")),
            (("key", "schema"),),
            (("key", "smoke"), ("key", "launch_evidence_dir")),
            (("key", "smoke"), ("key", "mode")),
            (("key", "smoke"), ("key", "run_root")),
            (("key", "smoke"), ("key", "stop_after_whitening")),
            (("key", "status"),),
        ),
    )

 def test_complete_contract_and_wrapper_synthetic_positive_path(self):
    with tempfile.TemporaryDirectory() as td:
        path, digest, config, _, _ = self._synthetic_contract_files(td)
        contract = validate_support_mask_contract(path, expected_config_sha256=digest)
        cov = validate_covariance(np.eye(4), expected_channels=4,
                                  contract_digest=contract.contract_digest)
        result = execute_with_validated_support_mask(
            config_path=path, expected_config_sha256=digest,
            covariance_validation=cov, runner=lambda value: value + 1,
            native_io=io, runner_kwargs={"value": 6})
        self.assertEqual(result, 7)
        self.assertIs(io.BinaryFiltered.__name__, "BinaryFiltered")

 def test_contract_and_wrapper_fail_closed_cases(self):
    with tempfile.TemporaryDirectory() as td:
        path, digest, config, manifest_path, ops_path = self._synthetic_contract_files(td)
        contract = validate_support_mask_contract(path, expected_config_sha256=digest)
        cov = validate_covariance(np.eye(4), expected_channels=4,
                                  contract_digest=contract.contract_digest)
        bad_cov = validate_covariance(np.diag([1., 1., 1., 1e-9]),
            expected_channels=4, contract_digest=contract.contract_digest)
        with self.assertRaises(PermissionError):
            execute_with_validated_support_mask(config_path=path,
                expected_config_sha256=digest, covariance_validation=bad_cov,
                runner=lambda: None, native_io=io)

 def test_exact_boolean_shared_validator_and_frozen_base_deltas(self):
    with tempfile.TemporaryDirectory() as td:
        path, digest, config, base_path, base_sha, _, _ = self._smoke_contract_files(td)
        with self._patched_smoke_base(base_path, base_sha):
            validate_support_mask_contract(path, expected_config_sha256=digest)
            with mock.patch.object(
                candidate_module, "SMOKE_ALLOWED_DELTAS",
                candidate_module.SMOKE_ALLOWED_DELTAS + ((("key", "execution_enabled"),),),
            ):
                with self.assertRaises(ValueError):
                    validate_support_mask_contract(path, expected_config_sha256=digest)
            for malformed in (1, 0, "true", "false", None):
                broken = json.loads(json.dumps(config))
                broken["execution_enabled"] = malformed
                path.write_text(json.dumps(broken))
                changed = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    validate_support_mask_contract(path, expected_config_sha256=changed)
            for key, value in (("do_CAR", True), ("save_extra_vars", False)):
                broken = json.loads(json.dumps(config))
                broken["native_invocation"][key] = value
                path.write_text(json.dumps(broken))
                changed = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    validate_support_mask_contract(path, expected_config_sha256=changed)
            for allowed_path in ("status", "smoke.stop_after_whitening"):
                broken = json.loads(json.dumps(config))
                if allowed_path == "status": broken["status"] = 17
                else: broken["smoke"]["stop_after_whitening"] = False
                path.write_text(json.dumps(broken))
                changed = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError, msg=allowed_path):
                    validate_support_mask_contract(path, expected_config_sha256=changed)
            mutations = (
                ("prohibitions", lambda value: value.append("undeclared")),
                ("covariance_gate.maximum_condition_number",
                 lambda value: value.__setitem__("maximum_condition_number", 5e7)),
                ("real_covariance_receipt",
                 lambda value: value.__setitem__("real_covariance_receipt", "fake")),
                ("arbitrary_nested.keep.value",
                 lambda value: value["keep"].__setitem__("value", 8)),
                ("undeclared_empty_dict",
                 lambda value: value.__setitem__("undeclared_empty_dict", {})),
                ("undeclared_empty_list",
                 lambda value: value.__setitem__("undeclared_empty_list", [])),
                ("empty_to_nonempty",
                 lambda value: value["structural"]["empty_dict"].__setitem__("added", 1)),
                ("nonempty_to_empty",
                 lambda value: value["structural"].__setitem__("nonempty_dict", {})),
                ("dict_list_type_substitution",
                 lambda value: value["structural"].__setitem__("nonempty_dict", [1])),
                ("nested_insertion",
                 lambda value: value["structural"]["nested"].__setitem__("added", [])),
                ("nested_removal",
                 lambda value: value["structural"]["nested"].pop("keep")),
                ("prohibitions_bracket_keys",
                 lambda value: (value.pop("prohibitions"),
                                value.__setitem__("prohibitions[0]", "recording read"),
                                value.__setitem__("prohibitions[1]", "sort"))),
                ("dotted_key_for_nested_dict",
                 lambda value: (value.pop("arbitrary_nested"),
                                value.__setitem__("arbitrary_nested.keep.value", 7))),
                ("bracket_key_for_list",
                 lambda value: (value["structural"].pop("nonempty_list"),
                                value["structural"].__setitem__("nonempty_list[0]", 1))),
                ("numeric_dict_key_for_index",
                 lambda value: value["structural"].__setitem__("nonempty_list", {"0": 1})),
            )
            calls = []
            for name, mutate in mutations:
                broken = json.loads(json.dumps(config))
                if name == "prohibitions": mutate(broken["prohibitions"])
                elif name.startswith("covariance_gate"): mutate(broken["covariance_gate"])
                elif name == "real_covariance_receipt": mutate(broken)
                elif name == "arbitrary_nested.keep.value": mutate(broken["arbitrary_nested"])
                else: mutate(broken)
                path.write_text(json.dumps(broken))
                changed = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError, msg=name):
                    execute_native_with_runtime_covariance(
                        config_path=path, expected_config_sha256=changed,
                        native_runner=lambda **kwargs: calls.append(kwargs), native_io=io,
                        native_preprocessing=preprocessing)
                self.assertEqual(calls, [], name)
                self.assertFalse(Path(config["smoke"]["run_root"]).exists(), name)

 def test_typed_structural_paths_are_injective_and_deterministic(self):
    value = {
        "a.b": 1, "a": {"b": 2},
        "items[0]": 3, "items": [4],
        "numeric": {"0": 5}, "indexed": [6],
        "slash/key~": 7,
    }
    nodes = candidate_module._typed_structural_map(value)
    paths = {
        (("key", "a.b"),),
        (("key", "a"), ("key", "b")),
        (("key", "items[0]"),),
        (("key", "items"), ("index", 0)),
        (("key", "numeric"), ("key", "0")),
        (("key", "indexed"), ("index", 0)),
    }
    self.assertTrue(paths.issubset(nodes))
    self.assertEqual(len(paths), 6)
    ordered_once = sorted(nodes, key=candidate_module._path_sort_key)
    ordered_twice = sorted(nodes, key=candidate_module._path_sort_key)
    self.assertEqual(ordered_once, ordered_twice)
    self.assertEqual(candidate_module._json_pointer((("key", "slash/key~"),)),
                     "/slash~1key~0")

 def test_every_before_whitening_exit_writes_failure_receipt(self):
    cases = (
        ("normal_return", lambda **kwargs: {"returned": True},
         "normal_return_without_required_whitening_boundary", "RequiredWhiteningBoundaryMissing"),
        ("exception", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("before W")),
         "exception_before_required_whitening_boundary", "RuntimeError"),
    )
    for label, runner, expected_reason, expected_exception in cases:
        with self.subTest(label=label), tempfile.TemporaryDirectory() as td:
            path, digest, config, base_path, base_sha, _, _ = self._smoke_contract_files(td)
            with self._patched_smoke_base(base_path, base_sha):
                with self.assertRaises(RuntimeError):
                    execute_native_with_runtime_covariance(
                        config_path=path, expected_config_sha256=digest,
                        native_runner=runner, native_io=io,
                        native_preprocessing=preprocessing)
            receipt = json.loads(
                (Path(config["smoke"]["launch_evidence_dir"]) / "run_failure.json").read_text()
            )
            self.assertEqual(receipt["stage"],
                             "verify_whitening_boundary" if label == "normal_return" else "native_runner")
            self.assertEqual(receipt["reason"], expected_reason)
            self.assertEqual(receipt["exception_type"], expected_exception)
            self.assertFalse(receipt["crossed_whitening_boundary"])
            self.assertEqual(receipt["mask_audit_batch_count"], 0)

 def test_invalid_or_reused_namespace_fails_before_native_runner(self):
    with tempfile.TemporaryDirectory() as td:
        path, digest, config, base_path, base_sha, _, _ = self._smoke_contract_files(td)
        calls = []
        with self._patched_smoke_base(base_path, base_sha):
            broken = json.loads(json.dumps(config)); broken["execution_enabled"] = "true"
            path.write_text(json.dumps(broken)); changed = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                execute_native_with_runtime_covariance(
                    config_path=path, expected_config_sha256=changed,
                    native_runner=lambda **kw: calls.append(kw), native_io=io,
                    native_preprocessing=preprocessing)
            self.assertEqual(calls, [])
            path.write_text(json.dumps(config)); digest = hashlib.sha256(path.read_bytes()).hexdigest()
            Path(config["smoke"]["run_root"]).mkdir()
            with self.assertRaises(FileExistsError):
                execute_native_with_runtime_covariance(
                    config_path=path, expected_config_sha256=digest,
                    native_runner=lambda **kw: calls.append(kw), native_io=io,
                    native_preprocessing=preprocessing)
            self.assertEqual(calls, [])

 def test_actual_native_boundary_smoke_stops_after_durable_finite_w(self):
    with tempfile.TemporaryDirectory() as td:
        path, digest, config, base_path, base_sha, _, _ = self._smoke_contract_files(td)
        original_class = io.BinaryFiltered
        original_whitener = preprocessing.get_whitening_matrix
        reached_after_whitening = []
        def instrumented_native(**kwargs):
            self.assertEqual(kwargs["filename"], str(Path(td) / "synthetic.raw"))
            self.assertTrue(np.isposinf(kwargs["settings"]["artifact_threshold"]))
            data = np.random.default_rng(33).normal(size=(50, 4)).astype("float32")
            bf = io.BinaryFiltered(filename="synthetic", n_chan_bin=4, fs=1000,
                NT=20, nt=3, nt0min=1, chan_map=np.arange(4), whiten_mat=None,
                do_CAR=False, artifact_threshold=kwargs["settings"]["artifact_threshold"],
                device=torch.device("cpu"), file_object=data)
            preprocessing.get_whitening_matrix(
                bf, np.zeros(4), np.arange(4) * 20., nskip=25, nrange=32)
            reached_after_whitening.append(True)
        with self._patched_smoke_base(base_path, base_sha):
            result, audit = execute_native_with_runtime_covariance(
                config_path=path, expected_config_sha256=digest,
                native_runner=instrumented_native, native_io=io,
                native_preprocessing=preprocessing)
        evidence = Path(config["smoke"]["launch_evidence_dir"])
        self.assertEqual(result["status"], "smoke_stop_after_covariance_whitening")
        self.assertEqual(reached_after_whitening, [])
        self.assertTrue(audit.covariance_validation.passed and audit.whitening_finite)
        for name in ("actual_covariance.npy", "covariance_validation.json",
                     "actual_whitening.npy", "whitening_validation.json",
                     "covariance_batch_journal.json", "smoke_stop.json", "run_outcome.json"):
            self.assertTrue((evidence / name).is_file(), name)
        saved_w = np.load(evidence / "actual_whitening.npy", allow_pickle=False)
        self.assertEqual(audit.whitening_sha256,
            hashlib.sha256(np.ascontiguousarray(saved_w).tobytes()).hexdigest())
        self.assertIs(io.BinaryFiltered, original_class)
        self.assertIs(preprocessing.get_whitening_matrix, original_whitener)

 def test_sort_mode_boundary_returns_exact_audited_w_and_failure_receipt_survives(self):
    with tempfile.TemporaryDirectory() as td:
        path, digest, config, base_path, base_sha, _, _ = self._smoke_contract_files(td)
        downstream = {}
        original_context = installed_validated_whitening
        def no_stop_context(native_preprocessing, contract, audit, **kwargs):
            kwargs["stop_after_whitening"] = False
            return original_context(native_preprocessing, contract, audit, **kwargs)
        def failing_native(**kwargs):
            data = np.random.default_rng(51).normal(size=(50, 4)).astype("float32")
            bf = io.BinaryFiltered(filename="synthetic", n_chan_bin=4, fs=1000,
                NT=20, nt=3, nt0min=1, chan_map=np.arange(4), whiten_mat=None,
                do_CAR=False, device=torch.device("cpu"), file_object=data)
            downstream["W"] = preprocessing.get_whitening_matrix(
                bf, np.zeros(4), np.arange(4) * 20., nskip=25, nrange=32)
            raise RuntimeError("injected post-covariance failure")
        with self._patched_smoke_base(base_path, base_sha), \
             mock.patch.object(candidate_module, "installed_validated_whitening", no_stop_context):
            with self.assertRaisesRegex(RuntimeError, "injected post-covariance"):
                execute_native_with_runtime_covariance(
                    config_path=path, expected_config_sha256=digest,
                    native_runner=failing_native, native_io=io,
                    native_preprocessing=preprocessing)
        evidence = Path(config["smoke"]["launch_evidence_dir"])
        saved_w = np.load(evidence / "actual_whitening.npy", allow_pickle=False)
        self.assertTrue(np.array_equal(downstream["W"].detach().cpu().numpy(), saved_w))
        self.assertTrue((evidence / "covariance_validation.json").is_file())
        self.assertTrue((evidence / "whitening_validation.json").is_file())
        failure = json.loads((evidence / "run_failure.json").read_text())
        self.assertEqual(failure["exception_type"], "RuntimeError")
        self.assertEqual(failure["audit"]["whitening_sha256"],
            hashlib.sha256(np.ascontiguousarray(saved_w).tobytes()).hexdigest())

 def test_legacy_malformed_infinity_and_disabled_gate(self):
    with tempfile.TemporaryDirectory() as td:
        path, digest, config, _, _ = self._synthetic_contract_files(td)
        for malformed in ("inf", "+Infinity", "unlimited", None, True, False):
            broken = json.loads(json.dumps(config)); broken["native_invocation"]["settings"]["artifact_threshold"] = malformed
            path.write_text(json.dumps(broken)); changed = hashlib.sha256(path.read_bytes()).hexdigest()
            validated = validate_support_mask_contract(path, expected_config_sha256=changed)
            with self.assertRaises(ValueError):
                candidate_module._native_invocation(path, validated)
        path, disabled_digest, _, _, _ = self._synthetic_contract_files(td, execution_enabled=False)
        disabled = validate_support_mask_contract(path, expected_config_sha256=disabled_digest)
        disabled_cov = validate_covariance(np.eye(4), expected_channels=4,
                                           contract_digest=disabled.contract_digest)
        with self.assertRaises(PermissionError):
            execute_with_validated_support_mask(config_path=path,
                expected_config_sha256=disabled_digest, covariance_validation=disabled_cov,
                runner=lambda: None, native_io=io)


if __name__ == "__main__":
    unittest.main()
