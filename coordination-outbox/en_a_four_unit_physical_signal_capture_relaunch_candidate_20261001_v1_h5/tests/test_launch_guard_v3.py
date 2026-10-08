from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SOURCE_VALIDATOR = Path(__file__).resolve().parents[1] / "source" / "en_a_four_unit_physical_signal_capture_launch_guard_v3.py"
PACKET_ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = Path("/home/huklaban5/Documents/SpikeSortingTools/SpikeSortingTools")
SCHEMA = "en-a-four-unit-physical-signal-capture-h1-go-v3"
BINDING_NAMES = (
    "packet_manifest",
    "config",
    "service",
    "capture_source",
    "validator",
    "governing_contract_manifest",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argv_sha256(argv: list[str]) -> str:
    payload = json.dumps(argv, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class Fixture:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.packet = self.root / "packet"
        self.packet_source = self.packet / "source"
        self.packet_source.mkdir(parents=True)
        self.paths = {
            "config": self.packet_source / "en_a_four_unit_physical_signal_capture.approved.v3.json",
            "service": self.packet_source / "en_a_four_unit_physical_signal_capture_approved_v3.service",
            "capture_source": self.packet_source / "en_a_four_unit_physical_signal_capture.py",
            "validator": self.packet_source / "en_a_four_unit_physical_signal_capture_launch_guard_v3.py",
            "governing_contract_manifest": self.root / "governing_contract.MANIFEST.sha256",
            "packet_manifest": self.packet / "MANIFEST.sha256",
        }
        self.paths["config"].write_text('{"fixture":"config"}\n')
        self.paths["service"].write_text("fixture service\n")
        self.paths["capture_source"].write_text("# fixture capture source\n")
        shutil.copyfile(SOURCE_VALIDATOR, self.paths["validator"])
        self.paths["governing_contract_manifest"].write_text("0" * 64 + "  contract\n")
        members = {
            "config": "source/en_a_four_unit_physical_signal_capture.approved.v3.json",
            "service": "source/en_a_four_unit_physical_signal_capture_approved_v3.service",
            "capture_source": "source/en_a_four_unit_physical_signal_capture.py",
            "validator": "source/en_a_four_unit_physical_signal_capture_launch_guard_v3.py",
        }
        self.paths["packet_manifest"].write_text("".join(
            f"{sha256(self.paths[name])}  {member}\n" for name, member in members.items()
        ))
        self.counter = self.root / "child_count.txt"
        self.test_double = self.root / "test_double.py"
        self.test_double.write_text(
            "from pathlib import Path\nimport sys\n"
            "p=Path(sys.argv[1]); n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1))\n"
        )
        self.launch = [sys.executable, str(self.test_double), str(self.counter)]
        self.receipt = self.root / "H1_GO.test.json"
        self.value = self.make_receipt()
        self.write_receipt()

    def close(self) -> None:
        self.temp.cleanup()

    def make_receipt(self) -> dict:
        return {
            "schema": SCHEMA,
            "verdict": "GO",
            "bindings": {
                name: {"path": str(path.resolve()), "sha256": sha256(path)}
                for name, path in self.paths.items()
            },
            "launch": {"argv": self.launch, "argv_sha256": argv_sha256(self.launch)},
        }

    def write_receipt(self) -> None:
        self.receipt.write_text(json.dumps(self.value, indent=2, sort_keys=True) + "\n")

    def command(self, *, validate_only: bool = False) -> list[str]:
        command = [
            sys.executable,
            str(self.paths["validator"]),
            "--receipt", str(self.receipt),
            "--packet-manifest", str(self.paths["packet_manifest"]),
            "--config", str(self.paths["config"]),
            "--service", str(self.paths["service"]),
            "--capture-source", str(self.paths["capture_source"]),
            "--validator", str(self.paths["validator"]),
            "--governing-contract-manifest", str(self.paths["governing_contract_manifest"]),
        ]
        if validate_only:
            command.append("--validate-only")
        return command + ["--launch-command", *self.launch]

    def run(self, *, validate_only: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(self.command(validate_only=validate_only), text=True, capture_output=True, check=False)

    def assert_never_launched(self, case: unittest.TestCase) -> None:
        case.assertFalse(self.counter.exists(), "refused gate invoked the child")


class LaunchGuardTests(unittest.TestCase):
    def test_absent_empty_malformed_and_no_go_receipts_refuse(self) -> None:
        variants = ("absent", "empty", "malformed", "NO-GO")
        for variant in variants:
            with self.subTest(variant=variant):
                fixture = Fixture()
                try:
                    if variant == "absent":
                        fixture.receipt.unlink()
                    elif variant == "empty":
                        fixture.receipt.write_text("")
                    elif variant == "malformed":
                        fixture.receipt.write_text("{not json")
                    else:
                        fixture.value["verdict"] = "NO-GO"
                        fixture.write_receipt()
                    self.assertNotEqual(fixture.run().returncode, 0)
                    fixture.assert_never_launched(self)
                finally:
                    fixture.close()

    def test_missing_fields_refuse(self) -> None:
        cases = [("top", name) for name in ("schema", "verdict", "bindings", "launch")]
        cases += [("binding", name) for name in BINDING_NAMES]
        cases += [("binding_member", name) for name in ("path", "sha256")]
        for location, name in cases:
            with self.subTest(location=location, name=name):
                fixture = Fixture()
                try:
                    if location == "top":
                        del fixture.value[name]
                    elif location == "binding":
                        del fixture.value["bindings"][name]
                    else:
                        del fixture.value["bindings"]["config"][name]
                    fixture.write_receipt()
                    self.assertNotEqual(fixture.run().returncode, 0)
                    fixture.assert_never_launched(self)
                finally:
                    fixture.close()

    def test_wrong_types_refuse(self) -> None:
        mutators = {
            "schema": lambda value: value.__setitem__("schema", 1),
            "verdict": lambda value: value.__setitem__("verdict", True),
            "bindings": lambda value: value.__setitem__("bindings", []),
            "binding_path": lambda value: value["bindings"]["config"].__setitem__("path", 1),
            "binding_sha": lambda value: value["bindings"]["config"].__setitem__("sha256", False),
            "launch_argv": lambda value: value["launch"].__setitem__("argv", "not-a-list"),
            "launch_sha": lambda value: value["launch"].__setitem__("argv_sha256", 1),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name):
                fixture = Fixture()
                try:
                    mutate(fixture.value)
                    fixture.write_receipt()
                    self.assertNotEqual(fixture.run().returncode, 0)
                    fixture.assert_never_launched(self)
                finally:
                    fixture.close()

    def test_stale_path_binding_refuses(self) -> None:
        fixture = Fixture()
        try:
            stale = fixture.root / "stale_config.json"
            shutil.copyfile(fixture.paths["config"], stale)
            fixture.value["bindings"]["config"]["path"] = str(stale)
            fixture.write_receipt()
            self.assertNotEqual(fixture.run().returncode, 0)
            fixture.assert_never_launched(self)
        finally:
            fixture.close()


    def test_receipt_inside_packet_refuses(self) -> None:
        fixture = Fixture()
        try:
            inside = fixture.packet / "H1_GO.json"
            fixture.receipt = inside
            fixture.write_receipt()
            self.assertNotEqual(fixture.run().returncode, 0)
            fixture.assert_never_launched(self)
        finally:
            fixture.close()


class ContractPreservationTests(unittest.TestCase):
    def test_v3_changes_only_consumed_namespaces_and_bindings(self) -> None:
        old = json.loads((LIVE_ROOT / "configs" / "en_a_four_unit_physical_signal_capture.approved.v2.json").read_text())
        new = json.loads((PACKET_ROOT / "source" / "en_a_four_unit_physical_signal_capture.approved.v3.json").read_text())
        for key in ("config_path", "service", "output"):
            old.pop(key, None)
            new.pop(key, None)
        self.assertEqual(old, new)
        self.assertEqual(
            sha256(LIVE_ROOT / "testing" / "en_a_four_unit_physical_signal_capture.py"),
            sha256(PACKET_ROOT / "source" / "en_a_four_unit_physical_signal_capture.py"),
        )

    def test_service_resources_and_single_run_output_are_preserved(self) -> None:
        old = (LIVE_ROOT / "testing" / "en_a_four_unit_physical_signal_capture_approved_v2.service").read_text()
        new = (PACKET_ROOT / "source" / "en_a_four_unit_physical_signal_capture_approved_v3.service").read_text()
        required = (
            "CPUQuota=400%", "MemoryMax=2G", "MemorySwapMax=0", "RuntimeMaxSec=300",
            "TimeoutStartSec=5", "TimeoutStopSec=5", "KillMode=control-group", "OOMPolicy=stop",
        )
        for line in required:
            self.assertIn(line, old)
            self.assertIn(line, new)
        self.assertIn("en_a_four_unit_physical_signal_capture_20261001_approved_v2.launch.log", new)
        self.assertIn("Restart=no", new)

    def test_packet_member_mismatch_refuses_even_when_manifest_binding_is_updated(self) -> None:
        fixture = Fixture()
        try:
            lines = fixture.paths["packet_manifest"].read_text().splitlines()
            lines[0] = "f" * 64 + "  source/en_a_four_unit_physical_signal_capture.approved.v3.json"
            fixture.paths["packet_manifest"].write_text("\n".join(lines) + "\n")
            fixture.value["bindings"]["packet_manifest"]["sha256"] = sha256(fixture.paths["packet_manifest"])
            fixture.write_receipt()
            self.assertNotEqual(fixture.run().returncode, 0)
            fixture.assert_never_launched(self)
        finally:
            fixture.close()

    def test_each_bound_hash_mismatch_refuses(self) -> None:
        for name in BINDING_NAMES:
            with self.subTest(name=name):
                fixture = Fixture()
                try:
                    fixture.value["bindings"][name]["sha256"] = "f" * 64
                    fixture.write_receipt()
                    self.assertNotEqual(fixture.run().returncode, 0)
                    fixture.assert_never_launched(self)
                finally:
                    fixture.close()

    def test_each_bound_artifact_mutation_refuses(self) -> None:
        for name in BINDING_NAMES:
            with self.subTest(name=name):
                fixture = Fixture()
                try:
                    with fixture.paths[name].open("a") as stream:
                        stream.write("# mutation\n")
                    self.assertNotEqual(fixture.run().returncode, 0)
                    fixture.assert_never_launched(self)
                finally:
                    fixture.close()

    def test_launch_argv_mismatch_refuses(self) -> None:
        fixture = Fixture()
        try:
            fixture.value["launch"]["argv"] = [*fixture.launch, "extra"]
            fixture.value["launch"]["argv_sha256"] = argv_sha256(fixture.value["launch"]["argv"])
            fixture.write_receipt()
            self.assertNotEqual(fixture.run().returncode, 0)
            fixture.assert_never_launched(self)
        finally:
            fixture.close()

    def test_exact_positive_invokes_test_double_once(self) -> None:
        fixture = Fixture()
        try:
            completed = fixture.run()
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(fixture.counter.read_text(), "1")
        finally:
            fixture.close()

    def test_validate_only_preflight_reads_no_recording_and_invokes_no_child(self) -> None:
        fixture = Fixture()
        try:
            completed = fixture.run(validate_only=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            value = json.loads(completed.stdout)
            self.assertEqual(value["recording_open_attempts"], 0)
            self.assertEqual(value["recording_reads"], 0)
            self.assertEqual(value["recording_bytes_read"], 0)
            self.assertEqual(value["child_invocations"], 0)
            fixture.assert_never_launched(self)
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main()
