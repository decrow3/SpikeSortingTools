#!/usr/bin/env python3
"""Known-answer whole-stage restart test through DARTsort run_peeler."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/candidate3-stage-restart-numba")

import h5py
import numpy as np
import torch
from spikeinterface.core import NumpyRecording

from dartsort.peel.peel_base import BasePeeler
from dartsort.util.data_util import SpikeDataset
from dartsort.util.internal_config import ComputationConfig, FeaturizationConfig
from dartsort.util.peel_util import run_peeler

from testing.candidate3_stage_restart_fallback import run_fresh_stage_attempt


CHUNK_STARTS = np.array([0, 100, 200], dtype=np.int64)
EXPECTED = {
    "times_samples": np.array([10, 20, 110, 120, 210, 220], dtype=np.int64),
    "channels": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
    "template_indices": np.array([100, 101, 110, 111, 120, 121], dtype=np.int64),
    "construction_memberships": np.array([1000, 1001, 1010, 1011, 1020, 1021], dtype=np.int64),
    "source_rows": np.arange(6, dtype=np.int64),
}


class KnownAnswerPeeler(BasePeeler):
    peel_kind = "candidate3-restart-known-answer"

    def __init__(self, recording, *, inject_append_failure=False):
        super().__init__(
            recording=recording,
            channel_index=np.array([[0, 1], [0, 1]]),
            featurization_pipeline=None,
            chunk_length_samples=100,
            chunk_margin_samples=0,
        )
        self.inject_append_failure = inject_append_failure

    def out_datasets(self):
        return super().out_datasets() + [
            SpikeDataset("template_indices", (), np.int64),
            SpikeDataset("construction_memberships", (), np.int64),
            SpikeDataset("source_rows", (), np.int64),
        ]

    def peel_chunk(
        self,
        traces,
        *,
        chunk_start_samples=0,
        left_margin=0,
        right_margin=0,
        return_residual=False,
        return_waveforms=True,
    ):
        del traces, left_margin, right_margin, return_waveforms
        chunk = chunk_start_samples // 100
        result = {
            "n_spikes": 2,
            "times_samples": torch.tensor(
                [chunk_start_samples + 10, chunk_start_samples + 20], dtype=torch.int64
            ),
            "channels": torch.tensor([0, 1], dtype=torch.int64),
            "template_indices": torch.tensor([100 + 10 * chunk, 101 + 10 * chunk]),
            "construction_memberships": torch.tensor(
                [1000 + 10 * chunk, 1001 + 10 * chunk]
            ),
            "source_rows": torch.tensor([2 * chunk, 2 * chunk + 1]),
        }
        if self.inject_append_failure and chunk_start_samples == 100:
            # The core writer has already advanced its chunk marker and written
            # earlier datasets when this incompatible final dataset is appended.
            result["source_rows"] = torch.tensor([[2, 3], [4, 5]])
        if return_residual:
            result["residual"] = torch.zeros((100, 2), dtype=torch.float32)
        return result


def make_recording():
    traces = np.zeros((300, 2), dtype=np.float32)
    rec = NumpyRecording(traces, sampling_frequency=1000.0)
    rec.set_channel_locations(np.array([[0.0, 0.0], [20.0, 0.0]]))
    return rec, hashlib.sha256(traces.tobytes()).hexdigest()


def run_actual_path(attempt_dir: Path, *, inject_append_failure: bool) -> Path:
    rec, _ = make_recording()
    peeler = KnownAnswerPeeler(rec, inject_append_failure=inject_append_failure)
    return run_peeler(
        peeler,
        output_directory=attempt_dir,
        hdf5_filename="events.h5",
        model_subdir="models",
        featurization_cfg=FeaturizationConfig(skip=True, do_localization=False),
        computation_cfg=ComputationConfig(
            n_jobs_cpu=0, n_jobs_gpu=0, n_jobs_small=0, device="cpu"
        ),
        chunk_starts_samples=CHUNK_STARTS,
        overwrite=False,
        skip_resid_snips=True,
        show_progress=False,
    )


def read_rows(path: Path):
    with h5py.File(path, "r") as h5:
        return {name: h5[name][:] for name in EXPECTED} | {
            "times_seconds": h5["times_seconds"][:],
            "last_chunk_index": int(h5["last_chunk_index"][()]),
            "last_chunk_start": int(h5["last_chunk_start"][()]),
        }


def validate(path: Path):
    rows = read_rows(path)
    exact = {name: bool(np.array_equal(rows[name], expected)) for name, expected in EXPECTED.items()}
    return {
        "valid": bool(
            all(exact.values())
            and rows["last_chunk_index"] == 2
            and rows["last_chunk_start"] == 200
            and len(np.unique(rows["source_rows"])) == 6
        ),
        "dataset_exact": exact,
        "event_rows": int(len(rows["times_samples"])),
        "last_chunk_index": rows["last_chunk_index"],
        "last_chunk_start": rows["last_chunk_start"],
        "source_rows_unique": int(len(np.unique(rows["source_rows"]))),
    }


def main(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    _, recording_sha = make_recording()
    bindings = {
        "chunk_starts": CHUNK_STARTS.tolist(),
        "recording_sha256": recording_sha,
        "source": "dartsort.util.peel_util.run_peeler",
    }

    failure_observed = False
    try:
        run_fresh_stage_attempt(
            attempts_directory=output / "attempts",
            attempt_id="injected",
            input_bindings=bindings,
            run_stage=lambda d: run_actual_path(d, inject_append_failure=True),
            validate_stage=validate,
        )
    except (RuntimeError, TypeError, ValueError):
        failure_observed = True

    failed_path = output / "attempts" / "injected.failed"
    with h5py.File(failed_path / "events.h5", "r") as h5:
        failed_state = {
            "last_chunk_index": int(h5["last_chunk_index"][()]),
            "last_chunk_start": int(h5["last_chunk_start"][()]),
            "dataset_lengths": {
                name: int(len(h5[name]))
                for name in ("times_samples", "channels", "template_indices", "construction_memberships", "source_rows")
            },
        }

    restarted = run_fresh_stage_attempt(
        attempts_directory=output / "attempts",
        attempt_id="restart",
        input_bindings=bindings,
        run_stage=lambda d: run_actual_path(d, inject_append_failure=False),
        validate_stage=validate,
    )
    baseline = run_fresh_stage_attempt(
        attempts_directory=output / "attempts",
        attempt_id="baseline",
        input_bindings=bindings,
        run_stage=lambda d: run_actual_path(d, inject_append_failure=False),
        validate_stage=validate,
    )
    restart_rows = read_rows(restarted)
    baseline_rows = read_rows(baseline)
    equality = {
        name: bool(np.array_equal(restart_rows[name], baseline_rows[name]))
        for name in (*EXPECTED, "times_seconds")
    }
    result = {
        "failure_observed": failure_observed,
        "failed_attempt_preserved": failed_path.is_dir(),
        "failed_state": failed_state,
        "input_bindings_unchanged": json.loads((failed_path / "ATTEMPT.json").read_text())["input_bindings"]
        == json.loads((restarted.parent / "ATTEMPT.json").read_text())["input_bindings"],
        "restart_validation": validate(restarted),
        "restart_equals_baseline": equality,
        "no_gaps_or_duplicates": bool(np.array_equal(restart_rows["source_rows"], np.arange(6))),
    }
    result["verdict"] = (
        "GO_WHOLE_STAGE_RESTART_FALLBACK"
        if failure_observed
        and result["failed_attempt_preserved"]
        and result["input_bindings_unchanged"]
        and result["restart_validation"]["valid"]
        and all(equality.values())
        and result["no_gaps_or_duplicates"]
        else "BLOCKED"
    )
    (output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if result["verdict"] != "GO_WHOLE_STAGE_RESTART_FALLBACK":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
