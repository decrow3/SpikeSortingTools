from pathlib import Path

import numpy as np
import pytest

from testing.joint_model_chunk_starts_entry import run_selected_chunks


def test_thin_entry_exact_delegation_and_fail_closed(tmp_path):
    calls = []

    def existing_runner(**kwargs):
        calls.append(kwargs)
        return "runner-result"

    output = tmp_path / "create-only-output"
    result = run_selected_chunks(
        [217620000, 217650000],
        {"segment_ownership": True, "pulse_family": False},
        output,
        c4_continuation_runner=existing_runner,
    )
    assert result == "runner-result"
    assert len(calls) == 1
    assert np.array_equal(calls[0]["chunk_starts_samples"], [217620000, 217650000])
    assert calls[0]["feature_switches"] == {
        "segment_ownership": True,
        "pulse_family": False,
    }
    assert calls[0]["output_dir"] == Path(output)

    output.mkdir()
    with pytest.raises(FileExistsError):
        run_selected_chunks(
            [217620000],
            {"segment_ownership": False, "pulse_family": False},
            output,
            c4_continuation_runner=existing_runner,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        run_selected_chunks(
            [217620000, 217620000],
            {"segment_ownership": False, "pulse_family": False},
            tmp_path / "unused",
            c4_continuation_runner=existing_runner,
        )
    assert len(calls) == 1
