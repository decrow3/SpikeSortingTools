"""Run-sheet configuration loading."""

from pathlib import Path

import pytest

from pipeline.run_config import (
    ENV_VAR,
    EXAMPLE_CONFIG_PATH,
    RunConfig,
    load_run_config,
    require_configured,
    resolve_config_path,
)

MINIMAL = """
data_dir = "/data/subject_g0"
stream_id = "imec0.ap"
output_dir = "/results/subject_imec0"
"""


def write(tmp_path, text, name="run.toml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_loads_minimal_config(tmp_path):
    config = load_run_config(write(tmp_path, MINIMAL))
    assert config.configured
    assert config.data_dir == Path("/data/subject_g0")
    assert config.stream_id == "imec0.ap"
    assert config.output_dir == Path("/results/subject_imec0")
    assert config.local_work_dir is None
    assert config.n_jobs == 1


def test_loads_full_config(tmp_path):
    config = load_run_config(
        write(
            tmp_path,
            MINIMAL
            + """
local_work_dir = "/local/nvme/work"
legacy_curated_output = "/results/legacy/cur/cur_sorter_output"
claim_mask_curated_output = "/results/patched/cur/cur_sorter_output"
n_jobs = 20
""",
        )
    )
    assert config.local_work_dir == Path("/local/nvme/work")
    assert config.legacy_curated_output == Path("/results/legacy/cur/cur_sorter_output")
    assert config.claim_mask_curated_output == Path("/results/patched/cur/cur_sorter_output")
    assert config.n_jobs == 20


def test_run_table_form_is_accepted(tmp_path):
    config = load_run_config(write(tmp_path, "[run]\n" + MINIMAL))
    assert config.configured
    assert config.data_dir == Path("/data/subject_g0")


def test_missing_required_key_is_rejected(tmp_path):
    text = MINIMAL.replace('output_dir = "/results/subject_imec0"', "")
    with pytest.raises(ValueError, match=r"missing required key\(s\) \['output_dir'\]"):
        load_run_config(write(tmp_path, text))


def test_unknown_key_is_rejected(tmp_path):
    """A typo must not be silently ignored -- it would run the wrong recording."""
    with pytest.raises(ValueError, match="unknown key"):
        load_run_config(write(tmp_path, MINIMAL + '\noutput_directory = "/typo"\n'))


@pytest.mark.parametrize("value", ["0", "-4", '"twenty"'])
def test_invalid_n_jobs_is_rejected(tmp_path, value):
    with pytest.raises(ValueError, match="n_jobs must be a positive integer"):
        load_run_config(write(tmp_path, MINIMAL + f"\nn_jobs = {value}\n"))


def test_empty_optional_path_is_none(tmp_path):
    config = load_run_config(write(tmp_path, MINIMAL + '\nlocal_work_dir = ""\n'))
    assert config.local_work_dir is None


def test_explicitly_named_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run_config(tmp_path / "absent.toml")


def test_env_var_is_honoured(tmp_path, monkeypatch):
    path = write(tmp_path, MINIMAL, name="elsewhere.toml")
    monkeypatch.setenv(ENV_VAR, str(path))
    assert resolve_config_path() == path
    assert load_run_config().data_dir == Path("/data/subject_g0")


def test_explicit_argument_beats_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(write(tmp_path, MINIMAL, name="env.toml")))
    explicit = write(
        tmp_path,
        MINIMAL.replace("/data/subject_g0", "/data/explicit_g0"),
        name="explicit.toml",
    )
    assert load_run_config(explicit).data_dir == Path("/data/explicit_g0")


def test_absent_default_config_yields_usable_placeholder(monkeypatch, tmp_path):
    """A fresh clone must still import the run sheet and build a plan."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(
        "pipeline.run_config.DEFAULT_CONFIG_PATH", tmp_path / "no" / "run.toml"
    )
    config = load_run_config()
    assert not config.configured
    assert isinstance(config.data_dir, Path)
    assert config.data_dir.resolve()  # plan building calls this
    assert config.source is None


def test_require_configured_gates_execution(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr(
        "pipeline.run_config.DEFAULT_CONFIG_PATH", tmp_path / "no" / "run.toml"
    )
    placeholder = load_run_config()
    with pytest.raises(RuntimeError, match="No run configuration found"):
        require_configured(placeholder)

    real = load_run_config(write(tmp_path, MINIMAL))
    assert require_configured(real) is real


def test_require_configured_message_survives_paths_outside_the_repo(monkeypatch, tmp_path):
    """The operator-facing message must never raise while being built."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.setattr("pipeline.run_config.DEFAULT_CONFIG_PATH", tmp_path / "run.toml")
    monkeypatch.setattr("pipeline.run_config.EXAMPLE_CONFIG_PATH", tmp_path / "example.toml")
    with pytest.raises(RuntimeError, match="No run configuration found"):
        require_configured(load_run_config())


def test_tracked_example_config_is_valid():
    """The tracked template must stay loadable, or new machines start broken."""
    assert EXAMPLE_CONFIG_PATH.is_file()
    config = load_run_config(EXAMPLE_CONFIG_PATH)
    assert config.configured
    assert config.stream_id == "imec0.ap"
    assert config.n_jobs == 20


def test_as_dict_is_json_safe():
    import json

    config = RunConfig(
        data_dir=Path("/a"),
        stream_id="imec0.ap",
        output_dir=Path("/b"),
        local_work_dir=None,
        legacy_curated_output=None,
        claim_mask_curated_output=None,
        n_jobs=4,
        source=Path("/c/run.toml"),
        configured=True,
    )
    json.dumps(config.as_dict())
    assert config.as_dict()["local_work_dir"] is None
    assert config.as_dict()["config_source"] == "/c/run.toml"
