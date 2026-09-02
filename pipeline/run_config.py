"""Machine- and recording-specific paths for the production run sheet.

The run sheet (``SpikeGLX_ext_ref_rescue.py``) stays a human-readable, cell-by-cell
document: stage switches, safety flags, and chunk sizes are still edited in place
at the top of that file. Only the values that differ between *machines* and
*recordings* live here -- the source folder, the stream, the output and scratch
directories, the comparison sorter outputs, and the worker count.

That split is deliberate. Stage switches are a per-run decision an operator makes
while reading the run sheet; paths are host state that should not have to be
edited into tracked source every time the pipeline moves to another machine.

Resolution order, first match wins:

1. an explicit path passed to :func:`load_run_config`
2. ``$RESCUE_RUN_CONFIG``
3. ``configs/run.toml`` beside the repository root

If none exists, a placeholder configuration is returned with ``configured``
False. Loading never raises, so the run sheet remains importable (and
``build_run_plan`` remains callable) on a fresh clone with no config yet. The run
sheet refuses to execute stages in that state -- see ``require_configured``.

The file is TOML, read with the standard library's ``tomllib``. No dependency is
added, so the locked production environment and its ``uv.lock`` identity are
untouched.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "run.toml"
EXAMPLE_CONFIG_PATH = REPO_ROOT / "configs" / "example.run.toml"
ENV_VAR = "RESCUE_RUN_CONFIG"

# Obviously-wrong placeholders used when no configuration file is present. They
# are real Path objects so that plan building and imports keep working.
UNSET_DIR = Path("/UNCONFIGURED/set-configs-run-toml")

_REQUIRED = ("data_dir", "stream_id", "output_dir")
_OPTIONAL_PATHS = ("local_work_dir", "legacy_curated_output", "claim_mask_curated_output")
_KNOWN = set(_REQUIRED) | set(_OPTIONAL_PATHS) | {"n_jobs"}


@dataclass(frozen=True)
class RunConfig:
    """Resolved per-machine, per-recording settings."""

    data_dir: Path
    stream_id: str
    output_dir: Path
    local_work_dir: Path | None
    legacy_curated_output: Path | None
    claim_mask_curated_output: Path | None
    n_jobs: int
    source: Path | None
    configured: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "stream_id": self.stream_id,
            "output_dir": str(self.output_dir),
            "local_work_dir": None if self.local_work_dir is None else str(self.local_work_dir),
            "legacy_curated_output": (
                None if self.legacy_curated_output is None else str(self.legacy_curated_output)
            ),
            "claim_mask_curated_output": (
                None
                if self.claim_mask_curated_output is None
                else str(self.claim_mask_curated_output)
            ),
            "n_jobs": self.n_jobs,
            "config_source": None if self.source is None else str(self.source),
            "configured": self.configured,
        }


def _placeholder() -> RunConfig:
    return RunConfig(
        data_dir=UNSET_DIR,
        stream_id="imec0.ap",
        output_dir=UNSET_DIR,
        local_work_dir=None,
        legacy_curated_output=None,
        claim_mask_curated_output=None,
        n_jobs=1,
        source=None,
        configured=False,
    )


def resolve_config_path(explicit: Path | str | None = None) -> Path | None:
    """Return the configuration file that would be used, or ``None``."""
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env)
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    return None


def load_run_config(path: Path | str | None = None) -> RunConfig:
    """Load the run configuration. Never raises for a missing default config."""
    resolved = resolve_config_path(path)
    if resolved is None:
        return _placeholder()
    if not resolved.is_file():
        # An explicitly requested file that does not exist is an operator error.
        raise FileNotFoundError(f"run configuration not found: {resolved}")

    with resolved.open("rb") as handle:
        raw = tomllib.load(handle)
    # Accept either a flat file or a [run] table.
    data = raw.get("run", raw)

    unknown = sorted(set(data) - _KNOWN)
    if unknown:
        raise ValueError(
            f"{resolved}: unknown key(s) {unknown}; expected any of {sorted(_KNOWN)}"
        )
    missing = [key for key in _REQUIRED if not data.get(key)]
    if missing:
        raise ValueError(f"{resolved}: missing required key(s) {missing}")

    def opt(key: str) -> Path | None:
        value = data.get(key)
        return None if value in (None, "") else Path(value)

    n_jobs = data.get("n_jobs", 1)
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError(f"{resolved}: n_jobs must be a positive integer, got {n_jobs!r}")

    return RunConfig(
        data_dir=Path(data["data_dir"]),
        stream_id=str(data["stream_id"]),
        output_dir=Path(data["output_dir"]),
        local_work_dir=opt("local_work_dir"),
        legacy_curated_output=opt("legacy_curated_output"),
        claim_mask_curated_output=opt("claim_mask_curated_output"),
        n_jobs=n_jobs,
        source=resolved,
        configured=True,
    )


def _display(path: Path) -> str:
    """Repository-relative when possible, absolute otherwise.

    Never raises: this is used to build the message an operator sees on a new
    machine, and it must not fail for a path outside the repository.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def require_configured(config: RunConfig) -> RunConfig:
    """Refuse to run stages against the placeholder configuration."""
    if config.configured:
        return config
    raise RuntimeError(
        "No run configuration found. Copy "
        f"{_display(EXAMPLE_CONFIG_PATH)} to {_display(DEFAULT_CONFIG_PATH)}, "
        "edit the paths for this machine and recording, then rerun. "
        f"Alternatively set ${ENV_VAR} to a configuration file path."
    )
