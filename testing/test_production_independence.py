"""The production package must not depend on the legacy ``pipelineold`` package.

``pipelineold`` stays in the research repository for the legacy run sheets and
the retired curation strategies. Production curation, QC, refractory,
truncation, and Kilosort-result access were extracted into ``pipeline`` so the
production pipeline can be ported without carrying the legacy package.

These tests fail loudly if that boundary is crossed again.
"""

import ast
import importlib
import pathlib
import sys

import pytest

PIPELINE_DIR = pathlib.Path(__file__).resolve().parents[1] / "pipeline"

PRODUCTION_MODULES = [
    "pipeline",
    "pipeline.curation",
    "pipeline.downstream",
    "pipeline.kilosort_results",
    "pipeline.qc",
    "pipeline.refractory",
    "pipeline.truncation",
]


def _imported_names(path: pathlib.Path) -> set[str]:
    """Every module name imported anywhere in ``path``, including inside functions."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "source", sorted(PIPELINE_DIR.glob("*.py")), ids=lambda p: p.name
)
def test_no_static_pipelineold_import(source):
    offenders = {
        name
        for name in _imported_names(source)
        if name == "pipelineold" or name.startswith("pipelineold.")
    }
    assert not offenders, f"{source.name} imports {sorted(offenders)}"


class _BlockPipelineold:
    """Meta path hook that makes any ``pipelineold`` import fail immediately."""

    def find_module(self, name, path=None):  # pragma: no cover - trivial
        if name == "pipelineold" or name.startswith("pipelineold."):
            raise ImportError(f"production code imported {name}")
        return None

    def find_spec(self, name, path=None, target=None):
        if name == "pipelineold" or name.startswith("pipelineold."):
            raise ImportError(f"production code imported {name}")
        return None


@pytest.mark.parametrize("module", PRODUCTION_MODULES)
def test_imports_with_pipelineold_blocked(module):
    """Importing production modules must not reach for the legacy package."""
    blocker = _BlockPipelineold()
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name == module or name.startswith("pipeline")
    }
    for name in list(saved):
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)
    try:
        importlib.import_module(module)
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_production_entry_points_are_callable():
    """The three symbols ``pipeline.downstream`` used to take from pipelineold."""
    from pipeline.curation import run_cur_final
    from pipeline.kilosort_results import KilosortResults
    from pipeline.qc import run_qc

    assert callable(run_cur_final)
    assert callable(run_qc)
    assert callable(KilosortResults)
