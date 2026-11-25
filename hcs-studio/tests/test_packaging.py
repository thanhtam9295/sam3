"""The Windows bundle (packaging/hcs_studio.spec) carries what the app needs at run time:
scripts that another Python runs by file path, data folders, migrations."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "hcs_studio.spec"
PACKAGE = ROOT / "src" / "hcs_studio"


def _spec_list(name: str) -> list[str]:
    tree = ast.parse(SPEC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            value = ast.literal_eval(node.value)
            assert isinstance(value, list)
            return [str(v) for v in value]
    raise AssertionError(f"{name} not found in the spec")


def test_worker_scripts_are_bundled_as_files() -> None:
    from hcs_studio.segmentation import sam3_backend
    from hcs_studio.segmentation.ilastik_backend.runners import python_api

    bundled = set(_spec_list("WORKER_SCRIPTS"))
    for worker in (sam3_backend.WORKER, python_api.WORKER):
        assert worker.suffix == ".py" and worker.exists(), worker
        relative = worker.resolve().relative_to(PACKAGE.resolve()).as_posix()
        assert relative in bundled, f"{relative} is run by path: add it to WORKER_SCRIPTS"


def test_every_script_run_by_path_is_listed() -> None:
    """Modules that start themselves with ``if __name__ == "__main__"`` and are launched as
    files by another interpreter: every such worker must be in WORKER_SCRIPTS."""
    bundled = set(_spec_list("WORKER_SCRIPTS"))
    workers = {
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*_worker.py")
        if '__name__ == "__main__"' in p.read_text(encoding="utf-8")
    }
    assert workers and workers <= bundled, workers - bundled


def test_package_data_is_collected() -> None:
    packages = _spec_list("packages")
    assert "hcs_studio" in packages and "lab_events" in packages  # collect_all: data files too
    text = SPEC.read_text(encoding="utf-8")
    assert 'includes=["db/migrations/**/*"], include_py_files=True' in text
    for folder in ("output/templates", "server/static", "db/migrations"):
        assert any((PACKAGE / folder).iterdir()), folder
