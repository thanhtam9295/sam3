"""Core package: imports stay headless, registry, synthetic data, CLI."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from hcs_studio import __version__
from hcs_studio.app import main
from hcs_studio.registry import Registry
from hcs_studio.utils.synthetic import make_cell_field


def test_core_import_does_not_load_qt_or_napari() -> None:
    code = (
        "import sys, hcs_studio, hcs_studio.app, hcs_studio.utils.gpu, hcs_studio.registry;"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('PySide6', 'napari', 'qtpy')];"
        "assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_info_command_prints_version_and_gpu(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out
    assert "GPU" in out or "CPU only" in out


def test_registry_register_get_and_duplicates() -> None:
    registry: Registry[type] = Registry("module")

    @registry.register("alpha")
    class Alpha:
        pass

    assert registry.get("alpha") is Alpha
    assert registry.names() == ["alpha"]
    assert "alpha" in registry and len(registry) == 1
    with pytest.raises(ValueError, match="already registered"):
        registry.register("alpha")(Alpha)
    with pytest.raises(KeyError, match="Registered: alpha"):
        registry.get("missing")


def test_synthetic_field_is_deterministic_and_has_known_cells() -> None:
    a = make_cell_field(shape=(128, 160), n_cells=5, seed=3)
    b = make_cell_field(shape=(128, 160), n_cells=5, seed=3)
    assert a.image.shape == (2, 128, 160)
    assert a.image.dtype == np.uint16
    assert np.array_equal(a.image, b.image)
    assert a.centers_yx.shape == (5, 2)
    # Every nucleus centre is far brighter than the ~100-count background.
    background = float(np.median(make_cell_field(shape=(128, 160), n_cells=0).image[0]))
    assert 90 < background < 110
    for cy, cx in a.centers_yx:
        assert a.image[0, round(cy), round(cx)] > background + 1000


def test_synthetic_field_rejects_negative_cell_count() -> None:
    with pytest.raises(ValueError):
        make_cell_field(n_cells=-1)


def test_processing_docs_are_up_to_date() -> None:
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "gen", root / "scripts" / "gen_processing_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = (root / "docs" / "processing.md").read_text(encoding="utf-8")
    assert current == module.render(), "run: python scripts/gen_processing_docs.py"
