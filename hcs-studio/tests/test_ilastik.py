"""ilastik backend, step (a): axis order, HDF5 files, settings and detection.

The tests marked ``real_ilastik`` run ilastik itself (``HCS_ILASTIK=<ilastik program or the
Python of an ilastik environment>``) and are skipped without it.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hcs_studio.segmentation.ilastik_backend import axes as ax
from hcs_studio.segmentation.ilastik_backend import h5io
from hcs_studio.segmentation.ilastik_backend.settings import (
    INCOMPATIBLE,
    IlastikSettings,
    check_version,
    detect,
    fiji_root,
    find_version,
    ilastik_candidates,
    ilastik_command,
    ilastik_version,
    version_key,
)

REAL_ILASTIK = os.environ.get("HCS_ILASTIK", "")
real_ilastik = pytest.mark.skipif(
    not REAL_ILASTIK, reason="set HCS_ILASTIK to an ilastik program to run ilastik itself"
)


def tczyx(t: int, c: int, z: int, y: int = 6, x: int = 7) -> np.ndarray:
    """Values encode their (t, c, z, y, x) position, so any mix-up shows."""
    grid = np.indices((t, c, z, y, x))
    values = grid[0] * 10000 + grid[1] * 1000 + grid[2] * 100 + grid[3] * 10 + grid[4]
    return np.asarray(values, dtype=np.int32)


# --------------------------------------------------------------------- axes


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((1, 1, 1), "yxc"),  # 2D single channel
        ((1, 3, 1), "yxc"),  # 2D multichannel
        ((1, 2, 5), "zyxc"),  # 3D multichannel
        ((4, 1, 1), "tyxc"),  # 2D + time
        ((4, 2, 3), "tzyxc"),  # 3D + time
    ],
)
def test_axes_round_trip(shape: tuple[int, int, int], expected: str) -> None:
    data = tczyx(*shape)
    axes = ax.ilastik_axes(data.shape)
    assert axes == expected
    out = ax.to_ilastik(data, axes)
    assert out.shape == ax.shape_in(axes, data.shape)
    # position check: element [t, c, z, y, x] lands at the ilastik index of those letters
    t, c, z = (n - 1 for n in shape)
    index = {"t": t, "c": c, "z": z, "y": 5, "x": 6}
    assert out[tuple(index[a] for a in axes)] == data[t, c, z, 5, 6]
    assert np.array_equal(ax.from_ilastik(out, axes), data)


def test_other_orders_and_missing_channel_axis() -> None:
    data = tczyx(1, 1, 1)
    xy = ax.to_ilastik(data, "xy")  # a single-channel project trained without c
    assert xy.shape == (7, 6) and xy[6, 5] == data[0, 0, 0, 5, 6]
    assert np.array_equal(ax.from_ilastik(xy, "xy"), data)
    cyx = ax.to_ilastik(tczyx(1, 2, 1), "cyx")
    assert cyx.shape == (2, 6, 7) and cyx[1, 2, 3] == 1023
    assert ax.ilastik_axes((1, 1, 1, 6, 7), channel=False) == "yx"
    assert ax.ilastik_axes((1, 1, 1, 6, 7), z=True, time=True) == "tzyxc"
    assert ax.describe("tzyxc") == "3D + time" and ax.describe("yxc") == "2D"


def test_axis_mismatches_have_clear_messages() -> None:
    with pytest.raises(ValueError, match="Z projection"):
        ax.ilastik_axes((1, 1, 5, 6, 7), z=False)
    with pytest.raises(ValueError, match="5 Z planes"):
        ax.to_ilastik(tczyx(1, 1, 5), "yxc")
    with pytest.raises(ValueError, match="2 channels"):
        ax.to_ilastik(tczyx(1, 2, 1), "yx")
    with pytest.raises(ValueError, match="repeats"):
        ax.check_axes("yyx")
    with pytest.raises(ValueError, match="unknown axis"):
        ax.check_axes("yxq")
    with pytest.raises(ValueError, match="needs both"):
        ax.check_axes("zc")
    with pytest.raises(ValueError, match="3 letters"):
        ax.from_ilastik(np.zeros((2, 2)), "yxc")


# ---------------------------------------------------------------------- HDF5


def test_h5_round_trip_with_ilastik_axistags(tmp_path: Path) -> None:
    data = ax.to_ilastik(tczyx(3, 2, 4, 300, 20).astype(np.uint16), "tzyxc")
    path = h5io.write_h5(tmp_path / "in.h5", data, "tzyxc", compression=4)
    assert not list(tmp_path.glob("*.partial"))
    import h5py

    with h5py.File(path) as f:
        ds = f["data"]
        tags = json.loads(ds.attrs["axistags"])
        assert ds.chunks == (1, 4, 256, 20, 2) and ds.compression == "gzip"
    # exactly the form ilastik 1.4.2 writes itself
    assert tags["axes"][0] == {"key": "t", "typeFlags": 8, "resolution": 0, "description": ""}
    assert [a["typeFlags"] for a in tags["axes"]] == [8, 2, 2, 2, 1]
    back, axes = h5io.read_h5(path)
    assert axes == "tzyxc" and back.dtype == np.uint16 and np.array_equal(back, data)
    assert h5io.input_spec(path) == f"{path}/data"
    assert h5io.is_ilastik_ready(path) and h5io.read_axes(path) == "tzyxc"
    with pytest.raises(ValueError, match="expected"):
        h5io.read_h5(path, axes="zyxc")


def test_h5_reads_ilastik_exports_and_untagged_files(tmp_path: Path) -> None:
    import h5py

    export = tmp_path / "probabilities.h5"
    with h5py.File(export, "w") as f:
        ds = f.create_dataset("exported_data", data=np.ones((5, 6, 2), np.float32))
        ds.attrs["axistags"] = h5io.axistags_json("yxc")
        f.create_dataset("other", data=np.zeros(3))
    array, axes = h5io.read_h5(export)  # prefers exported_data
    assert axes == "yxc" and array.shape == (5, 6, 2)
    plain = tmp_path / "plain.h5"
    with h5py.File(plain, "w") as f:
        f.create_dataset("seg", data=np.zeros((4, 4), np.uint32))
    assert not h5io.is_ilastik_ready(plain)
    with pytest.raises(ValueError, match="no axistags"):
        h5io.read_h5(plain)
    assert h5io.read_h5(plain, axes="yx")[1] == "yx"
    with pytest.raises(ValueError, match="not match"):
        h5io.write_h5(tmp_path / "bad.h5", np.zeros((2, 2)), "yxc")
    with pytest.raises(ValueError, match="Not a valid axistags"):
        h5io.parse_axistags("{}")


# ------------------------------------------------------------------ versions


def test_versions_and_blocked_releases() -> None:
    assert version_key("1.3.3") < version_key("1.3.3post3") < version_key("1.4.0b27")
    assert version_key("1.4.0b27") < version_key("1.4.0rc1") < version_key("1.4.0")
    assert version_key("1.4.2") < version_key("1.5.0a1") < version_key("1.5.0b1")
    assert find_version("ilastik-1.4.2-Linux") == "1.4.2"
    for version in INCOMPATIBLE:
        check = check_version(version)
        assert check.level == "error" and "not compatible" in check.message, version
    assert check_version("1.3.3post3").level == "warning"  # allowed, but old
    assert check_version("1.3.2").level == "error"
    assert check_version("1.4.2").level == "ok"
    assert check_version("1.5.0b1").level == "warning"  # pre-release
    assert check_version(None).level == "warning"


# ------------------------------------------------------------------ settings


def test_settings_save_load_validate_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    assert IlastikSettings.load() == IlastikSettings()
    assert IlastikSettings().environment() == {}  # ilastik's own defaults
    saved = IlastikSettings(ilastik="/x/run_ilastik.sh", threads=4, ram_mb=8000)
    saved.save()
    assert IlastikSettings.load() == saved
    assert saved.environment() == {"LAZYFLOW_THREADS": "4", "LAZYFLOW_TOTAL_RAM_MB": "8000"}
    for bad, message in (
        (IlastikSettings(ram_mb=300), "at least 500"),
        (IlastikSettings(threads=0), "at least 1"),
        (IlastikSettings(max_processes=0), "one ilastik process"),
        (IlastikSettings(runner="api"), "Unknown runner"),  # type: ignore[arg-type]
    ):
        with pytest.raises(ValueError, match=message):
            bad.validate()
    IlastikSettings.path().write_text(
        '{"ram_mb": 12}', encoding="utf-8"
    )  # a bad file falls back to defaults
    assert IlastikSettings.load() == IlastikSettings()


# ----------------------------------------------------------------- detection


def fake_ilastik(path: Path, version: str) -> Path:
    """A stand-in program that answers ``--version`` like ilastik."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\nprint('Starting ilastik')\nprint('{version}')\n", encoding="utf-8"
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.mark.skipif(sys.platform.startswith("win"), reason="shebang scripts")
def test_detection_finds_newest_ilastik_and_fiji_with_ilastik4ij(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "settings"))
    monkeypatch.delenv("HCS_ILASTIK", raising=False)
    monkeypatch.delenv("HCS_FIJI", raising=False)
    home = tmp_path / "home"
    fake_ilastik(home / "ilastik-1.3.3post3-Linux" / "run_ilastik.sh", "1.3.3post3")
    newest = fake_ilastik(home / "ilastik-1.4.2-Linux" / "run_ilastik.sh", "1.4.2")
    assert ilastik_candidates("linux", home)[0] == newest
    assert ilastik_version(newest) == "1.4.2"
    fiji = home / "Fiji.app"
    (fiji / "jars").mkdir(parents=True)
    (fiji / "ImageJ-linux64").write_text("", encoding="utf-8")
    (fiji / "jars" / "ilastik4ij-2.0.3.jar").write_text("", encoding="utf-8")
    found = detect(IlastikSettings(), platform="linux", home=home)
    assert found.ilastik == str(newest) and found.ilastik_source == "detected"
    assert found.ilastik_version == "1.4.2" and found.ilastik_ok
    assert found.fiji == str(fiji) and found.fiji_ok and found.ilastik4ij_version == "2.0.3"
    assert found.runners() == ["headless", "fiji"]
    assert "ilastik4ij-2.0.3.jar" in found.text()
    assert fiji_root(fiji / "ImageJ-linux64") == fiji

    # settings win over detection; a blocked version is reported, not used
    blocked = fake_ilastik(tmp_path / "old" / "ilastik-1.3.3-Linux" / "run_ilastik.sh", "1.3.3")
    found = detect(IlastikSettings(ilastik=str(blocked)), platform="linux", home=home)
    assert found.ilastik_source == "settings" and not found.ilastik_ok
    assert found.runners() == [] and "ERROR" in found.text()
    # Fiji without the plugin is found but not usable
    (fiji / "jars" / "ilastik4ij-2.0.3.jar").unlink()
    found = detect(IlastikSettings(), platform="linux", home=home, query_version=False)
    assert found.fiji and not found.fiji_ok and "update sites" in found.text()


def test_python_of_an_ilastik_environment_runs_the_module() -> None:
    python, script = "/envs/ilastik/bin/python", "/opt/ilastik-1.4.2-Linux/run_ilastik.sh"
    assert ilastik_command(python) == [str(Path(python)), "-m", "ilastik"]  # native separators
    assert ilastik_command(script) == [str(Path(script))]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="shebang scripts")
def test_cli_detect_and_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "settings"))
    monkeypatch.delenv("HCS_FIJI", raising=False)
    program = fake_ilastik(tmp_path / "ilastik-1.4.2-Linux" / "run_ilastik.sh", "1.4.2")
    env = {**os.environ, "HCS_STUDIO_HOME": str(tmp_path / "settings"), "HCS_ILASTIK": ""}
    cli = [sys.executable, "-m", "hcs_studio", "ilastik", "--detect"]
    out = subprocess.run(
        [*cli, "--set-ilastik", str(program), "--threads", "2", "--save"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "OK: ilastik 1.4.2" in out.stdout and "Saved" in out.stdout
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "settings"))
    assert IlastikSettings.load().threads == 2
    bad = subprocess.run(
        [*cli, "--ram-mb", "100"], capture_output=True, encoding="utf-8", errors="replace", env=env
    )
    assert bad.returncode == 2 and "at least 500" in bad.stdout


# ------------------------------------------------------------- real ilastik


@real_ilastik
def test_real_ilastik_version_is_supported() -> None:
    version = ilastik_version(REAL_ILASTIK)
    assert version and check_version(version).ok, version


@real_ilastik
@pytest.mark.parametrize("shape", [(1, 1, 1), (1, 3, 1), (1, 2, 5), (4, 1, 1)])
def test_real_ilastik_reads_our_h5_files(shape: tuple[int, int, int], tmp_path: Path) -> None:
    """ilastik's Data Conversion workflow reads our file and writes back the same array with
    the same axes: our HDF5 and axis conventions are the ones ilastik uses."""
    command = ilastik_command(REAL_ILASTIK)
    project = tmp_path / "convert.ilp"
    made = subprocess.run(
        [*command, "--headless", f"--new_project={project}", "--workflow=Data Conversion"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        cwd=tmp_path,
    )
    assert made.returncode == 0 and project.exists(), made.stdout[-2000:] + made.stderr[-2000:]
    data = tczyx(*shape, y=40, x=30).astype(np.uint16)
    axes = ax.ilastik_axes(data.shape)
    source = h5io.write_h5(tmp_path / "in.h5", ax.to_ilastik(data, axes), axes)
    target = tmp_path / "out.h5"
    run = subprocess.run(
        [
            *command,
            "--headless",
            f"--project={project}",
            "--output_format=hdf5",
            f"--output_filename_format={target}",
            "--raw_data",
            h5io.input_spec(source),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        cwd=tmp_path,
    )
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    array, out_axes = h5io.read_h5(target)
    assert out_axes == axes
    assert np.array_equal(ax.from_ilastik(array, out_axes), data)
