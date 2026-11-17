"""Vendor readers against synthetic plates whose pixel values encode their coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tifffile

from hcs_studio.io import nikon
from hcs_studio.io.base import ReaderError, detect_reader, open_plate
from hcs_studio.io.generic import TiffFolderReader
from hcs_studio.io.sidecar import apply_overrides, load_overrides, save_overrides, sidecar_path
from tests.builders import SHAPE, FakeND2, make_imagexpress, make_operetta, plane


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
    }


# ---------------------------------------------------------------- Operetta


@pytest.mark.parametrize("harmony_v5", [False, True])
def test_operetta_plate(tmp_path: Path, harmony_v5: bool) -> None:
    root = make_operetta(tmp_path / "MyPlate", [(1, 2), (3, 10)], harmony_v5=harmony_v5)
    before = _snapshot(root)
    assert detect_reader(root) is not None and detect_reader(root).vendor == "Operetta"  # type: ignore[union-attr]
    plate = open_plate(root)

    assert (plate.name, plate.barcode, plate.rows, plate.cols) == (
        "MyPlate",
        "BARCODE-0042",
        16,
        24,
    )
    assert sorted(plate.wells) == [(1, 2), (3, 10)]
    assert plate.well("B03").name == "B03"
    assert plate.shape_tczyx == (1, 2, 3, *SHAPE)
    meta = plate.metadata
    assert meta.pixel_size_um == pytest.approx(0.29901)
    assert meta.z_step_um == pytest.approx(2.0)
    assert (meta.objective_na, meta.magnification, meta.modality) == (0.8, 20, "confocal")
    assert [c.name for c in meta.channels] == ["HOECHST 33342", "Alexa 488"]
    assert meta.channels[1].excitation_nm == 488 and meta.channels[1].emission_nm == 522
    assert meta.channels[1].exposure_ms == pytest.approx(200)

    field = plate.wells[(3, 10)].fields[1]
    assert (
        field.index == 2 and field.x_um == pytest.approx(500) and field.y_um == pytest.approx(-250)
    )
    data = field.load()
    assert data.shape == (1, 2, 3, *SHAPE) and data.dtype == np.uint16
    for c in range(2):
        for z in range(3):
            assert np.array_equal(data[0, c, z].compute(), plane(3, 10, 1, c, z))
    assert _snapshot(root) == before, "reader modified the raw folder"


def test_operetta_missing_plane_reads_as_zeros_and_bad_state_is_skipped(tmp_path: Path) -> None:
    root = make_operetta(tmp_path / "P", [(0, 0)], n_fields=1, n_planes=2)
    index = root / "Images" / "Index.idx.xml"
    text = index.read_text(encoding="utf-8")
    # Mark the first image (channel 1, plane 1) as failed.
    index.write_text(text.replace("<State>Ok</State>", "<State>Error</State>", 1), encoding="utf-8")
    data = open_plate(root).wells[(0, 0)].fields[0].load()
    assert not data[0, 0, 0].compute().any()
    assert np.array_equal(data[0, 1, 1].compute(), plane(0, 0, 0, 1, 1))


def test_operetta_rejects_unknown_unit(tmp_path: Path) -> None:
    root = make_operetta(tmp_path / "P", [(0, 0)], n_fields=1, n_planes=1)
    index = root / "Images" / "Index.idx.xml"
    index.write_text(
        index.read_text(encoding="utf-8").replace(
            '<ImageResolutionX Unit="m">', '<ImageResolutionX Unit="furlong">'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ReaderError, match="furlong"):
        open_plate(root)


# ------------------------------------------------------------- ImageXpress


def test_imagexpress_plate(tmp_path: Path) -> None:
    root = make_imagexpress(tmp_path / "ixm", [(0, 0), (7, 11)])
    before = _snapshot(root)
    assert detect_reader(root).vendor == "ImageXpress"  # type: ignore[union-attr]
    plate = open_plate(root)

    assert (plate.name, plate.rows, plate.cols, plate.barcode) == ("Plate1", 8, 12, "Screen 7")
    assert sorted(plate.wells) == [(0, 0), (7, 11)]
    assert plate.shape_tczyx == (1, 2, 1, *SHAPE)
    meta = plate.metadata
    assert meta.pixel_size_um == pytest.approx(0.65)
    assert (meta.objective_na, meta.magnification, meta.bit_depth) == (0.75, 20, 16)
    assert [(c.name, c.emission_nm, c.exposure_ms) for c in meta.channels] == [
        ("DAPI", 447, 150),
        ("FITC", 520, 150),
    ]
    assert meta.modality == "unknown" and any("modality" in n for n in plate.notes)

    well = plate.wells[(7, 11)]
    assert [f.index for f in well.fields] == [1, 2]
    assert well.fields[1].x_um == pytest.approx(SHAPE[1] * 0.65)
    data = well.fields[1].load()
    for c in range(2):
        assert np.array_equal(data[0, c, 0].compute(), plane(7, 11, 1, c))  # thumbnails ignored
    assert _snapshot(root) == before


def test_imagexpress_z_series_ignores_projection(tmp_path: Path) -> None:
    root = make_imagexpress(tmp_path / "ixz", [(2, 3)], n_sites=1, z_steps=3)
    plate = open_plate(root)
    assert plate.shape_tczyx == (1, 2, 3, *SHAPE)
    data = plate.wells[(2, 3)].fields[0].load()
    for z in range(3):
        assert np.array_equal(data[0, 1, z].compute(), plane(2, 3, 0, 1, z))
    assert any("projection" in n for n in plate.notes)


def test_imagexpress_undeclared_wavelength_is_an_error(tmp_path: Path) -> None:
    root = make_imagexpress(tmp_path / "ix", [(0, 0)], n_sites=1)
    tifffile.imwrite(root / "Plate1_A01_s1_w3.tif", np.zeros(SHAPE, dtype=np.uint16))
    with pytest.raises(ReaderError, match="wavelength 3"):
        open_plate(root)


# ------------------------------------------------------------------- Nikon


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("WellB02_ChannelDAPI,Cy3_Seq0001", (1, 1)),
        ("B02", (1, 1)),
        ("C7_0003", (2, 6)),
        ("plate_AF48.x", (31, 47)),
        ("PLA_Day3_T01_Z03", None),
        ("Seq0004", None),
        ("", None),
    ],
)
def test_well_from_text(text: str, expected: tuple[int, int] | None) -> None:
    assert nikon.well_from_text(text) == expected


def _fake_nd2_files(monkeypatch: pytest.MonkeyPatch, files: dict[str, FakeND2]) -> None:
    monkeypatch.setattr(nikon, "_open_nd2", lambda path: files[Path(path).name])


def test_nikon_single_file_with_well_positions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # nd2 axis order P, Z, C, Y, X; value encodes (p, z, c).
    data = np.zeros((3, 4, 2, 6, 8), dtype=np.uint16)
    for p in range(3):
        for z in range(4):
            for c in range(2):
                data[p, z, c] = p * 100 + z * 10 + c
    path = tmp_path / "plate.nd2"
    path.write_bytes(b"")
    sizes = {"P": 3, "Z": 4, "C": 2, "Y": 6, "X": 8}
    _fake_nd2_files(monkeypatch, {"plate.nd2": FakeND2(sizes, ["B02", "B02", "C05"], data)})

    plate = open_plate(path)
    assert plate.vendor == "Nikon" and plate.shape_tczyx == (1, 2, 4, 6, 8)
    assert sorted(plate.wells) == [(1, 1), (2, 4)]
    assert [f.index for f in plate.wells[(1, 1)].fields] == [1, 2]
    assert plate.wells[(1, 1)].fields[1].x_um == 100.0
    meta = plate.metadata
    assert (meta.pixel_size_um, meta.z_step_um) == (0.108, 0.3)
    assert (meta.objective_na, meta.immersion_ri, meta.magnification) == (1.4, 1.515, 60.0)
    assert (meta.modality, meta.pinhole_um, meta.bit_depth) == ("confocal", 50.0, 12)
    assert [c.name for c in meta.channels] == ["DAPI", "PLA Cy3"]

    image = plate.wells[(2, 4)].fields[0].load()  # position index 2
    assert image.shape == (1, 2, 4, 6, 8)
    assert int(image[0, 1, 3, 0, 0].compute()) == 2 * 100 + 3 * 10 + 1


def test_nikon_folder_of_well_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = {"C": 2, "Y": 6, "X": 8}
    fakes = {}
    for i, name in enumerate(["WellA01_Seq0000.nd2", "WellA01_Seq0001.nd2", "WellD12_Seq0002.nd2"]):
        (tmp_path / name).write_bytes(b"")
        fakes[name] = FakeND2(sizes, [], np.full((2, 6, 8), i, dtype=np.uint16))
    _fake_nd2_files(monkeypatch, fakes)

    plate = open_plate(tmp_path)
    assert sorted(plate.wells) == [(0, 0), (3, 11)]
    assert (plate.rows, plate.cols) == (8, 12)
    assert len(plate.wells[(0, 0)].fields) == 2
    assert plate.metadata.z_step_um is None  # no Z axis
    assert int(plate.wells[(3, 11)].fields[0].load()[0, 0, 0, 0, 0].compute()) == 2
    assert not plate.notes


def test_nikon_files_without_well_names_are_flagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sizes = {"Y": 6, "X": 8}
    fakes = {}
    for name in ["ctrl.nd2", "treated.nd2"]:
        (tmp_path / name).write_bytes(b"")
        fakes[name] = FakeND2(sizes, [], np.zeros((6, 8), dtype=np.uint16))
    _fake_nd2_files(monkeypatch, fakes)
    plate = open_plate(tmp_path)
    assert [w.label for w in plate.wells.values()] == ["ctrl", "treated"]
    assert any("No well names" in n for n in plate.notes)
    assert [c.name for c in plate.metadata.channels] == ["DAPI"]


def test_nikon_rgb_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "rgb.nd2").write_bytes(b"")
    fake = FakeND2({"Y": 6, "X": 8, "S": 3}, [], np.zeros((6, 8, 3), dtype=np.uint8))
    _fake_nd2_files(monkeypatch, {"rgb.nd2": fake})
    with pytest.raises(ReaderError, match="RGB"):
        open_plate(tmp_path)


# ----------------------------------------------------------------- Generic


def test_ome_tiff(tmp_path: Path) -> None:
    data = np.arange(2 * 3 * 6 * 8, dtype=np.uint16).reshape(2, 3, 6, 8)
    path = tmp_path / "stack.ome.tif"
    tifffile.imwrite(
        path,
        data,
        metadata={
            "axes": "CZYX",
            "PhysicalSizeX": 0.2,
            "PhysicalSizeY": 0.2,
            "PhysicalSizeZ": 1.5,
            "Channel": {"Name": ["DNA", "Actin"]},
        },
    )
    plate = open_plate(path)
    assert plate.vendor == "OME-TIFF" and plate.shape_tczyx == (1, 2, 3, 6, 8)
    assert (plate.metadata.pixel_size_um, plate.metadata.z_step_um) == (0.2, 1.5)
    assert [c.name for c in plate.metadata.channels] == ["DNA", "Actin"]
    assert np.array_equal(plate.wells[(0, 0)].fields[0].load()[0].compute(), data)


def test_tiff_folder_with_pattern(tmp_path: Path) -> None:
    for well, (row, col) in {"A01": (0, 0), "B03": (1, 2)}.items():
        for f in range(2):
            for c in range(2):
                tifffile.imwrite(tmp_path / f"{well}_f{f + 1}_c{c + 1}.tif", plane(row, col, f, c))
    (tmp_path / "notes.txt").write_text("ignored")
    assert detect_reader(tmp_path).vendor == "TIFF"  # type: ignore[union-attr]
    plate = open_plate(tmp_path)
    assert sorted(plate.wells) == [(0, 0), (1, 2)] and plate.shape_tczyx == (1, 2, 1, *SHAPE)
    assert np.array_equal(
        plate.wells[(1, 2)].fields[1].load()[0, 1, 0].compute(), plane(1, 2, 1, 1)
    )

    custom = TiffFolderReader.read_with_pattern(
        tmp_path, r"(?P<well>[A-Z]\d+)_f(?P<field>\d)_c1", pixel_size_um=0.5
    )
    assert custom.shape_tczyx == (1, 1, 1, *SHAPE) and custom.metadata.pixel_size_um == 0.5
    assert any("4 TIFF file(s) did not match" in n for n in custom.notes)
    with pytest.raises(ReaderError, match="'well' group"):
        TiffFolderReader.read_with_pattern(tmp_path, r"_f(?P<field>\d)")
    with pytest.raises(ReaderError, match="matches none"):
        TiffFolderReader.read_with_pattern(tmp_path, r"(?P<well>Q\d\d)")


def test_ome_zarr_plate(tmp_path: Path) -> None:
    import zarr

    root: Any = zarr.open_group(str(tmp_path / "p.zarr"), mode="w", zarr_format=2)
    root.attrs["plate"] = {
        "name": "zplate",
        "rows": [{"name": r} for r in "ABCDEFGH"],
        "columns": [{"name": str(c)} for c in range(1, 13)],
        "wells": [{"path": "B/3", "rowIndex": 1, "columnIndex": 2}],
    }
    well = root.create_group("B/3")
    well.attrs["well"] = {"images": [{"path": "0"}]}
    image = well.create_group("0")
    image.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "nanometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1, 0.5, 500]}],
                }
            ],
        }
    ]
    image.attrs["omero"] = {"channels": [{"label": "DAPI"}, {"label": "GFP"}]}
    data = np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8)
    image.create_array("0", data=data)

    plate = open_plate(tmp_path / "p.zarr")
    assert (plate.vendor, plate.name, plate.rows, plate.cols) == ("OME-Zarr", "zplate", 8, 12)
    assert plate.metadata.pixel_size_um == pytest.approx(0.5)
    assert [c.name for c in plate.metadata.channels] == ["DAPI", "GFP"]
    assert np.array_equal(plate.well("B03").fields[0].load()[0, :, 0].compute(), data)


# ------------------------------------------------- Detection and overrides


def test_unrecognised_folder(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("hello")
    assert detect_reader(tmp_path) is None
    with pytest.raises(ReaderError, match="Could not recognise"):
        open_plate(tmp_path)
    with pytest.raises(FileNotFoundError):
        open_plate(tmp_path / "missing")


def test_overrides_live_outside_the_raw_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    root = make_operetta(tmp_path / "raw" / "P", [(0, 0)], n_fields=1, n_planes=1)
    before = _snapshot(root)
    plate = open_plate(root)
    assert load_overrides(root) == {} and apply_overrides(plate) is plate

    saved = save_overrides(root, {"pixel_size_um": 0.5, "modality": "widefield"})
    assert saved == sidecar_path(root) and tmp_path / "home" in saved.parents
    assert json.loads(saved.read_text(encoding="utf-8"))["overrides"]["pixel_size_um"] == 0.5
    patched = apply_overrides(plate)
    assert (patched.metadata.pixel_size_um, patched.metadata.modality) == (0.5, "widefield")
    assert _snapshot(root) == before

    save_overrides(root, {})
    assert not saved.exists()


def test_inspect_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from hcs_studio.app import main

    root = make_imagexpress(tmp_path / "ix", [(0, 0)], n_sites=1)
    assert main(["inspect", str(root)]) == 0
    out = capsys.readouterr().out
    assert "Format: ImageXpress" in out and "Plate1.HTD" in out
    assert "Pixel size: 0.65 µm" in out and "1. DAPI" in out and "NOTE:" in out
    assert main(["inspect", str(tmp_path / "nope")]) == 0
    assert "not found" in capsys.readouterr().out
    (tmp_path / "plain").mkdir()
    main(["inspect", str(tmp_path / "plain")])
    assert "not recognised" in capsys.readouterr().out
