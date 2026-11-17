"""Data model: well naming, plate geometry, metadata overrides, layout import."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from hcs_studio.model.layout import read_plate_layout
from hcs_studio.model.plate import (
    ChannelInfo,
    ImageMetadata,
    infer_plate_shape,
    parse_well_name,
    well_name,
)


def test_well_names() -> None:
    assert well_name(0, 0) == "A01"
    assert well_name(2, 6) == "C07"
    assert well_name(15, 23) == "P24"
    assert well_name(31, 47) == "AF48"  # last well of a 1536 plate
    assert parse_well_name("C7") == (2, 6)
    assert parse_well_name(" c07 ") == (2, 6)
    assert parse_well_name("AA01") == (26, 0)
    for bad in ("", "7C", "C00", "C", "ABC1", "C100"):
        with pytest.raises(ValueError):
            parse_well_name(bad)


@given(st.integers(0, 31), st.integers(0, 47))
def test_well_name_round_trip(row: int, col: int) -> None:
    assert parse_well_name(well_name(row, col)) == (row, col)


def test_infer_plate_shape() -> None:
    assert infer_plate_shape(1, 1) == (2, 3)
    assert infer_plate_shape(8, 12) == (8, 12)
    assert infer_plate_shape(9, 12) == (16, 24)
    assert infer_plate_shape(16, 25) == (32, 48)
    assert infer_plate_shape(40, 60) == (40, 60)


def test_metadata_overrides() -> None:
    meta = ImageMetadata(channels=(ChannelInfo("a"), ChannelInfo("b")), pixel_size_um=0.3)
    changed = meta.with_overrides({"pixel_size_um": 0.65, "channel_names": ["DAPI", "GFP"]})
    assert changed.pixel_size_um == 0.65
    assert [c.name for c in changed.channels] == ["DAPI", "GFP"]
    assert meta.pixel_size_um == 0.3  # original untouched
    with pytest.raises(ValueError):
        meta.with_overrides({"bit_depth": 8})
    with pytest.raises(ValueError):
        meta.with_overrides({"channel_names": ["only one"]})


def test_plate_layout_csv(tmp_path: Path) -> None:
    path = tmp_path / "layout.csv"
    path.write_text(
        "Well,Compound,Concentration,Treatment,Control Type\n"
        "A01,DMSO,,vehicle,negative\n"
        "b2,Olaparib,1.5,24h,\n",
        encoding="utf-8",
    )
    layout = read_plate_layout(path)
    assert layout[(0, 0)].control_type == "negative" and layout[(0, 0)].concentration is None
    assert layout[(1, 1)].compound == "Olaparib" and layout[(1, 1)].concentration == 1.5


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("compound\nDMSO\n", "'well' column"),
        ("well\nA01\nA1\n", "twice"),
        ("well,concentration\nA01,high\n", "not a number"),
        ("well,control_type\nA01,maybe\n", "control_type"),
        ("well\n7C\n", "line 2"),
    ],
)
def test_plate_layout_errors(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "layout.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_plate_layout(path)
