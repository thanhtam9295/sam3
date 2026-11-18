"""Phase 11: plate annotation model, units, import/export, helpers, validation, storage."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from hcs_studio.io.layout_io import (
    annotation_rows,
    load_layout,
    save_layout,
    suggest_mapping,
    write_layout_figure,
    write_platemap_xlsx,
)
from hcs_studio.io.layout_store import (
    LayoutStore,
    list_templates,
    load_template,
    plate_key,
    save_template,
)
from hcs_studio.model import units
from hcs_studio.model.annotation import (
    AnnotationFilter,
    PlateLayout,
    Treatment,
    WellAnnotation,
)
from hcs_studio.model.layout_tools import (
    apply_template,
    combination_matrix,
    copy_region,
    dilution_series,
    dilution_values,
    edge_wells,
    exclude_edges,
    paste,
    set_role,
    validate,
)
from hcs_studio.model.plate import parse_well_name

# ------------------------------------------------------------------------ units


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (100, "nM", "µM", 0.1),
        (1, "mM", "nM", 1e6),
        (2.5, "uM", "nM", 2500),
        (10, "µg/mL", "ng/mL", 10_000),
        (0.1, "%", "%", 0.1),
        (1, "M", "pM", 1e12),
    ],
)
def test_unit_conversion(value: float, source: str, target: str, expected: float) -> None:
    assert units.convert(value, source, target) == pytest.approx(expected)


def test_molar_mass_conversion_needs_molecular_weight() -> None:
    # Staurosporine, 466.5 g/mol: 1 µM = 466.5 ng/mL
    assert units.convert(1, "µM", "ng/mL", molecular_weight=466.5) == pytest.approx(466.5)
    assert units.convert(466.5, "ng/mL", "µM", molecular_weight=466.5) == pytest.approx(1)
    with pytest.raises(ValueError, match="molecular weight"):
        units.convert(1, "µM", "ng/mL")
    with pytest.raises(ValueError, match="cannot be converted"):
        units.convert(1, "%", "µM")
    with pytest.raises(ValueError, match="Unknown concentration unit"):
        units.parse_unit("furlongs")


def test_base_units_and_display() -> None:
    t = Treatment.of("X", 100, "nM")
    assert t.concentration == 1e-7 and t.value == 100 and t.label() == "X 100 nM"
    assert Treatment.of("Y", 0.1, "%").concentration == 0.001
    assert units.best_unit(3e-8, "molar") == "nM"
    assert units.format_concentration(1.5e-6, "µM") == "1.5 µM"
    # Repeated conversion does not drift.
    value = 1 / 3
    for _ in range(20):
        value = Treatment.of("Z", value, "µM").value
    assert value == pytest.approx(1 / 3, rel=1e-11)


def test_combinations_are_order_free() -> None:
    a, b = Treatment.of("A", 1, "µM"), Treatment.of("B", 10, "nM")
    assert WellAnnotation("MCF7", (a, b)) == WellAnnotation("MCF7", (b, a))
    well = WellAnnotation("MCF7", (a, b))
    assert well.is_combination and well.replicate_group == "MCF7 · A 1 µM + B 10 nM"
    edge = WellAnnotation("MCF7", (a, Treatment.of("B", 0, "nM")))
    assert not edge.is_combination and edge.concentration == 1  # single agent on the edge
    with pytest.raises(ValueError, match="twice"):
        WellAnnotation("MCF7", (a, Treatment.of("a", 2, "µM")))


# --------------------------------------------------------------------- helpers


def combo_plate() -> PlateLayout:
    """384 wells: four replicate 6×6 matrices of A × B on MCF7, vehicle and positive
    controls in columns 1 and 24 — what a user builds with three helper calls."""
    layout = PlateLayout.for_plate_type(384, barcode="P0001", assay="combo")
    layout = combination_matrix(
        layout,
        (1, 1),
        "Drug A",
        10,
        "Drug B",
        1,
        unit_b="µM",
        replicates=3,
        gap=1,
        cell_line="MCF7",
    )
    layout = combination_matrix(layout, (8, 1), "Drug A", 10, "Drug B", 1, cell_line="MCF7")
    layout = set_role(
        layout,
        [(r, 0) for r in range(16)],
        "negative_control",
        vehicle="DMSO",
        vehicle_percent=0.1,
        cell_line="MCF7",
    )
    return set_role(layout, [(r, 23) for r in range(16)], "positive_control", cell_line="MCF7")


def drc_plate() -> PlateLayout:
    """384 wells: 8 compounds × 10-point 1:3 dose-response curves in duplicate, controls in
    columns 1, 2 and 23, 24."""
    layout = PlateLayout.for_plate_type(384)
    for i in range(8):
        layout = dilution_series(
            layout,
            (2 * i, 2),
            f"Cpd {i + 1}",
            10,
            "µM",
            factor=3,
            points=10,
            replicates=2,
            cell_line="HeLa",
        )
    layout = set_role(
        layout,
        [(r, c) for r in range(16) for c in (0, 1)],
        "negative_control",
        vehicle="DMSO",
        vehicle_percent=0.1,
        cell_line="HeLa",
    )
    return set_role(
        layout,
        [(r, c) for r in range(16) for c in (22, 23)],
        "positive_control",
        cell_line="HeLa",
    )


def test_combination_matrix_helper() -> None:
    layout = combo_plate()
    block = {k: layout.get(k) for k in [(1 + i, 1 + j) for i in range(6) for j in range(6)]}
    corner = block[(1, 1)]
    assert corner.role == "negative_control" and not corner.active_treatments  # 0 × 0
    a_doses = sorted(
        {t.value for a in block.values() for t in a.treatments if t.compound == "Drug A"}
    )
    assert a_doses == pytest.approx([0, 10 / 81, 10 / 27, 10 / 9, 10 / 3, 10])
    assert block[(1, 3)].concentration is not None and block[(3, 1)].concentration is not None
    assert block[(6, 6)].is_combination
    assert all(layout.get((1, 1 + 7 * r)).role == "negative_control" for r in range(3))
    assert len([a for a in layout.wells.values() if a.is_combination]) == 4 * 25
    assert not [i for i in validate(layout) if i.level == "error"]


def test_dilution_series_helper() -> None:
    layout = drc_plate()
    row = [layout.get((0, c)) for c in range(2, 12)]
    assert [a.concentration for a in row] == pytest.approx(dilution_values(10, 3, 10))
    assert layout.get((1, 2)) == layout.get((0, 2))  # duplicate in the next row
    reverse = dilution_series(
        PlateLayout.for_plate_type(96),
        (0, 0),
        "X",
        1,
        "µM",
        points=4,
        direction="down",
        reverse=True,
        cell_line="A",
    )
    assert [reverse.get((r, 0)).concentration for r in range(4)] == pytest.approx(
        [1 / 27, 1 / 9, 1 / 3, 1]
    )
    with pytest.raises(ValueError, match="outside"):
        dilution_series(PlateLayout.for_plate_type(96), (0, 5), "X", 1, points=10)
    assert dilution_values(1, math.sqrt(10), 3) == pytest.approx([1, 10**-0.5, 0.1])
    assert not [i for i in validate(layout) if i.level == "error"]


def test_copy_paste_edges_and_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    layout = combo_plate()
    clip = copy_region(layout, [(1 + i, 1 + j) for i in range(6) for j in range(6)])
    assert clip.size == (6, 6)
    copied = paste(PlateLayout.for_plate_type(384), clip, (10, 15))
    assert copied.get((10, 15)) == layout.get((1, 1)) and copied.get((15, 20)) == layout.get((6, 6))
    with pytest.raises(ValueError, match="outside"):
        paste(copied, clip, (14, 20))
    excluded = exclude_edges(layout)
    assert all(excluded.get(k).role == "excluded" for k in edge_wells(layout))
    assert exclude_edges(excluded, False) == layout  # undone exactly
    save_template(layout, "384w 4x 6x6 combo")
    assert list_templates() == ["384w 4x 6x6 combo"]
    renamed = apply_template(
        load_template("384w 4x 6x6 combo"),
        cell_lines={"MCF7": "A549"},
        compounds={"Drug A": "Olaparib"},
        barcode="P0002",
    )
    assert renamed.barcode == "P0002" and renamed.cell_lines() == ["A549"]
    assert "Olaparib" in renamed.compounds() and "Drug A" not in renamed.compounds()


# ---------------------------------------------------------------- import / export


def test_round_trip_csv_xlsx_platemap_json(tmp_path: Path) -> None:
    layout = combo_plate()
    layout = layout.updated(
        {
            (15, 12): WellAnnotation(
                "MCF7",
                (Treatment.of("Drug C", 250, "ng/mL", vehicle="PBS"),),
                passage="P12",
                seeding_density=1500,
                time_point_h=48,
                notes="moved, 'odd', well",
                custom=(("plate_batch", "B7"),),
                replicate_override="custom group",
            ),
            (15, 13): WellAnnotation(role="blank"),
        },
    )
    for name in ("layout.csv", "layout.xlsx", "layout.json"):
        path = tmp_path / name
        save_layout(layout, path)
        again = load_layout(path).layout
        assert again == layout, name
        save_layout(again, tmp_path / f"again_{name}")
        assert load_layout(tmp_path / f"again_{name}").layout == layout
    first = (tmp_path / "layout.csv").read_text()
    assert first == (tmp_path / "again_layout.csv").read_text()  # byte-identical
    write_platemap_xlsx(layout, tmp_path / "map.xlsx")
    assert load_layout(tmp_path / "map.xlsx").layout == layout


def test_import_unfamiliar_columns_and_legacy(tmp_path: Path) -> None:
    path = tmp_path / "lab.csv"
    path.write_text(
        "Well ID,Cell Line,Drug A,Conc A,Unit A,Drug B,Conc B,Unit B,Well Type,Operator note\n"
        "C07,MCF7,Olaparib,1,uM,Talazoparib,10,nM,sample,ok\n"
        "C08,MCF7,DMSO,,,,,,negative,\n",
        encoding="utf-8",
    )
    mapping = suggest_mapping(
        [
            "Well ID",
            "Cell Line",
            "Drug A",
            "Conc A",
            "Unit A",
            "Drug B",
            "Conc B",
            "Unit B",
            "Well Type",
            "Operator note",
        ]
    )
    assert mapping["Drug B"] == "compound_2" and mapping["Unit A"] == "unit_1"
    assert mapping["Well Type"] == "role" and mapping["Operator note"] == "custom:Operator note"
    imported = load_layout(path)
    c07 = imported.layout.get(parse_well_name("C07"))
    assert c07.is_combination and {t.label() for t in c07.treatments} == {
        "Olaparib 1 µM",
        "Talazoparib 10 nM",
    }
    assert c07.custom_dict == {"Operator note": "ok"}
    c08 = imported.layout.get(parse_well_name("C08"))
    assert c08.role == "negative_control" and c08.compound == "DMSO"
    assert any("recorded at 0" in n for n in imported.notes)
    assert imported.layout.rows == 8  # fits a 96-well plate


def test_layout_figure(tmp_path: Path) -> None:
    for layer in ("compound", "concentration", "role", "cell_line", "replicate_group"):
        target = tmp_path / f"{layer}.png"
        write_layout_figure(combo_plate(), target, layer)
        assert target.stat().st_size > 10_000
    write_layout_figure(drc_plate(), tmp_path / "drc.pdf", "concentration")
    assert (tmp_path / "drc.pdf").read_bytes().startswith(b"%PDF")


# --------------------------------------------------------------------- validation


def test_validation_catches_each_seeded_error() -> None:
    layout = drc_plate()
    seeded = layout.updated(
        {
            (0, 2): WellAnnotation("", layout.get((0, 2)).treatments),  # no cell line
            (0, 3): WellAnnotation("HeLa"),  # sample without a treatment
            (0, 4): WellAnnotation("HeLa", (Treatment.of("Cpd 1", 1, "ng/mL"),)),  # unit kind
            (0, 5): WellAnnotation(
                "HeLa", (Treatment.of("Cpd 2", 3, "µM"), Treatment.of("Cpd 9", 1, "µM"))
            ),  # no single agent
            (0, 6): WellAnnotation(role="empty", treatments=(Treatment.of("Cpd 3", 1),)),
            (5, 2): WellAnnotation("U2OS", (Treatment.of("Cpd 3", 1),)),  # no U2OS controls
        }
    )
    codes = {i.code: i for i in validate(seeded)}
    assert codes["missing_cell_line"].wells == ((0, 2),)
    assert (0, 3) in codes["sample_without_treatment"].wells
    assert (0, 4) in codes["inconsistent_units"].wells
    assert codes["missing_single_agent"].wells == ((0, 5),)
    assert codes["treated_empty"].wells == ((0, 6),)
    assert codes["no_vehicle_controls"].wells == ((5, 2),)
    few = set_role(
        PlateLayout.for_plate_type(96), [(0, 0), (1, 0)], "negative_control", cell_line="A"
    )
    few = dilution_series(few, (0, 2), "X", 1, points=4, cell_line="A")
    assert [i.code for i in validate(few)] == ["few_controls"]
    assert not validate(combo_plate())


# --------------------------------------------------------------- storage / filters


def test_versions_are_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path))
    key = plate_key(tmp_path / "raw" / "plate1")
    assert plate_key(barcode="AB 12/3") == "barcode_AB_12_3"
    store = LayoutStore(key)
    first = store.save(drc_plate(), note="first")
    second = store.save(first.updated({(0, 2): None}))
    assert (first.version, second.version, second.parent_version) == (1, 2, 1)
    assert store.versions() == [1, 2]
    assert store.load(1) == first and store.load() == second
    assert "raw" not in str(store.folder.relative_to(tmp_path))


def test_annotation_filter() -> None:
    layout = drc_plate()
    hits = AnnotationFilter(
        cell_line="hela", compound="Cpd 1", min_concentration=100, unit="nM"
    ).select(layout)
    values = sorted(layout.get(k).concentration or 0 for k in hits)
    assert len(hits) == 2 * 5 and min(values) == pytest.approx(10 / 81)
    controls = AnnotationFilter(roles=("negative_control",)).select(layout)
    assert len(controls) == 32


# ------------------------------------------------------------------ runs


def test_layout_in_a_run_and_annotate(tmp_path: Path) -> None:
    from hcs_studio.io.base import open_plate
    from hcs_studio.output.results import RunResults
    from hcs_studio.pipeline.executor import annotate_run, pipeline_hash, run_pipeline
    from hcs_studio.pipeline.schema import Pipeline
    from tests.builders import make_imagexpress
    from tests.test_batch import PIPELINE, WELLS, _image

    plate = open_plate(make_imagexpress(tmp_path / "raw" / "ix", WELLS, n_sites=2, image_fn=_image))
    layout = PlateLayout.for_plate_type(96, barcode="RUN1")
    layout = set_role(layout, [(0, 0), (0, 1), (1, 0)], "negative_control", cell_line="MCF7")
    layout = dilution_series(layout, (0, 2), "X", 1, "µM", points=1, cell_line="MCF7")
    layout = dilution_series(layout, (1, 1), "X", 10, "µM", points=2, cell_line="MCF7")
    pipeline = Pipeline.from_text(PIPELINE)
    result = run_pipeline(
        plate, pipeline, output_root=tmp_path / "out", layout=layout, checksums=False
    )
    folder = result.folder
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert manifest["layout"]["wells"] == 6 and len(manifest["layout"]["sha256"]) == 64
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    assert list(summary.columns[:3]) == ["cell_line", "role", "compound_1"]
    assert summary.loc["B03", "conc_1"] == pytest.approx(10 / 3)
    results = RunResults(folder)
    assert "conc_1" not in results.readout_columns()
    assert results.layout["B02"].label() == "MCF7 · X 10 µM"

    # Renaming a compound does not change what the images are: same resume hash.
    renamed = apply_template(layout, compounds={"X": "Olaparib"})
    assert pipeline_hash(plate, pipeline, renamed) == pipeline_hash(plate, pipeline, layout)
    unstained = set_role(layout, [(1, 2)], "unstained")
    assert pipeline_hash(plate, pipeline, unstained) != pipeline_hash(plate, pipeline, layout)

    annotate_run(folder, renamed)
    summary = pd.read_csv(folder / "tables" / "per_well_summary.csv").set_index("well")
    assert summary.loc["B02", "compound_1"] == "Olaparib"
    manifest = json.loads((folder / "run_manifest.json").read_text())
    assert len(manifest["layout_history"]) == 1
    assert "Olaparib" in (folder / "index.html").read_text()
    with pytest.raises(ValueError, match="unstained or blank"):
        annotate_run(folder, unstained)
    rows = annotation_rows(renamed)
    assert rows["B02"]["conc_1"] == pytest.approx(10.0)


def test_plate_without_any_cell_line_is_a_warning_not_an_error() -> None:
    plate = PlateLayout.for_plate_type(96)
    plate = dilution_series(plate, (0, 0), "Cpd 1", 10.0, "µM", factor=10, points=4, cell_line="")
    plate = set_role(plate, [(1, c) for c in range(4)], "negative_control")
    issues = {i.code: i for i in validate(plate)}
    assert "missing_cell_line" not in issues
    assert issues["no_cell_lines"].level == "warning"
    assert len(issues["no_cell_lines"].wells) == 8
