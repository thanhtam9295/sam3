"""Phase 23a: scoring a primary screen and calling hits, on a synthetic screen with known
answers; library maps and assay layouts."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hcs_studio.screening import rules
from hcs_studio.screening.scores import ssmd, z_prime
from hcs_studio.screening.screen import ReadoutConfig, ScreenConfig, run_screen


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from hcs_studio.db.engine import dispose_engines

    monkeypatch.setenv("HCS_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("HCS_DATABASE_URL", raising=False)
    dispose_engines()
    yield tmp_path
    dispose_engines()


@pytest.fixture(scope="module")
def screen(tmp_path_factory: pytest.TempPathFactory) -> tuple[list[Path], dict[str, list[str]]]:
    from hcs_studio.synthetic.screening import write_screen

    return write_screen(tmp_path_factory.mktemp("screen") / "runs")


def _config(
    score: str = "b", rule: str = "rep_score <= -3 and count_pct > 60", **kw: object
) -> ScreenConfig:
    return ScreenConfig.model_validate(
        {
            "readouts": [
                {"column": "reporter_mean", "alias": "rep"},
                {"column": "nuclei_count", "alias": "count", "activity": "percent_of_control"},
            ],
            "score": score,
            "hit_rule": rule,
            **kw,
        }
    )


def test_seeded_hits_recovered_and_artifact_makes_no_false_hits(
    screen: tuple[list[Path], dict[str, list[str]]], tmp_path: Path
) -> None:
    """Done-when (23): 4 libraries × 2 replicates, 384 wells, 40 seeded hits (−35 %, noise 6 %),
    a row gradient (±15 %) and an edge offset (−25 %) differing per plate: with B-score ≥ 95 %
    of the hits are recovered and no other compound is a hit; robust Z alone loses hits and
    calls edge wells."""
    folders, truth = screen
    result = run_screen(folders, _config("b"), tmp_path / "b")
    hits, seeded = set(result.hits["compound"]), set(truth["hits"])
    assert len(hits & seeded) >= 0.95 * len(seeded)
    assert hits <= seeded
    assert set(result.plates.columns) >= {"rep_z_prime", "rep_b_score", "hit_rate_pct"}
    assert result.plates["rep_b_score"].all()
    without = run_screen(folders, _config("z"), tmp_path / "z")
    z_hits = set(without.hits["compound"])
    assert len(z_hits & seeded) < 0.6 * len(seeded) and z_hits - seeded  # the artifact bites
    files = {p.name for p in (tmp_path / "b").iterdir()}
    assert {"wells.csv", "compounds.csv", "hits.csv", "plates.csv", "screen.json"} <= files
    ranked = result.hits["rep_score"].tolist()
    assert ranked == sorted(ranked) and result.hits["rank"].tolist() == list(
        range(1, len(ranked) + 1)
    )


def test_cytotoxic_and_artifact_compounds_are_kept_out(
    screen: tuple[list[Path], dict[str, list[str]]], tmp_path: Path
) -> None:
    folders, truth = screen
    loose = run_screen(
        folders, _config("b", "rep_score <= -3", artifact_filters=[]), tmp_path / "loose"
    )
    loose_hits = set(loose.hits["compound"])
    assert set(truth["cytotoxic"]) <= loose_hits  # without the count rule they look like hits
    assert loose_hits & set(truth["autofluorescent"])  # the dark ones, without the QC filter
    strict = run_screen(folders, _config("b"), tmp_path / "strict")
    assert not set(strict.hits["compound"]) & (
        set(truth["cytotoxic"]) | set(truth["autofluorescent"])
    )
    flagged = strict.wells[strict.wells["compound"].isin(truth["autofluorescent"])]
    assert (flagged["artifact"].str.contains("autofluorescence")).all() and not flagged[
        "well_hit"
    ].any()


def test_replicate_rules(screen: tuple[list[Path], dict[str, list[str]]], tmp_path: Path) -> None:
    folders, _ = screen
    counts = {}
    for rule in ("all", "any", 2):
        result = run_screen(folders, _config("b", replicate_rule=rule), tmp_path / str(rule))
        counts[str(rule)] = len(result.hits)
        assert set(result.compounds["n"]) == {2}
    assert counts["all"] == counts["2"] <= counts["any"]


def test_manual_review_needs_a_reason(
    screen: tuple[list[Path], dict[str, list[str]]], home: Path
) -> None:
    from hcs_studio.app import main

    folders, truth = screen
    out = home / "screen"
    config = home / "screen.json"
    config.write_text(_config("b").model_dump_json())
    assert (
        main(["screen", "score", *map(str, folders), "--config", str(config), "--output", str(out)])
        == 0
    )
    first = pd.read_csv(out / "hits.csv")["compound"].iloc[0]
    assert (
        main(
            [
                "screen",
                "review",
                str(out),
                "--compound",
                first,
                "--action",
                "exclude",
                "--reason",
                "precipitate seen in the source plate",
            ]
        )
        == 0
    )
    assert first not in set(pd.read_csv(out / "hits.csv")["compound"])
    rescued = truth["cytotoxic"][0]
    assert (
        main(
            [
                "screen",
                "review",
                str(out),
                "--compound",
                rescued,
                "--action",
                "include",
                "--reason",
                "follow up: toxic but interesting",
            ]
        )
        == 0
    )
    hits = pd.read_csv(out / "hits.csv")
    assert rescued in set(hits["compound"])
    compounds = pd.read_csv(out / "compounds.csv").set_index("compound")
    assert compounds.loc[first, "override"] == "exclude"
    assert json.loads((out / "overrides.json").read_text())[0]["reason"].startswith("precipitate")
    with pytest.raises(SystemExit):
        main(["screen", "review", str(out), "--compound", first, "--action", "include"])


# -------------------------------------------------------------------- statistics


def test_ssmd_and_z_prime_known_values() -> None:
    values = ssmd([-3.0, -4.0, -5.0])
    assert values["ssmd_mm"] == pytest.approx(-4.0)  # mean −4, SD 1
    assert values["ssmd_umvue"] == pytest.approx(-4.0 / math.sqrt(math.pi))  # √(2/2)·Γ(1)/Γ(½)
    assert values["ssmd_robust"] == pytest.approx(-4.0 / 1.4826)
    assert math.isnan(ssmd([-3.0, -4.0])["ssmd_robust"])
    positive, negative = [10.0, 12.0, 14.0], [100.0, 102.0, 104.0]
    assert z_prime(positive, negative) == pytest.approx(1 - 3 * (2 + 2) / 90)
    assert z_prime(positive, negative, robust=True) == pytest.approx(1 - 3 * (2 * 1.4826 * 2) / 90)


def test_rules_language() -> None:
    table = pd.DataFrame({"a_z": [-4.0, -2.0, np.nan, -5.0], "b_pct": [80.0, 90.0, 95.0, 20.0]})
    assert rules.evaluate("a_z <= -3 and not b_pct < 50", table).tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert rules.evaluate("-6 < a_z <= -3", table).tolist() == [True, False, False, True]
    assert rules.evaluate("abs(a_z) > 3 or b_pct > 92", table).tolist() == [True, False, True, True]
    for bad in ("__import__('os')", "a_z.real < 1", "a_z < 'x'", "a_z <"):
        with pytest.raises(rules.RuleError):
            rules.evaluate(bad, table)
    with pytest.raises(rules.RuleError, match="unknown"):
        rules.evaluate("c_z < 1", table)
    with pytest.raises(ValueError, match="unknown name"):
        _config(rule="rep_score < -3 and viability < 2").check()


def test_b_score_refused_on_a_dose_response_plate(tmp_path: Path) -> None:
    """The Phase 17 guard rail: a compound diluted along a row blocks B-score; the screen falls
    back to robust Z with a note."""
    from hcs_studio.model.plate import well_name
    from hcs_studio.normalization.plate import WellInfo
    from hcs_studio.synthetic import write_values_run

    rng = np.random.default_rng(0)
    values, wells = {}, []
    for r in range(16):
        for c in range(24):
            w = well_name(r, c)
            role = "negative_control" if c < 2 else "sample"
            compounds = ((f"D{r}", 1e-9 * 3 ** (c - 2)),) if role == "sample" else ()
            values[w] = 100 + rng.normal(0, 3)
            wells.append(WellInfo(w, role, compounds))
    folder = write_values_run(
        tmp_path / "dr", values, wells, barcode="DR1", readout="reporter_mean"
    )
    config = ScreenConfig(
        readouts=[ReadoutConfig(column="reporter_mean", alias="rep")], hit_rule="rep_score <= -3"
    )
    result = run_screen([folder], config, tmp_path / "out")
    assert not result.plates["rep_b_score"].iloc[0]
    assert any("B-score not used" in n for n in result.notes)
    sample = result.wells["role"] == "sample"
    assert result.wells.loc[sample, "rep_score"].equals(result.wells.loc[sample, "rep_z"])


# ---------------------------------------------------------------- library maps


def _template(rows: int = 16, cols: int = 24) -> object:
    from hcs_studio.model.annotation import PlateLayout
    from hcs_studio.model.layout_tools import set_role

    layout = PlateLayout.for_plate_type(rows * cols)
    layout = set_role(layout, [(r, 0) for r in range(rows)], "negative_control")
    return set_role(layout, [(r, cols - 1) for r in range(rows)], "positive_control")


def test_library_maps_and_assay_layouts(tmp_path: Path) -> None:
    from hcs_studio.model.plate import parse_well_name
    from hcs_studio.screening.library import assay_layouts, read_library_map, read_transfers

    lines = ["Plate Barcode,Well,Compound ID,Concentration,SMILES"]
    for k in range(4):
        for well in ("A01", "B03", "H12"):
            lines.append(f"LIB{k},{well},C{k}-{well},10 mM,")
    lines.append("LIB0,C05,,10 mM,")  # an empty well
    path = tmp_path / "library.csv"
    path.write_text("\n".join(lines))
    library = read_library_map(path)
    assert len(library) == 12 and library["concentration_m"].eq(0.01).all()

    header_units = tmp_path / "units.csv"
    header_units.write_text("plate,well,compound,conc_uM\nL,A1,X,50\n")
    assert read_library_map(header_units)["concentration_m"].iloc[0] == pytest.approx(5e-5)

    template = _template()
    inner = library[library["well"] == "B03"]  # A01 and H12 land on control columns of 384
    quad = assay_layouts(
        inner,
        template,
        "quadrants",
        sources=["LIB0", "LIB1", "LIB2", "LIB3"],
        destinations={"x": "ASSAY-1"},
        dilution=1e-3,
    )
    layout = quad["ASSAY-1"]
    found = {t.compound: key for key, a in layout.wells.items() for t in a.treatments}
    # 96-well B03 = (1, 2) → 384-well (2 + k // 2, 4 + k % 2) for library plate k
    assert found == {"C0-B03": (2, 4), "C1-B03": (2, 5), "C2-B03": (3, 4), "C3-B03": (3, 5)}
    assert layout.wells[(2, 4)].treatments[0].concentration == pytest.approx(1e-5)
    assert layout.wells[(0, 0)].role == "negative_control"  # the template's controls stay
    with pytest.raises(ValueError, match="negative_control"):  # LIB0 A01 → 384 A01, a vehicle
        assay_layouts(library, template, "quadrants", sources=["LIB0", "LIB1", "LIB2", "LIB3"])
    with pytest.raises(ValueError, match="negative_control"):  # C0-A01 → A01, a vehicle well
        assay_layouts(library[library["library_plate"] == "LIB0"], template, "stamp")

    stamped = assay_layouts(
        library[library["well"] != "A01"],
        template,
        "stamp",
        destinations={"LIB0": "P-0"},
        dilution=1e-3,
    )
    assert set(stamped) == {"P-0", "LIB1-A", "LIB2-A", "LIB3-A"}
    assert stamped["P-0"].wells[parse_well_name("B03")].treatments[0].compound == "C0-B03"
    assert stamped["P-0"].barcode == "P-0"

    echo = tmp_path / "transfer.csv"
    echo.write_text(
        "[EXCEPTIONS]\nRun ID,1\n\n[DETAILS]\n"
        "Source Plate Barcode,Source Well,Destination Plate Barcode,Destination Well,"
        "Transfer Volume\n"
        "LIB0,B03,ASSAY-E,C05,40\nLIB0,B03,ASSAY-E,C05,40\nLIB1,H12,ASSAY-E,D07,20\n"
    )
    transfers = read_transfers(echo)
    assert len(transfers) == 3 and transfers["volume_nl"].sum() == 100
    layouts = assay_layouts(library, template, "echo", transfers=transfers, assay_volume_ul=40)
    c05 = layouts["ASSAY-E"].wells[parse_well_name("C05")].treatments[0]
    # 10 mM × 80 nL / (40 µL + 80 nL)
    assert c05.compound == "C0-B03" and c05.concentration == pytest.approx(0.01 * 80e-9 / 40.08e-6)


def test_library_registry_and_registered_layouts(home: Path) -> None:
    from hcs_studio.app import main
    from hcs_studio.db.library import library_map
    from hcs_studio.io.layout_io import save_layout
    from hcs_studio.io.layout_store import LayoutStore, plate_key

    path = home / "lib.csv"
    path.write_text(
        "barcode,well,compound,concentration,unit,smiles\n"
        "LIBX,B02,Imatinib,10,mM,CC1=CC=CC=C1\nLIBX,B03,Dasatinib,10,mM,\n"
    )
    assert main(["screen", "library", str(path), "--name", "kinase set"]) == 0
    stored = library_map(["LIBX"])
    assert set(stored["compound"]) == {"Imatinib", "Dasatinib"}
    assert stored["concentration_m"].eq(0.01).all()
    assert main(["screen", "library", str(path)]) == 0  # re-import replaces, does not duplicate
    assert len(library_map(["LIBX"])) == 2
    template = home / "template.json"
    save_layout(_template(), template)  # type: ignore[arg-type]
    assert (
        main(
            [
                "screen",
                "layouts",
                str(path),
                "--template",
                str(template),
                "--rule",
                "stamp",
                "--dest",
                "LIBX=AS-77",
                "--dilution",
                "0.001",
                "--output",
                str(home / "lay"),
                "--register",
            ]
        )
        == 0
    )
    registered = LayoutStore(plate_key(barcode="AS-77")).load()
    assert registered is not None and registered.barcode == "AS-77"
    assert registered.wells[(1, 1)].treatments[0].compound == "Imatinib"
