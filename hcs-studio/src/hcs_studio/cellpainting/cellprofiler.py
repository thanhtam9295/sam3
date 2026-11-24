"""Import CellProfiler outputs as single-cell tables (Phase 24a).

Existing CellProfiler data can be profiled here. Three layouts are read:

- **SQLite** written by ExportToDatabase / cytominer-database: an ``Image`` table (``Per_Image``)
  and one table per compartment (``Cells``, ``Cytoplasm``, ``Nuclei`` or ``Per_Cells`` …) whose
  feature columns already carry the compartment prefix;
- **CSV folder** written by ExportToSpreadsheet: ``Image.csv`` and ``<Compartment>.csv`` with
  unprefixed feature columns (``AreaShape_Area``) and parent columns ``Parent_Cells``,
  ``Parent_Nuclei`` (prefixes are added);
- **Parquet** (e.g. CytoTable or pycytominer output): already one row per cell; read as it is.

Compartments are joined as pycytominer does by default: each cytoplasm row to its cell
(``Cytoplasm_Parent_Cells``) and its nucleus (``Cytoplasm_Parent_Nuclei``), within the same image
(``TableNumber`` when present, ``ImageNumber``). Image metadata (``Image_Metadata_Plate`` …) is
attached and renamed ``Metadata_Plate`` … — so the result has the same features as pycytominer's
``SingleCells.merge_single_cells`` and the same ``Metadata_`` convention as HCS Studio's own
single-cell tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

COMPARTMENTS = ("cells", "cytoplasm", "nuclei")
LINKS = {"cells": "Cytoplasm_Parent_Cells", "nuclei": "Cytoplasm_Parent_Nuclei"}


def _image_keys(frame: pd.DataFrame) -> list[str]:
    return [c for c in ("TableNumber", "ImageNumber") if c in frame.columns]


def _merge(tables: dict[str, pd.DataFrame], image: pd.DataFrame) -> pd.DataFrame:
    """cytoplasm ⋈ cells ⋈ nuclei (pycytominer's default linking) ⋈ image metadata."""
    if "cytoplasm" not in tables:
        raise ValueError(
            "CellProfiler import needs the Cytoplasm table (it links cells and nuclei)"
        )
    keys = _image_keys(tables["cytoplasm"])
    merged = tables["cytoplasm"].rename(columns={"ObjectNumber": "Metadata_ObjectNumber_cytoplasm"})
    for name in ("cells", "nuclei"):
        if name not in tables:
            continue
        other = tables[name].rename(columns={"ObjectNumber": f"Metadata_ObjectNumber_{name}"})
        merged = merged.merge(
            other,
            how="inner",
            left_on=[*keys, LINKS[name]],
            right_on=[*keys, f"Metadata_ObjectNumber_{name}"],
        )
    meta = [c for c in image.columns if "Metadata_" in c or c in keys]
    merged = merged.merge(image[meta], how="left", on=keys)
    renamed = {
        c: c.replace("Image_Metadata_", "Metadata_")
        for c in merged.columns
        if c.startswith("Image_Metadata_")
    }
    renamed.update({c: f"Metadata_{c}" for c in keys})
    renamed.update({LINKS[n]: f"Metadata_{LINKS[n]}" for n in LINKS if LINKS[n] in merged})
    merged = merged.rename(columns=renamed)
    if "Metadata_ObjectNumber_nuclei" in merged:
        merged["Metadata_ObjectNumber"] = merged["Metadata_ObjectNumber_nuclei"]
    front = [c for c in merged.columns if c.startswith("Metadata_")]
    return merged[front + [c for c in merged.columns if c not in front]]


def read_sqlite(path: Path, compartments: tuple[str, ...] = COMPARTMENTS) -> pd.DataFrame:
    with sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True) as connection:
        names = {
            r[0] for r in connection.execute("select name from sqlite_master where type='table'")
        }

        def table(name: str) -> pd.DataFrame:
            for candidate in (name, f"Per_{name}", name.lower(), f"Per_{name.lower()}"):
                if candidate in names:
                    return pd.read_sql_query(f'select * from "{candidate}"', connection)
            raise ValueError(
                f"{Path(path).name}: no table {name!r} (tables: {', '.join(sorted(names))})"
            )

        image = table("Image")
        tables = {c: table(c.capitalize()) for c in compartments}
    return _merge(tables, image)


def read_csv_folder(folder: Path, compartments: tuple[str, ...] = COMPARTMENTS) -> pd.DataFrame:
    folder = Path(folder)

    def read(name: str) -> pd.DataFrame:
        for candidate in (f"{name}.csv", f"{name.lower()}.csv", f"Per_{name}.csv"):
            if (folder / candidate).exists():
                return pd.read_csv(folder / candidate)
        raise ValueError(f"{folder.name}: no {name}.csv")

    image = read("Image")
    image = image.rename(
        columns={c: f"Image_{c}" for c in image.columns if c.startswith("Metadata_")}
    )
    tables = {}
    for name in compartments:
        frame = read(name.capitalize())
        prefix = name.capitalize()
        keep = {"ImageNumber", "ObjectNumber", "TableNumber"}
        frame = frame.rename(
            columns={c: (c if c in keep else f"{prefix}_{c}") for c in frame.columns}
        )
        tables[name] = frame
    return _merge(tables, image)


def read_cellprofiler(path: Path, compartments: tuple[str, ...] = COMPARTMENTS) -> pd.DataFrame:
    """A single-cell table from a CellProfiler SQLite file, CSV folder or merged Parquet."""
    path = Path(path)
    if path.is_dir():
        return read_csv_folder(path, compartments)
    suffix = path.suffix.lower()
    if suffix in (".sqlite", ".db", ".sqlite3"):
        return read_sqlite(path, compartments)
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.rename(
            columns={
                c: c.replace("Image_Metadata_", "Metadata_")
                for c in frame.columns
                if c.startswith("Image_Metadata_")
            }
        )
    raise ValueError(
        f"Cannot read {path.name} as CellProfiler output (SQLite, CSV folder, Parquet)"
    )
