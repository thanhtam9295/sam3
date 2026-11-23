"""Library plate maps in the registry (Phase 23a): ``library_plate``, ``library_well`` and
``compound`` rows from a map read by ``screening.library.read_library_map``."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select

from hcs_studio.db.catalog import Compound, LibraryPlate, LibraryWell
from hcs_studio.db.engine import session_scope


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def import_library(library: pd.DataFrame, name: str = "") -> dict[str, Any]:
    """Create or replace the library plates of ``library`` (one map row per well); compounds
    are created by name (SMILES filled in when given and not yet known)."""
    counts = {"plates": 0, "wells": 0, "new_compounds": 0, "replaced_plates": 0}
    with session_scope() as session:
        known = {c.name: c for c in session.scalars(select(Compound)).all()}
        for barcode, group in library.groupby("library_plate"):
            plate = session.scalars(
                select(LibraryPlate).where(LibraryPlate.barcode == barcode)
            ).first()
            if plate is not None:
                session.delete(plate)
                session.flush()
                counts["replaced_plates"] += 1
            plate = LibraryPlate(barcode=str(barcode), name=name)
            session.add(plate)
            session.flush()
            for record in group.to_dict("records"):
                name_ = str(record["compound"])
                smiles = str(record.get("smiles") or "")
                compound = known.get(name_)
                if compound is None:
                    compound = Compound(name=name_, smiles=smiles)
                    session.add(compound)
                    session.flush()
                    known[name_] = compound
                    counts["new_compounds"] += 1
                elif smiles and not compound.smiles:
                    compound.smiles = smiles
                volume = _number(record.get("volume_nl"))
                concentration = _number(record.get("concentration_m"))
                session.add(
                    LibraryWell(
                        library_plate_id=plate.id,
                        well=str(record["well"]),
                        compound_id=compound.id,
                        concentration_m=concentration,
                        volume_nl=volume,
                    )
                )
                counts["wells"] += 1
            counts["plates"] += 1
    return counts


def library_map(barcodes: list[str] | None = None) -> pd.DataFrame:
    """The stored map (all plates, or the given ones) in the ``read_library_map`` columns."""
    with session_scope() as session:
        query = (
            select(
                LibraryPlate.barcode,
                LibraryWell.well,
                Compound.name,
                LibraryWell.concentration_m,
                LibraryWell.volume_nl,
                Compound.smiles,
            )
            .join(LibraryWell, LibraryWell.library_plate_id == LibraryPlate.id)
            .join(Compound, Compound.id == LibraryWell.compound_id)
        )
        if barcodes:
            query = query.where(LibraryPlate.barcode.in_(barcodes))
        rows = session.execute(query).all()
    return pd.DataFrame(
        rows,
        columns=["library_plate", "well", "compound", "concentration_m", "volume_nl", "smiles"],
    )
