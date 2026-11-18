"""Plate annotation: cell lines, treatments (single agents and combinations), well roles.

A ``PlateLayout`` is an immutable value: editing returns a new layout, and saving it makes
a new version (``io/layout_store.py``). Concentrations are stored in base units of their
kind (M, g/L or volume fraction; see ``model/units.py``) and shown in the unit the user
chose. Treatments are kept sorted, so a combination (A, B) equals (B, A).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from hcs_studio.model import units
from hcs_studio.model.plate import parse_well_name, well_name

FORMAT = "hcs-studio-layout/1"
ROLES = (
    "sample",
    "negative_control",
    "positive_control",
    "untreated",
    "blank",
    "empty",
    "excluded",
    "unstained",
)
#: Roles of wells without cells or not to be analysed.
NO_CELLS = ("blank", "empty")
PLATE_TYPES: dict[int, tuple[int, int]] = {
    6: (2, 3),
    12: (3, 4),
    24: (4, 6),
    48: (6, 8),
    96: (8, 12),
    384: (16, 24),
    1536: (32, 48),
}
_LEGACY_ROLES = {
    "negative": "negative_control",
    "positive": "positive_control",
    "unstained": "unstained",
    "blank": "blank",
    "": "sample",
}


def _canonical(value: float) -> float:
    """12 significant digits, so unit conversions round-trip exactly."""
    return float(f"{value:.12g}")


@dataclass(frozen=True, order=True)
class Treatment:
    """One compound in a well. ``concentration`` is in base units (M, g/L or fraction)."""

    compound: str
    concentration: float
    unit: str = "µM"
    vehicle: str = ""

    def __post_init__(self) -> None:
        if not self.compound.strip():
            raise ValueError("A treatment needs a compound name")
        if not self.concentration >= 0:
            raise ValueError(f"{self.compound}: concentration must be 0 or positive")
        object.__setattr__(self, "compound", self.compound.strip())
        object.__setattr__(self, "unit", units.parse_unit(self.unit).symbol)
        object.__setattr__(self, "concentration", _canonical(self.concentration))

    @classmethod
    def of(cls, compound: str, value: float, unit: str = "µM", vehicle: str = "") -> Treatment:
        """A treatment from a value in a display unit (e.g. ``Treatment.of("X", 100, "nM")``)."""
        return cls(compound, _canonical(units.to_base(value, unit)), unit, vehicle)

    @property
    def kind(self) -> units.Kind:
        return units.parse_unit(self.unit).kind

    @property
    def value(self) -> float:
        """The concentration in its display unit."""
        return _canonical(units.from_base(self.concentration, self.unit))

    def label(self) -> str:
        return f"{self.compound} {units.format_value(self.value)} {self.unit}"


@dataclass(frozen=True)
class WellAnnotation:
    cell_line: str = ""
    treatments: tuple[Treatment, ...] = ()
    role: str = "sample"
    passage: str = ""
    seeding_density: float | None = None
    #: Replicate group set by the user; empty = derived from cell line and treatments.
    replicate_override: str = ""
    time_point_h: float | None = None
    notes: str = ""
    #: User-defined fields as sorted (key, value) pairs.
    custom: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Unknown well role {self.role!r} (allowed: {', '.join(ROLES)})")
        compounds = [t.compound.lower() for t in self.treatments]
        if len(set(compounds)) != len(compounds):
            raise ValueError("A compound appears twice in one well")
        object.__setattr__(self, "treatments", tuple(sorted(self.treatments)))
        object.__setattr__(self, "custom", tuple(sorted(dict(self.custom).items())))
        object.__setattr__(self, "cell_line", self.cell_line.strip())

    # -------------------------------------------------------------- derived

    @property
    def is_combination(self) -> bool:
        return len([t for t in self.treatments if t.concentration > 0]) > 1

    @property
    def active_treatments(self) -> tuple[Treatment, ...]:
        """Treatments at a concentration above 0 (0 means the compound is absent)."""
        return tuple(t for t in self.treatments if t.concentration > 0)

    @property
    def replicate_group(self) -> str:
        if self.replicate_override:
            return self.replicate_override
        prefix = self.cell_line or "no cell line"
        if self.role in NO_CELLS or self.role == "excluded":
            return self.role
        if self.role != "sample":
            vehicle = {t.vehicle for t in self.treatments if t.vehicle}
            extra = f" ({', '.join(sorted(vehicle))})" if vehicle else ""
            active = " + ".join(t.label() for t in self.active_treatments)
            return f"{prefix} · {self.role.replace('_', ' ')}{extra}" + (
                f" · {active}" if active else ""
            )
        active = " + ".join(t.label() for t in self.active_treatments)
        return f"{prefix} · {active or 'no treatment'}"

    @property
    def custom_dict(self) -> dict[str, str]:
        return dict(self.custom)

    def label(self) -> str:
        """One line for tooltips and image labels."""
        parts = [self.cell_line] if self.cell_line else []
        active = " + ".join(t.label() for t in self.active_treatments)
        if active:
            parts.append(active)
        elif self.treatments:  # vehicle-only wells, the 0 × 0 corner of a matrix
            parts.append(" + ".join(t.compound for t in self.treatments) + " at 0")
        if self.role != "sample":
            parts.append(self.role.replace("_", " "))
        return " · ".join(parts) or "not annotated"

    # ------------------------------------------- compatibility (Phases 5–9)

    @property
    def compound(self) -> str:
        """Active compounds, or (vehicle-only wells) the compounds listed at 0."""
        shown = self.active_treatments or self.treatments
        return " + ".join(t.compound for t in shown)

    @property
    def concentration(self) -> float | None:
        """Concentration of a single-agent well in its display unit (None otherwise)."""
        active = self.active_treatments
        return active[0].value if len(active) == 1 else None

    @property
    def treatment(self) -> str:
        return self.custom_dict.get("treatment", "")

    @property
    def control_type(self) -> str:
        return {
            "negative_control": "negative",
            "positive_control": "positive",
            "unstained": "unstained",
            "blank": "blank",
        }.get(self.role, "")


EMPTY = WellAnnotation(role="empty")


@dataclass(frozen=True)
class PlateLayout:
    """Annotation of every well of a plate. Wells not listed are empty."""

    rows: int = 16
    cols: int = 24
    wells: Mapping[tuple[int, int], WellAnnotation] = field(default_factory=dict)
    plate_id: str = ""
    barcode: str = ""
    assay: str = ""
    date: str = ""
    operator: str = ""
    #: Order of user-defined columns (custom fields), for tables and the editor.
    custom_fields: tuple[str, ...] = ()
    version: int = 0
    parent_version: int | None = None

    def __post_init__(self) -> None:
        for row, col in self.wells:
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise ValueError(
                    f"Well {well_name(row, col)} is outside a {self.rows}×{self.cols} plate"
                )
        object.__setattr__(self, "wells", MappingProxyType(dict(self.wells)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlateLayout):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(repr(sorted(self.to_dict()["wells"].items())))

    @classmethod
    def for_plate_type(cls, plate_type: int, **metadata: Any) -> PlateLayout:
        if plate_type not in PLATE_TYPES:
            raise ValueError(f"Unknown plate type {plate_type} (known: {sorted(PLATE_TYPES)})")
        rows, cols = PLATE_TYPES[plate_type]
        return cls(rows=rows, cols=cols, **metadata)

    @property
    def plate_type(self) -> int | None:
        return next(
            (n for n, shape in PLATE_TYPES.items() if shape == (self.rows, self.cols)), None
        )

    def get(self, key: tuple[int, int]) -> WellAnnotation:
        return self.wells.get(key, EMPTY)

    def all_wells(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

    def updated(
        self, changes: Mapping[tuple[int, int], WellAnnotation | None], **metadata: Any
    ) -> PlateLayout:
        """A new layout with wells replaced (``None`` = cleared)."""
        wells = dict(self.wells)
        for key, annotation in changes.items():
            if annotation is None or annotation == EMPTY:
                wells.pop(key, None)
            else:
                wells[key] = annotation
        custom = list(self.custom_fields)
        for annotation in changes.values():
            for name, _ in annotation.custom if annotation else ():
                if name not in custom and not name.startswith("_"):
                    custom.append(name)
        return replace(self, wells=wells, custom_fields=tuple(custom), **metadata)

    def processing_signature(self) -> list[tuple[str, str]]:
        """The part of the layout image processing uses (unstained and blank wells): the
        only part that identifies a run's processed images."""
        return sorted(
            (well_name(*k), a.role)
            for k, a in self.wells.items()
            if a.role in ("unstained", "blank")
        )

    def compounds(self) -> list[str]:
        return sorted({t.compound for a in self.wells.values() for t in a.treatments})

    def cell_lines(self) -> list[str]:
        return sorted({a.cell_line for a in self.wells.values() if a.cell_line})

    # -------------------------------------------------------------------- JSON

    def to_dict(self) -> dict[str, Any]:
        def well(a: WellAnnotation) -> dict[str, Any]:
            out: dict[str, Any] = {"role": a.role}
            for name in ("cell_line", "passage", "replicate_override", "notes"):
                if getattr(a, name):
                    out[name] = getattr(a, name)
            for name in ("seeding_density", "time_point_h"):
                if getattr(a, name) is not None:
                    out[name] = getattr(a, name)
            if a.treatments:
                out["treatments"] = [
                    {
                        "compound": t.compound,
                        "concentration": t.concentration,
                        "unit": t.unit,
                        **({"vehicle": t.vehicle} if t.vehicle else {}),
                    }
                    for t in a.treatments
                ]
            if a.custom:
                out["custom"] = dict(a.custom)
            return out

        return {
            "format": FORMAT,
            "plate": {
                "rows": self.rows,
                "cols": self.cols,
                "plate_id": self.plate_id,
                "barcode": self.barcode,
                "assay": self.assay,
                "date": self.date,
                "operator": self.operator,
                "custom_fields": list(self.custom_fields),
                "version": self.version,
                "parent_version": self.parent_version,
            },
            "wells": {well_name(*k): well(a) for k, a in sorted(self.wells.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlateLayout:
        if data.get("format") != FORMAT:
            raise ValueError("Not an HCS Studio plate layout (format hcs-studio-layout/1)")
        plate = dict(data["plate"])
        wells = {}
        for name, item in data.get("wells", {}).items():
            treatments = tuple(
                Treatment(t["compound"], float(t["concentration"]), t["unit"], t.get("vehicle", ""))
                for t in item.get("treatments", [])
            )
            wells[parse_well_name(name)] = WellAnnotation(
                cell_line=item.get("cell_line", ""),
                treatments=treatments,
                role=item.get("role", "sample"),
                passage=str(item.get("passage", "")),
                seeding_density=item.get("seeding_density"),
                replicate_override=item.get("replicate_override", ""),
                time_point_h=item.get("time_point_h"),
                notes=item.get("notes", ""),
                custom=tuple(item.get("custom", {}).items()),
            )
        return cls(
            rows=int(plate["rows"]),
            cols=int(plate["cols"]),
            wells=wells,
            plate_id=plate.get("plate_id", ""),
            barcode=plate.get("barcode", ""),
            assay=plate.get("assay", ""),
            date=plate.get("date", ""),
            operator=plate.get("operator", ""),
            custom_fields=tuple(plate.get("custom_fields", [])),
            version=int(plate.get("version", 0)),
            parent_version=plate.get("parent_version"),
        )


def legacy_role(control_type: str) -> str:
    """Role for the ``control_type`` column of Phase 5–10 layout files."""
    key = control_type.strip().lower()
    if key in ROLES:
        return key
    if key not in _LEGACY_ROLES:
        raise ValueError(f"Unknown control type {control_type!r}")
    return _LEGACY_ROLES[key]


# --------------------------------------------------------------------- filters


@dataclass(frozen=True)
class AnnotationFilter:
    """Wells matching all given criteria (e.g. MCF7 wells with drug X ≥ 100 nM)."""

    cell_line: str = ""
    compound: str = ""
    min_concentration: float | None = None
    max_concentration: float | None = None
    unit: str = "µM"
    roles: tuple[str, ...] = ()
    text: str = ""

    def matches(self, annotation: WellAnnotation) -> bool:
        if self.cell_line and annotation.cell_line.lower() != self.cell_line.lower():
            return False
        if self.roles and annotation.role not in self.roles:
            return False
        if self.text and self.text.lower() not in annotation.label().lower():
            return False
        if self.compound:
            hits = [t for t in annotation.treatments if t.compound.lower() == self.compound.lower()]
            if not hits:
                return False
            target = units.parse_unit(self.unit)
            for treatment in hits:
                if treatment.kind != target.kind:
                    return False
                if self.min_concentration is not None and treatment.concentration < units.to_base(
                    self.min_concentration, target
                ) * (1 - 1e-9):
                    return False
                if self.max_concentration is not None and treatment.concentration > units.to_base(
                    self.max_concentration, target
                ) * (1 + 1e-9):
                    return False
        return True

    def select(self, layout: PlateLayout) -> list[tuple[int, int]]:
        return sorted(k for k, a in layout.wells.items() if self.matches(a))


def wells_in(names: Iterable[str]) -> list[tuple[int, int]]:
    return [parse_well_name(n) for n in names]
