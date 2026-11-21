"""The assay-module framework (Phase 20).

An :class:`AssayModule` is a table step with:

- ``roles``: the channels it needs (e.g. DNA, EdU, pH3), mapped to the plate's channel names by
  its parameters (``<role>_channel``);
- ``preset``: a complete pipeline (projection/background, segmentation, measurement, the assay);
- calibration in ``fit_table``: thresholds from the run's **control wells** (vehicle, positive
  control) found in the run's plate layout, stored in the run (``analysis_summary.json`` and
  ``assays/<name>/calibration.json``);
- ``run_warnings``: checks of how the images were made (e.g. a max projection cannot give DNA
  content);
- ``finish_table``: its report page ``assays/<name>/report.html`` and per-well CSV.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from hcs_studio.analysis.features import column_name
from hcs_studio.analysis.table_steps import TableModule, unique_rows
from hcs_studio.processing.base import MODULES, ModuleError

VEHICLE_ROLES = ("negative_control",)
POSITIVE_ROLES = ("positive_control",)


@dataclass(frozen=True)
class ChannelRole:
    key: str  # the parameter is ``<key>_channel``
    title: str
    description: str
    required: bool = True


def channel_column(channel: str, feature: str) -> str:
    """The measurement column of a channel (by name, as on the plate) and a feature."""
    return f"{column_name(channel)}_{feature}"


class AssayModule(TableModule):
    """Base of assay table steps (see the module docstring)."""

    needs_fit: ClassVar[bool] = True
    roles: ClassVar[tuple[ChannelRole, ...]] = ()
    #: Column prefix of the step's object and well columns.
    prefix: ClassVar[str] = ""

    # ------------------------------------------------------------ run context

    def manifest(self) -> dict[str, Any]:
        if self.run_folder is None:
            return {}
        try:
            data: dict[str, Any] = json.loads(
                (self.run_folder / "run_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return {}
        return data

    def well_roles(self) -> dict[str, str]:
        """Role of each well from the run's plate layout ({} without a layout)."""
        if self.run_folder is None:
            return {}
        from hcs_studio.model.layout import load_plate_layout, run_layout_file
        from hcs_studio.model.plate import well_name
        from hcs_studio.pharmacology.normalize import drug_treatments

        path = run_layout_file(self.run_folder)
        if path is None:
            return {}
        layout = load_plate_layout(path)
        roles = {}
        for key, annotation in layout.wells.items():
            role = annotation.role
            if role == "untreated" and not drug_treatments(annotation):
                role = "negative_control"
            roles[well_name(*key)] = role
        return roles

    def control_objects(
        self, roles: tuple[str, ...], columns: list[str] | None = None
    ) -> pd.DataFrame:
        """All objects (of ``params.object_type``, counted once) of the wells with ``roles``."""
        from hcs_studio.analysis.table_steps import read_well_objects

        wells = [w for w, r in self.well_roles().items() if r in roles]
        if self.run_folder is None or not wells:
            return pd.DataFrame()
        parts = []
        for well in wells:
            folder = self.run_folder / "wells" / well
            if not (folder / "objects.parquet").exists():
                continue
            objects = read_well_objects(folder)
            objects = objects[
                (objects["object_type"] == str(self.params.object_type)) & unique_rows(objects)
            ]
            parts.append(
                objects if columns is None else objects[[c for c in columns if c in objects]]
            )
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def require(self, frame: pd.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise ModuleError(
                f"column(s) {', '.join(missing)} are not measured; check the channel mapping "
                "and that measure_objects ran on these objects"
            )

    def describe_state(self, state: Any) -> dict[str, Any]:
        """The calibration, as stored in ``analysis_summary.json``."""
        return dict(state) if isinstance(state, dict) else {}

    def preview_values(self, state: Any) -> tuple[Any, str]:
        """Values of the vehicle objects to plot in the wizard's calibration preview, and the
        axis label (none by default)."""
        import numpy as np

        return np.array([]), ""

    def run_warnings(self) -> list[str]:
        """Checks of the acquisition and processing (overridden by assays)."""
        return []

    # ---------------------------------------------------------------- files

    def output_folder(self, folder: Path) -> Path:
        out = Path(folder) / "assays" / self.name
        out.mkdir(parents=True, exist_ok=True)
        return out

    @classmethod
    def preset(cls, mapping: dict[str, str], **options: Any) -> Any:
        """A complete pipeline for this assay (``mapping``: role → channel name)."""
        raise NotImplementedError


def assay_names() -> list[str]:
    from hcs_studio.processing.base import load_builtin_modules

    load_builtin_modules()
    return [n for n in MODULES.names() if issubclass(MODULES.get(n), AssayModule)]
