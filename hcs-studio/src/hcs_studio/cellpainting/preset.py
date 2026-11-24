"""The Cell Painting preset pipeline (Phase 24a).

Five stains by default (Bray et al. 2016): **DNA** (Hoechst), **ER** (concanavalin A),
**RNA** (SYTO 14: nucleoli and cytoplasmic RNA), **AGP** (phalloidin + WGA: actin, Golgi,
plasma membrane) and **Mito** (MitoTracker). ``mapping`` says which of the plate's channels holds
each (custom channel sets: any roles, at least ``dna``; ``cell`` names the channel that outlines
cells, AGP by default, else RNA).

Steps: nuclei by Otsu on DNA (split touching nuclei), cells grown from the nuclei up to
``cell_distance_um`` within the cell stain, cytoplasm = cell minus nucleus, then every feature
group of ``measure_objects`` on the three compartments — shape, intensity with percentiles,
radial distribution, Haralick texture at ``texture_scales_px``, granularity, the correlation of
every channel pair and neighbours — and image QC. The channel map for CellProfiler names comes
from the roles (``channel_map``).
"""

from __future__ import annotations

from typing import Any

from hcs_studio.cellpainting.naming import CELL_PAINTING

ROLES = {"dna": "DNA", "er": "ER", "rna": "RNA", "agp": "AGP", "mito": "Mito"}


def channel_map(mapping: dict[str, str]) -> dict[str, str]:
    """Column prefix of each mapped channel → its CellProfiler channel name."""
    from hcs_studio.analysis.features import column_name

    return {
        column_name(channel): ROLES.get(role, role.upper())
        for role, channel in mapping.items()
        if role != "cell"
    }


def preset(
    mapping: dict[str, str],
    *,
    channels: list[str],
    nucleus_diameter_um: float = 10.0,
    cell_distance_um: float = 12.0,
    texture_scales_px: tuple[int, ...] = (1, 3, 5),
) -> Any:
    from hcs_studio.pipeline.schema import Pipeline

    if "dna" not in mapping:
        raise ValueError("The Cell Painting preset needs the DNA channel (mapping['dna'])")

    def number(name: str) -> int:
        if name not in channels:
            raise ValueError(f"Channel {name!r} is not on the plate ({', '.join(channels)})")
        return channels.index(name) + 1

    cell = mapping.get("cell") or mapping.get("agp") or mapping.get("rna")
    expand: dict[str, Any] = {"distance_um": cell_distance_um}
    if cell:
        expand["mask_channel"] = number(cell)
    steps = [
        {
            "module": "segment_classical",
            "params": {
                "channel": number(mapping["dna"]),
                "method": "otsu",
                "min_distance_um": nucleus_diameter_um / 4,
            },
        },
        {"module": "expand_cells", "params": expand},
        {"module": "cytoplasm", "params": {"mode": "cells", "cells": "cells", "gap_um": 0.0}},
        {
            "module": "measure_objects",
            "params": {
                "labels": ["nuclei", "cells", "cytoplasm"],
                "texture": True,
                "texture_scales_px": list(texture_scales_px),
                "granularity": True,
                "correlation": True,
            },
        },
        {"module": "qc"},
    ]
    return Pipeline.model_validate({"name": "cell_painting", "steps": steps})


__all__ = ["CELL_PAINTING", "ROLES", "channel_map", "preset"]
