"""CellProfiler-compatible feature names (Phase 24a).

HCS Studio's object columns (``area_um2``, ``dna_mean``, ``er_haralick_contrast_s3`` …) are
renamed to CellProfiler's ``<Compartment>_<Category>_<Feature>_<Channel>[_<Scale>]`` so that
pycytominer and other Cell Painting tools find the features by their prefixes
(``Nuclei_``, ``Cells_``, ``Cytoplasm_``) without manual mapping:

| HCS Studio | CellProfiler style |
|---|---|
| ``area_um2``, ``perimeter_um``, ``eccentricity`` … | ``AreaShape_Area``, ``AreaShape_Perimeter``, ``AreaShape_Eccentricity`` … |
| ``zernike_n_m`` | ``AreaShape_Zernike_n_m`` |
| ``<ch>_mean`` / ``median`` / ``sd`` / ``min`` / ``max`` / ``integrated`` | ``Intensity_MeanIntensity_<CH>`` / ``MedianIntensity`` / ``StdIntensity`` / ``MinIntensity`` / ``MaxIntensity`` / ``IntegratedIntensity`` |
| ``<ch>_p25`` / ``p75`` / ``p05`` / ``p95`` | ``Intensity_LowerQuartileIntensity_<CH>`` / ``UpperQuartileIntensity`` / ``Percentile05Intensity`` / ``Percentile95Intensity`` |
| ``<ch>_radial_inner`` / ``middle`` / ``outer`` (``ring<i>``) | ``RadialDistribution_FracAtD_<CH>_<k>of<n>`` (1 = centre) |
| ``<ch>_center_periphery_ratio`` | ``RadialDistribution_CenterPeripheryRatio_<CH>`` |
| ``<ch>_haralick_<name>_s<d>`` | ``Texture_<Name>_<CH>_<d>`` (averaged over directions) |
| ``<ch>_granularity_r<k>`` | ``Granularity_<k>_<CH>`` |
| ``<a>_<b>_correlation`` | ``Correlation_Correlation_<A>_<B>`` |
| ``nearest_neighbour_um`` / ``neighbours_within_radius`` | ``Neighbors_FirstClosestDistance`` / ``Neighbors_NumberOfNeighbors`` |
| ``centroid_x_um`` / ``centroid_y_um`` | ``Location_Center_X`` / ``Location_Center_Y`` (µm) |
| ``<child>_count`` | ``Children_<Child>_Count`` |

Channels are named by a **channel map** (column prefix → CellProfiler channel name, e.g.
``hoechst`` → ``DNA``); unmapped channels keep their name with the first letter capitalised.
Columns with no CellProfiler counterpart keep the compartment prefix and ``Other_`` (so they
still count as features).
"""

from __future__ import annotations

import re

AREA_SHAPE = {
    "area_um2": "Area",
    "volume_um3": "Volume",
    "perimeter_um": "Perimeter",
    "surface_area_um2": "SurfaceArea",
    "equivalent_diameter_um": "EquivalentDiameter",
    "major_axis_um": "MajorAxisLength",
    "minor_axis_um": "MinorAxisLength",
    "eccentricity": "Eccentricity",
    "orientation_deg": "Orientation",
    "form_factor": "FormFactor",
    "solidity": "Solidity",
    "extent": "Extent",
}
INTENSITY = {
    "mean": "MeanIntensity",
    "median": "MedianIntensity",
    "sd": "StdIntensity",
    "min": "MinIntensity",
    "max": "MaxIntensity",
    "integrated": "IntegratedIntensity",
    "p25": "LowerQuartileIntensity",
    "p75": "UpperQuartileIntensity",
    "p05": "Percentile05Intensity",
    "p95": "Percentile95Intensity",
    "cytoplasm_mean": "MeanIntensityCytoplasm",
    "nc_ratio": "NuclearCytoplasmicRatio",
}
HARALICK = {
    "asm": "AngularSecondMoment",
    "contrast": "Contrast",
    "correlation": "Correlation",
    "variance": "Variance",
    "idm": "InverseDifferenceMoment",
    "sum_avg": "SumAverage",
    "sum_var": "SumVariance",
    "sum_entropy": "SumEntropy",
    "entropy": "Entropy",
    "diff_var": "DifferenceVariance",
    "diff_entropy": "DifferenceEntropy",
    "imc1": "InfoMeas1",
    "imc2": "InfoMeas2",
}
OTHER = {
    "nearest_neighbour_um": "Neighbors_FirstClosestDistance",
    "neighbours_within_radius": "Neighbors_NumberOfNeighbors",
    "centroid_x_um": "Location_Center_X",
    "centroid_y_um": "Location_Center_Y",
    "centroid_z_um": "Location_Center_Z",
}
#: Columns that are identifiers or context, never features.
CONTEXT = {
    "well",
    "row",
    "col",
    "field",
    "timepoint",
    "object_type",
    "object_id",
    "parent_id",
    "well_x_um",
    "well_y_um",
    "in_overlap_duplicate",
}

#: Cell Painting channels (Bray et al. 2016) and the stains usually behind them.
CELL_PAINTING = ("DNA", "ER", "RNA", "AGP", "Mito")


def compartment(object_type: str) -> str:
    """'nuclei' → 'Nuclei', 'cytoplasm' → 'Cytoplasm', 'cells' → 'Cells', other → capitalised."""
    return "".join(part.capitalize() for part in re.split(r"[^0-9a-zA-Z]+", object_type) if part)


def channel(prefix: str, channel_map: dict[str, str]) -> str:
    return channel_map.get(prefix) or prefix[:1].upper() + prefix[1:]


def cp_name(
    column: str, prefixes: list[str], channel_map: dict[str, str], radial_bins: int = 3
) -> str | None:
    """The CellProfiler-style name of one object column (without the compartment), or None for
    context columns. ``prefixes``: the run's channel column prefixes, longest first."""
    if column in CONTEXT:
        return None
    if column in AREA_SHAPE:
        return f"AreaShape_{AREA_SHAPE[column]}"
    if column in OTHER:
        return OTHER[column]
    match = re.fullmatch(r"zernike_(\d+)_(\d+)", column)
    if match:
        return f"AreaShape_Zernike_{match.group(1)}_{match.group(2)}"
    for prefix in prefixes:
        correlation = re.fullmatch(rf"{re.escape(prefix)}_(.+)_correlation", column)
        if correlation and correlation.group(1) in prefixes:
            return (
                f"Correlation_Correlation_{channel(prefix, channel_map)}_"
                f"{channel(correlation.group(1), channel_map)}"
            )
        if not column.startswith(prefix + "_"):
            continue
        rest = column[len(prefix) + 1 :]
        ch = channel(prefix, channel_map)
        if rest in INTENSITY:
            return f"Intensity_{INTENSITY[rest]}_{ch}"
        ring = {"inner": 1, "middle": 2, "outer": 3}
        if rest.startswith("radial_"):
            name = rest[len("radial_") :]
            if name in ring and radial_bins == 3:
                return f"RadialDistribution_FracAtD_{ch}_{ring[name]}of3"
            found = re.fullmatch(r"ring(\d+)", name)
            if found:  # ring0 = edge … ring(n−1) = centre → CellProfiler counts from the centre
                return f"RadialDistribution_FracAtD_{ch}_{radial_bins - int(found.group(1))}of{radial_bins}"
        if rest == "center_periphery_ratio":
            return f"RadialDistribution_CenterPeripheryRatio_{ch}"
        texture = re.fullmatch(r"haralick_(.+)_s(\d+)", rest)
        if texture and texture.group(1) in HARALICK:
            return f"Texture_{HARALICK[texture.group(1)]}_{ch}_{texture.group(2)}"
        granularity = re.fullmatch(r"granularity_r(\d+)", rest)
        if granularity:
            return f"Granularity_{granularity.group(1)}_{ch}"
        return f"Intensity_{rest}_{ch}"
    child = re.fullmatch(r"(.+)_count", column)
    if child:
        return f"Children_{compartment(child.group(1))}_Count"
    return f"Other_{column}"


def rename_objects(
    objects_columns: list[str],
    object_type: str,
    prefixes: list[str],
    channel_map: dict[str, str],
    radial_bins: int = 3,
) -> dict[str, str]:
    """{our column: Compartment_… name} for one object type's feature columns."""
    ordered = sorted(prefixes, key=len, reverse=True)
    comp = compartment(object_type)
    out = {}
    for column in objects_columns:
        name = cp_name(column, ordered, channel_map, radial_bins)
        if name is not None:
            out[column] = f"{comp}_{name}"
    return out
