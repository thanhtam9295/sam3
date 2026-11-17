"""Builders for small synthetic plates in each vendor's on-disk layout.

Every pixel value encodes its coordinates so tests can prove the right file
ended up at the right (well, field, t, c, z) index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from numpy.typing import NDArray

SHAPE = (12, 16)


def plane(
    row: int, col: int, field: int, channel: int, z: int = 0, t: int = 0
) -> NDArray[np.uint16]:
    """Constant image whose value identifies (row, col, field, channel, z, t), all zero-based."""
    value = 1 + row * 5000 + col * 300 + field * 40 + channel * 10 + z + t * 5
    return np.full(SHAPE, value, dtype=np.uint16)


def make_operetta(
    root: Path,
    wells: list[tuple[int, int]],
    n_fields: int = 2,
    n_planes: int = 3,
    harmony_v5: bool = False,
) -> Path:
    """Harmony export: ``<root>/Images/Index.idx.xml`` plus one TIFF per plane."""
    images_dir = root / "Images"
    images_dir.mkdir(parents=True)
    channels = [("HOECHST 33342", 405, 456, "0.1"), ("Alexa 488", 488, 522, "0.2")]
    image_xml: list[str] = []
    for row, col in wells:
        for f in range(n_fields):
            for p in range(n_planes):
                for c, (name, ex, em, exposure) in enumerate(channels):
                    stem = f"r{row + 1:02d}c{col + 1:02d}f{f + 1:02d}p{p + 1:02d}"
                    url = f"{stem}-ch{c + 1}sk1fk1fl1.tiff"
                    tifffile.imwrite(images_dir / url, plane(row, col, f, c, p))
                    per_channel = f"""
      <ChannelName>{name}</ChannelName>
      <AcquisitionType>NipkowConfocal</AcquisitionType>
      <ChannelType>Fluorescence</ChannelType>
      <ImageResolutionX Unit="m">2.9901E-07</ImageResolutionX>
      <ImageResolutionY Unit="m">2.9901E-07</ImageResolutionY>
      <ImageSizeX>{SHAPE[1]}</ImageSizeX>
      <ImageSizeY>{SHAPE[0]}</ImageSizeY>
      <MaxIntensity>65536</MaxIntensity>
      <MainExcitationWavelength Unit="nm">{ex}</MainExcitationWavelength>
      <MainEmissionWavelength Unit="nm">{em}</MainEmissionWavelength>
      <ObjectiveMagnification Unit="">20</ObjectiveMagnification>
      <ObjectiveNA Unit="">0.8</ObjectiveNA>
      <ExposureTime Unit="s">{exposure}</ExposureTime>"""
                    image_xml.append(f"""
    <Image Version="1">
      <id>{row + 1:02d}{col + 1:02d}K1F{f + 1}P{p + 1}R{c + 1}</id>
      <State>Ok</State>
      <URL>{url}</URL>
      <Row>{row + 1}</Row>
      <Col>{col + 1}</Col>
      <FieldID>{f + 1}</FieldID>
      <PlaneID>{p + 1}</PlaneID>
      <TimepointID>0</TimepointID>
      <ChannelID>{c + 1}</ChannelID>{"" if harmony_v5 else per_channel}
      <PositionX Unit="m">{f * 0.0005}</PositionX>
      <PositionY Unit="m">-0.00025</PositionY>
      <PositionZ Unit="m">{p * 2e-6}</PositionZ>
    </Image>""")
    maps = ""
    if harmony_v5:
        entries = "".join(
            f"""
      <Entry ChannelID="{c + 1}">
        <ChannelName>{name}</ChannelName>
        <AcquisitionType>NipkowConfocal</AcquisitionType>
        <ChannelType>Fluorescence</ChannelType>
        <ImageResolutionX Unit="m">2.9901E-07</ImageResolutionX>
        <ImageResolutionY Unit="m">2.9901E-07</ImageResolutionY>
        <ImageSizeX>{SHAPE[1]}</ImageSizeX>
        <ImageSizeY>{SHAPE[0]}</ImageSizeY>
        <MaxIntensity>65536</MaxIntensity>
        <MainExcitationWavelength Unit="nm">{ex}</MainExcitationWavelength>
        <MainEmissionWavelength Unit="nm">{em}</MainEmissionWavelength>
        <ObjectiveMagnification Unit="">20</ObjectiveMagnification>
        <ObjectiveNA Unit="">0.8</ObjectiveNA>
        <ExposureTime Unit="s">{exposure}</ExposureTime>
      </Entry>"""
            for c, (name, ex, em, exposure) in enumerate(channels)
        )
        maps = f"\n  <Maps>\n    <Map>{entries}\n    </Map>\n  </Maps>"
    version = "HarmonyV5" if harmony_v5 else "HarmonyV4"
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<EvaluationInputData xmlns="http://www.perkinelmer.com/PEHH/{version}" Version="1">
  <Plates>
    <Plate>
      <PlateID>BARCODE-0042</PlateID>
      <PlateTypeName>384 CellCarrier Ultra</PlateTypeName>
      <PlateRows>16</PlateRows>
      <PlateColumns>24</PlateColumns>
    </Plate>
  </Plates>{maps}
  <Images>{"".join(image_xml)}
  </Images>
</EvaluationInputData>
"""
    (images_dir / "Index.idx.xml").write_text(xml, encoding="utf-8")
    return root


def _metaseries_description(wave_name: str, wavelength: int, pixel_size: float) -> str:
    return f"""<MetaData>
<ApplicationName>MetaMorph</ApplicationName>
<ApplicationVersion>6.5</ApplicationVersion>
<PlaneInfo>
<prop id="plane-type" type="string" value="plane"/>
<prop id="bits-per-pixel" type="int" value="16"/>
<prop id="spatial-calibration-state" type="bool" value="on"/>
<prop id="spatial-calibration-x" type="float" value="{pixel_size}"/>
<prop id="spatial-calibration-y" type="float" value="{pixel_size}"/>
<prop id="spatial-calibration-units" type="string" value="um"/>
<prop id="image-name" type="string" value="{wave_name}"/>
<prop id="wavelength" type="float" value="{wavelength}"/>
<custom-prop id="_MagNA_" type="float" value="0.75"/>
<custom-prop id="_MagSetting_" type="string" value="20X Plan Apo"/>
<custom-prop id="Exposure Time" type="string" value="150 ms"/>
</PlaneInfo>
</MetaData>"""


def make_imagexpress(
    root: Path,
    wells: list[tuple[int, int]],
    n_sites: int = 2,
    z_steps: int = 1,
    image_fn: Any = None,
) -> Path:
    """MetaXpress export: ``.HTD`` + GUID-suffixed TIFFs + thumbnails, optional ZStep folders."""
    root.mkdir(parents=True)
    waves = [("DAPI", 447), ("FITC", 520)]
    guid = "0A1B2C3D-1111-2222-3333-444455556666"
    from hcs_studio.model.plate import well_name

    def write(folder: Path, row: int, col: int, s: int, w: int, z: int) -> None:
        stem = f"Plate1_{well_name(row, col)}_s{s + 1}_w{w + 1}"
        tifffile.imwrite(
            folder / f"{stem}{guid}.tif",
            plane(row, col, s, w, z) if image_fn is None else image_fn(row, col, s, w),
            description=_metaseries_description(waves[w][0], waves[w][1], 0.65),
            software="MetaSeries",
            metadata=None,
        )
        tifffile.imwrite(folder / f"{stem}_thumb{guid}.tif", np.zeros((4, 4), dtype=np.uint8))

    for row, col in wells:
        for s in range(n_sites):
            for w in range(len(waves)):
                if z_steps == 1:
                    write(root, row, col, s, w, 0)
                else:
                    # Projection at ZStep_0 (must be ignored), planes in ZStep_1..n.
                    for z in range(z_steps + 1):
                        folder = root / f"ZStep_{z}"
                        folder.mkdir(exist_ok=True)
                        write(folder, row, col, s, w, z - 1 if z else 99)
    lines = [
        '"HTSInfoFile", Version 1.0',
        '"Description", "Screen 7"',
        '"TimePoints", 1',
        f'"ZSeries", {"TRUE" if z_steps > 1 else "FALSE"}',
        f'"ZSteps", {z_steps}',
        '"XWells", 12',
        '"YWells", 8',
        '"Sites", TRUE',
        f'"XSites", {n_sites}',
        '"YSites", 1',
        '"Waves", TRUE',
        f'"NWavelengths", {len(waves)}',
        *[f'"WaveName{i + 1}", "{name}"' for i, (name, _) in enumerate(waves)],
        '"EndFile"',
    ]
    (root / "Plate1.HTD").write_text("\r\n".join(lines) + "\r\n", encoding="latin-1")
    return root


class FakeND2:
    """Stand-in for ``nd2.ND2File`` exposing the attributes the reader uses."""

    def __init__(self, sizes: dict[str, int], position_names: list[str], data: Any) -> None:
        from types import SimpleNamespace as NS

        self.sizes = sizes
        self._data = data
        points = [
            NS(name=n, stagePositionUm=(100.0 * i, -50.0, 0.0))
            for i, n in enumerate(position_names)
        ]
        self.experiment = (
            [NS(type="XYPosLoop", parameters=NS(points=points))] if position_names else []
        )
        microscope = NS(
            objectiveMagnification=60.0,
            objectiveName="Plan Apo λ 60x Oil",
            objectiveNumericalAperture=1.4,
            immersionRefractiveIndex=1.515,
            pinholeDiameterUm=50.0,
            modalityFlags=["fluorescence", "spinningDiskConfocal"],
        )
        names = [("DAPI", 405.0, 450.0), ("PLA Cy3", 561.0, 595.0)][: sizes.get("C", 1)]
        self.metadata = NS(
            channels=[
                NS(
                    channel=NS(name=n, excitationLambdaNm=ex, emissionLambdaNm=em),
                    microscope=microscope,
                )
                for n, ex, em in names
            ]
        )
        self.attributes = NS(bitsPerComponentSignificant=12)

    def voxel_size(self) -> Any:
        from types import SimpleNamespace as NS

        return NS(x=0.108, y=0.108, z=0.3)

    def to_dask(self) -> Any:
        import dask.array as da

        return da.from_array(self._data, chunks=self._data.shape)

    def __enter__(self) -> FakeND2:
        return self

    def __exit__(self, *args: object) -> None:
        return None
