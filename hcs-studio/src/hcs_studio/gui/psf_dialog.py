"""PSF tool: theoretical or bead-measured PSF, XY/XZ projections, FWHM, save."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.gui.workers import BackgroundTask
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing import psf as psf_tools
from hcs_studio.processing.base import ModuleError

_VIEW = 240


@dataclass
class PsfResult:
    psf: Any
    pixel_size_um: float
    z_step_um: float | None
    description: str
    notes: list[str] = field(default_factory=list)


def projection_images(
    psf: NDArray[Any], pixel_size_um: float, z_step_um: float | None
) -> tuple[NDArray[np.uint8], NDArray[np.uint8] | None]:
    """8-bit XY and XZ maximum projections, gamma 0.5 so the faint rings are visible.

    The XZ view is stretched along Z so that it has the physical aspect ratio.
    """
    from scipy import ndimage

    def to8(image: NDArray[Any]) -> NDArray[np.uint8]:
        peak = float(np.max(image))
        scaled = np.sqrt(np.clip(image / peak, 0, 1)) if peak > 0 else image
        return np.asarray(scaled * 255, dtype=np.uint8)

    xy = to8(psf.max(axis=0))
    if psf.shape[0] == 1 or z_step_um is None:
        return xy, None
    xz = psf.max(axis=1)
    stretch = z_step_um / pixel_size_um
    xz = ndimage.zoom(xz, (stretch, 1.0), order=1)
    return xy, to8(xz)


def _pixmap(gray: NDArray[np.uint8]) -> QPixmap:
    data = np.ascontiguousarray(gray)
    height, width = data.shape
    image = QImage(data.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(image).scaled(
        _VIEW, _VIEW, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
    )


class PsfDialog(QDialog):
    """Emits ``psf_saved(path)`` after a PSF has been written to disk."""

    psf_saved = Signal(object)

    def __init__(
        self,
        metadata: ImageMetadata | None,
        current_field: Any | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("PSF tool")
        self.resize(640, 640)
        self._metadata = metadata
        #: Lazy (T, C, Z, Y, X) array of the field in the viewer, used as a bead stack.
        self._current_field = current_field
        self._bead_file: Path | None = None
        self.psf_result: PsfResult | None = None
        self.task = BackgroundTask(self)
        self.task.finished.connect(self._on_finished)
        self.task.failed.connect(self._on_failed)

        self.source = QComboBox()
        self.source.addItems(["Theoretical (from metadata)", "Measured (bead stack)"])
        self.channel = QComboBox()
        if metadata is not None:
            self.channel.addItems([c.name for c in metadata.channels])

        theory = QWidget()
        theory_form = QFormLayout(theory)
        self.model = QComboBox()
        self.model.addItems(["vectorial", "scalar", "gaussian"])
        self.sample_ri = QDoubleSpinBox()
        self.sample_ri.setRange(1.0, 1.7)
        self.sample_ri.setDecimals(3)
        self.sample_ri.setValue(1.33)
        self.pinhole = QDoubleSpinBox()
        self.pinhole.setRange(0.0, 10.0)
        self.pinhole.setDecimals(2)
        self.pinhole.setSpecialValueText("from metadata")
        theory_form.addRow("Model", self.model)
        theory_form.addRow("Sample refractive index", self.sample_ri)
        theory_form.addRow("Confocal pinhole (Airy units)", self.pinhole)

        beads = QWidget()
        bead_form = QFormLayout(beads)
        self.bead_source = QComboBox()
        self.bead_source.addItems(["Field in the viewer", "File…"])
        self.bead_source.currentIndexChanged.connect(self._on_bead_source)
        self.bead_file_label = QLabel("")
        self.size_xy = QSpinBox()
        self.size_xy.setRange(5, 129)
        self.size_xy.setSingleStep(2)
        self.size_xy.setValue(33)
        self.size_z = QSpinBox()
        self.size_z.setRange(0, 65)
        self.size_z.setSingleStep(2)
        self.size_z.setSpecialValueText("whole stack")
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.01, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.2)
        self.saturation = QSpinBox()
        self.saturation.setRange(0, 1 << 30)
        self.saturation.setSpecialValueText("from bit depth")
        bead_form.addRow("Bead stack", self.bead_source)
        bead_form.addRow("", self.bead_file_label)
        bead_form.addRow("Crop XY (px, odd)", self.size_xy)
        bead_form.addRow("Crop Z (planes, odd)", self.size_z)
        bead_form.addRow("Detection threshold (of brightest)", self.threshold)
        bead_form.addRow("Saturation level", self.saturation)

        self.pages = QStackedWidget()
        self.pages.addWidget(theory)
        self.pages.addWidget(beads)
        self.source.currentIndexChanged.connect(self.pages.setCurrentIndex)

        self.compute_button = QPushButton("Compute PSF")
        self.compute_button.clicked.connect(self.compute)
        self.save_button = QPushButton("Save PSF…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._choose_save)

        self.xy_view = QLabel("XY")
        self.xz_view = QLabel("XZ")
        for view in (self.xy_view, self.xz_view):
            view.setFixedSize(_VIEW, _VIEW)
            view.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view.setStyleSheet("background: black; color: #888;")
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        form = QFormLayout()
        form.addRow("Source", self.source)
        form.addRow("Channel", self.channel)
        views = QHBoxLayout()
        for title, view in (
            ("XY (max projection)", self.xy_view),
            ("XZ (max projection)", self.xz_view),
        ):
            box = QGroupBox(title)
            QVBoxLayout(box).addWidget(view)
            views.addWidget(box)
        buttons = QHBoxLayout()
        buttons.addWidget(self.compute_button)
        buttons.addWidget(self.save_button)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.pages)
        layout.addLayout(buttons)
        layout.addLayout(views)
        layout.addWidget(self.info, 1)
        if metadata is None:
            self.info.setText("Open a plate first: the PSF needs its optical metadata.")
            self.compute_button.setEnabled(False)

    def _on_bead_source(self, index: int) -> None:
        if index == 1:
            file, _ = QFileDialog.getOpenFileName(
                self, "Bead stack", "", "Images (*.nd2 *.tif *.tiff *.ome.tif *.ome.tiff)"
            )
            self.set_bead_file(Path(file) if file else None)

    def set_bead_file(self, path: Path | None) -> None:
        self._bead_file = path
        self.bead_file_label.setText(path.name if path else "")
        if path is None:
            self.bead_source.setCurrentIndex(0)

    # --------------------------------------------------------------- compute

    def compute(self) -> None:
        metadata = self._metadata
        if metadata is None:
            return
        channel = self.channel.currentIndex()
        try:
            work = (
                self._theoretical(metadata, channel)
                if self.source.currentIndex() == 0
                else self._measured(metadata, channel)
            )
        except ModuleError as exc:
            self._on_failed(str(exc))
            return
        self.compute_button.setEnabled(False)
        self.info.setText("Computing …")
        self.task.request(work)

    def _theoretical(self, metadata: ImageMetadata, channel: int) -> Any:
        pinhole = self.pinhole.value() or None
        spec, notes = psf_tools.spec_from_metadata(
            metadata,
            channel,
            sample_ri=self.sample_ri.value(),
            pinhole_au=pinhole,
            model=self.model.currentText(),  # type: ignore[arg-type]
        )
        three_d = metadata.z_step_um is not None
        if not three_d:
            notes.append("no Z step in the metadata: 2D PSF")
        kind = "confocal" if spec.confocal else "widefield"

        def work() -> PsfResult:
            return PsfResult(
                psf=psf_tools.theoretical_psf(spec, three_d),
                pixel_size_um=spec.pixel_size_um,
                z_step_um=metadata.z_step_um if three_d else None,
                description=f"Theoretical {kind} PSF ({spec.model}), NA {spec.na}, "
                f"emission {spec.emission_nm:g} nm, immersion RI {spec.immersion_ri}",
                notes=notes,
            )

        return work

    def _measured(self, metadata: ImageMetadata, channel: int) -> Any:
        size_xy = self.size_xy.value() | 1
        size_z = (self.size_z.value() | 1) if self.size_z.value() else None
        threshold = self.threshold.value()
        from_file = self.bead_source.currentIndex() == 1
        saturation: float | None = float(self.saturation.value()) or None
        if saturation is None and metadata.bit_depth:
            saturation = float(2**metadata.bit_depth - 1)
        bead_file, field_data = self._bead_file, self._current_field
        if from_file and bead_file is None:
            raise ModuleError("Choose a bead-stack file")
        if not from_file and field_data is None:
            raise ModuleError("Open the bead field in the viewer first, or choose a file")

        def work() -> PsfResult:
            pixel, z_step, data = metadata.pixel_size_um, metadata.z_step_um, field_data
            if from_file:
                from hcs_studio.io.base import open_plate

                assert bead_file is not None
                plate = open_plate(bead_file)
                data = next(iter(plate.wells.values())).fields[0].load()
                pixel, z_step = plate.metadata.pixel_size_um, plate.metadata.z_step_um
            if pixel is None:
                raise ModuleError("The bead stack has no pixel size")
            assert data is not None
            if channel >= data.shape[1]:
                raise ModuleError(f"The bead stack has no channel {channel + 1}")
            stack = np.asarray(data[0, channel], dtype=np.float32)
            result = psf_tools.extract_psf_from_beads(
                stack,
                size_xy=size_xy,
                size_z=size_z,
                threshold_rel=threshold,
                saturation=saturation,
            )
            return PsfResult(
                psf=result.psf,
                pixel_size_um=pixel,
                z_step_um=z_step if stack.shape[0] > 1 else None,
                description=f"Measured PSF: average of {result.n_beads} beads "
                f"({result.n_rejected} rejected: edge, saturated, crowded or off-centre)",
            )

        return work

    def _on_finished(self, result: PsfResult) -> None:
        self.psf_result = result
        self.compute_button.setEnabled(True)
        self.save_button.setEnabled(True)
        xy, xz = projection_images(result.psf, result.pixel_size_um, result.z_step_um)
        self.xy_view.setPixmap(_pixmap(xy))
        if xz is None:
            self.xz_view.setPixmap(QPixmap())
            self.xz_view.setText("2D PSF")
        else:
            self.xz_view.setPixmap(_pixmap(xz))
        fwhm = psf_tools.measure_fwhm(
            result.psf, (result.z_step_um, result.pixel_size_um, result.pixel_size_um)
        )
        widths = ", ".join(
            f"{axis.upper()} {fwhm[axis] * 1000:.0f} nm"  # type: ignore[operator]
            for axis in "xyz"
            if fwhm[axis]
        )
        lines = [
            result.description,
            f"Size (Z, Y, X): {tuple(result.psf.shape)} voxels; voxel {result.pixel_size_um:g} µm"
            + (f" × {result.z_step_um:g} µm" if result.z_step_um else ""),
            f"FWHM: {widths or 'not measurable'}",
        ]
        lines += [f"Assumption: {note}" for note in result.notes]
        self.info.setText("\n".join(lines))

    def _on_failed(self, message: str) -> None:
        self.compute_button.setEnabled(self._metadata is not None)
        self.info.setText(f"Could not compute the PSF: {message}")

    # ------------------------------------------------------------------ save

    def _choose_save(self) -> None:
        file, _ = QFileDialog.getSaveFileName(
            self, "Save PSF", "psf.ome.tif", "OME-TIFF (*.ome.tif)"
        )
        if file:
            self.save(Path(file))

    def save(self, path: Path) -> None:
        if self.psf_result is None:
            return
        try:
            psf_tools.save_psf(
                path, self.psf_result.psf, self.psf_result.pixel_size_um, self.psf_result.z_step_um
            )
        except OSError as exc:
            self.info.setText(f"Not saved: {exc}")
            return
        self.info.setText(self.info.text() + f"\nSaved to {path}")
        self.psf_saved.emit(path)

    def closeEvent(self, event: Any) -> None:
        self.task.wait()
        super().closeEvent(event)
