"""Adapter around the embedded napari viewer.

All napari access goes through this class so the rest of the GUI does not depend
on napari internals. Embedding uses napari's own Qt window as a child widget,
which keeps its layer list and layer controls working.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import napari
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.utils.display import channel_colormaps

_CHANNEL_COLORMAPS = ("blue", "green", "red", "magenta", "cyan", "yellow", "gray")


class ViewerPanel:
    """Owns one napari viewer and exposes the widget to embed."""

    def __init__(self) -> None:
        self.viewer = napari.Viewer(show=False)
        with warnings.catch_warnings():
            # napari marks direct window access as private; there is no public
            # embedding API, so the dependency is confined to this adapter.
            warnings.simplefilter("ignore")
            window: QMainWindow = self.viewer.window._qt_window
        window.setWindowFlags(Qt.WindowType.Widget)
        window.menuBar().setVisible(False)
        window.statusBar().setVisible(False)
        self.widget: QWidget = window

    def show_channels(
        self,
        image: NDArray[np.generic],
        channel_names: Sequence[str],
        pixel_size_um: float,
        z_step_um: float | None = None,
    ) -> None:
        """Replace the displayed layers with a (C, [Z,] Y, X) image, scaled in micrometres."""
        if image.ndim not in (3, 4):
            raise ValueError(f"Expected (C, Y, X) or (C, Z, Y, X), got shape {image.shape}")
        if image.shape[0] != len(channel_names):
            raise ValueError(
                f"{image.shape[0]} channels in the image but {len(channel_names)} channel names"
            )
        if image.ndim == 4 and z_step_um is None:
            raise ValueError("z_step_um is required for a Z stack")
        scale: tuple[float, ...] = (pixel_size_um, pixel_size_um)
        if image.ndim == 4:
            assert z_step_um is not None
            scale = (z_step_um, *scale)
        self.viewer.layers.clear()
        self.viewer.add_image(
            image,
            channel_axis=0,
            name=list(channel_names),
            colormap=[
                _CHANNEL_COLORMAPS[i % len(_CHANNEL_COLORMAPS)] for i in range(image.shape[0])
            ],
            scale=scale,
            units=("µm",) * len(scale),
        )
        self.viewer.canvas.overlays.scale_bar.visible = True
        self.viewer.reset_view()

    def show_field(self, data: Any, metadata: ImageMetadata) -> None:
        """Show a lazy (T, C, Z, Y, X) field with channel names, colours and physical scale.

        Singleton T and Z axes are dropped. When the pixel size is unknown the image is
        shown in pixel units, without a scale bar.
        """
        if data.ndim != 5:
            raise ValueError(f"Expected (T, C, Z, Y, X), got shape {data.shape}")
        n_t, n_c, n_z = data.shape[:3]
        if n_c != len(metadata.channels):
            raise ValueError(f"{n_c} channels in the image but {len(metadata.channels)} described")
        calibrated = metadata.pixel_size_um is not None
        pixel = metadata.pixel_size_um or 1.0
        scale: list[float] = [pixel, pixel]
        units: list[str] = ["µm" if calibrated else "pixel"] * 2
        if n_z > 1:
            z_known = calibrated and metadata.z_step_um is not None
            scale.insert(0, metadata.z_step_um if z_known and metadata.z_step_um else pixel)
            units.insert(0, "µm" if z_known else "pixel")
        else:
            data = data[:, :, 0]
        if n_t > 1:
            scale.insert(0, 1.0)
            units.insert(0, "pixel")
        else:
            data = data[0]
        self.viewer.layers.clear()
        self.viewer.add_image(
            data,
            channel_axis=1 if n_t > 1 else 0,
            name=[c.name for c in metadata.channels],
            colormap=channel_colormaps(metadata.channels),
            scale=scale,
            units=units,
        )
        self.viewer.dims.axis_labels = [a for a, n in (("t", n_t), ("z", n_z)) if n > 1] + [
            "y",
            "x",
        ]
        self.viewer.canvas.overlays.scale_bar.visible = calibrated
        self.viewer.reset_view()

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.viewer.layers]

    def close(self) -> None:
        self.viewer.close()
