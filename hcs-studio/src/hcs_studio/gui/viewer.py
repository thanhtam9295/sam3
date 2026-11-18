"""Adapter around the embedded napari viewer.

All napari access goes through this class so the rest of the GUI does not depend
on napari internals. Embedding uses napari's own Qt window as a child widget,
which keeps its layer list and layer controls working.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import napari
import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QWidget

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import ImageData
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
        self._before_layers: list[Any] = []
        self._after_layers: list[Any] = []
        self._label_layers: list[Any] = []
        self._after_width = 0.0

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
        self._before_layers, self._after_layers, self._label_layers = [], [], []
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

    def show_comparison(self, before: ImageData, after: ImageData, mode: str = "After") -> None:
        """Show a field before and after processing. Both are (C, Z, Y, X) in memory."""
        self.viewer.layers.clear()
        self._label_layers = []
        self._before_layers = self._add_czyx(before, "")
        suffix = " ▸ corrected" if after.linear else " ▸ enhanced (display only)"
        self._after_layers = self._add_czyx(after, suffix)
        self._after_width = float(after.data.shape[-1] * (after.metadata.pixel_size_um or 1.0))
        self.viewer.canvas.overlays.scale_bar.visible = after.metadata.pixel_size_um is not None
        self.set_compare_mode(mode)

    def _add_czyx(self, image: ImageData, suffix: str) -> list[Any]:
        metadata = image.metadata
        calibrated = metadata.pixel_size_um is not None
        pixel = metadata.pixel_size_um or 1.0
        data = image.data
        scale: list[float] = [pixel, pixel]
        units = ["µm" if calibrated else "pixel"] * 2
        if data.shape[1] > 1:
            z_known = calibrated and metadata.z_step_um is not None
            scale.insert(0, metadata.z_step_um if z_known and metadata.z_step_um else pixel)
            units.insert(0, "µm" if z_known else "pixel")
        else:
            data = data[:, 0]
        layers = self.viewer.add_image(
            data,
            channel_axis=0,
            name=[c.name + suffix for c in metadata.channels],
            colormap=channel_colormaps(metadata.channels),
            scale=scale,
            units=units,
        )
        return list(layers) if isinstance(layers, list) else [layers]

    PROFILE_LAYER = "line profile"

    def start_line_profile(
        self,
        callback: Callable[[tuple[float, float], tuple[float, float]], None],
        pixel_um: float | None,
    ) -> Any:
        """Let the user draw a line; ``callback(start, end)`` gets its ends as (y, x) pixels
        of the field (the right-hand copy of side-by-side is mapped back)."""
        pixel = pixel_um or 1.0
        if self.PROFILE_LAYER in self.layer_names():
            layer = self.viewer.layers[self.PROFILE_LAYER]
        else:
            layer = self.viewer.add_shapes(
                name=self.PROFILE_LAYER,
                ndim=2,
                scale=[pixel, pixel],
                edge_color="yellow",
                edge_width=max(1.0, 2 * pixel) / pixel,
            )
        offset = self._after_width * 1.05 / pixel if self._after_width else 0.0

        def changed(event: Any = None) -> None:
            lines = [
                d for d, kind in zip(layer.data, layer.shape_type, strict=True) if kind == "line"
            ]
            if not lines:
                return
            (y0, x0), (y1, x1) = (tuple(float(v) for v in p[-2:]) for p in lines[-1][:2])
            if offset and min(x0, x1) >= offset:  # drawn on the "after" copy
                x0, x1 = x0 - offset, x1 - offset
            callback((y0, x0), (y1, x1))

        if not getattr(layer, "_hcs_profile_connected", False):
            layer.events.data.connect(changed)
            layer._hcs_profile_connected = True
        layer._hcs_profile_callback = changed
        self.viewer.layers.selection.active = layer
        layer.mode = "add_line"
        return layer

    def set_compare_mode(self, mode: str) -> None:
        """'After', 'Before' or 'Side by side' for the layers of ``show_comparison``."""
        if not self._after_layers:
            return
        if mode not in ("After", "Before", "Side by side"):
            raise ValueError(f"Unknown comparison mode {mode!r}")
        side = mode == "Side by side"
        for layer in self._before_layers:
            layer.visible = mode != "After"
        live_labels = [layer for layer in self._label_layers if layer in self.viewer.layers]
        for layer in [*self._after_layers, *live_labels]:
            layer.visible = mode != "Before"
            translate = [0.0] * layer.ndim
            translate[-1] = self._after_width * 1.05 if side else 0.0
            layer.translate = translate
        self.viewer.reset_view()

    # ------------------------------------------------------------- labels

    @staticmethod
    def _spatial(metadata: ImageMetadata, n_z: int) -> tuple[list[float], list[str]]:
        calibrated = metadata.pixel_size_um is not None
        pixel = metadata.pixel_size_um or 1.0
        scale: list[float] = [pixel, pixel]
        units = ["µm" if calibrated else "pixel"] * 2
        if n_z > 1:
            z_known = calibrated and metadata.z_step_um is not None
            scale.insert(0, metadata.z_step_um if z_known and metadata.z_step_um else pixel)
            units.insert(0, "µm" if z_known else "pixel")
        return scale, units

    def show_labels(
        self, label_sets: Any, metadata: ImageMetadata, translate_x: float = 0.0
    ) -> None:
        """Add (or replace) one editable labels layer per label set."""
        for label_set in label_sets:
            data = label_set.data if label_set.data.shape[0] > 1 else label_set.data[0]
            scale, units = self._spatial(metadata, label_set.data.shape[0])
            if label_set.name in self.viewer.layers:
                self.viewer.layers.remove(label_set.name)
            layer = self.viewer.add_labels(
                np.asarray(data, dtype=np.int32), name=label_set.name, scale=scale, units=units
            )
            translate = [0.0] * layer.ndim
            translate[-1] = translate_x
            layer.translate = translate
            layer.opacity = 0.45
            if layer not in self._label_layers:
                self._label_layers.append(layer)

    def labels_names(self) -> list[str]:
        return [layer.name for layer in self.viewer.layers if type(layer).__name__ == "Labels"]

    def get_labels(self, name: str) -> NDArray[np.int32]:
        """Labels of a layer as (Z, Y, X)."""
        data = np.asarray(self.viewer.layers[name].data, dtype=np.int32)
        return data if data.ndim == 3 else data[None]

    def set_labels(self, name: str, labels: NDArray[Any]) -> None:
        layer = self.viewer.layers[name]
        layer.data = np.asarray(labels if layer.ndim == 3 else labels[0], dtype=np.int32)

    def add_empty_labels(
        self, name: str, shape_zyx: tuple[int, int, int], metadata: ImageMetadata
    ) -> None:
        from hcs_studio.processing.base import LabelSet

        self.show_labels([LabelSet(name, np.zeros(shape_zyx, dtype=np.int32))], metadata)

    def prompt_layer(self, kind: str, name: str, like: str) -> Any:
        """Get or create a Points/Shapes layer for prompts, aligned with layer ``like``."""
        if name in self.viewer.layers:
            return self.viewer.layers[name]
        reference = self.viewer.layers[like]
        if kind == "points":
            layer = self.viewer.add_points(
                np.zeros((0, reference.ndim)),
                name=name,
                scale=reference.scale,
                size=6,
                face_color="lime" if "negative" not in name else "red",
            )
        else:
            layer = self.viewer.add_shapes(
                name=name,
                scale=reference.scale,
                ndim=reference.ndim,
                edge_color="lime",
                face_color="transparent",
                edge_width=2,
            )
        layer.translate = reference.translate
        return layer

    def current_z(self, like: str) -> int:
        """Z index of the displayed plane for layer ``like`` (0 for 2D)."""
        layer = self.viewer.layers[like]
        if layer.ndim < 3:
            return 0
        return int(np.rint(float(layer.world_to_data(self.viewer.dims.point)[-3])))

    def show_stages(self, stages: list[tuple[str, ImageData]]) -> None:
        """Show several versions of a field side by side (e.g. raw, corrected, enhanced)."""
        self.viewer.layers.clear()
        self._before_layers, self._after_layers, self._label_layers = [], [], []
        offset = 0.0
        for title, image in stages:
            layers = self._add_czyx(image, f" · {title}")
            for layer in layers:
                translate = [0.0] * layer.ndim
                translate[-1] = offset
                layer.translate = translate
            if image.labels:
                self.show_labels(image.labels, image.metadata, translate_x=offset)
                for layer in self.viewer.layers[-len(image.labels) :]:
                    layer.name = f"{layer.name} · {title}"
            offset += image.data.shape[-1] * (image.metadata.pixel_size_um or 1.0) * 1.05
        self.viewer.canvas.overlays.scale_bar.visible = True
        self.viewer.reset_view()

    # ------------------------------------------------- objects, categories, mosaics

    def on_object_click(self, layer_name: str, callback: Any) -> None:
        """Call ``callback(object_id)`` when the user clicks an object of a labels layer,
        whichever layer is selected and even while that layer is hidden (napari passes clicks
        only to the selected layer, so the handler sits on the viewer). Replaces any earlier
        click handler (``callback=None`` removes it)."""
        for owner, handler in getattr(self, "_click_handlers", []):
            if handler in owner.mouse_drag_callbacks:
                owner.mouse_drag_callbacks.remove(handler)
        self._click_handlers: list[tuple[Any, Any]] = []
        if callback is None or layer_name not in self.viewer.layers:
            return

        def on_click(viewer: Any, event: Any) -> None:
            if layer_name not in viewer.layers:
                return
            layer = viewer.layers[layer_name]
            data = np.asarray(layer.data)
            index = np.round(np.asarray(layer.world_to_data(event.position))).astype(int)
            index = index[-data.ndim :]
            if np.all(index >= 0) and np.all(index < data.shape):
                value = int(data[tuple(index)])
                if value > 0:
                    callback(value)

        self.viewer.mouse_drag_callbacks.append(on_click)
        self._click_handlers.append((self.viewer, on_click))

    def highlight_object(self, layer_name: str, object_id: int) -> None:
        """Show only one object of a labels layer (0 shows all again)."""
        if layer_name not in self.viewer.layers:
            return
        layer = self.viewer.layers[layer_name]
        layer.selected_label = max(1, object_id)
        layer.show_selected_label = object_id > 0

    def paint_categories(
        self,
        like: str,
        categories: dict[int, int],
        colours: Sequence[str],
        name: str = "categories",
    ) -> None:
        """Colour the objects of labels layer ``like`` by category: ``categories`` maps object
        ID → category index (into ``colours``); objects without a category are not drawn.
        The layer ``like`` is hidden while the categories are shown."""
        from napari.utils.colormaps import DirectLabelColormap

        reference = self.viewer.layers[like]
        labels = np.asarray(reference.data)
        lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
        for object_id, category in categories.items():
            if 0 < object_id < len(lut):
                lut[object_id] = category + 1
        painted = lut[labels]
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        colour_map: dict[Any, Any] = {None: "transparent", 0: "transparent"}
        colour_map.update({k + 1: c for k, c in enumerate(colours)})
        layer = self.viewer.add_labels(
            painted,
            name=name,
            scale=reference.scale,
            units=reference.units,
            colormap=DirectLabelColormap(color_dict=colour_map),
            opacity=0.85,
        )
        layer.translate = reference.translate
        # The plain masks (random colours) would hide the categories; they can be shown again
        # from the layer list.
        reference.visible = False
        if layer not in self._label_layers:
            self._label_layers.append(layer)

    def show_rgb(self, name: str, rgb: NDArray[Any], um_per_px: float | None) -> None:
        """Replace the view with one RGB image (e.g. a whole-well mosaic)."""
        self.viewer.layers.clear()
        self._before_layers, self._after_layers, self._label_layers = [], [], []
        scale = [um_per_px, um_per_px] if um_per_px else [1.0, 1.0]
        units = ["µm" if um_per_px else "pixel"] * 2
        self.viewer.add_image(rgb, name=name, rgb=True, scale=scale, units=units)
        self.viewer.reset_view()

    def layer_names(self) -> list[str]:
        return [layer.name for layer in self.viewer.layers]

    def close(self) -> None:
        self.viewer.close()
