"""Processing-module interface.

A module declares pydantic parameters, may learn plate-level state in ``fit`` (for
example a flat-field profile from many fields) and transforms one field in ``apply``.
Images carry a ``linear`` flag: once a step changes intensities non-linearly the
result is for viewing and segmentation only, never for intensity measurement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from hcs_studio.model.layout import WellAnnotation
from hcs_studio.model.plate import ImageMetadata, Plate
from hcs_studio.registry import Registry

FloatImage = NDArray[np.float32]


class ModuleError(Exception):
    """A module cannot run with the given parameters, metadata or data."""


class ModuleParams(BaseModel):
    """Base class for module parameters: unknown keys are errors, values are validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ChannelParams(ModuleParams):
    channels: list[int] | None = Field(
        default=None,
        description="Channel numbers to process, starting at 1. Empty = all channels.",
    )

    def selected(self, n_channels: int) -> list[int]:
        """Zero-based indices of the channels to process."""
        if not self.channels:
            return list(range(n_channels))
        bad = [c for c in self.channels if not 1 <= c <= n_channels]
        if bad:
            raise ModuleError(f"Channel(s) {bad} do not exist; the image has {n_channels} channels")
        return sorted({c - 1 for c in self.channels})


@dataclass(frozen=True)
class ImageData:
    """One timepoint of one field: ``data`` is float32 (C, Z, Y, X)."""

    data: FloatImage
    metadata: ImageMetadata
    #: False once any step has changed intensities non-linearly (display-only data).
    linear: bool = True
    history: tuple[str, ...] = ()
    #: Per-step reports (e.g. deconvolution iterations, residual, runtime).
    reports: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(f"ImageData needs (C, Z, Y, X); got shape {self.data.shape}")
        if self.data.shape[0] != len(self.metadata.channels):
            raise ValueError("Channel count of data and metadata differ")

    def derive(
        self,
        data: FloatImage,
        step: str,
        *,
        linear: bool = True,
        metadata: ImageMetadata | None = None,
        report: dict[str, Any] | None = None,
    ) -> ImageData:
        """New image after ``step``. Linearity can only be lost, never regained."""
        return ImageData(
            data=np.ascontiguousarray(data, dtype=np.float32),
            metadata=metadata or self.metadata,
            linear=self.linear and linear,
            history=(*self.history, step),
            reports=(*self.reports, report) if report is not None else self.reports,
        )

    @classmethod
    def from_array(cls, array: Any, metadata: ImageMetadata) -> ImageData:
        return cls(data=np.asarray(array, dtype=np.float32), metadata=metadata)


class FitContext(Protocol):
    """What a module may look at while learning plate-level state."""

    plate: Plate
    layout: dict[tuple[int, int], WellAnnotation]

    def sample(
        self, max_fields: int, wells: Sequence[tuple[int, int]] | None = None
    ) -> Iterator[ImageData]:
        """Fields spread evenly over the plate (or ``wells``), processed by all earlier steps."""
        ...


class Module(ABC):
    """One processing step. Subclasses set ``name``, ``title`` and ``Params``."""

    name: ClassVar[str]
    title: ClassVar[str]
    Params: ClassVar[type[ModuleParams]]
    #: False for steps whose output must not be used for intensity measurement.
    preserves_linearity: ClassVar[bool] = True
    #: True when ``fit`` must run on the plate before ``apply``.
    needs_fit: ClassVar[bool] = False

    def __init__(self, params: ModuleParams | None = None) -> None:
        self.params: Any = params if params is not None else self.Params()

    def validate(self, metadata: ImageMetadata) -> None:  # noqa: B027
        """Raise ``ModuleError`` if the step cannot run on images with this metadata."""

    def fit(self, context: FitContext) -> Any:
        return None

    @abstractmethod
    def apply(self, image: ImageData, state: Any) -> ImageData: ...

    def describe_state(self, state: Any) -> dict[str, Any]:
        """JSON-serialisable summary of the fitted state, for the run manifest."""
        return {}

    def output_metadata(self, metadata: ImageMetadata) -> ImageMetadata:
        """Metadata of the output image (changed by projection and unmixing)."""
        return metadata


MODULES: Registry[type[Module]] = Registry("module")


def load_builtin_modules() -> None:
    from hcs_studio.processing import (  # noqa: F401
        autofluorescence,
        deconvolution,
        enhance,
        flatfield,
    )


def create_module(name: str, params: dict[str, Any] | None = None) -> Module:
    """Instantiate a registered module, validating its parameters."""
    load_builtin_modules()
    cls = MODULES.get(name)
    return cls(cls.Params.model_validate(params or {}))


def um_to_px(value_um: float, metadata: ImageMetadata, what: str) -> float:
    if metadata.pixel_size_um is None:
        raise ModuleError(
            f"{what} is given in µm but the pixel size is unknown; set it in the Metadata panel"
        )
    return value_um / metadata.pixel_size_um


def block_reduce_mean(plane: NDArray[Any], max_size: int) -> FloatImage:
    """Downsample a 2D image by block averaging so its longest side is at most ``max_size``."""
    factor = max(1, -(-max(plane.shape) // max_size))
    if factor == 1:
        return plane.astype(np.float32)
    height, width = (plane.shape[0] // factor) * factor, (plane.shape[1] // factor) * factor
    if height == 0 or width == 0:
        return plane.astype(np.float32)
    blocks = plane[:height, :width].reshape(height // factor, factor, width // factor, factor)
    return np.asarray(blocks.mean(axis=(1, 3)), dtype=np.float32)


def resize_to(small: NDArray[Any], shape: tuple[int, int]) -> FloatImage:
    """Bilinear resize of a smooth 2D profile to ``shape``."""
    import cv2

    return np.asarray(
        cv2.resize(small.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR),
        dtype=np.float32,
    )


__all__ = [
    "MODULES",
    "ChannelParams",
    "FitContext",
    "FloatImage",
    "ImageData",
    "Module",
    "ModuleError",
    "ModuleParams",
    "block_reduce_mean",
    "create_module",
    "load_builtin_modules",
    "replace",
    "resize_to",
    "um_to_px",
]
