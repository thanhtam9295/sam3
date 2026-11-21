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
class LabelSet:
    """Segmented objects of one kind: int32 (Z, Y, X), 0 = background, 1..N = objects.

    ``parent_ids[i]`` is the ID of the parent object (in label set ``parent``) of object
    ``i``; index 0 is unused. Cells expanded from nuclei share their nucleus' ID.
    """

    name: str
    data: NDArray[np.int32]
    parent: str | None = None
    parent_ids: NDArray[np.int32] | None = None
    source: str = ""

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Label set '{self.name}' needs (Z, Y, X); got {self.data.shape}")
        if (self.parent is None) != (self.parent_ids is None):
            raise ValueError("parent and parent_ids must be given together")

    @property
    def count(self) -> int:
        """Number of objects (distinct non-zero IDs)."""
        ids = np.unique(self.data)
        return int((ids > 0).sum())


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
    #: Segmentations made so far, in order (nuclei, cells, spots, ...).
    labels: tuple[LabelSet, ...] = ()

    def __post_init__(self) -> None:
        if self.data.ndim != 4:
            raise ValueError(f"ImageData needs (C, Z, Y, X); got shape {self.data.shape}")
        if self.data.shape[0] != len(self.metadata.channels):
            raise ValueError("Channel count of data and metadata differ")
        for label_set in self.labels:
            if label_set.data.shape != self.data.shape[1:]:
                raise ValueError(
                    f"Label set '{label_set.name}' has shape {label_set.data.shape}; the image "
                    f"is {self.data.shape[1:]}"
                )

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
            labels=self.labels if data.shape[1:] == self.data.shape[1:] else (),
        )

    def label_set(self, name: str) -> LabelSet:
        for label_set in self.labels:
            if label_set.name == name:
                return label_set
        known = ", ".join(ls.name for ls in self.labels) or "none yet"
        raise ModuleError(f"No segmentation named '{name}' (available: {known})")

    def with_labels(
        self, label_set: LabelSet, step: str, report: dict[str, Any] | None = None
    ) -> ImageData:
        """Add (or replace, by name) a label set. Pixel data and linearity are unchanged."""
        kept = tuple(ls for ls in self.labels if ls.name != label_set.name)
        return ImageData(
            data=self.data,
            metadata=self.metadata,
            linear=self.linear,
            history=(*self.history, step),
            reports=(*self.reports, report) if report is not None else self.reports,
            labels=(*kept, label_set),
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

    #: True for steps that must run one at a time across worker processes (e.g. a large
    #: model loaded per process).
    exclusive: ClassVar[bool] = False

    def validate(self, metadata: ImageMetadata) -> None:  # noqa: B027
        """Raise ``ModuleError`` if the step cannot run on images with this metadata."""

    def uses_gpu(self) -> bool:
        """Whether this step will run on the GPU (GPU steps are serialised across workers)."""
        return False

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


@dataclass(frozen=True)
class MeasureContext:
    """What a measurement step sees for one field and timepoint."""

    #: The field as read, before any step (for saturation checks).
    raw: ImageData
    #: The last linear image: the data intensities are measured on.
    corrected: ImageData
    #: Segmentations available at this point in the pipeline.
    labels: tuple[LabelSet, ...]

    def check_shapes(self) -> None:
        for label_set in self.labels:
            if label_set.data.shape != self.corrected.data.shape[1:]:
                raise ModuleError(
                    f"Label set '{label_set.name}' has shape {label_set.data.shape}, but the "
                    f"linear image is {self.corrected.data.shape[1:]}: put Z projection before "
                    "any display-only step"
                )


class MeasurementModule(Module):
    """A step that produces tables instead of changing the image.

    ``apply`` passes the image through; the runner calls ``measure`` with the raw image,
    the last linear image and the current segmentations.
    """

    def apply(self, image: ImageData, state: Any) -> ImageData:
        return image

    @abstractmethod
    def measure(self, context: MeasureContext) -> dict[str, Any]:
        """Tables (pandas DataFrames) by name, e.g. {"objects": ..., "coloc": ...}."""


MODULES: Registry[type[Module]] = Registry("module")


def load_builtin_modules() -> None:
    from hcs_studio.analysis import (  # noqa: F401
        classifier,
        clustering,
        measure,
        run_steps,
        spatial,
        stitching,
        table_steps,
    )
    from hcs_studio.assays import apoptosis as _apoptosis  # noqa: F401
    from hcs_studio.assays import cell_cycle as _cell_cycle  # noqa: F401
    from hcs_studio.assays import foci as _foci  # noqa: F401
    from hcs_studio.assays import proliferation as _proliferation  # noqa: F401
    from hcs_studio.assays import translocation as _translocation  # noqa: F401
    from hcs_studio.processing import (  # noqa: F401
        autofluorescence,
        background,
        deconvolution,
        enhance,
        flatfield,
        projection,
    )
    from hcs_studio.qc import objects as _object_qc  # noqa: F401
    from hcs_studio.segmentation import (  # noqa: F401
        cellpose_backend,
        classical,
        objects,
        sam3_backend,
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
    "LabelSet",
    "MeasureContext",
    "MeasurementModule",
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
