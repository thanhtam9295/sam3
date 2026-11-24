"""Cellpose 4 backend (Cellpose-SAM and DINO models), plus fine-tuning on edited masks.

Cellpose is imported lazily so the rest of the app works without it. Models are cached
per process, so a plate run loads each model once.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    LabelSet,
    Module,
    ModuleError,
    ModuleParams,
    um_to_px,
)
from hcs_studio.segmentation.objects import overlap_parents, relabel_sequential

#: Built-in Cellpose 4.2 models. cpdino needs the dinov3 package (see the docs).
BUILTIN_MODELS = ("cpsam_v2", "cpsam", "cpdino", "cpdino-vitb")

_models: dict[tuple[str, bool], Any] = {}
_lock = threading.Lock()


def _cellpose() -> Any:
    try:
        from cellpose import models
    except ImportError:
        raise ModuleError(
            'Cellpose is not installed; install it with: pip install "hcs-studio[segmentation]"'
        ) from None
    return models


def use_gpu(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import torch

        available = bool(torch.cuda.is_available())
    except ImportError:
        available = False
    if device == "gpu" and not available:
        raise ModuleError("No CUDA GPU is available to PyTorch; choose device 'auto' or 'cpu'")
    return available


def get_model(model: str, gpu: bool) -> Any:
    """A loaded Cellpose model; cached. ``model`` is a built-in name or a model file path."""
    key = (model, gpu)
    with _lock:
        if key not in _models:
            models = _cellpose()
            if model not in BUILTIN_MODELS and not Path(model).is_file():
                raise ModuleError(
                    f"Unknown Cellpose model '{model}': use one of {', '.join(BUILTIN_MODELS)} "
                    "or the path of a trained model file"
                )
            try:
                _models[key] = models.CellposeModel(gpu=gpu, pretrained_model=model)
            except Exception as exc:  # download failures, missing dinov3, corrupt files
                raise ModuleError(f"Could not load Cellpose model '{model}': {exc}") from None
        return _models[key]


def model_file(model: str) -> str:
    """A built-in name or file path; registry references become the registered file."""
    from hcs_studio.training.registry import is_ref, resolve_path

    if not is_ref(model):
        return model
    try:
        return resolve_path(model, "cellpose")
    except ValueError as exc:
        raise ModuleError(str(exc)) from None


def clear_model_cache() -> None:
    with _lock:
        _models.clear()


class CellposeParams(ModuleParams):
    channel: int = Field(
        ge=1, description="Channel to segment (nuclear stain, or cytoplasm/membrane)."
    )
    second_channel: int | None = Field(
        default=None,
        ge=1,
        description="Optional extra channel (e.g. nuclei when segmenting cells).",
    )
    output: str = Field(default="nuclei", description="Name of the new label set.")
    model: str = Field(
        default="cpsam_v2",
        description=(
            "cpsam_v2, cpsam, cpdino, cpdino-vitb, a trained model file, or a registered "
            "model (registry:NAME@VERSION)."
        ),
    )
    diameter_um: float | None = Field(
        default=None,
        gt=0,
        description="Typical object diameter in µm; empty = let the model decide.",
    )
    flow_threshold: float = Field(
        default=0.4, ge=0, description="Higher = more (less certain) objects."
    )
    cellprob_threshold: float = Field(
        default=0.0, ge=-6, le=6, description="Lower = larger objects."
    )
    min_size_px: int = Field(default=15, ge=0, description="Remove objects smaller than this.")
    do_3d: bool = Field(default=False, description="Segment Z stacks in 3D (else plane by plane).")
    stitch_threshold: float = Field(
        default=0.0, ge=0, le=1, description="2D mode on stacks: stitch planes above this overlap."
    )
    parent: str | None = Field(
        default=None,
        description="Link each object to the object it overlaps most in this label set.",
    )
    batch_size: int = Field(default=8, ge=1, description="Tiles per GPU batch.")
    device: str = Field(default="auto", pattern="^(auto|cpu|gpu)$", description="auto, cpu or gpu.")


@MODULES.register("segment_cellpose")
class SegmentCellpose(Module):
    name = "segment_cellpose"
    title = "Segmentation: Cellpose"
    Params = CellposeParams

    def uses_gpu(self) -> bool:
        try:
            return use_gpu(self.params.device)
        except ModuleError:
            return False

    def validate(self, metadata: ImageMetadata) -> None:
        params: CellposeParams = self.params
        for channel in (params.channel, params.second_channel):
            if channel is not None and channel > len(metadata.channels):
                raise ModuleError(f"Channel {channel} does not exist")
        if params.diameter_um is not None:
            um_to_px(params.diameter_um, metadata, "The object diameter")
        model_file(params.model)

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: CellposeParams = self.params
        self.validate(image.metadata)
        gpu = use_gpu(params.device)
        model = get_model(model_file(params.model), gpu)
        channels = [params.channel - 1] + (
            [params.second_channel - 1] if params.second_channel else []
        )
        data = image.data[channels]  # (C, Z, Y, X)
        diameter = (
            um_to_px(params.diameter_um, image.metadata, "The object diameter")
            if params.diameter_um is not None
            else None
        )
        n_z = data.shape[1]
        kwargs: dict[str, Any] = {
            "diameter": diameter,
            "flow_threshold": params.flow_threshold,
            "cellprob_threshold": params.cellprob_threshold,
            "min_size": params.min_size_px,
            "batch_size": params.batch_size,
            "channel_axis": 0,
        }
        if n_z > 1 and params.do_3d:
            z_step, pixel = image.metadata.z_step_um, image.metadata.pixel_size_um
            anisotropy = z_step / pixel if z_step and pixel else None
            masks, *_ = model.eval(data, z_axis=1, do_3D=True, anisotropy=anisotropy, **kwargs)
            labels = np.asarray(masks, dtype=np.int32).reshape(data.shape[1:])
        elif n_z > 1 and params.stitch_threshold > 0:
            masks, *_ = model.eval(
                data, z_axis=1, stitch_threshold=params.stitch_threshold, **kwargs
            )
            labels = np.asarray(masks, dtype=np.int32).reshape(data.shape[1:])
        else:
            labels = np.zeros(data.shape[1:], dtype=np.int32)
            offset = 0
            for z in range(n_z):
                masks, *_ = model.eval(data[:, z], **kwargs)
                plane = relabel_sequential(np.asarray(masks))
                plane[plane > 0] += offset
                offset = max(offset, int(np.max(plane)))
                labels[z] = plane
        parent_ids = (
            overlap_parents(labels, image.label_set(params.parent).data) if params.parent else None
        )
        result = LabelSet(
            params.output, labels, params.parent, parent_ids, source=f"cellpose {params.model}"
        )
        report = {
            "module": self.name,
            "output": params.output,
            "model": params.model,
            "gpu": gpu,
            "objects": result.count,
        }
        return image.with_labels(result, self.name, report)


# ----------------------------------------------------------------- training


def training_pairs(folder: Path) -> tuple[list[Path], list[Path]]:
    """``*_img.tif`` images and their ``*_masks.tif`` label files in a training folder."""
    images = sorted(folder.glob("*_img.tif"))
    pairs = [(i, i.with_name(i.name.replace("_img.tif", "_masks.tif"))) for i in images]
    missing = [m.name for _, m in pairs if not m.exists()]
    if missing:
        raise ModuleError(f"Masks missing for: {', '.join(missing[:5])}")
    if not pairs:
        raise ModuleError(f"No *_img.tif / *_masks.tif pairs in {folder}")
    return [i for i, _ in pairs], [m for _, m in pairs]


def fine_tune(
    folder: Path,
    *,
    base_model: str = "cpsam_v2",
    model_name: str = "hcs_custom",
    epochs: int = 100,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.1,
    min_train_masks: int = 5,
    device: str = "auto",
    test_fraction: float = 0.0,
) -> tuple[Path, dict[str, Any]]:
    """Fine-tune a Cellpose model on edited masks. Returns the model file and loss history."""
    import tifffile

    _cellpose()
    from cellpose import train

    images, masks = training_pairs(folder)
    gpu = use_gpu(device)
    data = [np.asarray(tifffile.imread(p), dtype=np.float32) for p in images]
    labels = [np.asarray(tifffile.imread(p), dtype=np.int32) for p in masks]
    usable = [i for i, label in enumerate(labels) if len(np.unique(label)) - 1 >= min_train_masks]
    if not usable:
        raise ModuleError(f"No training image has at least {min_train_masks} objects")
    n_test = round(test_fraction * len(usable))
    test, train_idx = usable[:n_test], usable[n_test:]
    model = _cellpose().CellposeModel(gpu=gpu, pretrained_model=base_model)
    out = folder / "models"
    out.mkdir(exist_ok=True)
    path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=[data[i] for i in train_idx],
        train_labels=[labels[i] for i in train_idx],
        test_data=[data[i] for i in test] or None,
        test_labels=[labels[i] for i in test] or None,
        channel_axis=0 if data[0].ndim == 3 else None,
        n_epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        min_train_masks=min_train_masks,
        save_path=str(folder),
        model_name=model_name,
    )
    history = {
        "base_model": base_model,
        "images_train": len(train_idx),
        "images_test": len(test),
        "epochs": epochs,
        "train_loss": [float(v) for v in np.asarray(train_losses).ravel()],
        "test_loss": [float(v) for v in np.asarray(test_losses).ravel()] if test else [],
    }
    return Path(path), history
