"""``segment_model``: run a registered segmentation model made of pipeline steps (Phase 25).

Models of kind ``steps`` (e.g. a classical segmentation tuned on ground truth) store their
steps as JSON in the model registry. A pipeline references them as ``registry:NAME@VERSION``;
the steps run exactly as stored, so the same version always gives the same masks.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import MODULES, ImageData, Module, ModuleError, ModuleParams


class SegmentModelParams(ModuleParams):
    model: str = Field(
        default="", description="A registered segmentation model: registry:NAME@VERSION."
    )


def stored_steps(ref: str) -> list[Any]:
    """The pipeline steps of a registered ``steps`` model."""
    from hcs_studio.pipeline.schema import PipelineStep
    from hcs_studio.training.registry import resolve

    if not ref:
        raise ModuleError("Choose a registered model (registry:NAME@VERSION)")
    try:
        record = resolve(ref, "steps")
    except ValueError as exc:
        raise ModuleError(str(exc)) from None
    data = json.loads(record.path.read_text(encoding="utf-8"))
    steps = [PipelineStep.model_validate(s) for s in data["steps"]]
    if any(s.create().needs_fit for s in steps):
        raise ModuleError(f"{ref}: a registered model cannot contain plate-level fitted steps")
    return steps


@MODULES.register("segment_model")
class SegmentModel(Module):
    name = "segment_model"
    title = "Registered segmentation model"
    Params = SegmentModelParams

    def __init__(self, params: ModuleParams | None = None) -> None:
        super().__init__(params)
        self._modules: list[Module] | None = None

    def modules(self) -> list[Module]:
        if self._modules is None:
            self._modules = [s.create() for s in stored_steps(self.params.model)]
        return self._modules

    def validate(self, metadata: ImageMetadata) -> None:
        for module in self.modules():
            module.validate(metadata)

    def uses_gpu(self) -> bool:
        return any(m.uses_gpu() for m in self.modules())

    def apply(self, image: ImageData, state: Any) -> ImageData:
        for module in self.modules():
            image = module.apply(image, None)
        return image
