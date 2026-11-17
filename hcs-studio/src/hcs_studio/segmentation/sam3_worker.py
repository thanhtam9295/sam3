"""SAM 3 worker process. Runs in its own Python environment (SAM 3 needs numpy < 2).

Started by ``hcs_studio.segmentation.sam3_backend.Sam3Client`` as
``<sam3 python> sam3_worker.py``. Uses only the standard library, numpy, torch, PIL and
sam3, so the worker environment does not need HCS Studio installed.

Protocol on stdin/stdout, one message at a time: 4-byte big-endian header length, a
UTF-8 JSON header, then ``header["nbytes"]`` bytes of raw array data.
Requests: load, set_image, text, box, points, reset, shutdown.
Responses: {"ok": true, ...} or {"ok": false, "error": "..."}; segmentations come back
as an int32 (H, W) label image (higher-scoring objects painted on top) and their scores.

``--fake`` replaces the model by a simple threshold model. It exists only to test the
protocol and the user interface without the gated SAM 3 weights; never use it for data.
"""

from __future__ import annotations

import json
import struct
import sys
import traceback
from typing import Any, BinaryIO

import numpy as np


def read_message(stream: BinaryIO) -> tuple[dict[str, Any], bytes] | None:
    raw = stream.read(4)
    if len(raw) < 4:
        return None
    (length,) = struct.unpack(">I", raw)
    header = json.loads(stream.read(length).decode("utf-8"))
    payload = stream.read(int(header.get("nbytes", 0)))
    return header, payload


def write_message(stream: BinaryIO, header: dict[str, Any], payload: bytes = b"") -> None:
    header = {**header, "nbytes": len(payload)}
    data = json.dumps(header).encode("utf-8")
    stream.write(struct.pack(">I", len(data)) + data + payload)
    stream.flush()


def paint(masks: Any, scores: Any, shape: tuple[int, int]) -> tuple[np.ndarray, list[float]]:
    """Stack of boolean masks → label image; the highest score wins where masks overlap."""
    labels = np.zeros(shape, dtype=np.int32)
    masks = np.asarray(masks, dtype=bool).reshape(-1, *shape)
    scores = [float(s) for s in np.asarray(scores, dtype=float).ravel()]
    order = np.argsort(scores)[::-1]
    for rank, index in enumerate(order, start=1):
        labels[masks[index] & (labels == 0)] = rank
    return labels, [scores[i] for i in order]


def _cpu_shim() -> None:
    """Let SAM 3 build on a machine without an NVIDIA GPU.

    Upstream SAM 3 assumes CUDA in three places on the image path:
    two precomputed tensors with a hard-coded ``device="cuda"`` (positional-encoding cache
    in ``sam3/model/position_encoding.py``, decoder coordinates in ``sam3/model/decoder.py``)
    and ``Tensor.pin_memory()`` in ``sam3/model/geometry_encoders.py`` (box prompts).
    In this worker process, on machines without CUDA only, the two allocations go to the
    CPU and ``pin_memory`` (which only speeds up copies to a GPU) returns the tensor
    unchanged. Nothing else changes; results are the same, only slower.
    """
    import types

    import torch
    from sam3.model import decoder, position_encoding

    def cpu(device: Any) -> Any:
        return "cpu" if str(device).startswith("cuda") else device

    class TorchOnCpu(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            return getattr(torch, name)

        def zeros(self, *args: Any, device: Any = None, **kwargs: Any) -> Any:
            return torch.zeros(*args, device=cpu(device), **kwargs)

    position_encoding.torch = TorchOnCpu("torch")
    torch.Tensor.pin_memory = lambda self, *args, **kwargs: self
    for cls in vars(decoder).values():
        if isinstance(cls, type) and "_get_coords" in vars(cls):
            original = getattr(cls, "_get_coords")  # noqa: B009

            def get_coords(h: int, w: int, device: Any, _original: Any = original) -> Any:
                return _original(h, w, cpu(device))

            setattr(cls, "_get_coords", staticmethod(get_coords))  # noqa: B010


class RealModel:
    def __init__(
        self,
        checkpoint: str | None,
        device: str,
        confidence: float,
        interactive: bool,
        random_weights: bool,
    ) -> None:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and not torch.cuda.is_available():
            _cpu_shim()
        self.device = device
        self.model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint,
            load_from_HF=checkpoint is None and not random_weights,
            enable_inst_interactivity=interactive,
        )
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence)
        self.interactive = interactive
        self.state: dict[str, Any] = {}
        self.shape = (0, 0)
        # SAM 3 runs in bfloat16 (its fused MLP casts unconditionally); autocast keeps the
        # surrounding layers consistent on either device.
        self._autocast = torch.autocast(
            device_type="cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16
        )

    def set_image(self, rgb: np.ndarray) -> None:
        from PIL import Image

        self.shape = rgb.shape[:2]
        with self._autocast:
            self.state = self.processor.set_image(Image.fromarray(rgb))

    def _result(self) -> tuple[Any, Any]:
        masks = self.state.get("masks")
        scores = self.state.get("scores")
        if masks is None:
            return np.zeros((0, *self.shape), dtype=bool), []
        assert scores is not None
        return masks.squeeze(1).cpu().numpy(), scores.float().cpu().numpy()

    def text(self, prompt: str, threshold: float) -> tuple[Any, Any]:
        self.processor.reset_all_prompts(self.state)
        self.processor.confidence_threshold = threshold
        with self._autocast:
            self.state = self.processor.set_text_prompt(prompt, self.state)
        return self._result()

    def box(self, box_xyxy: list[float], positive: bool) -> tuple[Any, Any]:
        h, w = self.shape
        x0, y0, x1, y1 = box_xyxy
        normalised = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
        with self._autocast:
            self.state = self.processor.add_geometric_prompt(normalised, positive, self.state)
        return self._result()

    def points(self, points_xy: list[list[float]], labels: list[int]) -> tuple[Any, Any]:
        if not self.interactive:
            raise RuntimeError("Point prompts need the model loaded with interactive=true")
        with self._autocast:
            masks, scores, _ = self.model.predict_inst(
                self.state,
                point_coords=np.asarray(points_xy, dtype=np.float32),
                point_labels=np.asarray(labels, dtype=np.int32),
                multimask_output=True,
            )
        best = int(np.argmax(scores))
        return masks[best][None] > 0, [float(scores[best])]

    def reset(self) -> None:
        self.processor.reset_all_prompts(self.state)


class FakeModel:
    """Threshold model with the same interface; for protocol and UI tests only."""

    def __init__(self, *args: Any) -> None:
        self.image = np.zeros((1, 1), dtype=float)
        self.shape = (1, 1)

    def set_image(self, rgb: np.ndarray) -> None:
        self.image = rgb.astype(float).mean(axis=2)
        self.shape = self.image.shape

    def _components(self) -> np.ndarray:
        from scipy import ndimage

        level = self.image.mean() + self.image.std()
        labelled, _ = ndimage.label(self.image > level)
        return np.asarray(labelled)

    def text(self, prompt: str, threshold: float) -> tuple[Any, Any]:
        labelled = self._components()
        ids = [i for i in np.unique(labelled) if i > 0]
        return np.stack([labelled == i for i in ids]) if ids else np.zeros((0, *self.shape)), [
            0.9
        ] * len(ids)

    def box(self, box_xyxy: list[float], positive: bool) -> tuple[Any, Any]:
        labelled = self._components()
        x0, y0, x1, y1 = (round(v) for v in box_xyxy)
        inside = np.zeros(self.shape, dtype=bool)
        inside[y0:y1, x0:x1] = True
        ids = [i for i in np.unique(labelled[inside]) if i > 0]
        return np.stack([labelled == i for i in ids]) if ids else np.zeros((0, *self.shape)), [
            0.8
        ] * len(ids)

    def points(self, points_xy: list[list[float]], labels: list[int]) -> tuple[Any, Any]:
        labelled = self._components()
        mask = np.zeros(self.shape, dtype=bool)
        for (x, y), label in zip(points_xy, labels, strict=True):
            value = labelled[round(y), round(x)]
            if value and label == 1:
                mask |= labelled == value
        return mask[None], [0.95]

    def reset(self) -> None:
        pass


def serve(fake: bool) -> None:
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # library prints must not corrupt the protocol
    model: Any = None
    while True:
        message = read_message(stdin)
        if message is None:
            return
        header, payload = message
        command = header.get("cmd")
        try:
            if command == "shutdown":
                write_message(stdout, {"ok": True})
                return
            if command == "load":
                cls = FakeModel if fake else RealModel
                model = cls(
                    header.get("checkpoint"),
                    header.get("device", "auto"),
                    float(header.get("confidence", 0.5)),
                    bool(header.get("interactive", True)),
                    bool(header.get("random_weights", False)),
                )
                write_message(stdout, {"ok": True, "device": getattr(model, "device", "fake")})
                continue
            if model is None:
                raise RuntimeError("Send 'load' first")
            if command == "set_image":
                rgb = np.frombuffer(payload, dtype=np.uint8).reshape(header["shape"])
                model.set_image(rgb)
                write_message(stdout, {"ok": True})
                continue
            if command == "reset":
                model.reset()
                write_message(stdout, {"ok": True})
                continue
            if command == "text":
                masks, scores = model.text(header["prompt"], float(header.get("threshold", 0.5)))
            elif command == "box":
                masks, scores = model.box(header["box"], bool(header.get("positive", True)))
            elif command == "points":
                masks, scores = model.points(header["points"], header["labels"])
            else:
                raise RuntimeError(f"Unknown command {command!r}")
            labels, ordered = paint(masks, scores, model.shape)
            write_message(
                stdout,
                {"ok": True, "shape": list(labels.shape), "scores": ordered},
                labels.tobytes(),
            )
        except Exception as exc:
            write_message(
                stdout,
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(),
                },
            )


if __name__ == "__main__":
    serve(fake="--fake" in sys.argv[1:])
