"""SAM 3 backend: a client for the SAM 3 worker process and a pipeline step.

SAM 3 runs in a separate Python environment because it pins numpy < 2. The worker is
started on first use and kept alive (the model is large), one per process.
The weights are gated on Hugging Face: request access to facebook/sam3, then either
log in with ``huggingface-cli login`` in the SAM 3 environment, or download
``sam3.pt`` and give its path as ``checkpoint``.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from hcs_studio.io.sidecar import config_dir
from hcs_studio.model.plate import ImageMetadata
from hcs_studio.processing.base import (
    MODULES,
    ImageData,
    LabelSet,
    Module,
    ModuleError,
    ModuleParams,
)
from hcs_studio.segmentation import sam3_worker
from hcs_studio.segmentation.objects import overlap_parents

WORKER = Path(sam3_worker.__file__)


@dataclass(frozen=True)
class Sam3Settings:
    """Where SAM 3 lives. Stored in the user settings folder (first-run dialog)."""

    python: str = ""
    checkpoint: str = ""
    device: str = "auto"

    @classmethod
    def path(cls) -> Path:
        return config_dir() / "sam3.json"

    @classmethod
    def load(cls) -> Sam3Settings:
        try:
            data = json.loads(cls.path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        return cls(
            **{k: str(v) for k, v in data.items() if k in ("python", "checkpoint", "device")}
        )

    def save(self) -> None:
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @property
    def configured(self) -> bool:
        return bool(self.python) and Path(self.python).exists()


@dataclass(frozen=True)
class SamResult:
    labels: NDArray[np.int32]
    scores: list[float]


class Sam3Client:
    """Talks to one SAM 3 worker process. Thread-safe: one request at a time."""

    def __init__(
        self,
        python: str,
        *,
        checkpoint: str | None = None,
        device: str = "auto",
        confidence: float = 0.5,
        interactive: bool = True,
        fake: bool = False,
        random_weights: bool = False,
    ) -> None:
        args = [python, "-u", str(WORKER)] + (["--fake"] if fake else [])
        try:
            self._process = subprocess.Popen(
                args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                # Plain HTTPS downloads: the Xet transfer hosts are often blocked by
                # institutional proxies, while the standard CDN download works.
                env={"HF_HUB_DISABLE_XET": "1", **os.environ, "PYTHONNOUSERSITE": "1"},
            )
        except OSError as exc:
            raise ModuleError(f"Could not start the SAM 3 worker with {python}: {exc}") from None
        self._lock = threading.Lock()
        self._stderr: list[str] = []
        self._reader = threading.Thread(target=self._drain, daemon=True)
        self._reader.start()
        reply, _ = self._call(
            {
                "cmd": "load",
                "checkpoint": checkpoint or None,
                "device": device,
                "confidence": confidence,
                "interactive": interactive,
                "random_weights": random_weights,
            }
        )
        self.device = str(reply.get("device"))
        self._image_key: Any = None

    def _drain(self) -> None:
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr = [*self._stderr[-50:], line.decode("utf-8", "replace").rstrip()]

    def _call(self, header: dict[str, Any], payload: bytes = b"") -> tuple[dict[str, Any], bytes]:
        with self._lock:
            assert self._process.stdin is not None and self._process.stdout is not None
            try:
                sam3_worker.write_message(self._process.stdin, header, payload)  # type: ignore[arg-type]
                message = sam3_worker.read_message(self._process.stdout)  # type: ignore[arg-type]
            except (BrokenPipeError, OSError):
                message = None
            if message is None:
                # Let the process exit so its last error lines are read.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    self._process.wait(timeout=5)
                self._reader.join(timeout=2)
                tail = "\n".join(self._stderr[-8:])
                raise ModuleError(f"The SAM 3 worker stopped.\n{tail}")
            reply, data = message
            if not reply.get("ok"):
                raise ModuleError(f"SAM 3: {reply.get('error')}")
            return reply, data

    def set_image(self, rgb: NDArray[np.uint8], key: Any = None) -> None:
        """Encode an (H, W, 3) uint8 image. Skipped when ``key`` matches the last image."""
        if key is not None and key == self._image_key:
            return
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ModuleError(f"SAM 3 needs an RGB image; got shape {rgb.shape}")
        self._call({"cmd": "set_image", "shape": list(rgb.shape)}, rgb.tobytes())
        self._image_key = key

    def _labels(self, reply: dict[str, Any], data: bytes) -> SamResult:
        labels = np.frombuffer(data, dtype=np.int32).reshape(reply["shape"]).copy()
        return SamResult(labels=labels, scores=[float(s) for s in reply["scores"]])

    def text(self, prompt: str, threshold: float = 0.5) -> SamResult:
        return self._labels(*self._call({"cmd": "text", "prompt": prompt, "threshold": threshold}))

    def box(self, box_xyxy: list[float], positive: bool = True) -> SamResult:
        return self._labels(*self._call({"cmd": "box", "box": box_xyxy, "positive": positive}))

    def points(self, points_xy: list[list[float]], labels: list[int]) -> SamResult:
        return self._labels(*self._call({"cmd": "points", "points": points_xy, "labels": labels}))

    def reset(self) -> None:
        self._call({"cmd": "reset"})

    def close(self) -> None:
        if self._process.poll() is None:
            with contextlib.suppress(ModuleError):
                self._call({"cmd": "shutdown"})
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()


_clients: dict[tuple[str, str, str], Sam3Client] = {}
_clients_lock = threading.Lock()


def get_client(python: str, checkpoint: str, device: str, fake: bool = False) -> Sam3Client:
    key = (python, checkpoint, device + ("/fake" if fake else ""))
    with _clients_lock:
        client = _clients.get(key)
        if client is None or client._process.poll() is not None:
            client = Sam3Client(python, checkpoint=checkpoint or None, device=device, fake=fake)
            _clients[key] = client
        return client


@atexit.register
def close_clients() -> None:
    with _clients_lock:
        for client in _clients.values():
            client.close()
        _clients.clear()


def to_rgb(data: NDArray[Any], channels: list[int]) -> NDArray[np.uint8]:
    """(C, Y, X) → (Y, X, 3) uint8, each used channel stretched between its 0.5/99.8 percentiles.

    One channel → grey; two → red/green; three → red/green/blue.
    """
    if not 1 <= len(channels) <= 3:
        raise ModuleError("Choose one to three channels for SAM 3")
    rgb = np.zeros((*data.shape[1:], 3), dtype=np.float32)
    for slot, channel in enumerate(channels):
        plane = data[channel]
        low, high = np.percentile(plane, (0.5, 99.8))
        scaled = np.clip((plane - low) / (high - low), 0, 1) if high > low else np.zeros_like(plane)
        if len(channels) == 1:
            rgb[:] = scaled[..., None]
        else:
            rgb[..., slot] = scaled
    return np.asarray(rgb * 255, dtype=np.uint8)


class Sam3Params(ModuleParams):
    prompt: str = Field(min_length=1, description='Concept to segment, e.g. "nucleus", "spheroid".')
    channels: list[int] = Field(
        min_length=1, max_length=3, description="1-3 channels shown to SAM 3."
    )
    output: str = Field(default="objects", description="Name of the new label set.")
    confidence: float = Field(default=0.5, gt=0, lt=1, description="Minimum detection score.")
    min_size_px: int = Field(default=20, ge=0, description="Remove objects smaller than this.")
    parent: str | None = Field(
        default=None, description="Link objects to this label set by overlap."
    )
    python: str | None = Field(
        default=None, description="Python of the SAM 3 environment; empty = from settings."
    )
    checkpoint: str | None = Field(
        default=None, description="Path of sam3.pt; empty = from settings, else Hugging Face."
    )
    device: str = Field(default="auto", pattern="^(auto|cpu|gpu)$", description="auto, cpu or gpu.")


@MODULES.register("segment_sam3")
class SegmentSam3(Module):
    """Segment every instance of a text concept with SAM 3 (2D, plane by plane)."""

    name = "segment_sam3"
    title = "Segmentation: SAM 3 (text concept)"
    Params = Sam3Params
    #: Tests set this to use the worker's threshold stand-in instead of SAM 3.
    fake_model = False

    def validate(self, metadata: ImageMetadata) -> None:
        bad = [c for c in self.params.channels if not 1 <= c <= len(metadata.channels)]
        if bad:
            raise ModuleError(f"Channel(s) {bad} do not exist")

    def _client(self) -> Sam3Client:
        params: Sam3Params = self.params
        settings = Sam3Settings.load()
        python = params.python or settings.python or (sys.executable if self.fake_model else "")
        if not python:
            raise ModuleError(
                "SAM 3 is not set up: choose its Python environment in Tools → SAM 3 settings"
            )
        device = "cuda" if params.device == "gpu" else params.device
        return get_client(python, params.checkpoint or settings.checkpoint, device, self.fake_model)

    def apply(self, image: ImageData, state: Any) -> ImageData:
        params: Sam3Params = self.params
        self.validate(image.metadata)
        client = self._client()
        channels = [c - 1 for c in params.channels]
        labels = np.zeros(image.data.shape[1:], dtype=np.int32)
        scores: list[float] = []
        for z in range(image.data.shape[1]):
            client.set_image(to_rgb(image.data[:, z], channels))
            result = client.text(params.prompt, params.confidence)
            plane = result.labels
            if params.min_size_px:
                sizes = np.bincount(plane.ravel())
                plane[np.isin(plane, np.nonzero(sizes < params.min_size_px)[0][1:])] = 0
            plane[plane > 0] += len(scores)
            labels[z] = plane
            scores += result.scores
        parent_ids = (
            overlap_parents(labels, image.label_set(params.parent).data) if params.parent else None
        )
        result_set = LabelSet(
            params.output, labels, params.parent, parent_ids, source=f"sam3 '{params.prompt}'"
        )
        report = {
            "module": self.name,
            "output": params.output,
            "prompt": params.prompt,
            "objects": result_set.count,
            "device": client.device,
        }
        return image.with_labels(result_set, self.name, report)
