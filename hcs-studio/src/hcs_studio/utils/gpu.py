"""GPU auto-detection. Never raises: a missing or broken GPU stack means CPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Backend = Literal["cuda", "mps", "cpu"]


@dataclass(frozen=True)
class GpuStatus:
    """What acceleration is usable in this process."""

    backend: Backend
    device_name: str | None = None
    vram_gb: float | None = None
    torch_version: str | None = None
    cupy_available: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        return self.backend != "cpu"

    @property
    def summary(self) -> str:
        """One line for the status bar and ``hcs-studio info``."""
        if self.backend == "cuda":
            vram = f", {self.vram_gb:.1f} GB" if self.vram_gb is not None else ""
            cupy = ", CuPy" if self.cupy_available else ""
            name = self.device_name or "CUDA device"
            return f"GPU: {name}{vram} (PyTorch {self.torch_version}{cupy})"
        if self.backend == "mps":
            return f"GPU: Apple Metal (PyTorch {self.torch_version})"
        reason = self.notes[0] if self.notes else "no GPU detected"
        return f"CPU only ({reason})"


def _detect_torch() -> tuple[Backend, str | None, float | None, str | None, list[str]]:
    notes: list[str] = []
    try:
        import torch
    except ImportError:
        return "cpu", None, None, None, ["PyTorch not installed"]
    except Exception as exc:  # a broken install must not take the app down
        return "cpu", None, None, None, [f"PyTorch failed to load: {exc}"]

    version = str(torch.__version__)
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return "cuda", str(props.name), props.total_memory / 1024**3, version, notes
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "Apple Metal", None, version, notes
    except Exception as exc:
        notes.append(f"GPU query failed: {exc}")
        return "cpu", None, None, version, notes
    notes.append("PyTorch found no CUDA device")
    return "cpu", None, None, version, notes


def _cupy_available() -> bool:
    try:
        import cupy

        return bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        return False


def detect_gpu() -> GpuStatus:
    """Probe PyTorch (CUDA, then Apple MPS) and CuPy; fall back to CPU."""
    backend, name, vram, version, notes = _detect_torch()
    return GpuStatus(
        backend=backend,
        device_name=name,
        vram_gb=vram,
        torch_version=version,
        cupy_available=_cupy_available(),
        notes=tuple(notes),
    )
