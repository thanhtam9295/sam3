"""GPU detection must degrade to CPU without raising."""

from __future__ import annotations

import sys
import types

import pytest

from hcs_studio.utils.gpu import GpuStatus, detect_gpu


def _fake_torch(cuda: bool) -> types.ModuleType:
    torch = types.ModuleType("torch")
    props = types.SimpleNamespace(name="Fake RTX", total_memory=24 * 1024**3)
    torch.__version__ = "9.9.9"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda, get_device_properties=lambda index: props
    )
    torch.backends = types.SimpleNamespace()  # type: ignore[attr-defined]
    return torch


def test_detect_gpu_never_raises_and_summary_is_text() -> None:
    status = detect_gpu()
    assert status.backend in ("cuda", "mps", "cpu")
    assert status.summary


def test_missing_torch_means_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes ``import torch`` raise ImportError
    status = detect_gpu()
    assert status.backend == "cpu" and not status.available
    assert status.summary == "CPU only (PyTorch not installed)"


def test_cuda_device_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    monkeypatch.setitem(sys.modules, "cupy", None)
    status = detect_gpu()
    assert status.backend == "cuda" and status.available
    assert status.vram_gb == pytest.approx(24.0)
    assert status.summary == "GPU: Fake RTX, 24.0 GB (PyTorch 9.9.9)"


def test_torch_without_cuda_means_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    status = detect_gpu()
    assert status.backend == "cpu"
    assert "no CUDA device" in status.summary


def test_broken_gpu_query_means_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch(cuda=True)

    def boom(index: int) -> None:
        raise RuntimeError("driver mismatch")

    torch.cuda.get_device_properties = boom
    monkeypatch.setitem(sys.modules, "torch", torch)
    status = detect_gpu()
    assert status.backend == "cpu"
    assert "driver mismatch" in status.summary


def test_status_defaults() -> None:
    assert GpuStatus(backend="cpu").summary == "CPU only (no GPU detected)"
