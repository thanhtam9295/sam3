"""Shared pytest configuration for the SAM 3 test tiers.

Tiers (select with ``-m``):
  unit         fast, pure-function tests, CPU only, no network, no checkpoints
  integration  multi-module pipelines on synthetic or in-repo public data
  scientific   numerical regression + reference-implementation agreement
  gpu          requires CUDA (auto-skipped when unavailable)
  slow         > ~10 s on a laptop CPU
"""

import os
import random
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config, items):
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Every test starts from the same RNG state for reproducibility."""
    random.seed(0)
    np.random.seed(0)
    try:
        import torch

        torch.manual_seed(0)
    except ImportError:
        pass
    yield


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _no_network_tokens(monkeypatch):
    """Tests must never depend on real credentials."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", os.environ.get("SAM3_TEST_HF_OFFLINE", "1"))
    yield
