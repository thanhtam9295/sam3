"""Integration tests for the deployment front ends (CLI + REST service).

A fake processor stands in for the model, so these run without GPU/weights
and exercise the I/O, schema and security behaviour only.
"""

import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = pytest.mark.integration
REPO = Path(__file__).resolve().parents[2]


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "sam3_cli", REPO / "packaging/windows/sam3_cli.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sam3_cli"] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeProcessor:
    """Returns two square instances on any image."""

    def __init__(self):
        self.threshold = 0.5

    def set_confidence_threshold(self, t, state=None):
        self.threshold = t

    def set_image(self, image):
        return {"h": image.height, "w": image.width}

    def set_text_prompt(self, state, prompt):
        h, w = state["h"], state["w"]
        masks = torch.zeros(2, 1, h, w, dtype=torch.bool)
        masks[0, 0, :4, :4] = True
        masks[1, 0, 5:10, 5:7] = True
        return {
            "masks": masks,
            "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0], [5.0, 5.0, 7.0, 10.0]]),
            "scores": torch.tensor([0.9, 0.7]),
        }


def test_cli_summary_and_outputs(tmp_path):
    cli = _load_cli()
    img = Image.new("RGB", (16, 12), "gray")
    out = FakeProcessor().set_text_prompt(FakeProcessor().set_image(img), "x")
    inst = cli.summarize(out)
    assert [i["area_px"] for i in inst] == [16, 10]
    assert inst[0]["box_xyxy"] == [0.0, 0.0, 4.0, 4.0]
    cli.write_outputs(img, out, "img", tmp_path)
    label = np.array(Image.open(tmp_path / "img_masks.png"))
    assert label.shape == (12, 16) and set(np.unique(label)) == {0, 1, 2}
    assert (tmp_path / "img_overlay.png").exists()


def test_cli_help_does_not_load_model():
    cli = _load_cli()
    with pytest.raises(SystemExit) as e:
        cli.main(["--help"])
    assert e.value.code == 0


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    _load_cli()
    import deploy.service.app as svc

    monkeypatch.setattr(svc, "processor_factory", FakeProcessor)
    monkeypatch.setattr(svc, "_processor", None)
    return TestClient(svc.app)


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_service_requires_bearer_token(client, monkeypatch):
    monkeypatch.delenv("SAM3_SERVICE_ALLOW_ANONYMOUS", raising=False)
    r = client.post(
        "/v1/segment", files={"image": ("a.png", _png())}, data={"prompt": "x"}
    )
    assert r.status_code == 401


def test_service_segment_contract(client):
    r = client.post(
        "/v1/segment",
        files={"image": ("a.png", _png())},
        data={"prompt": "square", "threshold": "0.3"},
        headers={"Authorization": "Bearer test"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2 and body["threshold"] == 0.3
    assert body["image"] == {"width": 16, "height": 12}
    assert set(body["instances"][0]) == {"instance", "score", "box_xyxy", "area_px"}


def test_service_rejects_bad_input(client):
    h = {"Authorization": "Bearer test"}
    assert (
        client.post(
            "/v1/segment",
            files={"image": ("a.png", b"notanimage")},
            data={"prompt": "x"},
            headers=h,
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/v1/segment",
            files={"image": ("a.png", _png())},
            data={"prompt": "x" * 500},
            headers=h,
        ).status_code
        == 422
    )


def test_openapi_spec_matches_service_routes():
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load(
        (REPO / "deploy/m365-copilot/appPackage/openapi.yaml").read_text()
    )
    assert "/v1/segment" in spec["paths"]
    for name in ("manifest.json", "declarativeAgent.json", "ai-plugin.json"):
        json.loads((REPO / "deploy/m365-copilot/appPackage" / name).read_text())
