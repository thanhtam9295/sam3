"""SAM 3 segmentation REST service (for Microsoft 365 Copilot API plugins and
other internal clients).

  uvicorn deploy.service.app:app --host 127.0.0.1 --port 8080

Endpoints
  GET  /healthz              liveness (no model load)
  POST /v1/segment           multipart: image file + prompt [+ threshold]
                             -> instance count, scores, XYXY boxes, areas

Security model (read docs/deployment/M365_COPILOT.md before exposing this):
  * Binds to localhost by default. In production run it inside the approved
    enterprise network behind Azure API Management (or equivalent) that
    validates Microsoft Entra ID tokens; this process does not validate JWTs.
  * Requests without an Authorization header are rejected unless
    SAM3_SERVICE_ALLOW_ANONYMOUS=1 (local development only).
  * Images are processed in memory and never written to disk or logged.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packaging" / "windows"))
from sam3_cli import summarize  # noqa: E402  (shared output schema with the CLI)

MAX_IMAGE_BYTES = int(os.environ.get("SAM3_SERVICE_MAX_IMAGE_BYTES", 20 * 1024 * 1024))
MAX_PROMPT_CHARS = 200

app = FastAPI(
    title="SAM 3 Segmentation Service",
    version="0.1.0",
    description="Open-vocabulary instance segmentation with SAM 3.",
)

_lock = threading.Lock()
_processor = None
# Tests inject a fake factory; production builds the real model lazily.
processor_factory: Optional[Callable[[], Any]] = None


def _default_factory():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    ckpt = os.environ.get("SAM3_CHECKPOINT")
    model = build_sam3_image_model(checkpoint_path=ckpt, load_from_HF=ckpt is None)
    return Sam3Processor(model)


def get_processor():
    global _processor
    with _lock:
        if _processor is None:
            _processor = (processor_factory or _default_factory)()
        return _processor


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if os.environ.get("SAM3_SERVICE_ALLOW_ANONYMOUS") == "1":
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            401, "Bearer token required (validated by the API gateway)."
        )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/segment", dependencies=[Depends(require_auth)])
async def segment(
    image: UploadFile = File(..., description="PNG/JPEG/TIFF image"),
    prompt: str = Form(..., description="Short noun phrase, e.g. 'cell nucleus'"),
    threshold: float = Form(0.5, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    from PIL import Image, UnidentifiedImageError

    data = await image.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large.")
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(422, f"Prompt must be 1-{MAX_PROMPT_CHARS} characters.")
    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(415, "Unsupported image format.")

    import torch

    proc = get_processor()
    with _lock, torch.inference_mode():
        proc.set_confidence_threshold(threshold)
        state = proc.set_image(pil)
        out = proc.set_text_prompt(state=state, prompt=prompt)
    instances = summarize(out)
    return {
        "prompt": prompt,
        "threshold": threshold,
        "image": {"width": pil.width, "height": pil.height},
        "count": len(instances),
        "instances": instances,
        "caveat": "Automated segmentation; counts depend on the threshold. "
        "Validate on your own data before scientific use.",
    }
