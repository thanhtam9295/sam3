"""sam3-cli: portable command-line front end for SAM 3 image segmentation.

  sam3-cli segment --image IMG [--image IMG2 ...] --prompt "cell nucleus" \
      --checkpoint C:\\models\\sam3.pt --out results\\ [--threshold 0.5]

Writes, per image:  <stem>_masks.png (label image, 0 = background),
<stem>_overlay.png, and one results.json with boxes (XYXY px), scores, areas,
provenance (versions, prompt, threshold, checkpoint SHA-256).

Everything runs locally. No data leaves the machine. Weights are never
bundled; pass --checkpoint, or omit it to download from Hugging Face with the
user's own accepted licence and login (`hf auth login`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert processor output to plain JSON records (one per instance)."""
    masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
    areas = masks.flatten(1).sum(1) if masks.numel() else masks.new_zeros(0)
    return [
        {
            "instance": i + 1,
            "score": round(float(scores[i]), 6),
            "box_xyxy": [round(float(v), 2) for v in boxes[i].tolist()],
            "area_px": int(areas[i]),
        }
        for i in range(int(scores.shape[0]))
    ]


def write_outputs(image, out: Dict[str, Any], stem: str, out_dir: Path) -> None:
    import numpy as np
    from PIL import Image

    masks = out["masks"].detach().cpu().numpy().reshape(-1, image.height, image.width)
    label = np.zeros((image.height, image.width), dtype=np.uint16)
    for i, m in enumerate(masks):  # ascending score order not guaranteed: later wins
        label[m.astype(bool)] = i + 1
    Image.fromarray(label).save(out_dir / f"{stem}_masks.png")

    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    rng = np.random.RandomState(0)
    colors = rng.randint(40, 255, size=(max(len(masks), 1), 3))
    for i, m in enumerate(masks):
        sel = m.astype(bool)
        rgb[sel] = 0.5 * rgb[sel] + 0.5 * colors[i]
    Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).save(
        out_dir / f"{stem}_overlay.png"
    )


def cmd_segment(args: argparse.Namespace) -> int:
    import sam3
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not torch.cuda.is_available() and args.device == "cuda":
        print(
            "ERROR: SAM 3 currently requires an NVIDIA GPU with CUDA "
            "(docs/KNOWN_ISSUES.md KI-009).",
            file=sys.stderr,
        )
        return 3
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    model = build_sam3_image_model(
        device=args.device,
        checkpoint_path=str(ckpt) if ckpt else None,
        load_from_HF=ckpt is None,
    )
    proc = Sam3Processor(model, device=args.device, confidence_threshold=args.threshold)

    results = {
        "tool": "sam3-cli",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "sam3_version": sam3.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "checkpoint": str(ckpt) if ckpt else "huggingface:default",
            "checkpoint_sha256": sha256(ckpt) if ckpt else None,
            "prompt": args.prompt,
            "confidence_threshold": args.threshold,
        },
        "images": [],
    }
    for img_path in map(Path, args.image):
        image = Image.open(img_path)
        with torch.inference_mode():
            state = proc.set_image(image)
            out = proc.set_text_prompt(state=state, prompt=args.prompt)
        write_outputs(image, out, img_path.stem, out_dir)
        inst = summarize(out)
        results["images"].append(
            {"image": img_path.name, "count": len(inst), "instances": inst}
        )
        print(f"{img_path.name}: {len(inst)} instance(s) of '{args.prompt}'")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {out_dir / 'results.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="sam3-cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("segment", help="segment all instances of a text concept")
    s.add_argument("--image", action="append", required=True)
    s.add_argument("--prompt", required=True)
    s.add_argument("--checkpoint", help="local .pt checkpoint (recommended offline)")
    s.add_argument("--out", default="sam3_results")
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--device", default="cuda")
    s.set_defaults(func=cmd_segment)
    return ap


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
