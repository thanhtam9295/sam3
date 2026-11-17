"""Write a synthetic ImageXpress plate for performance tests (repository script, not shipped).

    python scripts/make_test_plate.py <folder> [--wells 384] [--sites 2] [--size 1080]

Each field has 30–400 nuclei (Hoechst, channel 1) with a cytoplasmic marker (channel 2),
uneven illumination, camera offset and Poisson noise. Columns 1–2 are negative controls,
columns 23–24 positive controls (brighter marker); a layout CSV is written next to it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.builders import make_imagexpress


def field_image(row: int, col: int, site: int, wave: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(((row * 100 + col) * 10 + site) * 7)  # same cells for both waves
    n = int(30 + (370 * ((row * 24 + col) % 37) / 36))
    centres = rng.uniform(12, size - 12, (n, 2))
    radii = rng.uniform(6, 10, n)
    image = np.zeros((size, size), dtype=np.float32)
    marker = 3.0 if col >= 22 else 1.0
    for (y, x), r in zip(centres, radii, strict=True):
        reach = int(2.2 * r) + 1
        y0, y1 = max(0, int(y) - reach), min(size, int(y) + reach)
        x0, x1 = max(0, int(x) - reach), min(size, int(x) + reach)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = np.hypot(yy - y, xx - x)
        if wave == 0:
            image[y0:y1, x0:x1] += 800 * (d <= r)
        else:
            image[y0:y1, x0:x1] += 250 * marker * ((d > r) & (d <= 2.1 * r))
    image = ndimage.gaussian_filter(image, 1.2)
    yy, xx = np.mgrid[:size, :size] / size - 0.5
    shading = 1 - 0.35 * (yy**2 + xx**2) / 0.5  # vignetting
    noise_rng = np.random.default_rng(((row * 100 + col) * 10 + site) * 7 + wave + 1)
    # Vignetting dims all collected light (cells and background); the camera offset is not.
    image = noise_rng.poisson(np.clip((image + 60) * shading, 0, None)) + 100  # offset 100
    return np.clip(image, 0, 65535).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--wells", type=int, default=384, choices=[6, 24, 96, 384])
    parser.add_argument("--sites", type=int, default=2)
    parser.add_argument("--size", type=int, default=1080)
    args = parser.parse_args()
    rows, cols = {6: (2, 3), 24: (4, 6), 96: (8, 12), 384: (16, 24)}[args.wells]
    wells = [(r, c) for r in range(rows) for c in range(cols)]
    started = time.monotonic()
    root = make_imagexpress(
        args.folder / "plate",
        wells,
        n_sites=args.sites,
        image_fn=lambda r, c, s, w: field_image(r, c, s, w, args.size),
    )
    names = "ABCDEFGHIJKLMNOP"
    lines = ["well,compound,concentration,treatment,control_type"]
    for r, c in wells:
        well = f"{names[r]}{c + 1:02d}"
        if c < 2:
            lines.append(f"{well},DMSO,,vehicle,negative")
        elif c >= cols - 2:
            lines.append(f"{well},Reference,1,24h,positive")
        else:
            lines.append(f"{well},Compound {1 + (c - 2) // 4},{10 ** ((c - 2) % 4 - 2)},24h,")
    (args.folder / "layout.csv").write_text("\n".join(lines) + "\n")
    print(
        f"Wrote {len(wells)} wells × {args.sites} sites to {root} in "
        f"{time.monotonic() - started:.0f} s"
    )


if __name__ == "__main__":
    main()
