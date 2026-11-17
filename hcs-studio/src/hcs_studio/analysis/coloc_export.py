"""Export the exact colocalization inputs of one field or object for Fiji's Coloc 2.

Writes background-subtracted channel A and B, the mask, HCS Studio's own results and a
Fiji macro, so both programs analyse identical pixels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hcs_studio.analysis.coloc import ColocResult, colocalize

MACRO = """// Compare HCS Studio with Fiji Coloc 2 on identical inputs.
// Open in Fiji (Plugins > Macros > Run...). Coloc 2's option names can differ between
// Fiji versions; if this fails, run Coloc 2 once with the Macro Recorder on and copy the
// recorded line over the run("Coloc 2", ...) call below.
dir = getDirectory("file");
print("\\Clear");
open(dir + "channel_a.tif");
open(dir + "channel_b.tif");
open(dir + "mask.tif");
run("Coloc 2", "channel_1=channel_a.tif channel_2=channel_b.tif roi_or_mask=mask.tif "
  + "threshold_regression=Costes li_icq spearman's_rank_correlation manders'_correlation "
  + "2d_instensity_histogram costes'_significance_test psf={psf} costes_randomisations={n}");
// Coloc 2 writes its results to the Log; keep them for `hcs-studio acceptance coloc`.
selectWindow("Log");
saveAs("Text", dir + "coloc2_log.txt");
// Compare with hcs_studio_results.json in the same folder:
//   Pearson's R value (no threshold)  ~ pearson
//   Manders' tM1 / tM2 (above autothreshold) ~ tm1 / tm2
//   Costes threshold ch1 / ch2        ~ threshold_a / threshold_b
//   Costes P-Value (Coloc 2 reports 1 - p) ~ 1 - p_value
"""


def export_for_fiji(
    folder: Path,
    a: NDArray[Any],
    b: NDArray[Any],
    mask: NDArray[np.bool_],
    *,
    names: tuple[str, str] = ("A", "B"),
    psf_px: int = 3,
    randomizations: int = 100,
    description: str = "",
) -> ColocResult:
    """Write the inputs and HCS Studio's results for one 2D region; returns the results."""
    import tifffile

    if a.shape != b.shape or a.shape != mask.shape or a.ndim != 2:
        raise ValueError("Channel A, channel B and the mask must be 2D images of the same shape")
    folder.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(folder / "channel_a.tif", np.asarray(a, dtype=np.float32))
    tifffile.imwrite(folder / "channel_b.tif", np.asarray(b, dtype=np.float32))
    tifffile.imwrite(folder / "mask.tif", (mask > 0).astype(np.uint8) * 255)
    result = colocalize(a, b, mask, randomizations=randomizations, block=(psf_px, psf_px))
    payload = {
        "description": description,
        "channel_a": names[0],
        "channel_b": names[1],
        "psf_px": psf_px,
        "randomizations": randomizations,
        "note": "Inputs are background-subtracted and clipped at 0, as HCS Studio used them.",
        "results": result.row(),
    }
    (folder / "hcs_studio_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (folder / "coloc2_compare.ijm").write_text(
        MACRO.format(psf=psf_px, n=randomizations), encoding="utf-8"
    )
    return result
