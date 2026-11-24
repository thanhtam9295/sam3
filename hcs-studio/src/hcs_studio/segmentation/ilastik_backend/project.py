"""Read an ilastik project (``.ilp``, HDF5) without changing or unpickling it.

What is read, and how sure we are (checked against ilastik's source; **not yet confirmed on
trained projects from the lab** — sample ``.ilp`` files are still awaited):

- ``workflowName`` and ``ilastikVersion`` (root datasets): confirmed in ilastik's project
  writer.
- class names (``PixelClassification/LabelNames``): confirmed for Pixel Classification;
  Autocontext stores one ``LabelNames`` per stage, found by searching the file.
- the training data's axis order and channel count: searched in ``Input Data`` (``axistags``
  datasets/attributes, ``shape``); returned with ``confirmed=False`` until checked on real
  projects.

The classifier itself is a pickle: it is never read. The file's SHA-256 is recorded in run
manifests so results trace back to the exact trained classifier.
"""

from __future__ import annotations

import contextlib
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: workflowName substring → HCS Studio's workflow key (most specific first)
WORKFLOWS = (
    ("Autocontext", "autocontext"),
    ("ObjectClassification", "objects"),
    ("Multicut", "multicut"),
    ("Tracking", "tracking"),
    ("PixelClassification", "pixel"),
    ("DataConversion", "conversion"),
)


@dataclass
class ProjectInfo:
    path: str
    sha256: str
    workflow: str  # pixel | autocontext | objects | multicut | tracking | unknown
    workflow_name: str = ""
    ilastik_version: str = ""
    label_names: list[str] = field(default_factory=list)
    #: axis order of the training data (ilastik letters), if found
    axes: str | None = None
    channels: int | None = None
    #: False until the location of axes/channels is confirmed on real projects
    confirmed: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def dims(self) -> str | None:
        if self.axes is None:
            return None
        return "3D" if "z" in self.axes else "2D"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def workflow_key(name: str) -> str:
    return next((key for text, key in WORKFLOWS if text.lower() in name.lower()), "unknown")


def read_project(path: str | Path) -> ProjectInfo:
    """The workflow, version, class names and (if found) training axes of a ``.ilp``."""
    import h5py

    from hcs_studio.segmentation.ilastik_backend.h5io import parse_axistags

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"ilastik project not found: {path}")
    info = ProjectInfo(path=str(path), sha256=file_sha256(path), workflow="unknown")
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise ValueError(f"{path.name} is not an ilastik project (not HDF5): {exc}") from exc
    with handle as f:
        if "workflowName" in f:
            info.workflow_name = _text(f["workflowName"][()])
            info.workflow = workflow_key(info.workflow_name)
        else:
            info.notes.append("no workflowName dataset")
        if "ilastikVersion" in f:
            info.ilastik_version = _text(f["ilastikVersion"][()])
        names: list[str] = []
        axes_found: list[str] = []
        shapes: list[tuple[int, ...]] = []

        def visit(name: str, obj: Any) -> None:
            leaf = name.rsplit("/", 1)[-1]
            if isinstance(obj, h5py.Dataset):
                if leaf == "LabelNames" and not names:
                    with contextlib.suppress(TypeError, ValueError):
                        names.extend(_text(v) for v in obj[()])
                elif leaf == "axistags" and name.startswith("Input Data"):
                    with contextlib.suppress(ValueError, KeyError):
                        axes_found.append(parse_axistags(_text(obj[()])))
                elif leaf == "shape" and name.startswith("Input Data"):
                    with contextlib.suppress(TypeError, ValueError):
                        shapes.append(tuple(int(v) for v in obj[()]))
            if name.startswith("Input Data") and "axistags" in obj.attrs:
                with contextlib.suppress(ValueError, KeyError):
                    axes_found.append(parse_axistags(_text(obj.attrs["axistags"])))

        f.visititems(visit)
        info.label_names = names
        if axes_found:
            info.axes = axes_found[0]
            if shapes and "c" in info.axes and len(shapes[0]) == len(info.axes):
                info.channels = shapes[0][info.axes.index("c")]
            elif "c" not in info.axes:
                info.channels = 1
        else:
            info.notes.append("training-data axes not found (unconfirmed project layout)")
    return info


def check_compatible(info: ProjectInfo, axes: str, channels: int) -> list[str]:
    """Problems running ``info`` on data with ``axes`` and ``channels`` (empty = fine).
    Only what the project states is checked; unknown training axes are not an error."""
    problems = []
    if info.axes is not None:
        if ("z" in info.axes) != ("z" in axes):
            problems.append(
                f"the project was trained on {info.dims} data but the input is "
                f"{'3D' if 'z' in axes else '2D'}: add or remove a Z projection step"
            )
        if info.channels is not None and info.channels != channels:
            problems.append(
                f"the project was trained with {info.channels} channel(s); {channels} are sent "
                "(choose the step's channels)"
            )
    return problems
