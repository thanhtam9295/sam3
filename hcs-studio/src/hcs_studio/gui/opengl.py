"""Software OpenGL for PCs whose graphics driver gives napari no OpenGL 2.1 context (Remote
Desktop sessions, virtual machines, old drivers, CI runners).

Must run before Qt, napari, vispy or PyOpenGL are imported (``app.main`` calls
:func:`prepare_opengl` for the ``gui`` command), and imports none of them.

Windows: Qt's software mode alone does not work for napari: Qt would create the context in
``opengl32sw.dll`` while PyOpenGL (vispy's backend) calls ``WinDLL("opengl32")``, i.e. the
system ``opengl32.dll``, which then has no current context (GLError 1282). So Qt's software
renderer (Mesa llvmpipe, shipped with PySide6 as ``opengl32sw.dll``) is copied to
``<settings>/opengl/opengl32.dll`` and loaded first: Windows resolves later loads of a module
named ``opengl32.dll`` to the one already loaded, so PyOpenGL gets it too, and Qt is pointed at
the same file with ``QT_OPENGL_DLL``. Linux/macOS: Mesa's ``LIBGL_ALWAYS_SOFTWARE=1``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

SOFTWARE_GL_ENV = "HCS_SOFTWARE_OPENGL"
SOFTWARE_DLL = "opengl32sw.dll"


def _config_dir() -> Path:
    from hcs_studio.io.sidecar import config_dir

    return config_dir()


def software_marker() -> Path:
    return _config_dir() / "software_opengl"


def use_software_opengl() -> bool:
    """When ``HCS_SOFTWARE_OPENGL=1`` or after the driver failed once. ``HCS_SOFTWARE_OPENGL=0``
    and deleting ``<settings>/software_opengl`` go back to the driver."""
    choice = os.environ.get(SOFTWARE_GL_ENV, "")
    if choice in ("0", "1"):
        return choice == "1"
    return software_marker().exists()


def find_software_dll() -> Path | None:
    """Qt's software renderer: next to the Qt DLLs of PySide6 (also inside the installed app)."""
    roots = [Path(getattr(sys, "_MEIPASS", "")) / "PySide6", Path(getattr(sys, "_MEIPASS", ""))]
    try:
        import importlib.util

        spec = importlib.util.find_spec("PySide6")  # locates the package without importing Qt
        if spec is not None and spec.submodule_search_locations:
            roots += [Path(p) for p in spec.submodule_search_locations]
    except (ImportError, ValueError):
        pass
    for root in roots:
        if str(root) not in ("", ".") and (root / SOFTWARE_DLL).is_file():
            return root / SOFTWARE_DLL
    return None


def prepare_opengl() -> str:
    """Switch this process to software OpenGL when wanted; returns what was done (for the log)."""
    if not use_software_opengl():
        return ""
    if sys.platform != "win32":
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        return "software OpenGL (Mesa, LIBGL_ALWAYS_SOFTWARE=1)"
    source = find_software_dll()
    if source is None:
        return f"software OpenGL requested, but {SOFTWARE_DLL} was not found"
    target = _config_dir() / "opengl" / "opengl32.dll"
    if not target.is_file() or target.stat().st_size != source.stat().st_size:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    import ctypes

    ctypes.WinDLL(str(target))  # first module named opengl32.dll: PyOpenGL's load resolves to it
    os.environ["QT_OPENGL_DLL"] = str(target)  # and Qt creates its contexts with the same DLL
    os.environ["QT_OPENGL"] = "desktop"
    return f"software OpenGL ({source.name} as {target})"
