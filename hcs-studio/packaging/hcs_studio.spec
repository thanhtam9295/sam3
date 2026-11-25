# PyInstaller spec for HCS Studio (one-folder bundle, two executables).
#
#   pyinstaller packaging/hcs_studio.spec --noconfirm
#
# Environment: HCS_BUNDLE_SEGMENTATION=0 leaves out PyTorch and Cellpose (a smaller
# bundle without Cellpose segmentation). SAM 3 is never bundled: it needs its own Python
# environment (Tools → SAM 3 settings).
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).parent  # noqa: F821 (defined by PyInstaller)
SEGMENTATION = os.environ.get("HCS_BUNDLE_SEGMENTATION", "1") != "0"

datas, binaries, hiddenimports = [], [], []
packages = [
    "hcs_studio",
    "napari",
    "napari_builtins",
    "napari_resources",
    "napari_console",
    "napari_plugin_engine",
    "in_n_out",
    "psygnal",
    "pint",
    "OpenGL",
    "napari_svg",
    "vispy",
    "magicgui",
    "app_model",
    "npe2",
    "superqt",
    "pyqtgraph",
    "psfmodels",
    "nd2",
    "imagecodecs",
    "skimage",
    "mahotas",
    "reportlab",
    "openpyxl",
    "imageio",
    "imageio_ffmpeg",
    "h5py",
    "sqlalchemy",
    "alembic",
    "jsonschema",
    "jsonschema_specifications",
    "keyring",
    "argon2",
    "_argon2_cffi_bindings",
    "lab_events",
    "anthropic",  # lab assistant (optional extra; skipped when not installed)
    "umap",
    "pynndescent",
]
if SEGMENTATION:
    packages += ["cellpose"]
for package in packages:
    try:
        d, b, h = collect_all(package)
    except Exception as exc:  # optional package not installed
        print(f"spec: skipping {package}: {exc}")
        continue
    datas += d
    binaries += b
    hiddenimports += h
# Packages that look up their own version or entry points at run time.
for package in ("napari", "npe2", "napari-builtins", "napari-svg", "napari-resources",
                "napari-console", "imagecodecs", "numba", "pydantic", "magicgui", "app-model",
                "tifffile", "nd2", "pint", "hcs-studio", "keyring", "alembic", "sqlalchemy",
                "jsonschema", "lab-events"):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass
hiddenimports += collect_submodules("sklearn")
# Alembic reads the registry migrations as .py files from disk at start-up.
datas += collect_data_files("hcs_studio", includes=["db/migrations/**/*"], include_py_files=True)
# Scripts another Python runs by file path (SAM 3's and ilastik's own environments): they
# must exist as .py files in the bundle, not only inside the PYZ archive.
WORKER_SCRIPTS = [
    "segmentation/sam3_worker.py",
    "segmentation/ilastik_backend/api_worker.py",
]
datas += collect_data_files("hcs_studio", includes=WORKER_SCRIPTS, include_py_files=True)
hiddenimports += collect_submodules("keyring.backends")
hiddenimports += ["sqlalchemy.dialects.sqlite", "sqlalchemy.dialects.postgresql"]
# ome-types (used by nd2) registers a napari plugin; its manifest must travel with it.
datas += collect_data_files("ome_types")
hiddenimports += ["scipy.special._cdflib", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets"]
# Loaded by name at run time (figure export): found by `hcs-studio selftest`.
hiddenimports += [
    "matplotlib.backends.backend_agg",
    "matplotlib.backends.backend_svg",
    "mpl_toolkits.mplot3d",
    "matplotlib.backends.backend_pdf",
]

excludes = ["tkinter", "matplotlib.tests", "numpy.tests", "sam3"]
if not SEGMENTATION:
    excludes += ["torch", "torchvision", "cellpose", "triton"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
icon = str(ROOT / "packaging" / "hcs_studio.ico") if sys.platform == "win32" else None
gui = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HCS Studio",
    console=False,
    icon=icon,
)
cli = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="hcs-studio",
    console=True,
    icon=icon,
)
COLLECT(gui, cli, a.binaries, a.datas, name="HCS Studio")  # noqa: F821
