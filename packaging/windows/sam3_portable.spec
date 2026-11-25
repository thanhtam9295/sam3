# PyInstaller spec for the portable SAM 3 CLI (one-folder bundle).
# Build: packaging/windows/build.ps1   (do not commit dist/ or build/)
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]

hidden = collect_submodules("sam3") + collect_submodules("timm")
datas = collect_data_files("sam3", includes=["assets/*.txt.gz"])  # BPE vocab only

a = Analysis(
    [str(ROOT / "packaging" / "windows" / "sam3_cli.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hidden,
    # Never bundle weights, data or credentials.
    excludes=["tkinter", "matplotlib", "IPython", "jupyter", "notebook", "tensorboard"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="sam3-cli", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="sam3-portable")
