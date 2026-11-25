"""Process resources: peak memory, available memory and free disk, on every platform."""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path


def peak_rss_bytes() -> int | None:
    """Peak resident memory of this process (None if it cannot be read)."""
    if sys.platform == "win32":
        return _windows_peak_working_set()
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if platform.system() == "Darwin" else int(peak) * 1024  # Linux: KiB


def _windows_peak_working_set() -> int | None:
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        windll = ctypes.windll  # type: ignore[attr-defined,unused-ignore]
        # HANDLE is pointer-sized: ctypes' default C int would truncate it on 64-bit Windows
        windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        info = windll.kernel32.K32GetProcessMemoryInfo  # kernel32 since Windows 7
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        info.restype = wintypes.BOOL
        handle = windll.kernel32.GetCurrentProcess()
        if info(handle, ctypes.byref(counters), counters.cb):
            return int(counters.PeakWorkingSetSize)
    except Exception:
        return None
    return None


def available_ram_bytes() -> int | None:
    """Physical memory available to new allocations (None if unknown)."""
    from hcs_studio.processing.deconvolution import _available_ram

    value = _available_ram()
    return int(value) if value is not None else None


def free_disk_bytes(path: Path) -> int | None:
    """Free space on the disk holding ``path`` (or its nearest existing parent)."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    try:
        return int(shutil.disk_usage(path).free)
    except OSError:
        return None
