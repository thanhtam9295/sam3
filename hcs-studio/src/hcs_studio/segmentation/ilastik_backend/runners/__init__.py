"""ilastik runners: each takes an ``IlastikJob`` and returns an ``IlastikResult``."""

from hcs_studio.segmentation.ilastik_backend.runners.headless import HeadlessRunner

__all__ = ["HeadlessRunner"]
