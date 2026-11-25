"""Background work for the GUI: opening plates and rendering thumbnails off the UI thread."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from hcs_studio.io.base import open_plate
from hcs_studio.model.plate import Field, Plate
from hcs_studio.utils.display import channel_colormaps, make_thumbnail


class _Task(QRunnable):
    def __init__(self, work: Callable[[], None]) -> None:
        super().__init__()
        self._work = work

    def run(self) -> None:
        self._work()


class PlateLoader(QObject):
    """Opens a plate in a worker thread. Results of superseded requests are dropped."""

    loaded = Signal(object)  # Plate, as read from the files (no overrides applied)
    failed = Signal(str, str)  # path, message

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def request(self, path: Path, opener: Callable[[Path], Plate] = open_plate) -> None:
        self._generation += 1
        generation = self._generation

        def work() -> None:
            try:
                plate = opener(path)
            except Exception as exc:
                if generation == self._generation:
                    self.failed.emit(str(path), f"{type(exc).__name__}: {exc}")
                return
            if generation == self._generation:
                self.loaded.emit(plate)

        self._pool.start(_Task(work))

    def wait(self) -> None:
        self._pool.waitForDone()


class ThumbnailLoader(QObject):
    """Renders field thumbnails in worker threads, with a per-plate cache."""

    ready = Signal(object, object)  # key (row, col, field index), RGB uint8 array

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._cache: dict[tuple[int, int, int], Any] = {}
        self._pending: set[tuple[int, int, int]] = set()
        self._colormaps: list[str] = []
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(2)

    def set_plate(self, plate: Plate | None) -> None:
        self._generation += 1
        self._pool.clear()
        self._cache.clear()
        self._pending.clear()
        self._colormaps = channel_colormaps(plate.metadata.channels) if plate else []

    def idle(self) -> bool:
        """No thumbnail is being rendered (all requested ones have been emitted)."""
        return not self._pending and self._pool.activeThreadCount() == 0

    def request(self, key: tuple[int, int, int], field: Field) -> None:
        if key in self._cache:
            self.ready.emit(key, self._cache[key])
            return
        if key in self._pending:
            return
        self._pending.add(key)
        generation, colormaps = self._generation, list(self._colormaps)

        def work() -> None:
            try:
                image = make_thumbnail(field.load(), colormaps)
            except Exception:
                image = None  # an unreadable field simply has no thumbnail
            if generation != self._generation:
                return
            self._pending.discard(key)
            if image is not None:
                self._cache[key] = image
                self.ready.emit(key, image)

        self._pool.start(_Task(work))

    def wait(self) -> None:
        self._pool.waitForDone()


class PreviewWorker(QObject):
    """Runs a pipeline on one field in a worker thread; only the latest request reports."""

    finished = Signal(object, object)  # raw ImageData, ProcessResult
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def request(self, work: Callable[[], tuple[Any, Any]]) -> None:
        self._generation += 1
        generation = self._generation

        def run() -> None:
            try:
                raw, result = work()
            except Exception as exc:
                if generation == self._generation:
                    self.failed.emit(str(exc) or type(exc).__name__)
                return
            if generation == self._generation:
                self.finished.emit(raw, result)

        self._pool.start(_Task(run))

    def wait(self) -> None:
        self._pool.waitForDone()


class BackgroundTask(QObject):
    """Runs one callable at a time in a worker thread; only the latest request reports."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._generation = 0
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)

    def request(self, work: Callable[[], Any]) -> None:
        self._generation += 1
        generation = self._generation

        def run() -> None:
            try:
                result = work()
            except Exception as exc:
                if generation == self._generation:
                    self.failed.emit(str(exc) or type(exc).__name__)
                return
            if generation == self._generation:
                self.finished.emit(result)

        self._pool.start(_Task(run))

    def wait(self) -> None:
        self._pool.waitForDone()
