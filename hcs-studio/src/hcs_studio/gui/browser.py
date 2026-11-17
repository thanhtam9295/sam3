"""Data browser dock: favourites plus a file-system tree with plate/vendor badges."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import QDir, QModelIndex, QPersistentModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.io.base import detect_reader
from hcs_studio.io.sidecar import config_dir

_FILE_FILTERS = ["*.nd2", "*.ome.tif", "*.ome.tiff", "*.htd", "*.HTD"]


def favorites_file() -> Path:
    return config_dir() / "favorites.json"


def load_favorites() -> list[Path]:
    try:
        data = json.loads(favorites_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [Path(p) for p in data if isinstance(p, str)]


def save_favorites(paths: list[Path]) -> None:
    file = favorites_file()
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps([str(p) for p in paths], indent=2), encoding="utf-8")


class PlateFileSystemModel(QFileSystemModel):
    """File-system model that marks folders and files recognised as plates."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._vendors: dict[str, str | None] = {}
        self.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        self.setNameFilters(_FILE_FILTERS)
        self.setNameFilterDisables(False)

    def vendor(self, path: str) -> str | None:
        """Vendor badge for a path, or ``None``. Cached: detection lists the directory."""
        if path not in self._vendors:
            try:
                reader = detect_reader(Path(path))
            except OSError:
                reader = None
            self._vendors[path] = reader.vendor if reader else None
        return self._vendors[path]

    def data(
        self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole
    ) -> Any:
        value = super().data(index, role)
        if index.column() == 0 and role == Qt.ItemDataRole.DisplayRole and self.isDir(index):
            # Drive roots are skipped: scanning them is slow and they are never plates.
            path = self.filePath(index)
            if Path(path).parent != Path(path):
                vendor = self.vendor(path)
                if vendor:
                    return f"▦ {value}  [{vendor}]"
        return value


class BrowserPanel(QWidget):
    """Emits ``plate_requested(path)`` when the user opens a folder or file."""

    plate_requested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model = PlateFileSystemModel(self)
        self.model.setRootPath("")
        self.tree = QTreeView()
        self.tree.setModel(self.model)
        for column in (1, 2, 3):
            self.tree.hideColumn(column)
        self.tree.setHeaderHidden(True)
        self.tree.doubleClicked.connect(self._on_activated)

        self.favorites = QListWidget()
        self.favorites.setMaximumHeight(110)
        self.favorites.itemClicked.connect(self._on_favorite)
        self._favorite_paths: list[Path] = []
        self._reload_favorites()

        self.open_button = QPushButton("Open as plate")
        self.open_button.clicked.connect(self.open_selected)
        self.add_button = QPushButton("★ Add favourite")
        self.add_button.clicked.connect(self.add_selected_favorite)
        self.remove_button = QPushButton("Remove")
        self.remove_button.clicked.connect(self.remove_selected_favorite)

        buttons = QHBoxLayout()
        buttons.addWidget(self.open_button)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Favourites"))
        layout.addWidget(self.favorites)
        layout.addWidget(QLabel("Drives and folders (double-click a plate to open)"))
        layout.addWidget(self.tree, 1)
        layout.addLayout(buttons)

    def selected_path(self) -> Path | None:
        index = self.tree.currentIndex()
        return Path(self.model.filePath(index)) if index.isValid() else None

    def reveal(self, path: Path) -> None:
        """Expand the tree to ``path`` and select it."""
        index = self.model.index(str(path))
        if index.isValid():
            self.tree.setCurrentIndex(index)
            self.tree.scrollTo(index)
            self.tree.expand(index)

    def open_selected(self) -> None:
        path = self.selected_path()
        if path is not None:
            self.plate_requested.emit(path)

    def _on_activated(self, index: QModelIndex) -> None:
        path = Path(self.model.filePath(index))
        if not self.model.isDir(index) or self.model.vendor(str(path)):
            self.plate_requested.emit(path)

    def _reload_favorites(self) -> None:
        self._favorite_paths = load_favorites()
        self.favorites.clear()
        for path in self._favorite_paths:
            item = QListWidgetItem(path.name or str(path))
            item.setToolTip(str(path))
            self.favorites.addItem(item)

    def add_selected_favorite(self) -> None:
        path = self.selected_path()
        if path is None or path in self._favorite_paths:
            return
        save_favorites([*self._favorite_paths, path])
        self._reload_favorites()

    def remove_selected_favorite(self) -> None:
        row = self.favorites.currentRow()
        if row < 0:
            return
        save_favorites([p for i, p in enumerate(self._favorite_paths) if i != row])
        self._reload_favorites()

    def _on_favorite(self, item: QListWidgetItem) -> None:
        path = self._favorite_paths[self.favorites.row(item)]
        self.reveal(path)
