"""Dialog to map a TIFF filename pattern to well / field / channel / z when detection fails."""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.io.generic import DEFAULT_PATTERNS


class PatternDialog(QDialog):
    def __init__(self, folder: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Describe the file names")
        self.resize(640, 420)
        self._names = sorted(
            p.name for p in folder.iterdir() if p.suffix.lower() in (".tif", ".tiff")
        )[:8]
        intro = QLabel(
            f"The format of <b>{folder.name}</b> was not recognised. Enter a regular expression "
            "with the named groups <code>well</code> (or <code>row</code> and <code>col</code>), "
            "and optionally <code>field</code>, <code>channel</code>, <code>z</code>, "
            "<code>t</code>."
        )
        intro.setWordWrap(True)
        self.pattern = QLineEdit(DEFAULT_PATTERNS[0])
        self.pixel_size = QLineEdit()
        self.pixel_size.setPlaceholderText("optional")
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        form = QFormLayout()
        form.addRow("Pattern", self.pattern)
        form.addRow("Pixel size (µm)", self.pixel_size)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(QLabel("How the first files are read:"))
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.buttons)
        self.pattern.textChanged.connect(self._update)
        self.pixel_size.textChanged.connect(self._update)
        self._update()

    def pixel_size_um(self) -> float | None:
        text = self.pixel_size.text().strip().replace(",", ".")
        if not text:
            return None
        value = float(text)
        if value <= 0:
            raise ValueError("must be positive")
        return value

    def _update(self) -> None:
        ok = True
        lines: list[str] = []
        try:
            compiled = re.compile(self.pattern.text())
        except re.error as exc:
            compiled, ok = None, False
            lines.append(f"Invalid pattern: {exc}")
        if compiled is not None:
            groups = set(compiled.groupindex)
            if "well" not in groups and not {"row", "col"} <= groups:
                ok = False
                lines.append("The pattern needs a 'well' group, or 'row' and 'col' groups.")
            matched = 0
            for name in self._names:
                match = compiled.search(name)
                if match is None:
                    lines.append(f"{name}  →  no match")
                else:
                    matched += 1
                    parts = ", ".join(f"{k}={v}" for k, v in match.groupdict().items() if v)
                    lines.append(f"{name}  →  {parts}")
            ok = ok and matched > 0
        try:
            self.pixel_size_um()
        except ValueError:
            ok = False
            lines.append("Pixel size must be a positive number.")
        self.preview.setPlainText("\n".join(lines))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)
