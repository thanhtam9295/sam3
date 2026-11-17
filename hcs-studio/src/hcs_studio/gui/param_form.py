"""Parameter form generated from a module's pydantic model."""

from __future__ import annotations

from typing import Any, Literal, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QWidget


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(yaml.safe_dump(value, default_flow_style=True)).replace("\n...", "").strip()


def format_errors(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        where = ".".join(str(p) for p in item["loc"]) or "parameters"
        lines.append(f"{where}: {item['msg']}")
    return "; ".join(lines)


class ParamForm(QWidget):
    """One editor per model field: combo for choices, checkbox for booleans, text otherwise.

    Text fields accept YAML, so lists (``[1, 2]``) and mappings (``{1: 0.4}``) can be typed.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._model: type[BaseModel] | None = None
        self._editors: dict[str, QWidget] = {}

    def set_model(
        self, model: type[BaseModel] | None, values: dict[str, Any] | None = None
    ) -> None:
        while self._layout.rowCount():
            self._layout.removeRow(0)
        self._editors.clear()
        self._model = model
        if model is None:
            return
        values = values or {}
        for name, info in model.model_fields.items():
            if info.exclude:
                continue
            value = values.get(
                name, None if info.is_required() else info.get_default(call_default_factory=True)
            )
            annotation = info.annotation
            editor: QWidget
            if get_origin(annotation) is Literal:
                combo = QComboBox()
                combo.addItems([str(a) for a in get_args(annotation)])
                combo.setCurrentText(str(value))
                combo.currentTextChanged.connect(self.changed)
                editor = combo
            elif annotation is bool:
                box = QCheckBox()
                box.setChecked(bool(value))
                box.toggled.connect(self.changed)
                editor = box
            else:
                edit = QLineEdit(_to_text(value))
                if info.is_required():
                    edit.setPlaceholderText("required")
                edit.editingFinished.connect(self.changed)
                editor = edit
            label = QLabel(name.replace("_", " "))
            if info.description:
                label.setToolTip(info.description)
                editor.setToolTip(info.description)
            self._layout.addRow(label, editor)
            self._editors[name] = editor

    def editor(self, name: str) -> QWidget:
        return self._editors[name]

    def values(self) -> dict[str, Any]:
        """Validated parameters. Raises ``ValueError`` describing every invalid field."""
        if self._model is None:
            return {}
        raw: dict[str, Any] = {}
        for name, editor in self._editors.items():
            if isinstance(editor, QComboBox):
                raw[name] = editor.currentText()
            elif isinstance(editor, QCheckBox):
                raw[name] = editor.isChecked()
            elif isinstance(editor, QLineEdit):
                text = editor.text().strip()
                if not text:
                    if not self._model.model_fields[name].is_required():
                        raw[name] = None
                    continue
                try:
                    raw[name] = yaml.safe_load(text)
                except yaml.YAMLError:
                    raise ValueError(f"{name}: cannot read {text!r}") from None
        try:
            return self._model.model_validate(raw).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(format_errors(exc)) from None
