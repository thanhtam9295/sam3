"""First-run dialog for SAM 3: Python environment, checkpoint, device."""

from __future__ import annotations

import subprocess

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hcs_studio.segmentation.sam3_backend import Sam3Settings

INTRO = (
    "SAM 3 runs in its own Python environment (it needs numpy &lt; 2). Install it there with "
    "<code>pip install -e &lt;sam3 folder&gt;</code> plus <code>einops pycocotools psutil "
    "'setuptools&lt;81'</code>.<br><br>The weights are <b>gated</b>: request access to "
    "<i>facebook/sam3</i> on Hugging Face and wait for approval. Then either run "
    "<code>huggingface-cli login</code> in that environment (the weights download on first "
    "use), or download <code>sam3.pt</code> and choose it below.<br><br>A GPU is strongly "
    "recommended; on the CPU, encoding one image takes about 10 s."
)


class Sam3SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SAM 3 settings")
        self.resize(620, 360)
        settings = Sam3Settings.load()
        intro = QLabel(INTRO)
        intro.setWordWrap(True)
        self.python = QLineEdit(settings.python)
        self.python.setPlaceholderText("e.g. C:\\envs\\sam3\\python.exe")
        self.checkpoint = QLineEdit(settings.checkpoint)
        self.checkpoint.setPlaceholderText("optional: sam3.pt (else downloaded from Hugging Face)")
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "cuda"])
        self.device.setCurrentText(settings.device or "auto")
        self.status = QLabel("")
        self.status.setWordWrap(True)

        def browse(edit: QLineEdit, title: str) -> QPushButton:
            button = QPushButton("…")
            button.setFixedWidth(32)
            button.clicked.connect(
                lambda: edit.setText(QFileDialog.getOpenFileName(self, title)[0] or edit.text())
            )
            return button

        def row(edit: QLineEdit, title: str) -> QHBoxLayout:
            layout = QHBoxLayout()
            layout.addWidget(edit)
            layout.addWidget(browse(edit, title))
            return layout

        form = QFormLayout()
        form.addRow("Python of the SAM 3 environment", row(self.python, "Python executable"))
        form.addRow("Checkpoint", row(self.checkpoint, "SAM 3 checkpoint"))
        form.addRow("Device", self.device)
        self.check_button = QPushButton("Check environment")
        self.check_button.clicked.connect(self.check)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.check_button)
        layout.addWidget(self.status)
        layout.addWidget(buttons)

    def check(self) -> bool:
        python = self.python.text().strip()
        try:
            result = subprocess.run(
                [
                    python,
                    "-c",
                    "import sam3, torch; print(torch.__version__, torch.cuda.is_available())",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.status.setText(f"Could not run {python or '(empty)'}: {exc}")
            return False
        if result.returncode != 0:
            last = (result.stderr.strip().splitlines() or ["unknown error"])[-1]
            self.status.setText(f"SAM 3 cannot be imported there: {last}")
            return False
        version, cuda = result.stdout.split()
        gpu = "a CUDA GPU" if cuda == "True" else "no GPU (CPU only, slow)"
        self.status.setText(f"OK: SAM 3 found, PyTorch {version}, {gpu}.")
        return True

    def save(self) -> None:
        from hcs_studio.services import context
        from hcs_studio.services.operations import save_sam3_settings

        settings = Sam3Settings(
            python=self.python.text().strip(),
            checkpoint=self.checkpoint.text().strip(),
            device=self.device.currentText(),
        )
        save_sam3_settings(context(), settings)
        self.accept()
