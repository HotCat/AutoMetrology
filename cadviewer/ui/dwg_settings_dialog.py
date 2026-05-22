"""
DWGSettingsDialog — configure DWG converter backend.

Supports both ODA File Converter and libredwg (dwg2dxf).
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QFileDialog, QDialogButtonBox,
)


class DWGSettingsDialog(QDialog):
    """Dialog for configuring DWG converter backend."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure DWG Converter")
        self.setMinimumWidth(500)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #cccccc; }
            QLabel { color: #cccccc; }
            QLineEdit {
                background: #2d2d2d; color: #cccccc;
                border: 1px solid #3d3d3d; border-radius: 3px; padding: 4px;
            }
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 6px 16px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
        """)

        layout = QVBoxLayout(self)

        # Auto-detect status
        self._detect_label = QLabel("Checking converters...")
        self._detect_label.setWordWrap(True)
        self._detect_label.setStyleSheet("color: #aaa; margin-bottom: 8px;")
        layout.addWidget(self._detect_label)

        # Path input for manual override
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(
            "Path to ODAFileConverter or dwg2dxf (auto-detected if empty)..."
        )
        path_layout.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Test button
        test_layout = QHBoxLayout()
        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._test_connection)
        test_layout.addWidget(self._test_btn)
        self._test_result = QLabel("")
        test_layout.addWidget(self._test_result)
        test_layout.addStretch()
        layout.addLayout(test_layout)

        # Install hints
        hint = QLabel(
            "Install options:\n"
            "  ODA File Converter: https://www.opendesign.com/guestfiles/oda_file_converter\n"
            "  libredwg: sudo apt install libredwg-utils  (Ubuntu/Debian)"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #777; font-size: 11px; margin-top: 8px;")
        layout.addWidget(hint)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        # Detect current installation
        self._detect_backends()

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Converter Executable",
            str(Path.cwd()), "All Files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def _detect_backends(self) -> None:
        """Auto-detect available converter backends."""
        from ..converters.oda_cli import ODACLI, LibreDWGCLI

        oda = ODACLI()
        oda_exe = oda.find_executable()

        libredwg = LibreDWGCLI()
        dwg2dxf_exe = libredwg.find_executable()

        lines = []
        if oda_exe:
            lines.append(f"ODA File Converter: found at {oda_exe}")
        else:
            lines.append("ODA File Converter: not found")

        if dwg2dxf_exe:
            lines.append(f"libredwg (dwg2dxf): found at {dwg2dxf_exe}")
        else:
            lines.append("libredwg (dwg2dxf): not found")

        self._detect_label.setText("Detection results:\n" + "\n".join(lines))

        if oda_exe:
            self._path_edit.setText(str(oda_exe))
            self._test_result.setText("ODA detected")
            self._test_result.setStyleSheet("color: #66cc66;")
        elif dwg2dxf_exe:
            self._path_edit.setText(str(dwg2dxf_exe))
            self._test_result.setText("dwg2dxf detected")
            self._test_result.setStyleSheet("color: #66cc66;")
        else:
            self._test_result.setText("No converter found")
            self._test_result.setStyleSheet("color: #ff6666;")

    def _test_connection(self) -> None:
        """Test if the configured path works."""
        path = self._path_edit.text().strip()
        if not path:
            # Test auto-detected backend
            from ..converters.oda_cli import auto_detect_backend
            backend = auto_detect_backend()
            if backend:
                info = backend.get_installation_info()
                if info.installed:
                    self._test_result.setText(f"OK — {backend.backend_name} detected")
                    self._test_result.setStyleSheet("color: #66cc66;")
                    return
            self._test_result.setText("No converter available")
            self._test_result.setStyleSheet("color: #ff6666;")
            return

        # Test specific path
        from ..converters.oda_cli import ODACLI
        cli = ODACLI(executable_path=Path(path))
        info = cli.get_installation_info()
        if info.installed:
            self._test_result.setText("OK — executable works")
            self._test_result.setStyleSheet("color: #66cc66;")
        else:
            self._test_result.setText("Failed — not a valid executable")
            self._test_result.setStyleSheet("color: #ff6666;")

    def get_converter_path(self) -> str:
        return self._path_edit.text().strip()
