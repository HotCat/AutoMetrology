"""
ImageLoadDialog — dialog for loading a telecentric image with calibration params.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QPushButton,
    QFileDialog, QGroupBox, QDialogButtonBox,
)


class ImageLoadDialog(QDialog):
    """Dialog for loading a telecentric product image."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Load Telecentric Image")
        self.setMinimumWidth(400)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #cccccc; }
            QLabel { color: #cccccc; }
            QLineEdit, QDoubleSpinBox {
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

        # File picker
        file_group = QGroupBox("Image File")
        file_group.setStyleSheet(
            "QGroupBox { color: #aaa; border: 1px solid #333; "
            "border-radius: 4px; margin-top: 8px; padding-top: 14px; }"
        )
        file_layout = QHBoxLayout(file_group)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select PNG, BMP, or TIF file...")
        self._path_edit.setReadOnly(True)
        file_layout.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        file_layout.addWidget(browse_btn)
        layout.addWidget(file_group)

        # Calibration params
        cal_group = QGroupBox("Calibration")
        cal_group.setStyleSheet(
            "QGroupBox { color: #aaa; border: 1px solid #333; "
            "border-radius: 4px; margin-top: 8px; padding-top: 14px; }"
        )
        cal_layout = QFormLayout(cal_group)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0.0001, 100.0)
        self._pixel_size.setValue(0.01)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setSuffix(" mm/pixel")
        cal_layout.addRow("Pixel Size:", self._pixel_size)

        layout.addWidget(cal_group)

        # Buttons
        btn_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", str(Path.cwd()),
            "Images (*.png *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._path_edit.setText(path)

    def get_values(self) -> Tuple[str, float]:
        return self._path_edit.text(), self._pixel_size.value()
