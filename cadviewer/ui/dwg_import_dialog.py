"""
DWGImportDialog — dark-themed progress dialog for DWG→DXF conversion.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QPushButton, QHBoxLayout,
)


class DWGImportDialog(QDialog):
    """Progress dialog for DWG import, showing conversion stages."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import DWG")
        self.setMinimumWidth(420)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #cccccc; }
            QLabel { color: #cccccc; }
            QProgressBar {
                background: #2d2d2d; color: #cccccc;
                border: 1px solid #3d3d3d; border-radius: 3px;
                text-align: center; min-height: 20px;
            }
            QProgressBar::chunk { background: #007acc; border-radius: 2px; }
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 6px 16px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
            QPushButton:disabled { color: #666; }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Status label
        self._status = QLabel("Preparing conversion...")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # Detail label
        self._detail = QLabel("")
        self._detail.setStyleSheet("color: #888; font-size: 11px;")
        self._detail.setWordWrap(True)
        layout.addWidget(self._detail)

        # Close button (hidden until completion/error)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.accept)
        self._close_btn.setVisible(False)
        btn_layout.addWidget(self._close_btn)
        layout.addLayout(btn_layout)

    def set_stage(self, stage: str, progress: int) -> None:
        """Update current conversion stage and progress."""
        self._status.setText(stage)
        self._progress.setValue(progress)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def set_detail(self, text: str) -> None:
        """Set detail text below progress bar."""
        self._detail.setText(text)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def set_error(self, message: str) -> None:
        """Show error state."""
        self._status.setText("Conversion failed")
        self._status.setStyleSheet("color: #ff6666; font-weight: bold;")
        self._detail.setText(message)
        self._detail.setStyleSheet("color: #ff6666; font-size: 11px;")
        self._progress.setValue(0)
        self._close_btn.setVisible(True)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()

    def set_complete(self, result) -> None:
        """Show completion summary."""
        self._status.setText("Conversion complete")
        self._status.setStyleSheet("color: #66cc66; font-weight: bold;")
        self._progress.setValue(100)

        lines = [
            f"Duration: {result.duration_seconds:.1f}s",
            f"Entities: {result.entity_count}",
            f"Layers: {result.layer_count}",
        ]
        if result.validation:
            v = result.validation
            if v.warnings:
                lines.append(f"Warnings: {len(v.warnings)}")
            lines.append(f"Validation: {'PASS' if v.is_valid else 'FAIL'}")

        self._detail.setText("\n".join(lines))
        self._detail.setStyleSheet("color: #66cc66; font-size: 11px;")
        self._close_btn.setVisible(True)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
