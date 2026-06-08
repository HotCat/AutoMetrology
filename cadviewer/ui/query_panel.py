"""
QueryPanel — dockable panel for the measurement query language.

Provides:
  - Text editor for query input
  - Load/Save query file buttons
  - Evaluate button
  - Results table
  - Export results button
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Slot, Signal, QSignalBlocker
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QSplitter, QAbstractItemView,
)

from ..models.query import QueryResult
from ..measurement.result_writer import ResultWriter
from ..core.signals import bus


class QueryPanel(QWidget):
    """Panel for writing and evaluating measurement queries."""

    result_selected = Signal(object)  # QueryResult | None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._results: List[QueryResult] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Measurement Queries")
        header.setStyleSheet(
            "font-weight: bold; padding: 6px; background: #2d2d2d; color: #ddd;"
        )
        layout.addWidget(header)

        # Query editor
        self._editor = QTextEdit()
        self._editor.setPlaceholderText(
            "# Enter measurement queries, one per line:\n"
            "# circles(ID1, ID2)   — center distance\n"
            "# lines(ID1, ID2)     — perpendicular distance\n"
        )
        self._editor.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a; color: #cccccc;
                border: none; font-family: monospace; font-size: 12px;
                padding: 4px;
            }
        """)
        self._editor.setMaximumHeight(110)
        layout.addWidget(self._editor)

        # Query file buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self._btn_load = QPushButton("Load")
        self._btn_load.clicked.connect(self._load_query_file)
        self._btn_save = QPushButton("Save")
        self._btn_save.clicked.connect(self._save_query_file)
        self._btn_evaluate = QPushButton("Evaluate")
        self._btn_evaluate.clicked.connect(self._evaluate)
        self._btn_export = QPushButton("Export Results")
        self._btn_export.clicked.connect(self._export_results)

        for btn in [self._btn_load, self._btn_save, self._btn_evaluate, self._btn_export]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #333; color: #ccc; border: 1px solid #555;
                    padding: 4px 10px; border-radius: 3px; font-size: 11px;
                }
                QPushButton:hover { background: #444; }
            """)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        # Results table
        self._table = QTableWidget(0, 5)
        self._table.setAlternatingRowColors(True)
        self._table.setHorizontalHeaderLabels(["Query", "Value", "Nominal", "Deviation", "Status"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(24)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a1a; color: #cccccc;
                alternate-background-color: #202020;
                border: none; font-size: 12px; gridline-color: #333;
            }
            QTableWidget::viewport {
                background-color: #1a1a1a;
            }
            QTableWidget::item:selected {
                background-color: #264f78;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #2d2d2d; color: #aaa;
                border: 1px solid #333; padding: 4px;
            }
            QTableCornerButton::section {
                background-color: #2d2d2d;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self._table, stretch=1)

        # Summary
        self._summary = QLabel("No queries evaluated")
        self._summary.setStyleSheet("color: #888; font-size: 10px; padding: 4px;")
        layout.addWidget(self._summary)

    def get_query_text(self) -> str:
        return self._editor.toPlainText()

    def set_results(self, results: List[QueryResult]) -> None:
        self._results = results
        with QSignalBlocker(self._table):
            self._table.clearSelection()
            self._table.setRowCount(len(results))

            ok_count = 0
            no_meas_count = 0
            for i, r in enumerate(results):
                query_text = r.instruction.raw_text if r.instruction else "—"
                value_text = f"{r.value:.3f}" if r.value is not None else "—"
                nominal_text = f"{r.nominal:.3f}" if r.nominal is not None else "—"
                dev_text = f"{r.deviation:+.3f}" if r.deviation is not None else "—"
                source_text = f"{r.status} [{r.geometry_source}]"

                items = [
                    query_text, value_text, nominal_text, dev_text, source_text,
                ]
                for col, text in enumerate(items):
                    item = QTableWidgetItem(text)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if r.status == "ok":
                        item.setForeground(Qt.white)
                        if col == 4:
                            item.setForeground(Qt.green)
                    elif r.status == "no_measurement":
                        item.setForeground(Qt.yellow)
                        if col == 4:
                            item.setForeground(Qt.yellow)
                    else:
                        item.setForeground(Qt.red)
                    self._table.setItem(i, col, item)

                if r.status == "ok":
                    ok_count += 1
                elif r.status == "no_measurement":
                    no_meas_count += 1

        error_count = len(results) - ok_count - no_meas_count
        parts = [f"OK: {ok_count}"]
        if no_meas_count:
            parts.append(f"No Measurement: {no_meas_count}")
        if error_count:
            parts.append(f"Errors: {error_count}")
        self._summary.setText(
            f"Evaluated: {len(results)} queries | " + " | ".join(parts)
        )
        self.result_selected.emit(None)

    @Slot()
    def _on_selection_changed(self) -> None:
        selected = self._table.selectionModel().selectedRows()
        if not selected:
            self.result_selected.emit(None)
            return
        row = selected[0].row()
        if 0 <= row < len(self._results):
            self.result_selected.emit(self._results[row])
        else:
            self.result_selected.emit(None)

    @Slot()
    def _load_query_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Query File", str(Path.cwd()),
            "Query Files (*.txt *.query);;All Files (*)",
        )
        if path:
            with open(path, 'r') as f:
                self._editor.setPlainText(f.read())

    @Slot()
    def _save_query_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Query File", "measurements.txt",
            "Query Files (*.txt)",
        )
        if path:
            with open(path, 'w') as f:
                f.write(self._editor.toPlainText())

    @Slot()
    def _evaluate(self) -> None:
        """Emit signal to trigger evaluation (handled by MainWindow)."""
        bus.queries_evaluated.emit(0)  # placeholder, MainWindow will handle

    @Slot()
    def _export_results(self) -> None:
        if not self._results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "results.txt",
            "Text Files (*.txt);;CSV Files (*.csv)",
        )
        if path:
            if path.endswith('.csv'):
                ResultWriter.write_csv(self._results, path)
            else:
                ResultWriter.write_results(self._results, path)
