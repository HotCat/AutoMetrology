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
from PySide6.QtGui import QTextCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QSplitter, QAbstractItemView, QDoubleSpinBox,
)

from ..models.query import QueryResult
from ..measurement.result_writer import ResultWriter
from ..core.signals import bus
from ..core.i18n import tr


class QueryPanel(QWidget):
    """Panel for writing and evaluating measurement queries."""

    result_selected = Signal(object)  # QueryResult | None
    pair_pick_requested = Signal(str)  # "lines" or "circles"
    pair_pick_cancelled = Signal()
    production_run_requested = Signal()
    production_log_requested = Signal()
    live_query_view_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._results: List[QueryResult] = []
        self._pair_pick_mode: Optional[str] = None
        self._updating_table = False
        self._log_viewer = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._layout = layout
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
            "# circles(ID1, ID2), 0.10  - center distance, abs threshold mm\n"
            "# lines(ID1, ID2), 0.30    - perpendicular distance, abs threshold mm\n"
            "# circle(ID), 0.20          - circle radius, abs threshold mm\n"
            "# arcs(ID), 0.40            - arc radius, abs threshold mm\n"
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
        self._btn_production_run = QPushButton("Run Production")
        self._btn_production_run.setToolTip("Capture camera frame, register, and evaluate queries (F5)")
        self._btn_production_run.clicked.connect(self._request_production_run)
        self._btn_evaluate = QPushButton("Evaluate")
        self._btn_evaluate.clicked.connect(self._evaluate)
        self._btn_export = QPushButton("Export Results")
        self._btn_export.clicked.connect(self._export_results)
        self._btn_logs = QPushButton("View Logs")
        self._btn_logs.setToolTip("Show production measurement logs")
        self._btn_logs.clicked.connect(self.show_production_log_view)
        self._btn_queries = QPushButton("Measurement Queries")
        self._btn_queries.setToolTip("Return to live measurement queries")
        self._btn_queries.clicked.connect(self.show_measurement_query_view)
        self._btn_queries.hide()

        for btn in [
            self._btn_load, self._btn_save, self._btn_production_run,
            self._btn_evaluate, self._btn_export, self._btn_logs, self._btn_queries,
        ]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #333; color: #ccc; border: 1px solid #555;
                    padding: 4px 10px; border-radius: 3px; font-size: 11px;
                }
                QPushButton:hover { background: #444; }
            """)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)

        self._production_shortcut = QShortcut(QKeySequence("F5"), self)
        self._production_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self._production_shortcut.activated.connect(self._request_production_run)

        # Interactive query pair builder
        pick_layout = QHBoxLayout()
        pick_layout.setSpacing(4)
        self._btn_pick_lines = QPushButton("Pick Lines Pair")
        self._btn_pick_lines.clicked.connect(lambda: self._start_pair_pick("lines"))
        self._btn_pick_circles = QPushButton("Pick Circles Pair")
        self._btn_pick_circles.clicked.connect(lambda: self._start_pair_pick("circles"))
        self._btn_pick_circle = QPushButton("Pick Circle")
        self._btn_pick_circle.clicked.connect(lambda: self._start_pair_pick("circle"))
        self._btn_pick_arc = QPushButton("Pick Arc")
        self._btn_pick_arc.clicked.connect(lambda: self._start_pair_pick("arcs"))
        self._btn_cancel_pick = QPushButton("Cancel Pick")
        self._btn_cancel_pick.clicked.connect(self._cancel_pair_pick)
        self._btn_cancel_pick.setEnabled(False)
        self._pair_pick_status = QLabel("Pair picker idle")
        self._pair_pick_status.setStyleSheet("color: #888; font-size: 10px; padding: 4px;")
        self._tol_percent_label = QLabel("Tol %:")
        self._tol_percent_label.setStyleSheet("color: #aaa; font-size: 10px; padding-left: 6px;")
        self._tol_percent = QDoubleSpinBox()
        self._tol_percent.setRange(0.0, 100.0)
        self._tol_percent.setDecimals(3)
        self._tol_percent.setSingleStep(0.1)
        self._tol_percent.setValue(1.0)
        self._tol_percent.setSuffix(" %")
        self._tol_percent.setToolTip("Tolerance percent used when generated queries are added")
        self._tol_percent.setStyleSheet("""
            QDoubleSpinBox {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 3px; border-radius: 3px; font-size: 11px;
            }
        """)

        for btn in [self._btn_pick_lines, self._btn_pick_circles, self._btn_pick_circle, self._btn_pick_arc, self._btn_cancel_pick]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #333; color: #ccc; border: 1px solid #555;
                    padding: 4px 10px; border-radius: 3px; font-size: 11px;
                }
                QPushButton:hover { background: #444; }
                QPushButton:disabled { background: #252525; color: #666; }
            """)
            pick_layout.addWidget(btn)
        pick_layout.addWidget(self._tol_percent_label)
        pick_layout.addWidget(self._tol_percent)
        pick_layout.addWidget(self._pair_pick_status, stretch=1)
        layout.addLayout(pick_layout)

        # Results table
        self._table = QTableWidget(0, 6)
        self._table.setAlternatingRowColors(True)
        self._table.setHorizontalHeaderLabels(["Query", "Value", "Nominal", "Deviation", "Threshold", "Status"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(24)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemChanged.connect(self._on_item_changed)
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

        self._query_view_widgets = [
            self._editor,
            self._btn_load, self._btn_save, self._btn_production_run,
            self._btn_evaluate, self._btn_export, self._btn_logs,
            self._btn_pick_lines, self._btn_pick_circles, self._btn_pick_circle,
            self._btn_pick_arc, self._btn_cancel_pick, self._pair_pick_status,
            self._tol_percent_label, self._tol_percent, self._table, self._summary,
        ]

    def set_production_log_viewer(self, viewer: QWidget) -> None:
        if self._log_viewer is not None:
            self._layout.removeWidget(self._log_viewer)
            self._log_viewer.setParent(None)
        self._log_viewer = viewer
        self._layout.addWidget(viewer, stretch=1)
        viewer.hide()

    @Slot()
    def show_production_log_view(self) -> None:
        if self._log_viewer is None:
            self.production_log_requested.emit()
            return
        for widget in self._query_view_widgets:
            widget.hide()
        self._btn_queries.show()
        self._log_viewer.show()
        refresh = getattr(self._log_viewer, "refresh", None)
        if callable(refresh):
            refresh()

    @Slot()
    def show_measurement_query_view(self) -> None:
        if self._log_viewer is not None:
            self._log_viewer.hide()
        for widget in self._query_view_widgets:
            widget.show()
        self._btn_queries.hide()
        self.live_query_view_requested.emit()

    def get_query_text(self) -> str:
        return self._editor.toPlainText()

    def tolerance_percent(self) -> float:
        return float(self._tol_percent.value())

    def results(self) -> List[QueryResult]:
        return list(self._results)

    @staticmethod
    def format_query_line(expression: str, tolerance_abs: Optional[float]) -> str:
        if tolerance_abs is None:
            return expression
        return f"{expression}, {tolerance_abs:.4f}"

    def _result_query_line(self, result: QueryResult) -> str:
        if result.instruction is None:
            return ""
        expression = self._query_expression_from_instruction(result.instruction)
        return self.format_query_line(expression, result.instruction.tolerance_abs)

    @staticmethod
    def _query_expression_from_instruction(inst) -> str:
        name = inst.query_type.name
        if name == "CIRCLE_DISTANCE":
            return f"circles({inst.feature_id_1}, {inst.feature_id_2})"
        if name == "LINE_DISTANCE":
            return f"lines({inst.feature_id_1}, {inst.feature_id_2})"
        if name == "CIRCLE_RADIUS":
            return f"circle({inst.feature_id_1})"
        if name == "ARC_RADIUS":
            return f"arcs({inst.feature_id_1})"
        return inst.raw_text

    def _sync_editor_from_results(self) -> None:
        lines = [self._result_query_line(r) for r in self._results if r.instruction]
        self._editor.setPlainText("\n".join(lines) + ("\n" if lines else ""))

    def append_query_expression(self, expression: str, tolerance_abs: Optional[float] = None) -> None:
        """Append one generated query expression to the editor."""
        line = self.format_query_line(expression, tolerance_abs)
        current = self._editor.toPlainText().rstrip()
        next_text = f"{current}\n{line}\n" if current else f"{line}\n"
        self._editor.setPlainText(next_text)
        cursor = self._editor.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._editor.setTextCursor(cursor)

    def set_pair_pick_active(self, mode: Optional[str], selected_count: int = 0) -> None:
        self._pair_pick_mode = mode
        active = mode is not None
        self._btn_pick_lines.setEnabled(not active)
        self._btn_pick_circles.setEnabled(not active)
        self._btn_pick_circle.setEnabled(not active)
        self._btn_pick_arc.setEnabled(not active)
        self._btn_cancel_pick.setEnabled(active)
        if mode == "lines":
            self._pair_pick_status.setText(f"{tr('Picking lines')}: {selected_count}/2")
        elif mode == "circles":
            self._pair_pick_status.setText(f"{tr('Picking circles')}: {selected_count}/2")
        elif mode == "circle":
            self._pair_pick_status.setText(f"{tr('Picking circle')}: {selected_count}/1")
        elif mode == "arcs":
            self._pair_pick_status.setText(f"{tr('Picking arc')}: {selected_count}/1")
        else:
            self._pair_pick_status.setText(tr("Pair picker idle"))

    def set_pair_pick_message(self, message: str) -> None:
        self._pair_pick_status.setText(message)

    def set_results(self, results: List[QueryResult]) -> None:
        self._results = results
        self._updating_table = True
        with QSignalBlocker(self._table):
            self._table.clearSelection()
            self._table.setRowCount(len(results))

            ok_count = 0
            ng_count = 0
            no_meas_count = 0
            for i, r in enumerate(results):
                if (
                    r.instruction is not None
                    and r.tolerance_abs is None
                    and r.nominal is not None
                ):
                    r.tolerance_abs = round(
                        abs(r.nominal) * self.tolerance_percent() / 100.0, 4
                    )
                    r.instruction.tolerance_abs = r.tolerance_abs
                    if r.deviation is not None:
                        r.status = (
                            "ok" if abs(r.deviation) <= r.tolerance_abs else "ng"
                        )

                query_text = self._result_query_line(r) if r.instruction else "—"
                value_text = f"{r.value:.3f}" if r.value is not None else "—"
                nominal_text = f"{r.nominal:.3f}" if r.nominal is not None else "—"
                dev_text = f"{r.deviation:+.3f}" if r.deviation is not None else "—"
                threshold_text = (
                    f"{r.tolerance_abs:.4f}"
                    if r.tolerance_abs is not None else ""
                )
                source_text = f"{r.status} [{r.geometry_source}]"

                items = [
                    query_text, value_text, nominal_text, dev_text,
                    threshold_text, source_text,
                ]
                for col, text in enumerate(items):
                    item = QTableWidgetItem(text)
                    if col == 4 and r.instruction is not None:
                        item.setFlags(item.flags() | Qt.ItemIsEditable)
                    else:
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    if r.status == "ok":
                        item.setForeground(Qt.white)
                        if col == 5:
                            item.setForeground(Qt.green)
                    elif r.status == "ng":
                        item.setForeground(Qt.red)
                        if col == 5:
                            item.setForeground(Qt.red)
                    elif r.status == "no_measurement":
                        item.setForeground(Qt.yellow)
                        if col == 5:
                            item.setForeground(Qt.yellow)
                    else:
                        item.setForeground(Qt.red)
                    self._table.setItem(i, col, item)

                if r.status == "ok":
                    ok_count += 1
                elif r.status == "ng":
                    ng_count += 1
                elif r.status == "no_measurement":
                    no_meas_count += 1

        self._updating_table = False
        error_count = len(results) - ok_count - ng_count - no_meas_count
        parts = [f"{tr('OK')}: {ok_count}"]
        if ng_count:
            parts.append(f"{tr('NG')}: {ng_count}")
        if no_meas_count:
            parts.append(f"{tr('No Measurement')}: {no_meas_count}")
        if error_count:
            parts.append(f"{tr('Errors')}: {error_count}")
        self._summary.setText(
            f"{tr('Evaluated')}: {len(results)} {tr('Query')} | " + " | ".join(parts)
        )
        self.result_selected.emit(None)

    @Slot(QTableWidgetItem)
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item.column() != 4:
            return
        row = item.row()
        if row < 0 or row >= len(self._results):
            return
        result = self._results[row]
        if result.instruction is None:
            return
        text = item.text().strip()
        try:
            tolerance = float(text) if text else None
            if tolerance is not None and tolerance < 0:
                raise ValueError
        except ValueError:
            item.setText(
                f"{result.instruction.tolerance_abs:.4f}"
                if result.instruction.tolerance_abs is not None else ""
            )
            return

        result.instruction.tolerance_abs = tolerance
        result.tolerance_abs = tolerance
        if result.deviation is not None and tolerance is not None:
            result.status = "ok" if abs(result.deviation) <= tolerance else "ng"
        elif result.deviation is not None:
            result.status = "ok"
        self._sync_editor_from_results()
        self.set_results(self._results)

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
    def _start_pair_pick(self, mode: str) -> None:
        self.set_pair_pick_active(mode, 0)
        self.pair_pick_requested.emit(mode)

    @Slot()
    def _cancel_pair_pick(self) -> None:
        self.set_pair_pick_active(None)
        self.pair_pick_cancelled.emit()

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
            if self._results:
                self._sync_editor_from_results()
            with open(path, 'w') as f:
                f.write(self._editor.toPlainText())

    @Slot()
    def _evaluate(self) -> None:
        """Emit signal to trigger evaluation (handled by MainWindow)."""
        bus.queries_evaluated.emit(0)  # placeholder, MainWindow will handle

    @Slot()
    def _request_production_run(self) -> None:
        self.production_run_requested.emit()

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
