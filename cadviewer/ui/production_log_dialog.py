"""Embedded production log viewer for SQLite-backed measurement records."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QDate, Qt, Signal, Slot
from PySide6.QtGui import QColor, QTextCharFormat
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCalendarWidget,
    QGroupBox,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..measurement.production_log import ProductionLogStore
from ..models.query import QueryResult
from ..core.i18n import tr


class ProductionLogViewer(QWidget):
    """Calendar-based production log browser."""

    record_selected = Signal(str)
    result_selected = Signal(object)  # QueryResult | None

    def __init__(self, store: ProductionLogStore, parent=None) -> None:
        super().__init__(parent)
        self._store = store
        self._current_record_id = ""
        self._results: list[QueryResult] = []
        self._setup_ui()
        self._refresh_month_marks()
        self._load_day(QDate.currentDate())

    def current_record_id(self) -> str:
        return self._current_record_id

    def refresh(self, select_record_id: str = "") -> None:
        self._refresh_month_marks()
        self._load_day(
            self._calendar.selectedDate(),
            preferred_record_id=select_record_id,
        )

    def _setup_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 6, 0)

        self._calendar = QCalendarWidget()
        self._calendar.setGridVisible(True)
        self._calendar.clicked.connect(self._load_day)
        self._calendar.currentPageChanged.connect(self._on_month_changed)
        self._calendar.selectionChanged.connect(self._on_calendar_selection_changed)
        left_layout.addWidget(self._calendar)

        self._month_summary = QLabel("No production records")
        self._month_summary.setStyleSheet("color: #aaa; padding: 4px;")
        left_layout.addWidget(self._month_summary)

        records_group = QGroupBox("Daily Records")
        records_layout = QVBoxLayout(records_group)
        self._records = QTreeWidget()
        self._records.setHeaderLabels(["Status / Time", "CAD", "Rows"])
        self._records.setSelectionMode(QAbstractItemView.SingleSelection)
        self._records.itemSelectionChanged.connect(self._on_record_selection_changed)
        self._records.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._records.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self._records.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        records_layout.addWidget(self._records)
        left_layout.addWidget(records_group, stretch=1)

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 0, 0, 0)
        self._record_title = QLabel("Select a production record")
        self._record_title.setStyleSheet("font-weight: bold; color: #ddd; padding: 4px;")
        right_layout.addWidget(self._record_title)

        self._record_meta = QLabel("")
        self._record_meta.setWordWrap(True)
        self._record_meta.setStyleSheet("color: #aaa; padding: 4px;")
        right_layout.addWidget(self._record_meta)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Query", "Value", "Nominal", "Deviation", "Threshold", "Status"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_result_selection_changed)
        for col in range(6):
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)
        self._table.verticalHeader().setDefaultSectionSize(24)
        right_layout.addWidget(self._table, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([390, 790])


    @Slot(int, int)
    def _on_month_changed(self, year: int, month: int) -> None:
        self._refresh_month_marks(year, month)

    @Slot()
    def _on_calendar_selection_changed(self) -> None:
        self._load_day(self._calendar.selectedDate())

    def _refresh_month_marks(
        self,
        year: Optional[int] = None,
        month: Optional[int] = None,
    ) -> None:
        if year is None or month is None:
            page = self._calendar.yearShown(), self._calendar.monthShown()
            year, month = page[0], page[1]

        default_format = QTextCharFormat()
        first = date(year, month, 1)
        if month == 12:
            last_day = 31
        else:
            last_day = (date(year, month + 1, 1) - first).days
        for day in range(1, last_day + 1):
            self._calendar.setDateTextFormat(QDate(year, month, day), default_format)

        counts = self._store.month_day_counts(year, month)
        total = sum(v["total"] for v in counts.values())
        ok = sum(v["ok"] for v in counts.values())
        ng = sum(v["ng"] for v in counts.values())
        self._month_summary.setText(
            f"{year:04d}-{month:02d}: {total} {tr('records')} | {tr('OK')} {ok} | {tr('NG')} {ng}"
        )

        for date_text, info in counts.items():
            y, m, d = [int(part) for part in date_text.split("-")]
            fmt = QTextCharFormat()
            fmt.setFontWeight(700)
            if info["ng"]:
                fmt.setBackground(QColor(120, 35, 35))
                fmt.setForeground(QColor(255, 230, 230))
            else:
                fmt.setBackground(QColor(35, 95, 55))
                fmt.setForeground(QColor(230, 255, 235))
            self._calendar.setDateTextFormat(QDate(y, m, d), fmt)

    @Slot(QDate)
    def _load_day(
        self,
        qdate: QDate,
        preferred_record_id: str = "",
    ) -> None:
        date_text = qdate.toString("yyyy-MM-dd")
        records = self._store.records_for_day(date_text)
        self._records.clear()
        groups = {
            "ok": QTreeWidgetItem([tr("OK"), "", ""]),
            "ng": QTreeWidgetItem([tr("NG"), "", ""]),
        }
        for group in groups.values():
            group.setFlags(group.flags() & ~Qt.ItemIsSelectable)
            self._records.addTopLevelItem(group)

        for record in records:
            group_key = "ok" if record["overall_status"] == "ok" else "ng"
            created = str(record["created_at"]).replace("T", " ")
            item = QTreeWidgetItem([
                created.split(" ")[-1],
                record.get("cad_filename", ""),
                str(record.get("query_count", 0)),
            ])
            item.setData(0, Qt.UserRole, record["id"])
            item.setToolTip(
                0,
                f"{record['id']}\nImage: {record.get('image_filename', '')}",
            )
            groups[group_key].addChild(item)

        for group in groups.values():
            group.setExpanded(True)
            group.setText(2, str(group.childCount()))

        if records:
            preferred_item = None
            if preferred_record_id:
                for group in groups.values():
                    for i in range(group.childCount()):
                        item = group.child(i)
                        if item.data(0, Qt.UserRole) == preferred_record_id:
                            preferred_item = item
                            break
                    if preferred_item is not None:
                        break
            if preferred_item is not None:
                self._records.setCurrentItem(preferred_item)
            else:
                first_group = groups["ng"] if groups["ng"].childCount() else groups["ok"]
                self._records.setCurrentItem(first_group.child(0))
        else:
            self._clear_record()

    @Slot()
    def _on_record_selection_changed(self) -> None:
        selected = self._records.selectedItems()
        if not selected:
            return
        record_id = selected[0].data(0, Qt.UserRole)
        if not record_id:
            return
        self._load_record(str(record_id))

    def _load_record(self, record_id: str) -> None:
        record = self._store.get_record(record_id)
        if record is None:
            self._clear_record()
            return
        self._current_record_id = record_id
        self._results = self._store.get_results(record_id)
        self._populate_results(self._results)
        status = str(record["overall_status"]).upper()
        status_label = tr("OK") if status == "OK" else tr("NG")
        self._record_title.setText(
            f"{status_label} | {record['created_at'].replace('T', ' ')}"
        )
        self._record_meta.setText(
            f"CAD: {record.get('cad_filename', '')} | "
            f"{tr('Image')}: {Path(record.get('image_path', '')).name} | "
            f"{tr('OK')} {record.get('ok_count', 0)} | {tr('NG')} {record.get('ng_count', 0)} | "
            f"{tr('No measurement')} {record.get('no_measurement_count', 0)} | "
            f"{tr('Errors')} {record.get('error_count', 0)}"
        )
        self.record_selected.emit(record_id)
        self.result_selected.emit(None)

    def _clear_record(self) -> None:
        self._current_record_id = ""
        self._results = []
        self._record_title.setText(tr("No production records for selected day"))
        self._record_meta.setText("")
        self._table.setRowCount(0)
        self.result_selected.emit(None)

    def _populate_results(self, results: list[QueryResult]) -> None:
        self._table.clearSelection()
        self._table.setRowCount(len(results))
        for row, result in enumerate(results):
            inst = result.instruction
            query_text = inst.raw_text if inst else ""
            values = [
                query_text,
                f"{result.value:.3f}" if result.value is not None else "-",
                f"{result.nominal:.3f}" if result.nominal is not None else "-",
                f"{result.deviation:+.3f}" if result.deviation is not None else "-",
                f"{result.tolerance_abs:.4f}" if result.tolerance_abs is not None else "",
                f"{result.status} [{result.geometry_source}]",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if result.status == "ok":
                    item.setForeground(Qt.green if col == 5 else Qt.white)
                elif result.status == "ng":
                    item.setForeground(Qt.red)
                elif result.status == "no_measurement":
                    item.setForeground(Qt.yellow)
                else:
                    item.setForeground(Qt.red)
                self._table.setItem(row, col, item)

    @Slot()
    def _on_result_selection_changed(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            self.result_selected.emit(None)
            return
        row = rows[0].row()
        if 0 <= row < len(self._results):
            self.result_selected.emit(self._results[row])
        else:
            self.result_selected.emit(None)


# Backward-compatible name for older imports.
ProductionLogDialog = ProductionLogViewer
