"""Tests for production log viewer selection behavior."""

from __future__ import annotations

import os
import unittest
from datetime import date

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from cadviewer.ui.production_log_dialog import ProductionLogViewer


class _FakeProductionLogStore:
    def __init__(self) -> None:
        today = date.today().isoformat()
        self.records = [
            {
                "id": "old-ng",
                "created_at": f"{today}T08:00:00",
                "created_date": today,
                "overall_status": "ng",
                "cad_filename": "xintai.dxf",
                "query_count": 1,
                "image_path": "",
                "ok_count": 0,
                "ng_count": 1,
                "no_measurement_count": 0,
                "error_count": 0,
            },
            {
                "id": "new-ok",
                "created_at": f"{today}T09:00:00",
                "created_date": today,
                "overall_status": "ok",
                "cad_filename": "hongyi.dxf",
                "query_count": 1,
                "image_path": "",
                "ok_count": 1,
                "ng_count": 0,
                "no_measurement_count": 0,
                "error_count": 0,
            },
        ]

    def month_day_counts(self, year: int, month: int) -> dict[str, dict[str, int]]:
        today = date.today()
        if year != today.year or month != today.month:
            return {}
        return {today.isoformat(): {"total": 2, "ok": 1, "ng": 1}}

    def records_for_day(self, date_text: str) -> list[dict]:
        return [r for r in self.records if r["created_date"] == date_text]

    def get_record(self, record_id: str) -> dict | None:
        for record in self.records:
            if record["id"] == record_id:
                return record
        return None

    def get_results(self, record_id: str) -> list:
        return []


class ProductionLogViewerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_refresh_can_select_newly_created_record(self) -> None:
        viewer = ProductionLogViewer(_FakeProductionLogStore())

        viewer.refresh(select_record_id="new-ok")

        self.assertEqual(viewer.current_record_id(), "new-ok")


if __name__ == "__main__":
    unittest.main()
