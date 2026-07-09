#!/usr/bin/env python3
"""Record a real screen interaction demo for Measurement Queries band selection."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCursor, QImage, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox

from cadviewer.core.i18n import LANG_ZH_CN, set_language
from cadviewer.models.feature import FeatureType
from cadviewer.ui.main_window import MainWindow


DXF_PATH = Path(
    "/home/hotcat/Downloads/cadrefs/cads/"
    "弘毅云佳-工位牌-（大号）无挂绳孔V1.1_窗口雕刻测量22222.dxf"
)
IMAGE_PATH = Path("/tmp/cadrefs_camera_capture.png")
OUT_DIR = Path("docs/demos")
VIDEO_PATH = OUT_DIR / "band_selection_demo.mp4"
SUMMARY_PATH = OUT_DIR / "band_selection_demo_summary.json"
TARGET_TOKEN = "AC66:3"
FPS = 25
CAPTURE_W = 1880
CAPTURE_H = 1040
_RECORDER = None


def pump(app: QApplication, seconds: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if _RECORDER is not None:
            _RECORDER.maybe_capture()
        time.sleep(0.005)


def move_cursor(app: QApplication, global_pos, seconds: float = 0.45, steps: int = 18) -> None:
    start = QCursor.pos()
    end = QPoint(int(global_pos.x()), int(global_pos.y()))
    for idx in range(1, steps + 1):
        t = idx / steps
        eased = 0.5 - 0.5 * math.cos(math.pi * t)
        x = round(start.x() + (end.x() - start.x()) * eased)
        y = round(start.y() + (end.y() - start.y()) * eased)
        QCursor.setPos(x, y)
        pump(app, seconds / steps)


def click_widget(app: QApplication, widget, local_pos: QPoint, pause: float = 0.4) -> None:
    move_cursor(app, widget.mapToGlobal(local_pos), 0.35)
    QTest.mouseClick(widget, Qt.LeftButton, Qt.NoModifier, local_pos)
    pump(app, pause)


def qpixmap_to_bgr(pixmap: QPixmap) -> np.ndarray:
    image = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


class ManualFrameRecorder:
    """Continuously records the real Qt windows while UI actions run."""

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.frames: list[np.ndarray] = []
        self.started_at = time.monotonic()
        self._last_frame_at = 0.0
        self._period = 1.0 / FPS

    def maybe_capture(self) -> None:
        now = time.monotonic()
        if now - self._last_frame_at < self._period:
            return
        self._last_frame_at = now
        self.frames.append(self._capture_frame())

    def _capture_frame(self) -> np.ndarray:
        frame = np.zeros((CAPTURE_H, CAPTURE_W, 3), dtype=np.uint8)
        self._overlay_widget(frame, self.window)
        self._overlay_widget(frame, self.window._query_window)
        combo = self._current_band_combo()
        if combo is not None:
            popup = combo.view().window()
            if combo.view().isVisible():
                self._overlay_widget(frame, combo.view())
            elif popup is not None and popup.isVisible():
                self._overlay_widget(frame, popup)
        self._draw_cursor(frame)
        return frame

    def _current_band_combo(self):
        table = self.window._query_panel._line_band_table
        row = table.currentRow()
        if row < 0:
            row = 0
        combo = table.cellWidget(row, 1)
        return combo if isinstance(combo, QComboBox) else None

    def _overlay_widget(self, frame: np.ndarray, widget) -> None:
        if widget is None or not widget.isVisible():
            return
        pixmap = widget.grab()
        image = qpixmap_to_bgr(pixmap)
        top_left = widget.mapToGlobal(QPoint(0, 0))
        x = max(0, top_left.x())
        y = max(0, top_left.y())
        if x >= CAPTURE_W or y >= CAPTURE_H:
            return
        h, w = image.shape[:2]
        x2 = min(CAPTURE_W, x + w)
        y2 = min(CAPTURE_H, y + h)
        frame[y:y2, x:x2] = image[: y2 - y, : x2 - x]

    def _draw_cursor(self, frame: np.ndarray) -> None:
        pos = QCursor.pos()
        x, y = int(pos.x()), int(pos.y())
        if x < 0 or y < 0 or x >= CAPTURE_W or y >= CAPTURE_H:
            return
        pts = np.array([[x, y], [x + 19, y + 8], [x + 8, y + 12], [x + 13, y + 25]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], (245, 245, 245))
        cv2.polylines(frame, [pts], True, (20, 20, 20), 1, cv2.LINE_AA)

    def write(self) -> None:
        if not self.frames:
            raise RuntimeError("No manual interaction frames captured")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = OUT_DIR / "_band_selection_manual_screen_raw.mp4"
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(FPS),
            (CAPTURE_W, CAPTURE_H),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open {raw_path}")
        try:
            for frame in self.frames:
                writer.write(frame)
        finally:
            writer.release()
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(raw_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(VIDEO_PATH),
            ],
            check=True,
        )
        try:
            raw_path.unlink()
        except OSError:
            pass


def find_line_id(window: MainWindow, token: str) -> str:
    feature_id = window._resolve_query_feature_id(token)
    if feature_id is None:
        raise RuntimeError(f"Cannot resolve line token {token}")
    return feature_id


def line_center(window: MainWindow, feature_id: str) -> tuple[float, float]:
    feature = window._repo.get(feature_id)
    if feature is None or feature.feature_type != FeatureType.LINE:
        raise RuntimeError(f"Feature is not a line: {feature_id}")
    geometry = feature.geometry
    return (
        (float(geometry["x1"]) + float(geometry["x2"])) / 2.0,
        (float(geometry["y1"]) + float(geometry["y2"])) / 2.0,
    )


def set_view(window: MainWindow, center: tuple[float, float], scale: float) -> None:
    canvas = window._viewer
    sx, sy = 450.0, 400.0
    cx, cy = center
    canvas._scale = scale
    canvas._offset_x = cx - (sx - canvas.width() / 2.0) / scale
    canvas._offset_y = cy + (sy - canvas.height() / 2.0) / scale
    canvas._cache_dirty = True
    canvas.update()


def animate_zoom(app: QApplication, window: MainWindow, center: tuple[float, float]) -> None:
    canvas = window._viewer
    move_cursor(app, canvas.mapToGlobal(QPoint(450, 400)), 0.7)
    for idx in range(80):
        t = idx / 79
        eased = 0.5 - 0.5 * math.cos(math.pi * t)
        set_view(window, center, 5.0 + (60.0 - 5.0) * eased)
        pump(app, 0.045)
    pump(app, 0.8)


def setup_app(app: QApplication) -> tuple[MainWindow, str, tuple[float, float]]:
    if not DXF_PATH.exists():
        raise RuntimeError(f"DXF not found: {DXF_PATH}")
    if not IMAGE_PATH.exists():
        raise RuntimeError(f"Image not found: {IMAGE_PATH}")

    window = MainWindow()
    set_language(LANG_ZH_CN)
    window._config.language = LANG_ZH_CN
    window.resize(1060, 960)
    window.move(0, 35)
    window.show()
    pump(app, 0.8)
    window.retranslate_ui()
    pump(app, 0.2)

    window._load_dxf(str(DXF_PATH))
    pump(app, 0.8)
    if not window._viewer.get_image_layer().load_image(str(IMAGE_PATH)):
        raise RuntimeError(f"Could not load image: {IMAGE_PATH}")
    window._viewer.get_image_layer().set_pixel_size_mm(float(window._config.pixel_size_mm))
    window._reg_panel._image_calibration_applied = True
    window._reg_panel._auto_source_image_path = str(IMAGE_PATH)
    window._reg_panel._image_path_label.setText(str(IMAGE_PATH))
    if not window._reg_panel._run_window_line_registration():
        raise RuntimeError("Window registration failed")

    window._query_panel_action.setChecked(True)
    query_window = window._query_window
    query_window.resize(790, 760)
    query_window.move(1070, 145)
    query_window.show()
    query_window.raise_()
    window.raise_()
    pump(app, 0.5)

    target_id = find_line_id(window, TARGET_TOKEN)
    center = line_center(window, target_id)
    window._viewer.set_highlighted_features([target_id])
    return window, target_id, center


def ensure_target_row(window: MainWindow) -> tuple[int, QComboBox]:
    panel = window._query_panel
    panel.add_line_band_override(TARGET_TOKEN, "positive")
    table = panel._line_band_table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and item.text().strip() == TARGET_TOKEN:
            combo = table.cellWidget(row, 1)
            if isinstance(combo, QComboBox):
                table.clearSelection()
                return row, combo
    raise RuntimeError(f"Could not find line-band row for {TARGET_TOKEN}")


def select_line_row(app: QApplication, window: MainWindow, row: int) -> None:
    table = window._query_panel._line_band_table
    table.scrollToItem(table.item(row, 0))
    pump(app, 0.2)
    rect = table.visualItemRect(table.item(row, 0))
    click_widget(app, table.viewport(), rect.center(), 0.8)


def choose_combo_index(app: QApplication, combo: QComboBox, index: int) -> None:
    click_widget(app, combo, combo.rect().center(), 1.2)
    combo.showPopup()
    pump(app, 0.8)
    view = combo.view()
    viewport = view.viewport()
    model_index = combo.model().index(index, 0)
    rect = view.visualRect(model_index)
    if not rect.isValid():
        rect = viewport.rect()
    move_cursor(app, viewport.mapToGlobal(rect.center()), 0.8)
    QTest.mouseClick(viewport, Qt.LeftButton, Qt.NoModifier, rect.center())
    pump(app, 0.9)


def click_evaluate(app: QApplication, window: MainWindow) -> None:
    button = window._query_panel._btn_evaluate
    click_widget(app, button, button.rect().center(), 1.1)
    window._query_panel._table.selectRow(2)
    pump(app, 0.7)


def result_summary(results) -> list[dict]:
    rows = []
    for result in results:
        instruction = result.instruction
        rows.append({
            "query": instruction.raw_text if instruction else "",
            "status": result.status,
            "value": result.value,
            "nominal": result.nominal,
            "deviation": result.deviation,
            "geometry_source": result.geometry_source,
        })
    return rows


def main() -> int:
    global _RECORDER
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AutoMetrology Manual Band Screen Recording")
    window, target_id, center = setup_app(app)
    row, combo = ensure_target_row(window)
    pump(app, 0.5)

    recorder = ManualFrameRecorder(window)
    _RECORDER = recorder
    positive_results = []
    negative_results = []
    try:
        pump(app, 1.2)
        animate_zoom(app, window, center)
        pump(app, 1.0)
        select_line_row(app, window, row)
        pump(app, 0.8)
        choose_combo_index(app, combo, 0)
        pump(app, 0.8)
        click_evaluate(app, window)
        positive_results = window._query_panel.results()
        pump(app, 2.6)
        choose_combo_index(app, combo, 1)
        pump(app, 0.8)
        click_evaluate(app, window)
        negative_results = window._query_panel.results()
        pump(app, 4.0)
    finally:
        _RECORDER = None
    recorder.write()

    SUMMARY_PATH.write_text(
        json.dumps({
            "video": str(VIDEO_PATH),
            "dxf": str(DXF_PATH),
            "image": str(IMAGE_PATH),
            "target_line": TARGET_TOKEN,
            "target_feature_id": target_id,
            "recording_type": "continuous Qt window recording with visible cursor, combo popup, and real UI clicks",
            "actions": [
                "zoom in the printed line",
                "select AC66:3 in Line ID table",
                "select +N band and click Evaluate",
                "select -N band and click Evaluate",
            ],
            "results": {
                "positive": result_summary(positive_results),
                "negative": result_summary(negative_results),
            },
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    window.close()
    pump(app, 0.2)
    print(f"wrote {VIDEO_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
