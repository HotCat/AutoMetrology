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
from PIL import Image, ImageDraw, ImageFont
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
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
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
        self._subtitle_title = ""
        self._subtitle_detail = ""
        self._title_font = ImageFont.truetype(str(FONT_PATH), 34)
        self._detail_font = ImageFont.truetype(str(FONT_PATH), 25)

    def set_subtitle(self, title: str, detail: str = "") -> None:
        self._subtitle_title = title
        self._subtitle_detail = detail

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
        self._draw_subtitle(frame)
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

    def _draw_subtitle(self, frame: np.ndarray) -> None:
        if not self._subtitle_title and not self._subtitle_detail:
            return
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        box_h = 136 if self._subtitle_detail else 88
        y0 = CAPTURE_H - box_h
        draw.rectangle((0, y0, CAPTURE_W, CAPTURE_H), fill=(0, 0, 0, 190))
        if self._subtitle_title:
            draw.text((34, y0 + 16), self._subtitle_title, font=self._title_font, fill=(255, 255, 255, 255))
        if self._subtitle_detail:
            draw.text((36, y0 + 72), self._subtitle_detail, font=self._detail_font, fill=(255, 232, 96, 255))
        composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        frame[:] = cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)

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
    for idx in range(140):
        t = idx / 139
        eased = 0.5 - 0.5 * math.cos(math.pi * t)
        set_view(window, center, 5.0 + (60.0 - 5.0) * eased)
        pump(app, 0.05)
    pump(app, 1.0)


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
    initial_results = []
    positive_results = []
    negative_results = []
    try:
        recorder.set_subtitle(
            "演示目标：让操作者看清 +N / -N 灰度带如何改变线拟合",
            "左侧是已配准的 CAD+图像视图；右侧是“测量查询”和每条 Line ID 的灰度带选择。",
        )
        pump(app, 3.0)

        recorder.set_subtitle(
            "第一步：放大印刷线区域",
            "先把画布放大到 6000%，这样能看清青色边缘采样点和绿色拟合线的位置。",
        )
        animate_zoom(app, window, center)

        recorder.set_subtitle(
            "第二步：先计算一次，停留观察图像拟合结果",
            "此时还没有操作 Line ID 表；请注意左侧青色点是参与拟合的边缘点，绿色线是当前拟合出来的印刷边。",
        )
        click_evaluate(app, window)
        initial_results = window._query_panel.results()
        pump(app, 9.0)

        recorder.set_subtitle(
            "第三步：选择要控制灰度带方向的 CAD 线",
            "在 Line ID 表中点击 AC66:3；这一步只是指定哪一条 CAD 线使用手动 +N/-N 灰度带。",
        )
        select_line_row(app, window, row)
        pump(app, 3.0)

        recorder.set_subtitle(
            "第四步：选择 +N 灰度带并点击“计算”",
            "+N 表示沿 CAD 线法向的一侧寻找印刷边；计算后看左侧绿色拟合线以及右侧第 3 行结果。",
        )
        choose_combo_index(app, combo, 0)
        pump(app, 1.5)
        click_evaluate(app, window)
        positive_results = window._query_panel.results()
        recorder.set_subtitle(
            "+N 灰度带结果：拟合线贴在当前选择的一侧",
            "第 3 行 lines(AC66:3, AB8E:7) 为 OK，绿色拟合线靠近 CAD 名义线所在的这一侧。",
        )
        pump(app, 7.0)

        recorder.set_subtitle(
            "第五步：切换为 -N 灰度带",
            "-N 表示选择相反一侧的灰度带；切换后再次点击“计算”，拟合线会跳到另一侧。",
        )
        choose_combo_index(app, combo, 1)
        pump(app, 1.5)
        click_evaluate(app, window)
        negative_results = window._query_panel.results()
        recorder.set_subtitle(
            "-N 灰度带结果：绿色拟合线跳到相反侧",
            "同一条 CAD 线 AC66:3，只改变灰度带方向，测量值和状态立即变化；这就是手动消除印刷双边歧义的作用。",
        )
        pump(app, 8.0)
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
            "recording_type": "continuous Qt window recording with visible cursor, Chinese subtitles, combo value changes, Evaluate clicks, and real UI actions",
            "actions": [
                "zoom in the printed line",
                "evaluate once and hold on the fitted green line before Line ID selection",
                "select AC66:3 in Line ID table",
                "select +N band and click Evaluate",
                "select -N band and click Evaluate",
            ],
            "results": {
                "initial": result_summary(initial_results),
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
