#!/usr/bin/env python3
"""Record a real UI interaction demo for line band selection."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication

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
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FPS = 15
TARGET_TOKEN = "AC66:3"
VIDEO_WIDTH = 1800
VIDEO_HEIGHT = 1080
CANVAS_W = 1060
CANVAS_H = 900
QUERY_X = 1082
QUERY_Y = 78
QUERY_W = 690
QUERY_H = 805


def pump(app: QApplication, seconds: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def qpixmap_to_bgr(pixmap) -> np.ndarray:
    temp = OUT_DIR / "_screen_grab_tmp.png"
    pixmap.save(str(temp))
    frame = cv2.imread(str(temp), cv2.IMREAD_COLOR)
    try:
        temp.unlink()
    except OSError:
        pass
    if frame is None:
        raise RuntimeError("Could not convert Qt grab to image")
    return frame


def fit_into(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Letterbox a real widget grab into a fixed video region."""
    h, w = frame.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    rw = max(1, int(round(w * scale)))
    rh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
    out = np.full((height, width, 3), (24, 26, 26), dtype=np.uint8)
    ox = (width - rw) // 2
    oy = (height - rh) // 2
    out[oy:oy + rh, ox:ox + rw] = resized
    return out


def grab_scene(window: MainWindow) -> np.ndarray:
    """Grab the live CAD canvas and live query panel into a stable layout."""
    canvas = qpixmap_to_bgr(window._viewer.grab())
    query = qpixmap_to_bgr(window._query_window.grab())
    out = np.full((VIDEO_HEIGHT, VIDEO_WIDTH, 3), (18, 20, 20), dtype=np.uint8)
    out[58:58 + CANVAS_H, 20:20 + CANVAS_W] = fit_into(canvas, CANVAS_W, CANVAS_H)
    out[QUERY_Y:QUERY_Y + QUERY_H, QUERY_X:QUERY_X + QUERY_W] = fit_into(query, QUERY_W, QUERY_H)
    cv2.rectangle(out, (20, 58), (20 + CANVAS_W, 58 + CANVAS_H), (70, 70, 70), 1)
    cv2.rectangle(out, (QUERY_X, QUERY_Y), (QUERY_X + QUERY_W, QUERY_Y + QUERY_H), (70, 70, 70), 1)
    return out


def draw_caption(frame: np.ndarray, title: str, note: str = "") -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = ImageFont.truetype(str(FONT_PATH), 36)
    note_font = ImageFont.truetype(str(FONT_PATH), 26)
    w, h = image.size
    box_h = 118 if note else 82
    draw.rectangle((0, h - box_h, w, h), fill=(0, 0, 0, 178))
    draw.text((28, h - box_h + 16), title, font=title_font, fill=(255, 255, 255, 255))
    if note:
        draw.text((30, h - 42), note, font=note_font, fill=(255, 230, 80, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_callout(frame: np.ndarray, text: str, pos: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype(str(FONT_PATH), 30)
    x, y = pos
    draw.rounded_rectangle((x, y, x + 470, y + 62), radius=12, fill=(0, 70, 120, 220))
    draw.text((x + 18, y + 13), text, font=font, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def record_frames(
    app: QApplication,
    window: MainWindow,
    seconds: float,
    title: str,
    note: str = "",
    callout: str = "",
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    count = max(1, int(round(seconds * FPS)))
    for _ in range(count):
        pump(app, 1.0 / FPS)
        frame = grab_scene(window)
        if callout:
            frame = draw_callout(frame, callout, (28, 92))
        frames.append(draw_caption(frame, title, note))
    return frames


def find_line_id(window: MainWindow, token: str) -> str:
    fid = window._resolve_query_feature_id(token)
    if fid is None:
        raise RuntimeError(f"Cannot resolve {token}")
    return fid


def line_center(window: MainWindow, feature_id: str) -> tuple[float, float]:
    feature = window._repo.get(feature_id)
    if feature is None or feature.feature_type != FeatureType.LINE:
        raise RuntimeError(f"Not a line feature: {feature_id}")
    g = feature.geometry
    return (
        (float(g["x1"]) + float(g["x2"])) / 2.0,
        (float(g["y1"]) + float(g["y2"])) / 2.0,
    )


def set_view(
    window: MainWindow,
    center: tuple[float, float],
    scale: float,
    target_screen: tuple[float, float] = (470.0, 395.0),
) -> None:
    canvas = window._viewer
    cx, cy = center
    sx, sy = target_screen
    canvas._scale = scale
    canvas._offset_x = cx - (sx - canvas.width() / 2.0) / scale
    canvas._offset_y = cy + (sy - canvas.height() / 2.0) / scale
    canvas._cache_dirty = True
    canvas.update()


def animate_zoom(
    app: QApplication,
    window: MainWindow,
    center: tuple[float, float],
    start_scale: float,
    end_scale: float,
    seconds: float,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    count = max(1, int(round(seconds * FPS)))
    for idx in range(count):
        t = idx / max(1, count - 1)
        eased = 0.5 - 0.5 * math.cos(math.pi * t)
        scale = start_scale + (end_scale - start_scale) * eased
        set_view(window, center, scale)
        pump(app, 1.0 / FPS)
        frame = draw_caption(
            grab_scene(window),
            "先放大 CAD 画布，直到能清楚看到印刷线的两个灰度边",
            "红色为 CAD 名义线，绿色/青色为图像拟合线。",
        )
        frames.append(frame)
    return frames


def set_band(window: MainWindow, token: str, band: str):
    overrides = dict(window._query_panel.line_fit_side_overrides())
    overrides[token] = band
    window._query_panel.set_line_fit_side_overrides(overrides)
    table = window._query_panel._line_band_table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and item.text().strip() == token:
            table.selectRow(row)
            break
    count = window._evaluate_current_queries()
    if count <= 0:
        raise RuntimeError("No measurement queries evaluated")
    window._query_panel._table.selectRow(2)
    QApplication.processEvents()
    return window._query_panel.results()


def setup_app(app: QApplication) -> tuple[MainWindow, str, tuple[float, float]]:
    window = MainWindow()
    window.resize(1680, 1040)
    window.show()
    pump(app, 0.5)
    set_language(LANG_ZH_CN)
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
    window._query_window.resize(940, 760)
    pump(app, 0.4)
    target_id = find_line_id(window, TARGET_TOKEN)
    center = line_center(window, target_id)
    window._viewer.set_highlighted_features([target_id])
    return window, target_id, center


def write_video(frames: list[np.ndarray]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "_band_selection_screen_raw.mp4"
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(FPS),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open {raw_path}")
    try:
        for frame in frames:
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


def result_summary(results) -> list[dict]:
    rows = []
    for result in results:
        inst = result.instruction
        rows.append({
            "query": inst.raw_text if inst else "",
            "status": result.status,
            "value": result.value,
            "nominal": result.nominal,
            "deviation": result.deviation,
            "geometry_source": result.geometry_source,
        })
    return rows


def main() -> int:
    if not DXF_PATH.exists():
        raise RuntimeError(f"DXF not found: {DXF_PATH}")
    if not IMAGE_PATH.exists():
        raise RuntimeError(f"Image not found: {IMAGE_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AutoMetrology Band Screen Recording")
    window, target_id, center = setup_app(app)

    frames: list[np.ndarray] = []
    frames.extend(record_frames(
        app, window, 3.0,
        "打开 Measurement Queries，并选中 Line ID 表中的 AC66:3",
        "下面演示同一条 CAD 线选择不同印刷灰度边，拟合线会立即改变。",
        "目标线：AC66:3",
    ))
    frames.extend(animate_zoom(app, window, center, 5.0, 60.0, 5.0))

    plus_results = set_band(window, TARGET_TOKEN, "positive")
    set_view(window, center, 60.0)
    frames.extend(record_frames(
        app, window, 7.0,
        "选择 +N band：拟合 CAD 法向一侧的印刷边",
        "观察左侧放大区域，绿色/青色拟合线贴到当前选择的边；结果为 OK。",
        "+N band",
    ))

    minus_results = set_band(window, TARGET_TOKEN, "negative")
    set_view(window, center, 60.0)
    frames.extend(record_frames(
        app, window, 8.0,
        "切换为 -N band：拟合线跳到相反侧印刷边",
        "同一条 Line ID，选择另一侧灰度带后，测量值和状态立即变化。",
        "-N band",
    ))

    plus_again_results = set_band(window, TARGET_TOKEN, "positive")
    set_view(window, center, 60.0)
    frames.extend(record_frames(
        app, window, 5.0,
        "再切回 +N band：拟合线回到原来的印刷边",
        "结论：Line ID 表中的 +N/-N 用来明确指定要拟合哪一侧印刷边。",
        "+N band",
    ))

    write_video(frames)
    SUMMARY_PATH.write_text(
        json.dumps({
            "video": str(VIDEO_PATH),
            "dxf": str(DXF_PATH),
            "image": str(IMAGE_PATH),
            "target_line": TARGET_TOKEN,
            "target_feature_id": target_id,
            "recording_type": "animated Qt screen/widget recording",
            "results": {
                "positive": result_summary(plus_results),
                "negative": result_summary(minus_results),
                "positive_again": result_summary(plus_again_results),
            },
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    window.close()
    pump(app, 0.1)
    print(f"wrote {VIDEO_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
