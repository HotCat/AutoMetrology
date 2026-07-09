#!/usr/bin/env python3
"""Create a short Chinese demo video for per-line band selection."""

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


def pump(app: QApplication, seconds: float = 0.2) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def pixmap_to_bgr(pixmap) -> np.ndarray:
    temp = OUT_DIR / "_grab_tmp.png"
    pixmap.save(str(temp))
    image = cv2.imread(str(temp), cv2.IMREAD_COLOR)
    try:
        temp.unlink()
    except OSError:
        pass
    if image is None:
        raise RuntimeError(f"Could not read grabbed pixmap: {temp}")
    return image


def widget_bgr(widget) -> np.ndarray:
    return pixmap_to_bgr(widget.grab())


def compose_scene(window: MainWindow, subtitle: str, note: str = "") -> np.ndarray:
    main = widget_bgr(window)
    query = widget_bgr(window._query_window)
    canvas_h, canvas_w = main.shape[:2]
    out = main.copy()
    x, y = 405, 230
    h, w = query.shape[:2]
    x2 = min(canvas_w, x + w)
    y2 = min(canvas_h, y + h)
    out[y:y2, x:x2] = query[: y2 - y, : x2 - x]
    return add_subtitles(out, subtitle, note)


def add_subtitles(frame: np.ndarray, title: str, note: str = "") -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype(str(FONT_PATH), 36)
    small = ImageFont.truetype(str(FONT_PATH), 25)
    w, h = image.size
    box_h = 118 if note else 82
    draw.rectangle((0, h - box_h, w, h), fill=(0, 0, 0, 175))
    draw.text((32, h - box_h + 18), title, font=font, fill=(255, 255, 255, 255))
    if note:
        draw.text((34, h - 42), note, font=small, fill=(255, 230, 120, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def add_pointer(frame: np.ndarray, text: str, xy: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype(str(FONT_PATH), 28)
    x, y = xy
    draw.rounded_rectangle((x, y, x + 450, y + 58), radius=12, fill=(0, 70, 120, 210))
    draw.text((x + 18, y + 12), text, font=font, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def find_line_id(window: MainWindow, token: str) -> str:
    fid = window._resolve_query_feature_id(token)
    if fid is None:
        raise RuntimeError(f"Cannot resolve {token}")
    return fid


def zoom_to_line(window: MainWindow, feature_id: str) -> None:
    feature = window._repo.get(feature_id)
    if feature is None or feature.feature_type != FeatureType.LINE:
        raise RuntimeError(f"Not a line feature: {feature_id}")
    g = feature.geometry
    cx = (float(g["x1"]) + float(g["x2"])) / 2.0
    cy = (float(g["y1"]) + float(g["y2"])) / 2.0
    canvas = window._viewer
    canvas._scale = 13.5
    # Keep the selected vertical print line visible to the left of the floating
    # Measurement Queries window, matching the demonstration reference layout.
    canvas._offset_x = cx + 50.0
    canvas._offset_y = cy + 8.0
    canvas._cache_dirty = True
    canvas.set_highlighted_features([feature_id])
    canvas.update()


def set_band(window: MainWindow, target_token: str, band: str) -> list:
    overrides = dict(window._query_panel.line_fit_side_overrides())
    overrides[target_token] = band
    window._query_panel.set_line_fit_side_overrides(overrides)
    for row in range(window._query_panel._line_band_table.rowCount()):
        item = window._query_panel._line_band_table.item(row, 0)
        if item is not None and item.text().strip() == target_token:
            window._query_panel._line_band_table.selectRow(row)
            break
    count = window._evaluate_current_queries()
    if count <= 0:
        raise RuntimeError("No measurement queries evaluated")
    window._query_panel._table.selectRow(2)
    QApplication.processEvents()
    return window._query_panel.results()


def setup_app(app: QApplication) -> MainWindow:
    window = MainWindow()
    window.resize(1800, 1220)
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
    pump(app, 0.2)
    if not window._reg_panel._run_window_line_registration():
        raise RuntimeError("Window registration failed")
    pump(app, 0.4)
    window._query_panel_action.setChecked(True)
    window._query_window.resize(1280, 760)
    pump(app, 0.3)
    return window


def write_video(frames: list[tuple[np.ndarray, float]], fps: int = 15) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    first = frames[0][0]
    height, width = first.shape[:2]
    temp_path = OUT_DIR / "_band_selection_demo_raw.mp4"
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temp_path}")
    try:
        for frame, seconds in frames:
            repeat = max(1, int(math.ceil(seconds * fps)))
            for _ in range(repeat):
                writer.write(frame)
    finally:
        writer.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(temp_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(VIDEO_PATH),
        ],
        check=True,
    )
    try:
        temp_path.unlink()
    except OSError:
        pass


def main() -> int:
    if not DXF_PATH.exists():
        raise RuntimeError(f"DXF not found: {DXF_PATH}")
    if not IMAGE_PATH.exists():
        raise RuntimeError(f"Image not found: {IMAGE_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("AutoMetrology Band Demo")
    window = setup_app(app)
    target = "AC66:3"
    target_id = find_line_id(window, target)
    zoom_to_line(window, target_id)

    frames: list[tuple[np.ndarray, float]] = []
    results_by_band: dict[str, dict] = {}

    plus_results = set_band(window, target, "positive")
    pump(app, 0.4)
    plus = compose_scene(
        window,
        "步骤 1：选中 Line ID = AC66:3，设置为 +N band",
        "+N 会沿 CAD 线法向的一侧寻找印刷灰度带；绿色拟合线贴到这一侧边缘。",
    )
    plus = add_pointer(plus, "+N band：拟合这一侧印刷边", (34, 88))
    cv2.imwrite(str(OUT_DIR / "band_plus_state.png"), plus)
    frames.append((plus, 8.0))
    results_by_band["positive"] = result_summary(plus_results)

    minus_results = set_band(window, target, "negative")
    pump(app, 0.4)
    minus = compose_scene(
        window,
        "步骤 2：把同一条 Line ID 切换为 -N band",
        "-N 会切换到 CAD 线法向的另一侧；观察绿色拟合线跳到另一条印刷边。",
    )
    minus = add_pointer(minus, "-N band：拟合相反侧印刷边", (34, 88))
    cv2.imwrite(str(OUT_DIR / "band_minus_state.png"), minus)
    frames.append((minus, 9.0))
    results_by_band["negative"] = result_summary(minus_results)

    plus_results2 = set_band(window, target, "positive")
    pump(app, 0.4)
    compare = compose_scene(
        window,
        "步骤 3：再次切回 +N，确认拟合线随灰度带选择稳定切换",
        "测量值和偏差会跟随被选中的印刷边变化；CAD 线只限定搜索区域，不决定最终边。",
    )
    cv2.imwrite(str(OUT_DIR / "band_compare_state.png"), compare)
    frames.append((compare, 8.0))
    results_by_band["positive_again"] = result_summary(plus_results2)

    ending = add_subtitles(
        compare,
        "结论：+N / -N 是逐线指定的印刷边选择",
        "当一条印刷线有两个灰度边时，用 Line ID 表精确指定要拟合哪一侧。",
    )
    frames.append((ending, 5.0))

    write_video(frames, fps=15)
    SUMMARY_PATH.write_text(
        json.dumps({
            "video": str(VIDEO_PATH),
            "dxf": str(DXF_PATH),
            "image": str(IMAGE_PATH),
            "target_line": target,
            "target_feature_id": target_id,
            "results": results_by_band,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    window.close()
    pump(app, 0.1)
    print(f"wrote {VIDEO_PATH}")
    print(f"wrote {SUMMARY_PATH}")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
