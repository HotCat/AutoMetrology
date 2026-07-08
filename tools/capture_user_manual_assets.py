#!/usr/bin/env python3
"""Capture current UI/manual screenshots using the real application widgets."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
from PySide6.QtWidgets import QApplication

from cadviewer.core.i18n import LANG_ZH_CN, set_language
from cadviewer.ui.main_window import MainWindow


DXF_PATH = Path(
    "/home/hotcat/Downloads/cadrefs/cads/"
    "弘毅云佳-工位牌-（大号）无挂绳孔V1.1_窗口雕刻测量22222.dxf"
)
SCREEN_DIR = Path("docs/screenshots")
SUMMARY_PATH = Path("docs/user_manual_capture_summary.json")


def pump(app: QApplication, seconds: float = 0.2) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def save_widget(widget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixmap = widget.grab()
    if pixmap.isNull():
        raise RuntimeError(f"Could not capture widget: {path}")
    if not pixmap.save(str(path)):
        raise RuntimeError(f"Could not save screenshot: {path}")


def crop_image(src: Path, dst: Path, rect: tuple[int, int, int, int]) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read screenshot: {src}")
    x, y, w, h = rect
    crop = img[y : y + h, x : x + w]
    if crop.size == 0:
        raise RuntimeError(f"Empty crop for {src}: {rect}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), crop)


def save_frame_preview(frame, path: Path, max_width: int = 1400) -> None:
    if frame is None:
        return
    img = frame.copy()
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(
            img,
            (max_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def main() -> int:
    if not DXF_PATH.exists():
        raise RuntimeError(f"DXF not found: {DXF_PATH}")

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("CAD Inspection Tool")

    window = MainWindow()
    window.resize(1680, 980)
    window.show()
    pump(app, 0.5)

    # The manual is Chinese. Keep this in the temporary HOME used by the caller.
    set_language(LANG_ZH_CN)
    window.retranslate_ui()
    pump(app, 0.2)

    window._load_dxf(str(DXF_PATH))
    pump(app, 0.8)
    window._viewer.fit_all()
    pump(app, 0.2)
    save_widget(window, SCREEN_DIR / "01_main_window.png")
    save_widget(window._viewer, SCREEN_DIR / "03_cad_canvas.png")

    window._reg_dock.show()
    window._reg_dock.raise_()
    pump(app, 0.4)

    reg = window._reg_panel
    reg._refresh_cameras()
    pump(app, 0.2)
    reg._open_camera()
    pump(app, 1.2)
    save_widget(window, SCREEN_DIR / "04_registration_camera_open.png")

    captured = reg._capture_from_camera(wait_for_fresh_frame=True)
    pump(app, 0.5)
    image_layer = window._viewer.get_image_layer()
    save_frame_preview(image_layer.image, SCREEN_DIR / "05_live_capture_frame.png")
    save_widget(window, SCREEN_DIR / "06_frame_captured.png")

    registered = reg._run_window_line_registration()
    pump(app, 0.6)
    save_widget(window, SCREEN_DIR / "07_window_registered.png")

    evaluated_count = window._evaluate_current_queries()
    pump(app, 0.5)
    window._query_panel_action.setChecked(True)
    pump(app, 0.5)
    save_widget(window._query_window, SCREEN_DIR / "08_measurement_queries.png")
    save_widget(window, SCREEN_DIR / "09_measurement_overlay_canvas.png")
    crop_image(
        SCREEN_DIR / "08_measurement_queries.png",
        SCREEN_DIR / "08a_measurement_results_crop.png",
        (0, 260, 1260, 430),
    )

    if window._query_panel.results():
        window._query_panel._table.selectRow(0)
        pump(app, 0.2)
        save_widget(window, SCREEN_DIR / "10_selected_measurement_overlay.png")

    record_id = window._save_current_production_log()
    pump(app, 0.3)
    if record_id:
        window._production_log_viewer.refresh(select_record_id=record_id)
        window._query_panel.show_production_log_view()
        pump(app, 0.5)
        save_widget(window._query_window, SCREEN_DIR / "11_production_log_viewer.png")

    results = []
    for result in window._query_panel.results():
        inst = result.instruction
        results.append({
            "query": inst.raw_text if inst else "",
            "status": result.status,
            "value": result.value,
            "nominal": result.nominal,
            "deviation": result.deviation,
            "geometry_source": result.geometry_source,
        })

    summary = {
        "dxf": str(DXF_PATH),
        "captured": bool(captured),
        "registered": bool(registered),
        "evaluated_count": int(evaluated_count),
        "record_id": record_id,
        "pixel_size_mm": float(window._config.pixel_size_mm),
        "apply_correction_map": bool(getattr(window._config, "apply_correction_map", True)),
        "active_profile": getattr(window._config, "active_production_profile", ""),
        "window_edges": list(getattr(reg, "_window_edge_ids", [])),
        "window_detection_mode": getattr(reg, "_window_detection_mode", ""),
        "registration": reg.last_auto_registration_snapshot(),
        "results": results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        reg._close_camera()
    except Exception:
        pass
    window.close()
    pump(app, 0.1)
    return 0 if captured and registered and evaluated_count else 2


if __name__ == "__main__":
    raise SystemExit(main())
