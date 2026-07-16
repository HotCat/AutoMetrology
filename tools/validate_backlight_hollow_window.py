#!/usr/bin/env python3
"""Validate one backlight hollow-window image using chessboard pixel scale only.

This tool intentionally does not use CAD dimensions, window-registration affine
scale, or any CAD nominal to compute measurement scale.  It measures the
backlight window directly in image pixels and converts with AppConfig.pixel_size_mm.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from cadviewer.core.config import AppConfig
from cadviewer.registration.auto_correspondence import undistort_if_calibrated

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"OpenCV is required: {exc}") from exc

try:
    import diplib as dip
    HAS_DIPLIB = True
except ImportError:  # pragma: no cover
    HAS_DIPLIB = False


@dataclass
class WindowFit:
    bbox: tuple[int, int, int, int]
    threshold: int
    confidence: float
    side_lines: dict[str, tuple[float, float, float]]
    corners: np.ndarray
    distances_px: dict[str, float]
    distances_mm: dict[str, float]
    errors_mm: dict[str, float]
    gradient_method: str
    fit_mode: str
    edge_bias: str
    light_fraction: float
    undistorted: bool
    pixel_size_mm: float


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _bright_threshold_candidates(gray: np.ndarray) -> list[int]:
    values = [170, 180, 190, 200, 210, 220, 230, 240, 245]
    try:
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        values.extend([
            int(round(float(otsu) + 10.0)),
            int(round(float(otsu) + 25.0)),
            int(round(float(otsu) + 40.0)),
        ])
    except Exception:
        pass
    percentiles = np.percentile(gray, [60, 70, 75, 80, 85, 90])
    values.extend(int(round(float(v))) for v in percentiles)
    return sorted({int(np.clip(v, 150, 250)) for v in values})


def _select_bright_window_component(
    gray: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], int, float]:
    h, w = gray.shape[:2]
    best = None
    best_score = float("-inf")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for threshold in _bright_threshold_candidates(gray):
        mask = (gray > threshold).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for idx in range(1, n):
            x, y, bw, bh, area = [float(v) for v in stats[idx]]
            if area < 80000 or bw < w * 0.25 or bh < h * 0.20:
                continue
            if bw > w * 0.90 or bh > h * 0.85:
                continue
            aspect = bw / max(bh, 1.0)
            if not 0.8 <= aspect <= 2.6:
                continue
            cx, cy = centroids[idx]
            center_penalty = abs(cx - w / 2.0) + abs(cy - h / 2.0)
            border_penalty = 0.0
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                border_penalty = area * 0.6
            score = area - center_penalty * 80.0 - border_penalty
            if score > best_score:
                best_score = score
                best = (labels == idx, (int(x), int(y), int(x + bw - 1), int(y + bh - 1)), threshold)
    if best is None:
        raise RuntimeError("No suitable bright hollow-window region detected")
    comp, bbox, threshold = best
    return comp, bbox, threshold, float(best_score)


def _scan_component_sides(
    comp: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> dict[str, float]:
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    top_vals: list[float] = []
    bottom_vals: list[float] = []
    for x in range(xmin + int(width * 0.12), xmax - int(width * 0.12) + 1):
        ys = np.flatnonzero(comp[:, x])
        if len(ys):
            top_vals.append(float(ys.min()))
            bottom_vals.append(float(ys.max()))

    left_vals: list[float] = []
    right_vals: list[float] = []
    for y in range(ymin + int(height * 0.12), ymax - int(height * 0.12) + 1):
        xs = np.flatnonzero(comp[y, :])
        if len(xs):
            left_vals.append(float(xs.min()))
            right_vals.append(float(xs.max()))

    if not top_vals or not bottom_vals or not left_vals or not right_vals:
        raise RuntimeError("Could not scan bright window component sides")

    return {
        "left": float(np.percentile(left_vals, 10)),
        "right": float(np.percentile(right_vals, 90)),
        "top": float(np.percentile(top_vals, 10)),
        "bottom": float(np.percentile(bottom_vals, 90)),
    }


def _gradient_magnitude(gray: np.ndarray, prefer_diplib: bool) -> tuple[np.ndarray, str]:
    if prefer_diplib and HAS_DIPLIB:
        try:
            dip_img = dip.Image(np.ascontiguousarray(gray))
            smoothed = dip.Gauss(dip_img, 0.4)
            grad = np.asarray(dip.Norm(dip.GradientMagnitude(smoothed)), dtype=np.float64)
            return grad, "diplib_gradient_magnitude"
        except Exception:
            pass
    grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
    return np.sqrt(grad_x ** 2 + grad_y ** 2), "opencv_scharr_gradient_magnitude"


def _profile_peaks(profile: np.ndarray) -> tuple[list[int], float]:
    if profile.size < 3:
        return [], 0.0
    threshold = max(45.0, float(np.mean(profile)) * 2.0)
    peaks: list[int] = []
    for idx in range(1, profile.size - 1):
        if (
            profile[idx] >= threshold
            and profile[idx] >= profile[idx - 1]
                and profile[idx] >= profile[idx + 1]
        ):
            peaks.append(idx)
    return peaks, threshold


def _select_profile_peak(
    profile: np.ndarray,
    edge_bias: str,
    inner_sign: int,
) -> Optional[int]:
    peaks, threshold = _profile_peaks(profile)
    if not peaks:
        idx = int(np.argmax(profile))
        if float(profile[idx]) < threshold:
            return None
        peaks = [idx]
    center = (profile.size - 1) / 2.0
    mode = str(edge_bias or "inner").strip().lower()
    if mode in {"inner", "outer"}:
        sign = inner_sign if mode == "inner" else -inner_sign
        biased = [idx for idx in peaks if (idx - center) * sign >= 0.0]
        if biased:
            peaks = biased
    return max(peaks, key=lambda idx: (float(profile[idx]), -abs(idx - center)))


def _light_inner_crossing(
    profile: np.ndarray,
    inner_sign: int,
    fraction: float,
) -> Optional[float]:
    """Return subpixel index of the bright-side threshold crossing.

    The profile is sampled across a bright backlight edge.  `inner_sign`
    indicates the bright-window interior direction in profile coordinates.
    A fraction near 1.0 biases toward the bright plateau, i.e. the inner side
    of the grayscale transition band.
    """
    if profile.size < 7:
        return None
    fraction = float(np.clip(fraction, 0.50, 0.98))
    values = profile.astype(np.float64)
    if inner_sign > 0:
        ordered = values
        base_index = 0
        step = 1
    else:
        ordered = values[::-1]
        base_index = values.size - 1
        step = -1

    edge_count = max(3, min(12, ordered.size // 5))
    outer_level = float(np.median(ordered[:edge_count]))
    inner_level = float(np.median(ordered[-edge_count:]))
    contrast = inner_level - outer_level
    if contrast < 20.0:
        return None
    target = outer_level + contrast * fraction

    crossings: list[float] = []
    for idx in range(ordered.size - 1):
        a = float(ordered[idx])
        b = float(ordered[idx + 1])
        if (a <= target <= b) or (b <= target <= a):
            denom = b - a
            if abs(denom) <= 1e-12:
                local = float(idx)
            else:
                local = float(idx) + (target - a) / denom
            crossings.append(float(base_index + step * local))
    if not crossings:
        return None
    center = (values.size - 1) / 2.0
    inner_crossings = [
        idx for idx in crossings if (idx - center) * inner_sign >= 0.0
    ]
    candidates = inner_crossings or crossings
    return min(candidates, key=lambda idx: abs(idx - center))


def _fit_line(points: np.ndarray) -> tuple[float, float, float]:
    if points.ndim != 2 or points.shape[0] < 20 or points.shape[1] != 2:
        raise RuntimeError("Not enough points for window side line fitting")
    line = cv2.fitLine(points.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01)
    vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]
    norm = float(np.hypot(vx, vy))
    if norm <= 1e-12:
        raise RuntimeError("Degenerate window side line")
    vx /= norm
    vy /= norm
    a = -vy
    b = vx
    c = -(a * x0 + b * y0)
    line_norm = float(np.hypot(a, b))
    if line_norm <= 1e-12:
        raise RuntimeError("Degenerate window side equation")
    return (a / line_norm, b / line_norm, c / line_norm)


def _refine_side_lines(
    gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    side_positions: dict[str, float],
    prefer_diplib: bool,
    fit_mode: str,
    edge_bias: str,
    light_fraction: float,
) -> tuple[dict[str, tuple[float, float, float]], str]:
    gradient, gradient_method = _gradient_magnitude(gray, prefer_diplib)
    h, w = gray.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    lines: dict[str, tuple[float, float, float]] = {}
    use_light_inner = str(fit_mode or "light-inner").lower() == "light-inner"

    for name in ("left", "right"):
        x0 = float(side_positions[name])
        pts: list[tuple[float, float]] = []
        y_values = np.linspace(
            ymin + height * 0.12,
            ymax - height * 0.12,
            max(80, int(height * 0.25)),
        )
        for y in y_values:
            yy = int(round(y))
            lo = max(0, int(round(x0 - 35.0)))
            hi = min(w - 1, int(round(x0 + 35.0)))
            if hi - lo < 6 or yy < 0 or yy >= h:
                continue
            inner_sign = 1 if name == "left" else -1
            if use_light_inner:
                peak_pos = _light_inner_crossing(
                    gray[yy, lo:hi + 1],
                    inner_sign=inner_sign,
                    fraction=light_fraction,
                )
            else:
                peak = _select_profile_peak(
                    gradient[yy, lo:hi + 1],
                    edge_bias=edge_bias,
                    inner_sign=inner_sign,
                )
                peak_pos = None if peak is None else float(peak)
            if peak_pos is not None:
                pts.append((float(lo + peak_pos), float(y)))
        lines[name] = _fit_line(np.asarray(pts, dtype=np.float64))

    for name in ("top", "bottom"):
        y0 = float(side_positions[name])
        pts = []
        x_values = np.linspace(
            xmin + width * 0.12,
            xmax - width * 0.12,
            max(80, int(width * 0.25)),
        )
        for x in x_values:
            xx = int(round(x))
            lo = max(0, int(round(y0 - 35.0)))
            hi = min(h - 1, int(round(y0 + 35.0)))
            if hi - lo < 6 or xx < 0 or xx >= w:
                continue
            inner_sign = 1 if name == "top" else -1
            if use_light_inner:
                peak_pos = _light_inner_crossing(
                    gray[lo:hi + 1, xx],
                    inner_sign=inner_sign,
                    fraction=light_fraction,
                )
            else:
                peak = _select_profile_peak(
                    gradient[lo:hi + 1, xx],
                    edge_bias=edge_bias,
                    inner_sign=inner_sign,
                )
                peak_pos = None if peak is None else float(peak)
            if peak_pos is not None:
                pts.append((float(x), float(lo + peak_pos)))
        lines[name] = _fit_line(np.asarray(pts, dtype=np.float64))

    return lines, gradient_method


def _intersect_lines(
    line_a: tuple[float, float, float],
    line_b: tuple[float, float, float],
) -> np.ndarray:
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-9:
        raise RuntimeError("Fitted window side lines are parallel")
    return np.array([
        (b1 * c2 - b2 * c1) / det,
        (c1 * a2 - c2 * a1) / det,
    ], dtype=np.float64)


def _side_line_corners(
    lines: dict[str, tuple[float, float, float]],
) -> np.ndarray:
    return np.array([
        _intersect_lines(lines["left"], lines["top"]),
        _intersect_lines(lines["right"], lines["top"]),
        _intersect_lines(lines["right"], lines["bottom"]),
        _intersect_lines(lines["left"], lines["bottom"]),
    ], dtype=np.float64)


def _measure_distances(corners: np.ndarray) -> dict[str, float]:
    top_left, top_right, bottom_right, bottom_left = corners
    top_width = float(np.linalg.norm(top_right - top_left))
    bottom_width = float(np.linalg.norm(bottom_right - bottom_left))
    left_height = float(np.linalg.norm(bottom_left - top_left))
    right_height = float(np.linalg.norm(bottom_right - top_right))
    return {
        "left_right_px": (top_width + bottom_width) * 0.5,
        "top_bottom_px": (left_height + right_height) * 0.5,
        "top_width_px": top_width,
        "bottom_width_px": bottom_width,
        "left_height_px": left_height,
        "right_height_px": right_height,
    }


def _fit_window(
    gray: np.ndarray,
    pixel_size_mm: float,
    gt_width_mm: float,
    gt_height_mm: float,
    prefer_diplib: bool,
    fit_mode: str,
    edge_bias: str,
    light_fraction: float,
    undistorted: bool,
) -> WindowFit:
    _comp, bbox, threshold, score = _select_bright_window_component(gray)
    side_positions = _scan_component_sides(_comp, bbox)
    side_lines, gradient_method = _refine_side_lines(
        gray, bbox, side_positions, prefer_diplib,
        fit_mode, edge_bias, light_fraction,
    )
    corners = _side_line_corners(side_lines)
    if not np.all(np.isfinite(corners)):
        raise RuntimeError("Invalid fitted window corners")
    px = _measure_distances(corners)
    width_px = px["left_right_px"]
    height_px = px["top_bottom_px"]
    distances_mm = {
        "width_mm": width_px * pixel_size_mm,
        "height_mm": height_px * pixel_size_mm,
    }
    errors_mm = {
        "width_error_mm": distances_mm["width_mm"] - gt_width_mm,
        "height_error_mm": distances_mm["height_mm"] - gt_height_mm,
    }
    return WindowFit(
        bbox=bbox,
        threshold=threshold,
        confidence=float(min(1.0, max(0.0, score / max(gray.size, 1)))),
        side_lines=side_lines,
        corners=corners,
        distances_px=px,
        distances_mm=distances_mm,
        errors_mm=errors_mm,
        gradient_method=gradient_method,
        fit_mode=fit_mode,
        edge_bias=edge_bias,
        light_fraction=light_fraction,
        undistorted=undistorted,
        pixel_size_mm=pixel_size_mm,
    )


def _line_segment_for_box(
    line: tuple[float, float, float],
    image_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = image_shape[:2]
    a, b, c = line
    pts = []
    if abs(b) > 1e-12:
        for x in (0.0, float(w - 1)):
            y = -(a * x + c) / b
            if -h <= y <= 2 * h:
                pts.append((x, y))
    if abs(a) > 1e-12:
        for y in (0.0, float(h - 1)):
            x = -(b * y + c) / a
            if -w <= x <= 2 * w:
                pts.append((x, y))
    if len(pts) < 2:
        return (0, 0), (0, 0)
    return tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int))


def _draw_overlay(
    image: np.ndarray,
    fit: WindowFit,
    gt_width_mm: float,
    gt_height_mm: float,
    path: Path,
) -> None:
    canvas = image.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    xmin, ymin, xmax, ymax = fit.bbox
    cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (80, 80, 255), 2)
    colors = {
        "left": (0, 255, 0),
        "right": (0, 255, 0),
        "top": (255, 160, 0),
        "bottom": (255, 160, 0),
    }
    for name, line in fit.side_lines.items():
        p1, p2 = _line_segment_for_box(line, canvas.shape[:2])
        cv2.line(canvas, p1, p2, colors[name], 3, cv2.LINE_AA)
    pts = np.round(fit.corners).astype(np.int32)
    cv2.polylines(canvas, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)

    top_left, top_right, bottom_right, bottom_left = fit.corners
    center_top = ((top_left + top_right) * 0.5).astype(int)
    center_left = ((top_left + bottom_left) * 0.5).astype(int)
    width_text = (
        f"{fit.distances_px['left_right_px']:.2f}px  "
        f"{fit.distances_mm['width_mm']:.4f}mm  "
        f"err {fit.errors_mm['width_error_mm']:+.4f}mm"
    )
    height_text = (
        f"{fit.distances_px['top_bottom_px']:.2f}px  "
        f"{fit.distances_mm['height_mm']:.4f}mm  "
        f"err {fit.errors_mm['height_error_mm']:+.4f}mm"
    )
    _put_label(canvas, width_text, tuple(center_top + np.array([-420, -35])))
    _put_label(canvas, height_text, tuple(center_left + np.array([20, 0])))
    meta = (
        f"pixel={fit.pixel_size_mm:.9f} mm/px  "
        f"GT=({gt_width_mm:.4f}, {gt_height_mm:.4f})mm  "
        f"{fit.fit_mode}  {fit.gradient_method}  bias={fit.edge_bias}"
    )
    _put_label(canvas, meta, (40, 80))

    h, w = canvas.shape[:2]
    scale = min(1.0, 2200.0 / max(w, h))
    if scale < 1.0:
        canvas = cv2.resize(
            canvas,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), canvas)


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    x, y = origin
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 0, 0), 2, cv2.LINE_AA)


def _fit_to_dict(fit: WindowFit, image_path: str, gt_width_mm: float, gt_height_mm: float) -> dict:
    return {
        "image": image_path,
        "undistorted": fit.undistorted,
        "pixel_size_mm": fit.pixel_size_mm,
        "gt_width_mm": gt_width_mm,
        "gt_height_mm": gt_height_mm,
        "threshold": fit.threshold,
        "component_bbox": list(fit.bbox),
        "gradient_method": fit.gradient_method,
        "fit_mode": fit.fit_mode,
        "edge_bias": fit.edge_bias,
        "light_fraction": fit.light_fraction,
        "distances_px": {k: float(v) for k, v in fit.distances_px.items()},
        "distances_mm": {k: float(v) for k, v in fit.distances_mm.items()},
        "errors_mm": {k: float(v) for k, v in fit.errors_mm.items()},
        "corners_px": fit.corners.tolist(),
        "side_lines_ax_by_c": {
            k: [float(x) for x in v] for k, v in fit.side_lines.items()
        },
        "scale_source": "AppConfig.pixel_size_mm only; no CAD/window affine scale",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="/tmp/cadrefs_camera_capture.png")
    parser.add_argument("--gt-width-mm", type=float, required=True)
    parser.add_argument("--gt-height-mm", type=float, required=True)
    parser.add_argument("--overlay", default="/tmp/backlight_hollow_window_validation.png")
    parser.add_argument("--json", default="/tmp/backlight_hollow_window_validation.json")
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Do not apply the saved lens undistortion model before fitting.",
    )
    parser.add_argument(
        "--no-diplib",
        action="store_true",
        help="Use OpenCV Scharr gradient even if DIPLib is installed.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["light-inner", "gradient"],
        default="light-inner",
        help=(
            "Line fitting mode. light-inner fits the bright-side grayscale "
            "crossing inside the backlight transition band; gradient fits the "
            "strongest gradient peak."
        ),
    )
    parser.add_argument(
        "--light-fraction",
        type=float,
        default=0.85,
        help=(
            "For --fit-mode light-inner, fraction from local dark level to "
            "bright level. Higher values bias further into the bright window."
        ),
    )
    parser.add_argument(
        "--edge-bias",
        choices=["inner", "strongest", "outer"],
        default="inner",
        help=(
            "Peak selection for the bright window band. 'inner' biases each "
            "side toward the hollow-window interior; 'strongest' preserves the "
            "old strongest-gradient behavior."
        ),
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.load()
    pixel_size_mm = float(getattr(cfg, "pixel_size_mm", 0.0) or 0.0)
    if pixel_size_mm <= 0.0:
        raise RuntimeError("AppConfig.pixel_size_mm must be positive")

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot load image: {args.image}")
    if args.no_undistort:
        working = image
        undistorted = False
    else:
        working, undistorted = undistort_if_calibrated(image, cfg)
    gray = _to_gray(working)
    fit = _fit_window(
        gray,
        pixel_size_mm=pixel_size_mm,
        gt_width_mm=float(args.gt_width_mm),
        gt_height_mm=float(args.gt_height_mm),
        prefer_diplib=not args.no_diplib,
        fit_mode=args.fit_mode,
        edge_bias=args.edge_bias,
        light_fraction=args.light_fraction,
        undistorted=bool(undistorted),
    )

    overlay_path = Path(args.overlay)
    json_path = Path(args.json)
    _draw_overlay(working, fit, args.gt_width_mm, args.gt_height_mm, overlay_path)
    output = _fit_to_dict(fit, args.image, args.gt_width_mm, args.gt_height_mm)
    output["overlay"] = str(overlay_path)
    json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "image": args.image,
        "undistorted": fit.undistorted,
        "pixel_size_mm": fit.pixel_size_mm,
        "gradient_method": fit.gradient_method,
        "fit_mode": fit.fit_mode,
        "edge_bias": fit.edge_bias,
        "left_right_distance_px": fit.distances_px["left_right_px"],
        "top_bottom_distance_px": fit.distances_px["top_bottom_px"],
        "width_mm": fit.distances_mm["width_mm"],
        "height_mm": fit.distances_mm["height_mm"],
        "width_error_mm": fit.errors_mm["width_error_mm"],
        "height_error_mm": fit.errors_mm["height_error_mm"],
        "overlay": str(overlay_path),
        "json": str(json_path),
        "scale_source": "chessboard pixel_size_mm only",
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
