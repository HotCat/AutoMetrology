"""Registration from the dark rounded-rectangle product window.

This method is intended as an alternate initial registration for xintai-style
parts where the window is visible as a large dark rounded rectangle.  Rounded
corners make corner detection unstable, so the solver uses the four straight
side positions and maps them to four CAD line handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.feature import FeatureType
from ..models.repository import FeatureRepository
from . import affine_solver

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


DEFAULT_WINDOW_LINE_HANDLES = {
    "right": "9e40c968",
    "top": "9c0bd3a0",
    "left": "7e6e8eb2",
    "bottom": "71490463",
}


@dataclass
class WindowLineRegistrationResult:
    affine: np.ndarray
    side_positions: dict[str, float]
    line_handles: dict[str, str]
    component_bbox: tuple[int, int, int, int]
    confidence: float
    homography: Optional[np.ndarray] = None
    transform_model: str = "edge_affine"
    side_lines: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    image_corners: Optional[np.ndarray] = None
    cad_corners: Optional[np.ndarray] = None
    homography_safety: str = ""
    method: str = "window_line_registration"

    @property
    def transform(self) -> np.ndarray:
        if self.transform_model == "edge_homography" and self.homography is not None:
            return self.homography
        return self.affine


def register_window_lines(
    repo: FeatureRepository,
    image: np.ndarray,
    line_handles: Optional[dict[str, str]] = None,
    edge_tokens: Optional[list[str]] = None,
    pixel_size_mm: Optional[float] = None,
    prefer_homography: bool = True,
) -> WindowLineRegistrationResult:
    """Compute a pixel -> CAD world transform from the dark product window."""
    if not HAS_CV2:
        raise RuntimeError("OpenCV is required for window line registration")
    if image is None:
        raise ValueError("image is required")

    if edge_tokens:
        cad_corners, handles = _cad_window_from_edges(repo, edge_tokens)
        target_aspect = _corner_aspect(cad_corners)
    else:
        handles = dict(DEFAULT_WINDOW_LINE_HANDLES)
        if line_handles:
            handles.update({k: v for k, v in line_handles.items() if v})
        cad_corners = _cad_window_corners(repo, handles)
        target_aspect = None
    gray = _to_gray(image)
    side_positions, bbox, confidence, side_lines, image_corners = (
        _detect_registration_geometry(gray, target_aspect)
    )
    if edge_tokens:
        cad_corners = _select_cad_corner_order(
            repo, edge_tokens, image_corners, cad_corners, gray,
        )
    affine = _build_affine_from_corners(image_corners, cad_corners)
    homography = _build_homography_from_corners(image_corners, cad_corners)
    transform_model = "edge_affine"
    safety_reason = ""
    if prefer_homography and homography is not None:
        if pixel_size_mm is None:
            safety_reason = "pixel size unavailable; using affine"
        else:
            try:
                from ..calibration.transform_safety import validate_pixel_to_world_transform
                safety = validate_pixel_to_world_transform(
                    homography,
                    float(pixel_size_mm),
                    image_size=(int(gray.shape[1]), int(gray.shape[0])),
                )
                if safety.safe:
                    transform_model = "edge_homography"
                else:
                    safety_reason = safety.reason
            except Exception as exc:
                safety_reason = str(exc)
    return WindowLineRegistrationResult(
        affine=affine,
        side_positions=side_positions,
        line_handles=handles,
        component_bbox=bbox,
        confidence=confidence,
        homography=homography,
        transform_model=transform_model,
        side_lines=side_lines,
        image_corners=image_corners,
        cad_corners=cad_corners,
        homography_safety=safety_reason,
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _detect_window_geometry(
    gray: np.ndarray,
    target_aspect: Optional[float] = None,
) -> tuple[
    dict[str, float],
    tuple[int, int, int, int],
    float,
    dict[str, tuple[float, float, float]],
    np.ndarray,
]:
    labels = None
    best = None
    best_score = float("-inf")
    for threshold in _window_threshold_candidates(gray):
        mask = (gray < threshold).astype(np.uint8)
        n, cur_labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            continue
        candidate, score = _select_window_component(
            gray.shape, stats, centroids, target_aspect,
        )
        if candidate is not None and score > best_score:
            best = (candidate, stats)
            labels = cur_labels
            best_score = score
    if best is None or labels is None:
        raise RuntimeError("No suitable dark window component detected")

    best_idx, stats = best
    x, y, bw, bh, _area = [int(v) for v in stats[best_idx]]
    xmin, ymin = x, y
    xmax, ymax = x + bw - 1, y + bh - 1
    comp = labels == best_idx

    scan = _scan_component_sides(comp, xmin, ymin, xmax, ymax)
    side_lines = _fit_component_side_lines(comp, xmin, ymin, xmax, ymax)
    image_corners = _side_line_corners(side_lines)

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    side_positions = {
        "left": _line_x_at_y(side_lines["left"], cy, scan["left"]),
        "right": _line_x_at_y(side_lines["right"], cy, scan["right"]),
        "top": _line_y_at_x(side_lines["top"], cx, scan["top"]),
        "bottom": _line_y_at_x(side_lines["bottom"], cx, scan["bottom"]),
    }

    width = side_positions["right"] - side_positions["left"]
    height = side_positions["bottom"] - side_positions["top"]
    if width <= 100 or height <= 100:
        raise RuntimeError(f"Invalid window side positions: {side_positions}")
    if not np.all(np.isfinite(image_corners)):
        raise RuntimeError("Invalid fitted window corners")
    if abs(_polygon_area(image_corners)) < 10000.0:
        raise RuntimeError("Fitted window corners are degenerate")

    coverage = min(
        1.0,
        max(0.0, width / max(float(bw), 1.0))
        * max(0.0, height / max(float(bh), 1.0)),
    )
    confidence = float(0.75 + 0.25 * coverage)
    return side_positions, (xmin, ymin, xmax, ymax), confidence, side_lines, image_corners


def _detect_registration_geometry(
    gray: np.ndarray,
    target_aspect: Optional[float],
) -> tuple[
    dict[str, float],
    tuple[int, int, int, int],
    float,
    dict[str, tuple[float, float, float]],
    np.ndarray,
]:
    dark = _detect_window_geometry(gray, target_aspect)
    if target_aspect is None or not np.isfinite(target_aspect) or target_aspect <= 0:
        return dark
    candidates = [dark]
    try:
        candidates.append(_detect_grid_cell_geometry(gray))
    except Exception:
        pass

    def score(candidate) -> float:
        aspect = _corner_aspect(candidate[4])
        if not np.isfinite(aspect) or aspect <= 0:
            return float("inf")
        bbox = candidate[1]
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        h, w = gray.shape[:2]
        center_penalty = (
            abs(cx - w / 2.0) / max(float(w), 1.0)
            + abs(cy - h / 2.0) / max(float(h), 1.0)
        )
        return abs(float(np.log(aspect / target_aspect))) + center_penalty * 0.08

    return min(candidates, key=score)


def _detect_grid_cell_geometry(
    gray: np.ndarray,
) -> tuple[
    dict[str, float],
    tuple[int, int, int, int],
    float,
    dict[str, tuple[float, float, float]],
    np.ndarray,
]:
    h, w = gray.shape[:2]
    threshold = int(np.clip(np.percentile(gray, 55), 120, 225))
    mask = (gray < threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(220, int(w * 0.14)), 1),
    )
    vert_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(180, int(h * 0.14))),
    )
    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vert_kernel)
    horizontal = _line_components(horiz, horizontal=True, image_shape=gray.shape)
    vertical = _line_components(vert, horizontal=False, image_shape=gray.shape)
    if len(horizontal) < 2 or len(vertical) < 2:
        raise RuntimeError("Could not detect enough printed grid lines")

    best = None
    best_score = float("inf")
    for top in horizontal:
        for bottom in horizontal:
            y_top = min(top["pos"], bottom["pos"])
            y_bottom = max(top["pos"], bottom["pos"])
            height = y_bottom - y_top
            if height < h * 0.20 or height > h * 0.75:
                continue
            for left in vertical:
                for right in vertical:
                    x_left = min(left["pos"], right["pos"])
                    x_right = max(left["pos"], right["pos"])
                    width = x_right - x_left
                    if width < w * 0.35 or width > w * 0.90:
                        continue
                    aspect = width / max(height, 1.0)
                    if not 1.2 <= aspect <= 3.2:
                        continue
                    if not _line_span_covers(top, x_left, x_right, "x"):
                        continue
                    if not _line_span_covers(bottom, x_left, x_right, "x"):
                        continue
                    if not _line_span_covers(left, y_top, y_bottom, "y"):
                        continue
                    if not _line_span_covers(right, y_top, y_bottom, "y"):
                        continue
                    cx = (x_left + x_right) / 2.0
                    cy = (y_top + y_bottom) / 2.0
                    center_penalty = abs(cx - w / 2.0) / w + abs(cy - h / 2.0) / h
                    score = center_penalty - min(width * height / float(w * h), 1.0) * 0.15
                    if score < best_score:
                        best_score = score
                        best = (x_left, y_top, x_right, y_bottom)
    if best is None:
        raise RuntimeError("No suitable printed grid cell detected")

    left, top, right, bottom = [float(v) for v in best]
    side_positions = {
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }
    side_lines = {
        "left": (1.0, 0.0, -left),
        "right": (1.0, 0.0, -right),
        "top": (0.0, 1.0, -top),
        "bottom": (0.0, 1.0, -bottom),
    }
    corners = _side_line_corners(side_lines)
    return (
        side_positions,
        (int(round(left)), int(round(top)), int(round(right)), int(round(bottom))),
        0.86,
        side_lines,
        corners,
    )


def _line_components(
    mask: np.ndarray,
    horizontal: bool,
    image_shape: tuple[int, int],
) -> list[dict[str, float]]:
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    h, w = image_shape[:2]
    result = []
    for idx in range(1, n):
        x, y, bw, bh, area = [float(v) for v in stats[idx]]
        if horizontal:
            if bw < w * 0.30 or bh > h * 0.05:
                continue
            result.append({
                "pos": y + bh / 2.0,
                "x0": x,
                "x1": x + bw,
                "y0": y,
                "y1": y + bh,
                "area": area,
            })
        else:
            if bh < h * 0.20 or bw > w * 0.05:
                continue
            result.append({
                "pos": x + bw / 2.0,
                "x0": x,
                "x1": x + bw,
                "y0": y,
                "y1": y + bh,
                "area": area,
            })
    return result


def _line_span_covers(line: dict[str, float], start: float, end: float, axis: str) -> bool:
    pad = 35.0
    return line[f"{axis}0"] <= start + pad and line[f"{axis}1"] >= end - pad


def _axis_aligned_side_lines(
    side_positions: dict[str, float],
) -> dict[str, tuple[float, float, float]]:
    left = float(side_positions["left"])
    right = float(side_positions["right"])
    top = float(side_positions["top"])
    bottom = float(side_positions["bottom"])
    return {
        "left": (1.0, 0.0, -left),
        "right": (1.0, 0.0, -right),
        "top": (0.0, 1.0, -top),
        "bottom": (0.0, 1.0, -bottom),
    }


def _window_threshold_candidates(gray: np.ndarray) -> list[int]:
    """Return dark thresholds robust to lighting shifts."""
    values = [100, 110, 115, 120, 130, 140, 160]
    try:
        otsu, _ = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        values.extend([
            int(round(float(otsu) - 60.0)),
            int(round(float(otsu) - 45.0)),
            int(round(float(otsu) - 30.0)),
        ])
    except Exception:
        pass
    percentiles = np.percentile(gray, [10, 20, 30, 40])
    values.extend(int(round(float(v))) for v in percentiles)
    return sorted({int(np.clip(v, 70, 180)) for v in values})


def _select_window_component(
    shape: tuple[int, int],
    stats: np.ndarray,
    centroids: np.ndarray,
    target_aspect: Optional[float] = None,
) -> tuple[Optional[int], float]:
    h, w = shape[:2]
    best = None
    best_score = float("-inf")
    for idx in range(1, len(stats)):
        x, y, bw, bh, area = stats[idx]
        if area < 80000 or bw < w * 0.25 or bh < h * 0.20:
            continue
        if bw > w * 0.82 or bh > h * 0.78:
            continue
        aspect = float(bw) / max(float(bh), 1.0)
        if not 0.9 <= aspect <= 2.3:
            continue
        cx, cy = centroids[idx]
        center_penalty = abs(cx - w / 2.0) + abs(cy - h / 2.0)
        border_penalty = 0.0
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            border_penalty = float(area) * 0.6
        if target_aspect is not None and np.isfinite(target_aspect) and target_aspect > 0:
            normalized_aspect = max(aspect, 1.0 / max(aspect, 1e-12))
            aspect_error = abs(float(np.log(normalized_aspect / target_aspect)))
            score = (
                -aspect_error * 1_000_000.0
                + float(area) * 0.02
                - center_penalty * 10.0
                - border_penalty
            )
        else:
            score = float(area) - center_penalty * 80.0 - border_penalty
        if score > best_score:
            best = idx
            best_score = score
    return best, best_score


def _scan_component_sides(
    comp: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int,
) -> dict[str, float]:
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    top_vals = []
    bottom_vals = []
    for x in range(xmin + int(width * 0.12), xmax - int(width * 0.12) + 1):
        ys = np.flatnonzero(comp[:, x])
        if len(ys):
            top_vals.append(float(ys.min()))
            bottom_vals.append(float(ys.max()))

    left_vals = []
    right_vals = []
    for y in range(ymin + int(height * 0.12), ymax - int(height * 0.12) + 1):
        xs = np.flatnonzero(comp[y, :])
        if len(xs):
            left_vals.append(float(xs.min()))
            right_vals.append(float(xs.max()))

    if not top_vals or not(bottom_vals) or not left_vals or not right_vals:
        raise RuntimeError("Could not scan dark window sides")

    return {
        "top": float(np.percentile(top_vals, 10)),
        "bottom": float(np.percentile(bottom_vals, 90)),
        "left": float(np.percentile(left_vals, 10)),
        "right": float(np.percentile(right_vals, 90)),
    }


def _hough_side_positions(
    gray: np.ndarray, bbox: tuple[int, int, int, int],
) -> dict[str, float]:
    xmin, ymin, xmax, ymax = bbox
    pad = 80
    x0 = max(0, xmin - pad)
    y0 = max(0, ymin - pad)
    x1 = min(gray.shape[1], xmax + pad + 1)
    y1 = min(gray.shape[0], ymax + pad + 1)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return {}

    edges = cv2.Canny(cv2.GaussianBlur(crop, (5, 5), 0), 20, 70, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=50,
        minLineLength=max(250, int(min(x1 - x0, y1 - y0) * 0.28)),
        maxLineGap=80,
    )
    if lines is None:
        return {}

    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    vertical_left = []
    vertical_right = []
    horizontal_top = []
    horizontal_bottom = []

    for raw in lines[:, 0, :]:
        lx1, ly1, lx2, ly2 = [float(v) for v in raw]
        lx1 += x0
        lx2 += x0
        ly1 += y0
        ly2 += y0
        dx = lx2 - lx1
        dy = ly2 - ly1
        length = float(np.hypot(dx, dy))
        if length < 250.0:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        mx = (lx1 + lx2) / 2.0
        my = (ly1 + ly2) / 2.0
        if angle >= 85.0:
            item = (length, (lx1 + lx2) / 2.0)
            if mx < cx:
                vertical_left.append(item)
            else:
                vertical_right.append(item)
        elif angle <= 5.0:
            item = (length, (ly1 + ly2) / 2.0)
            if my < cy:
                horizontal_top.append(item)
            else:
                horizontal_bottom.append(item)

    result = {}
    if vertical_left:
        result["left"] = float(np.average(
            [x for length, x in vertical_left],
            weights=[length for length, x in vertical_left],
        ))
    if vertical_right:
        result["right"] = float(np.average(
            [x for length, x in vertical_right],
            weights=[length for length, x in vertical_right],
        ))
    if horizontal_top:
        result["top"] = float(np.average(
            [y for length, y in horizontal_top],
            weights=[length for length, y in horizontal_top],
        ))
    if horizontal_bottom:
        result["bottom"] = float(np.average(
            [y for length, y in horizontal_bottom],
            weights=[length for length, y in horizontal_bottom],
        ))
    return result


def _fit_component_side_lines(
    comp: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int,
) -> dict[str, tuple[float, float, float]]:
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    top_pts = []
    bottom_pts = []
    for x in range(xmin + int(width * 0.12), xmax - int(width * 0.12) + 1):
        ys = np.flatnonzero(comp[:, x])
        if len(ys):
            top_pts.append((float(x), float(ys.min())))
            bottom_pts.append((float(x), float(ys.max())))

    left_pts = []
    right_pts = []
    for y in range(ymin + int(height * 0.12), ymax - int(height * 0.12) + 1):
        xs = np.flatnonzero(comp[y, :])
        if len(xs):
            left_pts.append((float(xs.min()), float(y)))
            right_pts.append((float(xs.max()), float(y)))

    return {
        "left": _fit_line_ransac(np.asarray(left_pts, dtype=np.float64)),
        "right": _fit_line_ransac(np.asarray(right_pts, dtype=np.float64)),
        "top": _fit_line_ransac(np.asarray(top_pts, dtype=np.float64)),
        "bottom": _fit_line_ransac(np.asarray(bottom_pts, dtype=np.float64)),
    }


def _fit_line_ransac(points: np.ndarray) -> tuple[float, float, float]:
    if points.ndim != 2 or points.shape[0] < 20 or points.shape[1] != 2:
        raise RuntimeError("Not enough side points for line fitting")

    try:
        line = cv2.fitLine(
            points.astype(np.float32),
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        )
        vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]
    except Exception:
        centered = points - np.mean(points, axis=0)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        vx, vy = vt[0]
        x0, y0 = np.mean(points, axis=0)

    norm = float(np.hypot(vx, vy))
    if norm <= 1e-12:
        raise RuntimeError("Degenerate side line fit")
    vx /= norm
    vy /= norm
    a = -vy
    b = vx
    c = -(a * x0 + b * y0)
    line_norm = float(np.hypot(a, b))
    if line_norm <= 1e-12:
        raise RuntimeError("Degenerate side line equation")
    return (a / line_norm, b / line_norm, c / line_norm)


def _side_line_corners(
    lines: dict[str, tuple[float, float, float]],
) -> np.ndarray:
    return np.array([
        _intersect_lines(lines["left"], lines["top"]),
        _intersect_lines(lines["right"], lines["top"]),
        _intersect_lines(lines["right"], lines["bottom"]),
        _intersect_lines(lines["left"], lines["bottom"]),
    ], dtype=np.float64)


def _intersect_lines(
    l1: tuple[float, float, float],
    l2: tuple[float, float, float],
) -> np.ndarray:
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) <= 1e-9:
        raise RuntimeError("Fitted window sides are parallel")
    return np.array([
        (b1 * c2 - b2 * c1) / det,
        (c1 * a2 - c2 * a1) / det,
    ], dtype=np.float64)


def _line_x_at_y(
    line: tuple[float, float, float],
    y: float,
    fallback: float,
) -> float:
    a, b, c = line
    if abs(a) <= 1e-9:
        return float(fallback)
    return float(-(b * y + c) / a)


def _line_y_at_x(
    line: tuple[float, float, float],
    x: float,
    fallback: float,
) -> float:
    a, b, c = line
    if abs(b) <= 1e-9:
        return float(fallback)
    return float(-(a * x + c) / b)


def _polygon_area(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _corner_aspect(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        return 0.0
    lengths = [
        float(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        for i in range(4)
    ]
    positives = [length for length in lengths if length > 1e-12]
    if not positives:
        return 0.0
    long_side = max(lengths)
    short_side = min(positives)
    if short_side <= 1e-12:
        return 0.0
    return float(long_side / short_side)


def _cad_window_corners(
    repo: FeatureRepository,
    handles: dict[str, str],
) -> np.ndarray:
    top = _resolve_line(repo, handles["top"])
    bottom = _resolve_line(repo, handles["bottom"])
    left = _resolve_line(repo, handles["left"])
    right = _resolve_line(repo, handles["right"])
    lines = {
        "left": _cad_line_equation(left.geometry),
        "right": _cad_line_equation(right.geometry),
        "top": _cad_line_equation(top.geometry),
        "bottom": _cad_line_equation(bottom.geometry),
    }
    return _side_line_corners(lines)


def _cad_window_from_edges(
    repo: FeatureRepository,
    edge_tokens: list[str],
) -> tuple[np.ndarray, dict[str, str]]:
    tokens = [str(token).strip() for token in edge_tokens if str(token).strip()]
    if len(tokens) != 4:
        raise ValueError("Window registration requires exactly four CAD edge lines")
    features = [_resolve_line(repo, token) for token in tokens]
    corners = _cad_corners_from_line_features(features)
    handles = {
        "edge1": _line_token(features[0]),
        "edge2": _line_token(features[1]),
        "edge3": _line_token(features[2]),
        "edge4": _line_token(features[3]),
    }
    return corners, handles


def _select_cad_corner_order(
    repo: FeatureRepository,
    edge_tokens: list[str],
    image_corners: np.ndarray,
    cad_corners: np.ndarray,
    gray: np.ndarray,
) -> np.ndarray:
    context_points = _sample_cad_context_points(repo, edge_tokens, cad_corners)
    if len(context_points) < 20:
        return cad_corners

    edge_distance = _image_edge_distance(gray)
    best = np.asarray(cad_corners, dtype=np.float64)
    best_score = float("inf")
    for candidate in _cad_corner_order_candidates(best):
        try:
            transform = _build_affine_from_corners(image_corners, candidate)
            score = _score_cad_to_image_edges(transform, context_points, edge_distance)
        except Exception:
            continue
        if score < best_score:
            best = candidate
            best_score = score
    return best


def _cad_corner_order_candidates(corners: np.ndarray) -> list[np.ndarray]:
    pts = np.asarray(corners, dtype=np.float64)
    candidates = []
    for shift in range(4):
        cur = np.roll(pts, -shift, axis=0)
        if not any(np.allclose(cur, prev, atol=1e-9) for prev in candidates):
            candidates.append(cur.copy())
    return candidates


def _image_edge_distance(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120, L2gradient=True)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def _score_cad_to_image_edges(
    pixel_to_cad: np.ndarray,
    cad_points: np.ndarray,
    edge_distance: np.ndarray,
) -> float:
    cad_to_pixel = np.linalg.inv(pixel_to_cad)
    px = affine_solver.apply_projective(cad_to_pixel, cad_points)
    h, w = edge_distance.shape[:2]
    valid = (
        np.isfinite(px[:, 0]) & np.isfinite(px[:, 1])
        & (px[:, 0] >= 0) & (px[:, 0] < w)
        & (px[:, 1] >= 0) & (px[:, 1] < h)
    )
    if int(np.count_nonzero(valid)) < 20:
        return float("inf")
    coords = np.round(px[valid]).astype(np.int32)
    dists = edge_distance[coords[:, 1], coords[:, 0]].astype(np.float64)
    return float(np.percentile(np.clip(dists, 0.0, 80.0), 70))


def _sample_cad_context_points(
    repo: FeatureRepository,
    edge_tokens: list[str],
    cad_corners: np.ndarray,
) -> np.ndarray:
    excluded = {
        _resolve_line(repo, token).feature_id
        for token in edge_tokens
        if str(token).strip()
    }
    pts = []
    corner_arr = np.asarray(cad_corners, dtype=np.float64)
    min_x, min_y = np.min(corner_arr, axis=0)
    max_x, max_y = np.max(corner_arr, axis=0)
    span = max(float(max_x - min_x), float(max_y - min_y), 1.0)
    margin = span * 1.5
    bbox = (min_x - margin, min_y - margin, max_x + margin, max_y + margin)
    for feature in repo.all_features():
        if feature.feature_id in excluded:
            continue
        sampled = _sample_feature_points(feature, density=0.45)
        if sampled.size == 0:
            continue
        mask = (
            (sampled[:, 0] >= bbox[0]) & (sampled[:, 0] <= bbox[2])
            & (sampled[:, 1] >= bbox[1]) & (sampled[:, 1] <= bbox[3])
        )
        if np.any(mask):
            pts.append(sampled[mask])
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack(pts)


def _sample_feature_points(feature, density: float) -> np.ndarray:
    geom = feature.geometry
    if feature.feature_type == FeatureType.LINE:
        p1 = np.array([geom["x1"], geom["y1"]], dtype=np.float64)
        p2 = np.array([geom["x2"], geom["y2"]], dtype=np.float64)
        return _sample_segment(p1, p2, density)
    if feature.feature_type == FeatureType.POLYLINE:
        points = np.asarray(geom.get("points", []), dtype=np.float64)
        if len(points) < 2:
            return np.empty((0, 2), dtype=np.float64)
        segs = []
        for idx in range(len(points) - 1):
            segs.append(_sample_segment(points[idx], points[idx + 1], density))
        if geom.get("closed", False) and len(points) > 2:
            segs.append(_sample_segment(points[-1], points[0], density))
        return np.vstack(segs) if segs else np.empty((0, 2), dtype=np.float64)
    if feature.feature_type in {FeatureType.CIRCLE, FeatureType.ARC}:
        cx = float(geom["cx"])
        cy = float(geom["cy"])
        radius = float(geom["radius"])
        if radius <= 1e-9:
            return np.empty((0, 2), dtype=np.float64)
        if feature.feature_type == FeatureType.CIRCLE:
            a0, a1 = 0.0, 2.0 * np.pi
            endpoint = False
        else:
            a0 = np.radians(float(geom.get("start_angle", 0.0)))
            a1 = np.radians(float(geom.get("end_angle", 360.0)))
            if a1 <= a0:
                a1 += 2.0 * np.pi
            endpoint = True
        arc_len = radius * abs(a1 - a0)
        count = max(8, int(round(arc_len * density)))
        angles = np.linspace(a0, a1, count, endpoint=endpoint)
        return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])
    return np.empty((0, 2), dtype=np.float64)


def _sample_segment(p1: np.ndarray, p2: np.ndarray, density: float) -> np.ndarray:
    length = float(np.linalg.norm(p2 - p1))
    count = max(2, int(round(length * density)))
    t = np.linspace(0.0, 1.0, count)
    return p1 + np.outer(t, p2 - p1)


def _line_token(feature) -> str:
    return str(feature.dxf_handle or feature.feature_id)


def _cad_corners_from_line_features(features: list) -> np.ndarray:
    endpoints = []
    for feature in features:
        geom = feature.geometry
        endpoints.append((float(geom["x1"]), float(geom["y1"])))
        endpoints.append((float(geom["x2"]), float(geom["y2"])))
    pts = np.asarray(endpoints, dtype=np.float64)
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    if max_x - min_x <= 1e-9 or max_y - min_y <= 1e-9:
        raise ValueError("Selected CAD window edges are degenerate")

    horizontal = []
    vertical = []
    for feature in features:
        geom = feature.geometry
        dx = abs(float(geom["x2"]) - float(geom["x1"]))
        dy = abs(float(geom["y2"]) - float(geom["y1"]))
        if dy <= max(0.02, dx * 0.01):
            horizontal.append(feature)
        elif dx <= max(0.02, dy * 0.01):
            vertical.append(feature)
    if len(horizontal) == 2 and len(vertical) == 2:
        return np.array([
            [min_x, min_y],
            [min_x, max_y],
            [max_x, max_y],
            [max_x, min_y],
        ], dtype=np.float64)

    lines = [_cad_line_equation(feature.geometry) for feature in features]
    intersections = []
    for i, line_a in enumerate(lines):
        for line_b in lines[i + 1:]:
            try:
                pt = _intersect_lines(line_a, line_b)
            except RuntimeError:
                continue
            if (
                min_x - 1.0 <= pt[0] <= max_x + 1.0
                and min_y - 1.0 <= pt[1] <= max_y + 1.0
            ):
                intersections.append(pt)
    unique = []
    for pt in intersections:
        if not any(np.linalg.norm(pt - prev) < 1e-3 for prev in unique):
            unique.append(pt)
    if len(unique) != 4:
        raise ValueError("Selected CAD edges do not form one four-sided window")
    return _order_cad_corners(np.asarray(unique, dtype=np.float64))


def _order_cad_corners(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    ordered = sorted(
        points,
        key=lambda pt: np.arctan2(float(pt[1] - center[1]), float(pt[0] - center[0])),
    )
    pts = np.asarray(ordered, dtype=np.float64)
    start = int(np.argmin(pts[:, 0] - pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    if _polygon_area(pts) < 0:
        pts = np.asarray([pts[0], pts[3], pts[2], pts[1]], dtype=np.float64)
    return pts


def _cad_line_equation(geom: dict) -> tuple[float, float, float]:
    x1 = float(geom["x1"])
    y1 = float(geom["y1"])
    x2 = float(geom["x2"])
    y2 = float(geom["y2"])
    dx = x2 - x1
    dy = y2 - y1
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-12:
        raise ValueError("Degenerate CAD line")
    a = -dy / norm
    b = dx / norm
    c = -(a * x1 + b * y1)
    return (a, b, c)


def _build_affine_from_corners(
    image_corners: np.ndarray,
    cad_corners: np.ndarray,
) -> np.ndarray:
    return affine_solver.solve_from_correspondences(
        np.asarray(image_corners, dtype=np.float64),
        np.asarray(cad_corners, dtype=np.float64),
    )


def _build_homography_from_corners(
    image_corners: np.ndarray,
    cad_corners: np.ndarray,
) -> Optional[np.ndarray]:
    if not HAS_CV2:
        return None
    h, _mask = cv2.findHomography(
        np.asarray(image_corners, dtype=np.float32),
        np.asarray(cad_corners, dtype=np.float32),
        0,
    )
    if h is None or h.shape != (3, 3) or not np.all(np.isfinite(h)):
        return None
    if abs(float(h[2, 2])) > 1e-12:
        h = h / float(h[2, 2])
    return h.astype(np.float64)


def _build_affine_from_sides(
    repo: FeatureRepository,
    handles: dict[str, str],
    side_positions: dict[str, float],
) -> np.ndarray:
    top = _resolve_line(repo, handles["top"])
    bottom = _resolve_line(repo, handles["bottom"])
    left = _resolve_line(repo, handles["left"])
    right = _resolve_line(repo, handles["right"])

    top_x = _vertical_line_x(top.geometry, handles["top"])
    bottom_x = _vertical_line_x(bottom.geometry, handles["bottom"])
    left_y = _horizontal_line_y(left.geometry, handles["left"])
    right_y = _horizontal_line_y(right.geometry, handles["right"])

    sx = (bottom_x - top_x) / (
        side_positions["bottom"] - side_positions["top"]
    )
    tx = top_x - sx * side_positions["top"]
    sy = (right_y - left_y) / (
        side_positions["right"] - side_positions["left"]
    )
    ty = left_y - sy * side_positions["left"]

    return np.array([
        [0.0, sx, tx],
        [sy, 0.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _resolve_line(repo: FeatureRepository, token: str):
    feat = repo.get(token) or repo.get_by_handle(token)
    if feat is None:
        needle = str(token).lower()
        for candidate in repo.all_features():
            handle = str(candidate.dxf_handle or "").lower()
            if (
                candidate.feature_id.lower().startswith(needle)
                or handle.startswith(needle)
            ):
                feat = candidate
                break
    if feat is None:
        raise ValueError(f"Cannot resolve CAD line: {token}")
    if feat.feature_type != FeatureType.LINE:
        raise ValueError(f"CAD feature is not a line: {token}")
    return feat


def _vertical_line_x(geom: dict, label: str) -> float:
    dx = abs(float(geom["x2"]) - float(geom["x1"]))
    dy = abs(float(geom["y2"]) - float(geom["y1"]))
    if dx > max(0.02, dy * 0.01):
        raise ValueError(f"CAD line must be vertical for image top/bottom role: {label}")
    return float((geom["x1"] + geom["x2"]) / 2.0)


def _horizontal_line_y(geom: dict, label: str) -> float:
    dx = abs(float(geom["x2"]) - float(geom["x1"]))
    dy = abs(float(geom["y2"]) - float(geom["y1"]))
    if dy > max(0.02, dx * 0.01):
        raise ValueError(f"CAD line must be horizontal for image left/right role: {label}")
    return float((geom["y1"] + geom["y2"]) / 2.0)
