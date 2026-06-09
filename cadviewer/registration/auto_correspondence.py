"""Automatic two-point CAD/image correspondence from circular fiducials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import diplib as dip
    HAS_DIP = True
except ImportError:
    HAS_DIP = False


@dataclass
class CircleDetection:
    center: tuple[float, float]
    radius: float
    confidence: float
    method: str

    def to_dict(self) -> dict:
        return {
            "center": [float(self.center[0]), float(self.center[1])],
            "radius": float(self.radius),
            "confidence": float(self.confidence),
            "method": self.method,
        }


def clamp_roi(roi: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Clamp an x,y,w,h ROI to image bounds."""
    h, w = image_shape[:2]
    x, y, rw, rh = [int(round(v)) for v in roi]
    x = max(0, min(x, max(0, w - 1)))
    y = max(0, min(y, max(0, h - 1)))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    return x, y, rw, rh


def undistort_if_calibrated(image: np.ndarray, config) -> tuple[np.ndarray, bool]:
    """Apply OpenCV lens undistortion when AppConfig has calibration data."""
    if not HAS_CV2 or config is None:
        return image, False
    lc = getattr(config, "lens_calibration", None)
    if lc is None or not getattr(lc, "calibrated", False):
        return image, False
    mtx = lc.get_camera_matrix()
    dist = lc.get_dist_coeffs()
    if mtx is None or dist is None:
        return image, False
    return cv2.undistort(image, mtx, dist), True


def detect_circle_in_roi(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    expected_radius_px: Optional[float] = None,
    expected_center: Optional[tuple[float, float]] = None,
) -> Optional[CircleDetection]:
    """Detect the strongest circular fiducial in a user-specified ROI.

    The primary path follows ../detectFiducial/watershed2_circle.py:
    DIPLib gradient watershed plus MeasurementTool Roundness/Radius/Gravity.
    Hough and binary-contour detectors remain as fallbacks.  When CAD provides
    an expected radius/center, candidate scoring favors the intended fiducial.
    """
    if not HAS_CV2 or image is None:
        return None

    x, y, w, h = clamp_roi(roi, image.shape)
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    if crop.ndim == 3:
        raw_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        raw_gray = crop.copy()
    gray = cv2.GaussianBlur(raw_gray, (5, 5), 0)

    min_dim = min(w, h)
    if expected_radius_px is not None and expected_radius_px > 0:
        min_radius = max(2, int(expected_radius_px * 0.45))
        max_radius = max(min_radius + 2, int(expected_radius_px * 1.80))
    else:
        min_radius = max(3, int(min_dim * 0.04))
        max_radius = max(min_radius + 2, int(min_dim * 0.30))

    detections: list[CircleDetection] = []
    detections.extend(_detect_circles_dip_watershed(
        raw_gray, x, y, min_radius, max_radius,
        expected_radius_px=expected_radius_px,
        expected_center=expected_center,
    ))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(8, min_dim // 3),
        param1=80,
        param2=14,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        edge = cv2.Canny(gray, 50, 150)
        for cx, cy, radius in circles[0].astype(float):
            if radius <= 0:
                continue
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(radius)), 255, 2)
            support = float(np.mean(edge[mask > 0]) / 255.0) if np.any(mask > 0) else 0.0
            confidence = min(1.0, 0.50 + 0.50 * support)
            detections.append(CircleDetection(
                center=(x + float(cx), y + float(cy)),
                radius=float(radius),
                confidence=confidence,
                method="hough",
            ))

    # Binary-contour fallback works for filled dark or bright fiducials.
    for invert in (False, True):
        flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, 0, 255, flag | cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            perim = cv2.arcLength(cnt, True)
            if perim <= 1e-6:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)
            if circularity < 0.45:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < min_radius or radius > max_radius:
                continue
            fill_ratio = area / (np.pi * radius * radius) if radius > 1e-6 else 0.0
            if fill_ratio < 0.35 or fill_ratio > 1.25:
                continue
            confidence = float(max(0.0, min(1.0, circularity * min(fill_ratio, 1.0))))
            detections.append(CircleDetection(
                center=(x + float(cx), y + float(cy)),
                radius=float(radius),
                confidence=confidence,
                method="contour_inv" if invert else "contour",
            ))

    if not detections:
        return None

    return max(
        detections,
        key=lambda det: _circle_detection_score(
            det, x, y, w, h,
            expected_radius_px=expected_radius_px,
            expected_center=expected_center,
        ),
    )


def _detect_circles_dip_watershed(
    gray: np.ndarray,
    x0: int,
    y0: int,
    min_radius: float,
    max_radius: float,
    expected_radius_px: Optional[float] = None,
    expected_center: Optional[tuple[float, float]] = None,
) -> list[CircleDetection]:
    if not HAS_DIP:
        return []

    configs = (
        (0.8, 5, 20, False),
        (0.6, 3, 20, False),
        (0.8, 5, 20, True),
        (0.4, 3, 20, False),
    )
    detections: list[CircleDetection] = []

    for sigma, morph_radius, min_size, remove_edge in configs:
        try:
            dip_img = dip.Image(np.ascontiguousarray(gray))
            smoothed = dip.Gauss(dip_img, sigma)
            gradient = dip.Norm(dip.GradientMagnitude(smoothed))
            gradient = dip.Opening(dip.Closing(gradient, morph_radius), morph_radius)
            labels = dip.Watershed(
                gradient,
                connectivity=1,
                maxDepth=3,
                flags={"correct", "labels"},
            )
            labels = dip.SmallObjectsRemove(labels, min_size)
            if remove_edge:
                labels = dip.EdgeObjectsRemove(labels)
            msr = dip.MeasurementTool.Measure(
                labels, labels,
                ["Size", "Roundness", "Radius", "Gravity"],
            )
        except Exception:
            continue

        for obj in msr.Objects():
            values = msr[obj]
            roundness = float(values["Roundness"][0])
            radius = float(values["Radius"][0])
            gravity = values["Gravity"]
            cx = float(gravity[0])
            cy = float(gravity[1])
            if radius < min_radius or radius > max_radius:
                continue
            if roundness < 0.55:
                continue

            center = (x0 + cx, y0 + cy)
            radius_score = 1.0
            if expected_radius_px is not None and expected_radius_px > 0:
                radius_score = max(
                    0.0,
                    1.0 - abs(radius - expected_radius_px) / max(expected_radius_px * 0.45, 3.0),
                )
            center_score = 1.0
            if expected_center is not None:
                dist = float(np.linalg.norm(np.array(center) - np.array(expected_center)))
                center_score = max(0.0, 1.0 - dist / max(expected_radius_px or radius, 5.0) / 4.0)
            confidence = float(max(0.0, min(
                1.0,
                0.68 * roundness + 0.22 * radius_score + 0.10 * center_score,
            )))
            detections.append(CircleDetection(
                center=center,
                radius=radius,
                confidence=confidence,
                method=f"dip_watershed_s{sigma:g}_m{morph_radius}",
            ))

    return detections


def _circle_detection_score(
    det: CircleDetection,
    x: int,
    y: int,
    w: int,
    h: int,
    expected_radius_px: Optional[float] = None,
    expected_center: Optional[tuple[float, float]] = None,
) -> float:
    center = np.array(det.center, dtype=np.float64)
    ref_center = (
        np.array(expected_center, dtype=np.float64)
        if expected_center is not None
        else np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    )
    min_dim = max(1.0, float(min(w, h)))
    center_score = 1.0 - min(1.0, float(np.linalg.norm(center - ref_center)) / min_dim)

    radius_score = 1.0
    if expected_radius_px is not None and expected_radius_px > 0:
        radius_score = max(
            0.0,
            1.0 - abs(det.radius - expected_radius_px) / max(expected_radius_px * 0.55, 3.0),
        )

    method_bonus = 0.06 if det.method.startswith("dip_watershed") else 0.0
    return 0.58 * det.confidence + 0.24 * radius_score + 0.18 * center_score + method_bonus


def auto_config_path(image_path: str, group_id: str) -> str:
    """Path for reproducible auto-correspondence config."""
    base_dir = Path(image_path).resolve().parent if image_path else Path.cwd()
    safe_group = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in group_id)
    return str(base_dir / f"{safe_group}_auto_correspondence.json")
