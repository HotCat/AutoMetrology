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


def detect_circle_in_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> Optional[CircleDetection]:
    """Detect the strongest circular fiducial in a user-specified ROI."""
    if not HAS_CV2 or image is None:
        return None

    x, y, w, h = clamp_roi(roi, image.shape)
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return None

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    detections: list[CircleDetection] = []

    min_dim = min(w, h)
    min_radius = max(3, int(min_dim * 0.04))
    max_radius = max(min_radius + 2, int(min_dim * 0.30))

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(10, min_dim // 3),
        param1=80,
        param2=18,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        for cx, cy, r in np.round(circles[0]).astype(float):
            if r <= 0:
                continue
            # Prefer strong edge support around the circumference.
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.circle(mask, (int(cx), int(cy)), int(r), 255, 2)
            edge = cv2.Canny(gray, 50, 150)
            support = float(np.mean(edge[mask > 0]) / 255.0)
            detections.append(CircleDetection(
                center=(x + cx, y + cy),
                radius=float(r),
                confidence=min(1.0, 0.45 + support),
                method="hough",
            ))

    # Contour fallback works for filled dark or bright fiducials.
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
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if r < min_radius or r > max_radius:
                continue
            fill_ratio = area / (np.pi * r * r) if r > 1e-6 else 0.0
            if fill_ratio < 0.35 or fill_ratio > 1.25:
                continue
            confidence = float(max(0.0, min(1.0, circularity * min(fill_ratio, 1.0))))
            detections.append(CircleDetection(
                center=(x + float(cx), y + float(cy)),
                radius=float(r),
                confidence=confidence,
                method="contour_inv" if invert else "contour",
            ))

    if not detections:
        return None

    roi_center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)

    def score(det: CircleDetection) -> float:
        center = np.array(det.center, dtype=np.float64)
        centrality = 1.0 - min(1.0, float(np.linalg.norm(center - roi_center)) / max(1.0, min_dim))
        return det.confidence * 0.8 + centrality * 0.2

    return max(detections, key=score)


def auto_config_path(image_path: str, group_id: str) -> str:
    """Path for reproducible auto-correspondence config."""
    base_dir = Path(image_path).resolve().parent if image_path else Path.cwd()
    safe_group = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in group_id)
    return str(base_dir / f"{safe_group}_auto_correspondence.json")
