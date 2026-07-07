"""Robust OpenCV chessboard detection helpers for calibration images."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit grayscale image suitable for OpenCV chessboard finders."""
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.dtype == np.uint8:
        return gray
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def detect_chessboard_corners(
    image: np.ndarray,
    cols: int,
    rows: int,
    max_detect_size: int = 1800,
) -> tuple[Optional[np.ndarray], bool, str]:
    """Detect chessboard inner corners with fallbacks for full-res captures."""
    if not HAS_CV2:
        return None, False, "opencv_unavailable"

    gray = to_gray_uint8(image)
    pattern = (int(cols), int(rows))

    corners = _detect_standard(gray, pattern)
    if corners is not None:
        return corners, True, "findChessboardCorners"

    scaled, scale = _scaled_for_detection(gray, max_detect_size)
    if scale < 1.0:
        corners = _detect_standard(scaled, pattern)
        if corners is not None:
            corners = corners.astype(np.float32) / np.float32(scale)
            return _refine_original(gray, corners), True, "findChessboardCorners_scaled"

        corners = _detect_sb(scaled, pattern)
        if corners is not None:
            corners = corners.astype(np.float32) / np.float32(scale)
            return _refine_original(gray, corners), True, "findChessboardCornersSB_scaled"

    if scale >= 1.0:
        corners = _detect_sb(gray, pattern)
        if corners is not None:
            return _refine_original(gray, corners), True, "findChessboardCornersSB"

    return None, False, "not_found"


def _detect_standard(
    gray: np.ndarray,
    pattern: tuple[int, int],
) -> Optional[np.ndarray]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return None
    return _refine_original(gray, corners)


def _detect_sb(
    gray: np.ndarray,
    pattern: tuple[int, int],
) -> Optional[np.ndarray]:
    finder = getattr(cv2, "findChessboardCornersSB", None)
    if finder is None:
        return None
    flags = int(getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0))
    flags |= int(getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0))
    flags |= int(getattr(cv2, "CALIB_CB_ACCURACY", 0))
    found, corners = finder(gray, pattern, flags=flags)
    if not found:
        return None
    return corners.astype(np.float32)


def _scaled_for_detection(
    gray: np.ndarray,
    max_detect_size: int,
) -> tuple[np.ndarray, float]:
    h, w = gray.shape[:2]
    longest = max(w, h)
    if longest <= max_detect_size:
        return gray, 1.0
    scale = float(max_detect_size) / float(longest)
    resized = cv2.resize(
        gray,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _refine_original(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    corners = corners.astype(np.float32)
    try:
        return cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
    except cv2.error:
        return corners
