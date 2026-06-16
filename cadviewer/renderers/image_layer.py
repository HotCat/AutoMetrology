"""
ImageLayerRenderer — displays a telecentric product image as canvas background.

Loads PNG/BMP/TIF images via OpenCV, converts to QImage for QPainter,
and renders with an affine transform (pixel → world coords) and adjustable
opacity. The image is drawn under the CAD geometry as a reference layer.
"""

from __future__ import annotations

import math
import tempfile
import os
from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import (
    QImage, QPainter, QPixmap, QTransform, QColor, QPen,
)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageLayerRenderer:
    """Renders a telecentric image as background under CAD geometry."""

    def __init__(self) -> None:
        self._image: Optional[np.ndarray] = None
        self._qimage: Optional[QImage] = None
        self._affine: np.ndarray = np.eye(3, dtype=np.float64)
        self._visible: bool = True
        self._opacity: float = 0.6
        self._path: str = ""
        self._pixel_size_mm: float = 0.01
        self._cached_pixmap: Optional[QPixmap] = None
        self._cache_dirty: bool = True

    def load_image(self, path: str) -> bool:
        """Load image from file (PNG/BMP/TIF)."""
        if not HAS_CV2:
            return False
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return False
        self._image = img
        self._path = path
        self._qimage = self._numpy_to_qimage(img)
        self._affine = np.eye(3, dtype=np.float64)
        self._cache_dirty = True
        return True

    def load_from_array(self, img: np.ndarray) -> bool:
        """
        Load image from a numpy array (BGR format from camera capture).

        Saves the array to a temp file so the registration pipeline can read it
        via cv2.imread(). Returns True on success.
        """
        if not HAS_CV2 or img is None:
            return False

        # Ensure BGR format
        if len(img.shape) == 2:
            # Grayscale → BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # BGRA → BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif img.shape[2] == 1:
            # Single channel → BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self._image = img
        self._affine = np.eye(3, dtype=np.float64)

        # Save to temp file for pipeline compatibility
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "cadrefs_camera_capture.png")
        cv2.imwrite(temp_path, img)
        self._path = temp_path

        self._qimage = self._numpy_to_qimage(img)
        self._cache_dirty = True
        return True

    def set_affine_transform(self, matrix: np.ndarray) -> None:
        """Set 3x3 affine matrix (pixel → world)."""
        self._affine = matrix.copy()
        self._cache_dirty = True

    def set_pixel_size_mm(self, size: float) -> None:
        """Set pixel size for initial placement."""
        self._pixel_size_mm = size

    def set_opacity(self, opacity: float) -> None:
        """Set image opacity (0.0 to 1.0)."""
        self._opacity = max(0.0, min(1.0, opacity))
        self._cache_dirty = True

    def set_visible(self, visible: bool) -> None:
        self._visible = visible

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def path(self) -> str:
        return self._path

    @property
    def has_image(self) -> bool:
        return self._image is not None

    @property
    def affine(self) -> np.ndarray:
        """3x3 affine matrix (pixel → CAD world)."""
        return self._affine

    @property
    def image(self) -> Optional[np.ndarray]:
        """BGR numpy image array, or None."""
        return self._image

    @property
    def image_size(self) -> Tuple[int, int]:
        if self._image is None:
            return (0, 0)
        h, w = self._image.shape[:2]
        return (w, h)

    def get_center_world(self) -> Tuple[float, float]:
        """Image center (w/2, h/2) transformed to world coords via _affine.

        Used as rotation pivot for manual alignment.
        """
        w, h = self.image_size
        if w == 0 or h == 0:
            return (0.0, 0.0)
        center_px = np.array([[w / 2.0, h / 2.0]], dtype=np.float64)
        from ..registration.affine_solver import apply
        center_world = apply(self._affine, center_px)
        return (float(center_world[0, 0]), float(center_world[0, 1]))

    def draw_image(
        self,
        painter: QPainter,
        world_to_screen_fn,
        widget_width: int,
        widget_height: int,
    ) -> None:
        """Draw the image transformed onto the canvas."""
        if not self._visible or self._qimage is None:
            return

        img_w, img_h = self.image_size
        if img_w <= 0 or img_h <= 0:
            return

        # Compose the full pixel -> CAD -> screen matrix.  The stored matrix is
        # allowed to be projective; using only three corners here would silently
        # collapse calibrated homography registration back to an affine display.
        cx = widget_width / 2.0
        cy = widget_height / 2.0
        wx0, wy0 = world_to_screen_fn(0.0, 0.0)
        wx1, wy1 = world_to_screen_fn(1.0, 0.0)
        wx2, wy2 = world_to_screen_fn(0.0, 1.0)
        world_to_screen = np.array([
            [wx1 - wx0, wx2 - wx0, wx0],
            [wy1 - wy0, wy2 - wy0, wy0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        pixel_to_screen = world_to_screen @ np.asarray(self._affine, dtype=np.float64)
        if not np.all(np.isfinite(pixel_to_screen)):
            return

        transform = QTransform()
        transform.setMatrix(
            float(pixel_to_screen[0, 0]),
            float(pixel_to_screen[1, 0]),
            float(pixel_to_screen[2, 0]),
            float(pixel_to_screen[0, 1]),
            float(pixel_to_screen[1, 1]),
            float(pixel_to_screen[2, 1]),
            float(pixel_to_screen[0, 2]),
            float(pixel_to_screen[1, 2]),
            float(pixel_to_screen[2, 2]),
        )

        painter.save()
        painter.setOpacity(self._opacity)
        painter.setTransform(transform, True)
        painter.drawImage(QPointF(0, 0), self._qimage)
        painter.restore()

    @staticmethod
    def _numpy_to_qimage(img: np.ndarray) -> QImage:
        """Convert OpenCV BGR numpy array to QImage."""
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            # Grayscale
            return QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        else:
            # BGR → RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
