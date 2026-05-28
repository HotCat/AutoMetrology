"""
Product silhouette extraction from telecentric images.

Extracts a robust foreground mask and the largest contour
from telecentric product images for global registration.

Pipeline:
  grayscale → threshold (Otsu / adaptive) → morphology → largest contour
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ProductSilhouetteExtractor:
    """Extract product foreground mask and silhouette from telecentric image."""

    def extract(
        self, image: np.ndarray, invert: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract product silhouette from telecentric image.

        Args:
            image: 2D uint8 grayscale or 3D BGR image
            invert: True if product is bright on dark background

        Returns:
            (mask, contour) where mask is HxW uint8 and contour is Mx2 float64.
            Returns (mask, empty) if no contour found.
        """
        if not HAS_CV2:
            return np.zeros_like(image), np.empty((0, 2), dtype=np.float64)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Binary mask
        mask = self._compute_mask(gray, invert)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Extract external contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return mask, np.empty((0, 2), dtype=np.float64)

        # Largest contour by area
        largest = max(contours, key=cv2.contourArea)
        contour = largest.reshape(-1, 2).astype(np.float64)

        return mask, contour

    def _compute_mask(self, gray: np.ndarray, invert: bool) -> np.ndarray:
        """Compute binary mask with automatic threshold selection.

        Heuristic: the product typically occupies less than 50% of the
        image area. If the initial threshold produces a foreground region
        larger than 50%, the mask is inverted.
        """
        _, mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        if invert:
            mask = cv2.bitwise_not(mask)

        # The product should be the smaller region.
        # If foreground > 50%, invert the mask.
        fg_ratio = np.count_nonzero(mask) / mask.size
        if fg_ratio > 0.5:
            mask = cv2.bitwise_not(mask)
            fg_ratio = 1.0 - fg_ratio

        # If still unreasonable, try adaptive threshold
        if fg_ratio < 0.05 or fg_ratio > 0.95:
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 51, 5,
            )
            if invert:
                mask = cv2.bitwise_not(mask)
            fg_ratio = np.count_nonzero(mask) / mask.size
            if fg_ratio > 0.5:
                mask = cv2.bitwise_not(mask)

        return mask
