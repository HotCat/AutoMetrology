"""
ResidualDistortionMap — thin-plate-spline correction for sub-pixel metrology.

After standard OpenCV calibration (cv2.calibrateCamera + cv2.undistort),
small residual geometric errors remain. This map models them as a smooth
vector field sampled at calibration-grid nodes and interpolated via TPS.

Usage:
    # Build from calibration data
    map = ResidualDistortionMap()
    map.build(sample_points, corrections)
    corrected = map.correct(raw_points)

    # Serialize
    data = map.to_dict()
    map2 = ResidualDistortionMap.from_dict(data)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    from scipy.interpolate import RBFInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ResidualDistortionMap:
    """Thin-plate-spline interpolation of residual distortion vectors.

    Stores sample points (undistorted pixel coordinates where residuals are
    known) and their (dx, dy) correction vectors. Uses scipy's RBFInterpolator
    with the thin_plate_spline kernel for smooth, physically-motivated
    interpolation across the entire image.

    The map operates in undistorted-pixel space — after OpenCV undistortion
    has been applied to the image.
    """

    def __init__(self) -> None:
        self._sample_points: Optional[np.ndarray] = None  # Nx2
        self._corrections: Optional[np.ndarray] = None     # Nx2 (dx, dy)
        self._interp_x = None
        self._interp_y = None
        self._built = False
        self._image_size: tuple[int, int] = (0, 0)  # (width, height)

    @property
    def is_built(self) -> bool:
        return self._built

    @property
    def n_samples(self) -> int:
        return len(self._sample_points) if self._sample_points is not None else 0

    @property
    def image_size(self) -> tuple[int, int]:
        return self._image_size

    def build(
        self,
        sample_points: np.ndarray,
        corrections: np.ndarray,
        image_size: tuple[int, int] = (0, 0),
        smoothing: float = 0.0,
    ) -> None:
        """Build the TPS interpolation model from calibration data.

        Args:
            sample_points: Nx2 array of (x, y) positions in undistorted pixels
                where residual errors are known (from calibration grid corners).
            corrections: Nx2 array of (dx, dy) correction vectors.
                dx = ideal_x - measured_x
                dy = ideal_y - measured_y
                So corrected = measured + correction.
            image_size: (width, height) of the image in pixels.
            smoothing: TPS smoothing parameter. 0 = exact interpolation.
                Small values (0.01-0.1) can help with noisy calibration data.
        """
        if not HAS_SCIPY:
            raise RuntimeError("scipy is required for ResidualDistortionMap")

        self._sample_points = np.asarray(sample_points, dtype=np.float64)
        self._corrections = np.asarray(corrections, dtype=np.float64)
        self._image_size = image_size

        if len(self._sample_points) < 10:
            raise ValueError(
                f"Need at least 10 sample points, got {len(self._sample_points)}"
            )

        # Build two separate TPS interpolators for dx and dy
        self._interp_x = RBFInterpolator(
            self._sample_points,
            self._corrections[:, 0],
            kernel="thin_plate_spline",
            smoothing=smoothing,
        )
        self._interp_y = RBFInterpolator(
            self._sample_points,
            self._corrections[:, 1],
            kernel="thin_plate_spline",
            smoothing=smoothing,
        )
        self._built = True

    def correct(self, points: np.ndarray) -> np.ndarray:
        """Apply residual distortion correction to pixel coordinates.

        Args:
            points: Nx2 array of (x, y) in undistorted pixel coordinates.

        Returns:
            Nx2 array of corrected (x, y) positions.
        """
        if not self._built:
            return np.asarray(points, dtype=np.float64)

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        dx = self._interp_x(pts)
        dy = self._interp_y(pts)
        corrected = pts.copy()
        corrected[:, 0] += dx
        corrected[:, 1] += dy
        return corrected

    def correction_vectors(self, points: np.ndarray) -> np.ndarray:
        """Return the (dx, dy) correction vectors without applying them.

        Useful for visualization.
        """
        if not self._built:
            return np.zeros_like(points)

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        dx = self._interp_x(pts)
        dy = self._interp_y(pts)
        return np.column_stack([dx, dy])

    def evaluate_rms_error(
        self,
        test_points: np.ndarray,
        test_corrections: np.ndarray,
    ) -> float:
        """Compute RMS correction error on a test set.

        Compares the TPS-predicted correction to the actual measured
        correction at test_points.
        """
        if not self._built:
            return 0.0
        pts = np.atleast_2d(np.asarray(test_points, dtype=np.float64))
        actual = np.atleast_2d(np.asarray(test_corrections, dtype=np.float64))
        predicted = self.correction_vectors(pts)
        diff = actual - predicted
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "sample_points": (self._sample_points.tolist()
                              if self._sample_points is not None else []),
            "corrections": (self._corrections.tolist()
                            if self._corrections is not None else []),
            "image_size": list(self._image_size),
            "built": self._built,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResidualDistortionMap:
        """Deserialize from a dict (as loaded from JSON)."""
        m = cls()
        pts = data.get("sample_points", [])
        corrs = data.get("corrections", [])
        img_size = tuple(data.get("image_size", [0, 0]))

        if pts and corrs and data.get("built", False):
            pts_arr = np.array(pts, dtype=np.float64)
            corr_arr = np.array(corrs, dtype=np.float64)
            # Rebuild with light smoothing for numerical stability on reload
            m.build(pts_arr, corr_arr, img_size, smoothing=0.001)
        return m
