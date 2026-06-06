"""
Coordinate correction models for telecentric imaging.

Replaces TPS-based residual distortion with simpler, more stable models:
  - Homography: handles planar projective distortion (camera tilt)
  - Affine: handles linear scale/rotation/shear errors

For telecentric lens + planar workpiece, the geometric errors are primarily
affine or projective, NOT nonlinear barrel/pincushion distortion. TPS is
overkill and introduces scale instability.

These models transform pixel coordinates → corrected coordinates directly,
without warping the source image. This keeps the scale calibration stable.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class CalibrationMetadata:
    """Metadata about the calibration process."""

    image_count: int = 0
    corner_count: int = 0
    chessboard_cols: int = 0
    chessboard_rows: int = 0
    cell_mm: float = 0.0
    rms_error_px: float = 0.0
    max_error_px: float = 0.0
    mean_error_px: float = 0.0
    calibrated: bool = False
    calibration_date: str = ""
    image_size: Tuple[int, int] = (0, 0)  # (width, height)


@dataclass
class HomographyCalibrationModel:
    """Homography-based coordinate correction.

    Handles planar projective distortion from camera mounting tilt.
    Transforms pixel coordinates to corrected coordinates using a 3x3
    homography matrix computed from calibration grid correspondences.
    """

    homography: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    inverse_homography: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    metadata: CalibrationMetadata = field(default_factory=CalibrationMetadata)
    model_type: str = "homography"
    calibrated: bool = False

    def build(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        metadata: Optional[CalibrationMetadata] = None,
    ) -> bool:
        """Compute homography from image→world correspondences.

        Args:
            image_points: Nx2 array of detected corner positions (pixels)
            world_points: Nx2 array of known world positions (mm)
            metadata: optional calibration metadata

        Returns:
            True if calibration succeeded
        """
        if not HAS_CV2:
            print("ERROR: cv2 required for homography calibration")
            return False

        if len(image_points) < 4:
            print(f"ERROR: Need at least 4 points, got {len(image_points)}")
            return False

        # Compute homography: image → world
        H, mask = cv2.findHomography(
            image_points.astype(np.float32),
            world_points.astype(np.float32),
            cv2.RANSAC,
            5.0,  # reproj threshold in pixels
        )

        if H is None:
            print("ERROR: cv2.findHomography failed")
            return False

        self.homography = H
        self.inverse_homography = np.linalg.inv(H)
        self.calibrated = True

        if metadata:
            self.metadata = metadata

        # Compute residuals for quality assessment
        projected = self.transform(image_points)
        residuals = projected - world_points
        distances = np.sqrt(np.sum(residuals ** 2, axis=1))

        self.metadata.rms_error_px = float(np.sqrt(np.mean(distances ** 2)))
        self.metadata.max_error_px = float(distances.max())
        self.metadata.mean_error_px = float(distances.mean())
        self.metadata.calibrated = True

        return True

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Apply homography to transform pixel → corrected coordinates.

        Args:
            points: Nx2 array in pixel coordinates

        Returns:
            Nx2 array in corrected coordinates (mm)
        """
        if not self.calibrated:
            return points.copy()

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        n = pts.shape[0]

        # Convert to homogeneous
        homogeneous = np.hstack([pts, np.ones((n, 1))])

        # Apply homography
        transformed = (self.homography @ homogeneous.T).T

        # Convert back from homogeneous
        return transformed[:, :2] / transformed[:, 2:3]

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """Transform corrected → pixel coordinates (inverse homography)."""
        if not self.calibrated:
            return points.copy()

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        n = pts.shape[0]

        homogeneous = np.hstack([pts, np.ones((n, 1))])
        transformed = (self.inverse_homography @ homogeneous.T).T
        return transformed[:, :2] / transformed[:, 2:3]

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "model_type": self.model_type,
            "homography": self.homography.tolist(),
            "inverse_homography": self.inverse_homography.tolist(),
            "calibrated": self.calibrated,
            "metadata": {
                "image_count": self.metadata.image_count,
                "corner_count": self.metadata.corner_count,
                "chessboard_cols": self.metadata.chessboard_cols,
                "chessboard_rows": self.metadata.chessboard_rows,
                "cell_mm": self.metadata.cell_mm,
                "rms_error_px": self.metadata.rms_error_px,
                "max_error_px": self.metadata.max_error_px,
                "mean_error_px": self.metadata.mean_error_px,
                "calibrated": self.metadata.calibrated,
                "calibration_date": self.metadata.calibration_date,
                "image_size": list(self.metadata.image_size),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> HomographyCalibrationModel:
        """Deserialize from dict."""
        model = cls()
        model.homography = np.array(data.get("homography", np.eye(3)), dtype=np.float64)
        model.inverse_homography = np.array(
            data.get("inverse_homography", np.eye(3)), dtype=np.float64
        )
        model.calibrated = data.get("calibrated", False)

        md = data.get("metadata", {})
        model.metadata = CalibrationMetadata(
            image_count=md.get("image_count", 0),
            corner_count=md.get("corner_count", 0),
            chessboard_cols=md.get("chessboard_cols", 0),
            chessboard_rows=md.get("chessboard_rows", 0),
            cell_mm=md.get("cell_mm", 0.0),
            rms_error_px=md.get("rms_error_px", 0.0),
            max_error_px=md.get("max_error_px", 0.0),
            mean_error_px=md.get("mean_error_px", 0.0),
            calibrated=md.get("calibrated", False),
            calibration_date=md.get("calibration_date", ""),
            image_size=tuple(md.get("image_size", [0, 0])),
        )
        return model


@dataclass
class AffineCalibrationModel:
    """Affine-based coordinate correction.

    Handles linear scale/rotation/shear errors without perspective effects.
    More stable than homography when the camera has negligible tilt.
    """

    affine: np.ndarray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float64))
    inverse_affine: np.ndarray = field(default_factory=lambda: np.eye(2, 3, dtype=np.float64))
    metadata: CalibrationMetadata = field(default_factory=CalibrationMetadata)
    model_type: str = "affine"
    calibrated: bool = False

    def build(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        metadata: Optional[CalibrationMetadata] = None,
    ) -> bool:
        """Compute affine transform from image→world correspondences.

        Args:
            image_points: Nx2 array of detected corner positions (pixels)
            world_points: Nx2 array of known world positions (mm)
            metadata: optional calibration metadata

        Returns:
            True if calibration succeeded
        """
        if not HAS_CV2:
            print("ERROR: cv2 required for affine calibration")
            return False

        if len(image_points) < 3:
            print(f"ERROR: Need at least 3 points, got {len(image_points)}")
            return False

        # Compute affine: image → world (2x3 matrix)
        A, inliers = cv2.estimateAffine2D(
            image_points.astype(np.float32),
            world_points.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )

        if A is None:
            print("ERROR: cv2.estimateAffine2D failed")
            return False

        self.affine = A
        self.calibrated = True

        # Compute inverse affine (3x3 for inversion, then extract 2x3)
        A_full = np.vstack([A, np.array([0, 0, 1])])
        A_inv_full = np.linalg.inv(A_full)
        self.inverse_affine = A_inv_full[:2, :]

        if metadata:
            self.metadata = metadata

        # Compute residuals
        projected = self.transform(image_points)
        residuals = projected - world_points
        distances = np.sqrt(np.sum(residuals ** 2, axis=1))

        self.metadata.rms_error_px = float(np.sqrt(np.mean(distances ** 2)))
        self.metadata.max_error_px = float(distances.max())
        self.metadata.mean_error_px = float(distances.mean())
        self.metadata.calibrated = True

        return True

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Apply affine to transform pixel → corrected coordinates."""
        if not self.calibrated:
            return points.copy()

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        # Affine is 2x3: [a b tx; c d ty]
        # points @ [a b; c d]^T + [tx, ty]
        return pts @ self.affine[:, :2].T + self.affine[:, 2]

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """Transform corrected → pixel coordinates (inverse affine)."""
        if not self.calibrated:
            return points.copy()

        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        return pts @ self.inverse_affine[:, :2].T + self.inverse_affine[:, 2]

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "model_type": self.model_type,
            "affine": self.affine.tolist(),
            "inverse_affine": self.inverse_affine.tolist(),
            "calibrated": self.calibrated,
            "metadata": {
                "image_count": self.metadata.image_count,
                "corner_count": self.metadata.corner_count,
                "chessboard_cols": self.metadata.chessboard_cols,
                "chessboard_rows": self.metadata.chessboard_rows,
                "cell_mm": self.metadata.cell_mm,
                "rms_error_px": self.metadata.rms_error_px,
                "max_error_px": self.metadata.max_error_px,
                "mean_error_px": self.metadata.mean_error_px,
                "calibrated": self.metadata.calibrated,
                "calibration_date": self.metadata.calibration_date,
                "image_size": list(self.metadata.image_size),
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> AffineCalibrationModel:
        """Deserialize from dict."""
        model = cls()
        model.affine = np.array(data.get("affine", np.eye(2, 3)), dtype=np.float64)
        model.inverse_affine = np.array(
            data.get("inverse_affine", np.eye(2, 3)), dtype=np.float64
        )
        model.calibrated = data.get("calibrated", False)

        md = data.get("metadata", {})
        model.metadata = CalibrationMetadata(
            image_count=md.get("image_count", 0),
            corner_count=md.get("corner_count", 0),
            chessboard_cols=md.get("chessboard_cols", 0),
            chessboard_rows=md.get("chessboard_rows", 0),
            cell_mm=md.get("cell_mm", 0.0),
            rms_error_px=md.get("rms_error_px", 0.0),
            max_error_px=md.get("max_error_px", 0.0),
            mean_error_px=md.get("mean_error_px", 0.0),
            calibrated=md.get("calibrated", False),
            calibration_date=md.get("calibration_date", ""),
            image_size=tuple(md.get("image_size", [0, 0])),
        )
        return model


class CoordinateTransformer:
    """Unified coordinate transformer supporting multiple correction models.

    Usage:
        transformer = CoordinateTransformer()
        transformer.load_model(model_data, model_type="homography")
        world_coords = transformer.transform(pixel_coords)
    """

    def __init__(
        self,
        pixel_size_mm: float = 0.01,
        model_type: str = "none",
    ) -> None:
        self._pixel_size_mm = pixel_size_mm
        self._model_type = model_type  # "none", "affine", "homography"
        self._homography_model: Optional[HomographyCalibrationModel] = None
        self._affine_model: Optional[AffineCalibrationModel] = None

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def is_calibrated(self) -> bool:
        if self._model_type == "homography":
            return self._homography_model is not None and self._homography_model.calibrated
        elif self._model_type == "affine":
            return self._affine_model is not None and self._affine_model.calibrated
        return False

    @property
    def metadata(self) -> Optional[CalibrationMetadata]:
        if self._model_type == "homography" and self._homography_model:
            return self._homography_model.metadata
        elif self._model_type == "affine" and self._affine_model:
            return self._affine_model.metadata
        return None

    def load_model(self, model_data: dict, model_type: str) -> bool:
        """Load a correction model from serialized data."""
        self._model_type = model_type

        if model_type == "homography":
            self._homography_model = HomographyCalibrationModel.from_dict(model_data)
            self._affine_model = None
            return self._homography_model.calibrated
        elif model_type == "affine":
            self._affine_model = AffineCalibrationModel.from_dict(model_data)
            self._homography_model = None
            return self._affine_model.calibrated
        elif model_type == "none":
            self._homography_model = None
            self._affine_model = None
            return True
        else:
            print(f"WARNING: Unknown model type '{model_type}'")
            return False

    def set_pixel_size(self, pixel_size_mm: float) -> None:
        """Set pixel size for 'none' mode (simple scaling)."""
        self._pixel_size_mm = pixel_size_mm

    def build_from_corners(
        self,
        corners_px: np.ndarray,
        cols: int,
        rows: int,
        cell_mm: float,
        model_type: str,
        image_size: Tuple[int, int] = (0, 0),
        image_count: int = 1,
    ) -> bool:
        """Build correction model from chessboard corner detections.

        Args:
            corners_px: Nx2 detected corner positions (pixels)
            cols: chessboard inner corners in X
            rows: chessboard inner corners in Y
            cell_mm: grid cell size in mm
            model_type: "affine" or "homography"
            image_size: (width, height)
            image_count: number of images used

        Returns:
            True if model built successfully
        """
        n_expected = cols * rows
        if len(corners_px) < n_expected:
            print(f"WARNING: Expected {n_expected} corners, got {len(corners_px)}")
            return False

        # Generate ideal grid coordinates in mm
        world_pts = np.zeros((n_expected, 2), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                world_pts[r * cols + c] = [c * cell_mm, r * cell_mm]

        # Use only the first N corners (may have more from multiple images)
        image_pts = corners_px[:n_expected].copy()

        metadata = CalibrationMetadata(
            image_count=image_count,
            corner_count=n_expected,
            chessboard_cols=cols,
            chessboard_rows=rows,
            cell_mm=cell_mm,
            image_size=image_size,
        )

        self._model_type = model_type

        if model_type == "homography":
            self._homography_model = HomographyCalibrationModel()
            success = self._homography_model.build(image_pts, world_pts, metadata)
            self._affine_model = None
            return success
        elif model_type == "affine":
            self._affine_model = AffineCalibrationModel()
            success = self._affine_model.build(image_pts, world_pts, metadata)
            self._homography_model = None
            return success
        else:
            print(f"ERROR: Cannot build model type '{model_type}'")
            return False

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Transform pixel coordinates to corrected world coordinates.

        This is the key method used by the measurement pipeline:
        fitted pixel coordinates → world coordinates for distance computation.

        Args:
            points: Nx2 in pixel coordinates

        Returns:
            Nx2 in world coordinates (mm)
        """
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))

        if self._model_type == "homography" and self._homography_model:
            return self._homography_model.transform(pts)
        elif self._model_type == "affine" and self._affine_model:
            return self._affine_model.transform(pts)
        else:
            # No correction: simple pixel → world scaling
            return pts * self._pixel_size_mm

    def inverse_transform(self, points: np.ndarray) -> np.ndarray:
        """Transform world coordinates back to pixel coordinates."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))

        if self._model_type == "homography" and self._homography_model:
            return self._homography_model.inverse_transform(pts)
        elif self._model_type == "affine" and self._affine_model:
            return self._affine_model.inverse_transform(pts)
        else:
            return pts / self._pixel_size_mm

    def get_model_dict(self) -> dict:
        """Get serialized model data for persistence."""
        if self._model_type == "homography" and self._homography_model:
            return self._homography_model.to_dict()
        elif self._model_type == "affine" and self._affine_model:
            return self._affine_model.to_dict()
        return {}

    def get_error_summary(self) -> str:
        """Get human-readable error summary."""
        if not self.is_calibrated:
            return "Not calibrated"

        md = self.metadata
        return (
            f"RMS error: {md.rms_error_px:.4f} px, "
            f"Max: {md.max_error_px:.4f} px, "
            f"Mean: {md.mean_error_px:.4f} px"
        )


# ── Calibration Validation ───────────────────────────────────────────

class CalibrationValidator:
    """Validate calibration by measuring known distances on the grid."""

    def __init__(
        self,
        corners_px: np.ndarray,
        cols: int,
        rows: int,
        cell_mm: float,
        transformer: CoordinateTransformer,
    ) -> None:
        self._corners = corners_px
        self._cols = cols
        self._rows = rows
        self._cell_mm = cell_mm
        self._transformer = transformer

    def validate(self) -> Dict[str, float]:
        """Measure known grid distances and return error statistics."""
        results = {}

        # Measure horizontal distances at multiple scales
        for n_cells in [1, 2, 3, 5]:
            known_mm = n_cells * self._cell_mm
            if n_cells >= self._cols:
                continue

            measured_dists = []
            for row in range(self._rows):
                col_start = 0
                col_end = n_cells
                idx_start = row * self._cols + col_start
                idx_end = row * self._cols + col_end

                if idx_end >= len(self._corners):
                    continue

                p1_px = self._corners[idx_start]
                p2_px = self._corners[idx_end]

                # Transform to world coordinates
                p1_world = self._transformer.transform(p1_px.reshape(1, 2))[0]
                p2_world = self._transformer.transform(p2_px.reshape(1, 2))[0]

                dist_mm = math.sqrt(
                    (p2_world[0] - p1_world[0]) ** 2 +
                    (p2_world[1] - p1_world[1]) ** 2
                )
                measured_dists.append(dist_mm)

            if measured_dists:
                mean_measured = sum(measured_dists) / len(measured_dists)
                abs_err = mean_measured - known_mm
                rel_err = abs_err / known_mm * 100

                results[f"{n_cells}_cells"] = {
                    "known_mm": known_mm,
                    "measured_mm": mean_measured,
                    "abs_error_mm": abs_err,
                    "rel_error_pct": rel_err,
                }

        return results

    def summary(self) -> str:
        """Human-readable validation summary."""
        results = self.validate()
        if not results:
            return "No validation results"

        lines = [
            f"{'Distance':>12} {'Known':>10} {'Measured':>12} {'Error':>12} {'Error%':>10}",
            "-" * 58,
        ]

        for key, r in results.items():
            lines.append(
                f"{key:>12} {r['known_mm']:>10.3f} {r['measured_mm']:>12.4f} "
                f"{r['abs_error_mm']:>+12.4f} {r['rel_error_pct']:>+10.3f}%"
            )

        # Compute overall statistics
        all_errors = [r["rel_error_pct"] for r in results.values()]
        mean_err = sum(all_errors) / len(all_errors)
        max_abs_err = max(abs(e) for e in all_errors)

        lines.extend([
            "",
            f"Mean error: {mean_err:+.4f}%",
            f"Max absolute error: {max_abs_err:.4f}%",
        ])

        return "\n".join(lines)