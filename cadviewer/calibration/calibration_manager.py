"""
CalibrationManager — orchestrates full metrology calibration pipeline.

Pipeline:
  1. Collect chessboard images (camera capture or file load)
  2. Run standard OpenCV calibration (camera matrix + distortion coefficients)
  3. Undistort images, re-detect corners
  4. Compute residual errors (detected vs ideal grid positions)
  5. Generate CalibrationReport with before/after statistics

The coordinate correction is handled by HomographyCalibrationModel and
AffineCalibrationModel (in coordinate_correction.py), not by TPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .chessboard_detection import detect_chessboard_corners, to_gray_uint8
from .report import CalibrationReport

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class _CollectedImage:
    """A chessboard image with detected corners."""
    image: np.ndarray
    corners: Optional[np.ndarray]  # refined corners in ORIGINAL image coords
    detected: bool
    source: str


@dataclass
class CalibrationResult:
    """Complete result from the calibration pipeline."""
    camera_matrix: Optional[np.ndarray] = None
    dist_coeffs: Optional[np.ndarray] = None
    opencv_rms: float = 0.0
    calibration_model: str = "standard"
    calibration_flags: int = 0
    report: Optional[CalibrationReport] = None
    image_count: int = 0
    corner_count: int = 0
    calibrated: bool = False
    rejected_image_count: int = 0
    rejected_sources: list[str] = field(default_factory=list)
    per_image_errors: list[tuple[str, float]] = field(default_factory=list)


class CalibrationManager:
    """Orchestrates the full metrology calibration pipeline.

    Usage:
        mgr = CalibrationManager()
        mgr.add_image(frame, "camera")
        ...
        result = mgr.run_calibration(cols=11, rows=8, cell_mm=21.0)
        if result.calibrated:
            # Use camera_matrix and dist_coeffs for undistortion
    """

    def __init__(self) -> None:
        self._images: list[_CollectedImage] = []
        self._result: Optional[CalibrationResult] = None

    @property
    def result(self) -> Optional[CalibrationResult]:
        return self._result

    def add_image(self, image: np.ndarray, source: str = "") -> None:
        """Add a BGR image for calibration."""
        self._images.append(_CollectedImage(
            image=image, corners=None, detected=False, source=source,
        ))

    def clear(self) -> None:
        """Remove all collected images."""
        self._images.clear()
        self._result = None

    @property
    def image_count(self) -> int:
        return len(self._images)

    @property
    def good_image_count(self) -> int:
        return sum(1 for img in self._images if img.detected)

    def run_calibration(
        self,
        cols: int,
        rows: int,
        cell_mm: float,
        image_size: tuple[int, int] | None = None,
        flags: int = 0,
        model: str = "standard",
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> CalibrationResult:
        """Run the full calibration pipeline.

        1. Detect corners in all images
        2. Run OpenCV calibration
        3. Generate error report

        Args:
            cols: Number of inner corners in X.
            rows: Number of inner corners in Y.
            cell_mm: Grid cell size in mm.
            image_size: (width, height). Auto-detected if None.
            flags: OpenCV cv2.calibrateCamera flags.
            model: Human/config-readable calibration model key.

        Returns:
            CalibrationResult with all outputs.
        """
        if not HAS_CV2:
            return CalibrationResult(calibration_model=model, calibration_flags=flags)

        # Step 1: Detect corners in all images
        total_images = max(len(self._images), 1)
        for pos, entry in enumerate(self._images, start=1):
            gray = self._to_gray(entry.image)
            entry.corners, entry.detected = self._detect_corners(
                gray, cols, rows,
            )
            if progress_callback is not None:
                progress_callback("Detecting chessboard corners", pos, total_images)

        good = [e for e in self._images if e.detected]
        if len(good) < 3:
            return CalibrationResult(
                image_count=len(good),
                calibration_model=model,
                calibration_flags=flags,
            )

        # Step 2: OpenCV calibration
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32) * cell_mm

        if image_size is None:
            h, w = good[0].image.shape[:2]
            image_size = (w, h)

        if progress_callback is not None:
            progress_callback("Solving camera calibration", 0, 1)
        rms, mtx, dist, rvecs, tvecs = self._calibrate_good_images(
            good, objp, image_size, flags,
        )
        per_image_errors = self._per_image_reprojection_errors(
            good, objp, mtx, dist, rvecs, tvecs,
        )
        filtered_good, rejected = self._filter_reprojection_outliers(
            good, per_image_errors,
        )
        if rejected:
            if progress_callback is not None:
                progress_callback("Solving filtered calibration", 0, 1)
            rms, mtx, dist, rvecs, tvecs = self._calibrate_good_images(
                filtered_good, objp, image_size, flags,
            )
            per_image_errors = self._per_image_reprojection_errors(
                filtered_good, objp, mtx, dist, rvecs, tvecs,
            )
            good_for_result = filtered_good
        else:
            good_for_result = good

        result = CalibrationResult(
            camera_matrix=mtx,
            dist_coeffs=dist,
            opencv_rms=rms,
            calibration_model=model,
            calibration_flags=int(flags),
            image_count=len(good_for_result),
            calibrated=True,
            rejected_image_count=len(rejected),
            rejected_sources=[entry.source for entry in rejected],
            per_image_errors=[
                (entry.source, float(error))
                for entry, error in zip(good_for_result, per_image_errors)
            ],
        )

        # Step 3: Generate error report
        try:
            report = self._build_report(
                good_for_result, mtx, dist, cols, rows, cell_mm,
                progress_callback=progress_callback,
            )
            result.report = report
        except Exception as e:
            print(f"Warning: report generation failed: {e}")

        total_corners = sum(
            e.corners.shape[0] for e in good_for_result if e.corners is not None
        )
        result.corner_count = total_corners
        self._result = result
        return result

    @staticmethod
    def _calibrate_good_images(
        good_images: list[_CollectedImage],
        object_template: np.ndarray,
        image_size: tuple[int, int],
        flags: int,
    ):
        object_points = [object_template] * len(good_images)
        image_points = [e.corners.astype(np.float32) for e in good_images]
        return cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None,
            flags=int(flags),
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-6,
            ),
        )

    @staticmethod
    def _per_image_reprojection_errors(
        good_images: list[_CollectedImage],
        object_template: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        rvecs,
        tvecs,
    ) -> list[float]:
        errors: list[float] = []
        for entry, rvec, tvec in zip(good_images, rvecs, tvecs):
            if entry.corners is None:
                errors.append(float("inf"))
                continue
            projected, _ = cv2.projectPoints(
                object_template, rvec, tvec, camera_matrix, dist_coeffs,
            )
            delta = entry.corners.reshape(-1, 2) - projected.reshape(-1, 2)
            errors.append(float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))))
        return errors

    @staticmethod
    def _filter_reprojection_outliers(
        good_images: list[_CollectedImage],
        per_image_errors: list[float],
    ) -> tuple[list[_CollectedImage], list[_CollectedImage]]:
        if len(good_images) < 8 or len(good_images) != len(per_image_errors):
            return good_images, []
        finite = np.array(
            [e for e in per_image_errors if np.isfinite(e)], dtype=np.float64,
        )
        if finite.size < 8:
            return good_images, []
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        robust_sigma = 1.4826 * mad
        threshold = max(0.40, median + max(3.0 * robust_sigma, median * 1.5))
        keep: list[_CollectedImage] = []
        rejected: list[_CollectedImage] = []
        for entry, error in zip(good_images, per_image_errors):
            if np.isfinite(error) and error <= threshold:
                keep.append(entry)
            else:
                rejected.append(entry)
        min_keep = max(3, int(np.ceil(len(good_images) * 0.60)))
        if len(keep) < min_keep:
            return good_images, []
        return keep, rejected

    def _build_report(
        self,
        good_images: list[_CollectedImage],
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        cols: int,
        rows: int,
        cell_mm: float,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> CalibrationReport:
        """Build calibration error report from calibrated images."""
        all_residuals = []

        total_images = max(len(good_images), 1)
        for pos, entry in enumerate(good_images, start=1):
            if entry.corners is not None:
                # Undistort and re-detect
                undistorted = cv2.undistort(
                    entry.image, camera_matrix, dist_coeffs,
                )
                gray = self._to_gray(undistorted)

                corners_undist, found, _method = detect_chessboard_corners(
                    gray, cols, rows,
                )
                if found:
                    detected = corners_undist.reshape(-1, 2)

                    # Compute ideal grid and residuals
                    ideal = self._compute_ideal_grid(detected, cols, rows)
                    residual = ideal - detected
                    all_residuals.append(residual)
            if progress_callback is not None:
                progress_callback("Building calibration report", pos, total_images)

        if not all_residuals:
            raise ValueError("No corners detected in undistorted images")

        residuals = np.vstack(all_residuals)
        h, w = good_images[0].image.shape[:2]

        report = CalibrationReport(
            n_images=len(good_images),
            image_size=(w, h),
            opencv_rms=float(
                np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
            ) if len(residuals) > 0 else 0.0,
        )
        report.compute(
            np.vstack([entry.corners.reshape(-1, 2) for entry in good_images if entry.corners is not None]),
            residuals,
            residuals,  # no correction applied, same residuals
        )
        return report

    @staticmethod
    def _compute_ideal_grid(
        detected: np.ndarray, cols: int, rows: int,
    ) -> np.ndarray:
        """Compute ideal grid positions from detected corners.

        Fits an affine transform from grid indices (col, row) to detected
        (x, y) positions. The ideal positions are the forward projection
        of the grid indices through this affine. This removes the effect
        of board rotation/tilt while preserving local geometric errors.

        Returns ideal positions as Nx2 array.
        """
        n = cols * rows
        if len(detected) != n:
            # Fall back: return detected as-is
            return detected.copy()

        # Grid indices
        grid = np.zeros((n, 2), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                grid[r * cols + c] = [c, r]

        # Fit affine: grid → detected  (using least squares)
        # [x] = [a b tx] [col]
        # [y]   [c d ty] [row]
        #                 [  1]
        A = np.column_stack([grid, np.ones(n)])
        # Solve for x
        coeff_x, _, _, _ = np.linalg.lstsq(A, detected[:, 0], rcond=None)
        coeff_y, _, _, _ = np.linalg.lstsq(A, detected[:, 1], rcond=None)

        ideal_x = A @ coeff_x
        ideal_y = A @ coeff_y
        return np.column_stack([ideal_x, ideal_y])

    @staticmethod
    def _compute_projective_ideal_grid(
        detected: np.ndarray, cols: int, rows: int,
    ) -> Optional[np.ndarray]:
        """Compute per-image ideal corner positions using a homography.

        This removes chessboard pose, perspective, and camera tilt from the
        residual samples. The remaining vector field is the local residual
        after the best projective explanation of that single board image.
        """
        if not HAS_CV2:
            return None
        n = cols * rows
        corners = np.asarray(detected, dtype=np.float64).reshape(-1, 2)
        if len(corners) != n:
            return None

        grid = np.zeros((n, 2), dtype=np.float64)
        for r in range(rows):
            for c in range(cols):
                grid[r * cols + c] = [c, r]

        H, _ = cv2.findHomography(
            grid.astype(np.float32),
            corners.astype(np.float32),
            0,
        )
        if H is None or not np.all(np.isfinite(H)):
            return None
        projected = cv2.perspectiveTransform(
            grid.reshape(-1, 1, 2).astype(np.float32), H,
        )
        return projected.reshape(-1, 2).astype(np.float64)

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        return to_gray_uint8(image)

    @staticmethod
    def _detect_corners(
        gray: np.ndarray, cols: int, rows: int,
    ) -> tuple[Optional[np.ndarray], bool]:
        corners, found, _method = detect_chessboard_corners(gray, cols, rows)
        return corners, found
