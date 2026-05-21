"""
ImageFeatureExtractor — edge detection and geometric feature extraction.

Extracts edges, lines, circles, and contours from grayscale images using
OpenCV. All outputs are in pixel coordinates — world coordinate conversion
happens via the affine transform in the registration pipeline.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional, Tuple

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageFeatureExtractor:
    """Extract geometric features from telecentric product images."""

    def __init__(self) -> None:
        if not HAS_CV2:
            raise RuntimeError(
                "OpenCV (cv2) is required for ImageFeatureExtractor. "
                "Install with: pip install opencv-python"
            )

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """
        Load image as grayscale.

        Returns: 2D uint8 numpy array.
        """
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    @staticmethod
    def extract_edges(
        image: np.ndarray,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> np.ndarray:
        """
        Detect edges using Canny and return edge point coordinates.

        Returns: Nx2 float64 array of (x, y) pixel coordinates.
        """
        edges = cv2.Canny(image, canny_low, canny_high)
        ys, xs = np.where(edges > 0)
        if len(xs) == 0:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])

    @staticmethod
    def extract_contours(
        image: np.ndarray,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> List[np.ndarray]:
        """
        Extract contours from edge image.

        Returns: list of Mx2 float64 arrays (per-contour pixel coords).
        """
        edges = cv2.Canny(image, canny_low, canny_high)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        result = []
        for cnt in contours:
            if len(cnt) >= 5:
                pts = cnt.reshape(-1, 2).astype(np.float64)
                result.append(pts)
        return result

    @staticmethod
    def detect_lines_hough(
        image: np.ndarray,
        resolution: float = 1.0,
        angle_res: float = np.pi / 180,
        threshold: int = 100,
        min_line_length: int = 50,
        max_line_gap: int = 10,
    ) -> List[dict]:
        """
        Detect lines using probabilistic Hough transform.

        Returns: list of dicts with 'x1', 'y1', 'x2', 'y2' (pixel coords).
        """
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(
            edges, resolution, angle_res, threshold,
            minLineLength=min_line_length, maxLineGap=max_line_gap,
        )
        result = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                result.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return result

    @staticmethod
    def detect_circles_hough(
        image: np.ndarray,
        dp: float = 1.0,
        min_dist: float = 50,
        param1: int = 100,
        param2: int = 30,
        min_radius: int = 0,
        max_radius: int = 0,
    ) -> List[dict]:
        """
        Detect circles using Hough circle transform.

        Returns: list of dicts with 'cx', 'cy', 'radius' (pixel coords).
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 2)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
            param1=param1, param2=param2,
            minRadius=min_radius, maxRadius=max_radius,
        )
        result = []
        if circles is not None:
            for c in circles[0]:
                result.append({
                    "cx": float(c[0]),
                    "cy": float(c[1]),
                    "radius": float(c[2]),
                })
        return result

    @staticmethod
    def compute_image_centroid(edges: np.ndarray) -> np.ndarray:
        """Compute centroid of Nx2 edge point array."""
        if len(edges) == 0:
            return np.zeros(2, dtype=np.float64)
        return edges.mean(axis=0)

    @staticmethod
    def extract_edge_points_in_roi(
        edges: np.ndarray,
        roi: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Filter edge points to those within a rectangular ROI.

        roi: (x_min, y_min, x_max, y_max) in pixel coords.
        """
        if len(edges) == 0:
            return np.empty((0, 2), dtype=np.float64)
        x_min, y_min, x_max, y_max = roi
        mask = (
            (edges[:, 0] >= x_min) & (edges[:, 0] <= x_max) &
            (edges[:, 1] >= y_min) & (edges[:, 1] <= y_max)
        )
        return edges[mask]

    @staticmethod
    def fit_line_subpixel(
        edge_points: np.ndarray,
    ) -> Tuple[Optional[dict], float]:
        """
        Subpixel line fitting via cv2.fitLine.

        Returns: ({x1, y1, x2, y2}, residual) or (None, inf).
        """
        if len(edge_points) < 2:
            return None, float("inf")
        pts = edge_points.reshape(-1, 1, 2).astype(np.float32)
        line = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line[0][0], line[1][0], line[2][0], line[3][0]
        # Extend line through the data extent
        ts = []
        for pt in edge_points:
            if abs(vx) > 1e-10:
                ts.append((pt[0] - x0) / vx)
            elif abs(vy) > 1e-10:
                ts.append((pt[1] - y0) / vy)
        if not ts:
            return None, float("inf")
        t_min, t_max = min(ts), max(ts)
        x1 = x0 + vx * t_min
        y1 = y0 + vy * t_min
        x2 = x0 + vx * t_max
        y2 = y0 + vy * t_max
        # Compute residual as mean point-to-line distance
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-10:
            return None, float("inf")
        nx, ny = -dy / length, dx / length
        residuals = np.abs((edge_points[:, 0] - x1) * nx + (edge_points[:, 1] - y1) * ny)
        mean_residual = float(residuals.mean())
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, mean_residual

    @staticmethod
    def fit_circle_subpixel(
        edge_points: np.ndarray,
    ) -> Tuple[Optional[dict], float]:
        """
        Subpixel circle fitting via Kasa algebraic method.

        Solves: [x, y, 1] @ [2*cx, 2*cy, r^2-cx^2-cy^2] = x^2 + y^2

        Returns: ({cx, cy, radius}, residual) or (None, inf).
        """
        if len(edge_points) < 3:
            return None, float("inf")
        x = edge_points[:, 0]
        y = edge_points[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x ** 2 + y ** 2
        try:
            result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None, float("inf")
        cx = result[0] / 2
        cy = result[1] / 2
        r_sq = result[2] + cx ** 2 + cy ** 2
        if r_sq < 0:
            return None, float("inf")
        radius = math.sqrt(r_sq)
        # Compute residual as mean radial deviation
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        residuals = np.abs(dist - radius)
        mean_residual = float(residuals.mean())
        return {"cx": cx, "cy": cy, "radius": radius}, mean_residual
