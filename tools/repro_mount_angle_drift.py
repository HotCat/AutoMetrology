from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cadviewer.core.config import AppConfig
from cadviewer.calibration.residual_map import residual_map_from_config
from cadviewer.registration.auto_correspondence import (
    _scaled_camera_matrix_for_image,
    undistort_if_calibrated,
)
from cadviewer.ui.calibration_window import _PixelSizeTab


IMAGES = [
    Path("/tmp/cadrefs_camera_capture.png"),
    Path("/tmp/cadrefs_camera_capture2.png"),
]


def _detect_corners(image: np.ndarray, cols: int, rows: int):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        (cols, rows),
        flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    method = "findChessboardCornersSB"
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        method = "findChessboardCorners"
        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
    return found, corners, method


def _spacing_stats(corners: np.ndarray, cols: int, rows: int) -> dict:
    pts = corners.reshape(-1, 2)
    h_dists = []
    v_dists = []
    for r in range(rows):
        for c in range(cols - 1):
            h_dists.append(float(np.linalg.norm(pts[r * cols + c + 1] - pts[r * cols + c])))
    for r in range(rows - 1):
        for c in range(cols):
            v_dists.append(float(np.linalg.norm(pts[(r + 1) * cols + c] - pts[r * cols + c])))
    return {
        "h_mean": float(np.mean(h_dists)),
        "h_std": float(np.std(h_dists)),
        "v_mean": float(np.mean(v_dists)),
        "v_std": float(np.std(v_dists)),
        "avg": float((np.mean(h_dists) + np.mean(v_dists)) / 2.0),
    }


def _grid_vector_stats(corners: np.ndarray, cols: int, rows: int) -> dict:
    pts = corners.reshape(rows, cols, 2).astype(np.float64)
    row_vecs = []
    col_vecs = []
    for r in range(rows):
        row_vecs.append((pts[r, -1] - pts[r, 0]) / float(cols - 1))
    for c in range(cols):
        col_vecs.append((pts[-1, c] - pts[0, c]) / float(rows - 1))
    row_vecs = np.asarray(row_vecs)
    col_vecs = np.asarray(col_vecs)
    row_len = np.linalg.norm(row_vecs, axis=1)
    col_len = np.linalg.norm(col_vecs, axis=1)
    row_ang = np.degrees(np.arctan2(row_vecs[:, 1], row_vecs[:, 0]))
    col_ang = np.degrees(np.arctan2(col_vecs[:, 1], col_vecs[:, 0]))
    return {
        "row_len_mean": float(np.mean(row_len)),
        "row_len_std": float(np.std(row_len)),
        "row_len_minmax": [float(np.min(row_len)), float(np.max(row_len))],
        "col_len_mean": float(np.mean(col_len)),
        "col_len_std": float(np.std(col_len)),
        "col_len_minmax": [float(np.min(col_len)), float(np.max(col_len))],
        "row_angle_mean": float(np.mean(row_ang)),
        "row_angle_std": float(np.std(row_ang)),
        "col_angle_mean": float(np.mean(col_ang)),
        "col_angle_std": float(np.std(col_ang)),
    }


def _solve_pnp(
    corners: np.ndarray,
    image_shape: tuple[int, ...],
    cfg: AppConfig,
    undistorted: bool,
) -> dict:
    cols = cfg.calibration.chessboard_cols
    rows = cfg.calibration.chessboard_rows
    cell_mm = float(cfg.calibration.chessboard_cell_mm)
    lc = cfg.lens_calibration
    camera_matrix = lc.get_camera_matrix()
    dist_coeffs = lc.get_dist_coeffs()
    if camera_matrix is None or dist_coeffs is None:
        return {"ok": False}
    camera_matrix = _scaled_camera_matrix_for_image(camera_matrix, lc, image_shape)
    pose_dist = np.zeros_like(dist_coeffs) if undistorted else dist_coeffs
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32) * cell_mm
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners.reshape(-1, 2).astype(np.float32),
        camera_matrix,
        pose_dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return {"ok": False}
    rotation, _ = cv2.Rodrigues(rvec)
    pitch, roll, yaw = _PixelSizeTab._rotation_to_pitch_roll_yaw(rotation)
    normal_cam = rotation @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    tilt_deg = float(np.degrees(np.arccos(
        np.clip(abs(normal_cam[2]) / max(np.linalg.norm(normal_cam), 1e-12), -1.0, 1.0)
    )))
    return {
        "ok": True,
        "pitch": pitch,
        "roll": roll,
        "yaw": yaw,
        "tilt": tilt_deg,
        "t_norm": float(np.linalg.norm(tvec.reshape(3))),
    }


def _analyze(path: Path, cfg: AppConfig, undistort: bool) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"path": str(path), "error": "read failed"}
    applied = False
    if undistort:
        image, applied = undistort_if_calibrated(image, cfg)
    cols = cfg.calibration.chessboard_cols
    rows = cfg.calibration.chessboard_rows
    found, corners, method = _detect_corners(image, cols, rows)
    result = {
        "path": str(path),
        "space": "undistorted" if undistort else "raw",
        "undistort_applied": bool(applied),
        "shape": list(image.shape),
        "found": bool(found),
        "method": method,
    }
    if not found:
        return result
    pts = corners.reshape(-1, 2)
    ortho = _PixelSizeTab._estimate_orthographic_mount(
        cols,
        rows,
        float(cfg.calibration.chessboard_cell_mm),
        pts,
    )
    result.update({
        "center": [float(v) for v in np.mean(pts, axis=0)],
        "spacing": _spacing_stats(corners, cols, rows),
        "grid_vectors": _grid_vector_stats(corners, cols, rows),
        "pixel_size": float(cfg.calibration.chessboard_cell_mm) / _spacing_stats(corners, cols, rows)["avg"],
        "orthographic": ortho,
        "solvepnp": _solve_pnp(corners, image.shape, cfg, undistorted=undistort),
    })
    return result


def _analyze_residual_corrected(path: Path, cfg: AppConfig) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"path": str(path), "error": "read failed"}
    image, applied = undistort_if_calibrated(image, cfg)
    cols = cfg.calibration.chessboard_cols
    rows = cfg.calibration.chessboard_rows
    found, corners, method = _detect_corners(image, cols, rows)
    result = {
        "path": str(path),
        "space": "undistorted_residual_corrected",
        "undistort_applied": bool(applied),
        "shape": list(image.shape),
        "found": bool(found),
        "method": method,
    }
    residual_map = residual_map_from_config(cfg)
    if not found or residual_map is None:
        result["residual_map"] = bool(residual_map is not None)
        return result
    pts = corners.reshape(-1, 2).astype(np.float64)
    map_w, map_h = residual_map.image_size
    h, w = image.shape[:2]
    scale = np.array([map_w / w, map_h / h], dtype=np.float64)
    inv_scale = np.array([w / map_w, h / map_h], dtype=np.float64)
    corrected = residual_map.correct(pts * scale) * inv_scale
    ortho = _PixelSizeTab._estimate_orthographic_mount(
        cols,
        rows,
        float(cfg.calibration.chessboard_cell_mm),
        corrected,
    )
    result.update({
        "center": [float(v) for v in np.mean(corrected, axis=0)],
        "orthographic": ortho,
        "residual_map_size": [map_w, map_h],
    })
    return result


def _analyze_point_undistorted(path: Path, cfg: AppConfig) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {"path": str(path), "error": "read failed"}
    cols = cfg.calibration.chessboard_cols
    rows = cfg.calibration.chessboard_rows
    found, corners, method = _detect_corners(image, cols, rows)
    result = {
        "path": str(path),
        "space": "raw_detect_then_undistort_points",
        "shape": list(image.shape),
        "found": bool(found),
        "method": method,
    }
    if not found:
        return result
    lc = cfg.lens_calibration
    mtx = lc.get_camera_matrix()
    dist = lc.get_dist_coeffs()
    if mtx is None or dist is None:
        return result
    mtx = _scaled_camera_matrix_for_image(mtx, lc, image.shape)
    pts = cv2.undistortPoints(corners.astype(np.float32), mtx, dist, P=mtx)
    pts = pts.reshape(-1, 2).astype(np.float64)
    ortho = _PixelSizeTab._estimate_orthographic_mount(
        cols,
        rows,
        float(cfg.calibration.chessboard_cell_mm),
        pts,
    )
    result.update({
        "center": [float(v) for v in np.mean(pts, axis=0)],
        "orthographic": ortho,
        "grid_vectors": _grid_vector_stats(pts.reshape(-1, 1, 2), cols, rows),
    })
    return result


def main() -> None:
    cfg = AppConfig.load()
    print(json.dumps({
        "config": {
            "cols": cfg.calibration.chessboard_cols,
            "rows": cfg.calibration.chessboard_rows,
            "cell_mm": cfg.calibration.chessboard_cell_mm,
            "pixel_size_mm": cfg.pixel_size_mm,
            "lens_image_size": cfg.lens_calibration.get_image_size(),
            "reprojection_error": cfg.lens_calibration.reprojection_error,
        }
    }, indent=2))
    results = []
    for path in IMAGES:
        for undistort in (False, True):
            results.append(_analyze(path, cfg, undistort))
        results.append(_analyze_point_undistorted(path, cfg))
        results.append(_analyze_residual_corrected(path, cfg))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
