#!/usr/bin/env python
"""
run_diagnostics.py — Run the full metrology diagnostic framework.

Reproduces the measurement pipeline:
  1. Load xintai.dxf → FeatureRepository
  2. Load camera image → undistort (if lens calibrated)
  3. Run fiducial registration → get affine
  4. Run measurement pipeline with/without TPS
  5. Run all 8 diagnostic phases

Usage:
  cd /home/hotcat/Downloads/cadrefs
  python run_diagnostics.py
"""

import sys
import os
import json
import math
import numpy as np
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

import cv2

# ── Load config ──
config_path = Path.home() / ".config" / "cadviewer" / "settings.json"
config = json.loads(config_path.read_text())
pixel_size_mm = config["pixel_size_mm"]
lens_cfg = config.get("lens_calibration", {})
cal_cfg = config.get("calibration", {})

print(f"Config: pixel_size_mm = {pixel_size_mm}")
print(f"Lens calibrated: {lens_cfg.get('calibrated', False)}")
print(f"Residual map: {len(lens_cfg.get('residual_map', {}).get('sample_points', []))} samples")

# ── Load DXF ──
from cadviewer.parsers.dxf_importer import DXFImporter

dxf_path = "/home/hotcat/Downloads/cadrefs/xintai.dxf"
if not os.path.exists(dxf_path):
    dxf_path = "/home/hotcat/Downloads/cadrefs/#xintai.dxf"

print(f"\nLoading DXF: {dxf_path}")
importer = DXFImporter()
repo = importer.import_file(dxf_path)
print(f"  Features: {repo.count()}")

# Compute CAD bounding box for diagnostics
all_feats = list(repo._features.values())
if all_feats:
    xs, ys = [], []
    for f in all_feats:
        g = f.geometry
        if f.feature_type.name == "LINE":
            xs.extend([g["x1"], g["x2"]])
            ys.extend([g["y1"], g["y2"]])
        elif f.feature_type.name in ("CIRCLE", "ARC"):
            r = g.get("radius", 0)
            xs.extend([g["cx"] - r, g["cx"] + r])
            ys.extend([g["cy"] - r, g["cy"] + r])
    cad_bbox = (min(xs), min(ys), max(xs), max(ys))
    print(f"  CAD bbox: ({cad_bbox[0]:.1f}, {cad_bbox[1]:.1f}) - ({cad_bbox[2]:.1f}, {cad_bbox[3]:.1f})")
    print(f"  CAD extent: {cad_bbox[2]-cad_bbox[0]:.1f} x {cad_bbox[3]-cad_bbox[1]:.1f} mm")
else:
    cad_bbox = None

# ── Load image ──
image_path = "/home/hotcat/Downloads/cadrefs/problems/cadrefs_camera_capture.png"
print(f"\nLoading image: {image_path}")
image_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
if image_raw is None:
    print("ERROR: Could not load image")
    sys.exit(1)
print(f"  Image: {image_raw.shape[1]}x{image_raw.shape[0]} px")

# ── Undistort image if lens calibrated ──
camera_matrix = None
dist_coeffs = None
image_undistorted = False

if lens_cfg.get("calibrated") and lens_cfg.get("camera_matrix"):
    camera_matrix = np.array(lens_cfg["camera_matrix"], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.array(lens_cfg["dist_coeffs"], dtype=np.float64)
    print(f"\nUndistorting image with lens calibration...")
    print(f"  Camera matrix fx={camera_matrix[0,0]:.2f} fy={camera_matrix[1,1]:.2f}")
    print(f"  Dist coeffs: {dist_coeffs}")
    image = cv2.undistort(image_raw, camera_matrix, dist_coeffs)
    image_undistorted = True
    print(f"  Undistorted: {image.shape[1]}x{image.shape[0]} px")
else:
    image = image_raw.copy()
    print("  No lens calibration — using raw image")

# ── Load residual map ──
from cadviewer.calibration.residual_map import ResidualDistortionMap

residual_map = None
rm_data = lens_cfg.get("residual_map", {})
if rm_data.get("built") and rm_data.get("sample_points"):
    try:
        residual_map = ResidualDistortionMap.from_dict(rm_data)
        print(f"\nResidual map loaded: {residual_map.n_samples} samples, "
              f"image_size={residual_map.image_size}")

        # Quick analysis of TPS correction magnitudes
        w, h = residual_map.image_size
        sample_pts = []
        for x in np.linspace(100, w - 100, 30):
            for y in np.linspace(100, h - 100, 20):
                sample_pts.append([x, y])
        sample_pts = np.array(sample_pts)
        corrections = residual_map.correction_vectors(sample_pts)
        mags = np.sqrt(corrections[:, 0] ** 2 + corrections[:, 1] ** 2)
        print(f"  TPS correction magnitude: mean={mags.mean():.3f}, max={mags.max():.3f} px")
        print(f"  TPS correction magnitude: mean={mags.mean()*pixel_size_mm:.3f}, "
              f"max={mags.max()*pixel_size_mm:.3f} mm")
    except Exception as e:
        print(f"  Failed to load residual map: {e}")
        residual_map = None

# ── Run fiducial registration ──
from cadviewer.models.registration import RegistrationManager
from cadviewer.registration.pipeline import RegistrationPipeline
from cadviewer.registration.strategy import FiducialStrategy

print("\n" + "=" * 60)
print("RUNNING FIDUCIAL REGISTRATION")
print("=" * 60)

reg_manager = RegistrationManager(repo)
# Create a group with all features (or use saved groups)
groups_data = config.get("registration_groups", [])
if groups_data:
    reg_manager.restore_groups(groups_data)
    print(f"Restored {reg_manager.group_count()} groups")
else:
    # Create a default group with all features
    group = reg_manager.create_group("All Features")
    for feat in repo.all_features():
        if feat.feature_type.name in ("LINE", "CIRCLE", "ARC"):
            reg_manager.add_feature_to_group(group.group_id, feat.feature_id)
    print(f"Created default group with {group.feature_count} features")

# Use the first group
groups = reg_manager.all_groups()
if not groups:
    print("ERROR: No registration groups")
    sys.exit(1)
group = groups[0]
print(f"Using group: {group.name} ({group.feature_count} features)")

pipeline = RegistrationPipeline(repo, reg_manager)
pipeline.set_strategy_by_key("fiducial")

try:
    result = pipeline.run_full(
        image_path,
        group.group_id,
        pixel_size_mm,
        anchor_handles=[],
    )

    T_reg = result["transform"]
    print(f"\nRegistration result:")
    from cadviewer.registration import affine_solver
    params = affine_solver.extract_params(T_reg)
    print(f"  scale={params['scale_x']:.6f}")
    print(f"  rotation={params['rotation_deg']:.4f} deg")
    print(f"  tx={params['tx']:.2f}, ty={params['ty']:.2f}")

    # Compute image affine (pixel → CAD world) — same as registration_panel._compute_image_affine
    T_pixel_to_imgworld = np.array([
        [pixel_size_mm,  0,  0],
        [0,  -pixel_size_mm,  0],
        [0,   0,   1],
    ], dtype=np.float64)
    T_imgworld_to_cad = np.linalg.inv(T_reg)
    affine = T_imgworld_to_cad @ T_pixel_to_imgworld

    affine_params = affine_solver.extract_params(affine)
    print(f"\nImage affine (pixel → CAD world):")
    print(f"  scale={affine_params['scale_x']:.6f}")
    print(f"  rotation={affine_params['rotation_deg']:.4f} deg")

except Exception as e:
    print(f"Registration failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nFalling back to identity affine")
    affine = np.eye(3, dtype=np.float64)
    affine[0, 0] = pixel_size_mm
    affine[1, 1] = pixel_size_mm

# ── Detect chessboard corners for calibration validation ──
chessboard_image_path = cal_cfg.get("chessboard_image_path", "")
corners_raw = None
cols = cal_cfg.get("chessboard_cols", 11)
rows = cal_cfg.get("chessboard_rows", 8)
cell_mm = cal_cfg.get("chessboard_cell_mm", 21.0)

if chessboard_image_path and os.path.exists(chessboard_image_path):
    print(f"\nDetecting chessboard corners in: {chessboard_image_path}")
    cb_img = cv2.imread(chessboard_image_path, cv2.IMREAD_GRAYSCALE)
    if cb_img is not None:
        found, corners = cv2.findChessboardCorners(
            cb_img, (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if found:
            corners = cv2.cornerSubPix(
                cb_img, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            corners_raw = corners.reshape(-1, 2)
            print(f"  Found {len(corners_raw)} corners ({cols}x{rows})")

            # Compute empirical pixel size from chessboard
            h_dists = []
            for r in range(rows):
                for c in range(cols - 1):
                    idx1 = r * cols + c
                    idx2 = r * cols + c + 1
                    dx = corners_raw[idx2, 0] - corners_raw[idx1, 0]
                    dy = corners_raw[idx2, 1] - corners_raw[idx1, 1]
                    h_dists.append(math.sqrt(dx*dx + dy*dy))
            mean_inter_corner = np.mean(h_dists)
            empirical_px_mm = cell_mm / mean_inter_corner
            print(f"  Mean inter-corner spacing: {mean_inter_corner:.4f} px")
            print(f"  Empirical pixel size: {empirical_px_mm:.6f} mm/px")
            print(f"  Config pixel size:    {pixel_size_mm:.6f} mm/px")
            print(f"  Difference:           {(empirical_px_mm - pixel_size_mm)/pixel_size_mm*100:+.4f}%")
        else:
            print("  Chessboard corners NOT found")
    else:
        print("  Could not load chessboard image")
else:
    print(f"\nNo chessboard image available (path: {chessboard_image_path})")

# ══════════════════════════════════════════════════════════════════
# RUN THE DIAGNOSTIC FRAMEWORK
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STARTING METROLOGY DIAGNOSTIC FRAMEWORK")
print("=" * 70)

from cadviewer.diagnostics.runner import DiagnosticRunner

runner = DiagnosticRunner(
    repo=repo,
    affine=affine,
    pixel_size_mm=pixel_size_mm,
    image=image,
    image_path=image_path,
    residual_map=residual_map,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    image_undistorted=image_undistorted,
    chessboard_cols=cols,
    chessboard_rows=rows,
    chessboard_cell_mm=cell_mm,
    corners_raw=corners_raw,
    cad_bbox=cad_bbox,
)

result = runner.run_all()

# ── Save and print ──
output_dir = "/home/hotcat/Downloads/cadrefs/diagnostics_output"
runner.save_reports(output_dir)
print("\n" + "=" * 70)
print("FULL SUMMARY")
print("=" * 70)
runner.print_summary()

print(f"\nReports saved to: {output_dir}/")
