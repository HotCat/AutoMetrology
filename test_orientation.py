"""Debug script to trace the 180° orientation bug in fiducial registration."""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from cadviewer.models.repository import FeatureRepository
from cadviewer.parsers.dxf_importer import DXFImporter
from cadviewer.models.registration import RegistrationManager
from cadviewer.registration.pipeline import RegistrationPipeline
from cadviewer.registration.strategy import FiducialStrategy, RegistrationContext
from cadviewer.registration import affine_solver

# Load DXF
print("Loading DXF...")
repo = DXFImporter().import_file("xintai.dxf")
print(f"  Features: {repo.count()}")

# Count feature types
counts = repo.type_counts()
for ft, c in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {ft.name}: {c}")

# Create registration context
reg_manager = RegistrationManager(repo)
groups = reg_manager.save_groups()
reg_manager.restore_groups(groups)
group_id = groups[0]["group_id"] if groups else "default"

ctx = RegistrationContext(
    repo=repo,
    reg_manager=reg_manager,
    group_id=group_id,
    image_path="problems/cadrefs_camera_capture.png",
    pixel_size_mm=0.1162,
)

# Run fiducial registration
print("\n" + "=" * 60)
print("RUNNING FIDUCIAL REGISTRATION")
print("=" * 60)

strategy = FiducialStrategy()

t0 = time.time()
result = strategy.run_coarse(ctx)
t1 = time.time()
print(f"\nCoarse registration took {t1 - t0:.2f}s")

T = result.transform
params = affine_solver.extract_params(T)
print(f"\nResult transform:")
print(f"  Scale: {params['scale_x']:.6f}")
print(f"  Rotation: {params['rotation_deg']:.2f}°")
print(f"  Translation: ({params['tx']:.2f}, {params['ty']:.2f}) mm")
print(f"  Error: {result.error:.4f}")

# Check if 180° is wrong
print(f"\n  Rotation mod 360: {params['rotation_deg'] % 360:.2f}°")
if abs(params['rotation_deg'] % 360) > 90 and abs(params['rotation_deg'] % 360) < 270:
    print("  *** LIKELY 180° FLIPPED ***")
else:
    print("  Orientation looks correct")

# Now manually test both orientations with edge overlap
print("\n" + "=" * 60)
print("MANUAL ORIENTATION COMPARISON")
print("=" * 60)

from cadviewer.registration.image_extractor import ImageFeatureExtractor
from cadviewer.registration.cad_silhouette import RegistrationContourGenerator
from scipy.spatial import cKDTree

# Load image and extract edges
image = ctx.image_path
img = ImageFeatureExtractor.load_image(image)
print(f"Image: {img.shape[1]}x{img.shape[0]} px")

img_edges = ImageFeatureExtractor.extract_edges(img)
img_edges_world = img_edges.astype(np.float64)
img_edges_world[:, 0] *= ctx.pixel_size_mm
img_edges_world[:, 1] *= -ctx.pixel_size_mm
print(f"Edge points: {len(img_edges_world)}")

# CAD points
sil_gen = RegistrationContourGenerator()
group_obj = reg_manager.get_group(group_id)
features = list(repo._features.values())
cad_points = sil_gen.generate_point_cloud(features, density=0.5)
print(f"CAD points: {len(cad_points)}")

# Score function
def score_edge_overlap(T, cad_pts, img_edges, pixel_size_mm):
    R = T[:2, :2]
    tx, ty = T[0, 2], T[1, 2]
    cad_rotated = cad_pts @ R.T

    # Filter to FOV
    img_w_mm = img.shape[1] * pixel_size_mm
    img_h_mm = img.shape[0] * pixel_size_mm
    test = cad_rotated + np.array([tx, ty])
    in_fov = (test[:, 0] > -10) & (test[:, 0] < img_w_mm + 10) & \
             (test[:, 1] > -img_h_mm - 10) & (test[:, 1] < 10)
    if in_fov.sum() < 5:
        return 0.0, in_fov.sum()

    scoring_pts = cad_rotated[in_fov]
    tree = cKDTree(img_edges)

    # Grid search for best translation
    best_score = -1
    best_t = np.array([tx, ty])
    sigma2 = 2 * 2.0 * 2.0

    for dx in np.arange(-100, 101, 4.0):
        for dy in np.arange(-100, 101, 4.0):
            shifted = scoring_pts + np.array([tx + dx, ty + dy])
            dists, _ = tree.query(shifted)
            s = float(np.mean(np.exp(-dists ** 2 / sigma2)))
            if s > best_score:
                best_score = s
                best_t = np.array([tx + dx, ty + dy])

    # Fine search
    for dx in np.arange(-4, 4.5, 0.5):
        for dy in np.arange(-4, 4.5, 0.5):
            shifted = scoring_pts + best_t + np.array([dx, dy])
            dists, _ = tree.query(shifted)
            s = float(np.mean(np.exp(-dists ** 2 / sigma2)))
            if s > best_score:
                best_score = s

    return best_score, in_fov.sum()

# Test original orientation
print("\nScoring ORIGINAL orientation...")
s_orig, n_orig = score_edge_overlap(T, cad_points, img_edges_world, ctx.pixel_size_mm)
print(f"  Score: {s_orig:.6f} ({n_orig} CAD pts in FOV)")

# Construct 180° candidate
T_180 = T.copy()
T_180[:2, :2] = -T[:2, :2]
p180 = affine_solver.extract_params(T_180)
print(f"\n180° candidate: rot={p180['rotation_deg']:.2f}°, scale={p180['scale_x']:.6f}")

print("Scoring 180° orientation...")
s_180, n_180 = score_edge_overlap(T_180, cad_points, img_edges_world, ctx.pixel_size_mm)
print(f"  Score: {s_180:.6f} ({n_180} CAD pts in FOV)")

print(f"\nResult: original={s_orig:.6f}, 180°={s_180:.6f}")
if s_180 > s_orig:
    print("  → 180° is BETTER (would be selected)")
else:
    print("  → Original is BETTER (selected)")

# Check if CAD silhouette is actually asymmetric
print("\n" + "=" * 60)
print("CAD SYMMETRY CHECK")
print("=" * 60)
cad_cx = cad_points[:, 0].mean()
cad_cy = cad_points[:, 1].mean()
print(f"CAD centroid: ({cad_cx:.1f}, {cad_cy:.1f}) mm")

# Reflect CAD points 180° around centroid
cad_flipped = 2 * np.array([cad_cx, cad_cy]) - cad_points
# Check overlap: how similar is the flipped point cloud?
tree_orig = cKDTree(cad_points)
dists, _ = tree_orig.query(cad_flipped)
mean_dist = float(np.mean(dists))
print(f"Mean distance CAD ↔ CAD_flipped: {mean_dist:.2f} mm")
if mean_dist < 2.0:
    print("  *** CAD is nearly 180°-SYMMETRIC! Edge overlap cannot disambiguate! ***")
elif mean_dist < 10.0:
    print("  CAD is partially symmetric — marginal disambiguation")
else:
    print("  CAD is asymmetric — edge overlap should work")
