# Camera Calibration Manual

## Overview

CAD Viewer supports a comprehensive metrology calibration pipeline designed for industrial measurement accuracy. The calibration system compensates for:

1. **Pixel size** — converts pixels to millimeters using a known reference pattern
2. **Lens distortion** — corrects radial and tangential distortion via OpenCV calibration
3. **Residual geometric errors** — sub-pixel correction via thin-plate-spline interpolation

Calibration results are automatically applied when:
- Loading images from the camera (real-time undistortion)
- Measuring features in the image (edge point correction before world coordinate conversion)

---

## Prerequisites

### Equipment

- **Telecentric lens camera** (recommended for metrology)
- **Chessboard calibration target** — printed on flat, rigid material
- Stable lighting conditions

### Software Requirements

- OpenCV (`cv2`)
- SciPy (for residual distortion map interpolation)

---

## Step 1: Prepare the Chessboard Pattern

### Default Configuration

The application uses an **11×8 inner corner** chessboard with **21 mm cell size**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Columns | 11 | Number of inner corners in X direction |
| Rows | 8 | Number of inner corners in Y direction |
| Cell Size | 21.0 mm | Physical size of each chessboard square |

### Pattern Generation

A PDF generator is included (`gen_chessboard.py`) that creates a printable pattern:

```bash
python gen_chessboard.py
```

This produces `chessboard_11x8_21mm.pdf` optimized for A4 landscape paper (76.4% coverage).

### Printing Guidelines

1. Print on **rigid, flat material** (photo paper, cardstock, or aluminum plate)
2. Ensure **no warping or bending**
3. Measure actual cell size with calipers and update the parameter if different
4. Avoid glossy surfaces that cause reflections

---

## Step 2: Open the Calibration Window

From the main menu:

```
Settings → Camera Calibration...
```

The calibration window has two tabs:

| Tab | Purpose |
|-----|---------|
| Pixel Size Calibration | Compute mm/pixel from a single chessboard photo |
| Lens Calibration | Multi-image calibration for distortion + residual map |

Shared chessboard parameters appear above both tabs:

```
┌─ Chessboard Pattern ──────────────────────────────────┐
│  Cols: [11]   Rows: [8]   Cell: [21.0] mm            │
└───────────────────────────────────────────────────────┘
```

Adjust these if your printed pattern differs from the defaults.

---

## Step 3: Pixel Size Calibration

This tab computes the **millimeters-per-pixel** scale factor from a single chessboard image.

### Procedure

1. **Capture or load a chessboard image**

   - **From Camera**: Connect your camera, ensure live preview shows, click "Capture"
   - **From File**: Click "Browse..." and select a saved chessboard photo

2. **Click "Calibrate Pixel Size"**

   The system will:
   - Detect all 88 corners (11×8) using `cv2.findChessboardCorners`
   - Refine to sub-pixel accuracy with `cv2.cornerSubPix`
   - Compute average inter-corner spacing in pixels
   - Derive mm/px = cell_mm / avg_pixel_spacing

3. **Review the result**

   ```
   Detected 11×8 — 42.15 px/cell → 0.4982 mm/px
   ```

4. **Close the window** — pixel size is automatically saved to configuration

### Notes

- Pixel size calibration is **separate from lens calibration**
- Works best with a **flat, fronto-parallel** chessboard view
- If corner detection fails, try:
  - Better lighting (reduce shadows)
  - Adjust camera focus
  - Ensure the full chessboard is visible in the image

---

## Step 4: Lens Calibration

This tab performs full camera calibration including distortion correction and residual error compensation.

### Why Lens Calibration Matters

Telecentric lenses minimize perspective distortion, but still have:
- Minor radial distortion (k1, k2)
- Tangential distortion from lens decentering (p1, p2)
- Manufacturing variations causing sub-pixel geometric errors

The residual distortion map corrects errors that remain **after** standard OpenCV undistortion, improving circle center and line position accuracy.

### Procedure

#### 4.1 Collect Calibration Images

You need **at least 10–20 images** for reliable calibration (minimum 3 required).

**Capture Strategy:**

| Coverage | Importance |
|----------|------------|
| Full FOV | Cover all corners and edges of the image |
| Multiple angles | Slight tilts (±15°) in X, Y, Z rotations |
| Varied positions | Move chessboard across the entire field |
| Good lighting | Avoid shadows and reflections on corners |

**From Camera:**

1. Ensure camera is connected and streaming
2. Position chessboard at different locations/angles
3. Click "Capture Frame" for each position
4. Thumbnails appear with detection status:
   - ✓ Green badge = corners detected
   - ✗ Red badge = corners not found

**From Files:**

1. Click "Add Files..."
2. Select multiple chessboard images
3. Detection runs automatically on each

#### 4.2 Review Collected Images

The thumbnail grid shows all captured images. Remove bad captures with "Remove Selected" or "Clear All".

Status bar shows:
```
Images: 15 | Corners detected: 14
```

#### 4.3 Run Calibration

Click **"Run Calibration"**

The pipeline executes:

1. **OpenCV Calibration**
   - Computes camera matrix (fx, fy, cx, cy)
   - Computes distortion coefficients (k1–k6, p1, p2)
   - Reports RMS reprojection error

2. **Residual Distortion Map**
   - Undistorts all images
   - Re-detects corners in undistorted images
   - Computes ideal grid positions via affine fit
   - Samples residual errors at each corner
   - Builds thin-plate-spline interpolation model

3. **Calibration Report**
   - Before residual correction: RMS error in pixels/mm
   - After residual correction: RMS error in pixels/mm
   - Improvement percentage

#### 4.4 Review Results

```
Reprojection error (RMS): 0.2341 px
Images used: 14
Corners: 1232

Camera Matrix:
  fx = 2456.32   fy = 2458.67
  cx = 1024.00   cy = 768.00

Distortion (5 coefficients):
  k1 = -0.012345
  k2 = 0.002341
  p1 = -0.000123
  p2 = 0.000234
  k3 = 0.000000

Residual Distortion Map:
  Sample points: 1232

Calibration Report:
  Before residual correction:
    RMS: 0.0823 px (0.412 mm)
  After residual correction:
    RMS: 0.0134 px (0.067 mm)
  Improvement: 83.7%
```

#### 4.5 Save to Configuration

Click **"Save to Config"**

All calibration data is persisted to `~/.config/cadviewer/settings.json`:
- Camera matrix and distortion coefficients
- Residual distortion map (sample points + corrections)
- Chessboard parameters

---

## Step 5: Verify Calibration

### Test with Live Camera

1. Close calibration window
2. Click **"Load Image"** → **"From Camera"**
3. Preview shows **undistorted** live feed (calibration applied automatically)
4. Capture a test image

### Test Measurement Accuracy

1. Load your CAD drawing (DXF/DWG)
2. Register the image to CAD (manual or automatic alignment)
3. Create measurement queries:
   ```
   DISTANCE(circle_1.center, circle_2.center)
   ```
4. Evaluate queries — residual correction improves accuracy

---

## Calibration Data Storage

### Configuration File

Location: `~/.config/cadviewer/settings.json`

```json
{
  "pixel_size_mm": 0.01,
  "calibration": {
    "chessboard_cols": 11,
    "chessboard_rows": 8,
    "chessboard_cell_mm": 21.0
  },
  "lens_calibration": {
    "camera_matrix": [2456.32, 0, 1024, 0, 2458.67, 768, 0, 0, 1],
    "dist_coeffs": [-0.012, 0.002, -0.0001, 0.0002, 0],
    "reprojection_error": 0.2341,
    "calibrated": true,
    "image_count": 14,
    "residual_map": {
      "sample_points": [[x1,y1], [x2,y2], ...],
      "corrections": [[dx1,dy1], [dx2,dy2], ...],
      "image_size": [2048, 1536],
      "built": true
    }
  }
}
```

### How Data Is Used

| Component | Application Point |
|-----------|-------------------|
| Camera matrix + dist_coeffs | `cv2.undistort()` on raw camera frames |
| Residual map | Edge point correction in `MeasurementPipeline` |
| Pixel size | Affine transform scaling (pixel → world) |

---

## Best Practices

### Chessboard Capture

- **Cover the full FOV** — edges and corners matter most for distortion
- **Vary orientation** — 10–15° tilts capture more distortion information
- **Avoid motion blur** — use adequate exposure time
- **Consistent lighting** — shadows cause false corner detection
- **Fronto-parallel reference** — include at least one flat-on capture

### Quality Indicators

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| OpenCV RMS | < 0.3 px | 0.3–0.5 px | > 0.5 px |
| Residual RMS (after) | < 0.02 px | 0.02–0.05 px | > 0.05 px |
| Improvement | > 70% | 50–70% | < 50% |

### Re-Calibration

Re-calibrate when:
- Lens is replaced or adjusted
- Focus distance changes significantly
- Measurement errors exceed tolerance
- Camera sensor or mounting is modified

---

## Troubleshooting

### "Corners not found"

**Causes:**
- Chessboard partially outside image bounds
- Poor lighting or shadows
- Motion blur
- Wrong cols/rows parameter

**Solutions:**
- Ensure full chessboard visible
- Add diffuse lighting
- Increase exposure time
- Verify pattern parameters match printed target

### High RMS Error (> 0.5 px)

**Causes:**
- Chessboard not flat (warped)
- Incorrect cell size parameter
- Too few images or poor coverage

**Solutions:**
- Use rigid backing for chessboard
- Measure cell size with calipers
- Capture more images covering edges

### Residual Map Not Built

**Causes:**
- SciPy not installed
- Fewer than 10 corners detected total

**Solutions:**
- Install scipy: `pip install scipy`
- Collect more images with good corner detection

---

## Technical Details

### Pipeline Flow

```
Raw Camera Frame
       │
       ▼
┌─────────────────────┐
│ cv2.undistort()     │  ← camera_matrix + dist_coeffs
│ (OpenCV correction) │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Edge Detection      │  ← Scharr gradient
│ (Circle/Line ROI)   │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Residual Correction │  ← TPS interpolation from residual_map
│ (sub-pixel adjust)  │
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│ Affine Transform    │  ← registration matrix + pixel_size_mm
│ (pixel → world mm)  │
└─────────────────────┘
       │
       ▼
   World Coordinates
```

### Thin-Plate-Spline Interpolation

The residual distortion map uses `scipy.interpolate.RBFInterpolator` with the `thin_plate_spline` kernel:

- **Physically motivated** — minimizes bending energy
- **Exact interpolation** — passes through all sample points (smoothing=0)
- **Smooth extrapolation** — reasonable behavior near image edges
- **Sample points** — chessboard corners in undistorted pixel coordinates
- **Correction vectors** — (dx, dy) = ideal_position - detected_position

---

## Quick Reference

| Action | Menu Path |
|--------|-----------|
| Open calibration window | Settings → Camera Calibration |
| Load calibration image | Pixel Size tab → Browse / Capture |
| Run pixel size calibration | "Calibrate Pixel Size" button |
| Add lens cal images | Lens tab → Capture Frame / Add Files |
| Run lens calibration | "Run Calibration" button |
| Save calibration | "Save to Config" button |

| Parameter | Default | Location |
|-----------|---------|----------|
| Chessboard cols | 11 | Settings → Camera Calibration |
| Chessboard rows | 8 | Settings → Camera Calibration |
| Cell size | 21.0 mm | Settings → Camera Calibration |
| Pixel size | 0.01 mm/px | Load Image dialog |

---

*Generated for CAD Viewer — Industrial Metrology Calibration System*