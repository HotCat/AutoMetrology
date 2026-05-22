# CAD Inspection Tool — User Manual

## Table of Contents

1. [Overview](#1-overview)
2. [Getting Started](#2-getting-started)
3. [Application Layout](#3-application-layout)
4. [Opening CAD Files](#4-opening-cad-files)
5. [Navigating the Canvas](#5-navigating-the-canvas)
6. [Feature Tree Browser](#6-feature-tree-browser)
7. [Property Inspector](#7-property-inspector)
8. [Registration Groups](#8-registration-groups)
9. [Image Registration](#9-image-registration)
10. [Camera Capture](#10-camera-capture)
11. [Measurement Queries](#11-measurement-queries)
12. [Keyboard Shortcuts Reference](#12-keyboard-shortcuts-reference)
13. [Supported File Formats](#13-supported-file-formats)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Overview

CAD Inspection Tool is a metrology-oriented DXF/DWG viewer designed for machine vision alignment and automatic dimension measurement. It allows you to:

- Load and inspect DXF and DWG CAD drawings
- Browse and select individual geometric features
- Capture images from an industrial camera (MindVision)
- Register captured images to CAD geometry for visual alignment
- Define measurement queries for automated inspection

The application uses a dark theme throughout and requires no persistent configuration.

**Technology**: Built with PySide6 (Qt) and QPainter for high-performance 2D rendering.

---

## 2. Getting Started

### Launching the Application

```bash
python main.py
```

To open a file on startup:

```bash
python main.py path/to/drawing.dxf
python main.py path/to/drawing.dwg
```

### Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| PySide6 | Yes | GUI framework |
| ezdxf | Yes | DXF parsing |
| numpy | Yes | Numerical operations |
| OpenCV | For registration | Image loading and edge detection |
| scipy | For ICP registration | Nearest-neighbor search |
| MindVision SDK | Optional | Industrial camera support |

---

## 3. Application Layout

```
+--------------------------------------------------------------+
|  Menu Bar: [File] [View] [Settings] [Help]                   |
|  Toolbar: [Open DXF] [Import DWG] | [Fit All] | [Pan][Select]|
+----------+----------------------------+----------------------+
| Feature  |                            |  Property            |
| Tree     |   CAD 2D Viewer Canvas     |  Panel               |
| Panel    |                            |                      |
+----------+----------------------------+----------------------+
|  Status Bar                                                  |
+--------------------------------------------------------------+
```

Two additional panels are available as dockable windows:

- **Registration Panel** (View → Registration Panel) — docks on the right
- **Measurement Queries** (View → Query Panel) — docks at the bottom

All panels and the main splitter are resizable by dragging their borders.

---

## 4. Opening CAD Files

### Opening a DXF File

1. Click **Open DXF** in the toolbar, or choose **File → Open DXF...** (Ctrl+O).
2. Select a `.dxf` file from the file dialog.
3. The file is parsed immediately and all features appear in the tree and canvas.

### Importing a DWG File

DWG files require a converter backend (ODA File Converter or libredwg) to be installed.

1. Click **Import DWG** in the toolbar, or choose **File → Import DWG...** (Ctrl+D).
2. Select a `.dwg` file.
3. A progress dialog shows the conversion stages:
   - Detecting DWG version
   - Running the converter
   - Validating the DXF output
   - Loading features
4. On completion, the converted DXF loads automatically.

If the "Import DWG" button is grayed out, no converter is installed. Install one of:

- **ODA File Converter**: https://www.opendesign.com/guestfiles/oda_file_converter
- **libredwg**: `sudo apt install libredwg-utils`

Then configure the path via **Settings → Configure DWG Converter...**.

### Supported DWG Versions

| Version Code | AutoCAD Version |
|-------------|-----------------|
| AC1032 | 2018+ |
| AC1027 | 2013 |
| AC1024 | 2010 |
| AC1021 | 2007 |
| AC1018 | 2004 |
| AC1015 | 2000 |
| AC1014 | R14 |

---

## 5. Navigating the Canvas

The central area displays the CAD drawing rendered on a dark background with an adaptive grid.

### Mouse Controls

| Action | Control |
|--------|---------|
| **Pan** | Middle-button drag or right-button drag |
| **Zoom in** | Scroll wheel up (1.15x, centered on cursor) |
| **Zoom out** | Scroll wheel down (1/1.15x, centered on cursor) |
| **Select feature** | Left-click on a feature |
| **Deselect** | Left-click on empty area |

### Canvas Overlays

- **Grid**: Adaptive dotted grid. Spacing adjusts automatically as you zoom.
- **Origin marker**: Red/green crosshair at coordinate (0, 0).
- **Zoom indicator**: Shows current zoom percentage in the bottom-left corner.
- **Feature highlight**: Selected features glow cyan with an outer halo effect.

### Fit All

Press **Ctrl+F** or click **Fit All** in the toolbar to fit all geometry into the viewport.

---

## 6. Feature Tree Browser

The left panel shows all loaded features organized by type and layer.

### Tree Structure

```
Geometry Features
  ├── Lines (142)
  │     ├── LINE [1A2B3C]
  │     └── LINE [2C3D4E]
  ├── Circles (28)
  ├── Arcs (54)
  ├── Polylines (12)
  └── Splines (8)
Dimensions (36)
Annotations (20)
Layers
  ├── 0 (210)
  └── Outline (50)
Registration Groups
  └── Group 1 (5)
```

### Searching

Type in the **Filter features...** text box at the top to search features by name. Matching is case-insensitive and filters in real time.

### Interacting with Features

- **Click a feature**: Highlights it on the canvas (cyan glow), zooms to fit it, and shows its properties in the Property Panel.
- **Click a group/layer/category node**: Deselects the current feature.
- **Double-click a node**: Expands or collapses it.

### Context Menu

Right-click a feature to open the context menu:

- **Add to Group → New Group...**: Creates a new registration group and adds the feature.
- **Add to Group → [Existing Group]**: Adds the feature to an existing group.

Right-click a group node:

- **Zoom to Group**: Fits the group's bounding box in the canvas.
- **Delete Group**: Removes the group.

### Feature Type Colors

| Type | Color |
|------|-------|
| Lines | White |
| Circles | Cyan |
| Arcs | Yellow |
| Polylines | Green |
| Splines | Magenta |
| Dimensions | Red |
| Annotations | Gray |
| Hatches | Dark gray |
| Points | Orange |
| Leaders | Amber |

---

## 7. Property Inspector

The right panel shows details for the currently selected feature.

### General Properties

| Field | Description |
|-------|-------------|
| Type | Feature type (LINE, CIRCLE, etc.) |
| Layer | DXF layer name |
| DXF Handle | Original DXF entity handle |
| ID | Internal feature identifier |
| Color | DXF color index |

### Geometry Properties (type-dependent)

- **Line**: Start point, end point, length
- **Circle**: Center point, radius
- **Arc**: Center point, radius, start angle, end angle
- **Polyline**: Vertex count, open/closed, extent
- **Spline**: Degree, control points count, fit points count
- **Text**: Text content, position, height, rotation

### Measurement Properties

Reserved for future integration with machine vision measurement. Fields: Nominal, Measured, Deviation, Status (PASS/FAIL).

All values are text-selectable for copy-paste.

---

## 8. Registration Groups

Registration groups are named collections of CAD features used for image-to-CAD alignment. Each group has a unique color.

### Creating Groups

**Method 1**: Right-click a feature in the tree → **Add to Group → New Group...** → enter a name.

**Method 2**: In the Registration Panel, click **New** → enter a name.

### Managing Groups

In the Registration Panel:

- **New**: Create a new group with an auto-generated name.
- **Rename**: Rename the selected group.
- **Delete**: Remove the selected group.

### Adding/Removing Features

1. Click a feature in the tree or canvas to highlight it.
2. In the Registration Panel, with the target group selected, click **Add Selected Feature**.
3. To remove a feature, select it in the group's feature list and click **Remove**.

### Group Statistics

When a group is selected, the Statistics section shows:

- **Features**: Total count
- **Centroid**: Center point (x, y) of all features
- **Types**: Breakdown by type (e.g., "LINE: 3, CIRCLE: 2")

### Group Visualization

Groups are displayed on the canvas as:

- Dashed colored boundary rectangle
- Semi-transparent colored fill
- Diamond-shaped centroid marker
- Group name and feature count labels

---

## 9. Image Registration

Image registration aligns a captured (or loaded) image with CAD geometry using the features in a registration group.

### Prerequisites

1. A CAD file is loaded.
2. A registration group is created with relevant features (circles and lines work best).
3. An image is loaded (from file or camera capture).

### Loading an Image from File

1. Open the Registration Panel (**View → Registration Panel**).
2. Click **Load Image...**.
3. Select an image file (PNG, BMP, or TIFF).
4. Set the **Pixel Size** (mm/pixel) for your imaging system. Default: 0.01 mm/pixel.
5. Click **OK**.

The image appears as a semi-transparent background layer under the CAD geometry.

### Registration Workflow

There are three registration modes:

**Coarse Registration**
- Aligns image and CAD by matching centroids and bounding boxes.
- Provides a rough initial alignment. Fast, but not pixel-accurate.

**Fine Registration (ICP)**
- Refines the coarse result using Iterative Closest Point.
- Requires a successful coarse registration first.
- Reports iteration count and convergence status.

**Full Registration**
- Runs coarse then fine in sequence. Recommended for most use cases.

### Steps

1. Select a registration group.
2. Load or capture an image.
3. Click **Full Registration**.
4. Check the status label for results (error values in mm, iteration count, convergence).
5. The image shifts to align with the CAD geometry on the canvas.

### Interpreting Results

The status label shows:

- **Coarse**: RMSE error in mm (lower is better)
- **Fine**: Iteration count, RMSE error, convergence flag
- **Full**: Coarse error → Fine error, iteration count

A well-registered result typically has a fine error below 0.5 mm.

---

## 10. Camera Capture

If a MindVision industrial camera is connected and the SDK is installed, you can capture images directly from the Registration Panel.

### Requirements

- MindVision camera (USB or GigE)
- MindVision MVCAM SDK (`libMVSDK.so`) installed on the system

If the SDK is not installed, the Camera Capture section is hidden and only file-based image loading is available.

### Camera Workflow

1. Open the Registration Panel (**View → Registration Panel**).
2. The **Camera Capture** section appears at the top of the panel.
3. Click **Refresh** to scan for connected cameras.
4. Select a camera from the dropdown list.
5. Click **Open** — the live preview starts.
6. Adjust camera settings as needed (see below).
7. Click **Capture Frame** — the current frame is frozen and loaded as the image layer.
8. The registration buttons become active. Proceed with registration as described in Section 9.
9. Click **Close** to stop the camera when finished.

### Camera Settings

Click the **Settings** button to expand the settings panel:

| Parameter | Description |
|-----------|-------------|
| **Auto Exposure** | Enables automatic exposure control. When on, the exposure slider is disabled. |
| **Exposure** | Manual exposure time in microseconds. Range depends on camera hardware. |
| **Gamma** | Gamma correction value. Range depends on camera hardware. |
| **Contrast** | Contrast adjustment. Range depends on camera hardware. |
| **Analog Gain** | Analog gain setting. Higher values increase brightness but may add noise. |
| **Mirror Horizontal** | Flips the image left-right. |
| **Mirror Vertical** | Flips the image top-bottom. |

Settings are applied to the camera hardware immediately when changed.

### Re-capturing

You can click **Capture Frame** again at any time while the camera is open. The previous capture is overwritten and the canvas updates with the new image.

---

## 11. Measurement Queries

The Query Panel allows you to define and evaluate automated measurement queries against loaded CAD features.

### Opening the Query Panel

Choose **View → Query Panel**.

### Query Language

Enter one query per line. Lines starting with `#` are comments.

**Circle center distance**:
```
circles(ID1, ID2)
```
Measures the center-to-center distance between two circle features.

**Line perpendicular distance**:
```
lines(ID1, ID2)
```
Measures the perpendicular distance between two line features.

### Feature ID Resolution

Queries can reference features by:

1. Full feature ID (UUID)
2. DXF handle
3. Partial UUID prefix

### Example

```
# Measure distance between two mounting holes
circles(1A2B, 3C4D)

# Measure gap between two edges
lines(F5E6, A7B8)
```

### Evaluating Queries

Click **Evaluate** to run all queries. Results appear in the table:

| Column | Description |
|--------|-------------|
| Query | The query expression |
| Value | Measured value in mm |
| Nominal | CAD nominal value |
| Deviation | Measured minus nominal |
| Status | "ok" (green) or error (red) |

A summary line shows: "Evaluated: N queries | OK: M | Errors: K"

### Saving and Loading

- **Load**: Opens a `.txt` or `.query` file into the editor.
- **Save**: Saves the editor contents to a `.txt` file.
- **Export Results**: Saves evaluation results to `.txt` or `.csv` format.

---

## 12. Keyboard Shortcuts Reference

| Shortcut | Action |
|----------|--------|
| Ctrl+O | Open DXF file |
| Ctrl+D | Import DWG file |
| Ctrl+F | Fit All (zoom to fit) |
| Ctrl+Q | Exit application |
| Scroll wheel | Zoom in/out (centered on cursor) |
| Middle/right drag | Pan |
| Left click | Select feature |
| Right-click (tree) | Context menu |

---

## 13. Supported File Formats

### Input Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| DXF | `.dxf` | Native support via ezdxf |
| DWG | `.dwg` | Converted to DXF via ODA or libredwg |
| Images | `.png`, `.bmp`, `.tif`, `.tiff` | Loaded as registration image layer |

### Supported DXF Entity Types

| Entity | Rendered | Notes |
|--------|----------|-------|
| LINE | Yes | |
| CIRCLE | Yes | |
| ARC | Yes | Circular and elliptical arcs |
| ELLIPSE | Yes | Rendered as elliptical arc |
| POLYLINE / LWPOLYLINE | Yes | Open, closed, and filled |
| SPLINE | Yes | Evaluated with ezdxf BSpline engine |
| TEXT / MTEXT | Yes | With formatting code conversion |
| DIMENSION | Yes | Decomposed into lines, solids, text |
| HATCH | Yes | All edge types supported |
| POINT | Yes | |
| LEADER | Yes | With arrowhead |
| INSERT | Decomposed | Block references exploded into child entities |

### Output Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Query results (text) | `.txt` | Human-readable results |
| Query results (CSV) | `.csv` | Structured data export |

---

## 14. Troubleshooting

### "Import DWG" button is disabled

No DWG converter backend is detected. Install ODA File Converter or libredwg, then configure via **Settings → Configure DWG Converter...**.

### DWG conversion fails

- Verify the DWG file is not corrupted (try opening in a CAD viewer).
- Check that the converter executable is in your PATH or configured correctly.
- For large files, the 300-second timeout may be insufficient. Contact support.

### Canvas appears blank after loading a file

- Click **Fit All** (Ctrl+F) to reset the viewport.
- The file may contain features at very different coordinate ranges. Check the Feature Tree to verify features were loaded.

### Camera not detected

- Ensure the MindVision SDK (`libMVSDK.so`) is installed.
- Verify the camera is connected (USB cable or Ethernet for GigE cameras).
- Click **Refresh** in the Camera Capture section.
- For GigE cameras, ensure the camera and computer are on the same network subnet.

### Registration produces poor alignment

- Ensure the registration group contains relevant features (circles and lines are most effective).
- Verify the **Pixel Size** is correct for your imaging system.
- Check that the captured image shows the correct area of the part.
- Try adding more features to the registration group.
- The image should contain clear edges corresponding to the CAD features in the group.

### Application crashes on DWG import

This was a known issue where DWG conversion callbacks executed GUI operations from a background thread. Ensure you are running the latest version where this has been fixed.
