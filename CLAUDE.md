# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This is a Python/PySide6 desktop application with no package metadata file in the repository. Run commands from the repository root so `cadviewer` imports resolve.

```bash
# Launch the application
python main.py

# Launch and auto-load a drawing
python main.py path/to/drawing.dxf
python main.py path/to/drawing.dwg

# Run the metrology integrity test module
python -m cadviewer.tests.test_forced_failure

# Run a specific test function with pytest, if pytest is installed
python -m pytest cadviewer/tests/test_forced_failure.py::test_correct_registration

# Run all pytest-discoverable tests, if pytest is installed
python -m pytest cadviewer/tests

# Syntax/import smoke check for tracked Python files
python -m compileall main.py cadviewer

# Generate the calibration chessboard PDF
python gen_chessboard.py

# Run the full diagnostics workflow; expects local sample/config paths used by the script
python run_diagnostics.py
```

Runtime dependencies documented by the app manuals: `PySide6`, `ezdxf`, `numpy`, `opencv-python`/`cv2`, and `scipy`; MindVision MVCAM SDK (`libMVSDK.so`) is optional for camera capture. DWG import requires an external converter: ODA File Converter or `libredwg-utils`.

## High-level architecture

- `main.py` creates the Qt `QApplication`, opens `cadviewer.ui.main_window.MainWindow`, and optionally auto-loads a CLI-provided DXF/DWG path.
- `cadviewer.ui.main_window.MainWindow` is the top-level coordinator. It owns the active `FeatureRepository`, `DXFImporter`, `RegistrationManager`, `AppConfig`, central `CADViewerCanvas`, tree/property panels, and docked registration/query panels. Loading a DXF replaces the repository, refreshes UI panels, restores registration groups, and creates a `RegistrationPipeline` for the new repository.
- `cadviewer.core.signals.bus` is the global Qt signal bus for cross-module communication between UI, rendering, registration, query evaluation, DWG conversion, and settings.
- `cadviewer.core.config.AppConfig` persists user settings to `~/.config/cadviewer/settings.json`, including pixel size, last file paths, camera settings, calibration settings, lens calibration, and registration groups.

## Data model and CAD import

- `cadviewer.parsers.dxf_importer.DXFImporter` converts ezdxf entities into pure-Python `CADFeature` objects; it does not depend on OpenCascade. It parses common geometry and annotation entities including LINE, CIRCLE, ARC/ELLIPSE, POLYLINE/LWPOLYLINE, SPLINE, DIMENSION block geometry, TEXT/MTEXT, HATCH, POINT, SOLID, LEADER, and decomposed INSERT virtual entities.
- `cadviewer.models.repository.FeatureRepository` is the single source of truth for loaded CAD geometry. It indexes features by stable feature id, type, layer, and DXF handle. Rendering, selection, registration, and measurement all read from this repository.
- Feature IDs are intended to be stable/deterministic via `cadviewer.models.feature._stable_id`; preserve this property when changing import behavior because queries and registration groups may reference IDs or DXF handles.

## Rendering and UI flow

- `cadviewer.renderers.cad_canvas.CADViewerCanvas` is the main QPainter-based 2D viewer. It displays CAD geometry, selection highlights, registration groups, image overlays, and measurement debug overlays.
- Left/right panels (`cadviewer.ui.tree_panel`, `property_panel`) provide feature browsing and inspection. Dock widgets (`registration_panel`, `query_panel`) handle image registration and measurement query workflows.
- Image overlays are managed through renderer/image-layer code and are distinct from CAD drawing pixels. Measurement code must use raw/captured image data, not rendered canvas composites.

## Registration

- `cadviewer.registration.pipeline.RegistrationPipeline` is a strategy-based orchestrator. It stores shared repo/group/debug context and delegates actual work to a `RegistrationStrategy` selected from `cadviewer.registration.strategy.STRATEGY_REGISTRY`.
- Public registration entry points are `run_coarse`, `run_fine`, and `run_full`. They return dictionaries consumed by UI code and debug overlay code, so preserve the existing result keys when refactoring.
- Registration groups live in `cadviewer.models.registration.RegistrationManager` and are persisted through `AppConfig.registration_groups`.
- The image affine used by measurement maps pixel coordinates to CAD/world coordinates. Be careful with pixel-size scaling and Y-axis direction when changing registration or image-layer transforms.

## Measurement and metrology integrity

- `cadviewer.measurement.measurement_pipeline.MeasurementPipeline` is the CAD-guided local feature measurement pipeline. Its contract is explicit: CAD features are only geometric priors for ROI prediction; measured values must come from raw grayscale camera/file image edge data.
- Measurement flow: CAD feature → ROI prediction → Scharr gradient edge sampling → circle/line fitting → residual distortion correction (if configured) → affine pixel-to-world conversion → `MeasuredFeature` with `source_type="FITTED"`.
- `cadviewer.measurement.evaluator.QueryEvaluator` evaluates query-panel expressions against measured features. It must not fall back to CAD geometry as measured output when image measurement is unavailable; no-measurement should remain a failure state rather than silently using nominal CAD values.
- The forced-failure test (`python -m cadviewer.tests.test_forced_failure`) specifically guards against CAD geometry leaking into measured values. Run it after changes to measurement, query evaluation, registration transforms, or image handling.

## Calibration, camera, and diagnostics

- Calibration UI and logic are under `cadviewer.ui.calibration_window` and `cadviewer.calibration`. Calibration covers pixel size, OpenCV lens calibration, and geometric correction data stored in `AppConfig`.
- Camera integration is under `cadviewer.camera` and is optional. The MindVision SDK driver should fail gracefully when SDK/hardware is unavailable so file-based image loading still works.
- `run_diagnostics.py` is a local full-pipeline diagnostic script that loads specific sample/config paths, runs registration and measurement diagnostics, and writes reports to `diagnostics_output/`. Treat its hard-coded paths as local workflow assumptions, not general application defaults.

## DWG conversion

- DWG import is implemented in `cadviewer.converters`. The UI validates converter availability at startup and disables DWG import when no converter is found.
- Conversion runs asynchronously and returns to the Qt main thread through `_DWGResultBridge` in `MainWindow`; keep GUI updates on the main thread when modifying conversion callbacks.
- Supported converter backends documented by the app are ODA File Converter and `libredwg-utils`.
