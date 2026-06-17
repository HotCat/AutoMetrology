"""
MainWindow — top-level application window.

Layout:
  ┌──────────────────────────────────────────────┐
  │  Toolbar: [Open DXF] [Fit All] [Pan] [Sel]   │
  ├─────────┬──────────────────────┬──────────────┤
  │ Feature │                      │  Property    │
  │ Tree    │   CAD 2D Viewer      │  Panel       │
  │ Panel   │   (QPainter canvas)  │              │
  │         │   [or OCC 3D viewer] │              │
  └─────────┴──────────────────────┴──────────────┘
  │  Status Bar                                   │
  └──────────────────────────────────────────────┘
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot, QSize, Signal, QObject, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QToolBar,
    QFileDialog, QStatusBar, QMessageBox, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout, QProgressBar, QDockWidget, QDialog,
)

from ..models.feature import FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from ..parsers.dxf_importer import DXFImporter
from ..renderers.cad_canvas import CADViewerCanvas
from ..ui.tree_panel import FeatureTreePanel
from ..ui.property_panel import PropertyPanel
from ..ui.registration_panel import RegistrationPanel
from ..ui.query_panel import QueryPanel
from ..ui.dwg_import_dialog import DWGImportDialog
from ..ui.dwg_settings_dialog import DWGSettingsDialog
from ..registration.pipeline import RegistrationPipeline
from ..converters.dwg_converter import DWGConverter
from ..converters.converter_config import ConversionConfig
from ..converters.oda_cli import ODACLI
from ..core.signals import bus
from ..core.config import AppConfig
from ..core.i18n import LANG_EN, LANG_ZH_CN, i18n, retranslate_widget_tree, set_language, tr
from ..measurement.production_log import ProductionLogStore
from ..ui.production_log_dialog import ProductionLogViewer


class _DWGResultBridge(QObject):
    """Thread-safe bridge: worker thread emits signal, main thread receives."""
    result_ready = Signal(object)  # ConversionResult


class MainWindow(QMainWindow):
    """Main application window for CAD Inspection Tool."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CAD Inspection Tool — Metrology DXF Viewer")
        self.resize(1600, 900)
        self.setMinimumSize(1024, 600)

        # Core data
        self._repo = FeatureRepository()
        self._importer = DXFImporter()
        self._reg_manager = RegistrationManager(self._repo)
        self._dwg_converter = DWGConverter()
        self._config = AppConfig.load()
        set_language(getattr(self._config, "language", LANG_EN))
        self._production_log_store = ProductionLogStore()
        self._production_log_viewer = None
        self._last_measurement_debug: dict = {}
        self._last_measurement_affine = None
        self._query_pair_pick_mode: Optional[str] = None
        self._query_pair_pick_ids: list[str] = []

        # Build UI
        self._setup_ui()
        self._setup_toolbar()
        self._setup_menu()
        self._setup_statusbar()
        self._setup_dock_widgets()
        self._connect_signals()
        self._check_dwg_converter()
        self.retranslate_ui()

        # Auto-load last DXF if available
        if self._config.last_dxf_path and Path(self._config.last_dxf_path).exists():
            QTimer.singleShot(0, lambda: self._load_dxf(self._config.last_dxf_path))

    def _setup_ui(self) -> None:
        """Create the main splitter layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Main splitter
        self._splitter = QSplitter(Qt.Horizontal)

        # Left: Feature Tree Panel
        self._tree_panel = FeatureTreePanel()
        self._splitter.addWidget(self._tree_panel)

        # Center: CAD Viewer Canvas (QPainter-based)
        self._viewer = CADViewerCanvas()
        self._splitter.addWidget(self._viewer)

        # Right: Property Panel
        self._property_panel = PropertyPanel(self._repo)
        self._splitter.addWidget(self._property_panel)

        # Set splitter sizes (tree:viewer:props = 250:flex:280)
        self._splitter.setSizes([250, 900, 280])
        self._splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self._splitter)

        # Apply dark theme
        self.setStyleSheet(self._dark_theme())

    def _setup_toolbar(self) -> None:
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setStyleSheet("QToolBar { background: #2d2d2d; border: none; padding: 4px; }")
        self.addToolBar(toolbar)

        # Open DXF
        open_action = QAction("Open DXF", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_dxf)
        toolbar.addAction(open_action)

        # Import DWG
        self._dwg_action = QAction("Import DWG", self)
        self._dwg_action.setShortcut(QKeySequence("Ctrl+D"))
        self._dwg_action.triggered.connect(self._open_dwg)
        toolbar.addAction(self._dwg_action)

        toolbar.addSeparator()

        # Fit All
        fit_action = QAction("Fit All", self)
        fit_action.setShortcut(QKeySequence("Ctrl+F"))
        fit_action.triggered.connect(lambda: bus.view_fit_all.emit())
        toolbar.addAction(fit_action)

        toolbar.addSeparator()

        # Pan mode
        pan_action = QAction("Pan", self)
        pan_action.setCheckable(True)
        pan_action.toggled.connect(self._toggle_pan)
        toolbar.addAction(pan_action)

        # Selection mode
        sel_action = QAction("Select", self)
        sel_action.setCheckable(True)
        sel_action.toggled.connect(self._toggle_selection)
        toolbar.addAction(sel_action)

    def _setup_menu(self) -> None:
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open DXF...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_dxf)
        file_menu.addAction(open_action)

        dwg_open_action = QAction("Import DWG...", self)
        dwg_open_action.setShortcut(QKeySequence("Ctrl+D"))
        dwg_open_action.triggered.connect(self._open_dwg)
        file_menu.addAction(dwg_open_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")
        fit_action = QAction("Fit All", self)
        fit_action.triggered.connect(lambda: bus.view_fit_all.emit())
        view_menu.addAction(fit_action)
        view_menu.addSeparator()
        reg_panel_action = QAction("Registration Panel", self)
        reg_panel_action.setCheckable(True)
        reg_panel_action.toggled.connect(self._toggle_reg_panel)
        view_menu.addAction(reg_panel_action)
        self._reg_panel_action = reg_panel_action
        query_panel_action = QAction("Measurement Window", self)
        query_panel_action.setCheckable(True)
        query_panel_action.toggled.connect(self._toggle_query_panel)
        view_menu.addAction(query_panel_action)
        self._query_panel_action = query_panel_action

        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        oda_config_action = QAction("Configure DWG Converter...", self)
        oda_config_action.triggered.connect(self._show_dwg_settings)
        settings_menu.addAction(oda_config_action)
        settings_menu.addSeparator()
        language_menu = settings_menu.addMenu("Language")
        self._language_menu = language_menu
        self._language_group = QActionGroup(self)
        self._language_group.setExclusive(True)
        self._english_action = QAction("English", self)
        self._english_action.setCheckable(True)
        self._english_action.setData(LANG_EN)
        self._chinese_action = QAction("Simplified Chinese", self)
        self._chinese_action.setCheckable(True)
        self._chinese_action.setData(LANG_ZH_CN)
        for action in [self._english_action, self._chinese_action]:
            self._language_group.addAction(action)
            language_menu.addAction(action)
        self._language_group.triggered.connect(self._on_language_action_triggered)
        settings_menu.addSeparator()
        cal_action = QAction("Camera Calibration...", self)
        cal_action.triggered.connect(self._open_calibration_window)
        settings_menu.addAction(cal_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self) -> None:
        """Create status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._status_label = QLabel("Ready — Open a DXF file to begin inspection")
        self._statusbar.addWidget(self._status_label)

        self._feature_count_label = QLabel("Features: 0")
        self._statusbar.addPermanentWidget(self._feature_count_label)

    def _setup_dock_widgets(self) -> None:
        """Create dockable panels for registration and future tools."""
        # Registration panel
        self._reg_panel = RegistrationPanel(self._reg_manager, self._repo, self._config)
        self._reg_panel._canvas = self._viewer
        reg_dock = QDockWidget("Registration", self)
        reg_dock.setWidget(self._reg_panel)
        reg_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, reg_dock)
        reg_dock.hide()

        # Store reference for toolbar toggle
        self._reg_dock = reg_dock

        # Query window: keep measurements in a wide standalone window so
        # operators can see many Value/Nominal/Deviation/Status rows at once.
        self._query_panel = QueryPanel()
        self._production_log_viewer = ProductionLogViewer(self._production_log_store)
        self._query_panel.set_production_log_viewer(self._production_log_viewer)
        self._query_window = QDialog(self)
        self._query_window.setWindowTitle("Measurement Queries")
        self._query_window.setWindowFlags(
            self._query_window.windowFlags() | Qt.WindowMaximizeButtonHint
        )
        self._query_window.resize(1280, 760)
        query_layout = QVBoxLayout(self._query_window)
        query_layout.setContentsMargins(0, 0, 0, 0)
        query_layout.addWidget(self._query_panel)
        self._query_window.rejected.connect(
            lambda: self._query_panel_action.setChecked(False)
        )
        self._query_window.hide()

    def _connect_signals(self) -> None:
        """Connect signals between components."""
        # Tree → highlight/viewer
        self._tree_panel.feature_selected.connect(self._on_feature_selected)
        self._tree_panel.feature_deselected.connect(self._on_feature_deselected)

        # Viewer → tree/property sync (click in viewer selects in tree)
        self._viewer.feature_clicked.connect(self._on_viewer_click)

        # Bus
        bus.features_loaded.connect(self._on_features_loaded)
        bus.feature_deselected.connect(self._on_feature_deselected)
        bus.queries_evaluated.connect(self._on_queries_evaluated)
        self._query_panel.result_selected.connect(self._on_query_result_selected)
        self._query_panel.production_run_requested.connect(self._run_production_measurement_cycle)
        self._query_panel.production_log_requested.connect(self._show_production_log_viewer)
        if self._production_log_viewer is not None:
            self._production_log_viewer.record_selected.connect(self._on_production_log_record_selected)
            self._production_log_viewer.result_selected.connect(self._on_query_result_selected)
        self._query_panel.pair_pick_requested.connect(self._on_query_pair_pick_requested)
        self._query_panel.pair_pick_cancelled.connect(self._on_query_pair_pick_cancelled)
        i18n.language_changed.connect(self._on_language_changed)

    def retranslate_ui(self) -> None:
        retranslate_widget_tree(self)
        self._sync_language_actions()
        if hasattr(self, "_query_window"):
            self._query_window.setWindowTitle(tr("Measurement Queries"))
        if hasattr(self, "_reg_dock"):
            self._reg_dock.setWindowTitle(tr("Registration"))
        if hasattr(self, "_feature_count_label"):
            self._feature_count_label.setText(tr("Features: 0") if self._repo.count() == 0 else f"{tr('Features')}: {self._repo.count()}")

    def _sync_language_actions(self) -> None:
        if not hasattr(self, "_english_action"):
            return
        self._english_action.setChecked(i18n.language == LANG_EN)
        self._chinese_action.setChecked(i18n.language == LANG_ZH_CN)

    @Slot(object)
    def _on_language_action_triggered(self, action) -> None:
        language = action.data() if action is not None else LANG_EN
        set_language(str(language))

    @Slot(str)
    def _on_language_changed(self, language: str) -> None:
        self._config.language = language
        self._config.save()
        self.retranslate_ui()

    # ── slot handlers ──────────────────────────────────────────────

    @Slot()
    def _open_dxf(self) -> None:
        """Open a DXF file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DXF File", str(Path.cwd()),
            "DXF Files (*.dxf);;All Files (*)"
        )
        if path:
            self._load_dxf(path)

    def _load_dxf(self, path: str) -> None:
        """Load and render a DXF file."""
        self._status_label.setText(f"{tr('Loading')} {Path(path).name}...")
        QApplication.processEvents()

        # Parse
        self._repo = self._importer.import_file(path)
        count = self._repo.count()

        self._status_label.setText(f"{tr('Loaded')} {count} {tr('features from')} {Path(path).name}")
        self._feature_count_label.setText(f"{tr('Features')}: {count}")

        # Populate tree
        self._tree_panel.populate(self._repo)

        # Update property panel repo reference
        self._property_panel._repo = self._repo

        # Render in viewer
        self._viewer.load_repository(self._repo)
        self._last_measurement_debug = {}
        self._last_measurement_affine = None
        self._viewer.set_measurement_debug({}, None)
        self._cancel_query_pair_pick(update_panel=True)

        # Update registration manager with new repo. Feature groups are no longer
        # exposed or persisted; auto correspondence uses the full CAD context.
        self._reg_panel.set_repository(self._repo)

        # Create registration pipeline for new repo
        self._pipeline = RegistrationPipeline(self._repo, self._reg_manager)
        self._reg_panel.set_pipeline(self._pipeline)
        self._reg_panel.apply_active_production_profile()

        # Track last DXF path for auto-restore
        self._last_dxf_path = path
        self._config.last_dxf_path = path
        self._config.save()

        # Print type summary
        counts = self._repo.type_counts()
        for ftype, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  {ftype.name}: {c}")

        bus.features_loaded.emit(count)

    @Slot(str)
    def _on_feature_selected(self, feature_id: str) -> None:
        """Handle feature selection from tree."""
        feature = self._repo.get(feature_id)
        if feature:
            if self._query_pair_pick_mode is not None:
                self._handle_query_pair_feature(feature_id)
                return
            self._status_label.setText(f"Selected: {feature.display_name}")

    @Slot()
    def _on_feature_deselected(self) -> None:
        """Handle feature deselection."""
        self._property_panel.clear()
        self._status_label.setText(tr("Ready"))

    @Slot(str)
    def _on_viewer_click(self, feature_id: str) -> None:
        """Handle feature click in viewer — sync with tree."""
        self._tree_panel.select_feature(feature_id)
        bus.property_update.emit({"feature_id": feature_id})
        if self._query_pair_pick_mode is not None:
            self._handle_query_pair_feature(feature_id)

    @Slot(int)
    def _on_features_loaded(self, count: int) -> None:
        self._feature_count_label.setText(f"{tr('Features')}: {count}")

    @Slot(int)
    def _on_queries_evaluated(self, _count: int) -> None:
        self._evaluate_current_queries()

    def _evaluate_current_queries(self) -> int:
        """Evaluate measurement queries using current image and registration."""
        from ..measurement.evaluator import QueryEvaluator
        from ..measurement.measurement_pipeline import MeasurementPipeline
        from ..calibration.residual_map import residual_map_from_config
        import numpy as np
        try:
            import cv2
            HAS_CV2 = True
        except ImportError:
            HAS_CV2 = False

        query_text = self._query_panel.get_query_text()

        # Build measurement pipeline from current image + registration
        pipeline = None
        image_layer = self._viewer.get_image_layer()
        if image_layer.has_image and HAS_CV2:
            bgr_image = image_layer.image
            affine = image_layer.affine
            if bgr_image is not None and affine is not None:
                image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
                if not np.allclose(affine, np.eye(3), atol=1e-6):
                    pipeline = MeasurementPipeline(
                        self._repo, image, affine,
                        pixel_size_mm=self._config.pixel_size_mm,
                        residual_map=residual_map_from_config(self._config),
                        pixel_to_world_transform=(
                            self._reg_panel.measurement_pixel_to_world_transform(image_layer.path)
                            if hasattr(self, "_reg_panel") else None
                        ),
                    )

        evaluator = QueryEvaluator(self._repo, measurement_pipeline=pipeline)
        results = evaluator.evaluate(query_text)
        self._query_panel.set_results(results)

        # Push measurement debug overlay to canvas. Keep the full set so a
        # selected query row can temporarily narrow the overlay to its features.
        if pipeline is not None:
            self._last_measurement_debug = dict(pipeline.get_debug_data())
            self._last_measurement_affine = pipeline.measurement_transform
            self._viewer.set_measurement_debug(
                self._last_measurement_debug, self._last_measurement_affine,
            )
        else:
            self._last_measurement_debug = {}
            self._last_measurement_affine = None
            self._viewer.set_measurement_debug({}, None)

        return len(results)

    @Slot()
    def _run_production_measurement_cycle(self) -> None:
        """Capture camera frame, auto-register, and evaluate loaded queries."""
        if self._query_pair_pick_mode is not None:
            self._cancel_query_pair_pick(update_panel=True)

        self._status_label.setText(tr("Production cycle: capturing camera frame..."))
        QApplication.processEvents()
        if not self._reg_panel.capture_current_frame_for_production():
            self._status_label.setText(tr("Production cycle failed during camera capture"))
            return

        self._status_label.setText(tr("Production cycle: applying window registration..."))
        QApplication.processEvents()
        if not self._reg_panel.run_registration_for_production():
            self._status_label.setText(tr("Production cycle failed during window registration"))
            return

        self._status_label.setText(tr("Production cycle: evaluating measurement queries..."))
        QApplication.processEvents()
        count = self._evaluate_current_queries()
        record_id = self._save_current_production_log()
        if record_id:
            if self._production_log_viewer is not None:
                self._production_log_viewer.refresh(select_record_id=record_id)
            self._status_label.setText(
                f"Production cycle complete — evaluated {count} queries; log {record_id[:8]}"
            )
        else:
            self._status_label.setText(
                f"Production cycle complete — evaluated {count} queries; log save failed"
            )

    def _save_current_production_log(self) -> str:
        """Persist the current production result set and replay context."""
        try:
            image_layer = self._viewer.get_image_layer()
            affine = image_layer.affine if image_layer.has_image else None
            calibration = {
                "pixel_size_mm": self._config.pixel_size_mm,
                "lens_calibration": getattr(self._config, "lens_calibration", None),
                "calibration_applied": self._reg_panel.image_calibration_applied(),
            }
            camera = getattr(self._config, "camera", {})
            return self._production_log_store.create_record(
                results=self._query_panel.results(),
                query_text=self._query_panel.get_query_text(),
                cad_path=getattr(self, "_last_dxf_path", self._config.last_dxf_path),
                source_image_path=image_layer.path,
                image=image_layer.image,
                pixel_size_mm=self._reg_panel.production_pixel_size_mm(),
                affine=affine,
                registration=self._reg_panel.last_auto_registration_snapshot(),
                production_profile=self._reg_panel.production_profile_snapshot(),
                calibration=calibration,
                camera=camera,
            )
        except Exception as e:
            self._status_label.setText(f"Production log save error: {e}")
            return ""

    @Slot()
    def _show_production_log_viewer(self) -> None:
        self._query_panel.show_production_log_view()
        if self._production_log_viewer is not None and self._production_log_viewer.current_record_id():
            self._on_production_log_record_selected(
                self._production_log_viewer.current_record_id()
            )

    @Slot(str)
    def _on_production_log_record_selected(self, record_id: str) -> None:
        record = self._production_log_store.get_record(record_id)
        if record is None:
            return
        results = self._production_log_store.get_results(record_id)
        if not self._load_production_log_context(record, results):
            return
        self._status_label.setText(
            f"Loaded production log {record_id[:8]} from {record.get('created_at', '')}"
        )

    def _load_production_log_context(self, record: dict, results: list) -> bool:
        cad_path = record.get("cad_path", "")
        image_path = record.get("image_path", "")
        if cad_path and Path(cad_path).exists() and cad_path != getattr(self, "_last_dxf_path", ""):
            self._load_dxf(cad_path)
        elif not self._repo.count() and cad_path and Path(cad_path).exists():
            self._load_dxf(cad_path)

        image_layer = self._viewer.get_image_layer()
        if image_path and Path(image_path).exists():
            image_layer.load_image(image_path)
        affine = record.get("affine")
        if affine is not None:
            import numpy as np
            image_layer.set_affine_transform(np.array(affine, dtype=float))
        pixel_size = record.get("pixel_size_mm")
        if pixel_size is not None:
            image_layer.set_pixel_size_mm(float(pixel_size))
        self._viewer.update()
        self._regenerate_log_measurement_debug(record, results)
        return True

    def _regenerate_log_measurement_debug(self, record: dict, results: list) -> None:
        try:
            from ..measurement.evaluator import QueryEvaluator
            from ..measurement.measurement_pipeline import MeasurementPipeline
            from ..calibration.residual_map import residual_map_from_config
            import cv2
            import numpy as np

            image_layer = self._viewer.get_image_layer()
            if image_layer.image is None or record.get("affine") is None:
                self._last_measurement_debug = {}
                self._last_measurement_affine = None
                self._viewer.set_measurement_debug({}, None)
                return
            gray = cv2.cvtColor(image_layer.image, cv2.COLOR_BGR2GRAY)
            affine = np.array(record["affine"], dtype=float)
            registration = record.get("registration") or {}
            measurement_transform = registration.get("measurement_pixel_to_world")
            if measurement_transform is not None:
                measurement_transform = np.array(measurement_transform, dtype=float)
            pipeline = MeasurementPipeline(
                self._repo, gray, affine,
                pixel_size_mm=float(record.get("pixel_size_mm") or self._config.pixel_size_mm),
                residual_map=residual_map_from_config(self._config),
                pixel_to_world_transform=measurement_transform,
            )
            query_text = "\n".join(
                r.instruction.raw_text for r in results if r.instruction is not None
            )
            QueryEvaluator(self._repo, pipeline).evaluate(query_text)
            self._last_measurement_debug = dict(pipeline.get_debug_data())
            self._last_measurement_affine = pipeline.measurement_transform
            self._viewer.set_measurement_debug(
                self._last_measurement_debug, self._last_measurement_affine,
            )
        except Exception:
            self._last_measurement_debug = {}
            self._last_measurement_affine = None
            self._viewer.set_measurement_debug({}, None)

    @Slot(str)
    def _on_query_pair_pick_requested(self, mode: str) -> None:
        """Start interactive pair selection for query expression generation."""
        if mode not in ("lines", "circles", "circle", "arcs"):
            return
        self._query_pair_pick_mode = mode
        self._query_pair_pick_ids = []
        self._viewer.set_highlighted_features([])
        self._query_panel.set_pair_pick_active(mode, 0)
        label = self._query_pick_feature_label(mode)
        if mode in ("circle", "arcs"):
            self._status_label.setText(f"Select {label} for measurement query")
        else:
            self._status_label.setText(f"Select first {label} for measurement query")

    @Slot()
    def _on_query_pair_pick_cancelled(self) -> None:
        self._cancel_query_pair_pick(update_panel=False)
        self._status_label.setText(tr("Measurement query pair selection cancelled"))

    def _cancel_query_pair_pick(self, update_panel: bool = True) -> None:
        self._query_pair_pick_mode = None
        self._query_pair_pick_ids = []
        self._viewer.set_highlighted_features([])
        if update_panel and hasattr(self, "_query_panel"):
            self._query_panel.set_pair_pick_active(None)

    def _handle_query_pair_feature(self, feature_id: str) -> None:
        """Consume one feature selection while the query pair picker is active."""
        mode = self._query_pair_pick_mode
        if mode is None:
            return

        feature = self._repo.get(feature_id)
        if feature is None:
            return

        expected_type = self._query_pick_feature_type(mode)
        expected_label = self._query_pick_feature_label(mode)
        if feature.feature_type != expected_type:
            message = f"Pick a {expected_label}; selected {feature.feature_type.name.lower()}"
            self._query_panel.set_pair_pick_message(message)
            self._status_label.setText(message)
            self._set_query_pair_highlight(self._query_pair_pick_ids)
            return

        if feature_id in self._query_pair_pick_ids:
            message = f"Pick a different second {expected_label}"
            self._query_panel.set_pair_pick_message(message)
            self._status_label.setText(message)
            self._set_query_pair_highlight(self._query_pair_pick_ids)
            return

        self._query_pair_pick_ids.append(feature_id)
        self._set_query_pair_highlight(self._query_pair_pick_ids)

        target_count = 1 if mode in ("circle", "arcs") else 2
        if len(self._query_pair_pick_ids) < target_count:
            self._query_panel.set_pair_pick_active(mode, len(self._query_pair_pick_ids))
            self._status_label.setText(f"Selected first {expected_label}; select second {expected_label}")
            return

        if mode in ("circle", "arcs"):
            fid = self._query_pair_pick_ids[0]
            feat = self._repo.get(fid)
            if feat is None:
                self._cancel_query_pair_pick(update_panel=True)
                return
            func = "circle" if mode == "circle" else "arcs"
            expression = f"{func}({self._query_token_for_feature(feat)})"
            tolerance = self._auto_query_tolerance(expression)
            self._query_panel.append_query_expression(expression, tolerance)
            self._query_panel.set_pair_pick_active(None)
            self._query_panel.set_pair_pick_message(f"Added {expression}")
            self._status_label.setText(f"Added measurement query: {expression}")
            self._query_pair_pick_mode = None
            self._query_pair_pick_ids = []
            self._set_query_pair_highlight([fid])
            return

        fid1, fid2 = self._query_pair_pick_ids[:2]
        feat1 = self._repo.get(fid1)
        feat2 = self._repo.get(fid2)
        if feat1 is None or feat2 is None:
            self._cancel_query_pair_pick(update_panel=True)
            return

        func = "lines" if mode == "lines" else "circles"
        token1 = self._query_token_for_feature(feat1)
        token2 = self._query_token_for_feature(feat2)
        expression = f"{func}({token1}, {token2})"
        tolerance = self._auto_query_tolerance(expression)
        self._query_panel.append_query_expression(expression, tolerance)
        self._query_panel.set_pair_pick_active(None)
        self._query_panel.set_pair_pick_message(f"Added {expression}")
        self._status_label.setText(f"Added measurement query: {expression}")
        self._query_pair_pick_mode = None
        self._query_pair_pick_ids = []
        self._set_query_pair_highlight([fid1, fid2])

    def _auto_query_tolerance(self, expression: str) -> Optional[float]:
        """Compute absolute tolerance from the query panel percentage setting."""
        try:
            from ..measurement.query_parser import QueryParser
            from ..measurement.evaluator import QueryEvaluator
            insts = QueryParser().parse(expression)
            if not insts:
                return None
            nominal = QueryEvaluator(self._repo).nominal_for_instruction(insts[0])
            if nominal is None:
                return None
            return round(abs(nominal) * self._query_panel.tolerance_percent() / 100.0, 4)
        except Exception:
            return None

    def _set_query_pair_highlight(self, feature_ids: list[str]) -> None:
        ids = list(feature_ids)
        self._viewer.set_highlighted_features(ids)
        QTimer.singleShot(0, lambda ids=ids: self._viewer.set_highlighted_features(ids))

    @staticmethod
    def _query_pick_feature_type(mode: str):
        if mode == "lines":
            return FeatureType.LINE
        if mode in ("circles", "circle"):
            return FeatureType.CIRCLE
        return FeatureType.ARC

    @staticmethod
    def _query_pick_feature_label(mode: str) -> str:
        if mode == "lines":
            return "line"
        if mode in ("circles", "circle"):
            return "circle"
        return "arc"

    @staticmethod
    def _query_token_for_feature(feature) -> str:
        if feature.dxf_handle:
            return feature.dxf_handle
        return feature.feature_id.split("-", 1)[0]

    @Slot(object)
    def _on_query_result_selected(self, result) -> None:
        """Highlight CAD and detected image features for the selected query."""
        if result is None or result.instruction is None:
            self._viewer.set_highlighted_features([])
            self._viewer.set_measurement_debug(
                self._last_measurement_debug, self._last_measurement_affine,
            )
            return

        raw_ids = [
            result.instruction.feature_id_1,
            result.instruction.feature_id_2,
        ]
        feature_ids = [
            fid for fid in (self._resolve_query_feature_id(raw) for raw in raw_ids)
            if fid is not None
        ]
        self._viewer.set_highlighted_features(feature_ids)

        selected_debug = {
            fid: self._last_measurement_debug[fid]
            for fid in feature_ids
            if fid in self._last_measurement_debug
        }
        self._viewer.set_measurement_debug(
            selected_debug, self._last_measurement_affine,
        )

        if feature_ids:
            labels = ", ".join(fid[:12] for fid in feature_ids)
            self._status_label.setText(f"Selected query features: {labels}")

    def _resolve_query_feature_id(self, raw_id: str) -> Optional[str]:
        """Resolve query IDs exactly as the measurement evaluator does."""
        if not raw_id:
            return None
        if self._repo.get(raw_id):
            return raw_id
        feat = self._repo.get_by_handle(raw_id)
        if feat:
            return feat.feature_id
        for feat in self._repo.all_features():
            if feat.feature_id.startswith(raw_id):
                return feat.feature_id
        return None

    def _toggle_pan(self, checked: bool) -> None:
        pass  # pan is always via middle/right mouse in canvas

    def _toggle_selection(self, checked: bool) -> None:
        pass  # selection is always via left click in canvas

    def _toggle_reg_panel(self, checked: bool) -> None:
        if checked:
            self._reg_dock.show()
        else:
            self._reg_dock.hide()

    def _toggle_query_panel(self, checked: bool) -> None:
        if checked:
            self._query_window.show()
            self._query_window.raise_()
            self._query_window.activateWindow()
        else:
            self._query_window.hide()

    @Slot()
    def _open_dwg(self) -> None:
        """Open a DWG file, convert to DXF, then load."""
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Import DWG File"), str(Path.cwd()),
            "DWG Files (*.dwg);;All Files (*)"
        )
        if not path:
            return
        self._convert_and_load_dwg(path)

    def _open_dwg_path(self, path: str) -> None:
        """Convert and load a DWG file from a given path (CLI usage)."""
        self._convert_and_load_dwg(path)

    def _convert_and_load_dwg(self, path: str) -> None:
        """Shared logic for DWG conversion and loading."""
        # Check converter availability
        info = self._dwg_converter.validate_installation()
        if not info.installed:
            QMessageBox.warning(
                self, tr("DWG Converter Not Found"),
                tr("No DWG converter is installed.") + "\n\n"
                "Install one of:\n"
                "  • ODA File Converter: https://www.opendesign.com/guestfiles/oda_file_converter\n"
                "  • libredwg: sudo apt install libredwg-utils\n\n"
                "Then configure via Settings → Configure DWG Converter.",
            )
            return

        # Show progress dialog
        dialog = DWGImportDialog(self)
        retranslate_widget_tree(dialog)
        dialog.show()
        QApplication.processEvents()

        dwg_path = Path(path)
        output_dir = Path(path).parent

        config = ConversionConfig(
            dwg_path=dwg_path,
            output_dir=output_dir,
        )

        # Detect DWG version
        dialog.set_stage("Detecting DWG version...", 5)
        version = DWGConverter.detect_dwg_version(dwg_path)
        if version:
            dialog.set_detail(f"DWG version: {version}")
        QApplication.processEvents()

        # Run conversion
        dialog.set_stage("Running ODA File Converter...", 20)
        bus.dwg_conversion_started.emit(path)

        # Thread-safe bridge: worker emits signal, slot runs on main thread
        bridge = _DWGResultBridge()
        bridge.result_ready.connect(
            lambda result: self._on_dwg_conversion_done(result, dialog),
            Qt.QueuedConnection,
        )

        def on_complete(result):
            bridge.result_ready.emit(result)

        self._dwg_converter.convert_async(config, on_complete)

    def _on_dwg_conversion_done(self, result, dialog) -> None:
        """Handle DWG conversion result on the main thread."""
        if result.success:
            dialog.set_stage("Validating DXF output...", 75)
            dialog.set_stage("Loading features...", 90)
            dialog.set_complete(result)
            bus.dwg_conversion_completed.emit({
                "dxf_path": str(result.dxf_path),
                "entity_count": result.entity_count,
                "duration": result.duration_seconds,
            })

            # Load the converted DXF through existing pipeline
            if result.dxf_path and result.dxf_path.exists():
                self._load_dxf(str(result.dxf_path))
        else:
            dialog.set_error(result.error_message or "Unknown error")
            bus.dwg_conversion_failed.emit(result.error_message or "Unknown error")

    def _show_dwg_settings(self) -> None:
        """Open DWG converter settings dialog."""
        dialog = DWGSettingsDialog(self)
        retranslate_widget_tree(dialog)
        if dialog.exec() == QDialog.Accepted:
            path = dialog.get_converter_path()
            if path:
                from ..converters.oda_cli import ODACLI
                self._dwg_converter = DWGConverter(
                    backend=ODACLI(executable_path=Path(path))
                )
                bus.oda_path_changed.emit(path)
            else:
                self._dwg_converter = DWGConverter()
            # Re-check installation status
            info = self._dwg_converter.validate_installation()
            self._dwg_action.setEnabled(info.installed)
            if info.installed:
                self._dwg_action.setToolTip("")
            else:
                self._dwg_action.setToolTip(
                    "DWG converter not found — configure in Settings"
                )

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About CAD Inspection Tool",
            "CAD Inspection Tool v1.0\n\n"
            "A metrology-oriented DXF feature inspection tool\n"
            "for machine vision alignment and automatic dimension measurement.\n\n"
            "Built with PySide6 + QPainter (OpenCascade optional)"
        )

    def _open_calibration_window(self) -> None:
        """Open the camera calibration window."""
        from .calibration_window import CalibrationWindow
        camera = getattr(self._reg_panel, '_camera', None)
        camera_open = getattr(self._reg_panel, '_camera_open', False)
        win = CalibrationWindow(
            parent=self,
            config=self._config,
            camera=camera if camera_open else None,
        )
        retranslate_widget_tree(win)
        win.exec()
        # Persist updated chessboard params
        params = win.get_chessboard_params()
        self._config.calibration.chessboard_cols = params["cols"]
        self._config.calibration.chessboard_rows = params["rows"]
        self._config.calibration.chessboard_cell_mm = params["cell_mm"]
        pixel_size = win.get_computed_pixel_size()
        if pixel_size is not None:
            self._config.pixel_size_mm = float(pixel_size)
        if hasattr(self, "_reg_panel"):
            self._reg_panel._pixel_size_mm = float(self._config.pixel_size_mm)
            layer = self._viewer.get_image_layer()
            if layer is not None:
                layer.set_pixel_size_mm(float(self._config.pixel_size_mm))
        self._config.save()

    def _check_dwg_converter(self) -> None:
        """Check ODA availability at startup, disable DWG button if missing."""
        info = self._dwg_converter.validate_installation()
        if not info.installed:
            self._dwg_action.setEnabled(False)
            self._dwg_action.setToolTip(
                "ODA File Converter not found — install and configure via Settings menu"
            )

    def closeEvent(self, event) -> None:
        """Handle window close — save config, cleanup camera resources."""
        # Persist settings from registration panel
        if hasattr(self, '_reg_panel'):
            self._config.pixel_size_mm = float(getattr(
                self._reg_panel, '_pixel_size_mm', self._config.pixel_size_mm,
            ))
            if hasattr(self, '_reg_panel') and hasattr(self._reg_panel, '_camera_settings_for_config'):
                cam = self._reg_panel._camera_settings_for_config()
                if cam is not None:
                    self._config.camera = cam
            if hasattr(self._reg_panel, 'cleanup'):
                self._reg_panel.cleanup()
        self._config.save()
        event.accept()

    # ── dark theme stylesheet ──────────────────────────────────────

    @staticmethod
    def _dark_theme() -> str:
        return """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: "Segoe UI", "Ubuntu", sans-serif;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #cccccc;
                border-bottom: 1px solid #3d3d3d;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
            }
            QMenu::item:selected {
                background-color: #264f78;
            }
            QStatusBar {
                background-color: #007acc;
                color: white;
                font-size: 12px;
            }
            QStatusBar QLabel {
                color: white;
                padding: 2px 8px;
            }
            QToolBar QToolButton {
                color: #cccccc;
                background: transparent;
                border: 1px solid transparent;
                padding: 4px 12px;
                border-radius: 3px;
            }
            QToolBar QToolButton:hover {
                background: #3d2d2d;
                border-color: #555;
            }
            QToolBar QToolButton:checked {
                background: #264f78;
                border-color: #007acc;
            }
            QSplitter::handle {
                background-color: #3d3d3d;
                width: 2px;
            }
            QLineEdit {
                background: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3d3d3d;
                border-radius: 3px;
                padding: 4px 8px;
            }
            QLineEdit:focus {
                border-color: #007acc;
            }
        """
