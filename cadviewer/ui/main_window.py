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

from PySide6.QtCore import Qt, Slot, QSize, Signal, QObject
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QToolBar,
    QFileDialog, QStatusBar, QMessageBox, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout, QProgressBar, QDockWidget, QDialog,
)

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

        # Build UI
        self._setup_ui()
        self._setup_toolbar()
        self._setup_menu()
        self._setup_statusbar()
        self._setup_dock_widgets()
        self._connect_signals()
        self._check_dwg_converter()

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
        query_panel_action = QAction("Query Panel", self)
        query_panel_action.setCheckable(True)
        query_panel_action.toggled.connect(self._toggle_query_panel)
        view_menu.addAction(query_panel_action)
        self._query_panel_action = query_panel_action

        # Settings menu
        settings_menu = menubar.addMenu("Settings")
        oda_config_action = QAction("Configure DWG Converter...", self)
        oda_config_action.triggered.connect(self._show_dwg_settings)
        settings_menu.addAction(oda_config_action)

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
        self._reg_panel = RegistrationPanel(self._reg_manager, self._repo)
        self._reg_panel._canvas = self._viewer
        reg_dock = QDockWidget("Registration", self)
        reg_dock.setWidget(self._reg_panel)
        reg_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, reg_dock)
        reg_dock.hide()

        # Store reference for toolbar toggle
        self._reg_dock = reg_dock

        # Query panel
        self._query_panel = QueryPanel()
        query_dock = QDockWidget("Measurement Queries", self)
        query_dock.setWidget(self._query_panel)
        query_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea | Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, query_dock)
        query_dock.hide()
        self._query_dock = query_dock

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
        self._status_label.setText(f"Loading {Path(path).name}...")
        QApplication.processEvents()

        # Parse
        self._repo = self._importer.import_file(path)
        count = self._repo.count()

        self._status_label.setText(f"Loaded {count} features from {Path(path).name}")
        self._feature_count_label.setText(f"Features: {count}")

        # Populate tree
        self._tree_panel.populate(self._repo)

        # Update property panel repo reference
        self._property_panel._repo = self._repo

        # Render in viewer
        self._viewer.load_repository(self._repo)

        # Update registration manager with new repo
        self._reg_manager.set_repository(self._repo)
        self._reg_panel.set_repository(self._repo)
        self._tree_panel.set_registration_manager(self._reg_manager)
        self._viewer.set_registration_manager(self._reg_manager)

        # Create registration pipeline for new repo
        self._pipeline = RegistrationPipeline(self._repo, self._reg_manager)
        self._reg_panel.set_pipeline(self._pipeline)

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
            self._status_label.setText(f"Selected: {feature.display_name}")

    @Slot()
    def _on_feature_deselected(self) -> None:
        """Handle feature deselection."""
        self._property_panel.clear()
        self._status_label.setText("Ready")

    @Slot(str)
    def _on_viewer_click(self, feature_id: str) -> None:
        """Handle feature click in viewer — sync with tree."""
        self._tree_panel.select_feature(feature_id)
        bus.property_update.emit({"feature_id": feature_id})

    @Slot(int)
    def _on_features_loaded(self, count: int) -> None:
        self._feature_count_label.setText(f"Features: {count}")

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
            self._query_dock.show()
        else:
            self._query_dock.hide()

    @Slot()
    def _open_dwg(self) -> None:
        """Open a DWG file, convert to DXF, then load."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import DWG File", str(Path.cwd()),
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
                self, "DWG Converter Not Found",
                "No DWG converter is installed.\n\n"
                "Install one of:\n"
                "  • ODA File Converter: https://www.opendesign.com/guestfiles/oda_file_converter\n"
                "  • libredwg: sudo apt install libredwg-utils\n\n"
                "Then configure via Settings → Configure DWG Converter.",
            )
            return

        # Show progress dialog
        dialog = DWGImportDialog(self)
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

    def _check_dwg_converter(self) -> None:
        """Check ODA availability at startup, disable DWG button if missing."""
        info = self._dwg_converter.validate_installation()
        if not info.installed:
            self._dwg_action.setEnabled(False)
            self._dwg_action.setToolTip(
                "ODA File Converter not found — install and configure via Settings menu"
            )

    def closeEvent(self, event) -> None:
        """Handle window close — cleanup camera resources."""
        # Cleanup registration panel (which handles camera cleanup)
        if hasattr(self, '_reg_panel') and hasattr(self._reg_panel, 'cleanup'):
            self._reg_panel.cleanup()
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