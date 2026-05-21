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

from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtGui import QAction, QKeySequence, QIcon
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QSplitter, QToolBar,
    QFileDialog, QStatusBar, QMessageBox, QLabel, QWidget,
    QHBoxLayout, QVBoxLayout, QProgressBar, QDockWidget,
)

from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from ..parsers.dxf_importer import DXFImporter
from ..renderers.cad_canvas import CADViewerCanvas
from ..ui.tree_panel import FeatureTreePanel
from ..ui.property_panel import PropertyPanel
from ..ui.registration_panel import RegistrationPanel
from ..core.signals import bus


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

        # Build UI
        self._setup_ui()
        self._setup_toolbar()
        self._setup_menu()
        self._setup_statusbar()
        self._setup_dock_widgets()
        self._connect_signals()

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
        reg_dock = QDockWidget("Registration", self)
        reg_dock.setWidget(self._reg_panel)
        reg_dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, reg_dock)
        reg_dock.hide()

        # Store reference for toolbar toggle
        self._reg_dock = reg_dock

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

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About CAD Inspection Tool",
            "CAD Inspection Tool v1.0\n\n"
            "A metrology-oriented DXF feature inspection tool\n"
            "for machine vision alignment and automatic dimension measurement.\n\n"
            "Built with PySide6 + QPainter (OpenCascade optional)"
        )

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