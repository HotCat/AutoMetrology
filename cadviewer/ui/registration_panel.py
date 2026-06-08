"""
RegistrationPanel — dockable panel for managing registration groups.

Provides:
  - Group list with color swatches
  - Create / Rename / Delete group buttons
  - Feature list for selected group
  - Add/remove features from groups
  - Group statistics (type counts, centroid, feature count)
  - Zoom to Group button
"""

from __future__ import annotations

import json
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit,
    QInputDialog, QAbstractItemView, QSplitter, QComboBox, QCheckBox,
)

from ..models.feature import FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationGroup, RegistrationManager
from ..core.signals import bus
from .image_load_dialog import ImageLoadDialog

# Optional camera import
try:
    from ..camera import HAS_CAMERA, MindVisionCamera, CameraSettings
    from ..camera.preview_widget import CameraPreviewWidget
    from ..camera.live_window import CameraLiveWindow
except ImportError:
    HAS_CAMERA = False
    MindVisionCamera = None
    CameraSettings = None
    CameraPreviewWidget = None
    CameraLiveWindow = None

import numpy as np


class RegistrationPanel(QWidget):
    """Panel for creating and managing registration groups."""

    def __init__(
        self,
        manager: RegistrationManager,
        repo: FeatureRepository,
        config=None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._repo = repo
        self._config = config
        self._selected_group_id: Optional[str] = None
        self._auto_cad_ids = ["", ""]
        self._image_calibration_applied = False
        self._auto_source_image_path = ""

        # Pixel size from config
        if config is not None:
            self._pixel_size_mm = config.pixel_size_mm
        else:
            self._pixel_size_mm = 0.01

        # Camera state
        self._camera: Optional[MindVisionCamera] = None
        self._camera_devices: list = []
        self._camera_open: bool = False
        self._live_window: Optional[CameraLiveWindow] = None

        self._setup_ui()
        self._setup_camera_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Registration Groups")
        header.setStyleSheet(
            "font-weight: bold; padding: 6px; background: #2d2d2d; color: #ddd;"
        )
        layout.addWidget(header)

        # Group list
        self._group_list = QListWidget()
        self._group_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._group_list.currentItemChanged.connect(self._on_group_selected)
        self._group_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: none;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background-color: #264f78;
            }
        """)
        layout.addWidget(self._group_list)

        # Group CRUD buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self._btn_create = QPushButton("New")
        self._btn_create.clicked.connect(self._create_group)
        self._btn_rename = QPushButton("Rename")
        self._btn_rename.clicked.connect(self._rename_group)
        self._btn_delete = QPushButton("Delete")
        self._btn_delete.clicked.connect(self._delete_group)

        for btn in [self._btn_create, self._btn_rename, self._btn_delete]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #333; color: #ccc; border: 1px solid #555;
                    padding: 4px 10px; border-radius: 3px;
                }
                QPushButton:hover { background: #444; }
            """)
            btn_layout.addWidget(btn)

        layout.addLayout(btn_layout)

        # Feature management section
        feat_group = QGroupBox("Group Features")
        feat_group.setStyleSheet("""
            QGroupBox {
                color: #aaa; font-weight: bold; font-size: 11px;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
            }
        """)
        feat_layout = QVBoxLayout(feat_group)

        self._feature_list = QListWidget()
        self._feature_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._feature_list.setStyleSheet("""
            QListWidget {
                background-color: #1a1a1a; color: #bbb;
                border: none; font-size: 11px;
            }
        """)
        feat_layout.addWidget(self._feature_list)

        feat_btn_layout = QHBoxLayout()
        self._btn_add = QPushButton("Add Selected Feature")
        self._btn_add.clicked.connect(self._add_selected_feature)
        self._btn_remove = QPushButton("Remove")
        self._btn_remove.clicked.connect(self._remove_feature)
        for btn in [self._btn_add, self._btn_remove]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #333; color: #ccc; border: 1px solid #555;
                    padding: 3px 8px; border-radius: 3px; font-size: 11px;
                }
                QPushButton:hover { background: #444; }
            """)
            feat_btn_layout.addWidget(btn)
        feat_layout.addLayout(feat_btn_layout)

        layout.addWidget(feat_group)

        # Statistics section
        stats_group = QGroupBox("Statistics")
        stats_group.setStyleSheet("""
            QGroupBox {
                color: #aaa; font-weight: bold; font-size: 11px;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
            }
        """)
        stats_layout = QFormLayout(stats_group)
        stats_layout.setLabelAlignment(Qt.AlignRight)

        self._stats_label = QLabel("—")
        self._stats_label.setStyleSheet("color: #ddd; font-size: 11px;")
        self._centroid_label = QLabel("—")
        self._centroid_label.setStyleSheet("color: #ddd; font-size: 11px;")
        self._types_label = QLabel("—")
        self._types_label.setStyleSheet("color: #ddd; font-size: 11px;")

        for label, widget in [
            ("Features:", self._stats_label),
            ("Centroid:", self._centroid_label),
            ("Types:", self._types_label),
        ]:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #888; font-size: 11px;")
            stats_layout.addRow(lbl, widget)

        layout.addWidget(stats_group)

        # ── Image Registration section ──
        reg_group = QGroupBox("Image Registration")
        reg_group.setStyleSheet("""
            QGroupBox {
                color: #aaa; font-weight: bold; font-size: 11px;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
            }
        """)
        reg_layout = QVBoxLayout(reg_group)

        self._btn_load_image = QPushButton("Load Image...")
        self._btn_load_image.clicked.connect(self._load_image)
        self._btn_load_image.setStyleSheet("""
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 4px 10px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
        """)
        reg_layout.addWidget(self._btn_load_image)

        self._image_path_label = QLabel("No image loaded")
        self._image_path_label.setStyleSheet("color: #666; font-size: 10px;")
        reg_layout.addWidget(self._image_path_label)

        # Registration method dropdown
        method_row = QHBoxLayout()
        method_label = QLabel("Method:")
        method_label.setStyleSheet("color: #aaa; font-size: 11px;")
        method_row.addWidget(method_label)

        self._method_combo = QComboBox()
        self._method_combo.setStyleSheet("""
            QComboBox {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 4px; border-radius: 3px; min-width: 160px;
            }
            QComboBox:drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #333; color: #ccc; selection-background-color: #264f78;
            }
        """)
        self._method_combo.addItem("Full Silhouette", "full_silhouette")
        self._method_combo.addItem("Convex Hull (partial FOV)", "convex_hull")
        self._method_combo.addItem("Fiducial-Based", "fiducial")
        self._method_combo.addItem("Teach + ICP", "teach_icp")
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        method_row.addWidget(self._method_combo)
        reg_layout.addLayout(method_row)

        # Anchor configuration
        anchor_row = QHBoxLayout()
        anchor_row.addWidget(QLabel("Anchors:"))
        self._anchor_edit = QLineEdit()
        self._anchor_edit.setPlaceholderText("DXF handles, e.g. 120C3,12121")
        self._anchor_edit.setStyleSheet(
            "QLineEdit { background: #333; color: #ccc; border: 1px solid #555; "
            "padding: 2px 4px; border-radius: 2px; font-size: 10px; }"
        )
        anchor_row.addWidget(self._anchor_edit)
        self._btn_auto_anchors = QPushButton("Auto")
        self._btn_auto_anchors.setFixedWidth(40)
        self._btn_auto_anchors.setStyleSheet(
            "QPushButton { background: #444; color: #ccc; border: 1px solid #555; "
            "padding: 2px; border-radius: 2px; font-size: 10px; }"
            "QPushButton:hover { background: #555; }"
        )
        self._btn_auto_anchors.clicked.connect(self._auto_detect_anchors)
        anchor_row.addWidget(self._btn_auto_anchors)
        reg_layout.addLayout(anchor_row)

        self._btn_run_coarse = QPushButton("Coarse Registration")
        self._btn_run_coarse.clicked.connect(self._run_coarse)
        self._btn_run_fine = QPushButton("Refine (Contour ICP)")
        self._btn_run_fine.clicked.connect(self._run_fine)
        self._btn_run_full = QPushButton("Full Registration")
        self._btn_run_full.clicked.connect(self._run_full)

        for btn in [self._btn_run_coarse, self._btn_run_fine, self._btn_run_full]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #264f78; color: white; border: none;
                    padding: 4px 8px; border-radius: 3px; font-size: 11px;
                }
                QPushButton:hover { background: #306898; }
                QPushButton:disabled { background: #333; color: #666; }
            """)
            btn.setEnabled(False)
            reg_layout.addWidget(btn)

        self._reg_status = QLabel("—")
        self._reg_status.setStyleSheet("color: #aaa; font-size: 10px;")
        reg_layout.addWidget(self._reg_status)

        # Debug checkbox
        self._btn_debug = QCheckBox("Show Debug Overlay")
        self._btn_debug.setStyleSheet("color: #aaa; font-size: 10px;")
        self._btn_debug.toggled.connect(self._toggle_debug)
        reg_layout.addWidget(self._btn_debug)

        # Teach pose controls
        teach_row = QHBoxLayout()

        self._btn_teach = QPushButton("Teach Initial Pose")
        self._btn_teach.setStyleSheet("""
            QPushButton {
                background: #446622; color: #bbdd66; border: none;
                padding: 4px 8px; border-radius: 3px; font-size: 11px;
            }
            QPushButton:hover { background: #558833; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self._btn_teach.setEnabled(False)
        self._btn_teach.clicked.connect(self._start_teach_mode)
        teach_row.addWidget(self._btn_teach)

        self._btn_save_pose = QPushButton("Save Pose Template")
        self._btn_save_pose.setStyleSheet("""
            QPushButton {
                background: #264f78; color: white; border: none;
                padding: 4px 8px; border-radius: 3px; font-size: 11px;
            }
            QPushButton:hover { background: #306898; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self._btn_save_pose.setEnabled(False)
        self._btn_save_pose.clicked.connect(self._save_pose_template)
        teach_row.addWidget(self._btn_save_pose)

        self._btn_clear_teach = QPushButton("Clear")
        self._btn_clear_teach.setStyleSheet("""
            QPushButton {
                background: #553333; color: #cc8888; border: none;
                padding: 4px 8px; border-radius: 3px; font-size: 11px;
            }
            QPushButton:hover { background: #664444; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        self._btn_clear_teach.setEnabled(False)
        self._btn_clear_teach.clicked.connect(self._clear_teach_points)
        teach_row.addWidget(self._btn_clear_teach)

        reg_layout.addLayout(teach_row)

        # Automatic two-point correspondence from selected CAD circles + image ROIs
        auto_group = QGroupBox("Auto 2-Point Correspondence")
        auto_group.setStyleSheet(reg_group.styleSheet())
        auto_layout = QVBoxLayout(auto_group)

        auto_cad1_row = QHBoxLayout()
        auto_cad1_row.addWidget(QLabel("CAD P1:"))
        self._auto_cad1_edit = QLineEdit()
        self._auto_cad1_edit.setPlaceholderText("Select CAD circle, click Use")
        self._auto_cad1_edit.setStyleSheet(self._anchor_edit.styleSheet())
        auto_cad1_row.addWidget(self._auto_cad1_edit)
        self._btn_auto_cad1 = QPushButton("Use")
        self._btn_auto_cad1.setFixedWidth(42)
        self._btn_auto_cad1.clicked.connect(lambda: self._set_auto_cad_fiducial(0))
        auto_cad1_row.addWidget(self._btn_auto_cad1)
        auto_layout.addLayout(auto_cad1_row)

        auto_cad2_row = QHBoxLayout()
        auto_cad2_row.addWidget(QLabel("CAD P2:"))
        self._auto_cad2_edit = QLineEdit()
        self._auto_cad2_edit.setPlaceholderText("Select CAD circle, click Use")
        self._auto_cad2_edit.setStyleSheet(self._anchor_edit.styleSheet())
        auto_cad2_row.addWidget(self._auto_cad2_edit)
        self._btn_auto_cad2 = QPushButton("Use")
        self._btn_auto_cad2.setFixedWidth(42)
        self._btn_auto_cad2.clicked.connect(lambda: self._set_auto_cad_fiducial(1))
        auto_cad2_row.addWidget(self._btn_auto_cad2)
        auto_layout.addLayout(auto_cad2_row)

        roi1_row = QHBoxLayout()
        roi1_row.addWidget(QLabel("ROI P1:"))
        self._auto_roi1_edit = QLineEdit()
        self._auto_roi1_edit.setPlaceholderText("x,y,w,h")
        self._auto_roi1_edit.setStyleSheet(self._anchor_edit.styleSheet())
        roi1_row.addWidget(self._auto_roi1_edit)
        auto_layout.addLayout(roi1_row)

        roi2_row = QHBoxLayout()
        roi2_row.addWidget(QLabel("ROI P2:"))
        self._auto_roi2_edit = QLineEdit()
        self._auto_roi2_edit.setPlaceholderText("x,y,w,h")
        self._auto_roi2_edit.setStyleSheet(self._anchor_edit.styleSheet())
        roi2_row.addWidget(self._auto_roi2_edit)
        auto_layout.addLayout(roi2_row)

        auto_btn_row = QHBoxLayout()
        self._btn_pick_auto_rois = QPushButton("Pick ROIs...")
        self._btn_pick_auto_rois.clicked.connect(self._pick_auto_rois)
        self._btn_auto_register = QPushButton("Auto Register")
        self._btn_auto_register.clicked.connect(self._run_auto_correspondence)
        self._btn_save_auto_cfg = QPushButton("Save Cfg")
        self._btn_save_auto_cfg.clicked.connect(self._save_auto_correspondence_config)
        self._btn_load_auto_cfg = QPushButton("Load Cfg")
        self._btn_load_auto_cfg.clicked.connect(self._load_auto_correspondence_config)
        for btn in [self._btn_pick_auto_rois, self._btn_auto_register, self._btn_save_auto_cfg, self._btn_load_auto_cfg]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #264f78; color: white; border: none;
                    padding: 4px 6px; border-radius: 3px; font-size: 10px;
                }
                QPushButton:hover { background: #306898; }
                QPushButton:disabled { background: #333; color: #666; }
            """)
            auto_btn_row.addWidget(btn)
        auto_layout.addLayout(auto_btn_row)

        reg_layout.addWidget(auto_group)

        layout.addWidget(reg_group)

        # Zoom button
        self._btn_zoom = QPushButton("Zoom to Group")
        self._btn_zoom.clicked.connect(self._zoom_to_group)
        self._btn_zoom.setStyleSheet("""
            QPushButton {
                background: #264f78; color: white; border: none;
                padding: 6px; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background: #306898; }
        """)
        layout.addWidget(self._btn_zoom)

    def _setup_camera_ui(self) -> None:
        """Add camera capture section above Image Registration."""
        if not HAS_CAMERA:
            return

        layout = self.layout()

        # Camera Capture group box
        cam_group = QGroupBox("Camera Capture")
        cam_group.setStyleSheet("""
            QGroupBox {
                color: #aaa; font-weight: bold; font-size: 11px;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
            }
        """)
        cam_layout = QVBoxLayout(cam_group)

        # Device selection row
        dev_row = QHBoxLayout()
        self._camera_combo = QComboBox()
        self._camera_combo.setStyleSheet("""
            QComboBox {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 4px; border-radius: 3px; min-width: 120px;
            }
            QComboBox:drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #333; color: #ccc; selection-background-color: #264f78;
            }
        """)
        self._refresh_cameras()
        dev_row.addWidget(self._camera_combo)

        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.clicked.connect(self._refresh_cameras)
        self._btn_refresh.setStyleSheet("""
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 4px 10px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
        """)
        dev_row.addWidget(self._btn_refresh)

        self._btn_open = QPushButton("Open")
        self._btn_open.clicked.connect(self._open_camera)
        self._btn_open.setStyleSheet("""
            QPushButton {
                background: #264f78; color: white; border: none;
                padding: 4px 10px; border-radius: 3px;
            }
            QPushButton:hover { background: #306898; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        dev_row.addWidget(self._btn_open)

        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self._close_camera)
        self._btn_close.setEnabled(False)
        self._btn_close.setStyleSheet("""
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 4px 10px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        dev_row.addWidget(self._btn_close)

        cam_layout.addLayout(dev_row)

        # Preview widget
        self._camera_preview = CameraPreviewWidget()
        cam_layout.addWidget(self._camera_preview)

        # Capture + Focus Preview row
        capture_row = QHBoxLayout()
        self._btn_capture = QPushButton("Capture Frame")
        self._btn_capture.clicked.connect(self._capture_from_camera)
        self._btn_capture.setEnabled(False)
        self._btn_capture.setStyleSheet("""
            QPushButton {
                background: #264f78; color: white; border: none;
                padding: 4px 10px; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background: #306898; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        capture_row.addWidget(self._btn_capture)

        self._btn_focus_preview = QPushButton("Focus Preview")
        self._btn_focus_preview.clicked.connect(self._open_focus_preview)
        self._btn_focus_preview.setEnabled(False)
        self._btn_focus_preview.setStyleSheet("""
            QPushButton {
                background: #3a6b35; color: white; border: none;
                padding: 4px 10px; border-radius: 3px; font-weight: bold;
            }
            QPushButton:hover { background: #4a8b45; }
            QPushButton:disabled { background: #333; color: #666; }
        """)
        capture_row.addWidget(self._btn_focus_preview)

        cam_layout.addLayout(capture_row)

        # Camera status
        self._camera_status = QLabel("No camera connected")
        self._camera_status.setStyleSheet("color: #666; font-size: 10px;")
        cam_layout.addWidget(self._camera_status)

        # Insert camera group before Image Registration (find its index)
        reg_group_idx = layout.indexOf(self.findChild(QGroupBox, "Image Registration"))
        if reg_group_idx >= 0:
            layout.insertWidget(reg_group_idx, cam_group)
        else:
            layout.addWidget(cam_group)

        # Initialize camera instance
        self._camera = MindVisionCamera()
        self._camera.signals.frame_ready.connect(self._camera_preview.display_frame)
        self._camera.signals.error.connect(self._on_camera_error)

    def _refresh_cameras(self) -> None:
        """Refresh the camera device list."""
        if not HAS_CAMERA or self._camera is None:
            return

        self._camera_devices = self._camera.enumerate_devices()
        self._camera_combo.clear()

        if not self._camera_devices:
            self._camera_combo.addItem("No camera detected")
            self._btn_open.setEnabled(False)
        else:
            for dev in self._camera_devices:
                self._camera_combo.addItem(dev["name"])
            self._btn_open.setEnabled(True)

    def _open_camera(self) -> None:
        """Open selected camera and start live view."""
        if not HAS_CAMERA or self._camera is None:
            return

        idx = self._camera_combo.currentIndex()
        if idx < 0 or idx >= len(self._camera_devices):
            return

        dev_info = self._camera_devices[idx]["dev_info"]
        try:
            self._camera.open(dev_info)
            self._camera.set_live_mode()
            self._camera_open = True
            # Apply saved camera settings
            if self._config is not None:
                try:
                    saved = self._config.camera
                    self._camera.apply_settings(CameraSettings(
                        exposure_us=saved.exposure_us,
                        gamma=saved.gamma,
                        contrast=saved.contrast,
                        analog_gain=saved.analog_gain,
                        ae_enabled=saved.ae_enabled,
                        reverse_x=saved.reverse_x,
                        reverse_y=saved.reverse_y,
                    ))
                except Exception:
                    pass
            self._btn_open.setEnabled(False)
            self._btn_close.setEnabled(True)
            self._btn_capture.setEnabled(True)
            self._btn_focus_preview.setEnabled(True)
            self._camera_status.setText(f"Camera open: {self._camera_devices[idx]['name']}")
        except Exception as e:
            self._camera_status.setText(f"Error: {e}")

    def _close_camera(self) -> None:
        """Close camera and stop live view."""
        if not HAS_CAMERA or self._camera is None:
            return

        self._camera.close()
        self._camera_open = False
        self._btn_open.setEnabled(len(self._camera_devices) > 0)
        self._btn_close.setEnabled(False)
        self._btn_capture.setEnabled(False)
        self._btn_focus_preview.setEnabled(False)
        self._camera_preview.set_placeholder_text("Camera closed")
        self._camera_status.setText("Camera closed")
        if self._live_window is not None:
            self._live_window.clear()

    def _capture_from_camera(self) -> None:
        """Capture current frame and load it as the image layer."""
        if not HAS_CAMERA or self._camera is None:
            return

        frame = self._camera_preview.get_latest_frame()
        if frame is None:
            self._camera_status.setText("No frame to capture")
            return

        # Load into image layer, applying lens undistortion if available.
        if hasattr(self, '_canvas'):
            from ..registration.auto_correspondence import undistort_if_calibrated
            frame, applied = undistort_if_calibrated(frame, self._config)
            self._image_calibration_applied = applied
            self._canvas.get_image_layer().load_from_array(frame)
            self._auto_source_image_path = self._canvas.get_image_layer().path
            self._image_path_label.setText("<camera capture undistorted>" if applied else "<camera capture>")
            # Keep pixel_size_mm from config (don't reset to default)
            self._btn_run_coarse.setEnabled(True)
            self._btn_run_fine.setEnabled(False)
            self._btn_run_full.setEnabled(True)
            self._btn_teach.setEnabled(True)
            self._reg_status.setText(f"Frame captured. Ready for registration. (pixel_size={self._pixel_size_mm:.4f} mm)")
            self._canvas.update()
            bus.image_loaded.emit("<camera_capture>")

    def _open_focus_preview(self) -> None:
        """Open a dedicated full-size live preview window for focus adjustment."""
        if not HAS_CAMERA or self._camera is None or not self._camera_open:
            return

        if self._live_window is not None and self._live_window.isVisible():
            self._live_window.raise_()
            self._live_window.activateWindow()
            return

        self._live_window = CameraLiveWindow(self)
        self._live_window._btn_capture.clicked.connect(self._capture_from_camera)

        # Populate settings ranges and current values in the live window
        ranges = self._camera.get_setting_ranges()
        self._live_window.settings_widget.set_ranges(ranges)
        current = self._camera.get_current_settings()
        self._live_window.settings_widget.set_values(current)
        self._live_window.settings_widget.settings_changed.connect(
            self._apply_camera_settings,
        )

        self._camera.signals.frame_ready.connect(self._live_window.display_frame)
        self._live_window.show()

    def _on_method_changed(self, index: int) -> None:
        """Switch registration strategy when method dropdown changes."""
        if not hasattr(self, '_pipeline'):
            return
        method_key = self._method_combo.currentData()
        self._pipeline.set_strategy_by_key(method_key)
        self._reg_status.setText(f"Method: {self._method_combo.currentText()}")

    def _apply_camera_settings(self, settings: CameraSettings) -> None:
        """Apply settings from the settings widget to the camera."""
        if not HAS_CAMERA or self._camera is None or not self._camera_open:
            return
        self._camera.apply_settings(settings)

    def _on_camera_error(self, msg: str) -> None:
        """Handle camera error signal."""
        self._camera_status.setText(f"Camera error: {msg}")

    def cleanup(self) -> None:
        """Cleanup camera resources on app close."""
        if HAS_CAMERA and self._camera is not None and self._camera_open:
            self._camera.close()
        if self._live_window is not None:
            self._live_window.close()

    def _camera_settings_for_config(self):
        """Return current camera settings as CameraConfig for persistence."""
        if not HAS_CAMERA or self._camera is None:
            return None
        from ..core.config import CameraConfig
        try:
            s = self._camera.get_current_settings()
            return CameraConfig(
                exposure_us=s.exposure_us,
                gamma=s.gamma,
                contrast=s.contrast,
                analog_gain=s.analog_gain,
                ae_enabled=s.ae_enabled,
                reverse_x=s.reverse_x,
                reverse_y=s.reverse_y,
            )
        except Exception:
            return None

    def _connect_signals(self) -> None:
        bus.group_created.connect(self._on_group_created)
        bus.group_deleted.connect(self._on_group_deleted)
        bus.group_contents_changed.connect(self._on_group_contents_changed)
        bus.highlight_feature.connect(self._on_feature_highlighted)

    # ── group CRUD ────────────────────────────────────────────────

    @Slot()
    def _create_group(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Create Group", "Group name:",
            text=f"Group {self._manager.group_count() + 1}",
        )
        if ok and name:
            group = self._manager.create_group(name)
            bus.group_created.emit(group.group_id)

    @Slot()
    def _rename_group(self) -> None:
        group = self._get_selected_group()
        if not group:
            return
        name, ok = QInputDialog.getText(
            self, "Rename Group", "New name:", text=group.name,
        )
        if ok and name:
            self._manager.rename_group(group.group_id, name)
            bus.group_renamed.emit(group.group_id)
            bus.group_contents_changed.emit(group.group_id)

    @Slot()
    def _delete_group(self) -> None:
        group = self._get_selected_group()
        if not group:
            return
        self._manager.delete_group(group.group_id)
        self._selected_group_id = None
        bus.group_deleted.emit(group.group_id)

    # ── feature management ────────────────────────────────────────

    @Slot()
    def _add_selected_feature(self) -> None:
        group = self._get_selected_group()
        if not group:
            return
        # Use the currently highlighted feature
        if not hasattr(self, '_last_highlighted_id') or not self._last_highlighted_id:
            return
        fid = self._last_highlighted_id
        if self._manager.add_feature_to_group(group.group_id, fid):
            bus.group_contents_changed.emit(group.group_id)

    @Slot()
    def _remove_feature(self) -> None:
        group = self._get_selected_group()
        if not group:
            return
        item = self._feature_list.currentItem()
        if not item:
            return
        fid = item.data(Qt.UserRole)
        self._manager.remove_feature_from_group(group.group_id, fid)
        bus.group_contents_changed.emit(group.group_id)

    @Slot()
    def _zoom_to_group(self) -> None:
        group = self._get_selected_group()
        if not group:
            return
        bbox = group.bbox(self._repo)
        if not bbox:
            return
        fmin_x, fmin_y, fmax_x, fmax_y = bbox
        pad = max(fmax_x - fmin_x, fmax_y - fmin_y) * 0.3
        if pad < 10:
            pad = 30
        dx = (fmax_x - fmin_x) + pad * 2
        dy = (fmax_y - fmin_y) + pad * 2
        w, h = self.width(), self.height()
        if w == 0 or h == 0:
            return
        bus.view_fit_all.emit()

    # ── signal handlers ──────────────────────────────────────────

    @Slot(str)
    def _on_feature_highlighted(self, feature_id: str) -> None:
        self._last_highlighted_id = feature_id

    @Slot(str)
    def _on_group_created(self, group_id: str) -> None:
        self._refresh_group_list()
        # Select the new group
        for i in range(self._group_list.count()):
            item = self._group_list.item(i)
            if item.data(Qt.UserRole) == group_id:
                self._group_list.setCurrentItem(item)
                break
        # Persist
        if self._config is not None:
            self._config.registration_groups = self._manager.save_groups()
            self._config.save()

    @Slot(str)
    def _on_group_deleted(self, group_id: str) -> None:
        self._refresh_group_list()
        self._refresh_feature_list()
        if self._config is not None:
            self._config.registration_groups = self._manager.save_groups()
            self._config.save()

    @Slot(str)
    def _on_group_contents_changed(self, group_id: str) -> None:
        if group_id == self._selected_group_id:
            self._refresh_feature_list()
            self._refresh_statistics()
        self._refresh_group_list()
        # Persist groups to config whenever contents change
        if self._config is not None:
            self._config.registration_groups = self._manager.save_groups()
            self._config.save()

    # ── selection ────────────────────────────────────────────────

    def _on_group_selected(self, current, previous) -> None:
        if current:
            self._selected_group_id = current.data(Qt.UserRole)
        else:
            self._selected_group_id = None
        self._refresh_feature_list()
        self._refresh_statistics()

    def _get_selected_group(self) -> Optional[RegistrationGroup]:
        if not self._selected_group_id:
            return None
        return self._manager.get_group(self._selected_group_id)

    # ── refresh helpers ──────────────────────────────────────────

    def _refresh_group_list(self) -> None:
        selected_id = self._selected_group_id
        self._group_list.clear()
        for group in self._manager.all_groups():
            item = QListWidgetItem(f"  {group.name} ({group.feature_count})")
            item.setData(Qt.UserRole, group.group_id)
            # Color swatch via text color
            item.setForeground(group.color)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            self._group_list.addItem(item)
            if group.group_id == selected_id:
                self._group_list.setCurrentItem(item)

    def _refresh_feature_list(self) -> None:
        self._feature_list.clear()
        group = self._get_selected_group()
        if not group:
            return
        for fid in group.feature_ids:
            feat = self._repo.get(fid)
            if feat:
                item = QListWidgetItem(feat.display_name)
                item.setData(Qt.UserRole, fid)
                self._feature_list.addItem(item)

    def _refresh_statistics(self) -> None:
        group = self._get_selected_group()
        if not group:
            self._stats_label.setText("—")
            self._centroid_label.setText("—")
            self._types_label.setText("—")
            return

        self._stats_label.setText(str(group.feature_count))
        centroid = group.centroid(self._repo)
        if centroid:
            self._centroid_label.setText(f"({centroid[0]:.2f}, {centroid[1]:.2f})")
        else:
            self._centroid_label.setText("—")
        stats = group.type_statistics(self._repo)
        if stats:
            parts = [f"{ft.name}: {c}" for ft, c in sorted(stats.items(), key=lambda x: x[0].name)]
            self._types_label.setText(", ".join(parts))
        else:
            self._types_label.setText("—")

    def set_repository(self, repo: FeatureRepository) -> None:
        self._repo = repo
        self._manager._repo = repo

    # ── image registration ────────────────────────────────────────

    def set_pipeline(self, pipeline) -> None:
        """Set the registration pipeline (from MainWindow)."""
        self._pipeline = pipeline

    def _compute_image_affine(self, registration_transform: np.ndarray) -> np.ndarray:
        """Convert registration transform (CAD→image_world) to image layer affine (pixel→CAD).

        The registration pipeline returns T such that T @ cad_point ≈ image_world_point.
        The image layer needs an affine mapping pixel → CAD world:
          pixel → image_world (scale by pixel_size_mm, flip Y)
          image_world → CAD (invert registration transform)
        """
        ps = self._pixel_size_mm
        T_pixel_to_imgworld = np.array([
            [ps,  0,   0],
            [0,  -ps,  0],
            [0,   0,   1],
        ], dtype=np.float64)
        T_imgworld_to_cad = np.linalg.inv(registration_transform)
        return T_imgworld_to_cad @ T_pixel_to_imgworld

    def _load_image(self) -> None:
        camera = self._camera if (HAS_CAMERA and self._camera_open) else None
        dialog = ImageLoadDialog(
            self,
            default_pixel_size=self._pixel_size_mm,
            camera=camera,
            config=self._config,
        )
        if dialog.exec() == ImageLoadDialog.Accepted:
            path, pixel_size = dialog.get_values()
            captured = dialog.get_captured_frame()
            if hasattr(self, '_canvas'):
                if captured is not None:
                    self._canvas.get_image_layer().load_from_array(captured)
                    self._image_calibration_applied = True
                    self._auto_source_image_path = self._canvas.get_image_layer().path
                    self._image_path_label.setText("<camera capture undistorted>")
                elif path:
                    self._canvas.get_image_layer().load_image(path)
                    self._image_calibration_applied = False
                    self._auto_source_image_path = path
                    self._image_path_label.setText(path.split('/')[-1])
                else:
                    return
                self._canvas.get_image_layer().set_pixel_size_mm(pixel_size)
                self._pixel_size_mm = pixel_size
                self._btn_run_coarse.setEnabled(True)
                self._btn_run_fine.setEnabled(False)
                self._btn_run_full.setEnabled(True)
                self._reg_status.setText("Image loaded. Ready for registration.")
                self._canvas.update()
                bus.image_loaded.emit(path or "<camera_capture>")
            if self._config:
                self._config.pixel_size_mm = pixel_size

    def _current_auto_rois(self) -> list[Optional[tuple[int, int, int, int]]]:
        rois: list[Optional[tuple[int, int, int, int]]] = [None, None]
        for i, edit in enumerate([self._auto_roi1_edit, self._auto_roi2_edit]):
            text = edit.text().strip()
            if not text:
                continue
            try:
                rois[i] = self._parse_auto_roi(text, f"ROI P{i + 1}")
            except Exception:
                rois[i] = None
        return rois

    def _pick_auto_rois(self) -> None:
        try:
            image = self._ensure_auto_detection_image()
            from .roi_selector_dialog import ROISelectorDialog
            dialog = ROISelectorDialog(image, self._current_auto_rois(), self)
            if dialog.exec() != ROISelectorDialog.Accepted:
                return
            rois = dialog.get_rois()
            if rois[0] is not None:
                self._auto_roi1_edit.setText(",".join(str(v) for v in rois[0]))
            if rois[1] is not None:
                self._auto_roi2_edit.setText(",".join(str(v) for v in rois[1]))
            self._reg_status.setText("Fiducial ROIs updated from image picker")
        except Exception as e:
            self._reg_status.setText(f"ROI picker error: {e}")

    def _auto_correspondence_path(self) -> str:
        group = self._get_selected_group()
        group_id = group.group_id if group else "default"
        image_path = self._auto_source_image_path
        if not image_path and hasattr(self, '_canvas'):
            image_path = self._canvas.get_image_layer().path
        from ..registration.auto_correspondence import auto_config_path
        return auto_config_path(image_path, group_id)

    def _set_auto_cad_fiducial(self, index: int) -> None:
        fid = getattr(self, '_last_highlighted_id', "")
        feat = self._repo.get(fid) if fid else None
        if feat is None:
            self._reg_status.setText("Select a CAD circle first")
            return
        if feat.feature_type != FeatureType.CIRCLE:
            self._reg_status.setText("Selected CAD feature is not a circle")
            return
        self._auto_cad_ids[index] = feat.feature_id
        text = feat.dxf_handle or feat.feature_id[:8]
        if index == 0:
            self._auto_cad1_edit.setText(text)
        else:
            self._auto_cad2_edit.setText(text)
        g = feat.geometry
        self._reg_status.setText(
            f"CAD P{index + 1}: circle at ({g['cx']:.3f}, {g['cy']:.3f})"
        )

    def _resolve_auto_cad_feature(self, index: int):
        fid = self._auto_cad_ids[index]
        edit = self._auto_cad1_edit if index == 0 else self._auto_cad2_edit
        token = edit.text().strip()
        feat = self._repo.get(fid) if fid else None
        if feat is None and token:
            feat = self._repo.get(token)
        if feat is None and token:
            feat = self._repo.get_by_handle(token)
        if feat is None:
            raise ValueError(f"CAD P{index + 1} circle is not configured")
        if feat.feature_type != FeatureType.CIRCLE:
            raise ValueError(f"CAD P{index + 1} is not a circle")
        self._auto_cad_ids[index] = feat.feature_id
        return feat

    @staticmethod
    def _parse_auto_roi(text: str, label: str) -> tuple[int, int, int, int]:
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        if len(parts) != 4:
            raise ValueError(f"{label} must be x,y,w,h")
        vals = [int(round(float(p))) for p in parts]
        if vals[2] <= 0 or vals[3] <= 0:
            raise ValueError(f"{label} width/height must be positive")
        return vals[0], vals[1], vals[2], vals[3]

    def _ensure_auto_detection_image(self):
        if not hasattr(self, '_canvas'):
            raise ValueError("Canvas is not available")
        layer = self._canvas.get_image_layer()
        image = layer.image
        if image is None:
            raise ValueError("Load a camera image first")
        if not self._image_calibration_applied:
            from ..registration.auto_correspondence import undistort_if_calibrated
            corrected, applied = undistort_if_calibrated(image, self._config)
            if applied:
                layer.load_from_array(corrected)
                layer.set_pixel_size_mm(self._pixel_size_mm)
                self._image_calibration_applied = True
                self._image_path_label.setText("<undistorted for registration>")
                image = layer.image
                self._canvas.update()
                self._reg_status.setText("Lens calibration applied to registration image")
        return image

    def _auto_cad_points(self, f1, f2) -> list[dict]:
        points = []
        for label, feat in [("P1", f1), ("P2", f2)]:
            g = feat.geometry
            points.append({
                "label": label,
                "world": [float(g["cx"]), float(g["cy"])],
                "feature_id": feat.feature_id,
                "dxf_handle": feat.dxf_handle,
                "radius": float(g.get("radius", 0.0)),
            })
        return points

    def _auto_image_points(self, d1, d2) -> list[dict]:
        return [
            {"label": "P1", "pixel": [float(d1.center[0]), float(d1.center[1])]},
            {"label": "P2", "pixel": [float(d2.center[0]), float(d2.center[1])]},
        ]

    def _save_auto_correspondence_config(
        self,
        detections: Optional[list] = None,
        transform=None,
        pose_path: str = "",
    ) -> str:
        f1 = self._resolve_auto_cad_feature(0)
        f2 = self._resolve_auto_cad_feature(1)
        roi1 = self._parse_auto_roi(self._auto_roi1_edit.text(), "ROI P1")
        roi2 = self._parse_auto_roi(self._auto_roi2_edit.text(), "ROI P2")
        group = self._get_selected_group()
        image_path = self._canvas.get_image_layer().path if hasattr(self, '_canvas') else ""
        data = {
            "version": 1,
            "group_id": group.group_id if group else "default",
            "pixel_size_mm": self._pixel_size_mm,
            "image_path": image_path,
            "source_image_path": self._auto_source_image_path,
            "calibration_applied": bool(self._image_calibration_applied),
            "cad_fiducials": self._auto_cad_points(f1, f2),
            "image_rois": [list(roi1), list(roi2)],
            "pose_template_path": pose_path,
        }
        if detections:
            data["detections"] = [d.to_dict() for d in detections]
        if transform is not None:
            from ..registration import affine_solver
            params = affine_solver.extract_params(transform)
            data["transform"] = {
                "translation": [float(params["tx"]), float(params["ty"])],
                "rotation_deg": float(params["rotation_deg"]),
                "scale": float(params["scale_x"]),
            }
        path = self._auto_correspondence_path()
        from pathlib import Path
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        self._reg_status.setText(f"Auto config saved: {path}")
        return path

    def _load_auto_correspondence_config(self) -> None:
        try:
            path = self._auto_correspondence_path()
            from pathlib import Path
            p = Path(path)
            if not p.exists():
                self._reg_status.setText(f"No auto config at {path}")
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            fiducials = data.get("cad_fiducials", [])
            if len(fiducials) >= 2:
                self._auto_cad_ids[0] = fiducials[0].get("feature_id", "")
                self._auto_cad_ids[1] = fiducials[1].get("feature_id", "")
                self._auto_cad1_edit.setText(fiducials[0].get("dxf_handle") or self._auto_cad_ids[0])
                self._auto_cad2_edit.setText(fiducials[1].get("dxf_handle") or self._auto_cad_ids[1])
            rois = data.get("image_rois", [])
            if len(rois) >= 2:
                self._auto_roi1_edit.setText(",".join(str(int(v)) for v in rois[0]))
                self._auto_roi2_edit.setText(",".join(str(int(v)) for v in rois[1]))
            self._reg_status.setText(f"Auto config loaded: {path}")
        except Exception as e:
            self._reg_status.setText(f"Error loading auto config: {e}")

    def _populate_auto_debug_data(self, transform, image) -> None:
        if not hasattr(self, '_pipeline'):
            return
        group = self._get_selected_group()
        if group and group.feature_ids:
            features = [self._repo.get(fid) for fid in group.feature_ids]
            features = [f for f in features if f is not None]
        else:
            features = list(self._repo._features.values())
        from ..registration.cad_silhouette import RegistrationContourGenerator
        from ..registration.image_extractor import ImageFeatureExtractor
        generator = RegistrationContourGenerator()
        cad_points = generator.generate_point_cloud(features, density=0.5)
        img_edges = ImageFeatureExtractor.extract_edges(image)
        img_edges_world = img_edges.astype(np.float64)
        img_edges_world[:, 0] *= self._pixel_size_mm
        img_edges_world[:, 1] *= -self._pixel_size_mm
        self._pipeline.set_debug_data("coarse", {
            "cad_points": cad_points,
            "cad_centroid": cad_points.mean(axis=0) if len(cad_points) > 0 else np.zeros(2),
            "image_edges": img_edges,
            "img_edges_world": img_edges_world,
            "img_contour_world": img_edges_world,
            "transform": transform,
            "pixel_size_mm": self._pixel_size_mm,
            "image_path": self._canvas.get_image_layer().path,
            "strategy": "auto_correspondence",
        })

    def _run_auto_correspondence(self) -> None:
        try:
            from ..registration.auto_correspondence import detect_circle_in_roi
            from ..registration.strategy import TeachICPStrategy
            from ..registration import affine_solver
            from datetime import datetime

            f1 = self._resolve_auto_cad_feature(0)
            f2 = self._resolve_auto_cad_feature(1)
            roi1 = self._parse_auto_roi(self._auto_roi1_edit.text(), "ROI P1")
            roi2 = self._parse_auto_roi(self._auto_roi2_edit.text(), "ROI P2")
            image = self._ensure_auto_detection_image()

            d1 = detect_circle_in_roi(image, roi1)
            d2 = detect_circle_in_roi(image, roi2)
            if d1 is None:
                raise ValueError("No circle detected in ROI P1")
            if d2 is None:
                raise ValueError("No circle detected in ROI P2")

            cad_points = self._auto_cad_points(f1, f2)
            img_points = self._auto_image_points(d1, d2)
            T = TeachICPStrategy._compute_transform_from_points(
                cad_points, img_points, self._pixel_size_mm,
            )
            params = affine_solver.extract_params(T)

            group = self._get_selected_group()
            group_id = group.group_id if group else "default"
            image_path = self._canvas.get_image_layer().path
            pose = {
                "version": 1,
                "source": "auto_correspondence",
                "group_id": group_id,
                "pixel_size_mm": self._pixel_size_mm,
                "translation": [params["tx"], params["ty"]],
                "rotation_deg": params["rotation_deg"],
                "scale": params["scale_x"],
                "cad_points": cad_points,
                "image_points": img_points,
                "image_rois": [list(roi1), list(roi2)],
                "detections": [d1.to_dict(), d2.to_dict()],
                "calibration_applied": bool(self._image_calibration_applied),
                "created": datetime.now().isoformat(),
                "image_path": image_path,
                "source_image_path": self._auto_source_image_path,
            }
            info = {"image_path": image_path, "group_id": group_id}
            pose_path = TeachICPStrategy._pose_template_path(info)
            TeachICPStrategy._save_pose_template(pose_path, pose)
            cfg_path = self._save_auto_correspondence_config([d1, d2], T, pose_path)

            T_img = self._compute_image_affine(T)
            self._canvas.get_image_layer().set_affine_transform(T_img)
            self._coarse_transform = T
            self._populate_auto_debug_data(T, image)
            self._push_debug_data()
            self._canvas.update()
            self._btn_run_fine.setEnabled(True)
            bus.registration_completed.emit({"transform": T, "stage": "auto_correspondence", "error": 0.0})
            self._reg_status.setText(
                f"Auto registered: P1 conf={d1.confidence:.2f}, P2 conf={d2.confidence:.2f}; "
                f"rot={params['rotation_deg']:.2f} deg, cfg={cfg_path}"
            )
        except Exception as e:
            self._reg_status.setText(f"Auto registration error: {e}")
            bus.registration_failed.emit(str(e))

    def _get_anchor_handles(self) -> list[str]:
        text = self._anchor_edit.text().strip()
        if not text:
            return []
        return [h.strip() for h in text.split(",") if h.strip()]

    def _run_coarse(self) -> None:
        group = self._get_selected_group()
        if not group:
            self._reg_status.setText("Error: select a group first")
            return
        if not hasattr(self, '_pipeline'):
            self._reg_status.setText("Error: pipeline not initialized")
            return
        try:
            result = self._pipeline.run_coarse(
                self._canvas.get_image_layer().path,
                group.group_id,
                self._pixel_size_mm,
                anchor_handles=self._get_anchor_handles(),
            )
            T_img = self._compute_image_affine(result["transform"])
            self._canvas.get_image_layer().set_affine_transform(T_img)
            self._push_debug_data()
            self._canvas.update()
            self._reg_status.setText(
                f"Coarse: error={result['error']:.4f}mm"
            )
            self._coarse_transform = result["transform"]
            self._btn_run_fine.setEnabled(True)
            bus.registration_completed.emit(result)
        except Exception as e:
            self._reg_status.setText(f"Error: {e}")
            bus.registration_failed.emit(str(e))

    def _auto_detect_anchors(self) -> None:
        from ..registration.anchor_detector import AnchorHeuristic
        heuristic = AnchorHeuristic()
        candidates = heuristic.find_anchor_candidates(self._repo)
        if candidates:
            handles = [c["handle"] for c in candidates[:4]]
            self._anchor_edit.setText(",".join(handles))
            info = ", ".join([f"{c['handle']}@({c['cx']:.0f},{c['cy']:.0f})" for c in candidates[:2]])
            self._reg_status.setText(f"Auto anchors: {info}")
        else:
            self._reg_status.setText("No anchor candidates found")

    def _run_fine(self) -> None:
        group = self._get_selected_group()
        if not group:
            return
        if not hasattr(self, '_pipeline') or not hasattr(self, '_coarse_transform'):
            return
        try:
            result = self._pipeline.run_fine(
                self._coarse_transform, group.group_id,
            )
            T_img = self._compute_image_affine(result["transform"])
            self._canvas.get_image_layer().set_affine_transform(T_img)
            self._push_debug_data()
            self._canvas.update()
            self._reg_status.setText(
                f"Fine: iters={result['iterations']}, "
                f"error={result['error']:.4f}mm, "
                f"converged={result.get('converged', False)}"
            )
            bus.registration_completed.emit(result)
        except Exception as e:
            self._reg_status.setText(f"Error: {e}")
            bus.registration_failed.emit(str(e))

    def _run_full(self) -> None:
        group = self._get_selected_group()
        if not group:
            self._reg_status.setText("Error: select a group first")
            return
        if not hasattr(self, '_pipeline'):
            return
        try:
            result = self._pipeline.run_full(
                self._canvas.get_image_layer().path,
                group.group_id,
                self._pixel_size_mm,
                anchor_handles=self._get_anchor_handles(),
            )
            T_img = self._compute_image_affine(result["transform"])
            self._canvas.get_image_layer().set_affine_transform(T_img)
            self._push_debug_data()
            self._canvas.update()
            self._reg_status.setText(
                f"Full: coarse={result.get('coarse_error', 0):.4f}mm → "
                f"fine={result.get('fine_error', 0):.4f}mm, "
                f"iters={result.get('iterations', 0)}"
            )
            self._coarse_transform = result.get("coarse_transform", result["transform"])
            self._btn_run_fine.setEnabled(True)
            bus.registration_completed.emit(result)
        except Exception as e:
            self._reg_status.setText(f"Error: {e}")
            bus.registration_failed.emit(str(e))

    def _toggle_debug(self, checked: bool) -> None:
        """Toggle debug overlay on canvas."""
        if hasattr(self, '_canvas'):
            self._canvas.set_debug_mode(checked)
            if checked:
                self._push_debug_data()

    def _push_debug_data(self) -> None:
        """Push pipeline debug data to canvas for overlay rendering."""
        if hasattr(self, '_canvas') and hasattr(self, '_pipeline'):
            self._canvas.set_debug_data(self._pipeline.get_debug_data())

    # ── teach mode ──────────────────────────────────────────────────

    PHASE_INSTRUCTIONS = {
        "cad_p1": "Teach: Click CAD Point P1",
        "cad_p2": "Teach: Click CAD Point P2",
        "img_p1": "Teach: Click corresponding Image Point P1",
        "img_p2": "Teach: Click corresponding Image Point P2",
    }

    def _start_teach_mode(self) -> None:
        if not hasattr(self, '_canvas'):
            return
        self._canvas.start_teach_mode()
        self._btn_teach.setEnabled(False)
        self._btn_clear_teach.setEnabled(True)
        self._btn_save_pose.setEnabled(False)
        self._update_teach_status()
        bus.teach_point_added.connect(self._on_teach_point_added)
        bus.teach_mode_completed.connect(self._on_teach_completed)

    def _clear_teach_points(self) -> None:
        if hasattr(self, '_canvas') and self._canvas.is_teach_mode():
            self._canvas.cancel_teach_mode()
        self._btn_teach.setEnabled(True)
        self._btn_clear_teach.setEnabled(False)
        self._btn_save_pose.setEnabled(False)
        self._reg_status.setText("Teach points cleared")

    def _on_teach_point_added(self, info: dict) -> None:
        self._update_teach_status()

    def _on_teach_completed(self, info: dict) -> None:
        self._btn_save_pose.setEnabled(True)
        self._reg_status.setText(
            "Teach complete. Click 'Save Pose Template' to store."
        )

    def _update_teach_status(self) -> None:
        if hasattr(self, '_canvas'):
            phase = self._canvas.teach_phase
            msg = self.PHASE_INSTRUCTIONS.get(phase, "")
            if msg:
                self._reg_status.setText(msg)

    def _save_pose_template(self) -> None:
        """Compute transform from teach points and save as JSON template."""
        if not hasattr(self, '_canvas') or not hasattr(self, '_pipeline'):
            return

        cad_points = self._canvas.teach_cad_points
        img_points = self._canvas.teach_img_points

        if len(cad_points) < 2 or len(img_points) < 2:
            self._reg_status.setText("Error: need 2 CAD + 2 image points")
            return

        try:
            from ..registration.strategy import TeachICPStrategy
            from datetime import datetime

            # Compute transform
            T = TeachICPStrategy._compute_transform_from_points(
                cad_points, img_points, self._pixel_size_mm,
            )

            from ..registration import affine_solver
            params = affine_solver.extract_params(T)

            group = self._get_selected_group()
            group_id = group.group_id if group else "default"

            image_path = self._canvas.get_image_layer().path
            template = {
                "version": 1,
                "group_id": group_id,
                "pixel_size_mm": self._pixel_size_mm,
                "translation": [params["tx"], params["ty"]],
                "rotation_deg": params["rotation_deg"],
                "scale": params["scale_x"],
                "cad_points": cad_points,
                "image_points": img_points,
                "created": datetime.now().isoformat(),
                "image_path": image_path,
                "source_image_path": self._auto_source_image_path,
            }

            info = {"image_path": image_path, "group_id": group_id}
            path = TeachICPStrategy._pose_template_path(info)
            TeachICPStrategy._save_pose_template(path, template)

            # Apply transform immediately so user can verify
            T_img = self._compute_image_affine(T)
            self._canvas.get_image_layer().set_affine_transform(T_img)
            self._canvas.update()

            self._reg_status.setText(
                f"Pose saved to {path} "
                f"(rot={params['rotation_deg']:.2f}°, "
                f"scale={params['scale_x']:.6f})"
            )
            self._btn_save_pose.setEnabled(False)
            self._btn_clear_teach.setEnabled(False)
            self._btn_teach.setEnabled(True)

        except Exception as e:
            self._reg_status.setText(f"Error saving pose: {e}")
