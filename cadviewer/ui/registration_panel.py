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
    from ..camera.settings_widget import CameraSettingsWidget
except ImportError:
    HAS_CAMERA = False
    MindVisionCamera = None
    CameraSettings = None
    CameraPreviewWidget = None
    CameraSettingsWidget = None

import numpy as np


class RegistrationPanel(QWidget):
    """Panel for creating and managing registration groups."""

    def __init__(
        self,
        manager: RegistrationManager,
        repo: FeatureRepository,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._repo = repo
        self._selected_group_id: Optional[str] = None

        # Camera state
        self._camera: Optional[MindVisionCamera] = None
        self._camera_devices: list = []
        self._camera_open: bool = False
        self._settings_visible: bool = False

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

        # Capture + Settings row
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

        self._btn_settings = QPushButton("Settings")
        self._btn_settings.clicked.connect(self._toggle_settings)
        self._btn_settings.setEnabled(False)
        self._btn_settings.setStyleSheet("""
            QPushButton {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 4px 10px; border-radius: 3px;
            }
            QPushButton:hover { background: #444; }
        """)
        capture_row.addWidget(self._btn_settings)

        cam_layout.addLayout(capture_row)

        # Settings widget (hidden by default)
        self._camera_settings = CameraSettingsWidget()
        self._camera_settings.settings_changed.connect(self._apply_camera_settings)
        self._camera_settings.setVisible(False)
        cam_layout.addWidget(self._camera_settings)

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

            # Populate settings ranges and current values
            ranges = self._camera.get_setting_ranges()
            self._camera_settings.set_ranges(ranges)
            current = self._camera.get_current_settings()
            self._camera_settings.set_values(current)

            self._camera_open = True
            self._btn_open.setEnabled(False)
            self._btn_close.setEnabled(True)
            self._btn_capture.setEnabled(True)
            self._btn_settings.setEnabled(True)
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
        self._btn_settings.setEnabled(False)
        self._camera_preview.set_placeholder_text("Camera closed")
        self._camera_status.setText("Camera closed")

    def _capture_from_camera(self) -> None:
        """Capture current frame and load it as the image layer."""
        if not HAS_CAMERA or self._camera is None:
            return

        frame = self._camera_preview.get_latest_frame()
        if frame is None:
            self._camera_status.setText("No frame to capture")
            return

        # Load into image layer
        if hasattr(self, '_canvas'):
            self._canvas.get_image_layer().load_from_array(frame)
            self._image_path_label.setText("<camera capture>")
            self._pixel_size_mm = 0.01  # Default pixel size; user may adjust
            self._btn_run_coarse.setEnabled(True)
            self._btn_run_fine.setEnabled(False)
            self._btn_run_full.setEnabled(True)
            self._reg_status.setText("Frame captured. Ready for registration.")
            self._canvas.update()
            bus.image_loaded.emit("<camera_capture>")

    def _toggle_settings(self) -> None:
        """Show/hide camera settings widget."""
        self._settings_visible = not self._settings_visible
        self._camera_settings.setVisible(self._settings_visible)

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

    @Slot(str)
    def _on_group_deleted(self, group_id: str) -> None:
        self._refresh_group_list()
        self._refresh_feature_list()

    @Slot(str)
    def _on_group_contents_changed(self, group_id: str) -> None:
        if group_id == self._selected_group_id:
            self._refresh_feature_list()
            self._refresh_statistics()
        self._refresh_group_list()

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
        self._manager.set_repository(repo)
        self._refresh_group_list()
        self._refresh_feature_list()

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
        dialog = ImageLoadDialog(self)
        if dialog.exec() == ImageLoadDialog.Accepted:
            path, pixel_size = dialog.get_values()
            if path and hasattr(self, '_canvas'):
                self._canvas.get_image_layer().load_image(path)
                self._canvas.get_image_layer().set_pixel_size_mm(pixel_size)
                self._image_path_label.setText(path.split('/')[-1])
                self._pixel_size_mm = pixel_size
                self._btn_run_coarse.setEnabled(True)
                self._btn_run_fine.setEnabled(False)
                self._btn_run_full.setEnabled(True)
                self._reg_status.setText("Image loaded. Ready for registration.")
                self._canvas.update()
                bus.image_loaded.emit(path)

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
