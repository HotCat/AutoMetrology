"""
RegistrationPanel — dockable panel for managing registration groups.

Provides:
  - Named production parameter profiles
  - Two-point CAD/image correspondence setup
  - Fiducial ROI picking and automatic registration
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QGroupBox, QFormLayout, QLineEdit,
    QInputDialog, QAbstractItemView, QSplitter, QComboBox, QCheckBox,
    QMessageBox, QApplication,
)

from ..models.feature import FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from ..core.signals import bus
from ..core.i18n import retranslate_widget_tree, tr
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
    """Panel for production image registration parameters."""

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
        self._auto_cad_ids = ["", ""]
        self._window_edge_ids: list[str] = []
        self._window_detection_mode = "auto"
        self._image_calibration_applied = False
        self._auto_source_image_path = ""
        self._last_auto_registration = {}
        self._last_measurement_pixel_to_world = None
        self._last_display_pixel_to_world = None
        self._loading_profile_combo = False

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
        header = QLabel("Auto Registration")
        header.setStyleSheet(
            "font-weight: bold; padding: 6px; background: #2d2d2d; color: #ddd;"
        )
        self._group_header = header
        layout.addWidget(header)

        profile_group = QGroupBox("Production Parameters")
        profile_group.setStyleSheet("""
            QGroupBox {
                color: #aaa; font-weight: bold; font-size: 11px;
                border: 1px solid #333; border-radius: 4px;
                margin-top: 8px; padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 8px; padding: 0 4px;
            }
        """)
        profile_layout = QVBoxLayout(profile_group)

        profile_row = QHBoxLayout()
        profile_label = QLabel("Profile:")
        profile_label.setStyleSheet("color: #aaa; font-size: 11px;")
        profile_row.addWidget(profile_label)

        self._profile_combo = QComboBox()
        self._profile_combo.setStyleSheet("""
            QComboBox {
                background: #333; color: #ccc; border: 1px solid #555;
                padding: 4px; border-radius: 3px; min-width: 150px;
            }
            QComboBox:drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #333; color: #ccc; selection-background-color: #264f78;
            }
        """)
        self._profile_combo.currentIndexChanged.connect(
            self._on_production_profile_selected
        )
        profile_row.addWidget(self._profile_combo)
        profile_layout.addLayout(profile_row)

        profile_btn_row = QHBoxLayout()
        self._btn_save_profile = QPushButton("Save")
        self._btn_save_profile.clicked.connect(self._save_selected_production_profile)
        self._btn_save_as_profile = QPushButton("Save As...")
        self._btn_save_as_profile.clicked.connect(self._save_production_profile_as)
        self._btn_delete_profile = QPushButton("Delete")
        self._btn_delete_profile.clicked.connect(self._delete_selected_production_profile)
        for btn in [
            self._btn_save_profile, self._btn_save_as_profile,
            self._btn_delete_profile,
        ]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #333; color: #ccc; border: 1px solid #555;
                    padding: 4px 8px; border-radius: 3px; font-size: 10px;
                }
                QPushButton:hover { background: #444; }
                QPushButton:disabled { background: #252525; color: #666; }
            """)
            profile_btn_row.addWidget(btn)
        profile_layout.addLayout(profile_btn_row)

        self._profile_hint = QLabel("Saves camera settings, fiducials, and ROIs.")
        self._profile_hint.setStyleSheet("color: #777; font-size: 10px;")
        self._profile_hint.setWordWrap(True)
        profile_layout.addWidget(self._profile_hint)

        layout.addWidget(profile_group)
        self._profile_group = profile_group
        self._refresh_production_profile_combo()

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
        self._method_label = method_label
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
        self._anchor_label = QLabel("Anchors:")
        anchor_row.addWidget(self._anchor_label)
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
        self._btn_window_register = QPushButton("Window Register")
        self._btn_window_register.clicked.connect(self._run_window_line_registration)
        for btn in [self._btn_pick_auto_rois, self._btn_auto_register, self._btn_window_register]:
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

        window_group = QGroupBox("Window CAD Edges")
        window_group.setStyleSheet(reg_group.styleSheet())
        window_layout = QVBoxLayout(window_group)

        mode_row = QHBoxLayout()
        mode_label = QLabel("Detect:")
        mode_label.setStyleSheet("color: #aaa; font-size: 11px;")
        mode_row.addWidget(mode_label)
        self._window_mode_combo = QComboBox()
        self._window_mode_combo.addItem("Auto", "auto")
        self._window_mode_combo.addItem("Dark window", "dark")
        self._window_mode_combo.addItem("Bright backlight", "bright")
        self._window_mode_combo.addItem("Printed grid", "grid")
        self._window_mode_combo.currentIndexChanged.connect(
            self._on_window_detection_mode_changed
        )
        self._window_mode_combo.setStyleSheet(self._method_combo.styleSheet())
        mode_row.addWidget(self._window_mode_combo)
        window_layout.addLayout(mode_row)

        self._window_edges_edit = QLineEdit()
        self._window_edges_edit.setReadOnly(True)
        self._window_edges_edit.setPlaceholderText("Select CAD edge, click Add; need 4")
        self._window_edges_edit.setStyleSheet(self._anchor_edit.styleSheet())
        window_layout.addWidget(self._window_edges_edit)

        window_btn_row = QHBoxLayout()
        self._btn_add_window_edge = QPushButton("Add")
        self._btn_add_window_edge.clicked.connect(self._add_selected_window_edge)
        self._btn_clear_window_edges = QPushButton("Clear")
        self._btn_clear_window_edges.clicked.connect(self._clear_window_edges)
        for btn in [self._btn_add_window_edge, self._btn_clear_window_edges]:
            btn.setStyleSheet("""
                QPushButton {
                    background: #333; color: #ccc; border: 1px solid #555;
                    padding: 4px 6px; border-radius: 3px; font-size: 10px;
                }
                QPushButton:hover { background: #444; }
                QPushButton:disabled { background: #252525; color: #666; }
            """)
            window_btn_row.addWidget(btn)
        window_layout.addLayout(window_btn_row)

        reg_layout.addWidget(window_group)

        layout.addWidget(reg_group)

        self._hide_legacy_registration_controls()

    def _production_profiles(self) -> list[dict]:
        if self._config is None:
            return []
        profiles = getattr(self._config, "production_profiles", [])
        if not isinstance(profiles, list):
            profiles = []
            self._config.production_profiles = profiles
        return profiles

    def _default_production_profile(self) -> dict:
        camera = asdict(self._config.camera) if self._config is not None else {}
        return {
            "version": 1,
            "name": "Default",
            "camera": camera,
            "auto_correspondence": {
                "cad_fiducials": [],
                "image_rois": [None, None],
                "roi_texts": ["", ""],
                "source_image_path": "",
                "image_path": "",
            },
            "window_registration": {
                "edge_ids": [],
                "edge_labels": [],
            },
        }

    def _ensure_production_profiles(self) -> list[dict]:
        profiles = self._production_profiles()
        if not profiles and self._config is not None:
            profiles.append(self._default_production_profile())
            self._config.active_production_profile = "Default"
        return profiles

    def _find_production_profile(self, name: str) -> Optional[dict]:
        needle = name.strip().lower()
        for profile in self._ensure_production_profiles():
            if str(profile.get("name", "")).strip().lower() == needle:
                return profile
        return None

    def _refresh_production_profile_combo(self, select_name: str = "") -> None:
        if not hasattr(self, "_profile_combo"):
            return
        profiles = self._ensure_production_profiles()
        active = select_name
        if not active and self._config is not None:
            active = getattr(self._config, "active_production_profile", "")
        if not active and profiles:
            active = str(profiles[0].get("name", ""))

        self._loading_profile_combo = True
        self._profile_combo.clear()
        selected_index = 0
        for i, profile in enumerate(profiles):
            name = str(profile.get("name", "")).strip() or f"Profile {i + 1}"
            profile["name"] = name
            self._profile_combo.addItem(name, name)
            if name == active:
                selected_index = i
        if profiles:
            self._profile_combo.setCurrentIndex(selected_index)
        self._loading_profile_combo = False

        enabled = bool(profiles) and self._config is not None
        self._btn_save_profile.setEnabled(enabled)
        self._btn_delete_profile.setEnabled(enabled)
        self._btn_save_as_profile.setEnabled(self._config is not None)

    def _current_profile_name(self) -> str:
        if not hasattr(self, "_profile_combo"):
            return "Default"
        name = self._profile_combo.currentData()
        if not name:
            name = self._profile_combo.currentText()
        return str(name).strip() or "Default"

    def _camera_profile_dict(self) -> dict:
        cam = self._camera_settings_for_config()
        if cam is None and self._config is not None:
            cam = self._config.camera
        return asdict(cam) if cam is not None else {}

    def _fiducial_profile_entry(self, index: int) -> dict:
        edit = self._auto_cad1_edit if index == 0 else self._auto_cad2_edit
        entry = {
            "label": f"P{index + 1}",
            "feature_id": self._auto_cad_ids[index],
            "edit_text": edit.text().strip(),
        }
        feat = None
        try:
            if entry["feature_id"] or entry["edit_text"]:
                feat = self._resolve_auto_cad_feature(index)
        except Exception:
            feat = None
        if feat is not None:
            g = feat.geometry
            entry.update({
                "feature_id": feat.feature_id,
                "dxf_handle": feat.dxf_handle,
                "world": [float(g["cx"]), float(g["cy"])],
                "radius": float(g.get("radius", 0.0)),
            })
        return entry

    def _auto_correspondence_profile(self) -> dict:
        roi_edits = [self._auto_roi1_edit, self._auto_roi2_edit]
        roi_texts = [edit.text().strip() for edit in roi_edits]
        parsed_rois = []
        for i, text in enumerate(roi_texts):
            if not text:
                parsed_rois.append(None)
                continue
            try:
                parsed_rois.append(list(self._parse_auto_roi(text, f"ROI P{i + 1}")))
            except Exception:
                parsed_rois.append(None)
        image_path = ""
        if hasattr(self, "_canvas"):
            image_path = self._canvas.get_image_layer().path
        data = {
            "cad_fiducials": [
                self._fiducial_profile_entry(0),
                self._fiducial_profile_entry(1),
            ],
            "image_rois": parsed_rois,
            "roi_texts": roi_texts,
            "source_image_path": self._auto_source_image_path,
            "image_path": image_path,
        }
        if self._last_auto_registration:
            data["last_registration"] = dict(self._last_auto_registration)
        return data

    def _window_registration_profile(self) -> dict:
        labels = []
        for feature_id in self._window_edge_ids:
            feature = self._repo.get(feature_id)
            labels.append(self._window_edge_label(feature_id, feature))
        return {
            "edge_ids": list(self._window_edge_ids),
            "edge_labels": labels,
            "detection_mode": self._window_detection_mode,
        }

    def _snapshot_production_profile(self, name: str) -> dict:
        return {
            "version": 1,
            "name": name,
            "camera": self._camera_profile_dict(),
            "auto_correspondence": self._auto_correspondence_profile(),
            "window_registration": self._window_registration_profile(),
        }

    def _upsert_production_profile(self, profile: dict, silent: bool = False) -> None:
        if self._config is None:
            return
        name = str(profile.get("name", "")).strip() or "Default"
        profile["name"] = name
        profiles = [
            p for p in self._ensure_production_profiles()
            if str(p.get("name", "")).strip().lower() != name.lower()
        ]
        profiles.append(profile)
        profiles.sort(key=lambda p: str(p.get("name", "")).lower())
        self._config.production_profiles = profiles
        self._config.active_production_profile = name
        from ..core.config import CameraConfig
        camera_data = profile.get("camera", {})
        if isinstance(camera_data, dict):
            allowed = CameraConfig.__dataclass_fields__.keys()
            filtered = {k: camera_data[k] for k in allowed if k in camera_data}
            self._config.camera = CameraConfig(**filtered)
        self._config.save()
        self._refresh_production_profile_combo(name)
        if not silent:
            self._reg_status.setText(f"Production profile saved: {name}")

    def _save_selected_production_profile(self, silent: bool = False) -> None:
        name = self._current_profile_name()
        self._upsert_production_profile(
            self._snapshot_production_profile(name), silent=silent,
        )

    def _save_production_profile_as(self) -> None:
        default = self._current_profile_name()
        name, ok = QInputDialog.getText(
            self, "Save Production Parameters", "Profile name:", text=default,
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            self._reg_status.setText(tr("Production profile name is empty"))
            return
        self._upsert_production_profile(self._snapshot_production_profile(name))

    def _delete_selected_production_profile(self) -> None:
        if self._config is None:
            return
        name = self._current_profile_name()
        answer = QMessageBox.question(
            self,
            "Delete Production Profile",
            f"Delete production profile '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        profiles = [
            p for p in self._ensure_production_profiles()
            if str(p.get("name", "")).strip().lower() != name.lower()
        ]
        if not profiles:
            profiles = [self._default_production_profile()]
        self._config.production_profiles = profiles
        self._config.active_production_profile = str(profiles[0].get("name", "Default"))
        self._config.save()
        self._refresh_production_profile_combo(self._config.active_production_profile)
        self.apply_active_production_profile()
        self._reg_status.setText(f"Production profile deleted: {name}")

    def _apply_camera_profile(self, camera_data: dict) -> None:
        if self._config is None or not isinstance(camera_data, dict):
            return
        from ..core.config import CameraConfig
        allowed = CameraConfig.__dataclass_fields__.keys()
        filtered = {k: camera_data[k] for k in allowed if k in camera_data}
        try:
            self._config.camera = CameraConfig(**filtered)
        except Exception:
            return
        if HAS_CAMERA and self._camera is not None and self._camera_open:
            try:
                self._camera.apply_settings(CameraSettings(**asdict(self._config.camera)))
            except Exception as e:
                self._reg_status.setText(f"Camera profile apply error: {e}")

    def _apply_auto_correspondence_profile(self, auto_data: dict) -> None:
        if not isinstance(auto_data, dict):
            return
        fiducials = auto_data.get("cad_fiducials", [])
        edits = [self._auto_cad1_edit, self._auto_cad2_edit]
        for i in range(2):
            if i >= len(fiducials) or not isinstance(fiducials[i], dict):
                self._auto_cad_ids[i] = ""
                edits[i].clear()
                continue
            item = fiducials[i]
            self._auto_cad_ids[i] = str(item.get("feature_id", ""))
            edits[i].setText(
                str(
                    item.get("edit_text")
                    or item.get("dxf_handle")
                    or item.get("feature_id")
                    or ""
                )
            )

        roi_texts = auto_data.get("roi_texts", [])
        rois = auto_data.get("image_rois", [])
        for i, edit in enumerate([self._auto_roi1_edit, self._auto_roi2_edit]):
            text = ""
            if i < len(roi_texts):
                text = str(roi_texts[i] or "")
            if not text and i < len(rois) and rois[i]:
                text = ",".join(str(int(v)) for v in rois[i])
            edit.setText(text)

        self._auto_source_image_path = str(auto_data.get("source_image_path", ""))

        last = auto_data.get("last_registration")
        self._last_auto_registration = dict(last) if isinstance(last, dict) else {}
        self._last_measurement_pixel_to_world = None
        self._last_display_pixel_to_world = None
        display_matrix = self._last_auto_registration.get("display_pixel_to_world")
        if display_matrix is not None:
            try:
                arr = np.asarray(display_matrix, dtype=np.float64)
                if self._display_transform_is_sane(arr):
                    self._last_display_pixel_to_world = arr
            except Exception:
                self._last_display_pixel_to_world = None
        matrix = self._last_auto_registration.get("measurement_pixel_to_world")
        if matrix is not None:
            try:
                arr = np.asarray(matrix, dtype=np.float64)
                image_size = None
                metadata = (
                    getattr(
                        getattr(self._config, "lens_calibration", None),
                        "coordinate_correction",
                        {},
                    )
                    or {}
                ).get("metadata", {})
                if isinstance(metadata, dict):
                    size = metadata.get("image_size")
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        image_size = (int(size[0]), int(size[1]))
                from ..calibration.transform_safety import validate_pixel_to_world_transform
                safety = validate_pixel_to_world_transform(
                    arr, float(self._pixel_size_mm), image_size=image_size,
                )
                if arr.shape == (3, 3) and safety.safe:
                    self._last_measurement_pixel_to_world = arr
            except Exception:
                self._last_measurement_pixel_to_world = None

    def _apply_window_registration_profile(self, window_data: dict) -> None:
        if not isinstance(window_data, dict):
            self._window_edge_ids = []
            self._refresh_window_edges_edit()
            return
        edge_ids = window_data.get("edge_ids", [])
        if not isinstance(edge_ids, list):
            edge_ids = []
        self._window_edge_ids = [
            str(feature_id) for feature_id in edge_ids
            if isinstance(feature_id, str) and self._repo.get(feature_id) is not None
        ][:4]
        mode = str(window_data.get("detection_mode", "auto")).strip().lower()
        if mode not in {"auto", "dark", "bright", "grid"}:
            mode = "auto"
        self._window_detection_mode = mode
        if hasattr(self, "_window_mode_combo"):
            idx = self._window_mode_combo.findData(mode)
            if idx >= 0:
                self._window_mode_combo.blockSignals(True)
                self._window_mode_combo.setCurrentIndex(idx)
                self._window_mode_combo.blockSignals(False)
        self._refresh_window_edges_edit()

    def _apply_production_profile(self, profile: dict) -> None:
        if self._config is None or not isinstance(profile, dict):
            return
        self._apply_camera_profile(profile.get("camera", {}))

        self._apply_auto_correspondence_profile(
            profile.get("auto_correspondence", {})
        )
        self._apply_window_registration_profile(
            profile.get("window_registration", {})
        )
        self._config.active_production_profile = str(profile.get("name", ""))
        self._config.save()
        if hasattr(self, "_canvas"):
            self._apply_saved_display_transform_to_image()
            self._canvas.update()

    def _apply_saved_display_transform_to_image(self) -> None:
        if self._last_display_pixel_to_world is None or not hasattr(self, "_canvas"):
            return
        layer = self._canvas.get_image_layer()
        if layer is None or not getattr(layer, "has_image", False):
            return
        image_path = self._last_auto_registration.get("image_path")
        if image_path and getattr(layer, "path", "") != image_path:
            return
        layer.set_affine_transform(
            np.asarray(self._last_display_pixel_to_world, dtype=np.float64),
        )

    def _display_transform_is_sane(self, transform, image_size=None) -> bool:
        try:
            arr = np.asarray(transform, dtype=np.float64)
            if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
                return False
            if image_size is None and hasattr(self, "_canvas"):
                layer = self._canvas.get_image_layer()
                image = getattr(layer, "image", None)
                if image is not None:
                    image_size = (int(image.shape[1]), int(image.shape[0]))
            if image_size is None:
                metadata = (
                    getattr(
                        getattr(self._config, "lens_calibration", None),
                        "coordinate_correction",
                        {},
                    )
                    or {}
                ).get("metadata", {})
                if isinstance(metadata, dict):
                    size = metadata.get("image_size")
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        image_size = (int(size[0]), int(size[1]))
            from ..calibration.transform_safety import validate_pixel_to_world_transform
            safety = validate_pixel_to_world_transform(
                arr, float(self._pixel_size_mm), image_size=image_size,
                max_scale_error=0.12,
                max_anisotropy=1.08,
                max_field_scale_change=1.08,
            )
            return bool(safety.safe)
        except Exception:
            return False

    def apply_active_production_profile(self) -> None:
        if self._config is None:
            return
        name = getattr(self._config, "active_production_profile", "")
        profile = self._find_production_profile(name) if name else None
        if profile is None and self._ensure_production_profiles():
            profile = self._ensure_production_profiles()[0]
        if profile is None:
            return
        self._refresh_production_profile_combo(str(profile.get("name", "")))
        self._apply_production_profile(profile)

    def _on_production_profile_selected(self, index: int) -> None:
        if self._loading_profile_combo or index < 0 or self._config is None:
            return
        name = self._current_profile_name()
        profile = self._find_production_profile(name)
        if profile is None:
            return
        self._apply_production_profile(profile)
        self._reg_status.setText(f"Production profile active: {name}")

    def _hide_legacy_registration_controls(self) -> None:
        """Keep only the production auto-registration controls visible."""
        for widget in [
            self._method_label, self._method_combo,
            self._anchor_label, self._anchor_edit, self._btn_auto_anchors,
            self._btn_run_coarse, self._btn_run_fine, self._btn_run_full,
            self._btn_teach, self._btn_save_pose, self._btn_clear_teach,
        ]:
            widget.hide()

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
        self._camera_status.setText(tr("Camera closed"))
        if self._live_window is not None:
            self._live_window.clear()

    def _capture_from_camera(self, wait_for_fresh_frame: bool = False) -> bool:
        """Capture current frame and load it as the image layer."""
        if not HAS_CAMERA or self._camera is None:
            self._reg_status.setText(tr("Camera support is not available"))
            return False
        if not self._camera_open:
            self._reg_status.setText(tr("Camera is not open"))
            return False
        if not hasattr(self, '_canvas'):
            self._reg_status.setText(tr("Canvas is not available"))
            return False

        if wait_for_fresh_frame:
            start_counter = int(getattr(self._camera_preview, "frame_counter", 0))
            import time
            deadline = time.monotonic() + 0.75
            fresh_frame = False
            while time.monotonic() < deadline:
                QApplication.processEvents()
                frame = self._camera_preview.get_latest_frame()
                counter = int(getattr(self._camera_preview, "frame_counter", 0))
                if frame is not None and counter > start_counter:
                    fresh_frame = True
                    break
                time.sleep(0.01)
            if not fresh_frame:
                self._camera_status.setText(tr("No fresh frame to capture"))
                self._reg_status.setText(tr("No fresh frame to capture"))
                return False

        frame = self._camera_preview.get_latest_frame()
        if frame is None:
            self._camera_status.setText(tr("No frame to capture"))
            self._reg_status.setText(tr("No frame to capture"))
            return False

        if self._config is not None:
            self._pixel_size_mm = float(self._config.pixel_size_mm)

        # Load into image layer, applying lens undistortion if available.
        from ..registration.auto_correspondence import undistort_if_calibrated
        frame, applied = undistort_if_calibrated(frame, self._config)
        self._image_calibration_applied = applied
        self._canvas.get_image_layer().load_from_array(frame)
        self._canvas.get_image_layer().set_pixel_size_mm(self._pixel_size_mm)
        self._coarse_transform = None
        self._last_measurement_pixel_to_world = None
        self._last_display_pixel_to_world = None
        self._last_auto_registration = {}
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
        return True

    def capture_current_frame_for_production(self) -> bool:
        return self._capture_from_camera(wait_for_fresh_frame=True)

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
        bus.highlight_feature.connect(self._on_feature_highlighted)

    # ── removed registration feature groups ───────────────────────

    @Slot(str)
    def _on_feature_highlighted(self, feature_id: str) -> None:
        self._last_highlighted_id = feature_id

    def _window_edge_label(self, feature_id: str, feature=None) -> str:
        if feature is None:
            feature = self._repo.get(feature_id)
        if feature is None:
            return feature_id[:8]
        token = feature.dxf_handle or feature.feature_id[:8]
        return f"{feature.feature_type.name}[{token}]"

    def _refresh_window_edges_edit(self) -> None:
        if not hasattr(self, "_window_edges_edit"):
            return
        labels = [
            self._window_edge_label(feature_id)
            for feature_id in self._window_edge_ids
        ]
        self._window_edges_edit.setText(", ".join(labels))

    def _add_selected_window_edge(self) -> None:
        feature_id = getattr(self, "_last_highlighted_id", "")
        feature = self._repo.get(feature_id) if feature_id else None
        if feature is None:
            self._reg_status.setText("Select a CAD line edge first.")
            return
        if feature.feature_type != FeatureType.LINE:
            self._reg_status.setText(
                f"Window edge must be LINE; selected {feature.feature_type.name}."
            )
            return
        if feature_id in self._window_edge_ids:
            self._reg_status.setText("Window edge already added.")
            return
        if len(self._window_edge_ids) >= 4:
            self._reg_status.setText("Window edge list already has 4 lines; clear it first.")
            return
        self._window_edge_ids.append(feature_id)
        self._refresh_window_edges_edit()
        self._save_selected_production_profile(silent=True)
        self._reg_status.setText(
            f"Window edge added ({len(self._window_edge_ids)}/4): "
            f"{self._window_edge_label(feature_id, feature)}"
        )

    def _clear_window_edges(self) -> None:
        self._window_edge_ids = []
        self._refresh_window_edges_edit()
        self._save_selected_production_profile(silent=True)
        self._reg_status.setText("Window CAD edges cleared.")

    def _on_window_detection_mode_changed(self, index: int) -> None:
        if index < 0 or not hasattr(self, "_window_mode_combo"):
            return
        mode = str(self._window_mode_combo.currentData() or "auto")
        if mode not in {"auto", "dark", "bright", "grid"}:
            mode = "auto"
        self._window_detection_mode = mode
        self._save_selected_production_profile(silent=True)
        self._reg_status.setText(f"Window detection mode: {self._window_mode_combo.currentText()}")

    def _get_selected_group(self):
        return None

    def set_repository(self, repo: FeatureRepository) -> None:
        self._repo = repo
        self._manager._repo = repo
        self._manager.clear()
        self._window_edge_ids = [
            feature_id for feature_id in self._window_edge_ids
            if self._repo.get(feature_id) is not None
        ]
        self._refresh_window_edges_edit()

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
        if self._config is not None:
            self._pixel_size_mm = float(self._config.pixel_size_mm)
        camera = self._camera if (HAS_CAMERA and self._camera_open) else None
        dialog = ImageLoadDialog(
            self,
            default_pixel_size=self._pixel_size_mm,
            camera=camera,
            config=self._config,
        )
        retranslate_widget_tree(dialog)
        if dialog.exec() == ImageLoadDialog.Accepted:
            path, pixel_size = dialog.get_values()
            captured = dialog.get_captured_frame()
            if hasattr(self, '_canvas'):
                if captured is not None:
                    self._canvas.get_image_layer().load_from_array(captured)
                    self._image_calibration_applied = dialog.calibration_applied()
                    self._auto_source_image_path = self._canvas.get_image_layer().path
                    self._image_path_label.setText(
                        "<camera capture undistorted>"
                        if self._image_calibration_applied else "<camera capture>"
                    )
                elif path:
                    self._canvas.get_image_layer().load_image(path)
                    self._image_calibration_applied = False
                    self._auto_source_image_path = path
                    self._image_path_label.setText(path.split('/')[-1])
                else:
                    return
                pixel_size = float(pixel_size)
                self._canvas.get_image_layer().set_pixel_size_mm(pixel_size)
                self._pixel_size_mm = pixel_size
                self._coarse_transform = None
                self._last_measurement_pixel_to_world = None
                self._last_display_pixel_to_world = None
                self._last_auto_registration = {}
                self._btn_run_coarse.setEnabled(True)
                self._btn_run_fine.setEnabled(False)
                self._btn_run_full.setEnabled(True)
                self._reg_status.setText(tr("Image loaded. Ready for registration."))
                self._canvas.update()
                bus.image_loaded.emit(path or "<camera_capture>")
            if self._config:
                self._config.pixel_size_mm = float(pixel_size)

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
            retranslate_widget_tree(dialog)
            if dialog.exec() != ROISelectorDialog.Accepted:
                return
            rois = dialog.get_rois()
            if rois[0] is not None:
                self._auto_roi1_edit.setText(",".join(str(v) for v in rois[0]))
            if rois[1] is not None:
                self._auto_roi2_edit.setText(",".join(str(v) for v in rois[1]))
            self._reg_status.setText(tr("Fiducial ROIs updated from image picker"))
        except Exception as e:
            self._reg_status.setText(f"ROI picker error: {e}")

    def _auto_correspondence_path(self) -> str:
        group_id = "default"
        image_path = self._auto_source_image_path
        if not image_path and hasattr(self, '_canvas'):
            image_path = self._canvas.get_image_layer().path
        from ..registration.auto_correspondence import auto_config_path
        return auto_config_path(image_path, group_id)

    def _set_auto_cad_fiducial(self, index: int) -> None:
        fid = getattr(self, '_last_highlighted_id', "")
        feat = self._repo.get(fid) if fid else None
        if feat is None:
            self._reg_status.setText(tr("Select a CAD circle first"))
            return
        if feat.feature_type != FeatureType.CIRCLE:
            self._reg_status.setText(tr("Selected CAD feature is not a circle"))
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
                self._reg_status.setText(tr("Lens calibration applied to registration image"))
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

    def _compute_calibrated_pixel_to_world(
        self, cad_points: list[dict], img_points: list[dict],
        enforce_measurement_safety: bool,
    ):
        """Build calibrated pixel -> CAD transform from lens coordinate model.

        Display registration may use the projective transform to compensate
        small camera tilt. Measurement only accepts it when local scale remains
        close to the calibrated pixel size.
        """
        cfg = self._config
        lc = getattr(cfg, "lens_calibration", None) if cfg is not None else None
        if lc is None or not getattr(lc, "coordinate_correction", None):
            return None
        model_type = getattr(lc, "correction_model_type", "none")
        if model_type not in {"homography", "affine"}:
            return None
        try:
            from ..calibration.coordinate_correction import CoordinateTransformer
            from ..registration import affine_solver
            transformer = CoordinateTransformer()
            if not transformer.load_model(lc.coordinate_correction, model_type):
                return None
            image_px = np.array([p["pixel"] for p in img_points], dtype=np.float64)
            from ..calibration.residual_map import residual_map_from_config
            residual_map = residual_map_from_config(cfg)
            if residual_map is not None:
                image_px = residual_map.correct(image_px)
            image_metric = transformer.transform(image_px)
            cad_metric = np.array([p["world"] for p in cad_points], dtype=np.float64)
            plane_to_world = affine_solver.solve_similarity(image_metric, cad_metric)
            if model_type == "homography":
                model_matrix = np.asarray(lc.coordinate_correction.get("homography"), dtype=np.float64)
            else:
                affine = np.asarray(lc.coordinate_correction.get("affine"), dtype=np.float64)
                model_matrix = np.vstack([affine, np.array([0.0, 0.0, 1.0])])
            if model_matrix.shape != (3, 3):
                return None
            pixel_to_world = plane_to_world @ model_matrix
            if not np.all(np.isfinite(pixel_to_world)):
                return None
            if not enforce_measurement_safety:
                return pixel_to_world
            metadata = (lc.coordinate_correction or {}).get("metadata", {})
            image_size = None
            if isinstance(metadata, dict):
                size = metadata.get("image_size")
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    image_size = (int(size[0]), int(size[1]))
            from ..calibration.transform_safety import validate_pixel_to_world_transform
            safety = validate_pixel_to_world_transform(
                pixel_to_world, float(self._pixel_size_mm),
                image_size=image_size,
            )
            if not safety.safe:
                return None
            return pixel_to_world
        except Exception:
            return None

    def _compute_measurement_pixel_to_world(self, cad_points: list[dict], img_points: list[dict]):
        """Build optional measurement-safe pixel -> CAD world transform."""
        return self._compute_calibrated_pixel_to_world(
            cad_points, img_points, enforce_measurement_safety=True,
        )

    def _compute_display_pixel_to_world(self, cad_points: list[dict], img_points: list[dict]):
        """Build a sane calibrated pixel -> CAD world transform for visual overlay."""
        transform = self._compute_calibrated_pixel_to_world(
            cad_points, img_points, enforce_measurement_safety=False,
        )
        if transform is None or not self._display_transform_is_sane(transform):
            return None
        return transform

    def measurement_pixel_to_world_transform(self, image_path: str = ""):
        """Return latest projective pixel -> CAD world measurement transform."""
        if self._last_measurement_pixel_to_world is None:
            return None
        if image_path and self._last_auto_registration.get("image_path") != image_path:
            return None
        transform = np.asarray(self._last_measurement_pixel_to_world, dtype=np.float64)
        image_size = None
        if hasattr(self, "_canvas"):
            layer = self._canvas.get_image_layer()
            image = getattr(layer, "image", None)
            if image is not None:
                image_size = (int(image.shape[1]), int(image.shape[0]))
        try:
            from ..calibration.transform_safety import validate_pixel_to_world_transform
            safety = validate_pixel_to_world_transform(
                transform, float(self._pixel_size_mm), image_size=image_size,
            )
            if not safety.safe:
                return None
        except Exception:
            return None
        return transform

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
        image_path = self._canvas.get_image_layer().path if hasattr(self, '_canvas') else ""
        data = {
            "version": 1,
            "group_id": "default",
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

    def _run_auto_correspondence(self) -> bool:
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

            expected_r1 = float(f1.geometry.get("radius", 0.0)) / max(self._pixel_size_mm, 1e-9)
            expected_r2 = float(f2.geometry.get("radius", 0.0)) / max(self._pixel_size_mm, 1e-9)
            d1 = detect_circle_in_roi(image, roi1, expected_radius_px=expected_r1)
            d2 = detect_circle_in_roi(image, roi2, expected_radius_px=expected_r2)
            if d1 is None:
                raise ValueError("No circle detected in ROI P1")
            if d2 is None:
                raise ValueError("No circle detected in ROI P2")

            cad_points = self._auto_cad_points(f1, f2)
            img_points = self._auto_image_points(d1, d2)
            T = TeachICPStrategy._compute_transform_from_points(
                cad_points, img_points, self._pixel_size_mm,
            )
            measurement_pixel_to_world = self._compute_measurement_pixel_to_world(
                cad_points, img_points,
            )
            display_pixel_to_world = self._compute_display_pixel_to_world(
                cad_points, img_points,
            )
            params = affine_solver.extract_params(T)

            group_id = "default"
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
                "measurement_pixel_to_world": (
                    measurement_pixel_to_world.tolist()
                    if measurement_pixel_to_world is not None else None
                ),
                "display_pixel_to_world": (
                    display_pixel_to_world.tolist()
                    if display_pixel_to_world is not None else None
                ),
                "display_transform_model": (
                    getattr(getattr(self._config, "lens_calibration", None), "correction_model_type", "none")
                    if display_pixel_to_world is not None else "affine"
                ),
                "measurement_transform_model": (
                    getattr(getattr(self._config, "lens_calibration", None), "correction_model_type", "none")
                    if measurement_pixel_to_world is not None else "affine"
                ),
                "created": datetime.now().isoformat(),
                "image_path": image_path,
                "source_image_path": self._auto_source_image_path,
            }
            info = {"image_path": image_path, "group_id": group_id}
            pose_path = TeachICPStrategy._pose_template_path(info)
            TeachICPStrategy._save_pose_template(pose_path, pose)
            self._last_measurement_pixel_to_world = measurement_pixel_to_world
            self._last_display_pixel_to_world = display_pixel_to_world
            self._last_auto_registration = {
                "pose_template_path": pose_path,
                "detections": [d1.to_dict(), d2.to_dict()],
                "transform": {
                    "translation": [float(params["tx"]), float(params["ty"])],
                    "rotation_deg": float(params["rotation_deg"]),
                    "scale": float(params["scale_x"]),
                },
                "measurement_pixel_to_world": (
                    measurement_pixel_to_world.tolist()
                    if measurement_pixel_to_world is not None else None
                ),
                "display_pixel_to_world": (
                    display_pixel_to_world.tolist()
                    if display_pixel_to_world is not None else None
                ),
                "display_transform_model": (
                    getattr(getattr(self._config, "lens_calibration", None), "correction_model_type", "none")
                    if display_pixel_to_world is not None else "affine"
                ),
                "measurement_transform_model": (
                    getattr(getattr(self._config, "lens_calibration", None), "correction_model_type", "none")
                    if measurement_pixel_to_world is not None else "affine"
                ),
                "image_path": image_path,
                "source_image_path": self._auto_source_image_path,
            }
            self._save_selected_production_profile(silent=True)
            profile_name = self._current_profile_name()

            T_img = (
                display_pixel_to_world
                if display_pixel_to_world is not None else self._compute_image_affine(T)
            )
            self._canvas.get_image_layer().set_affine_transform(T_img)
            self._coarse_transform = T
            self._populate_auto_debug_data(T, image)
            self._push_debug_data()
            self._canvas.update()
            self._btn_run_fine.setEnabled(True)
            bus.registration_completed.emit({"transform": T, "stage": "auto_correspondence", "error": 0.0})
            self._reg_status.setText(
                f"Auto registered: P1 conf={d1.confidence:.2f}, P2 conf={d2.confidence:.2f}; "
                f"rot={params['rotation_deg']:.2f} deg, profile={profile_name}"
            )
            return True
        except Exception as e:
            self._last_measurement_pixel_to_world = None
            self._last_display_pixel_to_world = None
            self._reg_status.setText(f"Auto registration error: {e}")
            bus.registration_failed.emit(str(e))
            return False

    def _run_window_line_registration(self) -> bool:
        try:
            from datetime import datetime
            from ..registration.window_line_registration import register_window_lines
            from ..registration import affine_solver

            image = self._ensure_auto_detection_image()
            edge_tokens = list(self._window_edge_ids)
            if edge_tokens and len(edge_tokens) != 4:
                self._reg_status.setText(
                    f"Window registration needs 4 CAD edges; currently {len(edge_tokens)}."
                )
                return False
            result = register_window_lines(
                self._repo,
                image,
                edge_tokens=edge_tokens if len(edge_tokens) == 4 else None,
                pixel_size_mm=self._pixel_size_mm,
                detection_mode=self._window_detection_mode,
            )
            transform = result.transform
            affine = result.affine
            params = affine_solver.extract_params(affine)
            image_path = self._canvas.get_image_layer().path

            self._last_measurement_pixel_to_world = transform
            self._last_display_pixel_to_world = transform
            self._last_auto_registration = {
                "source": result.method,
                "line_handles": dict(result.line_handles),
                "cad_edge_ids": edge_tokens,
                "side_positions": dict(result.side_positions),
                "component_bbox": list(result.component_bbox),
                "confidence": float(result.confidence),
                "transform_model": result.transform_model,
                "homography_safety": result.homography_safety,
                "side_lines": {
                    key: [float(v) for v in value]
                    for key, value in result.side_lines.items()
                },
                "image_corners": (
                    result.image_corners.tolist()
                    if result.image_corners is not None else None
                ),
                "cad_corners": (
                    result.cad_corners.tolist()
                    if result.cad_corners is not None else None
                ),
                "homography": (
                    result.homography.tolist()
                    if result.homography is not None else None
                ),
                "transform": {
                    "translation": [float(params["tx"]), float(params["ty"])],
                    "rotation_deg": float(params["rotation_deg"]),
                    "scale": float(params["scale_x"]),
                    "scale_y": float(params["scale_y"]),
                },
                "measurement_pixel_to_world": transform.tolist(),
                "display_pixel_to_world": transform.tolist(),
                "display_transform_model": result.transform_model,
                "measurement_transform_model": result.transform_model,
                "calibration_applied": bool(self._image_calibration_applied),
                "created": datetime.now().isoformat(),
                "image_path": image_path,
                "source_image_path": self._auto_source_image_path,
            }
            self._save_selected_production_profile(silent=True)
            profile_name = self._current_profile_name()

            self._canvas.get_image_layer().set_affine_transform(transform)
            self._coarse_transform = transform
            self._canvas.update()
            self._btn_run_fine.setEnabled(True)
            bus.registration_completed.emit({
                "transform": transform,
                "stage": result.method,
                "error": 0.0,
            })
            sides = result.side_positions
            self._reg_status.setText(
                "Window registered: "
                f"L={sides['left']:.1f}, R={sides['right']:.1f}, "
                f"T={sides['top']:.1f}, B={sides['bottom']:.1f}; "
                f"conf={result.confidence:.2f}, model={result.transform_model}, "
                f"profile={profile_name}"
            )
            return True
        except Exception as e:
            self._last_measurement_pixel_to_world = None
            self._last_display_pixel_to_world = None
            self._reg_status.setText(f"Window registration error: {e}")
            bus.registration_failed.emit(str(e))
            return False

    def run_auto_registration_for_production(self) -> bool:
        return self._run_auto_correspondence()

    def run_registration_for_production(self) -> bool:
        """Run the production registration method for the current captured frame."""
        # The current production workflow is window-registration based.  Keep the
        # old auto-correspondence method available for explicit UI use, but route
        # Run Production through the same fast window path as manual Evaluate.
        return self._run_window_line_registration()

    def production_profile_snapshot(self) -> dict:
        return self._snapshot_production_profile(self._current_profile_name())

    def last_auto_registration_snapshot(self) -> dict:
        return dict(self._last_auto_registration)

    def production_pixel_size_mm(self) -> float:
        return float(self._pixel_size_mm)

    def image_calibration_applied(self) -> bool:
        return bool(self._image_calibration_applied)

    def _get_anchor_handles(self) -> list[str]:
        text = self._anchor_edit.text().strip()
        if not text:
            return []
        return [h.strip() for h in text.split(",") if h.strip()]

    def _run_coarse(self) -> None:
        group_id = "default"
        if not hasattr(self, '_pipeline'):
            self._reg_status.setText(tr("Error: pipeline not initialized"))
            return
        try:
            result = self._pipeline.run_coarse(
                self._canvas.get_image_layer().path,
                group_id,
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
            self._reg_status.setText(tr("No anchor candidates found"))

    def _run_fine(self) -> None:
        group_id = "default"
        if not hasattr(self, '_pipeline') or not hasattr(self, '_coarse_transform'):
            return
        try:
            result = self._pipeline.run_fine(
                self._coarse_transform, group_id,
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
        group_id = "default"
        if not hasattr(self, '_pipeline'):
            return
        try:
            result = self._pipeline.run_full(
                self._canvas.get_image_layer().path,
                group_id,
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
        self._reg_status.setText(tr("Teach points cleared"))

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
            self._reg_status.setText(tr("Error: need 2 CAD + 2 image points"))
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

            group_id = "default"

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
