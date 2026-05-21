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
    QInputDialog, QAbstractItemView, QSplitter,
)

from ..models.feature import FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationGroup, RegistrationManager
from ..core.signals import bus


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

        self._setup_ui()
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
            parts = [f"{ft.name}: {c}" for ft, c in sorted(stats.items())]
            self._types_label.setText(", ".join(parts))
        else:
            self._types_label.setText("—")

    def set_repository(self, repo: FeatureRepository) -> None:
        self._repo = repo
        self._manager.set_repository(repo)
        self._refresh_group_list()
        self._refresh_feature_list()
