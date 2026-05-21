"""
FeatureTreePanel — left-side tree widget grouping features by type and layer.

Tree structure:
  Geometry Features
    Lines (N)
      [individual line entries]
    Circles (N)
      ...
    Arcs (N)
    Polylines (N)
    Splines (N)
  Dimensions
  Annotations (Text)
  Layers
    Layer Name 1
    Layer Name 2
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QIcon, QColor, QBrush
from PySide6.QtWidgets import (
    QTreeWidget, QTreeWidgetItem, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QLabel, QToolBar, QAbstractItemView, QMenu,
)

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..models.registration import RegistrationManager
from ..core.signals import bus


# Human-readable type labels
TYPE_LABELS = {
    FeatureType.LINE: "Lines",
    FeatureType.CIRCLE: "Circles",
    FeatureType.ARC: "Arcs",
    FeatureType.POLYLINE: "Polylines",
    FeatureType.SPLINE: "Splines",
    FeatureType.DIMENSION: "Dimensions",
    FeatureType.TEXT: "Annotations",
    FeatureType.HATCH: "Hatches",
    FeatureType.POINT: "Points",
}

# Colors for tree group icons
TYPE_COLORS = {
    FeatureType.LINE: "#FFFFFF",
    FeatureType.CIRCLE: "#00FFFF",
    FeatureType.ARC: "#FFFF00",
    FeatureType.POLYLINE: "#00FF00",
    FeatureType.SPLINE: "#FF00FF",
    FeatureType.DIMENSION: "#FF0000",
    FeatureType.TEXT: "#AAAAAA",
    FeatureType.HATCH: "#666666",
    FeatureType.POINT: "#FF8800",
}


class FeatureTreePanel(QWidget):
    """Left-side feature tree panel with type/layer grouping."""

    feature_selected = Signal(str)   # feature_id
    feature_deselected = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(400)

        self._feature_map: Dict[str, QTreeWidgetItem] = {}  # feature_id → tree item
        self._type_nodes: Dict[FeatureType, QTreeWidgetItem] = {}
        self._reg_manager: Optional[RegistrationManager] = None
        self._groups_root: Optional[QTreeWidgetItem] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Feature Browser")
        header.setStyleSheet("font-weight: bold; padding: 6px; background: #2d2d2d; color: #ddd;")
        layout.addWidget(header)

        # Search filter
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter features...")
        self._search.textChanged.connect(self._filter_tree)
        layout.addWidget(self._search)

        # Tree widget
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.setAnimated(True)
        self._tree.setExpandsOnDoubleClick(True)
        self._tree.itemClicked.connect(self._on_item_clicked)
        self._tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: none;
                font-size: 12px;
            }
            QTreeWidget::item:selected {
                background-color: #264f78;
                color: #ffffff;
            }
            QTreeWidget::item:hover {
                background-color: #2a2d2e;
            }
        """)
        self._tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self._tree)

    def populate(self, repo: FeatureRepository) -> None:
        """Build tree from repository."""
        self._tree.clear()
        self._feature_map.clear()
        self._type_nodes.clear()

        counts = repo.type_counts()

        # ── Geometry Features root ──
        geom_root = QTreeWidgetItem(self._tree, ["Geometry Features"])
        geom_root.setExpanded(True)
        font = geom_root.font(0)
        font.setBold(True)
        geom_root.setFont(0, font)

        geom_types = [FeatureType.LINE, FeatureType.CIRCLE, FeatureType.ARC,
                      FeatureType.POLYLINE, FeatureType.SPLINE]

        for ftype in geom_types:
            count = counts.get(ftype, 0)
            if count == 0:
                continue
            label = f"{TYPE_LABELS[ftype]} ({count})"
            type_node = QTreeWidgetItem(geom_root, [label])
            type_node.setExpanded(False)
            color = QColor(TYPE_COLORS.get(ftype, "#FFFFFF"))
            type_node.setForeground(0, QBrush(color))
            font = type_node.font(0)
            font.setBold(True)
            type_node.setFont(0, font)
            self._type_nodes[ftype] = type_node

            # Add individual features
            for feat in repo.features_by_type(ftype):
                item = QTreeWidgetItem(type_node, [feat.display_name])
                item.setData(0, Qt.UserRole, feat.feature_id)
                self._feature_map[feat.feature_id] = item

        # ── Dimensions root ──
        dim_count = counts.get(FeatureType.DIMENSION, 0)
        if dim_count > 0:
            dim_root = QTreeWidgetItem(self._tree, [f"Dimensions ({dim_count})"])
            dim_root.setExpanded(False)
            for feat in repo.features_by_type(FeatureType.DIMENSION):
                item = QTreeWidgetItem(dim_root, [feat.display_name])
                item.setData(0, Qt.UserRole, feat.feature_id)
                self._feature_map[feat.feature_id] = item

        # ── Annotations root ──
        text_count = counts.get(FeatureType.TEXT, 0)
        if text_count > 0:
            anno_root = QTreeWidgetItem(self._tree, [f"Annotations ({text_count})"])
            anno_root.setExpanded(False)
            for feat in repo.features_by_type(FeatureType.TEXT):
                item = QTreeWidgetItem(anno_root, [feat.display_name])
                item.setData(0, Qt.UserRole, feat.feature_id)
                self._feature_map[feat.feature_id] = item

        # ── Layers root ──
        layers_root = QTreeWidgetItem(self._tree, ["Layers"])
        layers_root.setExpanded(False)
        font = layers_root.font(0)
        font.setBold(True)
        layers_root.setFont(0, font)

        for layer_name in sorted(repo.all_layers()):
            layer_feats = repo.features_by_layer(layer_name)
            layer_node = QTreeWidgetItem(layers_root, [f"{layer_name} ({len(layer_feats)})"])
            layer_node.setData(0, Qt.UserRole, f"layer:{layer_name}")

    def set_registration_manager(self, manager: RegistrationManager) -> None:
        self._reg_manager = manager
        bus.group_created.connect(self._refresh_groups_tree)
        bus.group_deleted.connect(self._refresh_groups_tree)
        bus.group_contents_changed.connect(self._refresh_groups_tree)
        bus.groups_cleared.connect(self._refresh_groups_tree)

    def _refresh_groups_tree(self, *args) -> None:
        if not self._reg_manager:
            return
        if self._groups_root:
            self._tree.takeTopLevelItem(
                self._tree.indexOfTopLevelItem(self._groups_root)
            )
            self._groups_root = None
        groups = self._reg_manager.all_groups()
        if not groups:
            return
        self._groups_root = QTreeWidgetItem(self._tree, ["Registration Groups"])
        self._groups_root.setExpanded(False)
        font = self._groups_root.font(0)
        font.setBold(True)
        self._groups_root.setFont(0, font)
        for group in groups:
            grp_node = QTreeWidgetItem(
                self._groups_root, [f"{group.name} ({group.feature_count})"]
            )
            grp_node.setData(0, Qt.UserRole, f"group:{group.group_id}")
            grp_node.setForeground(0, QBrush(group.color))
            for fid in group.feature_ids:
                feat = self._reg_manager._repo.get(fid)
                if feat:
                    item = QTreeWidgetItem(grp_node, [feat.display_name])
                    item.setData(0, Qt.UserRole, fid)

    def _on_context_menu(self, pos) -> None:
        item = self._tree.itemAt(pos)
        if not item:
            return

        feature_id = item.data(0, Qt.UserRole)
        if not feature_id:
            return

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #2d2d2d; color: #cccccc; border: 1px solid #3d3d3d; }
            QMenu::item:selected { background-color: #264f78; }
        """)

        if feature_id.startswith("group:"):
            # Group node context menu
            group_id = feature_id[len("group:"):]
            zoom_action = menu.addAction("Zoom to Group")
            delete_action = menu.addAction("Delete Group")
            action = menu.exec_(self._tree.mapToGlobal(pos))
            if action == zoom_action:
                bus.view_fit_all.emit()
            elif action == delete_action and self._reg_manager:
                self._reg_manager.delete_group(group_id)
                bus.group_deleted.emit(group_id)
        elif not feature_id.startswith("layer:") and self._reg_manager:
            # Feature node — offer add to group
            add_menu = menu.addMenu("Add to Group")
            new_group_action = add_menu.addAction("New Group...")
            add_menu.addSeparator()
            for group in self._reg_manager.all_groups():
                action = add_menu.addAction(group.name)
                action.setData(group.group_id)

            action = menu.exec_(self._tree.mapToGlobal(pos))
            if action == new_group_action:
                group = self._reg_manager.create_group()
                self._reg_manager.add_feature_to_group(group.group_id, feature_id)
                bus.group_created.emit(group.group_id)
                bus.group_contents_changed.emit(group.group_id)
            elif action and action.data():
                gid = action.data()
                self._reg_manager.add_feature_to_group(gid, feature_id)
                bus.group_contents_changed.emit(gid)

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle tree item click — emit feature selection signal."""
        feature_id = item.data(0, Qt.UserRole)
        if feature_id and not feature_id.startswith("layer:") and not feature_id.startswith("group:"):
            self.feature_selected.emit(feature_id)
            bus.highlight_feature.emit(feature_id)
            bus.view_fit_feature.emit(feature_id)
            bus.property_update.emit({"feature_id": feature_id})
        else:
            self.feature_deselected.emit()
            bus.feature_deselected.emit()

    def select_feature(self, feature_id: str) -> None:
        """Programmatically select and scroll to a feature in the tree."""
        item = self._feature_map.get(feature_id)
        if item:
            self._tree.setCurrentItem(item)
            self._tree.scrollToItem(item)
            # Expand parent nodes
            parent = item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()

    def _filter_tree(self, text: str) -> None:
        """Filter tree items by search text."""
        text_lower = text.lower()
        for fid, item in self._feature_map.items():
            match = not text or text_lower in item.text(0).lower()
            item.setHidden(not match)
