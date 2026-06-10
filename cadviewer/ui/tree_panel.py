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
    QLineEdit, QLabel, QToolBar, QAbstractItemView,
)

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
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
    FeatureType.LEADER: "Leaders",
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
    FeatureType.LEADER: "#FFB450",
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

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QLabel("Feature Browser")
        header.setStyleSheet("font: bold; padding: 6px; background-color: #2d2d2d; color: #ddd;")
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

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        """Handle tree item click — emit feature selection signal."""
        feature_id = item.data(0, Qt.UserRole)
        if feature_id and not feature_id.startswith("layer:"):
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
