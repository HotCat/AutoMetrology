"""
PropertyPanel — right-side panel showing selected feature properties.

Displays:
  - Feature ID, type, layer
  - Geometry details (coordinates, radius, angles, etc.)
  - DXF handle
  - Measurement metadata (future)
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QFormLayout, QGroupBox,
    QScrollArea, QFrame, QSizePolicy,
)

from ..models.feature import CADFeature, FeatureType
from ..models.repository import FeatureRepository
from ..core.signals import bus


class PropertyPanel(QWidget):
    """Right-side property inspector panel."""

    def __init__(self, repo: FeatureRepository, parent=None) -> None:
        super().__init__(parent)
        self._repo = repo
        self.setMinimumWidth(220)
        self.setMaximumWidth(350)

        self._setup_ui()
        bus.property_update.connect(self._on_property_update)

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Header
        header = QLabel("Properties")
        header.setStyleSheet("font-weight: bold; padding: 6px; background: #2d2d2d; color: #ddd;")
        layout.addWidget(header)

        # Scroll area for property content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #1e1e1e; color: #cccccc; }")

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)

        # Placeholder
        self._placeholder = QLabel("Select a feature to view properties")
        self._placeholder.setAlignment(Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #666; font-style: italic;")
        self._content_layout.addWidget(self._placeholder)
        self._content_layout.addStretch()

        scroll.setWidget(self._content)
        layout.addWidget(scroll)

    @Slot(dict)
    def _on_property_update(self, props: dict) -> None:
        """Update property panel from signal."""
        feature_id = props.get("feature_id")
        if not feature_id:
            return
        feature = self._repo.get(feature_id)
        if feature:
            self._display_feature(feature)

    def _display_feature(self, feature: CADFeature) -> None:
        """Populate panel with feature data."""
        # Clear existing content
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # ── General group ──
        general_group = QGroupBox("General")
        general_form = QFormLayout()
        general_form.setLabelAlignment(Qt.AlignRight)
        self._add_form_row(general_form, "Type:", feature.feature_type.name)
        self._add_form_row(general_form, "Layer:", feature.layer)
        self._add_form_row(general_form, "DXF Handle:", feature.dxf_handle)
        self._add_form_row(general_form, "ID:", feature.feature_id[:12] + "...")
        self._add_form_row(general_form, "Color:", str(feature.color))
        general_group.setLayout(general_form)
        general_group.setStyleSheet(self._group_style())
        self._content_layout.addWidget(general_group)

        # ── Geometry group ──
        geom_group = QGroupBox("Geometry")
        geom_form = QFormLayout()
        geom_form.setLabelAlignment(Qt.AlignRight)
        summary = feature.geometry_summary()
        for key, value in summary.items():
            if key != "type":
                self._add_form_row(geom_form, f"{key}:", str(value))
        geom_group.setLayout(geom_form)
        geom_group.setStyleSheet(self._group_style())
        self._content_layout.addWidget(geom_group)

        # ── Measurement group (placeholder) ──
        meas_group = QGroupBox("Measurement")
        meas_form = QFormLayout()
        meas_form.setLabelAlignment(Qt.AlignRight)
        m = feature.measurement
        self._add_form_row(meas_form, "Nominal:", str(m.nominal_value) if m.nominal_value else "—")
        self._add_form_row(meas_form, "Measured:", str(m.measured_value) if m.measured_value else "—")
        self._add_form_row(meas_form, "Deviation:", str(m.deviation) if m.deviation else "—")
        self._add_form_row(meas_form, "Status:", "—" if m.is_passing is None else ("PASS" if m.is_passing else "FAIL"))
        meas_group.setLayout(meas_form)
        meas_group.setStyleSheet(self._group_style())
        self._content_layout.addWidget(meas_group)

        self._content_layout.addStretch()

    def _add_form_row(self, form: QFormLayout, label: str, value: str) -> None:
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #888; font-size: 11px;")
        val = QLabel(value)
        val.setStyleSheet("color: #ddd; font-size: 11px;")
        val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        val.setWordWrap(True)
        form.addRow(lbl, val)

    @staticmethod
    def _group_style() -> str:
        return """
            QGroupBox {
                color: #aaa;
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #333;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """

    def clear(self) -> None:
        """Reset panel to placeholder."""
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        placeholder = QLabel("Select a feature to view properties")
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-style: italic;")
        self._content_layout.addWidget(placeholder)
        self._content_layout.addStretch()
