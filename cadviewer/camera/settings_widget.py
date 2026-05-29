"""
CameraSettingsWidget — embeddable camera parameter adjustment panel.

Provides sliders/spinboxes for exposure, gamma, contrast, gain, plus
checkboxes for auto-exposure and mirror modes. Emits settings_changed
signal when any parameter changes.

The exposure slider uses a logarithmic mapping so that equal perceptual
resolution is available across the full range (e.g. 100–1 000 000 us).
The internal slider range is 0–10000; the mapping is:
    slider → exposure:  min * exp(s / SLIDER_MAX * ln(max / min))
    exposure → slider:  SLIDER_MAX * ln(e / min) / ln(max / min)
"""

from __future__ import annotations

import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QHBoxLayout, QSlider, QSpinBox, QCheckBox,
)

from .device import CameraSettings, CameraSettingRanges

_LOG_SLIDER_MAX = 10000


class CameraSettingsWidget(QWidget):
    """Embeddable camera settings panel with slider+spinbox pairs."""

    settings_changed = Signal(CameraSettings)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._block_signals = False
        self._exp_min = 100
        self._exp_max = 1_000_000
        self._setup_ui()
        self.setStyleSheet("""
            QSlider {
                background: transparent;
            }
            QSpinBox {
                background-color: #333;
                color: #ccc;
                border: 1px solid #555;
                padding: 2px;
            }
            QCheckBox {
                color: #ccc;
            }
            QLabel {
                color: #888;
            }
        """)

    # ── log-scale helpers for exposure slider ─────────────────────────

    def _exp_from_slider(self, s: int) -> int:
        """Convert internal slider position to exposure time (us)."""
        if self._exp_min <= 0 or self._exp_max <= self._exp_min:
            return s
        ratio = self._exp_max / self._exp_min
        us = self._exp_min * math.exp(s / _LOG_SLIDER_MAX * math.log(ratio))
        return max(self._exp_min, min(self._exp_max, int(round(us))))

    def _slider_from_exp(self, us: int) -> int:
        """Convert exposure time (us) to internal slider position."""
        if us <= 0 or self._exp_min <= 0 or self._exp_max <= self._exp_min:
            return 0
        us = max(self._exp_min, min(self._exp_max, us))
        ratio = self._exp_max / self._exp_min
        s = _LOG_SLIDER_MAX * math.log(us / self._exp_min) / math.log(ratio)
        return max(0, min(_LOG_SLIDER_MAX, int(round(s))))

    # ── UI setup ──────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setSpacing(8)

        # Auto Exposure checkbox
        self._ae_check = QCheckBox("Auto Exposure")
        self._ae_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._ae_check)

        # Exposure: log-scale slider + spinbox
        self._exposure_slider = QSlider(Qt.Horizontal)
        self._exposure_slider.setRange(0, _LOG_SLIDER_MAX)
        self._exposure_slider.setSingleStep(10)
        self._exposure_slider.setPageStep(200)

        self._exposure_spin = QSpinBox()
        self._exposure_spin.setSuffix(" us")
        self._exposure_spin.setMinimum(100)
        self._exposure_spin.setMaximum(1000000)
        self._exposure_spin.setSingleStep(100)

        self._exposure_slider.valueChanged.connect(self._on_exp_slider_changed)
        self._exposure_spin.valueChanged.connect(self._on_exp_spin_changed)

        exp_row = QHBoxLayout()
        exp_row.addWidget(self._exposure_slider, 1)
        exp_row.addWidget(self._exposure_spin)
        layout.addRow("Exposure:", exp_row)

        # Gamma: slider + spinbox
        self._gamma_slider = QSlider(Qt.Horizontal)
        self._gamma_spin = QSpinBox()
        self._gamma_slider.valueChanged.connect(self._gamma_spin.setValue)
        self._gamma_spin.valueChanged.connect(self._gamma_slider.setValue)
        self._gamma_spin.valueChanged.connect(self._on_setting_changed)
        gamma_row = QHBoxLayout()
        gamma_row.addWidget(self._gamma_slider, 1)
        gamma_row.addWidget(self._gamma_spin)
        layout.addRow("Gamma:", gamma_row)

        # Contrast: slider + spinbox
        self._contrast_slider = QSlider(Qt.Horizontal)
        self._contrast_spin = QSpinBox()
        self._contrast_slider.valueChanged.connect(self._contrast_spin.setValue)
        self._contrast_spin.valueChanged.connect(self._contrast_slider.setValue)
        self._contrast_spin.valueChanged.connect(self._on_setting_changed)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self._contrast_slider, 1)
        contrast_row.addWidget(self._contrast_spin)
        layout.addRow("Contrast:", contrast_row)

        # Analog Gain: slider + spinbox
        self._gain_slider = QSlider(Qt.Horizontal)
        self._gain_spin = QSpinBox()
        self._gain_slider.valueChanged.connect(self._gain_spin.setValue)
        self._gain_spin.valueChanged.connect(self._gain_slider.setValue)
        self._gain_spin.valueChanged.connect(self._on_setting_changed)
        gain_row = QHBoxLayout()
        gain_row.addWidget(self._gain_slider, 1)
        gain_row.addWidget(self._gain_spin)
        layout.addRow("Analog Gain:", gain_row)

        # Mirror checkboxes
        self._reverse_x_check = QCheckBox("Mirror Horizontal")
        self._reverse_x_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._reverse_x_check)

        self._reverse_y_check = QCheckBox("Mirror Vertical")
        self._reverse_y_check.stateChanged.connect(self._on_setting_changed)
        layout.addRow(self._reverse_y_check)

    # ── exposure slider ↔ spinbox sync (log-scale) ────────────────────

    def _on_exp_slider_changed(self, slider_val: int) -> None:
        if self._block_signals:
            return
        us = self._exp_from_slider(slider_val)
        self._block_signals = True
        self._exposure_spin.setValue(us)
        self._block_signals = False
        self._on_setting_changed()

    def _on_exp_spin_changed(self, us: int) -> None:
        if self._block_signals:
            return
        s = self._slider_from_exp(us)
        self._block_signals = True
        self._exposure_slider.setValue(s)
        self._block_signals = False
        self._on_setting_changed()

    # ── public API ────────────────────────────────────────────────────

    def set_ranges(self, ranges: CameraSettingRanges) -> None:
        """Set min/max/step for all sliders and spinboxes."""
        self._block_signals = True
        self._exp_min = ranges.exposure_min_us
        self._exp_max = ranges.exposure_max_us
        self._exposure_slider.setRange(0, _LOG_SLIDER_MAX)
        self._exposure_slider.setSingleStep(10)
        self._exposure_slider.setPageStep(200)
        self._exposure_spin.setRange(ranges.exposure_min_us, ranges.exposure_max_us)
        self._exposure_spin.setSingleStep(ranges.exposure_step_us)
        self._gamma_slider.setRange(ranges.gamma_min, ranges.gamma_max)
        self._gamma_spin.setRange(ranges.gamma_min, ranges.gamma_max)
        self._contrast_slider.setRange(ranges.contrast_min, ranges.contrast_max)
        self._contrast_spin.setRange(ranges.contrast_min, ranges.contrast_max)
        self._gain_slider.setRange(ranges.analog_gain_min, ranges.analog_gain_max)
        self._gain_spin.setRange(ranges.analog_gain_min, ranges.analog_gain_max)
        self._block_signals = False

    def set_values(self, settings: CameraSettings) -> None:
        """Set all controls to match current camera settings."""
        self._block_signals = True
        self._ae_check.setChecked(settings.ae_enabled)
        self._exposure_slider.setValue(self._slider_from_exp(settings.exposure_us))
        self._exposure_spin.setValue(settings.exposure_us)
        self._exposure_slider.setEnabled(not settings.ae_enabled)
        self._exposure_spin.setEnabled(not settings.ae_enabled)
        self._gamma_slider.setValue(settings.gamma)
        self._gamma_spin.setValue(settings.gamma)
        self._contrast_slider.setValue(settings.contrast)
        self._contrast_spin.setValue(settings.contrast)
        self._gain_slider.setValue(settings.analog_gain)
        self._gain_spin.setValue(settings.analog_gain)
        self._reverse_x_check.setChecked(settings.reverse_x)
        self._reverse_y_check.setChecked(settings.reverse_y)
        self._block_signals = False

    def _on_setting_changed(self) -> None:
        """Collect current values and emit settings_changed signal."""
        if self._block_signals:
            return
        settings = CameraSettings(
            exposure_us=self._exposure_spin.value(),
            gamma=self._gamma_spin.value(),
            contrast=self._contrast_spin.value(),
            analog_gain=self._gain_spin.value(),
            ae_enabled=self._ae_check.isChecked(),
            reverse_x=self._reverse_x_check.isChecked(),
            reverse_y=self._reverse_y_check.isChecked(),
        )
        self._exposure_slider.setEnabled(not settings.ae_enabled)
        self._exposure_spin.setEnabled(not settings.ae_enabled)
        self.settings_changed.emit(settings)
