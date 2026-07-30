"""Dialog for manual light-source controller tuning."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.config import AppConfig, LightChannelConfig
from ..core.i18n import tr
from ..hardware.light_controller import LightController


class _ChannelControls:
    def __init__(self, title: str, role: str, channel: int, config: LightChannelConfig):
        self.title = title
        self.role = role
        self.channel = channel
        self.group = QGroupBox(title)
        self.enabled = QCheckBox("Off")
        self.enabled.setChecked(bool(config.enabled))
        self.enabled.setText("On" if config.enabled else "Off")
        self.enabled.setToolTip("Switch this controller channel output on or off")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.setValue(int(config.brightness))

        self.spin = QSpinBox()
        self.spin.setRange(0, 255)
        self.spin.setValue(int(config.brightness))
        self.spin.setSuffix(" /255")

        layout = QGridLayout(self.group)
        layout.addWidget(QLabel("Output:"), 0, 0)
        layout.addWidget(self.enabled, 0, 1)
        layout.addWidget(QLabel("Brightness:"), 1, 0)
        layout.addWidget(self.slider, 1, 1)
        layout.addWidget(self.spin, 1, 2)

    def brightness(self) -> int:
        return int(self.spin.value())

    def is_enabled(self) -> bool:
        return bool(self.enabled.isChecked())

    def set_brightness(self, value: int) -> None:
        value = max(0, min(255, int(value)))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(value)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled.blockSignals(True)
        self.enabled.setChecked(bool(enabled))
        self.enabled.setText(tr("On") if enabled else tr("Off"))
        self.enabled.blockSignals(False)

    def to_config(self) -> LightChannelConfig:
        return LightChannelConfig(
            brightness=self.brightness(),
            enabled=self.is_enabled(),
        )


class LightControlDialog(QDialog):
    """Manual three-group light-control panel for the RS232 controller."""

    light_profile_changed = Signal()

    CHANNELS = (
        ("Ring Light CH1", "ring_ch1", 1),
        ("Ring Light CH2", "ring_ch2", 2),
        ("Backlight CH4", "backlight_ch4", 4),
    )

    def __init__(
        self,
        config: AppConfig,
        parent: QWidget | None = None,
        controller_owner=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Light Source Control")
        self.setMinimumWidth(620)
        self._config = config
        self._controller_owner = controller_owner
        self._controller: LightController | None = None
        self._controller_key: tuple[str, int, float] | None = None

        self._build_ui()
        self._connect_ui()
        self._load_config()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        connection_group = QGroupBox("Controller Connection")
        form = QFormLayout(connection_group)
        self._device_edit = QLineEdit()
        self._device_edit.setPlaceholderText("/dev/ttyUSB0 or COM3")
        self._baud_combo = QComboBox()
        for baud in (9600, 19200, 38400, 57600, 115200):
            self._baud_combo.addItem(str(baud), baud)
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(100, 5000)
        self._timeout_spin.setSingleStep(100)
        self._timeout_spin.setSuffix(" ms")
        self._backlight_delay_spin = QSpinBox()
        self._backlight_delay_spin.setRange(0, 10000)
        self._backlight_delay_spin.setSingleStep(50)
        self._backlight_delay_spin.setSuffix(" ms")
        self._ring_delay_spin = QSpinBox()
        self._ring_delay_spin.setRange(0, 10000)
        self._ring_delay_spin.setSingleStep(50)
        self._ring_delay_spin.setSuffix(" ms")
        form.addRow("Device:", self._device_edit)
        form.addRow("Baud:", self._baud_combo)
        form.addRow("Timeout:", self._timeout_spin)
        form.addRow("Backlight settle delay:", self._backlight_delay_spin)
        form.addRow("Ring-light settle delay:", self._ring_delay_spin)

        connection_buttons = QHBoxLayout()
        self._btn_test = QPushButton("Test Connection")
        connection_buttons.addWidget(self._btn_test)
        connection_buttons.addStretch(1)
        form.addRow(connection_buttons)
        layout.addWidget(connection_group)

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _connect_ui(self) -> None:
        self._btn_test.clicked.connect(self._test_connection)
        self._device_edit.textChanged.connect(self._connection_settings_changed)
        self._baud_combo.currentIndexChanged.connect(self._connection_settings_changed)
        self._timeout_spin.valueChanged.connect(self._connection_settings_changed)
        self._backlight_delay_spin.valueChanged.connect(lambda _: self._save_config())
        self._ring_delay_spin.valueChanged.connect(lambda _: self._save_config())

    def _load_config(self) -> None:
        light_config = self._config.light_controller
        widgets = [
            self._device_edit,
            self._baud_combo,
            self._timeout_spin,
            self._backlight_delay_spin,
            self._ring_delay_spin,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self._device_edit.setText(light_config.device or "/dev/ttyUSB0")
        idx = self._baud_combo.findData(int(light_config.baud))
        self._baud_combo.setCurrentIndex(max(idx, 0))
        self._timeout_spin.setValue(max(100, int(float(light_config.timeout_s) * 1000)))
        self._backlight_delay_spin.setValue(
            max(0, int(getattr(light_config, "backlight_settle_delay_ms", 200))),
        )
        self._ring_delay_spin.setValue(
            max(0, int(getattr(light_config, "ring_light_settle_delay_ms", 200))),
        )
        for widget in widgets:
            widget.blockSignals(False)

    def accept(self) -> None:
        self._save_config()
        self._close_controller()
        super().accept()

    def reject(self) -> None:
        self._close_controller()
        super().reject()

    def closeEvent(self, event) -> None:
        self._close_controller()
        super().closeEvent(event)

    def _connection_settings_changed(self) -> None:
        self._close_controller()

    def _current_key(self) -> tuple[str, int, float]:
        return (
            self._device_edit.text().strip() or "/dev/ttyUSB0",
            int(self._baud_combo.currentData()),
            self._timeout_spin.value() / 1000.0,
        )

    def _ensure_controller(self) -> LightController:
        if self._controller_owner is not None:
            controller = self._controller_owner._light_controller(self._current_key())
            self._controller = controller
            self._controller_key = self._current_key()
            return controller
        key = self._current_key()
        if self._controller is not None and self._controller_key == key:
            return self._controller
        self._close_controller()
        device, baud, timeout_s = key
        controller = LightController(device=device, baud=baud, timeout_s=timeout_s)
        controller.open()
        self._controller = controller
        self._controller_key = key
        return controller

    def _close_controller(self) -> None:
        if self._controller_owner is not None:
            return
        if self._controller is not None:
            self._controller.close()
            self._controller = None
            self._controller_key = None

    def _save_config(self) -> None:
        light_config = self._config.light_controller
        device, baud, timeout_s = self._current_key()
        light_config.device = device
        light_config.baud = baud
        light_config.timeout_s = timeout_s
        light_config.backlight_settle_delay_ms = int(self._backlight_delay_spin.value())
        light_config.ring_light_settle_delay_ms = int(self._ring_delay_spin.value())
        self._config.save()
        self.light_profile_changed.emit()

    def refresh_from_config(self) -> None:
        light_config = self._config.light_controller
        self._device_edit.setText(light_config.device or "/dev/ttyUSB0")
        idx = self._baud_combo.findData(int(light_config.baud))
        self._baud_combo.setCurrentIndex(max(idx, 0))
        self._timeout_spin.setValue(max(100, int(float(light_config.timeout_s) * 1000)))
        self._backlight_delay_spin.setValue(
            max(0, int(getattr(light_config, "backlight_settle_delay_ms", 200))),
        )
        self._ring_delay_spin.setValue(
            max(0, int(getattr(light_config, "ring_light_settle_delay_ms", 200))),
        )

    def _show_error(self, message: str) -> None:
        self._status_label.setText(message)
        QMessageBox.warning(self, tr("Light Source Control"), message)

    def _test_connection(self) -> None:
        try:
            ctrl = self._ensure_controller()
            first = ctrl.read_brightness(1)
            self._status_label.setText(
                f"Controller OK. CH{first.channel} brightness={first.brightness}"
            )
            self._save_config()
        except Exception as exc:
            self._show_error(f"{tr('Light controller connection failed:')} {exc}")

    def _read_brightness(self) -> None:
        try:
            ctrl = self._ensure_controller()
            messages = []
            for controls in self._channels.values():
                result = ctrl.read_brightness(controls.channel)
                controls.set_brightness(result.brightness)
                messages.append(
                    f"CH{result.channel}={result.brightness}"
                    + ("" if result.valid_checksum else " checksum warning")
                )
            self._status_label.setText("Read brightness: " + ", ".join(messages))
            self._save_config()
        except Exception as exc:
            self._show_error(f"{tr('Read brightness failed:')} {exc}")

    def _apply_all(self) -> None:
        try:
            ctrl = self._ensure_controller()
            for controls in self._channels.values():
                ctrl.set_brightness(controls.channel, controls.brightness())
                if controls.is_enabled():
                    ctrl.open_channel(controls.channel)
                else:
                    ctrl.close_channel(controls.channel)
            self._save_config()
            self._status_label.setText("Light settings applied")
        except Exception as exc:
            self._show_error(f"{tr('Apply light settings failed:')} {exc}")

    def _all_off(self) -> None:
        try:
            ctrl = self._ensure_controller()
            for controls in self._channels.values():
                ctrl.close_channel(controls.channel)
                controls.set_enabled(False)
            self._save_config()
            self._status_label.setText("All configured light channels are off")
        except Exception as exc:
            self._show_error(f"{tr('Turning lights off failed:')} {exc}")

    def _toggle_channel(self, controls: _ChannelControls, checked: bool) -> None:
        controls.enabled.setText(tr("On") if checked else tr("Off"))
        try:
            ctrl = self._ensure_controller()
            if checked:
                ctrl.set_brightness(controls.channel, controls.brightness())
                ctrl.open_channel(controls.channel)
            else:
                ctrl.close_channel(controls.channel)
            self._save_config()
            state = "ON" if checked else "OFF"
            self._status_label.setText(f"{controls.title} {state}")
        except Exception as exc:
            controls.set_enabled(not checked)
            self._show_error(f"Set {controls.title} output failed: {exc}")

    def _brightness_changed(self, controls: _ChannelControls, value: int) -> None:
        controls.set_brightness(value)
        if controls.is_enabled():
            self._debounce[controls.role].start(180)

    def _send_brightness(self, controls: _ChannelControls) -> None:
        try:
            ctrl = self._ensure_controller()
            ctrl.set_brightness(controls.channel, controls.brightness())
            self._status_label.setText(
                f"{controls.title} brightness set to {controls.brightness()}"
            )
            self._save_config()
        except Exception as exc:
            self._show_error(f"Set {controls.title} brightness failed: {exc}")
