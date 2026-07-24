"""Compact light-source controls for the camera live preview."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.config import AppConfig, LightChannelConfig
from ..core.i18n import tr
from ..hardware.light_controller import LightController


class _CompactChannel:
    def __init__(self, title: str, role: str, channel: int, config: LightChannelConfig):
        self.title = title
        self.role = role
        self.channel = int(channel)
        self.group = QGroupBox(title)

        self.enabled = QCheckBox("On" if config.enabled else "Off")
        self.enabled.setChecked(bool(config.enabled))
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
        self.spin.setFixedWidth(78)

        layout = QGridLayout(self.group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(4)
        layout.addWidget(self.enabled, 0, 0)
        layout.addWidget(self.slider, 1, 0, 1, 2)
        layout.addWidget(self.spin, 0, 1)

    def brightness(self) -> int:
        return int(self.spin.value())

    def is_enabled(self) -> bool:
        return bool(self.enabled.isChecked())

    def set_enabled(self, enabled: bool) -> None:
        self.enabled.blockSignals(True)
        self.enabled.setChecked(bool(enabled))
        self.enabled.setText(tr("On") if enabled else tr("Off"))
        self.enabled.blockSignals(False)

    def set_brightness(self, value: int) -> None:
        value = max(0, min(255, int(value)))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(value)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)

    def to_config(self) -> LightChannelConfig:
        return LightChannelConfig(
            brightness=self.brightness(),
            enabled=self.is_enabled(),
        )


class LightControlPanel(QGroupBox):
    """Live-preview light controls using the saved light-controller settings."""

    CHANNELS = (
        ("Ring Light CH1", "ring_ch1", 1),
        ("Ring Light CH2", "ring_ch2", 2),
        ("Backlight CH4", "backlight_ch4", 4),
    )

    def __init__(self, config: AppConfig, parent: QWidget | None = None):
        super().__init__("Light Source Control", parent)
        self._config = config
        self._controller: LightController | None = None
        self._controller_key: tuple[str, int, float] | None = None
        self._channels: dict[str, _CompactChannel] = {}
        self._debounce: dict[str, QTimer] = {}

        self._build_ui()
        self._connect_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        light_config = self._config.light_controller
        for title, role, channel in self.CHANNELS:
            item = _CompactChannel(title, role, channel, getattr(light_config, role))
            self._channels[role] = item
            layout.addWidget(item.group)

        self._status = QLabel(self._connection_label())
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._status)

    def _connect_ui(self) -> None:
        for role, item in self._channels.items():
            item.enabled.toggled.connect(
                lambda checked, channel_item=item: self._toggle_channel(channel_item, checked)
            )
            item.slider.valueChanged.connect(
                lambda value, channel_item=item: self._brightness_changed(channel_item, value)
            )
            item.spin.valueChanged.connect(
                lambda value, channel_item=item: self._brightness_changed(channel_item, value)
            )
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda channel_item=item: self._apply_brightness(channel_item))
            self._debounce[role] = timer

    def close_controller(self) -> None:
        if self._controller is not None:
            self._controller.close()
            self._controller = None
            self._controller_key = None

    def _connection_label(self) -> str:
        light_config = self._config.light_controller
        return f"{light_config.device or '/dev/ttyUSB0'} @ {int(light_config.baud or 9600)}"

    def _current_key(self) -> tuple[str, int, float]:
        light_config = self._config.light_controller
        return (
            str(light_config.device or "/dev/ttyUSB0"),
            int(light_config.baud or 9600),
            float(light_config.timeout_s or 0.7),
        )

    def _ensure_controller(self) -> LightController:
        key = self._current_key()
        if self._controller is not None and self._controller_key == key:
            return self._controller
        self.close_controller()
        device, baud, timeout_s = key
        controller = LightController(device=device, baud=baud, timeout_s=timeout_s)
        controller.open()
        self._controller = controller
        self._controller_key = key
        return controller

    def _save_config(self) -> None:
        light_config = self._config.light_controller
        for role, item in self._channels.items():
            setattr(light_config, role, item.to_config())
        self._config.save()

    def _set_error(self, message: str) -> None:
        self._status.setText(message)

    def _toggle_channel(self, item: _CompactChannel, checked: bool) -> None:
        item.enabled.setText(tr("On") if checked else tr("Off"))
        try:
            ctrl = self._ensure_controller()
            if checked:
                ctrl.set_brightness(item.channel, item.brightness())
                ctrl.open_channel(item.channel)
            else:
                ctrl.close_channel(item.channel)
            self._save_config()
            self._status.setText(f"{item.title} {'ON' if checked else 'OFF'}")
        except Exception as exc:
            item.set_enabled(not checked)
            self._set_error(f"{item.title}: {exc}")

    def _brightness_changed(self, item: _CompactChannel, value: int) -> None:
        item.set_brightness(value)
        self._debounce[item.role].start(180)

    def _apply_brightness(self, item: _CompactChannel) -> None:
        try:
            if item.is_enabled():
                self._ensure_controller().set_brightness(item.channel, item.brightness())
                self._status.setText(f"{item.title} brightness {item.brightness()}")
            self._save_config()
        except Exception as exc:
            self._set_error(f"{item.title}: {exc}")
