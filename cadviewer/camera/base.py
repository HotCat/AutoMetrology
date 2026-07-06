"""Shared camera API data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal


@dataclass
class CameraSettings:
    exposure_us: int = 30000
    gamma: int = 100
    contrast: int = 100
    analog_gain: int = 16
    ae_enabled: bool = False
    reverse_x: bool = False
    reverse_y: bool = False


@dataclass
class CameraSettingRanges:
    exposure_min_us: int = 100
    exposure_max_us: int = 1000000
    exposure_step_us: int = 100
    gamma_min: int = 1
    gamma_max: int = 500
    contrast_min: int = 1
    contrast_max: int = 500
    analog_gain_min: int = 0
    analog_gain_max: int = 100


class CameraSignalEmitter(QObject):
    frame_ready = Signal(np.ndarray)
    grab_done = Signal(np.ndarray)
    error = Signal(str)
