"""
AppConfig — persistent application settings stored as JSON.

Saved to ~/.config/cadviewer/settings.json on exit, loaded on startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

_CONFIG_DIR = Path.home() / ".config" / "cadviewer"
_CONFIG_FILE = _CONFIG_DIR / "settings.json"


@dataclass
class CameraConfig:
    exposure_us: int = 30000
    gamma: int = 100
    contrast: int = 100
    analog_gain: int = 16
    ae_enabled: bool = False
    reverse_x: bool = False
    reverse_y: bool = False


@dataclass
class CalibrationConfig:
    chessboard_cols: int = 11
    chessboard_rows: int = 8
    chessboard_cell_mm: float = 21.0
    chessboard_image_path: str = ""


@dataclass
class LensCalibrationConfig:
    camera_matrix: list = field(default_factory=list)
    dist_coeffs: list = field(default_factory=list)
    reprojection_error: float = 0.0
    calibrated: bool = False
    image_count: int = 0
    coordinate_correction: dict = field(default_factory=dict)
    correction_model_type: str = "none"  # "none", "affine", "homography"

    def get_camera_matrix(self):
        if HAS_NUMPY and len(self.camera_matrix) == 9:
            return np.array(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        return None

    def get_dist_coeffs(self):
        if HAS_NUMPY and self.dist_coeffs:
            return np.array(self.dist_coeffs, dtype=np.float64)
        return None

    def set_from_results(self, camera_matrix, dist_coeffs, rms_error: float,
                         image_count: int) -> None:
        if HAS_NUMPY:
            self.camera_matrix = camera_matrix.flatten().tolist()
            self.dist_coeffs = dist_coeffs.flatten().tolist()
        else:
            self.camera_matrix = list(camera_matrix.flatten())
            self.dist_coeffs = list(dist_coeffs.flatten())
        self.reprojection_error = rms_error
        self.image_count = image_count
        self.calibrated = True


@dataclass
class AppConfig:
    pixel_size_mm: float = 0.01
    last_image_path: str = ""
    last_dxf_path: str = ""
    camera: CameraConfig = field(default_factory=CameraConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    lens_calibration: LensCalibrationConfig = field(default_factory=LensCalibrationConfig)
    production_profiles: list = field(default_factory=list)
    active_production_profile: str = ""
    last_dxf_file: str = ""
    language: str = "en"

    @staticmethod
    def load() -> AppConfig:
        if not _CONFIG_FILE.exists():
            return AppConfig()
        try:
            data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            cam_data = data.pop("camera", {})
            cal_data = data.pop("calibration", {})
            lens_data = data.pop("lens_calibration", {})
            data.pop("registration_groups", None)
            production_profiles = data.pop("production_profiles", [])
            active_production_profile = data.pop("active_production_profile", "")
            # Backward compat: remove old TPS residual_map field
            lens_data.pop("residual_map", None)
            cfg = AppConfig(**data)
            cfg.camera = CameraConfig(**cam_data)
            cfg.calibration = CalibrationConfig(**cal_data)
            cfg.lens_calibration = LensCalibrationConfig(**lens_data)
            cfg.production_profiles = (
                production_profiles if isinstance(production_profiles, list) else []
            )
            cfg.active_production_profile = (
                active_production_profile
                if isinstance(active_production_profile, str) else ""
            )
            return cfg
        except Exception:
            return AppConfig()

    def save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        _CONFIG_FILE.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
