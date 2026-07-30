"""
AppConfig — persistent application settings stored as JSON.

Saved to ~/.config/cadviewer/settings.json on exit, loaded on startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

def _default_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "cadviewer"
        return Path.home() / "AppData" / "Roaming" / "cadviewer"
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "cadviewer"
    return Path.home() / ".config" / "cadviewer"


_CONFIG_DIR = _default_config_dir()
_CONFIG_FILE = _CONFIG_DIR / "settings.json"
_CONFIG_BACKUP_FILE = _CONFIG_DIR / "settings.json.bak"
_LAST_LOAD_STATUS = "ok"
_LAST_LOAD_ERROR = ""




def _json_safe(value: Any) -> Any:
    if HAS_NUMPY:
        if isinstance(value, np.ndarray):
            return [_json_safe(v) for v in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


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
class LightChannelConfig:
    brightness: int = 180
    enabled: bool = False


@dataclass
class LightControllerConfig:
    device: str = "/dev/ttyUSB0"
    baud: int = 9600
    timeout_s: float = 0.7
    backlight_settle_delay_ms: int = 200
    ring_light_settle_delay_ms: int = 200
    ring_ch1: LightChannelConfig = field(default_factory=LightChannelConfig)
    ring_ch2: LightChannelConfig = field(default_factory=LightChannelConfig)
    backlight_ch4: LightChannelConfig = field(default_factory=LightChannelConfig)


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
    calibration_model: str = "standard"
    calibration_flags: int = 0
    calibrated: bool = False
    image_count: int = 0
    image_size: list = field(default_factory=list)
    coordinate_correction: dict = field(default_factory=dict)
    correction_model_type: str = "none"  # "none", "affine", "homography"
    residual_map: dict = field(default_factory=dict)

    def get_camera_matrix(self):
        if HAS_NUMPY and len(self.camera_matrix) == 9:
            return np.array(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        return None

    def get_dist_coeffs(self):
        if HAS_NUMPY and self.dist_coeffs:
            return np.array(self.dist_coeffs, dtype=np.float64)
        return None

    def get_image_size(self) -> tuple[int, int] | None:
        """Return calibration image size as (width, height), if known."""
        candidates = [self.image_size]
        if isinstance(self.residual_map, dict):
            candidates.append(self.residual_map.get("image_size"))
        if isinstance(self.coordinate_correction, dict):
            metadata = self.coordinate_correction.get("metadata", {})
            if isinstance(metadata, dict):
                candidates.append(metadata.get("image_size"))

        for candidate in candidates:
            if not candidate or len(candidate) < 2:
                continue
            width, height = int(candidate[0]), int(candidate[1])
            if width > 0 and height > 0:
                return width, height
        return None

    def set_from_results(self, camera_matrix, dist_coeffs, rms_error: float,
                         image_count: int,
                         image_size: tuple[int, int] | None = None,
                         calibration_model: str = "standard",
                         calibration_flags: int = 0) -> None:
        if HAS_NUMPY:
            self.camera_matrix = camera_matrix.flatten().tolist()
            self.dist_coeffs = dist_coeffs.flatten().tolist()
        else:
            self.camera_matrix = list(camera_matrix.flatten())
            self.dist_coeffs = list(dist_coeffs.flatten())
        self.reprojection_error = rms_error
        self.calibration_model = str(calibration_model or "standard")
        self.calibration_flags = int(calibration_flags or 0)
        self.image_count = image_count
        self.image_size = list(image_size) if image_size is not None else []
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
    light_controller: LightControllerConfig = field(default_factory=LightControllerConfig)
    line_fit_side_overrides: dict = field(default_factory=dict)
    apply_correction_map: bool = True
    dual_light_orientation_guard_enabled: bool = True
    measurement_queries: str = ""

    @staticmethod
    def load() -> AppConfig:
        global _LAST_LOAD_STATUS, _LAST_LOAD_ERROR
        _LAST_LOAD_STATUS = "ok"
        _LAST_LOAD_ERROR = ""
        if not _CONFIG_FILE.exists():
            return AppConfig()
        try:
            data = json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
            _snapshot_config_backup()
            return _build_config_from_dict(data)
        except Exception as exc:
            _LAST_LOAD_ERROR = str(exc)
            backup = _try_load_backup()
            if backup is not None:
                _LAST_LOAD_STATUS = "recovered_from_backup"
                return backup
            _LAST_LOAD_STATUS = "load_failed"
            return AppConfig()

    def save(self) -> None:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if _LAST_LOAD_STATUS == "load_failed" and _CONFIG_FILE.exists():
            # Refuse to overwrite the only on-disk config with defaults when we
            # know startup could not parse the existing file.
            return
        data = _json_safe(asdict(self))
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=str(_CONFIG_DIR),
                prefix="settings.",
                suffix=".tmp",
            ) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                tmp_path = Path(handle.name)
            if _CONFIG_FILE.exists():
                try:
                    shutil.copy2(_CONFIG_FILE, _CONFIG_BACKUP_FILE)
                except Exception:
                    pass
            os.replace(tmp_path, _CONFIG_FILE)
            try:
                shutil.copy2(_CONFIG_FILE, _CONFIG_BACKUP_FILE)
            except Exception:
                pass
        finally:
            if tmp_path is not None and tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass


def _build_config_from_dict(data: dict[str, Any]) -> AppConfig:
    cam_data = data.pop("camera", {})
    cal_data = data.pop("calibration", {})
    lens_data = data.pop("lens_calibration", {})
    light_data = data.pop("light_controller", {})
    data.pop("registration_groups", None)
    production_profiles = data.pop("production_profiles", [])
    active_production_profile = data.pop("active_production_profile", "")
    cfg = AppConfig(**data)
    cfg.camera = CameraConfig(**cam_data)
    cfg.calibration = CalibrationConfig(**cal_data)
    cfg.lens_calibration = LensCalibrationConfig(**lens_data)
    cfg.light_controller = _load_light_controller_config(light_data)
    cfg.production_profiles = (
        production_profiles if isinstance(production_profiles, list) else []
    )
    cfg.active_production_profile = (
        active_production_profile
        if isinstance(active_production_profile, str) else ""
    )
    return cfg


def _try_load_backup() -> AppConfig | None:
    if not _CONFIG_BACKUP_FILE.exists():
        return None
    try:
        data = json.loads(_CONFIG_BACKUP_FILE.read_text(encoding="utf-8"))
        return _build_config_from_dict(data)
    except Exception:
        return None


def _snapshot_config_backup() -> None:
    try:
        if _CONFIG_FILE.exists():
            _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_CONFIG_FILE, _CONFIG_BACKUP_FILE)
    except Exception:
        pass


def _load_light_channel_config(data: Any) -> LightChannelConfig:
    if not isinstance(data, dict):
        return LightChannelConfig()
    return LightChannelConfig(
        brightness=max(0, min(255, int(data.get("brightness", 180)))),
        enabled=bool(data.get("enabled", False)),
    )


def _load_light_controller_config(data: Any) -> LightControllerConfig:
    if not isinstance(data, dict):
        return LightControllerConfig()
    return LightControllerConfig(
        device=str(data.get("device", "/dev/ttyUSB0") or "/dev/ttyUSB0"),
        baud=int(data.get("baud", 9600) or 9600),
        timeout_s=float(data.get("timeout_s", 0.7) or 0.7),
        backlight_settle_delay_ms=max(
            0, int(data.get("backlight_settle_delay_ms", 200) or 0),
        ),
        ring_light_settle_delay_ms=max(
            0, int(data.get("ring_light_settle_delay_ms", 200) or 0),
        ),
        ring_ch1=_load_light_channel_config(data.get("ring_ch1")),
        ring_ch2=_load_light_channel_config(data.get("ring_ch2")),
        backlight_ch4=_load_light_channel_config(data.get("backlight_ch4")),
    )
