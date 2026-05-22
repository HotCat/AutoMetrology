"""
converter_config — data classes and error hierarchy for DWG conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .validation import ValidationReport


# ── Error hierarchy ──────────────────────────────────────────────────


class ConversionError(Exception):
    """Base error for DWG conversion."""


class ODANotFoundError(ConversionError):
    """ODA File Converter executable not found."""


class ODAExecutionError(ConversionError):
    """ODA subprocess failed (non-zero exit code)."""

    def __init__(self, exit_code: int, stderr: str) -> None:
        self.exit_code = exit_code
        self.stderr = stderr
        super().__init__(f"ODA exited with code {exit_code}: {stderr[:500]}")


class ODATimeoutError(ConversionError):
    """ODA subprocess exceeded timeout."""


class DXFValidationError(ConversionError):
    """Post-conversion DXF validation failed."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__(f"DXF validation failed: {'; '.join(errors[:3])}")


class DWGFileError(ConversionError):
    """Source DWG file is invalid or unreadable."""


class DWGVersionUnsupportedError(ConversionError):
    """DWG version not supported by installed ODA."""


# ── Data classes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConversionConfig:
    """Configuration for a single DWG→DXF conversion."""

    dwg_path: Path
    output_dir: Path
    output_version: str = "ACAD2013"
    audit: bool = True
    recurse: bool = False
    file_filter: str = "*.dwg"
    overwrite: bool = True
    timeout_seconds: int = 300


@dataclass
class ConversionResult:
    """Result of a DWG→DXF conversion attempt."""

    success: bool
    dwg_path: Path
    dxf_path: Optional[Path]
    config: ConversionConfig
    duration_seconds: float
    validation: Optional[ValidationReport]
    error_message: Optional[str] = None
    oda_exit_code: int = 0
    oda_stderr: str = ""
    entity_count: int = 0
    layer_count: int = 0


@dataclass
class ODAInstallationInfo:
    """Information about ODA File Converter installation."""

    installed: bool
    executable_path: Optional[Path] = None
    version: Optional[str] = None
    platform: str = ""
