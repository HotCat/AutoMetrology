"""
converters — standalone DWG-to-DXF conversion package.

Supports multiple backends:
  - ODA File Converter (preferred)
  - libredwg dwg2dxf (fallback)

Zero Qt dependencies. Auto-detects available backend.
"""

from .converter_config import (
    ConversionConfig,
    ConversionResult,
    ODAInstallationInfo,
    ConversionError,
    ODANotFoundError,
    ODAExecutionError,
    ODATimeoutError,
    DXFValidationError,
    DWGFileError,
    DWGVersionUnsupportedError,
)
from .oda_cli import ODACLI, LibreDWGCLI, auto_detect_backend
from .dwg_converter import DWGConverter
from .validation import DXFValidator, ValidationReport
from .handle_mapper import HandleMapper
