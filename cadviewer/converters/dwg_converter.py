"""
dwg_converter — orchestrates DWG→DXF conversion via multiple backends.

Supports:
  - ODA File Converter (directory-based, preferred)
  - libredwg dwg2dxf (single-file, fallback)

Auto-detects the best available backend at construction time.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from .converter_config import (
    ConversionConfig,
    ConversionResult,
    ConversionError,
    DWGFileError,
    ODAExecutionError,
    ODANotFoundError,
    ODATimeoutError,
)
from .oda_cli import (
    ODACLI,
    LibreDWGCLI,
    auto_detect_backend,
    detect_dwg_version,
    read_dwg_magic,
)
from .validation import DXFValidator

logger = logging.getLogger(__name__)


class DWGConverter:
    """Orchestrates DWG to DXF conversion using available backend."""

    def __init__(
        self,
        backend: Optional[ODACLI | LibreDWGCLI] = None,
        oda_cli: Optional[ODACLI] = None,
    ) -> None:
        if backend is not None:
            self._backend = backend
        elif oda_cli is not None:
            self._backend = oda_cli
        else:
            self._backend = auto_detect_backend()

    @property
    def backend_name(self) -> str:
        if self._backend is None:
            return "none"
        return self._backend.backend_name

    def convert(self, config: ConversionConfig) -> ConversionResult:
        """Run a synchronous DWG→DXF conversion."""
        start = time.monotonic()
        source_dir: Optional[Path] = None
        output_dir: Optional[Path] = None

        try:
            self._validate_input(config)

            # ODA needs temp dirs; dwg2dxf works on single files
            if isinstance(self._backend, ODACLI):
                source_dir, output_dir = self._prepare_working_dirs(config)
                work_config = ConversionConfig(
                    dwg_path=source_dir / config.dwg_path.name,
                    output_dir=output_dir,
                    output_version=config.output_version,
                    audit=config.audit,
                    recurse=config.recurse,
                    file_filter="*.dwg",
                    overwrite=config.overwrite,
                    timeout_seconds=config.timeout_seconds,
                )
            else:
                # LibreDWG: output dir for single-file conversion
                os.makedirs(str(config.output_dir), exist_ok=True)
                work_config = config

            # Build args and run
            args = self._backend.build_args(work_config)
            logger.info("Running %s: %s", self.backend_name, " ".join(args))
            proc = self._run_subprocess(args, config.timeout_seconds)

            if proc.returncode != 0:
                raise ODAExecutionError(proc.returncode, proc.stderr)

            # Locate output DXF
            expected = self._backend.output_dxf_path(work_config)
            if not expected.exists():
                dxf_files = list(work_config.output_dir.glob("*.dxf"))
                if not dxf_files:
                    raise ODAExecutionError(
                        proc.returncode,
                        f"No DXF output found. stderr: {proc.stderr[:500]}",
                    )
                expected = dxf_files[0]

            # Validate
            validator = DXFValidator(original_dwg_path=config.dwg_path)
            report = validator.validate(expected)

            # Copy to final location if needed
            final_dxf = config.output_dir / expected.name
            if expected != final_dxf:
                os.makedirs(str(config.output_dir), exist_ok=True)
                shutil.copy2(str(expected), str(final_dxf))

            duration = time.monotonic() - start
            logger.info(
                "DWG conversion completed in %.1fs: %s (%d entities)",
                duration, final_dxf.name, report.entity_count,
            )

            return ConversionResult(
                success=report.is_valid,
                dwg_path=config.dwg_path,
                dxf_path=final_dxf,
                config=config,
                duration_seconds=duration,
                validation=report,
                error_message=None if report.is_valid else "; ".join(report.errors),
                oda_exit_code=proc.returncode,
                oda_stderr=proc.stderr,
                entity_count=report.entity_count,
                layer_count=report.layer_count,
            )

        except ConversionError as e:
            duration = time.monotonic() - start
            logger.error("DWG conversion failed: %s", e)
            return ConversionResult(
                success=False, dwg_path=config.dwg_path, dxf_path=None,
                config=config, duration_seconds=duration, validation=None,
                error_message=str(e),
            )
        except Exception as e:
            duration = time.monotonic() - start
            logger.exception("Unexpected error during DWG conversion")
            return ConversionResult(
                success=False, dwg_path=config.dwg_path, dxf_path=None,
                config=config, duration_seconds=duration, validation=None,
                error_message=f"Unexpected error: {e}",
            )
        finally:
            for d in (source_dir, output_dir):
                if d:
                    self._cleanup(d)

    def convert_async(
        self,
        config: ConversionConfig,
        callback: Callable[[ConversionResult], None],
    ) -> threading.Thread:
        def _worker():
            callback(self.convert(config))
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return thread

    def validate_installation(self):
        """Check backend availability."""
        if self._backend is None:
            from .converter_config import ODAInstallationInfo
            return ODAInstallationInfo(installed=False, platform="unknown")
        return self._backend.get_installation_info()

    def _validate_input(self, config: ConversionConfig) -> None:
        if not config.dwg_path.exists():
            raise DWGFileError(f"DWG file not found: {config.dwg_path}")
        if not self.is_dwg_file(config.dwg_path):
            raise DWGFileError(f"Not a DWG file: {config.dwg_path}")
        if self._backend is None or self._backend.find_executable() is None:
            raise ODANotFoundError(
                "No DWG converter found. Install ODA File Converter or "
                "libredwg (dwg2dxf), then configure via Settings."
            )

    def _prepare_working_dirs(
        self, config: ConversionConfig
    ) -> Tuple[Path, Path]:
        source_dir = Path(tempfile.mkdtemp(prefix="dwg_src_"))
        output_dir = Path(tempfile.mkdtemp(prefix="dwg_out_"))
        link_path = source_dir / config.dwg_path.name
        shutil.copy2(str(config.dwg_path), str(link_path))
        return source_dir, output_dir

    def _run_subprocess(
        self, args: list[str], timeout: int
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                args, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise ODATimeoutError(f"Converter timed out after {timeout}s")
        except FileNotFoundError:
            raise ODANotFoundError("Converter executable not found at runtime")

    def _cleanup(self, d: Path) -> None:
        try:
            shutil.rmtree(str(d), ignore_errors=True)
        except Exception as e:
            logger.warning("Failed to cleanup temp dir %s: %s", d, e)

    @staticmethod
    def is_dwg_file(path: Path) -> bool:
        if path.suffix.lower() != ".dwg":
            return False
        magic = read_dwg_magic(path)
        return magic is not None and magic.startswith("AC")

    @staticmethod
    def detect_dwg_version(path: Path) -> Optional[str]:
        return detect_dwg_version(path)
