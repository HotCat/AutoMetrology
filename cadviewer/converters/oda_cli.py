"""
oda_cli — ODA File Converter and libredwg command-line interface wrapper.

Handles executable discovery for both ODA File Converter and libredwg's
dwg2dxf tool, argument construction, version detection, and DWG magic-byte
version reading.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

from .converter_config import ConversionConfig, ODAInstallationInfo

logger = logging.getLogger(__name__)

# DWG version magic bytes → AutoCAD version
DWG_VERSION_MAP: dict[str, str] = {
    "AC1032": "AutoCAD 2018+",
    "AC1027": "AutoCAD 2013",
    "AC1024": "AutoCAD 2010",
    "AC1021": "AutoCAD 2007",
    "AC1018": "AutoCAD 2004",
    "AC1015": "AutoCAD 2000",
    "AC1014": "AutoCAD R14",
    "AC1012": "AutoCAD R13",
}

SUPPORTED_OUTPUT_VERSIONS = [
    "ACAD2013", "ACAD2010", "ACAD2007", "ACAD2004", "ACAD2000", "R14",
]


def _which(name: str) -> Optional[Path]:
    """Find executable on PATH."""
    path_env = os.environ.get("PATH", "").split(os.pathsep)
    for dir_str in path_env:
        candidate = Path(dir_str) / name
        if candidate.is_file():
            return candidate
    return None


def detect_dwg_version(path: Path) -> Optional[str]:
    """Read DWG magic bytes to determine AutoCAD version."""
    try:
        with open(path, "rb") as f:
            magic = f.read(6).decode("ascii", errors="ignore")
        return DWG_VERSION_MAP.get(magic)
    except (OSError, ValueError):
        return None


def read_dwg_magic(path: Path) -> Optional[str]:
    """Read raw DWG magic bytes string (e.g. 'AC1021')."""
    try:
        with open(path, "rb") as f:
            return f.read(6).decode("ascii", errors="ignore")
    except (OSError, ValueError):
        return None


class ODACLI:
    """Encapsulates ODA File Converter command-line interface."""

    EXECUTABLE_NAMES: dict[str, list[str]] = {
        "Linux": ["ODAFileConverter", "odafileconverter"],
        "Windows": ["ODAFileConverter.exe"],
        "Darwin": ["ODAFileConverter"],
    }

    SEARCH_PATHS: dict[str, list[Path]] = {
        "Linux": [
            Path("/usr/bin"), Path("/usr/local/bin"),
            Path("/opt/oda"), Path.home() / ".local/bin",
        ],
        "Windows": [
            Path("C:/Program Files/ODA/ODAFileConverter"),
            Path("C:/Program Files (x86)/ODA/ODAFileConverter"),
        ],
        "Darwin": [
            Path("/Applications/ODAFileConverter.app/Contents/MacOS"),
            Path("/usr/local/bin"),
        ],
    }

    def __init__(self, executable_path: Optional[Path] = None) -> None:
        self._override_path = executable_path

    @property
    def backend_name(self) -> str:
        return "ODA File Converter"

    def find_executable(self) -> Optional[Path]:
        if self._override_path and self._override_path.is_file():
            return self._override_path
        system = platform.system()
        names = self.EXECUTABLE_NAMES.get(system, self.EXECUTABLE_NAMES["Linux"])
        for search_dir in self.SEARCH_PATHS.get(system, []):
            for name in names:
                candidate = search_dir / name
                if candidate.is_file():
                    return candidate
        for name in names:
            path = _which(name)
            if path:
                return path
        return None

    def get_installation_info(self) -> ODAInstallationInfo:
        exe = self.find_executable()
        if exe is None:
            return ODAInstallationInfo(installed=False, platform=platform.system())
        return ODAInstallationInfo(
            installed=True, executable_path=exe,
            version=self.get_version(), platform=platform.system(),
        )

    def build_args(self, config: ConversionConfig) -> list[str]:
        """Build ODA CLI argument list.

        ODA CLI format (all positional):
          ODAFileConverter <Input Folder> <Output Folder>
                           <Output version> <Output File type>
                           <Recurse> <Audit> [Input files filter]

        Example:
          ODAFileConverter /tmp/src /tmp/out ACAD2013 DXF 0 1 "H89-上壳.dwg"
        """
        exe = self.find_executable()
        if exe is None:
            raise FileNotFoundError("ODA File Converter not found")
        return [
            str(exe),
            str(config.dwg_path.parent),
            str(config.output_dir),
            config.output_version,
            "DXF",
            "1" if config.recurse else "0",
            "1" if config.audit else "0",
            config.file_filter,
        ]

    def get_version(self) -> Optional[str]:
        """Return version string. ODA is a GUI app — cannot be queried via CLI."""
        exe = self.find_executable()
        if exe is None:
            return None
        # ODA File Converter is a GUI application that shows a dialog when
        # called with --help or --version. Just report that it exists.
        return f"ODA File Converter ({exe.name})"

    @staticmethod
    def output_dxf_path(config: ConversionConfig) -> Path:
        return config.output_dir / f"{config.dwg_path.stem}.dxf"

    @staticmethod
    def supported_output_versions() -> list[str]:
        return list(SUPPORTED_OUTPUT_VERSIONS)


class LibreDWGCLI:
    """Encapsulates libredwg's dwg2dxf command-line interface."""

    EXECUTABLE_NAMES = ["dwg2dxf"]

    def __init__(self, executable_path: Optional[Path] = None) -> None:
        self._override_path = executable_path

    @property
    def backend_name(self) -> str:
        return "libredwg (dwg2dxf)"

    def find_executable(self) -> Optional[Path]:
        if self._override_path and self._override_path.is_file():
            return self._override_path
        for name in self.EXECUTABLE_NAMES:
            path = _which(name)
            if path:
                return path
        return None

    def get_installation_info(self) -> ODAInstallationInfo:
        exe = self.find_executable()
        if exe is None:
            return ODAInstallationInfo(installed=False, platform=platform.system())
        return ODAInstallationInfo(
            installed=True, executable_path=exe,
            version=self.get_version(), platform=platform.system(),
        )

    def build_args(self, config: ConversionConfig) -> list[str]:
        """Build dwg2dxf argument list.

        dwg2dxf [-v N] [-o output.dxf] input.dwg
        """
        exe = self.find_executable()
        if exe is None:
            raise FileNotFoundError("dwg2dxf not found")
        output = config.output_dir / f"{config.dwg_path.stem}.dxf"
        return [
            str(exe),
            "-v", "1",
            "-o", str(output),
            str(config.dwg_path),
        ]

    def get_version(self) -> Optional[str]:
        exe = self.find_executable()
        if exe is None:
            return None
        try:
            result = subprocess.run(
                [str(exe), "--version"],
                capture_output=True, text=True, timeout=10,
            )
            output = (result.stdout + result.stderr).strip()
            return output.splitlines()[0] if output else None
        except (subprocess.TimeoutExpired, OSError):
            return None

    @staticmethod
    def output_dxf_path(config: ConversionConfig) -> Path:
        return config.output_dir / f"{config.dwg_path.stem}.dxf"

    @staticmethod
    def supported_output_versions() -> list[str]:
        return []  # libredwg auto-detects version


def auto_detect_backend(oda_path: Optional[Path] = None) -> Optional[ODACLI | LibreDWGCLI]:
    """Find the best available conversion backend.

    Returns the first available backend, or None if nothing is found.
    Priority: ODA File Converter > libredwg dwg2dxf
    """
    oda = ODACLI(executable_path=oda_path)
    if oda.find_executable():
        logger.info("Using ODA File Converter backend")
        return oda

    libredwg = LibreDWGCLI()
    if libredwg.find_executable():
        logger.info("Using libredwg (dwg2dxf) backend")
        return libredwg

    return None
