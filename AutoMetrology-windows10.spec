# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building AutoMetrology on Windows 10 x64.

Build command, from an activated Windows Python environment:

    pyinstaller --clean --noconfirm AutoMetrology-windows10.spec

The output is a one-folder application under:

    dist\\AutoMetrology\\AutoMetrology.exe

MindVision camera support:
    The application can run without a camera SDK installed, but production
    camera capture requires MVCAMSDK_X64.dll. This spec intentionally keeps
    the SDK optional:

    1. Preferred deployment: install the official MindVision SDK on the target
       machine and add its Runtime\\Win64_x64 directory to PATH.
    2. Optional bundled deployment: set MINDVISION_SDK_RUNTIME before building,
       pointing to the folder that contains MVCAMSDK_X64.dll. The DLL will be
       copied beside the EXE so ctypes.windll can find it.

DWG support:
    DWG import uses an external ODA File Converter installation. The converter
    executable is not bundled here because ODA licensing and install paths are
    machine-specific. Install it separately if DWG import is required.
"""

from __future__ import annotations

import os
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


ROOT = Path.cwd()


def optional_collect_submodules(package: str) -> list[str]:
    """Collect hidden imports only when an optional package is installed."""
    try:
        return collect_submodules(package)
    except Exception:
        return []


def optional_collect_data_files(package: str) -> list[tuple[str, str]]:
    """Collect package data only when an optional package is installed."""
    try:
        return collect_data_files(package)
    except Exception:
        return []


def optional_collect_dynamic_libs(package: str) -> list[tuple[str, str]]:
    """Collect package binary libraries only when an optional package is installed."""
    try:
        return collect_dynamic_libs(package)
    except Exception:
        return []


def mindvision_binaries() -> list[tuple[str, str]]:
    """Optionally bundle the MindVision runtime DLL beside AutoMetrology.exe."""
    runtime_dir = os.environ.get(
        "MINDVISION_SDK_RUNTIME",
        r"C:\Program Files\MindVision\MVCAMSDK\Runtime\Win64_x64",
    )
    dll_path = Path(runtime_dir) / "MVCAMSDK_X64.dll"
    if dll_path.exists():
        return [(str(dll_path), ".")]
    return []


# Hidden imports are intentionally explicit. PyInstaller usually discovers
# normal Python imports, but Qt plugins, OpenCV extension modules, ezdxf add-ons,
# scipy compiled modules, and optional DIPLib/OCC packages can otherwise be
# missed depending on the builder machine.
hiddenimports = [
    "cv2",
    "numpy",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "cadviewer.camera.driver.mvsdk",
]
hiddenimports += optional_collect_submodules("ezdxf")
hiddenimports += optional_collect_submodules("scipy.spatial")
hiddenimports += optional_collect_submodules("scipy.interpolate")
hiddenimports += optional_collect_submodules("diplib")

# OpenCascade is optional in this application. The default runtime uses the
# QPainter canvas, but this keeps packaged builds usable if pythonocc-core is
# installed in the build environment.
hiddenimports += optional_collect_submodules("OCC")


datas = []
datas += optional_collect_data_files("ezdxf")
datas += optional_collect_data_files("PySide6")


binaries = []
binaries += optional_collect_dynamic_libs("cv2")
binaries += optional_collect_dynamic_libs("numpy")
binaries += optional_collect_dynamic_libs("scipy")
binaries += optional_collect_dynamic_libs("diplib")
binaries += mindvision_binaries()


a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Keep the deployment smaller by excluding modules that are not used by
        # the desktop inspection application.
        "matplotlib",
        "pandas",
        "pytest",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoMetrology",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AutoMetrology",
)
