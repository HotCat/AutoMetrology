# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building AutoMetrology on Windows 10 x64.

Build command, from an activated Windows Python environment:

    pyinstaller --clean --noconfirm AutoMetrology-windows10.spec

The output is a one-folder application under:

    dist\\AutoMetrology\\AutoMetrology.exe

Hikvision/Hikrobot MVS camera support:
    The application can run without the MVS SDK, but Hikvision camera capture
    requires the official SDK. This spec supports two deployment styles:

    1. Preferred deployment: install MVS on the target machine in its standard
       location or set HIKVISION_MVS_ROOT before launching AutoMetrology.
    2. Optional bundled deployment: if the MVS SDK exists on the build machine,
       this spec copies the Python wrapper and 64-bit runtime DLLs into the
       PyInstaller folder. Set AUTOMETROLOGY_BUNDLE_MVS=0 to skip this.

    Optional build-time path overrides:
       HIKVISION_MVS_ROOT      root SDK folder, for example C:\\Program Files (x86)\\MVS
       HIKVISION_MVS_MVIMPORT  folder containing MvCameraControl_class.py
       HIKVISION_MVS_RUNTIME   folder containing MvCameraControl.dll

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


ROOT = Path(globals().get("SPECPATH", ".")).resolve()
MVS_ROOT = Path(os.environ.get("HIKVISION_MVS_ROOT", r"C:\Program Files (x86)\MVS"))
BUNDLE_MVS = os.environ.get("AUTOMETROLOGY_BUNDLE_MVS", "1").lower() not in {
    "0",
    "false",
    "no",
}


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


def first_existing(paths: list[Path]) -> Path | None:
    """Return the first existing path from likely vendor SDK locations."""
    for path in paths:
        if path.exists():
            return path
    return None


def mvs_import_dir() -> Path | None:
    """Find Hikvision's flat Python wrapper module directory."""
    override = os.environ.get("HIKVISION_MVS_MVIMPORT")
    if override:
        path = Path(override)
        return path if path.exists() else None

    roots = [
        MVS_ROOT,
        Path(r"C:\Program Files (x86)\MVS"),
        Path(r"C:\Program Files\MVS"),
    ]
    candidates: list[Path] = []
    for root in roots:
        candidates += [
            root / "Development" / "Samples" / "Python" / "MvImport",
            root / "Samples" / "64" / "Python" / "MvImport",
            root / "Samples" / "Python" / "MvImport",
        ]
    return first_existing(candidates)


def mvs_runtime_dir() -> Path | None:
    """Find the 64-bit MVS runtime DLL directory."""
    override = os.environ.get("HIKVISION_MVS_RUNTIME")
    if override:
        path = Path(override)
        return path if path.exists() else None

    roots = [
        MVS_ROOT,
        Path(r"C:\Program Files (x86)\MVS"),
        Path(r"C:\Program Files\MVS"),
    ]
    candidates: list[Path] = []
    for root in roots:
        candidates += [
            root / "Development" / "Runtime" / "Win64_x64",
            root / "Runtime" / "Win64_x64",
            root / "bin" / "win64",
            root / "Bin" / "Win64",
            root / "lib" / "win64",
        ]
    return first_existing(candidates)


MVS_IMPORT_DIR = mvs_import_dir()
MVS_RUNTIME_DIR = mvs_runtime_dir()


def mvs_pathex() -> list[str]:
    """Expose Hikvision's flat Python wrapper modules to PyInstaller analysis."""
    if BUNDLE_MVS and MVS_IMPORT_DIR is not None:
        return [str(MVS_IMPORT_DIR)]
    return []


def mvs_hiddenimports() -> list[str]:
    """Hikvision wrapper modules use flat imports from MvImport."""
    if not (BUNDLE_MVS and MVS_IMPORT_DIR is not None):
        return []
    return [
        "MvCameraControl_class",
        "CameraParams_const",
        "CameraParams_header",
        "MvErrorDefine_const",
        "MvISPErrorDefine_const",
        "PixelType_header",
    ]


def mvs_datas() -> list[tuple[str, str]]:
    """Bundle Hikvision Python wrapper and non-DLL transport metadata."""
    if not BUNDLE_MVS:
        return []

    datas: list[tuple[str, str]] = []
    if MVS_IMPORT_DIR is not None:
        datas += [
            (str(path), "mvs/MvImport")
            for path in MVS_IMPORT_DIR.glob("*.py")
        ]

    if MVS_RUNTIME_DIR is not None:
        for pattern in ("*.cti", "*.ini", "*.xml"):
            datas += [
                (str(path), "mvs/runtime")
                for path in MVS_RUNTIME_DIR.rglob(pattern)
                if path.is_file()
            ]
    return datas


def mvs_binaries() -> list[tuple[str, str]]:
    """Bundle Hikvision runtime DLLs where the Windows runtime hook can find them."""
    if not (BUNDLE_MVS and MVS_RUNTIME_DIR is not None):
        return []

    return [
        (str(path), "mvs/runtime")
        for path in MVS_RUNTIME_DIR.rglob("*.dll")
        if path.is_file()
    ]


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
    "cadviewer.camera.hikvision",
    "cadviewer.camera.device",
    "cadviewer.camera.driver.mvsdk",
    "diplib",
    "scipy.interpolate",
    "scipy.interpolate._rbfinterp",
    "scipy.spatial",
    "scipy.spatial._ckdtree",
    "scipy.spatial.distance",
]
hiddenimports += mvs_hiddenimports()
hiddenimports += optional_collect_submodules("ezdxf")
hiddenimports += optional_collect_submodules("serial")

# OpenCascade is optional in this application. The default runtime uses the
# QPainter canvas, but this keeps packaged builds usable if pythonocc-core is
# installed in the build environment.
hiddenimports += optional_collect_submodules("OCC")


datas = []
datas += optional_collect_data_files("ezdxf")
datas += optional_collect_data_files("PySide6")
datas += mvs_datas()


binaries = []
binaries += optional_collect_dynamic_libs("cv2")
binaries += optional_collect_dynamic_libs("numpy")
binaries += optional_collect_dynamic_libs("scipy")
binaries += optional_collect_dynamic_libs("diplib")
binaries += mvs_binaries()
binaries += mindvision_binaries()


a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)] + mvs_pathex(),
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(ROOT / "cadviewer" / "packaging" / "pyinstaller_windows_runtime.py"),
    ],
    excludes=[
        # Keep the deployment smaller by excluding modules that are not used by
        # the desktop inspection application.
        "matplotlib",
        "pandas",
        "pytest",
        "tkinter",
        "IPython",
        "jupyter",
        "scipy.spatial.tests",
        "scipy.interpolate.tests",
        "torch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "triton",
        "nvidia",
        "sklearn",
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
