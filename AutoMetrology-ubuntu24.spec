# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for building AutoMetrology on Ubuntu 24.04 x86_64.

Build command, from an activated Python environment on Ubuntu 24.04:

    pyinstaller --clean --noconfirm AutoMetrology-ubuntu24.spec

The output is a one-folder application under:

    dist/AutoMetrology/AutoMetrology

Hikvision/Hikrobot MVS camera support:
    The application can run without the MVS SDK, but camera capture requires
    the official SDK. This spec supports two deployment styles:

    1. Preferred deployment: install MVS on the target machine under /opt/MVS
       or set HIKVISION_MVS_ROOT before launching AutoMetrology.
    2. Optional bundled deployment: if /opt/MVS exists on the build machine,
       this spec copies the 64-bit MVS Python wrapper and runtime files into
       the PyInstaller folder. Set AUTOMETROLOGY_BUNDLE_MVS=0 to skip this.

MindVision camera support:
    Linux MindVision ctypes bindings are still imported as an optional
    fallback if available in the source tree. The vendor shared libraries are
    not bundled by this spec; install the SDK separately if that backend is
    needed on a Linux station.

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
MVS_ROOT = Path(os.environ.get("HIKVISION_MVS_ROOT", "/opt/MVS"))
MVS_IMPORT_DIR = MVS_ROOT / "Samples" / "64" / "Python" / "MvImport"
MVS_LIB64_DIR = MVS_ROOT / "lib" / "64"
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


def mvs_pathex() -> list[str]:
    """Expose Hikvision's flat Python wrapper modules to PyInstaller analysis."""
    if BUNDLE_MVS and MVS_IMPORT_DIR.exists():
        return [str(MVS_IMPORT_DIR)]
    return []


def mvs_hiddenimports() -> list[str]:
    """Hikvision wrapper modules use flat imports from MvImport."""
    if not (BUNDLE_MVS and MVS_IMPORT_DIR.exists()):
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
    """Bundle MVS Python wrapper files and non-library transport metadata."""
    if not BUNDLE_MVS:
        return []

    datas: list[tuple[str, str]] = []
    if MVS_IMPORT_DIR.exists():
        datas += [
            (str(path), "mvs/MvImport")
            for path in MVS_IMPORT_DIR.glob("*.py")
        ]

    if MVS_LIB64_DIR.exists():
        for pattern in ("*.cti", "*.ini"):
            datas += [
                (str(path), "mvs/lib/64")
                for path in MVS_LIB64_DIR.glob(pattern)
            ]
    return datas


def mvs_binaries() -> list[tuple[str, str]]:
    """Bundle MVS shared libraries in the layout expected by the SDK wrapper."""
    if not (BUNDLE_MVS and MVS_LIB64_DIR.exists()):
        return []

    binaries: list[tuple[str, str]] = []
    binaries += [
        (str(path), "mvs/lib/64")
        for path in MVS_LIB64_DIR.glob("*.so*")
        if path.is_file()
    ]

    third_party = MVS_LIB64_DIR / "ThirdParty"
    if third_party.exists():
        binaries += [
            (str(path), "mvs/lib/64/ThirdParty")
            for path in third_party.glob("*.so*")
            if path.is_file()
        ]
    return binaries


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


a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)] + mvs_pathex(),
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        str(ROOT / "cadviewer" / "packaging" / "pyinstaller_ubuntu_runtime.py"),
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
