"""PyInstaller runtime setup for Ubuntu/Linux deployments.

This hook prepares the Hikvision/Hikrobot MVS Python wrapper before the
application imports cadviewer.camera.hikvision.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)


def _configure_hikvision_mvs() -> None:
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    bundled = base / "mvs"

    if bundled.exists():
        os.environ.setdefault("HIKVISION_MVS_ROOT", str(bundled))
        os.environ["MVCAM_COMMON_RUNENV"] = str(bundled / "lib")
        _prepend_sys_path(bundled / "MvImport")
        return

    root = Path(os.environ.get("HIKVISION_MVS_ROOT", "/opt/MVS"))
    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(root / "lib"))
    _prepend_sys_path(root / "Samples" / "64" / "Python" / "MvImport")


_configure_hikvision_mvs()
