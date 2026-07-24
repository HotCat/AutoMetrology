"""PyInstaller runtime setup for Windows deployments.

This hook prepares the Hikvision/Hikrobot MVS Python wrapper before the
application imports cadviewer.camera.hikvision.  The vendor wrapper loads
MvCameraControl.dll by name on Windows, so the runtime DLL folder must be added
to the process DLL search path before that import happens.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)


def _prepend_path_env(path: Path) -> None:
    if not path.exists():
        return
    text = str(path)
    entries = os.environ.get("PATH", "").split(os.pathsep)
    if text not in entries:
        os.environ["PATH"] = text + os.pathsep + os.environ.get("PATH", "")


def _add_dll_directory(path: Path) -> None:
    if not path.exists() or not hasattr(os, "add_dll_directory"):
        return
    try:
        os.add_dll_directory(str(path))
    except OSError:
        pass


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _configure_bundled_mvs(base: Path) -> bool:
    bundled = base / "mvs"
    import_dir = bundled / "MvImport"
    runtime_dir = bundled / "runtime"
    if not bundled.exists():
        return False

    os.environ.setdefault("HIKVISION_MVS_ROOT", str(bundled))
    os.environ.setdefault("MVCAM_COMMON_RUNENV", str(runtime_dir))
    _prepend_sys_path(import_dir)
    _prepend_path_env(runtime_dir)
    _add_dll_directory(runtime_dir)
    return True


def _configure_installed_mvs() -> None:
    root = Path(os.environ.get("HIKVISION_MVS_ROOT", r"C:\Program Files (x86)\MVS"))
    import_dir_text = os.environ.get("HIKVISION_MVS_MVIMPORT")
    runtime_dir_text = os.environ.get("HIKVISION_MVS_RUNTIME")
    import_dir = Path(import_dir_text) if import_dir_text else None
    runtime_dir = Path(runtime_dir_text) if runtime_dir_text else None
    roots = [
        root,
        Path(r"C:\Program Files (x86)\MVS"),
        Path(r"C:\Program Files\MVS"),
    ]

    if import_dir is None:
        import_candidates: list[Path] = []
        for candidate_root in roots:
            import_candidates += [
                candidate_root / "Development" / "Samples" / "Python" / "MvImport",
                candidate_root / "Samples" / "64" / "Python" / "MvImport",
                candidate_root / "Samples" / "Python" / "MvImport",
            ]
        import_dir = _first_existing(import_candidates)

    if runtime_dir is None:
        runtime_candidates: list[Path] = []
        for candidate_root in roots:
            runtime_candidates += [
                candidate_root / "Development" / "Runtime" / "Win64_x64",
                candidate_root / "Runtime" / "Win64_x64",
                candidate_root / "bin" / "win64",
                candidate_root / "Bin" / "Win64",
                candidate_root / "lib" / "win64",
            ]
        runtime_dir = _first_existing(runtime_candidates)

    if import_dir is not None:
        _prepend_sys_path(import_dir)
    if runtime_dir is not None:
        os.environ.setdefault("MVCAM_COMMON_RUNENV", str(runtime_dir))
        _prepend_path_env(runtime_dir)
        _add_dll_directory(runtime_dir)


def _configure_hikvision_mvs() -> None:
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    if _configure_bundled_mvs(base):
        return
    _configure_installed_mvs()


_configure_hikvision_mvs()
