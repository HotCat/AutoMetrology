"""
Optional industrial camera integration.

The UI talks to a small camera API shared by the available SDK backends.  The
Hikvision/Hikrobot MVS backend is preferred when installed; MindVision remains
available as a fallback for existing machines.
"""

from __future__ import annotations

from .base import CameraSettings, CameraSettingRanges

try:
    from .device import MindVisionCamera
except Exception:
    MindVisionCamera = None

try:
    from .hikvision import HikvisionCamera
except Exception:
    HikvisionCamera = None

CameraClass = HikvisionCamera or MindVisionCamera
HAS_CAMERA = CameraClass is not None
