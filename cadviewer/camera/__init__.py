"""
Camera module — optional MindVision industrial camera integration.

This module vendors the MindVision MVCAM SDK ctypes wrapper and provides
Qt-aware camera abstraction for live preview and frame capture.

If the SDK (libMVSDK.so) is not installed or import fails, HAS_CAMERA is False
and all camera classes are None — the application works fine without a camera.
"""

from __future__ import annotations

# Optional-import gate — SDK may not be installed
try:
    from .device import MindVisionCamera, CameraSettings, CameraSettingRanges
    HAS_CAMERA = True
except ImportError:
    HAS_CAMERA = False
    MindVisionCamera = None
    CameraSettings = None
    CameraSettingRanges = None