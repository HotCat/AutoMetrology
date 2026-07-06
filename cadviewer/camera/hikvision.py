"""
Qt-aware Hikvision/Hikrobot camera backend using the MVS SDK.

The class mirrors the MindVisionCamera API so the registration UI can use
either backend without knowing SDK-specific details.
"""

from __future__ import annotations

import os
import sys
from ctypes import POINTER, byref, c_bool, c_ubyte, cast, memset, sizeof
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from .base import CameraSettingRanges, CameraSettings, CameraSignalEmitter


MVS_ROOT = os.environ.get("HIKVISION_MVS_ROOT", "/opt/MVS")
MVS_RUNENV = os.environ.get("MVCAM_COMMON_RUNENV") or os.path.join(MVS_ROOT, "lib")
MVS_IMPORT_DIR = os.path.join(MVS_ROOT, "Samples", "64", "Python", "MvImport")

os.environ.setdefault("MVCAM_COMMON_RUNENV", MVS_RUNENV)
if MVS_IMPORT_DIR not in sys.path:
    sys.path.insert(0, MVS_IMPORT_DIR)

try:
    from MvCameraControl_class import (  # type: ignore
        MV_ACCESS_Exclusive,
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT_INFO_EX,
        MV_GENTL_CAMERALINK_DEVICE,
        MV_GENTL_CXP_DEVICE,
        MV_GENTL_GIGE_DEVICE,
        MV_GENTL_XOF_DEVICE,
        MV_GIGE_DEVICE,
        MV_OK,
        MV_TRIGGER_MODE_OFF,
        MV_TRIGGER_MODE_ON,
        MV_TRIGGER_SOURCE_SOFTWARE,
        MV_USB_DEVICE,
        MVCC_FLOATVALUE,
        MVCC_INTVALUE_EX,
        MvCamera,
    )
except Exception as exc:  # pragma: no cover - depends on local SDK install
    raise ImportError(f"Hikvision MVS SDK import failed: {exc}") from exc


_SDK_INITIALIZED = False


def _ensure_sdk_initialized() -> None:
    global _SDK_INITIALIZED
    if not _SDK_INITIALIZED:
        ret = MvCamera.MV_CC_Initialize()
        if ret != MV_OK:
            raise RuntimeError(f"Hikvision SDK initialize failed: 0x{ret:x}")
        _SDK_INITIALIZED = True


def _decode_ctypes_string(value) -> str:
    data = memoryview(value).tobytes()
    data = data.split(b"\x00", 1)[0]
    for encoding in ("utf-8", "gbk", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _check(ret: int, action: str) -> None:
    if ret != MV_OK:
        raise RuntimeError(f"{action} failed: 0x{ret:x}")


def _is_gige(tlayer_type: int) -> bool:
    return tlayer_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE)


def _device_name_and_sn(info: MV_CC_DEVICE_INFO) -> tuple[str, str]:
    tlayer_type = int(info.nTLayerType)
    special = info.SpecialInfo
    if tlayer_type in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
        model = _decode_ctypes_string(special.stGigEInfo.chModelName)
        sn = _decode_ctypes_string(special.stGigEInfo.chSerialNumber)
    elif tlayer_type == MV_USB_DEVICE:
        model = _decode_ctypes_string(special.stUsb3VInfo.chModelName)
        sn = _decode_ctypes_string(special.stUsb3VInfo.chSerialNumber)
    elif tlayer_type == MV_GENTL_CAMERALINK_DEVICE:
        model = _decode_ctypes_string(special.stCMLInfo.chModelName)
        sn = _decode_ctypes_string(special.stCMLInfo.chSerialNumber)
    elif tlayer_type == MV_GENTL_CXP_DEVICE:
        model = _decode_ctypes_string(special.stCXPInfo.chModelName)
        sn = _decode_ctypes_string(special.stCXPInfo.chSerialNumber)
    elif tlayer_type == MV_GENTL_XOF_DEVICE:
        model = _decode_ctypes_string(special.stXoFInfo.chModelName)
        sn = _decode_ctypes_string(special.stXoFInfo.chSerialNumber)
    else:
        model = f"Transport 0x{tlayer_type:x}"
        sn = ""
    return model.strip() or "Hikvision Camera", sn.strip()


class _HikvisionLiveViewThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, camera: "HikvisionCamera"):
        super().__init__()
        self._camera = camera
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running and not self.isInterruptionRequested():
            try:
                frame = self._camera._grab_frame(timeout_ms=200)
            except RuntimeError as exc:
                if self._running:
                    self._camera.signals.error.emit(f"Live view error: {exc}")
                continue
            if frame is not None:
                self.frame_ready.emit(frame)

    def stop(self) -> None:
        self._running = False
        self.requestInterruption()
        self.wait(3000)


class HikvisionCamera:
    """High-level camera abstraction wrapping the Hikvision MVS SDK."""

    def __init__(self):
        _ensure_sdk_initialized()
        self._cam: Optional[MvCamera] = None
        self._dev_info: Optional[MV_CC_DEVICE_INFO] = None
        self._signals = CameraSignalEmitter()
        self._live_thread: Optional[_HikvisionLiveViewThread] = None
        self._width = 0
        self._height = 0
        self._mode = "closed"
        self._grabbing = False

    @property
    def signals(self) -> CameraSignalEmitter:
        return self._signals

    @property
    def is_open(self) -> bool:
        return self._cam is not None

    @property
    def resolution(self) -> tuple[int, int]:
        return self._width, self._height

    def enumerate_devices(self) -> list[dict]:
        _ensure_sdk_initialized()
        device_list = MV_CC_DEVICE_INFO_LIST()
        tlayer_type = (
            MV_GIGE_DEVICE
            | MV_USB_DEVICE
            | MV_GENTL_GIGE_DEVICE
            | MV_GENTL_CAMERALINK_DEVICE
            | MV_GENTL_CXP_DEVICE
            | MV_GENTL_XOF_DEVICE
        )
        ret = MvCamera.MV_CC_EnumDevices(tlayer_type, device_list)
        if ret != MV_OK:
            return []

        devices: list[dict] = []
        for idx in range(int(device_list.nDeviceNum)):
            info = cast(
                device_list.pDeviceInfo[idx],
                POINTER(MV_CC_DEVICE_INFO),
            ).contents
            model, sn = _device_name_and_sn(info)
            label = f"Hikvision {model}"
            if sn:
                label += f" ({sn})"
            devices.append({
                "name": label,
                "sn": sn,
                "backend": "hikvision",
                "dev_info": info,
            })
        return devices

    def open(self, dev_info) -> None:
        if self._cam is not None:
            self.close()

        self._cam = MvCamera()
        self._dev_info = dev_info
        try:
            _check(self._cam.MV_CC_CreateHandle(dev_info), "create camera handle")
            _check(self._cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "open camera")

            if _is_gige(int(dev_info.nTLayerType)):
                packet_size = self._cam.MV_CC_GetOptimalPacketSize()
                if int(packet_size) > 0:
                    self._try_set_int("GevSCPSPacketSize", int(packet_size))

            self._try_set_enum_string("AcquisitionMode", "Continuous")
            self._try_set_enum("TriggerMode", MV_TRIGGER_MODE_OFF)
            self._try_set_bool("AcquisitionFrameRateEnable", False)
            self._width = max(self._get_int("Width", 0), 0)
            self._height = max(self._get_int("Height", 0), 0)
            self.apply_settings(CameraSettings())
            self._mode = "open"
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._stop_worker()
        if self._cam is not None:
            if self._grabbing:
                self._try_stop_grabbing()
            try:
                self._cam.MV_CC_CloseDevice()
            except Exception:
                pass
            try:
                self._cam.MV_CC_DestroyHandle()
            except Exception:
                pass
        self._cam = None
        self._dev_info = None
        self._grabbing = False
        self._mode = "closed"

    def set_live_mode(self) -> None:
        if self._cam is None:
            return
        self._stop_worker()
        if self._grabbing:
            self._try_stop_grabbing()
        self._try_set_enum("TriggerMode", MV_TRIGGER_MODE_OFF)
        _check(self._cam.MV_CC_StartGrabbing(), "start live grabbing")
        self._grabbing = True
        self._start_worker()
        self._mode = "live"

    def set_trigger_mode(self) -> None:
        if self._cam is None:
            return
        self._stop_worker()
        if self._grabbing:
            self._try_stop_grabbing()
        self._try_set_enum("TriggerMode", MV_TRIGGER_MODE_ON)
        self._try_set_enum("TriggerSource", MV_TRIGGER_SOURCE_SOFTWARE)
        _check(self._cam.MV_CC_StartGrabbing(), "start trigger grabbing")
        self._grabbing = True
        self._try_clear_buffer()
        self._mode = "trigger"

    def software_trigger(self) -> None:
        if self._cam is None or self._mode != "trigger":
            self._signals.error.emit("Camera not in trigger mode")
            return
        try:
            self._try_clear_buffer()
            _check(self._cam.MV_CC_SetCommandValue("TriggerSoftware"), "software trigger")
            frame = self._grab_frame(timeout_ms=2000)
            if frame is None:
                self._signals.error.emit("Software trigger: no frame received")
            else:
                self._signals.grab_done.emit(frame)
        except RuntimeError as exc:
            self._signals.error.emit(str(exc))

    def get_setting_ranges(self) -> CameraSettingRanges:
        return CameraSettingRanges(
            exposure_min_us=int(self._get_float_min("ExposureTime", 100)),
            exposure_max_us=int(self._get_float_max("ExposureTime", 1000000)),
            exposure_step_us=1,
            gamma_min=int(self._get_float_min("Gamma", 1)),
            gamma_max=int(self._get_float_max("Gamma", 500)),
            contrast_min=1,
            contrast_max=500,
            analog_gain_min=int(self._get_float_min("Gain", 0)),
            analog_gain_max=int(self._get_float_max("Gain", 100)),
        )

    def apply_settings(self, settings: CameraSettings) -> None:
        if self._cam is None:
            return
        self._try_set_enum_string(
            "ExposureAuto",
            "Continuous" if settings.ae_enabled else "Off",
        )
        if not settings.ae_enabled:
            self._try_set_float("ExposureTime", float(settings.exposure_us))
        self._try_set_float("Gain", float(settings.analog_gain))
        self._try_set_float("Gamma", float(settings.gamma))
        self._try_set_float("Contrast", float(settings.contrast))
        self._try_set_bool("ReverseX", bool(settings.reverse_x))
        self._try_set_bool("ReverseY", bool(settings.reverse_y))

    def get_current_settings(self) -> CameraSettings:
        if self._cam is None:
            return CameraSettings()
        return CameraSettings(
            exposure_us=int(self._get_float("ExposureTime", 30000)),
            gamma=int(self._get_float("Gamma", 100)),
            contrast=int(self._get_float("Contrast", 100)),
            analog_gain=int(self._get_float("Gain", 16)),
            ae_enabled=False,
            reverse_x=self._get_bool("ReverseX", False),
            reverse_y=self._get_bool("ReverseY", False),
        )

    def _grab_frame(self, timeout_ms: int = 200) -> Optional[np.ndarray]:
        if self._cam is None or not self._grabbing:
            return None

        width = self._width or self._get_int("Width", 0)
        height = self._height or self._get_int("Height", 0)
        if width <= 0 or height <= 0:
            raise RuntimeError("Camera width/height unavailable")

        buffer_size = int(width * height * 3)
        output = (c_ubyte * buffer_size)()
        frame_info = MV_FRAME_OUT_INFO_EX()
        memset(byref(frame_info), 0, sizeof(frame_info))
        ret = self._cam.MV_CC_GetImageForBGR(output, buffer_size, frame_info, timeout_ms)
        if ret != MV_OK:
            return None

        frame_width = int(frame_info.nWidth) or width
        frame_height = int(frame_info.nHeight) or height
        byte_count = frame_width * frame_height * 3
        frame = np.frombuffer(output, dtype=np.uint8, count=byte_count)
        return frame.reshape((frame_height, frame_width, 3)).copy()

    def _start_worker(self) -> None:
        self._live_thread = _HikvisionLiveViewThread(self)
        self._live_thread.frame_ready.connect(self._signals.frame_ready)
        self._live_thread.start()

    def _stop_worker(self) -> None:
        if self._live_thread is not None:
            self._live_thread.stop()
            self._live_thread = None

    def _try_stop_grabbing(self) -> None:
        if self._cam is None:
            return
        try:
            self._cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        self._grabbing = False

    def _try_clear_buffer(self) -> None:
        if self._cam is None:
            return
        try:
            self._cam.MV_CC_ClearImageBuffer()
        except Exception:
            pass

    def _try_set_int(self, key: str, value: int) -> bool:
        if self._cam is None:
            return False
        try:
            return self._cam.MV_CC_SetIntValue(key, int(value)) == MV_OK
        except Exception:
            return False

    def _try_set_float(self, key: str, value: float) -> bool:
        if self._cam is None:
            return False
        try:
            return self._cam.MV_CC_SetFloatValue(key, float(value)) == MV_OK
        except Exception:
            return False

    def _try_set_enum(self, key: str, value: int) -> bool:
        if self._cam is None:
            return False
        try:
            return self._cam.MV_CC_SetEnumValue(key, int(value)) == MV_OK
        except Exception:
            return False

    def _try_set_enum_string(self, key: str, value: str) -> bool:
        if self._cam is None:
            return False
        try:
            return self._cam.MV_CC_SetEnumValueByString(key, value) == MV_OK
        except Exception:
            return False

    def _try_set_bool(self, key: str, value: bool) -> bool:
        if self._cam is None:
            return False
        try:
            return self._cam.MV_CC_SetBoolValue(key, bool(value)) == MV_OK
        except Exception:
            return False

    def _get_int(self, key: str, default: int) -> int:
        if self._cam is None:
            return default
        try:
            value = MVCC_INTVALUE_EX()
            memset(byref(value), 0, sizeof(value))
            if self._cam.MV_CC_GetIntValueEx(key, value) == MV_OK:
                return int(value.nCurValue)
        except Exception:
            pass
        return default

    def _get_float(self, key: str, default: float) -> float:
        if self._cam is None:
            return default
        try:
            value = MVCC_FLOATVALUE()
            memset(byref(value), 0, sizeof(value))
            if self._cam.MV_CC_GetFloatValue(key, value) == MV_OK:
                return float(value.fCurValue)
        except Exception:
            pass
        return default

    def _get_float_min(self, key: str, default: float) -> float:
        if self._cam is None:
            return default
        try:
            value = MVCC_FLOATVALUE()
            memset(byref(value), 0, sizeof(value))
            if self._cam.MV_CC_GetFloatValue(key, value) == MV_OK:
                return float(value.fMin)
        except Exception:
            pass
        return default

    def _get_float_max(self, key: str, default: float) -> float:
        if self._cam is None:
            return default
        try:
            value = MVCC_FLOATVALUE()
            memset(byref(value), 0, sizeof(value))
            if self._cam.MV_CC_GetFloatValue(key, value) == MV_OK:
                return float(value.fMax)
        except Exception:
            pass
        return default

    def _get_bool(self, key: str, default: bool) -> bool:
        if self._cam is None:
            return default
        try:
            value = c_bool(default)
            if self._cam.MV_CC_GetBoolValue(key, value) == MV_OK:
                return bool(value.value)
        except Exception:
            pass
        return default
