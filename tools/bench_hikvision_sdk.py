from __future__ import annotations

import os
import sys
import time
from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof


os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
sys.path.insert(0, "/opt/MVS/Samples/64/Python/MvImport")

from MvCameraControl_class import *  # type: ignore  # noqa: F403


def _check(ret: int, what: str) -> None:
    if ret != MV_OK:  # noqa: F405
        raise RuntimeError(f"{what} failed: 0x{ret:x}")


def _open_first_camera():
    MvCamera.MV_CC_Initialize()  # noqa: F405
    device_list = MV_CC_DEVICE_INFO_LIST()  # noqa: F405
    tlayer_type = (
        MV_GIGE_DEVICE  # noqa: F405
        | MV_USB_DEVICE  # noqa: F405
        | MV_GENTL_GIGE_DEVICE  # noqa: F405
        | MV_GENTL_CAMERALINK_DEVICE  # noqa: F405
        | MV_GENTL_CXP_DEVICE  # noqa: F405
        | MV_GENTL_XOF_DEVICE  # noqa: F405
    )
    _check(MvCamera.MV_CC_EnumDevices(tlayer_type, device_list), "enum")  # noqa: F405
    if device_list.nDeviceNum <= 0:
        raise RuntimeError("no Hikvision camera found")
    info = cast(device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents  # noqa: F405
    cam = MvCamera()  # noqa: F405
    _check(cam.MV_CC_CreateHandle(info), "create handle")
    _check(cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0), "open")  # noqa: F405
    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)  # noqa: F405
    cam.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImagesOnly)  # noqa: F405
    width = MVCC_INTVALUE_EX()  # noqa: F405
    height = MVCC_INTVALUE_EX()  # noqa: F405
    cam.MV_CC_GetIntValueEx("Width", width)
    cam.MV_CC_GetIntValueEx("Height", height)
    print("resolution", int(width.nCurValue), int(height.nCurValue))
    return cam, int(width.nCurValue), int(height.nCurValue)


def _bench_raw(cam, seconds: float) -> None:
    _check(cam.MV_CC_StartGrabbing(), "start raw")
    frame_out = MV_FRAME_OUT()  # noqa: F405
    memset(byref(frame_out), 0, sizeof(frame_out))
    count = 0
    bytes_total = 0
    first = None
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        ret = cam.MV_CC_GetImageBuffer(frame_out, 1000)
        if ret != MV_OK:  # noqa: F405
            continue
        info = frame_out.stFrameInfo
        if first is None:
            first = (int(info.nWidth), int(info.nHeight), int(info.enPixelType), int(info.nFrameLen))
        bytes_total += int(info.nFrameLen)
        count += 1
        cam.MV_CC_FreeImageBuffer(frame_out)
    elapsed = time.perf_counter() - start
    cam.MV_CC_StopGrabbing()
    print("raw_getbuffer", {"fps": count / elapsed, "frames": count, "first": first, "MBps": bytes_total / 1024 / 1024 / elapsed})


def _bench_bgr(cam, width: int, height: int, seconds: float) -> None:
    _check(cam.MV_CC_StartGrabbing(), "start bgr")
    frame_info = MV_FRAME_OUT_INFO_EX()  # noqa: F405
    buffer_size = width * height * 3
    buffer = (c_ubyte * buffer_size)()
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        memset(byref(frame_info), 0, sizeof(frame_info))
        ret = cam.MV_CC_GetImageForBGR(buffer, buffer_size, frame_info, 1000)
        if ret != MV_OK:  # noqa: F405
            continue
        count += 1
    elapsed = time.perf_counter() - start
    cam.MV_CC_StopGrabbing()
    print("get_image_for_bgr", {"fps": count / elapsed, "frames": count})


def main() -> None:
    cam, width, height = _open_first_camera()
    try:
        _bench_raw(cam, 3.0)
        _bench_bgr(cam, width, height, 3.0)
    finally:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()


if __name__ == "__main__":
    main()
