"""RS232 light-controller protocol used by AutoMetrology.

The controller protocol was first validated by ``tools/explore_light_controller.py``.
Frames are ASCII:

    "$" + command + channel + data3 + checksum2

Commands:
    1: open channel output
    2: close channel output
    3: set channel brightness
    4: read channel brightness

The checksum is XOR of the six bytes before the checksum, encoded as two
lowercase hexadecimal ASCII characters.
"""

from __future__ import annotations

import os
import select
import sys
import time
from dataclasses import dataclass


BAUD_RATES: dict[int, int] = {}
if os.name != "nt":
    import fcntl
    import termios

    BAUD_RATES = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
    }
else:  # pragma: no cover - exercised on Windows only
    fcntl = None
    termios = None

try:  # Optional Windows-friendly backend.
    import serial  # type: ignore
except Exception:  # pragma: no cover - pyserial is optional
    serial = None


@dataclass
class ReadResult:
    channel: int
    brightness: int
    raw: str
    valid_checksum: bool


def checksum(payload: str) -> str:
    value = 0
    for byte in payload.encode("ascii"):
        value ^= byte
    return f"{value:02x}"


def verify_checksum(frame: str) -> bool:
    if len(frame) < 8:
        return False
    return checksum(frame[:6]).lower() == frame[6:8].lower()


class LightController:
    """Small synchronous RS232 client for the four-channel light controller."""

    def __init__(
        self,
        device: str,
        baud: int = 9600,
        timeout_s: float = 0.7,
        suffix: bytes = b"",
        rts: bool | None = None,
        dtr: bool | None = None,
    ):
        self.device = device
        self.baud = int(baud)
        self.timeout_s = float(timeout_s)
        self.suffix = suffix
        self.rts = rts
        self.dtr = dtr
        self.fd: int | None = None
        self._serial = None

    def __enter__(self) -> "LightController":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self.fd is not None or self._serial is not None

    def open(self) -> None:
        if self.is_open:
            return
        if os.name != "nt":
            self._open_posix()
            return
        if serial is None:
            raise RuntimeError("pyserial is required for light control on Windows")
        self._open_pyserial()

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def frame(self, command: int, channel: int, value: int = 0x64) -> str:
        if command not in (1, 2, 3, 4):
            raise ValueError("command must be 1, 2, 3, or 4")
        if channel not in (1, 2, 3, 4):
            raise ValueError("channel must be 1, 2, 3, or 4")
        if not 0 <= int(value) <= 0xFF:
            raise ValueError("value must be in 0..255")
        payload = f"${command}{channel}0{int(value):02X}"
        return payload + checksum(payload)

    def transact(
        self,
        command: int,
        channel: int,
        value: int = 0x64,
        expected: int | None = None,
    ) -> str:
        if not self.is_open:
            raise RuntimeError("Serial device is not open")
        frame = self.frame(command, channel, value)
        if expected is None:
            expected = 8 if command == 4 else 1
        if self._serial is not None:
            return self._transact_pyserial(frame, expected)
        return self._transact_posix(frame, expected)

    def read_brightness(self, channel: int) -> ReadResult:
        raw = self.transact(4, channel, 0x64, expected=8)
        if raw == "&":
            raise RuntimeError(f"Controller rejected read command for CH{channel}")
        if len(raw) < 8 or raw[0] != "$" or raw[1] != "4":
            raise RuntimeError(f"Unexpected read response for CH{channel}: {raw!r}")
        return ReadResult(
            channel=int(raw[2]),
            brightness=int(raw[4:6], 16),
            raw=raw,
            valid_checksum=verify_checksum(raw),
        )

    def set_brightness(self, channel: int, brightness: int) -> str:
        return self._ack_command(3, channel, brightness, "set brightness")

    def open_channel(self, channel: int) -> str:
        return self._ack_command(1, channel, 0x64, "open channel")

    def close_channel(self, channel: int) -> str:
        return self._ack_command(2, channel, 0x64, "close channel")

    def _ack_command(self, command: int, channel: int, value: int, label: str) -> str:
        raw = self.transact(command, channel, value, expected=1)
        if raw == "$":
            return raw
        if raw == "&":
            raise RuntimeError(f"Controller rejected {label} CH{channel}")
        raise RuntimeError(f"No valid ACK for {label} CH{channel}: {raw!r}")

    def _open_pyserial(self) -> None:
        self._serial = serial.Serial(  # type: ignore[union-attr]
            port=self.device,
            baudrate=self.baud,
            timeout=0,
            write_timeout=self.timeout_s,
        )
        if self.rts is not None:
            self._serial.rts = bool(self.rts)
        if self.dtr is not None:
            self._serial.dtr = bool(self.dtr)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def _open_posix(self) -> None:
        if self.baud not in BAUD_RATES:
            raise ValueError(f"Unsupported baud rate: {self.baud}")
        self.fd = os.open(self.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        attrs = termios.tcgetattr(self.fd)
        attrs[0] = 0
        attrs[1] = 0
        attrs[2] = BAUD_RATES[self.baud] | termios.CS8 | termios.CREAD | termios.CLOCAL
        attrs[3] = 0
        attrs[4] = BAUD_RATES[self.baud]
        attrs[5] = BAUD_RATES[self.baud]
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        self._set_modem_line("rts", self.rts)
        self._set_modem_line("dtr", self.dtr)
        termios.tcflush(self.fd, termios.TCIOFLUSH)

    def _transact_pyserial(self, frame: str, expected: int) -> str:
        self._serial.reset_input_buffer()
        self._serial.write(frame.encode("ascii") + self.suffix)
        self._serial.flush()
        return self._read_response_pyserial(expected)

    def _transact_posix(self, frame: str, expected: int) -> str:
        if self.fd is None:
            raise RuntimeError("Serial device is not open")
        termios.tcflush(self.fd, termios.TCIFLUSH)
        os.write(self.fd, frame.encode("ascii") + self.suffix)
        termios.tcdrain(self.fd)
        return self._read_response_posix(expected)

    def _read_response_pyserial(self, expected: int) -> str:
        deadline = time.monotonic() + self.timeout_s
        chunks: list[bytes] = []
        while time.monotonic() < deadline:
            data = self._serial.read(64)
            if data:
                chunks.append(data)
                joined = b"".join(chunks)
                if b"&" in joined:
                    return "&"
                if len(joined) >= expected:
                    return joined[:expected].decode("ascii", errors="replace")
            else:
                time.sleep(0.01)
        return b"".join(chunks).decode("ascii", errors="replace")

    def _read_response_posix(self, expected: int) -> str:
        if self.fd is None:
            raise RuntimeError("Serial device is not open")
        deadline = time.monotonic() + self.timeout_s
        chunks: list[bytes] = []
        while time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            rlist, _, _ = select.select([self.fd], [], [], min(0.05, remaining))
            if not rlist:
                continue
            try:
                data = os.read(self.fd, 64)
            except BlockingIOError:
                continue
            if data:
                chunks.append(data)
                joined = b"".join(chunks)
                if b"&" in joined:
                    return "&"
                if len(joined) >= expected:
                    return joined[:expected].decode("ascii", errors="replace")
        return b"".join(chunks).decode("ascii", errors="replace")

    def _set_modem_line(self, name: str, enabled: bool | None) -> None:
        if enabled is None or self.fd is None:
            return
        const_name = "TIOCM_RTS" if name == "rts" else "TIOCM_DTR"
        bit = getattr(termios, const_name, None)
        if bit is None:
            return
        request = termios.TIOCMBIS if enabled else termios.TIOCMBIC
        fcntl.ioctl(self.fd, request, int(bit).to_bytes(4, sys.byteorder))
