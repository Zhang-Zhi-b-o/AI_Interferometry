"""Narrow Interception IOCTL adapter, opening only the C216:010B keyboard.

Protocol: oblitum/Interception library/interception.c and interception.h, v1.0.1.
Unlike the upstream DLL, this adapter does not require a mouse driver.
"""
import ctypes as c
from ctypes import wintypes as w

TARGET = 'VID_C216&PID_010B'


class RawKey(c.Structure):
    _fields_ = [('unit', w.USHORT), ('code', w.USHORT), ('state', w.USHORT),
                ('reserved', w.USHORT), ('information', w.ULONG)]


class Device:
    def __init__(self):
        self.k = c.WinDLL('kernel32', use_last_error=True)
        self.k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
        self.k.CreateFileW.restype = w.HANDLE
        self.k.CloseHandle.argtypes = [w.HANDLE]
        self.k.CreateEventW.argtypes = [c.c_void_p, w.BOOL, w.BOOL, w.LPCWSTR]
        self.k.CreateEventW.restype = w.HANDLE
        self.k.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        self.k.DeviceIoControl.argtypes = [w.HANDLE, w.DWORD, c.c_void_p, w.DWORD,
                                           c.c_void_p, w.DWORD, c.POINTER(w.DWORD), c.c_void_p]
        self.handle = self.event = None
        self.hardware_id = None

    def ioctl(self, function, source=None, destination=None):
        count = w.DWORD()
        ok = self.k.DeviceIoControl(self.handle, (0x22 << 16) | (function << 2),
                                   c.byref(source) if source is not None else None,
                                   c.sizeof(source) if source is not None else 0,
                                   c.byref(destination) if destination is not None else None,
                                   c.sizeof(destination) if destination is not None else 0,
                                   c.byref(count), None)
        if not ok:
            raise c.WinError(c.get_last_error())
        return count.value

    def open(self):
        matches = []
        try:
            for index in range(10):
                handle = self.k.CreateFileW(r'\\.\interception%02d' % index,
                                             0x80000000, 0, None, 3, 0, None)
                if handle == c.c_void_p(-1).value:
                    continue
                self.handle = handle
                buf = c.create_unicode_buffer(512)
                try:
                    try:
                        self.ioctl(0x880, destination=buf)
                    except OSError as exc:
                        # Driver control slots exist even when no physical device owns one.
                        if exc.winerror in (1, 2, 1167):
                            continue
                        raise
                    if TARGET in buf.value.upper():
                        matches.append((handle, buf.value))
                        handle = None
                finally:
                    if handle is not None:
                        self.k.CloseHandle(handle)
                    self.handle = None
            if len(matches) != 1:
                raise RuntimeError('Expected one configured C216:010B device; found %d. '
                                   'Connect it and run device-only setup first.' % len(matches))
            self.handle, self.hardware_id = matches.pop()
            self.event = self.k.CreateEventW(None, True, False, None)
            if not self.event:
                raise c.WinError(c.get_last_error())
            events = (w.HANDLE * 2)(self.event, None)
            self.ioctl(0x810, source=events)
            self.ioctl(0x804, source=w.USHORT(0xFFFF))
            # The signed v1.0.1 driver implements SET_FILTER but not GET_FILTER.
            # Unknown GET_FILTER completes with zero bytes; it is not a status test.
        except BaseException:
            for handle, _ in matches:
                self.k.CloseHandle(handle)
            self.close()
            raise

    def read(self, timeout=100):
        result = self.k.WaitForSingleObject(self.event, timeout)
        if result == 258:
            return None
        if result != 0:
            raise RuntimeError('Device wait failed')
        key = RawKey()
        size = self.ioctl(0x840, destination=key)
        if not size:
            return None
        if size != c.sizeof(key):
            raise RuntimeError('Unexpected keyboard packet size')
        return key.code, key.state

    def close(self):
        if self.handle:
            try:
                self.ioctl(0x804, source=w.USHORT(0))
            finally:
                self.k.CloseHandle(self.handle)
                self.handle = None
                if self.event:
                    self.k.CloseHandle(self.event)
                    self.event = None
