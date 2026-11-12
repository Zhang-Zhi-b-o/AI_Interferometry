"""Configure only the micrometer's device filter; never install class-wide filters."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import json
from pathlib import Path
import subprocess
import uuid
import winreg
import time
import sys
import os
import shutil
from contextlib import contextmanager

TARGET = 'HID\\VID_C216&PID_010B\\'
CLASS = '{4d36e96b-e325-11ce-bfc1-08002be10318}'
ROOT = Path(__file__).resolve().parents[2] / 'logs'
DRIVER = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32/drivers/keyboard.sys'


class Guid(c.Structure):
    _fields_ = [('bytes', c.c_ubyte * 16)]


class Info(c.Structure):
    _fields_ = [('size', w.DWORD), ('guid', Guid), ('instance', w.DWORD), ('reserved', c.c_size_t)]


def api():
    s = c.WinDLL('setupapi', use_last_error=True)
    s.SetupDiGetClassDevsW.argtypes = [c.POINTER(Guid), w.LPCWSTR, w.HWND, w.DWORD]
    s.SetupDiGetClassDevsW.restype = w.HANDLE
    s.SetupDiEnumDeviceInfo.argtypes = [w.HANDLE, w.DWORD, c.POINTER(Info)]
    s.SetupDiGetDeviceInstanceIdW.argtypes = [w.HANDLE, c.POINTER(Info), w.LPWSTR, w.DWORD, c.POINTER(w.DWORD)]
    s.SetupDiGetDeviceRegistryPropertyW.argtypes = [w.HANDLE, c.POINTER(Info), w.DWORD, c.POINTER(w.DWORD), c.c_void_p, w.DWORD, c.POINTER(w.DWORD)]
    s.SetupDiSetDeviceRegistryPropertyW.argtypes = [w.HANDLE, c.POINTER(Info), w.DWORD, c.c_void_p, w.DWORD]
    s.SetupDiDestroyDeviceInfoList.argtypes = [w.HANDLE]
    return s


def class_filters(guid):
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 'SYSTEM\\CurrentControlSet\\Control\\Class\\' + guid) as key:
        return winreg.QueryValueEx(key, 'UpperFilters')[0]


def check_class_filters():
    for guid, unwanted in [(CLASS, 'keyboard'), ('{4d36e96f-e325-11ce-bfc1-08002be10318}', 'mouse')]:
        filters = class_filters(guid)
        print('CLASS FILTERS', guid, filters)
        if unwanted in [x.lower() for x in filters]:
            raise RuntimeError('Class-wide Interception is still installed; restore it before device-only setup')

@contextmanager
def target_devices():
    """Enumerate present matching keyboards; device instances may change with USB ports."""
    s = api()
    guid = Guid((c.c_ubyte * 16).from_buffer_copy(uuid.UUID(CLASS).bytes_le))
    devices = s.SetupDiGetClassDevsW(c.byref(guid), None, None, 2)
    if devices == c.c_void_p(-1).value:
        raise c.WinError(c.get_last_error())
    try:
        matches = []
        i = 0
        while True:
            info = Info(); info.size = c.sizeof(info)
            if not s.SetupDiEnumDeviceInfo(devices, i, c.byref(info)):
                if c.get_last_error() != 259:
                    raise c.WinError(c.get_last_error())
                break
            i += 1
            name = c.create_unicode_buffer(512); needed = w.DWORD()
            if not s.SetupDiGetDeviceInstanceIdW(devices, c.byref(info), name, 512, c.byref(needed)):
                raise c.WinError(c.get_last_error())
            if name.value.upper().startswith(TARGET):
                buf = c.create_unicode_buffer(1024); kind = w.DWORD()
                if s.SetupDiGetDeviceRegistryPropertyW(devices, c.byref(info), 0x11, c.byref(kind), buf, c.sizeof(buf), c.byref(needed)):
                    filters = c.string_at(c.addressof(buf), needed.value).decode('utf-16-le').strip('\0').split('\0')
                elif c.get_last_error() == 13:
                    filters = []
                else:
                    raise c.WinError(c.get_last_error())
                print('TARGET', name.value, 'FILTERS', filters)
                matches.append((info, name.value, filters))
        yield s, devices, matches
    finally:
        s.SetupDiDestroyDeviceInfoList(devices)


def configure(*, remove=False):
    check_class_filters()
    with target_devices() as (s, devices, matches):
        if len(matches) != 1:
            raise RuntimeError('Connect exactly one C216:010B micrometer before setup')
        if not c.windll.shell32.IsUserAnAdmin():
            raise RuntimeError('Administrator authorization required for device-only setup')
        info, instance, old = matches[0]
        restored = False
        if not remove:
            # The bundled signed x64 driver is extracted from official Interception v1.0.1.
            driver = DRIVER if DRIVER.is_file() else Path(__file__).parent / 'vendor/keyboard.sys'
            cmd = ["$s=Get-AuthenticodeSignature -LiteralPath '" + str(driver).replace("'", "''") + "'; if($s.Status -ne 'Valid'){exit 1}"]
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', cmd[0]], check=True, creationflags=0x08000000)
            if driver != DRIVER:
                shutil.copyfile(driver, DRIVER)
                restored = True
            query = subprocess.run(['sc.exe', 'query', 'keyboard'], capture_output=True)
            if query.returncode == 1060:
                subprocess.run(['sc.exe', 'create', 'keyboard', 'type=', 'kernel', 'start=', 'demand',
                                'binPath=', r'\SystemRoot\System32\drivers\keyboard.sys'], check=True)
            elif query.returncode:
                raise RuntimeError('Cannot inspect keyboard driver service')
            new = old if 'keyboard' in [x.lower() for x in old] else old + ['keyboard']
        else:
            new = [x for x in old if x.lower() != 'keyboard']
        if new == old and not restored:
            print('DEVICE FILTER ALREADY CONFIGURED')
            return
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / ('device-filter-backup-%d.json' % time.time_ns())).write_text(json.dumps({'instance': instance, 'before': old, 'after': new, 'configured_at': time.time()}, indent=2), encoding='utf-8')
        raw = c.create_unicode_buffer('\0'.join(new) + '\0') if new else None
        if not s.SetupDiSetDeviceRegistryPropertyW(devices, c.byref(info), 0x11, raw, c.sizeof(raw) if raw is not None else 0):
            raise c.WinError(c.get_last_error())
        # Restart only this HID keyboard device, never normal keyboards or mice.
        result = subprocess.run(['pnputil.exe', '/restart-device', instance], capture_output=True)
        print('MICROMETER RESTART RESULT', result.returncode)
        if result.returncode:
            # Restore this device's old filters if its restart fails.
            previous = c.create_unicode_buffer('\0'.join(old) + '\0') if old else None
            s.SetupDiSetDeviceRegistryPropertyW(devices, c.byref(info), 0x11, previous, c.sizeof(previous) if previous is not None else 0)
            raise RuntimeError('Micrometer restart failed; old device filters restored. Restart Windows manually if requested.')
        print('DEVICE-ONLY CONFIGURATION COMPLETE. Normal keyboard/mouse class filters unchanged.')


def ensure_configured():
    """Configure a newly discovered port on demand, using Windows UAC when needed."""
    check_class_filters()
    with target_devices() as (_, _, matches):
        if len(matches) != 1:
            raise RuntimeError('未找到微分头 C216:010B，请检查 USB 连接' if not matches else
                               '检测到多个 C216:010B 微分头，请只连接一个')
        if 'keyboard' in [x.lower() for x in matches[0][2]] and DRIVER.is_file():
            return
    if c.windll.shell32.IsUserAnAdmin():
        configure()
        return
    # Quote paths as PowerShell literals and pass one correctly quoted Windows argument string.
    exe = "'" + sys.executable.replace("'", "''") + "'"
    arguments = "'" + subprocess.list2cmdline([
        '-X', 'utf8', str(Path(__file__).resolve()), '--configure', '--log'
    ]).replace("'", "''") + "'"
    command = ("$ErrorActionPreference='Stop'; $p=Start-Process -FilePath " + exe +
               " -ArgumentList " + arguments +
               " -Verb RunAs -WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, creationflags=0x08000000)
    if result.returncode:
        raise RuntimeError('微分头驱动自动配置未完成（管理员确认被取消或配置失败），'
                           '请重试并允许 Windows 管理员确认；详情见 app/logs/signal-setup.log')
    with target_devices() as (_, _, matches):
        if len(matches) != 1 or 'keyboard' not in [x.lower() for x in matches[0][2]]:
            raise RuntimeError('微分头驱动配置后未就绪，请重新插拔微分头后重试')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--configure', action='store_true')
    mode.add_argument('--remove', action='store_true')
    p.add_argument('--log', action='store_true')
    args = p.parse_args()
    if args.log:
        ROOT.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = (ROOT / 'signal-setup.log').open('w', encoding='utf-8', buffering=1)
    if args.configure or args.remove:
        configure(remove=args.remove)
    else:
        check_class_filters()
        with target_devices() as (_, _, matches):
            print('PRESENT TARGET COUNT', len(matches))


if __name__ == '__main__':
    main()
