"""User-invoked removal of the micrometer-only input filter."""
import ctypes
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
script = root / "engine" / "hardware" / "signal_setup.py"
arguments = ["-X", "utf8", str(script), "--remove", "--log"]
if ctypes.windll.shell32.IsUserAnAdmin():
    raise SystemExit(subprocess.call([sys.executable, *arguments]))
exe = "'" + sys.executable.replace("'", "''") + "'"
quoted = "'" + subprocess.list2cmdline(arguments).replace("'", "''") + "'"
command = ("$ErrorActionPreference='Stop'; $p=Start-Process -FilePath " + exe +
           " -ArgumentList " + quoted + " -Verb RunAs -WindowStyle Hidden -Wait -PassThru; exit $p.ExitCode")
raise SystemExit(subprocess.call(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command]))
