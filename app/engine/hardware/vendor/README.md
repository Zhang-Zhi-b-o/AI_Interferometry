keyboard.sys is the unmodified, signed Windows x64 keyboard filter extracted from
the KBDNT61A64 / DRIVER resource in the project's official Interception v1.0.1
install-interception.exe. Upstream: https://github.com/oblitum/Interception
License: LICENSE.txt (LGPL 3.0, non-commercial distribution).

signal_setup.py verifies its Windows Authenticode signature before restoring a
missing system copy. It attaches the filter only to the present C216:010B HID
keyboard instance, using the existing device setup implementation.
