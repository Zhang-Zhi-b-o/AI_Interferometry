@echo off
setlocal
cd /d "%~dp0"
if exist "..\runtime\python.exe" (
  "..\runtime\python.exe" -X utf8 distribution\remove_filter.py
) else (
  if not exist ".venv\Scripts\python.exe" call setup.bat
  ".venv\Scripts\python.exe" -X utf8 distribution\remove_filter.py
)
echo Check logs\signal-setup.log for the result. Replug the micrometer or restart Windows if required.
pause
