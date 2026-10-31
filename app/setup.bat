@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3.11 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
  py -3.11 -m venv .venv
  goto check
)
py -3.10 -c "import sys; assert sys.maxsize > 2**32" >nul 2>&1
if not errorlevel 1 (
  py -3.10 -m venv .venv
  goto check
)
python -c "import sys; assert sys.version_info[:2] in ((3,10),(3,11)) and sys.maxsize > 2**32" >nul 2>&1
if errorlevel 1 (
  echo Please install 64-bit Python 3.10 or 3.11 from https://www.python.org/downloads/windows/
  goto fail
)
python -m venv .venv
:check
if not exist ".venv\Scripts\python.exe" goto fail
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 goto fail
echo Installation complete. Run start.bat or start.bat --demo.
exit /b 0
:fail
echo Installation failed. Check the message above, then run setup.bat again.
pause
exit /b 1
