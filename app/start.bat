@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto run
call setup.bat
if errorlevel 1 exit /b 1
:run
".venv\Scripts\python.exe" run.py %*
if errorlevel 1 (
  echo Startup failed. Run setup.bat to install or update dependencies.
  pause
  exit /b 1
)
