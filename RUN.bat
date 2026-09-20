@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
  start "" /b ".venv\Scripts\pythonw.exe" app.py
) else (
  start "" /b pythonw.exe app.py
)
exit /b 0
