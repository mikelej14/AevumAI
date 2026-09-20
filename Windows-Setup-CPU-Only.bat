@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo Aevum AI 0.2.4 - CPU-only diagnostic setup
echo ============================================================
where python >nul 2>nul || (echo ERROR: Python was not found on PATH.& pause & exit /b 1)
python --version
echo Verifying Python/Tkinter...
python -c "import tkinter; print('Python runtime OK'); print('Tip: Python 3.11/3.12 usually has the widest llama-cpp prebuilt-wheel coverage.')" || goto :fail
if not exist .venv python -m venv .venv || goto :fail
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip setuptools wheel || goto :fail
echo Installing official prebuilt CPU wheel...
python -m pip install --upgrade --force-reinstall --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu llama-cpp-python==0.3.35
if errorlevel 1 goto :fail
python -c "import llama_cpp; print('llama-cpp-python', getattr(llama_cpp, '__version__', 'unknown'), 'loaded successfully')" || goto :fail
python -m pip install -r requirements.txt || goto :fail
python tools\setup_probe.py --require-runtime-deps || goto :fail
echo.
echo Setup complete. This environment is CPU-only. Configure the Executive and optional semantic annotator in Settings.
pause
exit /b 0
:fail
echo.
echo CPU-only setup failed. Review the error immediately above.
pause
exit /b 1
