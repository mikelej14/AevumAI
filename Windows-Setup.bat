@echo off
setlocal EnableExtensions
cd /d "%~dp0"
echo ============================================================
echo Aevum AI 0.2.4 - Windows Vulkan setup
echo The Executive GGUF may use GPU offload; the optional semantic Qwen stays CPU-oriented.
echo GGUF files do NOT need to be in this folder.
echo ============================================================
echo.
where python >nul 2>nul || (echo ERROR: Python was not found on PATH. Install 64-bit Python 3.10 or newer. Python 3.11 or 3.12 is recommended for the widest prebuilt-wheel coverage.& pause & exit /b 1)
python --version
echo Verifying Python/Tkinter...
python -c "import tkinter; print('Python runtime OK'); print('Tip: Python 3.11/3.12 usually has the widest llama-cpp prebuilt-wheel coverage.')" || goto :venv_fail
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :venv_fail
)
call .venv\Scripts\activate.bat || goto :venv_fail
python -m pip install --upgrade pip setuptools wheel || goto :pip_fail

echo.
echo [1/2] Trying the official prebuilt llama-cpp-python Vulkan wheel...
python -m pip install --upgrade --force-reinstall --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan llama-cpp-python==0.3.35
if not errorlevel 1 goto :verify

echo.
echo No compatible prebuilt Vulkan wheel was found for this Python environment.
echo Falling back to a local Vulkan source build using a SHORT Windows temp path.
echo This avoids the MAX_PATH extraction failure caused by the normal user TEMP path.
echo.
set "AEVUM_SHORT_TEMP=%SystemDrive%\AEVUM_BUILD"
if "%SystemDrive%"=="" set "AEVUM_SHORT_TEMP=C:\AEVUM_BUILD"
if exist "%AEVUM_SHORT_TEMP%" rmdir /s /q "%AEVUM_SHORT_TEMP%" >nul 2>nul
mkdir "%AEVUM_SHORT_TEMP%" || goto :temp_fail
set "TEMP=%AEVUM_SHORT_TEMP%"
set "TMP=%AEVUM_SHORT_TEMP%"
set "TMPDIR=%AEVUM_SHORT_TEMP%"
set "PIP_NO_CACHE_DIR=1"
set "CMAKE_ARGS=-DGGML_NATIVE=off -DGGML_VULKAN=on"
set "FORCE_CMAKE=1"
python -m pip install --upgrade cmake ninja build || goto :source_fail
python -m pip install --upgrade --force-reinstall --no-binary=llama-cpp-python llama-cpp-python==0.3.35
if errorlevel 1 goto :source_fail

goto :verify

:verify
echo.
echo Verifying llama-cpp-python import...
python -c "import llama_cpp; print('llama-cpp-python', getattr(llama_cpp, '__version__', 'unknown'), 'loaded successfully')"
if errorlevel 1 goto :verify_fail
if defined AEVUM_SHORT_TEMP if exist "%AEVUM_SHORT_TEMP%" rmdir /s /q "%AEVUM_SHORT_TEMP%" >nul 2>nul
echo Installing Aevum support packages...
python -m pip install -r requirements.txt || goto :pip_fail
python tools\setup_probe.py --require-runtime-deps || goto :verify_fail
echo.
echo ============================================================
echo Setup complete.
echo Run RUN.bat, open Settings, select the Granite/Executive GGUF and optionally a semantic Qwen GGUF,
echo then click Start model.
echo ============================================================
pause
exit /b 0

:venv_fail
echo.
echo ERROR: Could not create or activate .venv.
goto :fail

:pip_fail
echo.
echo ERROR: Python packaging tools could not be installed.
goto :fail

:temp_fail
echo.
echo ERROR: Could not create short build directory "%AEVUM_SHORT_TEMP%".
goto :fail

:source_fail
echo.
echo ERROR: The official Vulkan wheel was unavailable AND the fallback source build failed.
echo The original Windows path-length problem has been bypassed by using "%AEVUM_SHORT_TEMP%".
echo If the error above now mentions Vulkan, CMake, MSVC, cl.exe, or a compiler,
echo install the Vulkan SDK and Microsoft C++ Build Tools, then rerun Windows-Setup.bat.
goto :fail

:verify_fail
echo.
echo ERROR: Runtime verification failed. Review the message above (llama-cpp-python, OpenAI client, assets, or config).
goto :fail

:fail
if defined AEVUM_SHORT_TEMP if exist "%AEVUM_SHORT_TEMP%" rmdir /s /q "%AEVUM_SHORT_TEMP%" >nul 2>nul
echo.
echo Setup did not complete. The error immediately above is the relevant one.
pause
exit /b 1
