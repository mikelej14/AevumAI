@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo ============================================================
echo Aevum AI 0.2.4 - sharded self-test
echo ============================================================
%PY% tools\setup_probe.py --require-runtime-deps
if errorlevel 1 goto :fail

echo.
echo [1/3] Fast runtime / context / minimap contracts...
%PY% -m pytest -q tests\test_chat_store.py tests\test_local_context_runtime.py tests\test_neural_minimap.py tests\test_openai_runtime_contract.py tests\test_runtime_adapter_neural_activity.py tests\test_semantic_state.py
if errorlevel 1 goto :fail

echo.
echo [2/3] Real neural memory roundtrip...
%PY% -m pytest -q tests\test_product_memory_runtime.py::test_timestamped_chat_memory_roundtrip
if errorlevel 1 goto :fail

echo.
echo [3/3] Full multi-chunk document recall...
%PY% -m pytest -q tests\test_product_memory_runtime.py::test_explicit_recall_reassembles_complete_multichunk_document
if errorlevel 1 goto :fail

echo.
echo All Aevum AI 0.2.4 neural-memory tests passed.
pause
exit /b 0

:fail
echo.
echo Aevum AI 0.2.4 self-test FAILED.
pause
exit /b 1
