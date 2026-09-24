@echo off
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Python environment .venv was not found in this folder.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run "%~dp0src\app.py" %*
if errorlevel 1 pause
