@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo DriftLens's Python environment is missing.
    echo Run the project setup first, then open this launcher again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8510 --browser.gatherUsageStats false
if errorlevel 1 pause
endlocal
