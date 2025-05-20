@echo off
setlocal enabledelayedexpansion

REM Set environment variables
set PYTHONHASHSEED=123
set PYTHONPATH=%~dp0

REM Display startup information
echo ================================================
echo Starting Seed Analysis System
echo ================================================
echo Working directory: %~dp0
echo PYTHONHASHSEED: %PYTHONHASHSEED%
echo PYTHONPATH: %PYTHONPATH%
echo ================================================

REM Activate conda environment
echo Activating conda environment...
call conda activate seed-parse
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate conda environment.
    echo Please make sure conda is installed and the seed-parse environment exists.
    pause
    exit /b 1
)

REM Set process priority
echo Setting process priority...
wmic process where name="python.exe" CALL setpriority 128

REM Run the project
echo Running project...
python %~dp0run_project.py

if %ERRORLEVEL% NEQ 0 (
    echo Error running the project. See error message above.
)

pause
