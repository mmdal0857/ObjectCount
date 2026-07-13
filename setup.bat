@echo off
setlocal

cd /d "%~dp0"

set "EXTRAS=dev,capture"
set "RUN_TESTS=1"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--tools" (
  set "EXTRAS=dev,capture,tools"
  shift
  goto parse_args
)
if /I "%~1"=="--skip-tests" (
  set "RUN_TESTS=0"
  shift
  goto parse_args
)
if /I "%~1"=="--help" goto help
echo Unknown option: %~1
goto help

:args_done
echo [ObjectCount] Setting up Windows development environment
echo Project: %CD%
echo Extras:  %EXTRAS%

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo Creating .venv...
  py -3.11 -m venv .venv 2>nul
  if errorlevel 1 (
    py -m venv .venv 2>nul
  )
  if errorlevel 1 (
    python -m venv .venv
  )
  if errorlevel 1 (
    echo.
    echo ERROR: Python 3.11+ was not found. Install Python, then run setup.bat again.
    exit /b 1
  )
) else (
  echo.
  echo Reusing existing .venv
)

echo.
echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo.
echo Installing package in editable mode...
".venv\Scripts\python.exe" -m pip install -e ".[%EXTRAS%]"
if errorlevel 1 exit /b 1

if "%RUN_TESTS%"=="1" (
  echo.
  echo Running tests...
  ".venv\Scripts\python.exe" -m pytest -q
  if errorlevel 1 exit /b 1
) else (
  echo.
  echo Skipping tests.
)

echo.
echo Setup complete.
echo.
echo Common commands:
echo   .venv\Scripts\python.exe -m pytest -q
echo   .venv\Scripts\objectcount-detect --package models\packages\coco-demo --input ^<image^|folder^|video^> --out out\detect
echo   .venv\Scripts\objectcount-capture --package models\packages\coco-demo --out out\captures
exit /b 0

:help
echo Usage: setup.bat [--tools] [--skip-tests]
echo.
echo Options:
echo   --tools       Also install the tools extra, including ultralytics.
echo   --skip-tests  Install dependencies without running pytest.
exit /b 2
