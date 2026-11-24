@echo off
title BubbleMetrix Desktop Studio
cd /d "%~dp0"
echo ============================================================
echo   Starting BubbleMetrix Desktop Studio (No Server Required)
echo ============================================================
if exist "BubMask\.venv\Scripts\python.exe" (
    "BubMask\.venv\Scripts\python.exe" bubble_gui.py
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" bubble_gui.py
) else (
    python bubble_gui.py
)
if %errorlevel% neq 0 (
    echo.
    echo An error occurred while running the GUI.
    pause
)
