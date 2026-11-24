@echo off
rem Double-click: outline froth bubbles on the measured region of one frame (complete tiles are scored)
title Froth frame annotator
cd /d "%~dp0"
if exist "..\BubMask\.venv\Scripts\python.exe" (
  "..\BubMask\.venv\Scripts\python.exe" "eval\frame_annotator.py" "eval\to_label_frame"
) else (
  ".venv\Scripts\python.exe" "eval\frame_annotator.py" "eval\to_label_frame"
)
if errorlevel 1 pause
