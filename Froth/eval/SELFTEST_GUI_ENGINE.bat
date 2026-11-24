@echo off
title Froth GUI engine self-test
cd /d "%~dp0..\.."
"BubMask\.venv\Scripts\python.exe" "Froth\eval\selftest_gui_engine.py"
echo.
echo Done - see Froth\eval\selftest_gui_engine.txt
timeout /t 15
