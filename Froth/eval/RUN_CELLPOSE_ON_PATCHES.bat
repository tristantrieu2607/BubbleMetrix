@echo off
rem Cellpose (zero-shot) on the frames the labelled patches come from, exactly as the GUI
rem runs it: the in-focus region of the full frame. Output: Froth\eval\results\patch_runs\
title Cellpose on labelling frames
cd /d "%~dp0..\.."
set PY=Froth\.venv\Scripts\python.exe
rem Frames already done are skipped (delete their folder to redo).
for %%F in (150 240 300 360 420) do (
  echo === frame t=%%Fs
  if not exist "Froth\eval\results\patch_runs\t0%%Fs\cellpose_labels.npy" "%PY%" "Froth\cellpose\run_cellpose.py" --image "Froth\eval\data\source_frames\frame_t0%%Fs.png" --auto-roi --out-dir "Froth\eval\results\patch_runs\t0%%Fs" --out-prefix cellpose
)
echo ALL DONE > "Froth\eval\results\patch_runs\DONE.txt"
echo.
echo Done. You can close this window.
timeout /t 20
