@echo off
call conda activate seed-parse
wmic process where name="python.exe" CALL setpriority 128
python main.py
pause