@echo off
REM 设置环境变量
set PYTHONHASHSEED=123
set PYTHONPATH=%~dp0

REM 显示当前环境信息
echo 启动环境:
echo 工作目录: %~dp0
echo PYTHONHASHSEED: %PYTHONHASHSEED%
echo PYTHONPATH: %PYTHONPATH%

REM 直接运行Python脚本
python backend/main.py

REM 脚本结束后暂停
pause
