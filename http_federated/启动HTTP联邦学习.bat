@echo off
chcp 65001 >nul
title 联邦学习 - 启动脚本

set "ALGORITHM=fedavg"
set "MU=0.01"

:: 解析命令行参数
:parse_args
if "%~1"=="" goto :check_args
if "%~1"=="--fedprox" (
    set "ALGORITHM=fedprox"
    shift
    goto :parse_args
)
if "%~1"=="-mu" (
    set "MU=%~2"
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args

:check_args
echo ============================================================
echo 联邦学习 - HTTP版本
echo ============================================================
echo.
echo 当前配置:
echo   - 算法: %ALGORITHM%
if "%ALGORITHM%"=="fedprox" (
    echo   - FedProx μ: %MU%
)
echo.

echo 检查虚拟环境...
if not exist "..\.venv\Scripts\python.exe" (
    echo [ERROR] 找不到虚拟环境: ..\.venv
    pause
    exit /b 1
)

set PYTHON_PATH=..\.venv\Scripts\python.exe

echo [OK] 虚拟环境已就绪
echo.
echo 即将启动 4 个终端:
echo   - 1 个中心服务器 (端口 5001)
echo   - 3 个客户端 (端口 6000-6002)
echo.
echo 系统将自动等待客户端连接，然后开始训练
echo.
pause

echo.
echo ============================================================
echo 启动客户端 0...
echo ============================================================
start "FL-Client-0" cmd /k "%PYTHON_PATH% client.py 0 6000"

timeout /t 1 /nobreak >nul

echo.
echo ============================================================
echo 启动客户端 1...
echo ============================================================
start "FL-Client-1" cmd /k "%PYTHON_PATH% client.py 1 6001"

timeout /t 1 /nobreak >nul

echo.
echo ============================================================
echo 启动客户端 2...
echo ============================================================
start "FL-Client-2" cmd /k "%PYTHON_PATH% client.py 2 6002"

timeout /t 2 /nobreak >nul

echo.
echo ============================================================
echo 启动中心服务器 (自动模式, %ALGORITHM%)...
echo ============================================================
if "%ALGORITHM%"=="fedprox" (
    start "FL-Server" cmd /k "%PYTHON_PATH% server.py --auto --fedprox -mu %MU%"
) else (
    start "FL-Server" cmd /k "%PYTHON_PATH% server.py --auto"
)

echo.
echo ============================================================
echo 所有终端已启动！
echo ============================================================
echo.
echo 训练将自动开始，请观察各终端的日志输出
echo.
echo 按 Ctrl+C 可以强制退出所有进程
echo.

pause