@echo off
chcp 65001 >nul
title HTTP联邦学习 - 前端启动

echo ============================================================
echo HTTP联邦学习前端界面
echo ============================================================
echo.

echo 启动前端页面...
start "" "http://localhost:5002/http_fl_start.html"

echo.
echo 前端页面已启动！
echo.
pause