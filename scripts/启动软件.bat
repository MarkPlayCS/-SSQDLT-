@echo off
chcp 65001
echo ========================================
echo 彩票预测软件 v1.0
echo ========================================
echo 正在启动应用程序...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖包
echo 检查依赖包...
python -c "import pandas, numpy, sklearn, matplotlib, requests, bs4" >nul 2>&1
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖包安装失败，请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM 启动应用程序
echo 启动应用程序...
python main.py

if errorlevel 1 (
    echo.
    echo 应用程序启动失败，请检查错误信息
    pause
)
