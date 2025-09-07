@echo off
chcp 65001
echo ========================================
echo 彩票历史数据爬取工具
echo ========================================
echo 此工具将爬取从彩票诞生之日起的所有历史数据
echo.
echo 大乐透诞生日期: 2007-05-28
echo 双色球诞生日期: 2003-02-23
echo.
echo 预计数据量:
echo   大乐透: 6000+ 期
echo   双色球: 6000+ 期
echo.
echo 预计耗时: 10-30分钟
echo.
echo 注意: 此过程需要稳定的网络连接
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo 开始爬取历史数据...
echo.

REM 运行爬取脚本
python crawl_all_data.py

echo.
echo ========================================
echo 历史数据爬取完成！
echo ========================================
echo.
echo 数据已保存到:
echo   - 数据库文件: lottery_data.db
echo   - CSV文件: DLT_historical_data_*.csv
echo   - CSV文件: SSQ_historical_data_*.csv
echo.
echo 您现在可以:
echo   1. 运行 python main.py 启动软件
echo   2. 在软件中进行数据分析和预测
echo.
pause
