"""
彩票预测软件主程序
整合所有模块，提供完整的彩票预测功能
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import setup_logging
from gui.main_window import LotteryPredictionGUI


def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 
        'matplotlib', 'seaborn', 'plotly', 'requests', 
        'beautifulsoup4', 'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'scikit-learn':
                import sklearn
            elif package == 'beautifulsoup4':
                import bs4
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        error_msg = f"缺少以下依赖包: {', '.join(missing_packages)}\n"
        error_msg += "请运行以下命令安装:\n"
        error_msg += "pip install -r requirements.txt"
        
        print(error_msg)
        return False
    
    return True


def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("彩票预测软件启动")
    
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败")
        return
    
    try:
        # 创建主窗口
        root = tk.Tk()
        
        # 设置窗口图标（如果有的话）
        try:
            # root.iconbitmap('icon.ico')  # 如果有图标文件
            pass
        except:
            pass
        
        # 创建应用程序
        app = LotteryPredictionGUI(root)
        
        # 显示欢迎信息
        logger.info("应用程序界面已创建")
        
        # 启动主循环
        root.mainloop()
        
    except Exception as e:
        logger.error(f"应用程序启动失败: {str(e)}")
        messagebox.showerror("错误", f"应用程序启动失败: {str(e)}")
    
    logger.info("彩票预测软件退出")


if __name__ == "__main__":
    main()
