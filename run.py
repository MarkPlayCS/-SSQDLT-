"""
简化启动脚本
用于快速启动彩票预测软件
"""

import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """主函数"""
    print("=" * 50)
    print("彩票预测软件 v1.0")
    print("=" * 50)
    print("正在启动应用程序...")
    
    try:
        from main import main as app_main
        app_main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖包:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"启动失败: {e}")
        input("按回车键退出...")

if __name__ == "__main__":
    main()
