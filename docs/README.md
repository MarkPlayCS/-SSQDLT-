# 彩票预测软件

一个基于机器学习和深度学习的彩票号码预测系统，支持超级大乐透和双色球的历史数据分析和预测。

## 功能特点

- 🕷️ 自动爬取历史开奖数据
- 📊 多种统计分析方法
- 🤖 机器学习预测算法
- 🧠 深度学习神经网络模型
- 🖥️ 友好的图形用户界面
- 📈 数据可视化分析

## 安装说明

1. 安装Python 3.8+
2. 安装依赖包：
```bash
pip install -r requirements.txt
```

## 使用方法

运行主程序：
```bash
python main.py
```

## 项目结构

```
彩票预测软件/
├── main.py                 # 主程序入口
├── crawler/               # 爬虫模块
│   ├── __init__.py
│   ├── dlt_crawler.py     # 大乐透爬虫
│   └── ssq_crawler.py     # 双色球爬虫
├── data/                  # 数据管理模块
│   ├── __init__.py
│   ├── database.py        # 数据库操作
│   └── data_processor.py  # 数据处理
├── analysis/              # 统计分析模块
│   ├── __init__.py
│   ├── statistics.py      # 统计分析
│   └── visualization.py   # 数据可视化
├── ml/                    # 机器学习模块
│   ├── __init__.py
│   ├── traditional_ml.py  # 传统ML算法
│   └── deep_learning.py   # 深度学习模型
├── gui/                   # 用户界面模块
│   ├── __init__.py
│   └── main_window.py     # 主窗口
└── utils/                 # 工具模块
    ├── __init__.py
    └── helpers.py         # 辅助函数
```

## 免责声明

本软件仅供学习和研究使用，不构成任何投资建议。彩票具有随机性，任何预测都不能保证准确性。请理性购彩，量力而行。
