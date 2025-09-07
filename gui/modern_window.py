"""
现代化主窗口
使用现代UI设计风格
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.modern_theme import theme
from crawler.dlt_crawler import DLTCrawler
from crawler.ssq_crawler import SSQCrawler
from crawler.historical_crawler import HistoricalCrawler
from data.database import LotteryDatabase
from data.data_processor import LotteryDataProcessor
from analysis.statistics import LotteryStatistics
from analysis.visualization import LotteryVisualization
from ml.traditional_ml import TraditionalMLPredictor
from ml.deep_learning import DeepLearningPredictor


class ModernLotteryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("彩票预测软件 v2.0 - 现代化界面")
        self.root.geometry("1400x900")
        self.root.configure(bg=theme.colors['bg_primary'])
        
        # 设置窗口图标和属性
        self.root.resizable(True, True)
        self.root.minsize(1200, 800)
        
        # 初始化组件
        self.db = LotteryDatabase()
        self.processor = LotteryDataProcessor()
        self.stats = LotteryStatistics()
        self.viz = LotteryVisualization()
        self.ml_predictor = TraditionalMLPredictor()
        self.dl_predictor = DeepLearningPredictor()
        
        # 数据存储
        self.current_data = None
        self.current_lottery_type = 'DLT'
        
        # 配置样式
        self.style = theme.configure_style()
        
        # 创建界面
        self.create_modern_interface()
        
    def create_modern_interface(self):
        """创建现代化界面"""
        # 创建主容器
        main_container = tk.Frame(self.root, bg=theme.colors['bg_primary'])
        main_container.pack(fill='both', expand=True)
        
        # 创建顶部标题栏
        self.create_header(main_container)
        
        # 创建主要内容区域
        content_frame = tk.Frame(main_container, bg=theme.colors['bg_primary'])
        content_frame.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=theme.spacing['md'])
        
        # 创建侧边栏和主内容区
        self.create_sidebar_and_content(content_frame)
        
        # 创建底部状态栏
        self.create_footer(main_container)
    
    def create_header(self, parent):
        """创建顶部标题栏"""
        header_frame = theme.create_card_frame(parent)
        header_frame.pack(fill='x', padx=theme.spacing['lg'], pady=(theme.spacing['lg'], theme.spacing['md']))
        
        # 标题和副标题
        title_frame = tk.Frame(header_frame, bg=theme.colors['bg_card'])
        title_frame.pack(fill='x', padx=theme.spacing['xl'], pady=theme.spacing['lg'])
        
        title_label = theme.create_modern_label(title_frame, "彩票预测软件", 'Title')
        title_label.pack(side='left')
        
        subtitle_label = theme.create_modern_label(title_frame, "基于机器学习的智能预测系统", 'Secondary')
        subtitle_label.pack(side='left', padx=(theme.spacing['lg'], 0))
        
        # 版本信息
        version_label = theme.create_modern_label(title_frame, "v2.0", 'Small')
        version_label.pack(side='right')
    
    def create_sidebar_and_content(self, parent):
        """创建侧边栏和主内容区"""
        # 主水平容器
        main_horizontal = tk.Frame(parent, bg=theme.colors['bg_primary'])
        main_horizontal.pack(fill='both', expand=True)
        
        # 侧边栏
        self.create_sidebar(main_horizontal)
        
        # 主内容区
        self.create_main_content(main_horizontal)
    
    def create_sidebar(self, parent):
        """创建侧边栏"""
        sidebar_frame = tk.Frame(parent, 
                                bg=theme.colors['bg_sidebar'],
                                width=280,
                                relief='solid',
                                bd=1)
        sidebar_frame.pack(side='left', fill='y', padx=(0, theme.spacing['md']))
        sidebar_frame.pack_propagate(False)
        
        # 侧边栏标题
        sidebar_title = theme.create_modern_label(sidebar_frame, "控制面板", 'Heading')
        sidebar_title.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_white'])
        sidebar_title.pack(pady=(theme.spacing['xl'], theme.spacing['lg']))
        
        # 彩票类型选择
        self.create_lottery_selector(sidebar_frame)
        
        # 操作按钮组
        self.create_action_buttons(sidebar_frame)
        
        # 数据统计信息
        self.create_data_stats(sidebar_frame)
        
        # 系统信息
        self.create_system_info(sidebar_frame)
    
    def create_lottery_selector(self, parent):
        """创建彩票类型选择器"""
        selector_frame = theme.create_card_frame(parent)
        selector_frame.config(bg=theme.colors['bg_sidebar'])
        selector_frame.pack(fill='x', padx=theme.spacing['lg'], pady=theme.spacing['md'])
        
        # 标题
        selector_title = theme.create_modern_label(selector_frame, "彩票类型", 'Subheading')
        selector_title.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_white'])
        selector_title.pack(pady=(theme.spacing['md'], theme.spacing['sm']))
        
        # 选择按钮组
        button_frame = tk.Frame(selector_frame, bg=theme.colors['bg_sidebar'])
        button_frame.pack(fill='x', padx=theme.spacing['md'], pady=(0, theme.spacing['md']))
        
        self.lottery_var = tk.StringVar(value="DLT")
        
        dlt_button = tk.Radiobutton(button_frame,
                                   text="超级大乐透",
                                   variable=self.lottery_var,
                                   value="DLT",
                                   command=self.on_lottery_type_change,
                                   bg=theme.colors['bg_sidebar'],
                                   fg=theme.colors['text_white'],
                                   selectcolor=theme.colors['primary'],
                                   font=theme.fonts['body'],
                                   activebackground=theme.colors['bg_sidebar'],
                                   activeforeground=theme.colors['text_white'])
        dlt_button.pack(fill='x', pady=theme.spacing['xs'])
        
        ssq_button = tk.Radiobutton(button_frame,
                                   text="双色球",
                                   variable=self.lottery_var,
                                   value="SSQ",
                                   command=self.on_lottery_type_change,
                                   bg=theme.colors['bg_sidebar'],
                                   fg=theme.colors['text_white'],
                                   selectcolor=theme.colors['primary'],
                                   font=theme.fonts['body'],
                                   activebackground=theme.colors['bg_sidebar'],
                                   activeforeground=theme.colors['text_white'])
        ssq_button.pack(fill='x', pady=theme.spacing['xs'])
    
    def create_action_buttons(self, parent):
        """创建操作按钮组"""
        action_frame = theme.create_card_frame(parent)
        action_frame.config(bg=theme.colors['bg_sidebar'])
        action_frame.pack(fill='x', padx=theme.spacing['lg'], pady=theme.spacing['md'])
        
        # 标题
        action_title = theme.create_modern_label(action_frame, "数据操作", 'Subheading')
        action_title.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_white'])
        action_title.pack(pady=(theme.spacing['md'], theme.spacing['sm']))
        
        # 按钮容器
        button_container = tk.Frame(action_frame, bg=theme.colors['bg_sidebar'])
        button_container.pack(fill='x', padx=theme.spacing['md'], pady=(0, theme.spacing['md']))
        
        # 数据获取按钮
        fetch_btn = theme.create_rounded_button(button_container, 
                                               "📊 获取数据", 
                                               self.fetch_data, 
                                               'Primary')
        fetch_btn.pack(fill='x', pady=theme.spacing['xs'])
        
        # 历史数据按钮
        historical_btn = theme.create_rounded_button(button_container, 
                                                    "📈 历史数据", 
                                                    self.fetch_historical_data, 
                                                    'Secondary')
        historical_btn.pack(fill='x', pady=theme.spacing['xs'])
        
        # 分析按钮
        analyze_btn = theme.create_rounded_button(button_container, 
                                                 "🔍 统计分析", 
                                                 self.analyze_data, 
                                                 'Success')
        analyze_btn.pack(fill='x', pady=theme.spacing['xs'])
        
        # 预测按钮组
        predict_frame = tk.Frame(button_container, bg=theme.colors['bg_sidebar'])
        predict_frame.pack(fill='x', pady=(theme.spacing['md'], 0))
        
        ml_btn = theme.create_rounded_button(predict_frame, 
                                            "🤖 ML预测", 
                                            self.ml_predict, 
                                            'Warning')
        ml_btn.pack(side='left', fill='x', expand=True, padx=(0, theme.spacing['xs']))
        
        dl_btn = theme.create_rounded_button(predict_frame, 
                                            "🧠 DL预测", 
                                            self.dl_predict, 
                                            'Danger')
        dl_btn.pack(side='right', fill='x', expand=True, padx=(theme.spacing['xs'], 0))
        
        # 保存按钮
        save_btn = theme.create_rounded_button(button_container, 
                                              "💾 保存结果", 
                                              self.save_results, 
                                              'Secondary')
        save_btn.pack(fill='x', pady=(theme.spacing['md'], 0))
    
    def create_data_stats(self, parent):
        """创建数据统计信息"""
        stats_frame = theme.create_card_frame(parent)
        stats_frame.config(bg=theme.colors['bg_sidebar'])
        stats_frame.pack(fill='x', padx=theme.spacing['lg'], pady=theme.spacing['md'])
        
        # 标题
        stats_title = theme.create_modern_label(stats_frame, "数据统计", 'Subheading')
        stats_title.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_white'])
        stats_title.pack(pady=(theme.spacing['md'], theme.spacing['sm']))
        
        # 统计信息容器
        self.stats_container = tk.Frame(stats_frame, bg=theme.colors['bg_sidebar'])
        self.stats_container.pack(fill='x', padx=theme.spacing['md'], pady=(0, theme.spacing['md']))
        
        self.update_data_stats()
    
    def create_system_info(self, parent):
        """创建系统信息"""
        system_frame = theme.create_card_frame(parent)
        system_frame.config(bg=theme.colors['bg_sidebar'])
        system_frame.pack(fill='x', padx=theme.spacing['lg'], pady=theme.spacing['md'])
        
        # 标题
        system_title = theme.create_modern_label(system_frame, "系统信息", 'Subheading')
        system_title.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_white'])
        system_title.pack(pady=(theme.spacing['md'], theme.spacing['sm']))
        
        # 系统信息容器
        info_container = tk.Frame(system_frame, bg=theme.colors['bg_sidebar'])
        info_container.pack(fill='x', padx=theme.spacing['md'], pady=(0, theme.spacing['md']))
        
        # 显示系统信息
        info_text = f"""版本: v2.0
状态: 运行中
时间: {datetime.now().strftime('%H:%M:%S')}"""
        
        info_label = theme.create_modern_label(info_container, info_text, 'Small')
        info_label.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_light'])
        info_label.pack(anchor='w')
    
    def create_main_content(self, parent):
        """创建主内容区"""
        content_frame = tk.Frame(parent, bg=theme.colors['bg_primary'])
        content_frame.pack(side='right', fill='both', expand=True)
        
        # 创建现代化选项卡
        self.create_modern_notebook(content_frame)
    
    def create_modern_notebook(self, parent):
        """创建现代化选项卡"""
        # 创建选项卡容器
        notebook_frame = theme.create_card_frame(parent)
        notebook_frame.pack(fill='both', expand=True)
        
        # 创建选项卡
        self.notebook = ttk.Notebook(notebook_frame, style='Modern.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=theme.spacing['lg'])
        
        # 创建各个选项卡
        self.create_data_tab()
        self.create_analysis_tab()
        self.create_prediction_tab()
        self.create_chart_tab()
    
    def create_data_tab(self):
        """创建数据查看选项卡"""
        data_frame = tk.Frame(self.notebook, bg=theme.colors['bg_primary'])
        self.notebook.add(data_frame, text="📊 数据查看")
        
        # 数据表格容器
        table_frame = theme.create_card_frame(data_frame)
        table_frame.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=theme.spacing['lg'])
        
        # 表格标题
        table_title = theme.create_modern_label(table_frame, "历史开奖数据", 'Heading')
        table_title.pack(pady=(theme.spacing['lg'], theme.spacing['md']))
        
        # 创建现代化表格
        self.create_modern_data_table(table_frame)
    
    def create_modern_data_table(self, parent):
        """创建现代化数据表格"""
        # 表格容器
        table_container = tk.Frame(parent, bg=theme.colors['bg_card'])
        table_container.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=(0, theme.spacing['lg']))
        
        # 创建Treeview
        columns = ('期号', '日期', '红球1', '红球2', '红球3', '红球4', '红球5', '蓝球1', '蓝球2')
        self.data_tree = ttk.Treeview(table_container, 
                                     columns=columns, 
                                     show='headings', 
                                     style='Modern.Treeview',
                                     height=20)
        
        # 设置列标题和宽度
        column_widths = {'期号': 80, '日期': 100, '红球1': 60, '红球2': 60, 
                        '红球3': 60, '红球4': 60, '红球5': 60, '蓝球1': 60, '蓝球2': 60}
        
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=column_widths.get(col, 80), anchor='center')
        
        # 滚动条
        scrollbar = ttk.Scrollbar(table_container, orient='vertical', command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.data_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def create_analysis_tab(self):
        """创建统计分析选项卡"""
        analysis_frame = tk.Frame(self.notebook, bg=theme.colors['bg_primary'])
        self.notebook.add(analysis_frame, text="🔍 统计分析")
        
        # 分析结果容器
        result_frame = theme.create_card_frame(analysis_frame)
        result_frame.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=theme.spacing['lg'])
        
        # 分析标题
        analysis_title = theme.create_modern_label(result_frame, "统计分析结果", 'Heading')
        analysis_title.pack(pady=(theme.spacing['lg'], theme.spacing['md']))
        
        # 分析结果文本框
        text_frame = tk.Frame(result_frame, bg=theme.colors['bg_card'])
        text_frame.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=(0, theme.spacing['lg']))
        
        self.analysis_text = tk.Text(text_frame, 
                                    wrap=tk.WORD, 
                                    bg=theme.colors['white'],
                                    fg=theme.colors['text_primary'],
                                    font=theme.fonts['body'],
                                    relief='solid',
                                    bd=1,
                                    padx=theme.spacing['md'],
                                    pady=theme.spacing['md'])
        
        analysis_scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        
        self.analysis_text.pack(side='left', fill='both', expand=True)
        analysis_scrollbar.pack(side='right', fill='y')
    
    def create_prediction_tab(self):
        """创建预测结果选项卡"""
        prediction_frame = tk.Frame(self.notebook, bg=theme.colors['bg_primary'])
        self.notebook.add(prediction_frame, text="🎯 预测结果")
        
        # 预测结果容器
        result_frame = theme.create_card_frame(prediction_frame)
        result_frame.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=theme.spacing['lg'])
        
        # 预测标题
        prediction_title = theme.create_modern_label(result_frame, "预测结果", 'Heading')
        prediction_title.pack(pady=(theme.spacing['lg'], theme.spacing['md']))
        
        # 预测结果文本框
        text_frame = tk.Frame(result_frame, bg=theme.colors['bg_card'])
        text_frame.pack(fill='both', expand=True, padx=theme.spacing['lg'], pady=(0, theme.spacing['lg']))
        
        self.prediction_text = tk.Text(text_frame, 
                                      wrap=tk.WORD, 
                                      bg=theme.colors['white'],
                                      fg=theme.colors['text_primary'],
                                      font=theme.fonts['body'],
                                      relief='solid',
                                      bd=1,
                                      padx=theme.spacing['md'],
                                      pady=theme.spacing['md'])
        
        prediction_scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.prediction_text.yview)
        self.prediction_text.configure(yscrollcommand=prediction_scrollbar.set)
        
        self.prediction_text.pack(side='left', fill='both', expand=True)
        prediction_scrollbar.pack(side='right', fill='y')
    
    def create_chart_tab(self):
        """创建图表展示选项卡"""
        chart_frame = tk.Frame(self.notebook, bg=theme.colors['bg_primary'])
        self.notebook.add(chart_frame, text="📈 图表展示")
        
        # 图表容器
        self.chart_frame = chart_frame
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
    
    def create_footer(self, parent):
        """创建底部状态栏"""
        footer_frame, self.status_label = theme.create_status_bar(parent)
        footer_frame.pack(fill='x', side='bottom')
    
    def update_data_stats(self):
        """更新数据统计信息"""
        # 清空现有统计信息
        for widget in self.stats_container.winfo_children():
            widget.destroy()
        
        # 获取数据库统计
        stats = self.db.get_database_stats()
        
        # 显示统计信息
        dlt_count = stats.get('dlt_count', 0)
        ssq_count = stats.get('ssq_count', 0)
        
        dlt_label = theme.create_modern_label(self.stats_container, f"大乐透: {dlt_count} 期", 'Body')
        dlt_label.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_light'])
        dlt_label.pack(anchor='w', pady=theme.spacing['xs'])
        
        ssq_label = theme.create_modern_label(self.stats_container, f"双色球: {ssq_count} 期", 'Body')
        ssq_label.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_light'])
        ssq_label.pack(anchor='w', pady=theme.spacing['xs'])
        
        total_label = theme.create_modern_label(self.stats_container, f"总计: {dlt_count + ssq_count} 期", 'Body')
        total_label.config(bg=theme.colors['bg_sidebar'], fg=theme.colors['text_white'])
        total_label.pack(anchor='w', pady=(theme.spacing['sm'], 0))
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def on_lottery_type_change(self):
        """彩票类型改变事件"""
        self.current_lottery_type = self.lottery_var.get()
        self.update_status(f"已切换到 {self.current_lottery_type}")
        self.update_data_stats()
    
    def fetch_data(self):
        """获取数据"""
        def fetch_thread():
            try:
                self.update_status("正在获取数据...")
                
                if self.current_lottery_type == 'DLT':
                    crawler = DLTCrawler()
                    data = crawler.get_lottery_data()
                    self.db.insert_dlt_data(data)
                else:  # SSQ
                    crawler = SSQCrawler()
                    data = crawler.get_lottery_data()
                    self.db.insert_ssq_data(data)
                
                # 更新数据表格和统计
                self.root.after(0, self.update_data_table)
                self.root.after(0, self.update_data_stats)
                self.root.after(0, lambda: self.update_status("数据获取完成"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"获取数据失败: {str(e)}"))
                self.root.after(0, lambda: self.update_status("数据获取失败"))
        
        threading.Thread(target=fetch_thread, daemon=True).start()
    
    def fetch_historical_data(self):
        """获取历史数据"""
        def fetch_thread():
            try:
                self.update_status("正在获取历史数据...")
                
                # 创建历史数据爬虫
                historical_crawler = HistoricalCrawler()
                
                # 爬取当前选择的彩票类型的历史数据
                data = historical_crawler.crawl_all_historical_data(
                    self.current_lottery_type, 
                    save_to_file=True
                )
                
                if not data.empty:
                    # 保存到数据库
                    if self.current_lottery_type == 'DLT':
                        self.db.insert_dlt_data(data)
                    else:
                        self.db.insert_ssq_data(data)
                    
                    # 更新数据表格和统计
                    self.root.after(0, self.update_data_table)
                    self.root.after(0, self.update_data_stats)
                    self.root.after(0, lambda: self.update_status(f"历史数据获取完成，共 {len(data)} 条"))
                else:
                    self.root.after(0, lambda: self.update_status("历史数据获取失败"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"获取历史数据失败: {str(e)}"))
                self.root.after(0, lambda: self.update_status("历史数据获取失败"))
        
        # 确认对话框
        result = messagebox.askyesno(
            "确认", 
            f"确定要爬取 {self.current_lottery_type} 的所有历史数据吗？\n"
            f"这可能需要较长时间（预计10-30分钟）。\n"
            f"大乐透：从2007-05-28开始\n"
            f"双色球：从2003-02-23开始"
        )
        
        if result:
            threading.Thread(target=fetch_thread, daemon=True).start()
    
    def update_data_table(self):
        """更新数据表格"""
        # 清空现有数据
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # 获取数据
        if self.current_lottery_type == 'DLT':
            data = self.db.get_dlt_data(limit=100)
        else:  # SSQ
            data = self.db.get_ssq_data(limit=100)
        
        if data.empty:
            return
        
        # 插入数据
        for _, row in data.iterrows():
            if self.current_lottery_type == 'DLT':
                values = (row['period'], row['date'], 
                         row['red_ball_1'], row['red_ball_2'], row['red_ball_3'], 
                         row['red_ball_4'], row['red_ball_5'],
                         row['blue_ball_1'], row['blue_ball_2'])
            else:  # SSQ
                values = (row['period'], row['date'], 
                         row['red_ball_1'], row['red_ball_2'], row['red_ball_3'], 
                         row['red_ball_4'], row['red_ball_5'], row['red_ball_6'],
                         row['blue_ball'], '')
            
            self.data_tree.insert('', 'end', values=values)
        
        self.current_data = data
    
    def analyze_data(self):
        """分析数据"""
        if self.current_data is None or self.current_data.empty:
            messagebox.showwarning("警告", "请先获取数据")
            return
        
        def analyze_thread():
            try:
                self.update_status("正在进行统计分析...")
                
                # 预处理数据
                processed_data = self.processor.preprocess_dlt_data(self.current_data) if self.current_lottery_type == 'DLT' else self.processor.preprocess_ssq_data(self.current_data)
                
                # 统计分析
                analysis_results = self.stats.comprehensive_analysis(processed_data, self.current_lottery_type)
                
                # 更新分析结果
                self.root.after(0, lambda: self.update_analysis_text(analysis_results))
                self.root.after(0, lambda: self.update_status("统计分析完成"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"分析失败: {str(e)}"))
                self.root.after(0, lambda: self.update_status("分析失败"))
        
        threading.Thread(target=analyze_thread, daemon=True).start()
    
    def update_analysis_text(self, results):
        """更新分析结果文本"""
        self.analysis_text.delete(1.0, tk.END)
        
        text = f"=== {self.current_lottery_type} 统计分析结果 ===\n\n"
        
        # 频率分析
        freq = results['frequency']
        text += "1. 频率分析:\n"
        text += f"   总开奖期数: {freq['total_draws']}\n"
        text += f"   红球期望频率: {freq['red_expected']:.2f}\n"
        text += f"   蓝球期望频率: {freq['blue_expected']:.2f}\n\n"
        
        # 热号冷号
        hot_cold = results['hot_cold']
        text += "2. 热号冷号分析 (最近20期):\n"
        text += f"   红球热号: {hot_cold['red_hot_numbers'][:5]}\n"
        text += f"   红球冷号: {hot_cold['red_cold_numbers'][-5:]}\n"
        text += f"   蓝球热号: {hot_cold['blue_hot_numbers'][:3]}\n"
        text += f"   蓝球冷号: {hot_cold['blue_cold_numbers'][-3:]}\n\n"
        
        # 和值分析
        sum_analysis = results['sum']
        text += "3. 和值分析:\n"
        text += f"   红球和值 - 平均: {sum_analysis['red_sum_stats']['mean']:.2f}, "
        text += f"标准差: {sum_analysis['red_sum_stats']['std']:.2f}\n"
        text += f"   蓝球和值 - 平均: {sum_analysis['blue_sum_stats']['mean']:.2f}, "
        text += f"标准差: {sum_analysis['blue_sum_stats']['std']:.2f}\n\n"
        
        # 奇偶分析
        odd_even = results['odd_even']
        text += "4. 奇偶分析:\n"
        text += f"   红球奇偶分布: {dict(odd_even['red_odd_distribution'])}\n"
        text += f"   蓝球奇偶分布: {dict(odd_even['blue_odd_distribution'])}\n\n"
        
        # 大小号分析
        size_analysis = results['size']
        text += "5. 大小号分析:\n"
        text += f"   红球大小分布: {dict(size_analysis['red_big_distribution'])}\n"
        text += f"   蓝球大小分布: {dict(size_analysis['blue_big_distribution'])}\n\n"
        
        # 连号分析
        consecutive = results['consecutive']
        text += "6. 连号分析:\n"
        text += f"   红球连号分布: {dict(consecutive['red_consecutive_distribution'])}\n"
        text += f"   蓝球连号分布: {dict(consecutive['blue_consecutive_distribution'])}\n\n"
        
        # AC值分析
        ac_value = results['ac_value']
        text += "7. AC值分析:\n"
        text += f"   平均AC值: {ac_value['mean_ac']:.2f}\n"
        text += f"   AC值标准差: {ac_value['std_ac']:.2f}\n"
        text += f"   AC值分布: {dict(ac_value['ac_distribution'])}\n\n"
        
        # 趋势分析
        trend = results['trend']
        text += "8. 趋势分析:\n"
        text += f"   当前趋势方向: {'上升' if trend['current_trend'] > 0 else '下降' if trend['current_trend'] < 0 else '平稳'}\n"
        text += f"   平均趋势强度: {trend['avg_trend_strength']:.4f}\n"
        
        self.analysis_text.insert(1.0, text)
    
    def ml_predict(self):
        """机器学习预测"""
        if self.current_data is None or self.current_data.empty:
            messagebox.showwarning("警告", "请先获取数据")
            return
        
        def predict_thread():
            try:
                self.update_status("正在进行机器学习预测...")
                
                # 预处理数据
                processed_data = self.processor.preprocess_dlt_data(self.current_data) if self.current_lottery_type == 'DLT' else self.processor.preprocess_ssq_data(self.current_data)
                
                # 准备特征
                X, y = self.processor.prepare_ml_data(processed_data, self.current_lottery_type)
                
                if len(X) < 50:
                    self.root.after(0, lambda: messagebox.showwarning("警告", "数据量不足，无法进行机器学习预测"))
                    return
                
                # 训练模型
                self.ml_predictor.train_models(X, y, self.current_lottery_type)
                
                # 预测
                prediction = self.ml_predictor.predict_next_period(processed_data, self.current_lottery_type)
                ensemble_pred = self.ml_predictor.ensemble_predict(processed_data, self.current_lottery_type)
                
                # 更新预测结果
                self.root.after(0, lambda: self.update_prediction_text('ML', prediction, ensemble_pred))
                self.root.after(0, lambda: self.update_status("机器学习预测完成"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"预测失败: {str(e)}"))
                self.root.after(0, lambda: self.update_status("预测失败"))
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def dl_predict(self):
        """深度学习预测"""
        if self.current_data is None or self.current_data.empty:
            messagebox.showwarning("警告", "请先获取数据")
            return
        
        def predict_thread():
            try:
                self.update_status("正在进行深度学习预测...")
                
                # 预处理数据
                processed_data = self.processor.preprocess_dlt_data(self.current_data) if self.current_lottery_type == 'DLT' else self.processor.preprocess_ssq_data(self.current_data)
                
                if len(processed_data) < 100:
                    self.root.after(0, lambda: messagebox.showwarning("警告", "数据量不足，无法进行深度学习预测"))
                    return
                
                # 准备序列数据
                sequence_data = self.dl_predictor.prepare_sequence_data(processed_data, self.current_lottery_type)
                
                # 训练模型
                self.dl_predictor.train_model(sequence_data, self.current_lottery_type, 'LSTM', epochs=50)
                
                # 预测
                prediction = self.dl_predictor.predict_next_period(processed_data, self.current_lottery_type)
                ensemble_pred = self.dl_predictor.ensemble_deep_predict(processed_data, self.current_lottery_type)
                
                # 更新预测结果
                self.root.after(0, lambda: self.update_prediction_text('DL', prediction, ensemble_pred))
                self.root.after(0, lambda: self.update_status("深度学习预测完成"))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("错误", f"预测失败: {str(e)}"))
                self.root.after(0, lambda: self.update_status("预测失败"))
        
        threading.Thread(target=predict_thread, daemon=True).start()
    
    def update_prediction_text(self, method, prediction, ensemble_pred):
        """更新预测结果文本"""
        self.prediction_text.delete(1.0, tk.END)
        
        text = f"=== {method} 预测结果 ===\n\n"
        
        if prediction:
            text += "1. 单模型预测:\n"
            text += f"   红球: {prediction['red_balls']}\n"
            text += f"   蓝球: {prediction['blue_balls']}\n\n"
        
        if ensemble_pred:
            text += "2. 集成预测:\n"
            text += f"   红球: {ensemble_pred['red_balls']}\n"
            text += f"   蓝球: {ensemble_pred['blue_balls']}\n"
            text += f"   权重: {ensemble_pred['ensemble_weights']}\n\n"
        
        # 添加模型性能信息
        if method == 'ML':
            performance = self.ml_predictor.get_model_performance()
            text += "3. 模型性能:\n"
            for model_name, scores in performance.head(3).iterrows():
                text += f"   {model_name}: R² = {scores['r2']:.4f}, MSE = {scores['mse']:.4f}\n"
        
        text += "\n注意: 彩票具有随机性，任何预测都不能保证准确性。请理性购彩！"
        
        self.prediction_text.insert(1.0, text)
    
    def save_results(self):
        """保存结果"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=== 彩票预测结果 ===\n\n")
                    f.write(f"彩票类型: {self.current_lottery_type}\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # 保存分析结果
                    analysis_text = self.analysis_text.get(1.0, tk.END)
                    if analysis_text.strip():
                        f.write("=== 统计分析结果 ===\n")
                        f.write(analysis_text)
                        f.write("\n")
                    
                    # 保存预测结果
                    prediction_text = self.prediction_text.get(1.0, tk.END)
                    if prediction_text.strip():
                        f.write("=== 预测结果 ===\n")
                        f.write(prediction_text)
                
                messagebox.showinfo("成功", f"结果已保存到: {filename}")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = ModernLotteryGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
