"""
主窗口GUI模块
提供用户友好的图形界面
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

from crawler.dlt_crawler import DLTCrawler
from crawler.ssq_crawler import SSQCrawler
from crawler.historical_crawler import HistoricalCrawler
from data.database import LotteryDatabase
from data.data_processor import LotteryDataProcessor
from analysis.statistics import LotteryStatistics
from analysis.visualization import LotteryVisualization
from ml.traditional_ml import TraditionalMLPredictor
from ml.deep_learning import DeepLearningPredictor


class LotteryPredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("彩票预测软件 v1.0")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
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
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="彩票预测软件", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 控制面板
        self.create_control_panel(main_frame)
        
        # 创建选项卡
        self.create_notebook(main_frame)
        
        # 状态栏
        self.create_status_bar(main_frame)
    
    def create_control_panel(self, parent):
        """创建控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 彩票类型选择
        ttk.Label(control_frame, text="彩票类型:").grid(row=0, column=0, padx=(0, 5))
        self.lottery_var = tk.StringVar(value="DLT")
        lottery_combo = ttk.Combobox(control_frame, textvariable=self.lottery_var, 
                                   values=["DLT", "SSQ"], state="readonly", width=10)
        lottery_combo.grid(row=0, column=1, padx=(0, 20))
        lottery_combo.bind('<<ComboboxSelected>>', self.on_lottery_type_change)
        
        # 数据获取按钮
        ttk.Button(control_frame, text="获取数据", 
                  command=self.fetch_data).grid(row=0, column=2, padx=(0, 10))
        
        # 历史数据按钮
        ttk.Button(control_frame, text="历史数据", 
                  command=self.fetch_historical_data).grid(row=0, column=3, padx=(0, 10))
        
        # 数据分析按钮
        ttk.Button(control_frame, text="统计分析", 
                  command=self.analyze_data).grid(row=0, column=4, padx=(0, 10))
        
        # 机器学习预测按钮
        ttk.Button(control_frame, text="ML预测", 
                  command=self.ml_predict).grid(row=0, column=5, padx=(0, 10))
        
        # 深度学习预测按钮
        ttk.Button(control_frame, text="DL预测", 
                  command=self.dl_predict).grid(row=0, column=6, padx=(0, 10))
        
        # 保存结果按钮
        ttk.Button(control_frame, text="保存结果", 
                  command=self.save_results).grid(row=0, column=7)
    
    def create_notebook(self, parent):
        """创建选项卡"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 数据查看选项卡
        self.create_data_tab()
        
        # 统计分析选项卡
        self.create_analysis_tab()
        
        # 预测结果选项卡
        self.create_prediction_tab()
        
        # 图表展示选项卡
        self.create_chart_tab()
    
    def create_data_tab(self):
        """创建数据查看选项卡"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="数据查看")
        
        # 数据表格
        self.create_data_table(data_frame)
    
    def create_data_table(self, parent):
        """创建数据表格"""
        # 创建Treeview
        columns = ('期号', '日期', '红球1', '红球2', '红球3', '红球4', '红球5', '蓝球1', '蓝球2')
        self.data_tree = ttk.Treeview(parent, columns=columns, show='headings', height=15)
        
        # 设置列标题
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=80, anchor='center')
        
        # 滚动条
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        self.data_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
    
    def create_analysis_tab(self):
        """创建统计分析选项卡"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="统计分析")
        
        # 分析结果文本框
        self.analysis_text = tk.Text(analysis_frame, wrap=tk.WORD, height=20)
        analysis_scrollbar = ttk.Scrollbar(analysis_frame, orient=tk.VERTICAL, 
                                         command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        
        self.analysis_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        analysis_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(0, weight=1)
    
    def create_prediction_tab(self):
        """创建预测结果选项卡"""
        prediction_frame = ttk.Frame(self.notebook)
        self.notebook.add(prediction_frame, text="预测结果")
        
        # 预测结果文本框
        self.prediction_text = tk.Text(prediction_frame, wrap=tk.WORD, height=20)
        prediction_scrollbar = ttk.Scrollbar(prediction_frame, orient=tk.VERTICAL, 
                                           command=self.prediction_text.yview)
        self.prediction_text.configure(yscrollcommand=prediction_scrollbar.set)
        
        self.prediction_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        prediction_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        prediction_frame.columnconfigure(0, weight=1)
        prediction_frame.rowconfigure(0, weight=1)
    
    def create_chart_tab(self):
        """创建图表展示选项卡"""
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="图表展示")
        
        # 图表框架
        self.chart_frame = chart_frame
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
    
    def create_status_bar(self, parent):
        """创建状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def on_lottery_type_change(self, event=None):
        """彩票类型改变事件"""
        self.current_lottery_type = self.lottery_var.get()
        self.update_status(f"已切换到 {self.current_lottery_type}")
    
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
                
                # 更新数据表格
                self.root.after(0, self.update_data_table)
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
                    
                    # 更新数据表格
                    self.root.after(0, self.update_data_table)
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
    app = LotteryPredictionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
