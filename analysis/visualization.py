"""
数据可视化模块
提供各种图表和可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class LotteryVisualization:
    def __init__(self):
        self.colors = {
            'red': '#FF6B6B',
            'blue': '#4ECDC4',
            'green': '#45B7D1',
            'orange': '#FFA07A',
            'purple': '#98D8C8',
            'gray': '#6C7B7F'
        }
    
    def plot_frequency_chart(self, freq_data, lottery_type, save_path=None):
        """绘制频率分析图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 红球频率
        red_freq = freq_data['red_frequency']
        red_numbers = list(red_freq.keys())
        red_counts = list(red_freq.values())
        
        bars1 = ax1.bar(red_numbers, red_counts, color=self.colors['red'], alpha=0.7)
        ax1.axhline(y=freq_data['red_expected'], color='red', linestyle='--', 
                   label=f'期望频率: {freq_data["red_expected"]:.1f}')
        ax1.set_title(f'{lottery_type} 红球频率分析', fontsize=16, fontweight='bold')
        ax1.set_xlabel('号码')
        ax1.set_ylabel('出现次数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars1, red_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=8)
        
        # 蓝球频率
        blue_freq = freq_data['blue_frequency']
        blue_numbers = list(blue_freq.keys())
        blue_counts = list(blue_freq.values())
        
        bars2 = ax2.bar(blue_numbers, blue_counts, color=self.colors['blue'], alpha=0.7)
        ax2.axhline(y=freq_data['blue_expected'], color='blue', linestyle='--',
                   label=f'期望频率: {freq_data["blue_expected"]:.1f}')
        ax2.set_title(f'{lottery_type} 蓝球频率分析', fontsize=16, fontweight='bold')
        ax2.set_xlabel('号码')
        ax2.set_ylabel('出现次数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars2, blue_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_hot_cold_chart(self, hot_cold_data, lottery_type, save_path=None):
        """绘制冷热号分析图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 热号
        hot_numbers = hot_cold_data['red_hot_numbers'][:10]
        hot_counts = [hot_cold_data['red_hot_cold'][num] for num in hot_numbers]
        
        bars1 = ax1.bar(range(len(hot_numbers)), hot_counts, color=self.colors['red'], alpha=0.7)
        ax1.set_title(f'{lottery_type} 红球热号 (最近{hot_cold_data["window_size"]}期)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('热号排名')
        ax1.set_ylabel('出现次数')
        ax1.set_xticks(range(len(hot_numbers)))
        ax1.set_xticklabels(hot_numbers)
        ax1.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars1, hot_counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=10)
        
        # 冷号
        cold_numbers = hot_cold_data['red_cold_numbers'][-10:]
        cold_counts = [hot_cold_data['red_hot_cold'][num] for num in cold_numbers]
        
        bars2 = ax2.bar(range(len(cold_numbers)), cold_counts, color=self.colors['blue'], alpha=0.7)
        ax2.set_title(f'{lottery_type} 红球冷号 (最近{hot_cold_data["window_size"]}期)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('冷号排名')
        ax2.set_ylabel('出现次数')
        ax2.set_xticks(range(len(cold_numbers)))
        ax2.set_xticklabels(cold_numbers)
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars2, cold_counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sum_trend(self, sum_data, lottery_type, save_path=None):
        """绘制和值趋势图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 红球和值趋势
        periods = range(len(sum_data['red_sums']))
        ax1.plot(periods, sum_data['red_sums'], color=self.colors['red'], linewidth=2, alpha=0.8)
        ax1.axhline(y=sum_data['red_sum_stats']['mean'], color='red', linestyle='--',
                   label=f'平均值: {sum_data["red_sum_stats"]["mean"]:.1f}')
        ax1.fill_between(periods, sum_data['red_sums'], alpha=0.3, color=self.colors['red'])
        ax1.set_title(f'{lottery_type} 红球和值趋势', fontsize=16, fontweight='bold')
        ax1.set_xlabel('期数')
        ax1.set_ylabel('和值')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 蓝球和值趋势
        ax2.plot(periods, sum_data['blue_sums'], color=self.colors['blue'], linewidth=2, alpha=0.8)
        ax2.axhline(y=sum_data['blue_sum_stats']['mean'], color='blue', linestyle='--',
                   label=f'平均值: {sum_data["blue_sum_stats"]["mean"]:.1f}')
        ax2.fill_between(periods, sum_data['blue_sums'], alpha=0.3, color=self.colors['blue'])
        ax2.set_title(f'{lottery_type} 蓝球和值趋势', fontsize=16, fontweight='bold')
        ax2.set_xlabel('期数')
        ax2.set_ylabel('和值')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distribution_chart(self, data, title, save_path=None):
        """绘制分布图表"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if isinstance(data, dict):
            numbers = list(data.keys())
            counts = list(data.values())
        else:
            numbers = data.index.tolist()
            counts = data.values.tolist()
        
        bars = ax.bar(numbers, counts, color=self.colors['green'], alpha=0.7)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('数值')
        ax.set_ylabel('频次')
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_heatmap(self, df, lottery_type, save_path=None):
        """绘制相关性热力图"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
        
        # 计算相关性矩阵
        all_cols = red_cols + blue_cols
        corr_matrix = df[all_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'{lottery_type} 号码相关性热力图', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_trend(self, df, lottery_type):
        """绘制交互式趋势图"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
        
        # 计算和值
        red_sums = df[red_cols].sum(axis=1)
        
        # 创建交互式图表
        fig = go.Figure()
        
        # 添加和值线
        fig.add_trace(go.Scatter(
            x=list(range(len(red_sums))),
            y=red_sums,
            mode='lines+markers',
            name='和值',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        ))
        
        # 添加移动平均线
        window = 10
        moving_avg = red_sums.rolling(window=window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=list(range(len(moving_avg))),
            y=moving_avg,
            mode='lines',
            name=f'{window}期移动平均',
            line=dict(color='blue', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{lottery_type} 和值趋势分析',
            xaxis_title='期数',
            yaxis_title='和值',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def plot_number_heatmap(self, df, lottery_type, save_path=None):
        """绘制号码热力图"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            red_range = (1, 35)
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            red_range = (1, 33)
        
        # 创建号码出现矩阵
        period_count = len(df)
        number_range = red_range[1] - red_range[0] + 1
        
        # 初始化矩阵
        heatmap_data = np.zeros((period_count, number_range))
        
        # 填充数据
        for i, (_, row) in enumerate(df.iterrows()):
            for col in red_cols:
                num = row[col]
                heatmap_data[i, num - red_range[0]] = 1
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 绘制热力图
        im = ax.imshow(heatmap_data.T, cmap='Reds', aspect='auto')
        
        # 设置标签
        ax.set_xlabel('期数')
        ax.set_ylabel('号码')
        ax.set_title(f'{lottery_type} 号码出现热力图', fontsize=16, fontweight='bold')
        
        # 设置y轴标签
        ax.set_yticks(range(0, number_range, 5))
        ax.set_yticklabels(range(red_range[0], red_range[1] + 1, 5))
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('出现 (1) / 未出现 (0)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_dashboard(self, analysis_results, lottery_type, save_path=None):
        """创建综合分析仪表板"""
        fig = plt.figure(figsize=(20, 15))
        
        # 创建子图网格
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 频率分析
        ax1 = fig.add_subplot(gs[0, 0])
        red_freq = analysis_results['frequency']['red_frequency']
        red_numbers = list(red_freq.keys())[:10]
        red_counts = [red_freq[num] for num in red_numbers]
        ax1.bar(red_numbers, red_counts, color=self.colors['red'], alpha=0.7)
        ax1.set_title('红球频率 Top10')
        ax1.grid(True, alpha=0.3)
        
        # 2. 和值分布
        ax2 = fig.add_subplot(gs[0, 1])
        sum_dist = analysis_results['sum']['red_sum_distribution']
        ax2.hist(sum_dist.index, weights=sum_dist.values, bins=20, 
                color=self.colors['blue'], alpha=0.7, edgecolor='black')
        ax2.set_title('和值分布')
        ax2.grid(True, alpha=0.3)
        
        # 3. 奇偶分布
        ax3 = fig.add_subplot(gs[0, 2])
        odd_dist = analysis_results['odd_even']['red_odd_distribution']
        odd_counts = list(odd_dist.keys())
        odd_freq = list(odd_dist.values())
        ax3.bar(odd_counts, odd_freq, color=self.colors['green'], alpha=0.7)
        ax3.set_title('奇偶分布')
        ax3.set_xlabel('奇数个数')
        ax3.grid(True, alpha=0.3)
        
        # 4. 大小号分布
        ax4 = fig.add_subplot(gs[1, 0])
        size_dist = analysis_results['size']['red_big_distribution']
        size_counts = list(size_dist.keys())
        size_freq = list(size_dist.values())
        ax4.bar(size_counts, size_freq, color=self.colors['orange'], alpha=0.7)
        ax4.set_title('大小号分布')
        ax4.set_xlabel('大号个数')
        ax4.grid(True, alpha=0.3)
        
        # 5. 连号分布
        ax5 = fig.add_subplot(gs[1, 1])
        consecutive_dist = analysis_results['consecutive']['red_consecutive_distribution']
        consecutive_counts = list(consecutive_dist.keys())
        consecutive_freq = list(consecutive_dist.values())
        ax5.bar(consecutive_counts, consecutive_freq, color=self.colors['purple'], alpha=0.7)
        ax5.set_title('连号分布')
        ax5.set_xlabel('连号个数')
        ax5.grid(True, alpha=0.3)
        
        # 6. AC值分布
        ax6 = fig.add_subplot(gs[1, 2])
        ac_dist = analysis_results['ac_value']['ac_distribution']
        ac_values = list(ac_dist.keys())
        ac_freq = list(ac_dist.values())
        ax6.bar(ac_values, ac_freq, color=self.colors['gray'], alpha=0.7)
        ax6.set_title('AC值分布')
        ax6.set_xlabel('AC值')
        ax6.grid(True, alpha=0.3)
        
        # 7. 和值趋势
        ax7 = fig.add_subplot(gs[2, :])
        red_sums = analysis_results['sum']['red_sums']
        ax7.plot(red_sums, color=self.colors['red'], linewidth=2, alpha=0.8)
        ax7.axhline(y=analysis_results['sum']['red_sum_stats']['mean'], 
                   color='blue', linestyle='--', label='平均值')
        ax7.set_title('和值趋势')
        ax7.set_xlabel('期数')
        ax7.set_ylabel('和值')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        plt.suptitle(f'{lottery_type} 综合分析仪表板', fontsize=20, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # 测试可视化功能
    viz = LotteryVisualization()
    
    # 创建测试数据
    test_freq_data = {
        'red_frequency': {i: np.random.randint(10, 50) for i in range(1, 36)},
        'blue_frequency': {i: np.random.randint(5, 25) for i in range(1, 13)},
        'red_expected': 20.0,
        'blue_expected': 10.0
    }
    
    # 绘制频率图表
    fig = viz.plot_frequency_chart(test_freq_data, 'DLT')
    plt.show()
