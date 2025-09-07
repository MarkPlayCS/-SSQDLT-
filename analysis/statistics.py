"""
统计分析模块
提供各种统计分析方法用于彩票数据分析
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, normaltest
import warnings
warnings.filterwarnings('ignore')


class LotteryStatistics:
    def __init__(self):
        self.dlt_red_range = (1, 35)
        self.dlt_blue_range = (1, 12)
        self.ssq_red_range = (1, 33)
        self.ssq_blue_range = (1, 16)
    
    def frequency_analysis(self, df, lottery_type):
        """频率分析"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
            red_range = self.dlt_red_range
            blue_range = self.dlt_blue_range
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
            red_range = self.ssq_red_range
            blue_range = self.ssq_blue_range
        
        # 红球频率统计
        red_freq = {}
        for col in red_cols:
            for num in df[col]:
                red_freq[num] = red_freq.get(num, 0) + 1
        
        # 蓝球频率统计
        blue_freq = {}
        for col in blue_cols:
            for num in df[col]:
                blue_freq[num] = blue_freq.get(num, 0) + 1
        
        # 计算期望频率
        total_draws = len(df)
        red_expected = total_draws * len(red_cols) / (red_range[1] - red_range[0] + 1)
        blue_expected = total_draws * len(blue_cols) / (blue_range[1] - blue_range[0] + 1)
        
        # 计算偏差
        red_deviation = {}
        for num in range(red_range[0], red_range[1] + 1):
            observed = red_freq.get(num, 0)
            red_deviation[num] = observed - red_expected
        
        blue_deviation = {}
        for num in range(blue_range[0], blue_range[1] + 1):
            observed = blue_freq.get(num, 0)
            blue_deviation[num] = observed - blue_expected
        
        return {
            'red_frequency': red_freq,
            'blue_frequency': blue_freq,
            'red_deviation': red_deviation,
            'blue_deviation': blue_deviation,
            'red_expected': red_expected,
            'blue_expected': blue_expected,
            'total_draws': total_draws
        }
    
    def hot_cold_analysis(self, df, lottery_type, window_size=20):
        """冷热号分析"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
            red_range = self.dlt_red_range
            blue_range = self.dlt_blue_range
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
            red_range = self.ssq_red_range
            blue_range = self.ssq_blue_range
        
        # 最近N期的频率
        recent_df = df.tail(window_size)
        
        red_hot_cold = {}
        for num in range(red_range[0], red_range[1] + 1):
            count = 0
            for col in red_cols:
                count += (recent_df[col] == num).sum()
            red_hot_cold[num] = count
        
        blue_hot_cold = {}
        for num in range(blue_range[0], blue_range[1] + 1):
            count = 0
            for col in blue_cols:
                count += (recent_df[col] == num).sum()
            blue_hot_cold[num] = count
        
        # 排序
        red_sorted = sorted(red_hot_cold.items(), key=lambda x: x[1], reverse=True)
        blue_sorted = sorted(blue_hot_cold.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'red_hot_cold': red_hot_cold,
            'blue_hot_cold': blue_hot_cold,
            'red_hot_numbers': [x[0] for x in red_sorted[:10]],
            'red_cold_numbers': [x[0] for x in red_sorted[-10:]],
            'blue_hot_numbers': [x[0] for x in blue_sorted[:5]],
            'blue_cold_numbers': [x[0] for x in blue_sorted[-5:]],
            'window_size': window_size
        }
    
    def gap_analysis(self, df, lottery_type):
        """间隔分析 - 分析号码出现的间隔"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
            red_range = self.dlt_red_range
            blue_range = self.dlt_blue_range
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
            red_range = self.ssq_red_range
            blue_range = self.ssq_blue_range
        
        red_gaps = {}
        blue_gaps = {}
        
        # 计算每个号码的出现间隔
        for num in range(red_range[0], red_range[1] + 1):
            appearances = []
            for idx, row in df.iterrows():
                for col in red_cols:
                    if row[col] == num:
                        appearances.append(idx)
            
            if len(appearances) > 1:
                gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                red_gaps[num] = {
                    'gaps': gaps,
                    'avg_gap': np.mean(gaps),
                    'max_gap': max(gaps),
                    'min_gap': min(gaps),
                    'last_appearance': len(df) - 1 - appearances[-1] if appearances else None
                }
        
        for num in range(blue_range[0], blue_range[1] + 1):
            appearances = []
            for idx, row in df.iterrows():
                for col in blue_cols:
                    if row[col] == num:
                        appearances.append(idx)
            
            if len(appearances) > 1:
                gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                blue_gaps[num] = {
                    'gaps': gaps,
                    'avg_gap': np.mean(gaps),
                    'max_gap': max(gaps),
                    'min_gap': min(gaps),
                    'last_appearance': len(df) - 1 - appearances[-1] if appearances else None
                }
        
        return {
            'red_gaps': red_gaps,
            'blue_gaps': blue_gaps
        }
    
    def sum_analysis(self, df, lottery_type):
        """和值分析"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
        
        # 计算和值
        red_sums = df[red_cols].sum(axis=1)
        blue_sums = df[blue_cols].sum(axis=1)
        
        # 统计信息
        red_sum_stats = {
            'mean': red_sums.mean(),
            'std': red_sums.std(),
            'min': red_sums.min(),
            'max': red_sums.max(),
            'median': red_sums.median(),
            'mode': red_sums.mode().iloc[0] if not red_sums.mode().empty else None
        }
        
        blue_sum_stats = {
            'mean': blue_sums.mean(),
            'std': blue_sums.std(),
            'min': blue_sums.min(),
            'max': blue_sums.max(),
            'median': blue_sums.median(),
            'mode': blue_sums.mode().iloc[0] if not blue_sums.mode().empty else None
        }
        
        # 和值分布
        red_sum_dist = red_sums.value_counts().sort_index()
        blue_sum_dist = blue_sums.value_counts().sort_index()
        
        return {
            'red_sum_stats': red_sum_stats,
            'blue_sum_stats': blue_sum_stats,
            'red_sum_distribution': red_sum_dist,
            'blue_sum_distribution': blue_sum_dist,
            'red_sums': red_sums.tolist(),
            'blue_sums': blue_sums.tolist()
        }
    
    def odd_even_analysis(self, df, lottery_type):
        """奇偶分析"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
        
        # 计算每期奇偶比例
        red_odd_counts = []
        blue_odd_counts = []
        
        for _, row in df.iterrows():
            red_odd = sum(1 for col in red_cols if row[col] % 2 == 1)
            blue_odd = sum(1 for col in blue_cols if row[col] % 2 == 1)
            
            red_odd_counts.append(red_odd)
            blue_odd_counts.append(blue_odd)
        
        # 统计奇偶分布
        red_odd_dist = Counter(red_odd_counts)
        blue_odd_dist = Counter(blue_odd_counts)
        
        return {
            'red_odd_distribution': red_odd_dist,
            'blue_odd_distribution': blue_odd_dist,
            'red_odd_counts': red_odd_counts,
            'blue_odd_counts': blue_odd_counts
        }
    
    def size_analysis(self, df, lottery_type):
        """大小号分析"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
            red_mid = 18  # 35/2 + 0.5
            blue_mid = 6.5  # 12/2 + 0.5
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
            red_mid = 17  # 33/2 + 0.5
            blue_mid = 8.5  # 16/2 + 0.5
        
        # 计算每期大小号比例
        red_big_counts = []
        blue_big_counts = []
        
        for _, row in df.iterrows():
            red_big = sum(1 for col in red_cols if row[col] > red_mid)
            blue_big = sum(1 for col in blue_cols if row[col] > blue_mid)
            
            red_big_counts.append(red_big)
            blue_big_counts.append(blue_big)
        
        # 统计大小号分布
        red_big_dist = Counter(red_big_counts)
        blue_big_dist = Counter(blue_big_counts)
        
        return {
            'red_big_distribution': red_big_dist,
            'blue_big_distribution': blue_big_dist,
            'red_big_counts': red_big_counts,
            'blue_big_counts': blue_big_counts,
            'red_mid_point': red_mid,
            'blue_mid_point': blue_mid
        }
    
    def consecutive_analysis(self, df, lottery_type):
        """连号分析"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
        
        red_consecutive_counts = []
        blue_consecutive_counts = []
        
        for _, row in df.iterrows():
            # 红球连号
            red_numbers = sorted([row[col] for col in red_cols])
            red_consecutive = 0
            for i in range(len(red_numbers) - 1):
                if red_numbers[i+1] - red_numbers[i] == 1:
                    red_consecutive += 1
            red_consecutive_counts.append(red_consecutive)
            
            # 蓝球连号
            blue_numbers = sorted([row[col] for col in blue_cols])
            blue_consecutive = 0
            for i in range(len(blue_numbers) - 1):
                if blue_numbers[i+1] - blue_numbers[i] == 1:
                    blue_consecutive += 1
            blue_consecutive_counts.append(blue_consecutive)
        
        # 统计连号分布
        red_consecutive_dist = Counter(red_consecutive_counts)
        blue_consecutive_dist = Counter(blue_consecutive_counts)
        
        return {
            'red_consecutive_distribution': red_consecutive_dist,
            'blue_consecutive_distribution': blue_consecutive_dist,
            'red_consecutive_counts': red_consecutive_counts,
            'blue_consecutive_counts': blue_consecutive_counts
        }
    
    def ac_value_analysis(self, df, lottery_type):
        """AC值分析 (算术复杂度)"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
        
        ac_values = []
        
        for _, row in df.iterrows():
            numbers = sorted([row[col] for col in red_cols])
            differences = []
            
            # 计算所有差值
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    diff = abs(numbers[j] - numbers[i])
                    if diff not in differences:
                        differences.append(diff)
            
            # AC值 = 不同差值的个数 - 号码个数 + 1
            ac_value = len(differences) - len(numbers) + 1
            ac_values.append(ac_value)
        
        # 统计AC值分布
        ac_dist = Counter(ac_values)
        
        return {
            'ac_distribution': ac_dist,
            'ac_values': ac_values,
            'mean_ac': np.mean(ac_values),
            'std_ac': np.std(ac_values)
        }
    
    def trend_analysis(self, df, lottery_type, window_size=10):
        """趋势分析"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
        
        # 计算和值
        red_sums = df[red_cols].sum(axis=1)
        
        # 移动平均
        moving_avg = red_sums.rolling(window=window_size, min_periods=1).mean()
        
        # 趋势方向
        trend_direction = []
        for i in range(1, len(moving_avg)):
            if moving_avg.iloc[i] > moving_avg.iloc[i-1]:
                trend_direction.append(1)  # 上升
            elif moving_avg.iloc[i] < moving_avg.iloc[i-1]:
                trend_direction.append(-1)  # 下降
            else:
                trend_direction.append(0)  # 平稳
        
        # 计算趋势强度
        trend_strength = []
        for i in range(window_size, len(red_sums)):
            recent_data = red_sums.iloc[i-window_size:i]
            slope, _, _, _, _ = stats.linregress(range(len(recent_data)), recent_data)
            trend_strength.append(abs(slope))
        
        return {
            'moving_average': moving_avg.tolist(),
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'current_trend': trend_direction[-1] if trend_direction else 0,
            'avg_trend_strength': np.mean(trend_strength) if trend_strength else 0
        }
    
    def comprehensive_analysis(self, df, lottery_type):
        """综合分析"""
        results = {}
        
        # 各种分析
        results['frequency'] = self.frequency_analysis(df, lottery_type)
        results['hot_cold'] = self.hot_cold_analysis(df, lottery_type)
        results['gap'] = self.gap_analysis(df, lottery_type)
        results['sum'] = self.sum_analysis(df, lottery_type)
        results['odd_even'] = self.odd_even_analysis(df, lottery_type)
        results['size'] = self.size_analysis(df, lottery_type)
        results['consecutive'] = self.consecutive_analysis(df, lottery_type)
        results['ac_value'] = self.ac_value_analysis(df, lottery_type)
        results['trend'] = self.trend_analysis(df, lottery_type)
        
        return results


if __name__ == "__main__":
    # 测试统计分析功能
    stats_analyzer = LotteryStatistics()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'red_ball_1': [1, 5, 10, 15, 20],
        'red_ball_2': [2, 6, 11, 16, 21],
        'red_ball_3': [3, 7, 12, 17, 22],
        'red_ball_4': [4, 8, 13, 18, 23],
        'red_ball_5': [5, 9, 14, 19, 24],
        'blue_ball_1': [1, 2, 3, 4, 5],
        'blue_ball_2': [6, 7, 8, 9, 10]
    })
    
    # 频率分析
    freq_result = stats_analyzer.frequency_analysis(test_data, 'DLT')
    print("频率分析结果:")
    print(f"红球频率: {freq_result['red_frequency']}")
    print(f"蓝球频率: {freq_result['blue_frequency']}")
    
    # 冷热号分析
    hot_cold_result = stats_analyzer.hot_cold_analysis(test_data, 'DLT')
    print(f"\n热号: {hot_cold_result['red_hot_numbers']}")
    print(f"冷号: {hot_cold_result['red_cold_numbers']}")
    
    # 和值分析
    sum_result = stats_analyzer.sum_analysis(test_data, 'DLT')
    print(f"\n红球和值统计: {sum_result['red_sum_stats']}")
