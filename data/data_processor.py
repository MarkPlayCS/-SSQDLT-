"""
数据处理模块
负责彩票数据的预处理、特征工程和数据转换
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re


class LotteryDataProcessor:
    def __init__(self):
        self.dlt_red_range = (1, 35)
        self.dlt_blue_range = (1, 12)
        self.ssq_red_range = (1, 33)
        self.ssq_blue_range = (1, 16)
    
    def preprocess_dlt_data(self, df):
        """预处理大乐透数据"""
        if df.empty:
            return df
        
        # 确保数据类型正确
        df = df.copy()
        # 处理日期格式，移除括号中的内容
        df['date'] = df['date'].astype(str).str.replace(r'\([^)]*\)', '', regex=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['period'] = df['period'].astype(str)
        
        # 排序
        df = df.sort_values('period')
        
        # 添加衍生特征
        df = self._add_dlt_features(df)
        
        return df
    
    def preprocess_ssq_data(self, df):
        """预处理双色球数据"""
        if df.empty:
            return df
        
        # 确保数据类型正确
        df = df.copy()
        # 处理日期格式，移除括号中的内容
        df['date'] = df['date'].astype(str).str.replace(r'\([^)]*\)', '', regex=True)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['period'] = df['period'].astype(str)
        
        # 排序
        df = df.sort_values('period')
        
        # 添加衍生特征
        df = self._add_ssq_features(df)
        
        return df
    
    def _add_dlt_features(self, df):
        """为大乐透数据添加特征"""
        # 红球特征
        red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
        blue_cols = ['blue_ball_1', 'blue_ball_2']
        
        # 红球和
        df['red_sum'] = df[red_cols].sum(axis=1)
        
        # 红球平均值
        df['red_mean'] = df[red_cols].mean(axis=1)
        
        # 红球标准差
        df['red_std'] = df[red_cols].std(axis=1)
        
        # 红球跨度
        df['red_span'] = df[red_cols].max(axis=1) - df[red_cols].min(axis=1)
        
        # 红球奇偶比
        red_odd = df[red_cols].apply(lambda x: sum(x % 2), axis=1)
        df['red_odd_ratio'] = red_odd / 5
        
        # 红球大小比 (大于17的号码)
        red_big = df[red_cols].apply(lambda x: sum(x > 17), axis=1)
        df['red_big_ratio'] = red_big / 5
        
        # 红球质数个数
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        red_prime = df[red_cols].apply(lambda x: sum(x.isin(primes)), axis=1)
        df['red_prime_count'] = red_prime
        
        # 蓝球和
        df['blue_sum'] = df[blue_cols].sum(axis=1)
        
        # 蓝球奇偶比
        blue_odd = df[blue_cols].apply(lambda x: sum(x % 2), axis=1)
        df['blue_odd_ratio'] = blue_odd / 2
        
        # 蓝球大小比
        blue_big = df[blue_cols].apply(lambda x: sum(x > 6), axis=1)
        df['blue_big_ratio'] = blue_big / 2
        
        # 连号个数
        df['red_consecutive'] = df[red_cols].apply(self._count_consecutive, axis=1)
        df['blue_consecutive'] = df[blue_cols].apply(self._count_consecutive, axis=1)
        
        # AC值 (算术复杂度)
        df['red_ac_value'] = df[red_cols].apply(self._calculate_ac_value, axis=1)
        
        # 和值尾数
        df['red_sum_tail'] = df['red_sum'] % 10
        
        # 期号特征
        df['period_num'] = df['period'].astype(int)
        df['period_mod_7'] = df['period_num'] % 7  # 星期几
        df['period_mod_30'] = df['period_num'] % 30  # 月内第几天
        
        return df
    
    def _add_ssq_features(self, df):
        """为双色球数据添加特征"""
        # 红球特征
        red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
        blue_col = 'blue_ball'
        
        # 红球和
        df['red_sum'] = df[red_cols].sum(axis=1)
        
        # 红球平均值
        df['red_mean'] = df[red_cols].mean(axis=1)
        
        # 红球标准差
        df['red_std'] = df[red_cols].std(axis=1)
        
        # 红球跨度
        df['red_span'] = df[red_cols].max(axis=1) - df[red_cols].min(axis=1)
        
        # 红球奇偶比
        red_odd = df[red_cols].apply(lambda x: sum(x % 2), axis=1)
        df['red_odd_ratio'] = red_odd / 6
        
        # 红球大小比 (大于16的号码)
        red_big = df[red_cols].apply(lambda x: sum(x > 16), axis=1)
        df['red_big_ratio'] = red_big / 6
        
        # 红球质数个数
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        red_prime = df[red_cols].apply(lambda x: sum(x.isin(primes)), axis=1)
        df['red_prime_count'] = red_prime
        
        # 蓝球特征
        df['blue_odd'] = df[blue_col] % 2
        df['blue_big'] = (df[blue_col] > 8).astype(int)
        df['blue_prime'] = df[blue_col].isin(primes).astype(int)
        
        # 连号个数
        df['red_consecutive'] = df[red_cols].apply(self._count_consecutive, axis=1)
        
        # AC值 (算术复杂度)
        df['red_ac_value'] = df[red_cols].apply(self._calculate_ac_value, axis=1)
        
        # 和值尾数
        df['red_sum_tail'] = df['red_sum'] % 10
        
        # 期号特征
        df['period_num'] = df['period'].astype(int)
        df['period_mod_7'] = df['period_num'] % 7  # 星期几
        df['period_mod_30'] = df['period_num'] % 30  # 月内第几天
        
        return df
    
    def _count_consecutive(self, numbers):
        """计算连号个数"""
        sorted_nums = sorted(numbers)
        consecutive_count = 0
        for i in range(len(sorted_nums) - 1):
            if sorted_nums[i+1] - sorted_nums[i] == 1:
                consecutive_count += 1
        return consecutive_count
    
    def _calculate_ac_value(self, numbers):
        """计算AC值 (算术复杂度)"""
        sorted_nums = sorted(numbers)
        differences = []
        for i in range(len(sorted_nums)):
            for j in range(i+1, len(sorted_nums)):
                diff = abs(sorted_nums[j] - sorted_nums[i])
                if diff not in differences:
                    differences.append(diff)
        return len(differences) - len(numbers) + 1
    
    def create_frequency_features(self, df, lottery_type, window_size=10):
        """创建频率特征"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball'] if lottery_type == 'SSQ' else ['blue_ball_1', 'blue_ball_2']
        
        # 计算每个号码的出现频率
        for col in red_cols + blue_cols:
            freq_col = f'{col}_freq_{window_size}'
            df[freq_col] = df[col].rolling(window=window_size, min_periods=1).apply(
                lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0
            )
        
        return df
    
    def create_trend_features(self, df, lottery_type):
        """创建趋势特征"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
        
        # 计算和值趋势
        df['red_sum_trend_3'] = df['red_sum'].rolling(window=3, min_periods=1).mean()
        df['red_sum_trend_5'] = df['red_sum'].rolling(window=5, min_periods=1).mean()
        df['red_sum_trend_10'] = df['red_sum'].rolling(window=10, min_periods=1).mean()
        
        # 计算跨度趋势
        df['red_span_trend_3'] = df['red_span'].rolling(window=3, min_periods=1).mean()
        df['red_span_trend_5'] = df['red_span'].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def create_sequence_features(self, df, lottery_type, sequence_length=5):
        """创建序列特征"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
        
        # 创建历史序列特征
        for i in range(1, sequence_length + 1):
            df[f'red_sum_lag_{i}'] = df['red_sum'].shift(i)
            df[f'red_span_lag_{i}'] = df['red_span'].shift(i)
            df[f'red_odd_ratio_lag_{i}'] = df['red_odd_ratio'].shift(i)
        
        return df
    
    def prepare_ml_data(self, df, lottery_type, target_cols=None):
        """准备机器学习数据"""
        if target_cols is None:
            if lottery_type == 'DLT':
                target_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 
                             'blue_ball_1', 'blue_ball_2']
            else:  # SSQ
                target_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 
                             'red_ball_6', 'blue_ball']
        
        # 选择特征列
        feature_cols = [col for col in df.columns if col not in target_cols + ['period', 'date', 'lottery_type']]
        
        # 移除包含NaN的行
        df_clean = df.dropna()
        
        return df_clean[feature_cols], df_clean[target_cols]
    
    def normalize_data(self, X, method='minmax'):
        """数据标准化"""
        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            return scaler.fit_transform(X), scaler
        elif method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            return scaler.fit_transform(X), scaler
        else:
            return X, None


if __name__ == "__main__":
    # 测试数据处理功能
    processor = LotteryDataProcessor()
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'period': ['23001', '23002', '23003'],
        'date': ['2023-01-01', '2023-01-04', '2023-01-07'],
        'red_ball_1': [1, 5, 10],
        'red_ball_2': [10, 15, 20],
        'red_ball_3': [20, 25, 30],
        'red_ball_4': [30, 35, 5],
        'red_ball_5': [12, 18, 25],
        'blue_ball_1': [1, 3, 5],
        'blue_ball_2': [5, 7, 9],
        'lottery_type': ['DLT', 'DLT', 'DLT']
    })
    
    # 预处理数据
    processed_data = processor.preprocess_dlt_data(test_data)
    print("预处理后的大乐透数据:")
    print(processed_data[['period', 'red_sum', 'red_span', 'red_odd_ratio', 'red_ac_value']])
    
    # 准备机器学习数据
    X, y = processor.prepare_ml_data(processed_data, 'DLT')
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"目标矩阵形状: {y.shape}")
