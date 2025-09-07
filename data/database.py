"""
数据库管理模块
负责彩票数据的存储、查询和管理
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
import os


class LotteryDatabase:
    def __init__(self, db_path="lottery_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建大乐透数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dlt_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT UNIQUE NOT NULL,
                date TEXT NOT NULL,
                red_ball_1 INTEGER NOT NULL,
                red_ball_2 INTEGER NOT NULL,
                red_ball_3 INTEGER NOT NULL,
                red_ball_4 INTEGER NOT NULL,
                red_ball_5 INTEGER NOT NULL,
                blue_ball_1 INTEGER NOT NULL,
                blue_ball_2 INTEGER NOT NULL,
                lottery_type TEXT DEFAULT 'DLT',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建双色球数据表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ssq_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period TEXT UNIQUE NOT NULL,
                date TEXT NOT NULL,
                red_ball_1 INTEGER NOT NULL,
                red_ball_2 INTEGER NOT NULL,
                red_ball_3 INTEGER NOT NULL,
                red_ball_4 INTEGER NOT NULL,
                red_ball_5 INTEGER NOT NULL,
                red_ball_6 INTEGER NOT NULL,
                blue_ball INTEGER NOT NULL,
                lottery_type TEXT DEFAULT 'SSQ',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建统计分析结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lottery_type TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                period TEXT,
                result_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建预测结果表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lottery_type TEXT NOT NULL,
                model_type TEXT NOT NULL,
                prediction_data TEXT NOT NULL,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dlt_period ON dlt_data(period)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dlt_date ON dlt_data(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ssq_period ON ssq_data(period)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ssq_date ON ssq_data(date)')
        
        conn.commit()
        conn.close()
    
    def insert_dlt_data(self, df):
        """插入大乐透数据"""
        conn = sqlite3.connect(self.db_path)
        
        # 使用pandas的to_sql方法，如果存在则替换
        df.to_sql('dlt_data', conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
    
    def insert_ssq_data(self, df):
        """插入双色球数据"""
        conn = sqlite3.connect(self.db_path)
        
        # 使用pandas的to_sql方法，如果存在则替换
        df.to_sql('ssq_data', conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
    
    def get_dlt_data(self, limit=None, start_date=None, end_date=None):
        """获取大乐透数据"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM dlt_data"
        conditions = []
        params = []
        
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY period DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_ssq_data(self, limit=None, start_date=None, end_date=None):
        """获取双色球数据"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM ssq_data"
        conditions = []
        params = []
        
        if start_date:
            conditions.append("date >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("date <= ?")
            params.append(end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY period DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_latest_period(self, lottery_type):
        """获取最新期号"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if lottery_type == 'DLT':
            cursor.execute("SELECT period FROM dlt_data ORDER BY period DESC LIMIT 1")
        elif lottery_type == 'SSQ':
            cursor.execute("SELECT period FROM ssq_data ORDER BY period DESC LIMIT 1")
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def save_analysis_result(self, lottery_type, analysis_type, period, result_data):
        """保存统计分析结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results (lottery_type, analysis_type, period, result_data)
            VALUES (?, ?, ?, ?)
        ''', (lottery_type, analysis_type, period, json.dumps(result_data)))
        
        conn.commit()
        conn.close()
    
    def get_analysis_results(self, lottery_type, analysis_type=None):
        """获取统计分析结果"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM analysis_results WHERE lottery_type = ?"
        params = [lottery_type]
        
        if analysis_type:
            query += " AND analysis_type = ?"
            params.append(analysis_type)
        
        query += " ORDER BY created_at DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def save_prediction(self, lottery_type, model_type, prediction_data, confidence_score=None):
        """保存预测结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (lottery_type, model_type, prediction_data, confidence_score)
            VALUES (?, ?, ?, ?)
        ''', (lottery_type, model_type, json.dumps(prediction_data), confidence_score))
        
        conn.commit()
        conn.close()
    
    def get_predictions(self, lottery_type, model_type=None, limit=10):
        """获取预测结果"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM predictions WHERE lottery_type = ?"
        params = [lottery_type]
        
        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def clear_old_data(self, days=365):
        """清理旧数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 清理超过指定天数的数据
        cursor.execute('''
            DELETE FROM dlt_data 
            WHERE date < date('now', '-{} days')
        '''.format(days))
        
        cursor.execute('''
            DELETE FROM ssq_data 
            WHERE date < date('now', '-{} days')
        '''.format(days))
        
        conn.commit()
        conn.close()
    
    def get_database_stats(self):
        """获取数据库统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # 大乐透数据统计
        cursor.execute("SELECT COUNT(*) FROM dlt_data")
        stats['dlt_count'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM dlt_data")
        dlt_dates = cursor.fetchone()
        stats['dlt_date_range'] = dlt_dates
        
        # 双色球数据统计
        cursor.execute("SELECT COUNT(*) FROM ssq_data")
        stats['ssq_count'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(date), MAX(date) FROM ssq_data")
        ssq_dates = cursor.fetchone()
        stats['ssq_date_range'] = ssq_dates
        
        # 预测结果统计
        cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['prediction_count'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats


if __name__ == "__main__":
    # 测试数据库功能
    db = LotteryDatabase()
    
    # 创建测试数据
    test_dlt_data = pd.DataFrame({
        'period': ['23001', '23002'],
        'date': ['2023-01-01', '2023-01-04'],
        'red_ball_1': [1, 5],
        'red_ball_2': [10, 15],
        'red_ball_3': [20, 25],
        'red_ball_4': [30, 35],
        'red_ball_5': [12, 18],
        'blue_ball_1': [1, 3],
        'blue_ball_2': [5, 7],
        'lottery_type': ['DLT', 'DLT']
    })
    
    # 插入测试数据
    db.insert_dlt_data(test_dlt_data)
    
    # 查询数据
    data = db.get_dlt_data(limit=5)
    print("大乐透数据:")
    print(data)
    
    # 获取统计信息
    stats = db.get_database_stats()
    print("\n数据库统计:")
    print(stats)
