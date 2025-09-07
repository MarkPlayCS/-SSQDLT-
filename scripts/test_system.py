"""
系统测试脚本
测试彩票预测软件的各个模块
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_crawler():
    """测试爬虫模块"""
    print("=" * 50)
    print("测试爬虫模块")
    print("=" * 50)
    
    try:
        from crawler.dlt_crawler import DLTCrawler
        from crawler.ssq_crawler import SSQCrawler
        
        # 测试大乐透爬虫
        print("测试大乐透爬虫...")
        dlt_crawler = DLTCrawler()
        dlt_data = dlt_crawler.get_lottery_data()
        print(f"大乐透数据: {len(dlt_data)} 条")
        
        # 测试双色球爬虫
        print("测试双色球爬虫...")
        ssq_crawler = SSQCrawler()
        ssq_data = ssq_crawler.get_lottery_data()
        print(f"双色球数据: {len(ssq_data)} 条")
        
        print("✓ 爬虫模块测试通过")
        return dlt_data, ssq_data
        
    except Exception as e:
        print(f"✗ 爬虫模块测试失败: {e}")
        return None, None

def test_database(dlt_data, ssq_data):
    """测试数据库模块"""
    print("\n" + "=" * 50)
    print("测试数据库模块")
    print("=" * 50)
    
    try:
        from data.database import LotteryDatabase
        
        db = LotteryDatabase()
        print("数据库初始化成功")
        
        # 插入数据
        if dlt_data is not None and not dlt_data.empty:
            db.insert_dlt_data(dlt_data)
            print("大乐透数据插入成功")
        
        if ssq_data is not None and not ssq_data.empty:
            db.insert_ssq_data(ssq_data)
            print("双色球数据插入成功")
        
        # 查询数据
        dlt_query = db.get_dlt_data(limit=5)
        ssq_query = db.get_ssq_data(limit=5)
        
        print(f"查询到大乐透数据: {len(dlt_query)} 条")
        print(f"查询到双色球数据: {len(ssq_query)} 条")
        
        # 获取统计信息
        stats = db.get_database_stats()
        print(f"数据库统计: {stats}")
        
        print("✓ 数据库模块测试通过")
        return db, dlt_query, ssq_query
        
    except Exception as e:
        print(f"✗ 数据库模块测试失败: {e}")
        return None, None, None

def test_data_processor(dlt_data, ssq_data):
    """测试数据处理模块"""
    print("\n" + "=" * 50)
    print("测试数据处理模块")
    print("=" * 50)
    
    try:
        from data.data_processor import LotteryDataProcessor
        
        processor = LotteryDataProcessor()
        print("数据处理器初始化成功")
        
        # 处理大乐透数据
        if dlt_data is not None and not dlt_data.empty:
            processed_dlt = processor.preprocess_dlt_data(dlt_data)
            print(f"大乐透数据预处理完成，特征数: {len(processed_dlt.columns)}")
            
            # 准备机器学习数据
            X_dlt, y_dlt = processor.prepare_ml_data(processed_dlt, 'DLT')
            print(f"大乐透ML数据准备完成: X{X_dlt.shape}, y{y_dlt.shape}")
        
        # 处理双色球数据
        if ssq_data is not None and not ssq_data.empty:
            processed_ssq = processor.preprocess_ssq_data(ssq_data)
            print(f"双色球数据预处理完成，特征数: {len(processed_ssq.columns)}")
            
            # 准备机器学习数据
            X_ssq, y_ssq = processor.prepare_ml_data(processed_ssq, 'SSQ')
            print(f"双色球ML数据准备完成: X{X_ssq.shape}, y{y_ssq.shape}")
        
        print("✓ 数据处理模块测试通过")
        return processed_dlt if dlt_data is not None and not dlt_data.empty else None, \
               processed_ssq if ssq_data is not None and not ssq_data.empty else None
        
    except Exception as e:
        print(f"✗ 数据处理模块测试失败: {e}")
        return None, None

def test_statistics(processed_dlt, processed_ssq):
    """测试统计分析模块"""
    print("\n" + "=" * 50)
    print("测试统计分析模块")
    print("=" * 50)
    
    try:
        from analysis.statistics import LotteryStatistics
        
        stats = LotteryStatistics()
        print("统计分析器初始化成功")
        
        # 分析大乐透数据
        if processed_dlt is not None and not processed_dlt.empty:
            dlt_analysis = stats.comprehensive_analysis(processed_dlt, 'DLT')
            print("大乐透统计分析完成")
            print(f"  频率分析: {len(dlt_analysis['frequency']['red_frequency'])} 个红球号码")
            print(f"  热号: {dlt_analysis['hot_cold']['red_hot_numbers'][:3]}")
            print(f"  冷号: {dlt_analysis['hot_cold']['red_cold_numbers'][-3:]}")
        
        # 分析双色球数据
        if processed_ssq is not None and not processed_ssq.empty:
            ssq_analysis = stats.comprehensive_analysis(processed_ssq, 'SSQ')
            print("双色球统计分析完成")
            print(f"  频率分析: {len(ssq_analysis['frequency']['red_frequency'])} 个红球号码")
            print(f"  热号: {ssq_analysis['hot_cold']['red_hot_numbers'][:3]}")
            print(f"  冷号: {ssq_analysis['hot_cold']['red_cold_numbers'][-3:]}")
        
        print("✓ 统计分析模块测试通过")
        
    except Exception as e:
        print(f"✗ 统计分析模块测试失败: {e}")

def test_ml_predictor(processed_dlt, processed_ssq):
    """测试机器学习预测模块"""
    print("\n" + "=" * 50)
    print("测试机器学习预测模块")
    print("=" * 50)
    
    try:
        from ml.traditional_ml import TraditionalMLPredictor
        from data.data_processor import LotteryDataProcessor
        
        ml_predictor = TraditionalMLPredictor()
        processor = LotteryDataProcessor()
        print("机器学习预测器初始化成功")
        
        # 测试大乐透预测
        if processed_dlt is not None and len(processed_dlt) >= 50:
            X_dlt, y_dlt = processor.prepare_ml_data(processed_dlt, 'DLT')
            print(f"大乐透ML数据: X{X_dlt.shape}, y{y_dlt.shape}")
            
            # 训练模型
            ml_predictor.train_models(X_dlt, y_dlt, 'DLT')
            print("大乐透ML模型训练完成")
            
            # 预测
            prediction = ml_predictor.predict_next_period(processed_dlt, 'DLT')
            if prediction:
                print(f"大乐透预测结果: 红球{prediction['red_balls']}, 蓝球{prediction['blue_balls']}")
        
        # 测试双色球预测
        if processed_ssq is not None and len(processed_ssq) >= 50:
            X_ssq, y_ssq = processor.prepare_ml_data(processed_ssq, 'SSQ')
            print(f"双色球ML数据: X{X_ssq.shape}, y{y_ssq.shape}")
            
            # 训练模型
            ml_predictor.train_models(X_ssq, y_ssq, 'SSQ')
            print("双色球ML模型训练完成")
            
            # 预测
            prediction = ml_predictor.predict_next_period(processed_ssq, 'SSQ')
            if prediction:
                print(f"双色球预测结果: 红球{prediction['red_balls']}, 蓝球{prediction['blue_balls']}")
        
        print("✓ 机器学习预测模块测试通过")
        
    except Exception as e:
        print(f"✗ 机器学习预测模块测试失败: {e}")

def test_helpers():
    """测试辅助工具模块"""
    print("\n" + "=" * 50)
    print("测试辅助工具模块")
    print("=" * 50)
    
    try:
        from utils.helpers import validate_lottery_numbers, calculate_win_probability, generate_random_numbers
        
        # 测试号码验证
        test_numbers = {
            'red_balls': [1, 5, 10, 15, 20],
            'blue_balls': [1, 5]
        }
        
        is_valid, message = validate_lottery_numbers(test_numbers, 'DLT')
        print(f"号码验证: {is_valid}, {message}")
        
        # 测试概率计算
        prob_info = calculate_win_probability(test_numbers, 'DLT')
        print(f"中奖概率: {prob_info['description']}")
        
        # 测试随机号码生成
        random_nums = generate_random_numbers('DLT', 3)
        print(f"随机号码生成: {len(random_nums)} 组")
        
        print("✓ 辅助工具模块测试通过")
        
    except Exception as e:
        print(f"✗ 辅助工具模块测试失败: {e}")

def main():
    """主测试函数"""
    print("彩票预测软件系统测试")
    print("=" * 60)
    
    # 测试各个模块
    dlt_data, ssq_data = test_crawler()
    db, dlt_query, ssq_query = test_database(dlt_data, ssq_data)
    processed_dlt, processed_ssq = test_data_processor(dlt_data, ssq_data)
    test_statistics(processed_dlt, processed_ssq)
    test_ml_predictor(processed_dlt, processed_ssq)
    test_helpers()
    
    print("\n" + "=" * 60)
    print("系统测试完成！")
    print("=" * 60)
    
    # 显示系统信息
    try:
        from utils.helpers import get_system_info
        info = get_system_info()
        print("\n系统信息:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except:
        print("无法获取系统信息")

if __name__ == "__main__":
    main()
