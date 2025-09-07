"""
历史数据爬取脚本
爬取从双色球和大乐透诞生之日起的所有历史数据
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crawler.historical_crawler import HistoricalCrawler
from data.database import LotteryDatabase


def progress_callback(current, total, message):
    """进度回调函数"""
    if total > 0:
        percent = (current / total) * 100
        print(f"[{percent:6.2f}%] {message}")
    else:
        print(f"[      ] {message}")


def crawl_all_historical_data():
    """爬取所有历史数据"""
    print("=" * 60)
    print("彩票历史数据爬取工具")
    print("=" * 60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 初始化爬虫和数据库
    crawler = HistoricalCrawler()
    db = LotteryDatabase()
    
    total_start_time = time.time()
    
    # 爬取大乐透数据
    print("1. 开始爬取大乐透历史数据...")
    print("   大乐透诞生日期: 2007-05-28")
    print("   预计数据量: 6000+ 期")
    print()
    
    dlt_start_time = time.time()
    dlt_data = crawler.crawl_all_historical_data('DLT', save_to_file=True)
    dlt_end_time = time.time()
    
    if not dlt_data.empty:
        print(f"   大乐透数据爬取完成: {len(dlt_data)} 条")
        print(f"   耗时: {dlt_end_time - dlt_start_time:.2f} 秒")
        
        # 保存到数据库
        print("   正在保存到数据库...")
        db.insert_dlt_data(dlt_data)
        print("   数据库保存完成")
    else:
        print("   大乐透数据爬取失败")
    
    print()
    
    # 爬取双色球数据
    print("2. 开始爬取双色球历史数据...")
    print("   双色球诞生日期: 2003-02-23")
    print("   预计数据量: 6000+ 期")
    print()
    
    ssq_start_time = time.time()
    ssq_data = crawler.crawl_all_historical_data('SSQ', save_to_file=True)
    ssq_end_time = time.time()
    
    if not ssq_data.empty:
        print(f"   双色球数据爬取完成: {len(ssq_data)} 条")
        print(f"   耗时: {ssq_end_time - ssq_start_time:.2f} 秒")
        
        # 保存到数据库
        print("   正在保存到数据库...")
        db.insert_ssq_data(ssq_data)
        print("   数据库保存完成")
    else:
        print("   双色球数据爬取失败")
    
    print()
    
    # 统计信息
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("=" * 60)
    print("爬取完成统计")
    print("=" * 60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {total_time:.2f} 秒")
    print()
    
    # 数据库统计
    stats = db.get_database_stats()
    print("数据库统计:")
    print(f"  大乐透数据: {stats['dlt_count']} 条")
    if stats['dlt_date_range'][0]:
        print(f"  大乐透日期范围: {stats['dlt_date_range'][0]} 到 {stats['dlt_date_range'][1]}")
    print(f"  双色球数据: {stats['ssq_count']} 条")
    if stats['ssq_date_range'][0]:
        print(f"  双色球日期范围: {stats['ssq_date_range'][0]} 到 {stats['ssq_date_range'][1]}")
    print(f"  预测记录: {stats['prediction_count']} 条")
    
    print()
    print("数据文件:")
    if not dlt_data.empty:
        dlt_filename = f"DLT_historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"  大乐透: {dlt_filename}")
    if not ssq_data.empty:
        ssq_filename = f"SSQ_historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"  双色球: {ssq_filename}")
    
    print()
    print("=" * 60)
    print("爬取任务完成！")
    print("=" * 60)


def crawl_single_lottery(lottery_type):
    """爬取单个彩票类型的数据"""
    print(f"开始爬取 {lottery_type} 历史数据...")
    
    crawler = HistoricalCrawler()
    db = LotteryDatabase()
    
    start_time = time.time()
    data = crawler.crawl_all_historical_data(lottery_type, save_to_file=True)
    end_time = time.time()
    
    if not data.empty:
        print(f"数据爬取完成: {len(data)} 条")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        
        # 保存到数据库
        if lottery_type == 'DLT':
            db.insert_dlt_data(data)
        else:
            db.insert_ssq_data(data)
        print("数据库保存完成")
        
        return data
    else:
        print("数据爬取失败")
        return None


def main():
    """主函数"""
    if len(sys.argv) > 1:
        lottery_type = sys.argv[1].upper()
        if lottery_type in ['DLT', 'SSQ']:
            crawl_single_lottery(lottery_type)
        else:
            print("无效的彩票类型，请使用 DLT 或 SSQ")
    else:
        crawl_all_historical_data()


if __name__ == "__main__":
    main()
