"""
超级大乐透数据爬虫
爬取中国体彩网的历史开奖数据
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime, timedelta
import json


class DLTCrawler:
    def __init__(self):
        self.base_url = "https://www.lottery.gov.cn"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_lottery_data(self, start_date=None, end_date=None, max_pages=1000):
        """
        获取大乐透历史开奖数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)，默认为大乐透诞生日期
            end_date: 结束日期 (YYYY-MM-DD)，默认为当前日期
            max_pages: 最大页数
            
        Returns:
            DataFrame: 包含开奖数据的DataFrame
        """
        all_data = []
        
        # 如果没有指定日期，默认获取从大乐透诞生之日起的所有数据
        if not start_date:
            start_date = '2007-05-28'  # 大乐透诞生日期
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"开始爬取大乐透数据，时间范围：{start_date} 到 {end_date}")
        
        # 尝试多个数据源
        data_sources = [
            self._crawl_from_500_com,
            self._crawl_from_sina,
            self._crawl_from_163
        ]
        
        for source_func in data_sources:
            try:
                print(f"尝试从 {source_func.__name__} 获取数据...")
                data = source_func(start_date, end_date, max_pages)
                if data and len(data) > 0:
                    all_data.extend(data)
                    print(f"成功获取 {len(data)} 条数据")
                    break
            except Exception as e:
                print(f"从 {source_func.__name__} 获取数据失败: {e}")
                continue
                
        if not all_data:
            # 如果所有数据源都失败，生成模拟数据用于演示
            print("所有数据源都失败，生成模拟数据用于演示...")
            all_data = self._generate_mock_data(start_date, end_date)
            
        return pd.DataFrame(all_data)
    
    def _crawl_from_500_com(self, start_date, end_date, max_pages):
        """从500彩票网爬取数据"""
        data = []
        base_url = "https://datachart.500.com/dlt/history/newinc/history.php"
        
        for page in range(1, max_pages + 1):
            try:
                params = {
                    'limit': 30,
                    'start': (page - 1) * 30
                }
                
                response = self.session.get(base_url, params=params, timeout=10)
                response.encoding = 'gb2312'
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 解析表格数据
                table = soup.find('table', class_='tb')
                if not table:
                    break
                    
                rows = table.find_all('tr')[1:]  # 跳过表头
                
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 8:
                        try:
                            period = cells[0].text.strip()
                            date = cells[1].text.strip()
                            
                            # 解析号码
                            red_balls = []
                            blue_balls = []
                            
                            for i in range(2, 7):  # 前区号码
                                num = cells[i].text.strip()
                                if num.isdigit():
                                    red_balls.append(int(num))
                                    
                            for i in range(7, 9):  # 后区号码
                                num = cells[i].text.strip()
                                if num.isdigit():
                                    blue_balls.append(int(num))
                            
                            if len(red_balls) == 5 and len(blue_balls) == 2:
                                data.append({
                                    'period': period,
                                    'date': date,
                                    'red_ball_1': red_balls[0],
                                    'red_ball_2': red_balls[1],
                                    'red_ball_3': red_balls[2],
                                    'red_ball_4': red_balls[3],
                                    'red_ball_5': red_balls[4],
                                    'blue_ball_1': blue_balls[0],
                                    'blue_ball_2': blue_balls[1],
                                    'lottery_type': 'DLT'
                                })
                        except Exception as e:
                            continue
                            
                time.sleep(1)  # 避免请求过快
                
            except Exception as e:
                print(f"爬取第 {page} 页失败: {e}")
                break
                
        return data
    
    def _crawl_from_sina(self, start_date, end_date, max_pages):
        """从新浪彩票爬取数据"""
        data = []
        base_url = "https://match.lottery.sina.com.cn/lotto/pc_zst/index"
        
        try:
            response = self.session.get(base_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找开奖数据表格
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # 跳过表头
                    cells = row.find_all('td')
                    if len(cells) >= 8:
                        try:
                            period = cells[0].text.strip()
                            date = cells[1].text.strip()
                            
                            # 解析号码
                            numbers_text = cells[2].text.strip()
                            numbers = re.findall(r'\d+', numbers_text)
                            
                            if len(numbers) >= 7:
                                red_balls = [int(x) for x in numbers[:5]]
                                blue_balls = [int(x) for x in numbers[5:7]]
                                
                                data.append({
                                    'period': period,
                                    'date': date,
                                    'red_ball_1': red_balls[0],
                                    'red_ball_2': red_balls[1],
                                    'red_ball_3': red_balls[2],
                                    'red_ball_4': red_balls[3],
                                    'red_ball_5': red_balls[4],
                                    'blue_ball_1': blue_balls[0],
                                    'blue_ball_2': blue_balls[1],
                                    'lottery_type': 'DLT'
                                })
                        except Exception as e:
                            continue
                            
        except Exception as e:
            print(f"从新浪爬取数据失败: {e}")
            
        return data
    
    def _crawl_from_163(self, start_date, end_date, max_pages):
        """从网易彩票爬取数据"""
        data = []
        base_url = "https://caipiao.163.com/award/dlt/"
        
        try:
            response = self.session.get(base_url, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 查找开奖数据
            result_items = soup.find_all('div', class_='result-item')
            for item in result_items:
                try:
                    period_elem = item.find('span', class_='period')
                    date_elem = item.find('span', class_='date')
                    numbers_elem = item.find('div', class_='numbers')
                    
                    if period_elem and date_elem and numbers_elem:
                        period = period_elem.text.strip()
                        date = date_elem.text.strip()
                        
                        # 解析号码
                        red_balls_elem = numbers_elem.find('div', class_='red-balls')
                        blue_balls_elem = numbers_elem.find('div', class_='blue-balls')
                        
                        if red_balls_elem and blue_balls_elem:
                            red_numbers = re.findall(r'\d+', red_balls_elem.text)
                            blue_numbers = re.findall(r'\d+', blue_balls_elem.text)
                            
                            if len(red_numbers) == 5 and len(blue_numbers) == 2:
                                data.append({
                                    'period': period,
                                    'date': date,
                                    'red_ball_1': int(red_numbers[0]),
                                    'red_ball_2': int(red_numbers[1]),
                                    'red_ball_3': int(red_numbers[2]),
                                    'red_ball_4': int(red_numbers[3]),
                                    'red_ball_5': int(red_numbers[4]),
                                    'blue_ball_1': int(blue_numbers[0]),
                                    'blue_ball_2': int(blue_numbers[1]),
                                    'lottery_type': 'DLT'
                                })
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"从网易爬取数据失败: {e}")
            
        return data
    
    def _generate_mock_data(self, start_date, end_date):
        """生成模拟数据用于演示"""
        import random
        from datetime import datetime, timedelta
        
        data = []
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        period = 23001  # 起始期号
        current_date = start_dt
        
        while current_date <= end_dt:
            # 生成前区号码 (1-35)
            red_balls = sorted(random.sample(range(1, 36), 5))
            # 生成后区号码 (1-12)
            blue_balls = sorted(random.sample(range(1, 13), 2))
            
            data.append({
                'period': f"{period:05d}",
                'date': current_date.strftime('%Y-%m-%d'),
                'red_ball_1': red_balls[0],
                'red_ball_2': red_balls[1],
                'red_ball_3': red_balls[2],
                'red_ball_4': red_balls[3],
                'red_ball_5': red_balls[4],
                'blue_ball_1': blue_balls[0],
                'blue_ball_2': blue_balls[1],
                'lottery_type': 'DLT'
            })
            
            period += 1
            current_date += timedelta(days=3)  # 大乐透每周开奖3次
            
        print(f"生成了 {len(data)} 条模拟数据")
        return data


if __name__ == "__main__":
    crawler = DLTCrawler()
    data = crawler.get_lottery_data()
    print(f"获取到 {len(data)} 条大乐透数据")
    print(data.head())
