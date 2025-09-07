"""
辅助工具模块
提供各种辅助函数和工具
"""

import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
import os
import logging


def setup_logging(log_file="lottery_prediction.log"):
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate_lottery_numbers(numbers, lottery_type):
    """验证彩票号码"""
    if lottery_type == 'DLT':
        red_range = (1, 35)
        blue_range = (1, 12)
        red_count = 5
        blue_count = 2
    else:  # SSQ
        red_range = (1, 33)
        blue_range = (1, 16)
        red_count = 6
        blue_count = 1
    
    red_balls = numbers.get('red_balls', [])
    blue_balls = numbers.get('blue_balls', [])
    
    # 检查数量
    if len(red_balls) != red_count:
        return False, f"红球数量应为{red_count}个"
    
    if len(blue_balls) != blue_count:
        return False, f"蓝球数量应为{blue_count}个"
    
    # 检查范围
    for num in red_balls:
        if not (red_range[0] <= num <= red_range[1]):
            return False, f"红球号码{num}超出范围{red_range}"
    
    for num in blue_balls:
        if not (blue_range[0] <= num <= blue_range[1]):
            return False, f"蓝球号码{num}超出范围{blue_range}"
    
    # 检查重复
    if len(set(red_balls)) != len(red_balls):
        return False, "红球号码有重复"
    
    if len(set(blue_balls)) != len(blue_balls):
        return False, "蓝球号码有重复"
    
    return True, "号码验证通过"


def calculate_win_probability(numbers, lottery_type):
    """计算中奖概率"""
    if lottery_type == 'DLT':
        # 大乐透概率计算
        total_combinations = 1
        # 前区5个号码从35个中选
        for i in range(5):
            total_combinations *= (35 - i) / (i + 1)
        # 后区2个号码从12个中选
        for i in range(2):
            total_combinations *= (12 - i) / (i + 1)
        
        # 一等奖概率
        first_prize_prob = 1 / total_combinations
        
        return {
            'first_prize': first_prize_prob,
            'total_combinations': int(total_combinations),
            'description': f"一等奖中奖概率约为 1/{int(total_combinations):,}"
        }
    
    else:  # SSQ
        # 双色球概率计算
        red_combinations = 1
        for i in range(6):
            red_combinations *= (33 - i) / (i + 1)
        
        blue_combinations = 16
        
        total_combinations = red_combinations * blue_combinations
        
        # 一等奖概率
        first_prize_prob = 1 / total_combinations
        
        return {
            'first_prize': first_prize_prob,
            'total_combinations': int(total_combinations),
            'description': f"一等奖中奖概率约为 1/{int(total_combinations):,}"
        }


def format_prediction_result(prediction, lottery_type):
    """格式化预测结果"""
    if not prediction:
        return "预测失败"
    
    red_balls = prediction.get('red_balls', [])
    blue_balls = prediction.get('blue_balls', [])
    
    # 格式化红球
    red_str = " ".join([f"{num:02d}" for num in sorted(red_balls)])
    
    # 格式化蓝球
    blue_str = " ".join([f"{num:02d}" for num in sorted(blue_balls)])
    
    result = f"红球: {red_str} | 蓝球: {blue_str}"
    
    # 添加概率信息
    prob_info = calculate_win_probability(prediction, lottery_type)
    result += f"\n中奖概率: {prob_info['description']}"
    
    return result


def save_prediction_history(prediction, lottery_type, method, filepath="prediction_history.json"):
    """保存预测历史"""
    history = []
    
    # 加载现有历史
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []
    
    # 添加新预测
    new_entry = {
        'timestamp': datetime.now().isoformat(),
        'lottery_type': lottery_type,
        'method': method,
        'prediction': prediction,
        'formatted': format_prediction_result(prediction, lottery_type)
    }
    
    history.append(new_entry)
    
    # 保存历史
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_prediction_history(filepath="prediction_history.json"):
    """加载预测历史"""
    if not os.path.exists(filepath):
        return []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return []


def compare_predictions(predictions):
    """比较多个预测结果"""
    if not predictions:
        return None
    
    # 统计红球和蓝球的出现频率
    red_freq = {}
    blue_freq = {}
    
    for pred in predictions:
        for num in pred.get('red_balls', []):
            red_freq[num] = red_freq.get(num, 0) + 1
        
        for num in pred.get('blue_balls', []):
            blue_freq[num] = blue_freq.get(num, 0) + 1
    
    # 找出最常出现的号码
    red_consensus = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)
    blue_consensus = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'red_consensus': red_consensus,
        'blue_consensus': blue_consensus,
        'red_frequency': red_freq,
        'blue_frequency': blue_freq
    }


def generate_random_numbers(lottery_type, count=1):
    """生成随机号码"""
    results = []
    
    for _ in range(count):
        if lottery_type == 'DLT':
            red_balls = sorted(np.random.choice(range(1, 36), 5, replace=False))
            blue_balls = sorted(np.random.choice(range(1, 13), 2, replace=False))
        else:  # SSQ
            red_balls = sorted(np.random.choice(range(1, 34), 6, replace=False))
            blue_balls = [np.random.randint(1, 17)]
        
        results.append({
            'red_balls': red_balls,
            'blue_balls': blue_balls
        })
    
    return results if count > 1 else results[0]


def calculate_number_statistics(df, lottery_type):
    """计算号码统计信息"""
    if lottery_type == 'DLT':
        red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
        blue_cols = ['blue_ball_1', 'blue_ball_2']
    else:  # SSQ
        red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
        blue_cols = ['blue_ball']
    
    # 红球统计
    red_stats = {}
    for col in red_cols:
        for num in df[col]:
            red_stats[num] = red_stats.get(num, 0) + 1
    
    # 蓝球统计
    blue_stats = {}
    for col in blue_cols:
        for num in df[col]:
            blue_stats[num] = blue_stats.get(num, 0) + 1
    
    return {
        'red_statistics': red_stats,
        'blue_statistics': blue_stats,
        'total_draws': len(df)
    }


def export_data_to_excel(df, filename, sheet_name='数据'):
    """导出数据到Excel"""
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        return True, f"数据已导出到 {filename}"
    except Exception as e:
        return False, f"导出失败: {str(e)}"


def import_data_from_excel(filename, sheet_name=0):
    """从Excel导入数据"""
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        return True, df
    except Exception as e:
        return False, f"导入失败: {str(e)}"


def create_backup(filepath, backup_dir="backups"):
    """创建备份"""
    if not os.path.exists(filepath):
        return False, "文件不存在"
    
    # 创建备份目录
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # 生成备份文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    backup_filename = f"{name}_{timestamp}{ext}"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    try:
        # 复制文件
        import shutil
        shutil.copy2(filepath, backup_path)
        return True, f"备份已创建: {backup_path}"
    except Exception as e:
        return False, f"备份失败: {str(e)}"


def cleanup_old_files(directory, days=30, pattern="*.log"):
    """清理旧文件"""
    if not os.path.exists(directory):
        return False, "目录不存在"
    
    import glob
    import time
    
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    files_deleted = 0
    
    try:
        for filepath in glob.glob(os.path.join(directory, pattern)):
            if os.path.getmtime(filepath) < cutoff_time:
                os.remove(filepath)
                files_deleted += 1
        
        return True, f"已删除 {files_deleted} 个旧文件"
    except Exception as e:
        return False, f"清理失败: {str(e)}"


def get_system_info():
    """获取系统信息"""
    import platform
    import psutil
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        'memory_available': f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        'disk_usage': f"{psutil.disk_usage('/').percent:.1f}%"
    }
    
    return info


def validate_data_quality(df, lottery_type):
    """验证数据质量"""
    issues = []
    
    if df.empty:
        issues.append("数据为空")
        return issues
    
    # 检查必要列
    if lottery_type == 'DLT':
        required_cols = ['period', 'date', 'red_ball_1', 'red_ball_2', 'red_ball_3', 
                        'red_ball_4', 'red_ball_5', 'blue_ball_1', 'blue_ball_2']
    else:  # SSQ
        required_cols = ['period', 'date', 'red_ball_1', 'red_ball_2', 'red_ball_3', 
                        'red_ball_4', 'red_ball_5', 'red_ball_6', 'blue_ball']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"缺少必要列: {missing_cols}")
    
    # 检查重复期号
    if 'period' in df.columns:
        duplicates = df['period'].duplicated().sum()
        if duplicates > 0:
            issues.append(f"发现 {duplicates} 个重复期号")
    
    # 检查日期格式
    if 'date' in df.columns:
        try:
            pd.to_datetime(df['date'])
        except:
            issues.append("日期格式不正确")
    
    # 检查号码范围
    if lottery_type == 'DLT':
        red_range = (1, 35)
        blue_range = (1, 12)
    else:  # SSQ
        red_range = (1, 33)
        blue_range = (1, 16)
    
    red_cols = [col for col in df.columns if col.startswith('red_ball_')]
    blue_cols = [col for col in df.columns if col.startswith('blue_ball')]
    
    for col in red_cols:
        invalid_red = df[(df[col] < red_range[0]) | (df[col] > red_range[1])]
        if not invalid_red.empty:
            issues.append(f"列 {col} 中有 {len(invalid_red)} 个值超出范围 {red_range}")
    
    for col in blue_cols:
        invalid_blue = df[(df[col] < blue_range[0]) | (df[col] > blue_range[1])]
        if not invalid_blue.empty:
            issues.append(f"列 {col} 中有 {len(invalid_blue)} 个值超出范围 {blue_range}")
    
    return issues


if __name__ == "__main__":
    # 测试辅助函数
    logger = setup_logging()
    logger.info("辅助工具模块测试开始")
    
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
    
    # 测试格式化结果
    formatted = format_prediction_result(test_numbers, 'DLT')
    print(f"格式化结果: {formatted}")
    
    # 测试随机号码生成
    random_nums = generate_random_numbers('DLT', 3)
    print(f"随机号码: {random_nums}")
    
    logger.info("辅助工具模块测试完成")
