"""
配置文件
包含应用程序的各种配置参数
"""

# 数据库配置
DATABASE_CONFIG = {
    'db_path': 'lottery_data.db',
    'backup_interval': 7,  # 天
    'max_backups': 10
}

# 爬虫配置
CRAWLER_CONFIG = {
    'timeout': 10,
    'retry_times': 3,
    'delay_between_requests': 1,  # 秒
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# 机器学习配置
ML_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cross_validation_folds': 5,
    'models': ['RandomForest', 'GradientBoosting', 'ExtraTrees', 'LinearRegression', 'Ridge', 'Lasso', 'SVR', 'KNN', 'DecisionTree']
}

# 深度学习配置
DL_CONFIG = {
    'sequence_length': 20,
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001,
    'early_stopping_patience': 20,
    'models': ['LSTM', 'GRU', 'CNN_LSTM', 'Transformer']
}

# 统计分析配置
STATS_CONFIG = {
    'hot_cold_window': 20,
    'trend_window': 10,
    'frequency_analysis': True,
    'gap_analysis': True,
    'sum_analysis': True,
    'odd_even_analysis': True,
    'size_analysis': True,
    'consecutive_analysis': True,
    'ac_value_analysis': True,
    'trend_analysis': True
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'colors': {
        'red': '#FF6B6B',
        'blue': '#4ECDC4',
        'green': '#45B7D1',
        'orange': '#FFA07A',
        'purple': '#98D8C8',
        'gray': '#6C7B7F'
    },
    'save_charts': True,
    'chart_format': 'png'
}

# 彩票规则配置
LOTTERY_RULES = {
    'DLT': {
        'name': '超级大乐透',
        'red_range': (1, 35),
        'blue_range': (1, 12),
        'red_count': 5,
        'blue_count': 2,
        'draw_days': ['周一', '周三', '周六'],
        'draw_time': '21:15'
    },
    'SSQ': {
        'name': '双色球',
        'red_range': (1, 33),
        'blue_range': (1, 16),
        'red_count': 6,
        'blue_count': 1,
        'draw_days': ['周二', '周四', '周日'],
        'draw_time': '21:15'
    }
}

# 界面配置
GUI_CONFIG = {
    'window_size': '1200x800',
    'theme': 'default',
    'font_family': 'Arial',
    'font_size': 10,
    'auto_save': True,
    'auto_save_interval': 300  # 秒
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'lottery_prediction.log',
    'max_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# 预测配置
PREDICTION_CONFIG = {
    'min_data_points': 50,
    'confidence_threshold': 0.1,
    'ensemble_weights': 'auto',  # 'auto', 'equal', 'performance'
    'save_predictions': True,
    'prediction_history_file': 'prediction_history.json'
}

# 数据源配置
DATA_SOURCES = {
    'DLT': [
        'https://datachart.500.com/dlt/history/newinc/history.php',
        'https://match.lottery.sina.com.cn/lotto/pc_zst/index',
        'https://caipiao.163.com/award/dlt/'
    ],
    'SSQ': [
        'https://datachart.500.com/ssq/history/newinc/history.php',
        'https://match.lottery.sina.com.cn/lotto/pc_zst/index',
        'https://caipiao.163.com/award/ssq/',
        'https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice'
    ]
}

# 免责声明
DISCLAIMER = """
免责声明：
1. 本软件仅供学习和研究使用，不构成任何投资建议。
2. 彩票具有随机性，任何预测都不能保证准确性。
3. 请理性购彩，量力而行，切勿沉迷。
4. 使用本软件产生的任何损失，开发者不承担责任。
5. 请遵守当地法律法规，合法购彩。
"""

# 版本信息
VERSION_INFO = {
    'version': '1.0.0',
    'release_date': '2024-01-01',
    'author': 'AI Assistant',
    'description': '基于机器学习和深度学习的彩票号码预测软件',
    'features': [
        '自动数据爬取',
        '多种统计分析方法',
        '传统机器学习预测',
        '深度学习神经网络预测',
        '友好的图形用户界面',
        '数据可视化分析'
    ]
}
