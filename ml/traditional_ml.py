"""
传统机器学习预测模块
使用各种传统ML算法进行彩票号码预测
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class TraditionalMLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_scores = {}
        
    def prepare_features(self, df, lottery_type, lookback_periods=10):
        """准备特征数据"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
        
        # 创建特征矩阵
        features = []
        targets = []
        
        for i in range(lookback_periods, len(df)):
            # 历史特征
            feature_row = []
            
            # 历史期数的号码
            for j in range(lookback_periods):
                period_data = df.iloc[i - lookback_periods + j]
                
                # 红球特征
                for col in red_cols:
                    feature_row.append(period_data[col])
                
                # 蓝球特征
                for col in blue_cols:
                    feature_row.append(period_data[col])
                
                # 衍生特征
                red_numbers = [period_data[col] for col in red_cols]
                blue_numbers = [period_data[col] for col in blue_cols]
                
                # 和值
                feature_row.append(sum(red_numbers))
                feature_row.append(sum(blue_numbers))
                
                # 跨度
                feature_row.append(max(red_numbers) - min(red_numbers))
                if len(blue_numbers) > 1:
                    feature_row.append(max(blue_numbers) - min(blue_numbers))
                else:
                    feature_row.append(0)
                
                # 奇偶比
                red_odd = sum(1 for x in red_numbers if x % 2 == 1)
                feature_row.append(red_odd / len(red_numbers))
                
                blue_odd = sum(1 for x in blue_numbers if x % 2 == 1)
                feature_row.append(blue_odd / len(blue_numbers))
                
                # 大小比
                red_big = sum(1 for x in red_numbers if x > (max(red_numbers) + min(red_numbers)) / 2)
                feature_row.append(red_big / len(red_numbers))
                
                blue_big = sum(1 for x in blue_numbers if x > (max(blue_numbers) + min(blue_numbers)) / 2)
                feature_row.append(blue_big / len(blue_numbers))
            
            features.append(feature_row)
            
            # 目标值（当前期数的号码）
            current_period = df.iloc[i]
            target_row = []
            for col in red_cols + blue_cols:
                target_row.append(current_period[col])
            targets.append(target_row)
        
        return np.array(features), np.array(targets)
    
    def train_models(self, X, y, lottery_type):
        """训练多个模型"""
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[lottery_type] = scaler
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # 定义模型
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'DecisionTree': DecisionTreeRegressor(random_state=42)
        }
        
        # 训练和评估模型
        for name, model in models.items():
            print(f"训练 {name} 模型...")
            
            # 对于多输出回归，需要特殊处理
            if y_train.ndim > 1 and y_train.shape[1] > 1:
                # 使用MultiOutputRegressor包装单输出模型
                from sklearn.multioutput import MultiOutputRegressor
                if name in ['LinearRegression', 'Ridge', 'Lasso', 'SVR', 'KNN', 'DecisionTree']:
                    model = MultiOutputRegressor(model)
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            cv_mean = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # 保存模型和分数
            self.models[f"{lottery_type}_{name}"] = model
            self.model_scores[f"{lottery_type}_{name}"] = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            # 特征重要性（如果支持）
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[f"{lottery_type}_{name}"] = model.feature_importances_
            
            print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            print(f"交叉验证 MSE: {cv_mean:.4f} ± {cv_std:.4f}")
            print("-" * 50)
    
    def hyperparameter_tuning(self, X, y, lottery_type, model_name='RandomForest'):
        """超参数调优"""
        scaler = self.scalers.get(lottery_type, StandardScaler())
        X_scaled = scaler.fit_transform(X)
        
        # 定义参数网格
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVR': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
        }
        
        if model_name not in param_grids:
            print(f"模型 {model_name} 不支持超参数调优")
            return None
        
        # 创建基础模型
        base_models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR()
        }
        
        # 网格搜索
        grid_search = GridSearchCV(
            base_models[model_name],
            param_grids[model_name],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳分数: {-grid_search.best_score_:.4f}")
        
        # 保存最佳模型
        best_model = grid_search.best_estimator_
        self.models[f"{lottery_type}_{model_name}_tuned"] = best_model
        
        return best_model
    
    def predict_next_period(self, df, lottery_type, model_name='RandomForest'):
        """预测下一期号码"""
        model_key = f"{lottery_type}_{model_name}"
        if model_key not in self.models:
            print(f"模型 {model_key} 不存在")
            return None
        
        model = self.models[model_key]
        scaler = self.scalers.get(lottery_type)
        
        if scaler is None:
            print(f"标准化器 {lottery_type} 不存在")
            return None
        
        # 准备特征
        X, _ = self.prepare_features(df, lottery_type)
        if len(X) == 0:
            print("没有足够的历史数据")
            return None
        
        # 使用最新的特征
        latest_features = X[-1].reshape(1, -1)
        latest_features_scaled = scaler.transform(latest_features)
        
        # 预测
        prediction = model.predict(latest_features_scaled)[0]
        
        # 后处理预测结果
        processed_prediction = self._post_process_prediction(prediction, lottery_type)
        
        return processed_prediction
    
    def _post_process_prediction(self, prediction, lottery_type):
        """后处理预测结果"""
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
        
        # 分离红球和蓝球预测
        red_pred = prediction[:red_count]
        blue_pred = prediction[red_count:red_count + blue_count]
        
        # 将预测值映射到有效范围
        red_numbers = []
        for pred in red_pred:
            # 使用sigmoid函数将预测值映射到0-1范围，然后映射到号码范围
            sigmoid_val = 1 / (1 + np.exp(-pred))
            number = int(red_range[0] + sigmoid_val * (red_range[1] - red_range[0]))
            number = max(red_range[0], min(red_range[1], number))
            red_numbers.append(number)
        
        blue_numbers = []
        for pred in blue_pred:
            sigmoid_val = 1 / (1 + np.exp(-pred))
            number = int(blue_range[0] + sigmoid_val * (blue_range[1] - blue_range[0]))
            number = max(blue_range[0], min(blue_range[1], number))
            blue_numbers.append(number)
        
        # 去重并排序
        red_numbers = sorted(list(set(red_numbers)))
        blue_numbers = sorted(list(set(blue_numbers)))
        
        # 如果去重后数量不足，补充随机号码
        while len(red_numbers) < red_count:
            rand_num = np.random.randint(red_range[0], red_range[1] + 1)
            if rand_num not in red_numbers:
                red_numbers.append(rand_num)
        red_numbers = sorted(red_numbers[:red_count])
        
        while len(blue_numbers) < blue_count:
            rand_num = np.random.randint(blue_range[0], blue_range[1] + 1)
            if rand_num not in blue_numbers:
                blue_numbers.append(rand_num)
        blue_numbers = sorted(blue_numbers[:blue_count])
        
        return {
            'red_balls': red_numbers,
            'blue_balls': blue_numbers,
            'raw_prediction': prediction.tolist()
        }
    
    def ensemble_predict(self, df, lottery_type, models=None):
        """集成预测"""
        if models is None:
            models = ['RandomForest', 'GradientBoosting', 'ExtraTrees']
        
        predictions = []
        weights = []
        
        for model_name in models:
            model_key = f"{lottery_type}_{model_name}"
            if model_key in self.models:
                pred = self.predict_next_period(df, lottery_type, model_name)
                if pred:
                    predictions.append(pred)
                    # 使用模型分数作为权重
                    score = self.model_scores.get(model_key, {}).get('r2', 0.5)
                    weights.append(max(0.1, score))  # 确保权重为正
        
        if not predictions:
            return None
        
        # 加权平均
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化
        
        # 对红球和蓝球分别进行加权平均
        red_balls = []
        blue_balls = []
        
        for i, pred in enumerate(predictions):
            red_balls.extend(pred['red_balls'])
            blue_balls.extend(pred['blue_balls'])
        
        # 计算加权平均
        red_avg = np.average(red_balls, weights=np.repeat(weights, len(predictions[0]['red_balls'])))
        blue_avg = np.average(blue_balls, weights=np.repeat(weights, len(predictions[0]['blue_balls'])))
        
        # 转换为整数并确保在有效范围内
        if lottery_type == 'DLT':
            red_range = (1, 35)
            blue_range = (1, 12)
        else:  # SSQ
            red_range = (1, 33)
            blue_range = (1, 16)
        
        red_numbers = [max(red_range[0], min(red_range[1], int(red_avg)))]
        blue_numbers = [max(blue_range[0], min(blue_range[1], int(blue_avg)))]
        
        return {
            'red_balls': red_numbers,
            'blue_balls': blue_numbers,
            'ensemble_weights': weights.tolist(),
            'individual_predictions': predictions
        }
    
    def save_models(self, filepath_prefix):
        """保存模型"""
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}_{model_name}.joblib"
            joblib.dump(model, filename)
            print(f"模型已保存: {filename}")
        
        # 保存标准化器
        for scaler_name, scaler in self.scalers.items():
            filename = f"{filepath_prefix}_scaler_{scaler_name}.joblib"
            joblib.dump(scaler, filename)
            print(f"标准化器已保存: {filename}")
    
    def load_models(self, filepath_prefix):
        """加载模型"""
        import glob
        import os
        
        # 加载模型
        model_files = glob.glob(f"{filepath_prefix}_*.joblib")
        for file in model_files:
            if 'scaler' not in file:
                model_name = os.path.basename(file).replace(f"{filepath_prefix}_", "").replace(".joblib", "")
                self.models[model_name] = joblib.load(file)
                print(f"模型已加载: {model_name}")
        
        # 加载标准化器
        scaler_files = glob.glob(f"{filepath_prefix}_scaler_*.joblib")
        for file in scaler_files:
            scaler_name = os.path.basename(file).replace(f"{filepath_prefix}_scaler_", "").replace(".joblib", "")
            self.scalers[scaler_name] = joblib.load(file)
            print(f"标准化器已加载: {scaler_name}")
    
    def get_model_performance(self):
        """获取模型性能报告"""
        performance_df = pd.DataFrame(self.model_scores).T
        performance_df = performance_df.sort_values('r2', ascending=False)
        return performance_df


if __name__ == "__main__":
    # 测试传统机器学习功能
    ml_predictor = TraditionalMLPredictor()
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    
    test_data = pd.DataFrame({
        'red_ball_1': np.random.randint(1, 36, n_samples),
        'red_ball_2': np.random.randint(1, 36, n_samples),
        'red_ball_3': np.random.randint(1, 36, n_samples),
        'red_ball_4': np.random.randint(1, 36, n_samples),
        'red_ball_5': np.random.randint(1, 36, n_samples),
        'blue_ball_1': np.random.randint(1, 13, n_samples),
        'blue_ball_2': np.random.randint(1, 13, n_samples)
    })
    
    # 准备特征
    X, y = ml_predictor.prepare_features(test_data, 'DLT')
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标矩阵形状: {y.shape}")
    
    # 训练模型
    ml_predictor.train_models(X, y, 'DLT')
    
    # 预测下一期
    prediction = ml_predictor.predict_next_period(test_data, 'DLT')
    print(f"\n预测结果: {prediction}")
    
    # 集成预测
    ensemble_pred = ml_predictor.ensemble_predict(test_data, 'DLT')
    print(f"集成预测结果: {ensemble_pred}")
    
    # 性能报告
    performance = ml_predictor.get_model_performance()
    print(f"\n模型性能报告:")
    print(performance)
