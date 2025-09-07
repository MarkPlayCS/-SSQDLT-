"""
深度学习预测模块
使用神经网络进行彩票号码预测
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
tf.random.set_seed(42)
np.random.seed(42)


class DeepLearningPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.model_configs = {}
        
    def prepare_sequence_data(self, df, lottery_type, sequence_length=20, prediction_length=1):
        """准备序列数据用于深度学习"""
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
        
        # 提取号码数据
        red_data = df[red_cols].values
        blue_data = df[blue_cols].values
        
        # 标准化数据
        red_scaler = MinMaxScaler()
        blue_scaler = MinMaxScaler()
        
        red_scaled = red_scaler.fit_transform(red_data)
        blue_scaled = blue_scaler.fit_transform(blue_data)
        
        # 保存标准化器
        self.scalers[f"{lottery_type}_red"] = red_scaler
        self.scalers[f"{lottery_type}_blue"] = blue_scaler
        
        # 创建序列
        X_red, y_red = self._create_sequences(red_scaled, sequence_length, prediction_length)
        X_blue, y_blue = self._create_sequences(blue_scaled, sequence_length, prediction_length)
        
        return {
            'red': {'X': X_red, 'y': y_red},
            'blue': {'X': X_blue, 'y': y_blue}
        }
    
    def _create_sequences(self, data, sequence_length, prediction_length):
        """创建时间序列数据"""
        X, y = [], []
        
        for i in range(sequence_length, len(data) - prediction_length + 1):
            X.append(data[i-sequence_length:i])
            y.append(data[i:i+prediction_length])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape, output_shape, model_name='LSTM'):
        """构建LSTM模型"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_gru_model(self, input_shape, output_shape, model_name='GRU'):
        """构建GRU模型"""
        model = models.Sequential([
            layers.GRU(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.GRU(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.GRU(32),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape, output_shape, model_name='CNN_LSTM'):
        """构建CNN-LSTM混合模型"""
        model = models.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            layers.LSTM(128, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(output_shape, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_transformer_model(self, input_shape, output_shape, model_name='Transformer'):
        """构建Transformer模型"""
        # 简化的Transformer实现
        inputs = layers.Input(shape=input_shape)
        
        # 位置编码
        x = layers.Dense(64)(inputs)
        x = layers.LayerNormalization()(x)
        
        # 多头注意力
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=64, dropout=0.1
        )(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # 前馈网络
        ffn = layers.Dense(128, activation='relu')(x)
        ffn = layers.Dense(64)(ffn)
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization()(x)
        
        # 全局平均池化
        x = layers.GlobalAveragePooling1D()(x)
        
        # 输出层
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        outputs = layers.Dense(output_shape, activation='sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, data, lottery_type, model_type='LSTM', epochs=100, batch_size=32):
        """训练深度学习模型"""
        # 准备数据
        red_data = data['red']
        blue_data = data['blue']
        
        # 分割数据
        X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(
            red_data['X'], red_data['y'], test_size=0.2, random_state=42
        )
        
        X_blue_train, X_blue_test, y_blue_train, y_blue_test = train_test_split(
            blue_data['X'], blue_data['y'], test_size=0.2, random_state=42
        )
        
        # 构建模型
        input_shape = (X_red_train.shape[1], X_red_train.shape[2])
        
        if model_type == 'LSTM':
            red_model = self.build_lstm_model(input_shape, y_red_train.shape[-1])
            blue_model = self.build_lstm_model(input_shape, y_blue_train.shape[-1])
        elif model_type == 'GRU':
            red_model = self.build_gru_model(input_shape, y_red_train.shape[-1])
            blue_model = self.build_gru_model(input_shape, y_blue_train.shape[-1])
        elif model_type == 'CNN_LSTM':
            red_model = self.build_cnn_lstm_model(input_shape, y_red_train.shape[-1])
            blue_model = self.build_cnn_lstm_model(input_shape, y_blue_train.shape[-1])
        elif model_type == 'Transformer':
            red_model = self.build_transformer_model(input_shape, y_red_train.shape[-1])
            blue_model = self.build_transformer_model(input_shape, y_blue_train.shape[-1])
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 回调函数
        callbacks_list = [
            callbacks.EarlyStopping(patience=20, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7),
            callbacks.ModelCheckpoint(
                f'best_{lottery_type}_{model_type}_red_model.h5',
                save_best_only=True, monitor='val_loss'
            )
        ]
        
        # 训练红球模型
        print(f"训练 {lottery_type} {model_type} 红球模型...")
        red_history = red_model.fit(
            X_red_train, y_red_train,
            validation_data=(X_red_test, y_red_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 训练蓝球模型
        print(f"训练 {lottery_type} {model_type} 蓝球模型...")
        blue_history = blue_model.fit(
            X_blue_train, y_blue_train,
            validation_data=(X_blue_test, y_blue_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 保存模型
        self.models[f"{lottery_type}_{model_type}_red"] = red_model
        self.models[f"{lottery_type}_{model_type}_blue"] = blue_model
        self.history[f"{lottery_type}_{model_type}_red"] = red_history.history
        self.history[f"{lottery_type}_{model_type}_blue"] = blue_history.history
        
        # 评估模型
        red_loss, red_mae = red_model.evaluate(X_red_test, y_red_test, verbose=0)
        blue_loss, blue_mae = blue_model.evaluate(X_blue_test, y_blue_test, verbose=0)
        
        print(f"红球模型 - Loss: {red_loss:.4f}, MAE: {red_mae:.4f}")
        print(f"蓝球模型 - Loss: {blue_loss:.4f}, MAE: {blue_mae:.4f}")
        
        return {
            'red_model': red_model,
            'blue_model': blue_model,
            'red_history': red_history.history,
            'blue_history': blue_history.history,
            'red_metrics': {'loss': red_loss, 'mae': red_mae},
            'blue_metrics': {'loss': blue_loss, 'mae': blue_mae}
        }
    
    def predict_next_period(self, df, lottery_type, model_type='LSTM', sequence_length=20):
        """预测下一期号码"""
        red_model_key = f"{lottery_type}_{model_type}_red"
        blue_model_key = f"{lottery_type}_{model_type}_blue"
        
        if red_model_key not in self.models or blue_model_key not in self.models:
            print(f"模型 {model_type} 未训练")
            return None
        
        red_model = self.models[red_model_key]
        blue_model = self.models[blue_model_key]
        
        # 准备最新序列
        if lottery_type == 'DLT':
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5']
            blue_cols = ['blue_ball_1', 'blue_ball_2']
        else:  # SSQ
            red_cols = ['red_ball_1', 'red_ball_2', 'red_ball_3', 'red_ball_4', 'red_ball_5', 'red_ball_6']
            blue_cols = ['blue_ball']
        
        # 获取最新数据
        recent_data = df.tail(sequence_length)
        red_data = recent_data[red_cols].values
        blue_data = recent_data[blue_cols].values
        
        # 标准化
        red_scaler = self.scalers[f"{lottery_type}_red"]
        blue_scaler = self.scalers[f"{lottery_type}_blue"]
        
        red_scaled = red_scaler.transform(red_data)
        blue_scaled = blue_scaler.transform(blue_data)
        
        # 准备输入
        X_red = red_scaled.reshape(1, sequence_length, -1)
        X_blue = blue_scaled.reshape(1, sequence_length, -1)
        
        # 预测
        red_pred = red_model.predict(X_red, verbose=0)[0]
        blue_pred = blue_model.predict(X_blue, verbose=0)[0]
        
        # 反标准化
        red_pred_original = red_scaler.inverse_transform(red_pred.reshape(1, -1))[0]
        blue_pred_original = blue_scaler.inverse_transform(blue_pred.reshape(1, -1))[0]
        
        # 后处理
        processed_prediction = self._post_process_deep_prediction(
            red_pred_original, blue_pred_original, lottery_type
        )
        
        return processed_prediction
    
    def _post_process_deep_prediction(self, red_pred, blue_pred, lottery_type):
        """后处理深度学习预测结果"""
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
        
        # 将预测值映射到有效范围
        red_numbers = []
        for pred in red_pred:
            number = int(round(pred))
            number = max(red_range[0], min(red_range[1], number))
            red_numbers.append(number)
        
        blue_numbers = []
        for pred in blue_pred:
            number = int(round(pred))
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
            'raw_red_prediction': red_pred.tolist(),
            'raw_blue_prediction': blue_pred.tolist()
        }
    
    def ensemble_deep_predict(self, df, lottery_type, models=None, sequence_length=20):
        """深度学习集成预测"""
        if models is None:
            models = ['LSTM', 'GRU', 'CNN_LSTM']
        
        predictions = []
        weights = []
        
        for model_type in models:
            pred = self.predict_next_period(df, lottery_type, model_type, sequence_length)
            if pred:
                predictions.append(pred)
                # 使用模型历史性能作为权重
                red_history = self.history.get(f"{lottery_type}_{model_type}_red", {})
                blue_history = self.history.get(f"{lottery_type}_{model_type}_blue", {})
                
                # 计算平均验证损失作为权重（损失越小权重越大）
                red_val_loss = np.mean(red_history.get('val_loss', [1.0])[-10:])  # 最近10个epoch
                blue_val_loss = np.mean(blue_history.get('val_loss', [1.0])[-10:])
                avg_loss = (red_val_loss + blue_val_loss) / 2
                weight = 1.0 / (avg_loss + 1e-8)  # 避免除零
                weights.append(weight)
        
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
        
        red_numbers = [max(red_range[0], min(red_range[1], int(round(red_avg))))]
        blue_numbers = [max(blue_range[0], min(blue_range[1], int(round(blue_avg))))]
        
        return {
            'red_balls': red_numbers,
            'blue_balls': blue_numbers,
            'ensemble_weights': weights.tolist(),
            'individual_predictions': predictions
        }
    
    def plot_training_history(self, lottery_type, model_type, save_path=None):
        """绘制训练历史"""
        import matplotlib.pyplot as plt
        
        red_history = self.history.get(f"{lottery_type}_{model_type}_red", {})
        blue_history = self.history.get(f"{lottery_type}_{model_type}_blue", {})
        
        if not red_history or not blue_history:
            print("没有找到训练历史数据")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 红球模型损失
        axes[0, 0].plot(red_history['loss'], label='训练损失')
        axes[0, 0].plot(red_history['val_loss'], label='验证损失')
        axes[0, 0].set_title(f'{lottery_type} {model_type} 红球模型损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 红球模型MAE
        axes[0, 1].plot(red_history['mae'], label='训练MAE')
        axes[0, 1].plot(red_history['val_mae'], label='验证MAE')
        axes[0, 1].set_title(f'{lottery_type} {model_type} 红球模型MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 蓝球模型损失
        axes[1, 0].plot(blue_history['loss'], label='训练损失')
        axes[1, 0].plot(blue_history['val_loss'], label='验证损失')
        axes[1, 0].set_title(f'{lottery_type} {model_type} 蓝球模型损失')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 蓝球模型MAE
        axes[1, 1].plot(blue_history['mae'], label='训练MAE')
        axes[1, 1].plot(blue_history['val_mae'], label='验证MAE')
        axes[1, 1].set_title(f'{lottery_type} {model_type} 蓝球模型MAE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_models(self, filepath_prefix):
        """保存深度学习模型"""
        for model_name, model in self.models.items():
            filename = f"{filepath_prefix}_{model_name}.h5"
            model.save(filename)
            print(f"模型已保存: {filename}")
        
        # 保存标准化器
        import joblib
        for scaler_name, scaler in self.scalers.items():
            filename = f"{filepath_prefix}_scaler_{scaler_name}.joblib"
            joblib.dump(scaler, filename)
            print(f"标准化器已保存: {filename}")
    
    def load_models(self, filepath_prefix):
        """加载深度学习模型"""
        import glob
        import os
        import joblib
        
        # 加载模型
        model_files = glob.glob(f"{filepath_prefix}_*.h5")
        for file in model_files:
            if 'scaler' not in file:
                model_name = os.path.basename(file).replace(f"{filepath_prefix}_", "").replace(".h5", "")
                self.models[model_name] = keras.models.load_model(file)
                print(f"模型已加载: {model_name}")
        
        # 加载标准化器
        scaler_files = glob.glob(f"{filepath_prefix}_scaler_*.joblib")
        for file in scaler_files:
            scaler_name = os.path.basename(file).replace(f"{filepath_prefix}_scaler_", "").replace(".joblib", "")
            self.scalers[scaler_name] = joblib.load(file)
            print(f"标准化器已加载: {scaler_name}")


if __name__ == "__main__":
    # 测试深度学习功能
    dl_predictor = DeepLearningPredictor()
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 200
    
    test_data = pd.DataFrame({
        'red_ball_1': np.random.randint(1, 36, n_samples),
        'red_ball_2': np.random.randint(1, 36, n_samples),
        'red_ball_3': np.random.randint(1, 36, n_samples),
        'red_ball_4': np.random.randint(1, 36, n_samples),
        'red_ball_5': np.random.randint(1, 36, n_samples),
        'blue_ball_1': np.random.randint(1, 13, n_samples),
        'blue_ball_2': np.random.randint(1, 13, n_samples)
    })
    
    # 准备序列数据
    sequence_data = dl_predictor.prepare_sequence_data(test_data, 'DLT', sequence_length=10)
    print(f"红球序列数据形状: {sequence_data['red']['X'].shape}")
    print(f"蓝球序列数据形状: {sequence_data['blue']['X'].shape}")
    
    # 训练LSTM模型
    result = dl_predictor.train_model(sequence_data, 'DLT', 'LSTM', epochs=50)
    print(f"训练完成，红球模型损失: {result['red_metrics']['loss']:.4f}")
    print(f"蓝球模型损失: {result['blue_metrics']['loss']:.4f}")
    
    # 预测下一期
    prediction = dl_predictor.predict_next_period(test_data, 'DLT', 'LSTM')
    print(f"预测结果: {prediction}")
    
    # 集成预测
    ensemble_pred = dl_predictor.ensemble_deep_predict(test_data, 'DLT')
    print(f"集成预测结果: {ensemble_pred}")
