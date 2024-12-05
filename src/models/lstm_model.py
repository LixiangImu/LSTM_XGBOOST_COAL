import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import logging
from pathlib import Path
import matplotlib.pyplot as plt

class LSTMPredictor:
    def __init__(self, sequence_length=48, log_dir=None):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = StandardScaler()
        self.log_dir = Path(log_dir) if log_dir else Path('logs/lstm')
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志系统"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('LSTMPredictor')
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_dir / 'lstm_training.log')
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(fh)
            self.logger.setLevel(logging.INFO)

    def _plot_training_history(self, history, output_dir):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png')
        plt.close()

    def prepare_sequences(self, data, targets=None, is_training=True):
        """准备序列数据"""
        if is_training:
            data = self.feature_scaler.fit_transform(data)
            if targets is not None:
                targets = self.scaler.fit_transform(targets.reshape(-1, 1))
        else:
            data = self.feature_scaler.transform(data)
            if targets is not None:
                targets = self.scaler.transform(targets.reshape(-1, 1))

        sequences = []
        target_values = []
        
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:(i + self.sequence_length)])
            if targets is not None:
                target_values.append(targets[i + self.sequence_length])
        
        return np.array(sequences), np.array(target_values) if targets is not None else None

    def build_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential([
            # 第一层: GRU层
            GRU(256, return_sequences=True, 
                input_shape=input_shape,
                kernel_regularizer=l2(1e-5)),
            Dropout(0.3),
            
            # 第二层: LSTM层
            LSTM(128, return_sequences=True,
                 kernel_regularizer=l2(1e-5)),
            Dropout(0.3),
            
            # 第三层: LSTM层
            LSTM(64, return_sequences=False,
                 kernel_regularizer=l2(1e-5)),
            Dropout(0.3),
            
            # 全连接层
            Dense(32, activation='relu', kernel_regularizer=l2(1e-5)),
            Dropout(0.2),
            Dense(16, activation='relu', kernel_regularizer=l2(1e-5)),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                     loss='huber',
                     metrics=['mae', 'mse'])
        return model

    def train(self, data_path, output_dir='models/'):
        try:
            self.logger.info("开始训练LSTM模型...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载数据
            df = pd.read_csv(data_path)
            
            # 选择特征
            features = [
                'rolling_mean_wait', 
                'coal_type_mean_wait_30min',
                'coal_type_mean_wait_rolling',
                'wait_time_ratio',
                'coal_type_truck_count',
                'hourly_truck_count',
                'load_efficiency',
                'hour'
            ]
            
            X = df[features].values
            y = df['wait_time'].values
            
            # 数据集划分
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42,
                stratify=pd.qcut(y, q=10, labels=False, duplicates='drop')
            )
            
            # 准备序列数据
            X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train, True)
            X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test, False)
            
            # 构建模型
            self.model = self.build_model((self.sequence_length, X_train.shape[1]))
            
            # 回调函数
            callbacks = [
                ModelCheckpoint(
                    output_dir / 'best_lstm_model.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            # 训练模型
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=128,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估和可视化
            self._evaluate_model(X_test_seq, y_test_seq)
            self._plot_training_history(history, output_dir)
            
        except Exception as e:
            self.logger.error(f"训练失败: {str(e)}")
            raise

    def _evaluate_model(self, X_test, y_test):
        """评估模型"""
        pred = self.model.predict(X_test)
        pred = self.scaler.inverse_transform(pred)
        y_test = self.scaler.inverse_transform(y_test)
        
        mae = np.mean(np.abs(pred - y_test))
        mse = np.mean((pred - y_test)**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum((y_test - pred)**2) / np.sum((y_test - np.mean(y_test))**2)
        
        self.logger.info(f"评估结果:")
        self.logger.info(f"MAE: {mae:.4f}")
        self.logger.info(f"MSE: {mse:.4f}")
        self.logger.info(f"RMSE: {rmse:.4f}")
        self.logger.info(f"R2: {r2:.4f}")