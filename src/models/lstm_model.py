import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class LSTMPredictor:
    def __init__(self, sequence_length=10, log_dir=None):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.log_dir = Path(log_dir) if log_dir else Path('logs/lstm')
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'lstm_training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('LSTMPredictor')
    
    def prepare_sequences(self, data):
        sequences = []
        targets = []
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:(i + self.sequence_length)]
            target = data[i + self.sequence_length]
            sequences.append(sequence)
            targets.append(target)
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(1024, return_sequences=True, 
                              kernel_regularizer=l2(1e-4)), 
                         input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.5),
            
            Bidirectional(LSTM(512, return_sequences=True)),
            BatchNormalization(),
            Dropout(0.5),
            
            Bidirectional(LSTM(256)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(64, activation='relu', kernel_regularizer=l2(1e-4)),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.01),
                     loss='huber',
                     metrics=['mae'])
        return model
    
    def train(self, data_path, output_dir='models/'):
        try:
            self.logger.info("开始训练LSTM模型...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载和准备数据
            df = pd.read_csv(data_path)
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features].values
            y = df['wait_time'].values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 准备序列数据
            X_train_seq, y_train_seq = self.prepare_sequences(X_train)
            X_test_seq, y_test_seq = self.prepare_sequences(X_test)
            
            # 构建和训练模型
            self.model = self.build_model((self.sequence_length, X_train.shape[1]))
            
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
                    factor=0.1,
                    patience=2,
                    min_lr=1e-6,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=2,
                    restore_best_weights=True,
                    mode='min',
                    verbose=1
                )
            ]
            
            history = self.model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=64,
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估和保存
            self._evaluate_model(X_test_seq, y_test_seq)
            self._plot_training_history(history, output_dir)
            
        except Exception as e:
            self.logger.error(f"LSTM模型训练失败: {str(e)}")
            raise
    
    def predict(self, X):
        X_seq = X[np.newaxis, :self.sequence_length, :]
        return self.model.predict(X_seq)[0]
    
    def _evaluate_model(self, X_test, y_test):
        pred = self.model.predict(X_test)
        mae = np.mean(np.abs(pred - y_test))
        self.logger.info(f"LSTM MAE: {mae:.4f}")
    
    def _plot_training_history(self, history, output_dir):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        
        plt.tight_layout()
        # 保存到日志目录
        plt.savefig(self.log_dir / f'lstm_training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()