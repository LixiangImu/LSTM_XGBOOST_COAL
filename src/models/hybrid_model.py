import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

class HybridPredictor:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self._setup_logging()
        
    def _setup_logging(self):
        """配置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('HybridPredictor')
    
    def prepare_sequences(self, data):
        """准备LSTM的序列数据"""
        sequences = []
        targets = []
        
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:(i + self.sequence_length)]
            target = data[i + self.sequence_length]
            sequences.append(sequence)
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential([
            # 双向LSTM层
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
        
        model.compile(optimizer=Adam(learning_rate=0.01,
                                   beta_1=0.9,
                                   beta_2=0.999,
                                   epsilon=1e-07),
                     loss='huber',
                     metrics=['mae'])
        return model
    
    def build_xgb_model(self):
        """构建XGBoost模型"""
        return xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=9,
            learning_rate=0.01,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            tree_method='gpu_hist',
            early_stopping_rounds=10
        )
    
    def train(self, data_path, output_dir='models/'):
        try:
            self.logger.info("开始训练混合模型...")
            # 确保输出目录存在
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 删除重复的ModelCheckpoint，只保留这一个
            lstm_checkpoint = ModelCheckpoint(
                output_dir / 'best_lstm_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            
            # 1. 加载数据
            df = pd.read_csv(data_path)
            
            # 2. 分离特征和目标
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features].values
            y = df['wait_time'].values
            
            # 3. 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 4. 训练LSTM模型
            X_train_seq, y_train_seq = self.prepare_sequences(X_train)
            X_test_seq, y_test_seq = self.prepare_sequences(X_test)
            
            self.lstm_model = self.build_lstm_model((self.sequence_length, X_train.shape[1]))
            
            # 添加模型检查点
            model_checkpoint = ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
            
            # 调整学习率调度器
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,           # 更激进的降低
                patience=2,           # 更快反应
                min_lr=1e-6,
                verbose=1
            )
            
            # 添加早停
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=2,          # 适当调整
                restore_best_weights=True,
                mode='min',
                min_delta=1e-4,
                verbose=1
            )
            
            # LSTM训练
            lstm_history = self.lstm_model.fit(
                X_train_seq, y_train_seq,
                epochs=100,
                batch_size=64,
                validation_split=0.2,
                verbose=1,
                shuffle=True,
                callbacks=[early_stopping, reduce_lr, lstm_checkpoint]
            )
            
            # 5. 训练XGBoost模型
            self.xgb_model = self.build_xgb_model()
            self.xgb_model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=True
            )
            
            # 保存XGBoost最佳模型
            xgb_model_path = output_dir / 'best_xgb_model.json'
            self.xgb_model.save_model(str(xgb_model_path))
            self.logger.info(f"XGBoost模型已保存到: {xgb_model_path}")
            
            # 添加日志确认模型评估和保存
            self.logger.info("开始评估模型...")
            self._evaluate_models(X_test, y_test, X_test_seq, y_test_seq)
            
            self.logger.info("开始保存模型...")
            self._save_models(output_dir)
            
            self.logger.info("开始绘制训练历史...")
            self._plot_training_history(lstm_history, output_dir)
            
            self.logger.info(f"模型训练完成，所有文件已保存到: {output_dir}")
            
        except Exception as e:
            self.logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def predict(self, X):
        """混合预测"""
        # LSTM预测
        X_seq = X[np.newaxis, :self.sequence_length, :]
        lstm_pred = self.lstm_model.predict(X_seq)
        
        # XGBoost预测
        xgb_pred = self.xgb_model.predict(X[-1:])
        
        # 组合预测（简单平均）
        final_pred = (lstm_pred[0] + xgb_pred[0]) / 2
        
        return final_pred
    
    def _evaluate_models(self, X_test, y_test, X_test_seq, y_test_seq):
        """评估模型性能"""
        # LSTM评估
        lstm_pred = self.lstm_model.predict(X_test_seq)
        lstm_mae = np.mean(np.abs(lstm_pred - y_test_seq))
        
        # XGBoost评估
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_mae = np.mean(np.abs(xgb_pred - y_test))
        
        # 输出评估结果
        self.logger.info(f"LSTM MAE: {lstm_mae:.4f}")
        self.logger.info(f"XGBoost MAE: {xgb_mae:.4f}")
        
        # 特征重要性
        importance = pd.DataFrame({
            'feature': self.xgb_model.feature_names_in_,
            'importance': self.xgb_model.feature_importances_
        })
        self.logger.info("\nXGBoost特征重要性:")
        self.logger.info(importance.sort_values('importance', ascending=False))
    
    def _save_models(self, output_dir):
        """保存模型"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 保存LSTM模型
        self.lstm_model.save(Path(output_dir) / 'lstm_model.keras')
        
        # 保存XGBoost模型
        self.xgb_model.save_model(Path(output_dir) / 'xgb_model.json')
    
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
        plt.savefig(Path(output_dir) / 'training_history.png')
        plt.close()
    
    def train_xgboost_only(self, data_path, output_dir='models/'):
        try:
            self.logger.info("开始训练XGBoost模型...")
            # 确保输出目录存在
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 加载数据
            df = pd.read_csv(data_path)
            
            # 2. 分离特征和目标
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features]  # 保持为DataFrame以保留特征名
            y = df['wait_time'].values
            
            # 3. 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 4. 训练XGBoost模型
            self.xgb_model = self.build_xgb_model()
            self.xgb_model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=True
            )
            
            # 5. 保存XGBoost模型
            xgb_model_path = output_dir / 'best_xgb_model.json'
            self.xgb_model.save_model(str(xgb_model_path))
            self.logger.info(f"XGBoost模型已保存到: {xgb_model_path}")
            
            # 6. 评估XGBoost模型
            xgb_pred = self.xgb_model.predict(X_test)
            xgb_mae = np.mean(np.abs(xgb_pred - y_test))
            self.logger.info(f"XGBoost MAE: {xgb_mae:.4f}")
            
            # 7. 输出特征重要性
            importance = pd.DataFrame({
                'feature': X.columns,  # 使用DataFrame的列名
                'importance': self.xgb_model.feature_importances_
            })
            self.logger.info("\nXGBoost特征重要性:")
            self.logger.info(importance.sort_values('importance', ascending=False))
            
        except Exception as e:
            self.logger.error(f"XGBoost模型训练失败: {str(e)}")
            raise

def main():
    # 设置路径
    data_path = '/home/lx/LSTMCoal/data/processed/processed_data.csv'
    output_dir = '/home/lx/LSTMCoal/models/'
    
    # 创建并训练模型
    predictor = HybridPredictor(sequence_length=10)
    # predictor.train(data_path, output_dir)
    predictor.train_xgboost_only(data_path, output_dir)

if __name__ == "__main__":
    main()