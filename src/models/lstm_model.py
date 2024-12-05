import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

class LSTMPredictor:
    def __init__(self, log_dir, seq_length=24):
        """初始化 LSTM 预测器
        
        Args:
            log_dir: 日志和模型保存路径
            seq_length: 序列长度，默认为24
        """
        self.log_dir = log_dir
        self.seq_length = seq_length
        self.is_fitted = False
        self.scaler = RobustScaler()
        self.feature_scaler = RobustScaler()
        
    def preprocess_data(self, df):
        """增强的数据预处理"""
        # 1. 特征工程
        df = df.copy()  # 创建副本避免修改原始数据
        df['time_of_day'] = pd.to_datetime(df['hour'], format='%H').dt.hour
        df['is_peak_hour'] = df['hour'].between(9, 17).astype(int)
        
        # 2. 选择最重要的特征
        selected_features = [
            'rolling_mean_wait', 
            'coal_type_mean_wait',
            'hourly_truck_count',
            'load_period',
            'wait_time_ratio',
            'time_of_day',
            'is_peak_hour'
        ]
        
        # 3. 处理异常值
        for col in selected_features:
            if col in df.columns:  # 只处理存在的列
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df[col] = df[col].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
        
        return df[selected_features]

    def create_sequences(self, data, target):
        """创建序列数据"""
        X, y = [], []
        for i in range(len(data) - self.seq_length):
            X.append(data[i:(i + self.seq_length)])
            y.append(target[i + self.seq_length])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """进一步优化的模型架构"""
        # 创建输入层
        inputs = tf.keras.Input(shape=input_shape)
        
        # LSTM层 + 残差连接
        x = LSTM(256, return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)
        
        # 双向LSTM + 注意力机制
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(128, return_sequences=True,
                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        )(x)
        
        # 添加注意力层
        attention = tf.keras.layers.Dense(1)(lstm_out)
        attention = tf.keras.layers.Softmax(axis=1)(attention)
        attended = tf.keras.layers.Multiply()([lstm_out, attention])
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(attended)
        
        # 全连接层
        x = Dense(64, activation='selu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='selu',
                  kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        
        # 输出层
        outputs = Dense(1)(x)
        
        # 创建模型
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # 使用自定义的优化器配置
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0005,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # 编译模型
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )
        
        return model

    def train(self, X, y):
        """优化后的训练过程"""
        # 1. 数据预处理
        X_processed = self.preprocess_data(X)
        
        # 2. 特征缩放
        X_scaled = self.feature_scaler.fit_transform(X_processed)
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))
        
        # 3. 创建序列
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        
        # 4. 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 5. 训练模型
        best_val_loss = float('inf')
        best_model = None
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y_seq[train_idx], y_seq[val_idx]
            
            # 构建模型
            model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                ModelCheckpoint(
                    str(self.log_dir / f'best_model_fold_{fold}.keras'),
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min',
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=7,
                    min_lr=1e-6,
                    verbose=1,
                    cooldown=3
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.log_dir / 'tensorboard' / f'fold_{fold}'),
                    histogram_freq=1,
                    update_freq='epoch',
                    profile_batch=2
                )
            ]
            
            # 训练模型
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=48,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            val_loss = min(history.history['val_loss'])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
        
        self.model = best_model
        self.is_fitted = True
        
        return history

    def predict(self, X):
        """预测方法"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
            
        X_processed = self.preprocess_data(X)
        X_scaled = self.feature_scaler.transform(X_processed)
        X_seq, _ = self.create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        predictions = self.model.predict(X_seq)
        return self.scaler.inverse_transform(predictions)