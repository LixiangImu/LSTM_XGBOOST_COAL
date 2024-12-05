class Config:
    """配置类，存储项目所有配置参数"""
    
    # 路径配置
    RAW_DATA_PATH = 'data/raw/coaltime.csv'
    PROCESSED_DATA_PATH = 'data/processed/processed_data.csv'
    MODEL_SAVE_PATH = 'models/saved_models/'
    
    # 数据处理配置
    DATE_COLUMNS = ['打卡日期', '入口操作日期', '回皮操作日期', '过重操作日期', '出口操作日期']
    TIME_COLUMNS = ['打卡时间', '入口操作时间', '回皮操作时间', '过重操作时间', '出口操作时间']
    TIMESTAMP_PREFIXES = ['打卡', '入口操作', '回皮操作', '过重操作', '出口操作']
    
    # 特征工程配置
    TIME_WINDOWS = [15, 30, 45, 60, 90, 120, 180, 240]  # 分钟
    SEQUENCE_LENGTH = 25
    CATEGORICAL_FEATURES = ['煤种编号']
    
    # LSTM模型参数
    LSTM_PARAMS = {
        'units': [512, 256, 128, 64],
        'dropout_rate': [0.4, 0.3, 0.2, 0.1],
        'batch_size': 64,
        'epochs': 100,
        'validation_split': 0.2,
        'optimizer': 'adam',
        'learning_rate': 0.002,
        'learning_rate_decay': {
            'factor': 0.7,
            'patience': 4,
            'min_lr': 0.00001
        },
        'patience': 15,
        'kernel_regularizer': 0.01,
        'recurrent_regularizer': 0.01,
        
        # 早停配置
        'early_stopping': {
            'monitor': 'val_loss',     # 监控验证集损失
            'min_delta': 0.0005,       # 最小改善阈值
            'patience': 10,            # 容忍多少个epoch没有改善
            'mode': 'min',             # 监控指标是越小越好
            'restore_best_weights': True,  # 恢复最佳权重
            'verbose': 1               # 显示早停信息
        },
        
        # 检查点配置
        'model_checkpoint': {
            'monitor': 'val_loss',
            'save_best_only': True,    # 只保存最佳模型
            'mode': 'min',
            'verbose': 1,
            'save_weights_only': False
        },
        
        # 学习率衰减配置
        'learning_rate_decay': {
            'factor': 0.8,
            'patience': 5,
            'min_lr': 0.00001
        }
    }
    
    # XGBoost参数
    XGB_PARAMS = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 10
    }
    
    # 训练参数
    TRAIN_PARAMS = {
        'test_size': 0.2,
        'random_state': 42,
        'shuffle': True,
        'stratify': True
    }
    
    # 添加交叉验证配置
    CV_PARAMS = {
        'n_splits': 5,
        'shuffle': True
    }
    
    # 模型融合权重
    MODEL_WEIGHTS = {
        'xgboost': 0.5,
        'lstm': 0.5
    }
    
    # 评估指标配置
    METRICS = ['mse', 'mae', 'rmse', 'r2']
    
    # 可视化配置
    PLOT_PARAMS = {
        'figsize': (12, 6),
        'style': 'seaborn',
        'dpi': 100
    }
    
    # 添加新的训练配置
    TRAINING_STRATEGY = {
        'gradient_clip_norm': 1.0,
        'use_weighted_loss': True,
        'loss_weights': {
            'mae': 0.4,
            'mse': 0.6
        }
    }
