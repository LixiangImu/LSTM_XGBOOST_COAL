import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import logging
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class XGBoostPredictor:
    def __init__(self, log_dir=None):
        self.model = None
        self.log_dir = Path(log_dir) if log_dir else Path('logs/xgboost')
        self._setup_logging()
    
    def _setup_logging(self):
        """配置日志系统"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成带时间戳的日志文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.log_dir / f'xgboost_training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('XGBoostPredictor')
    
    def build_model(self):
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
            self.logger.info("开始训练XGBoost模型...")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 加载数据
            df = pd.read_csv(data_path)
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features]
            y = df['wait_time'].values
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # 训练模型
            self.model = self.build_model()
            self.model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=True
            )
            
            # 评估模型
            self._evaluate_model(X_test, y_test)
            
            # 保存模型
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = output_dir / f'xgboost_model_{timestamp}.json'
            self.model.save_model(str(model_path))
            self.logger.info(f"模型已保存到: {model_path}")
            
            # 输出并保存特征重要性
            self._show_feature_importance(X.columns)
            
        except Exception as e:
            self.logger.error(f"XGBoost模型训练失败: {str(e)}")
            raise
    
    def predict(self, X):
        return self.model.predict(X)
    
    def _evaluate_model(self, X_test, y_test):
        pred = self.model.predict(X_test)
        mae = np.mean(np.abs(pred - y_test))
        self.logger.info(f"XGBoost MAE: {mae:.4f}")
    
    def _show_feature_importance(self, feature_names):
        # 计算特征重要性
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 记录到日志
        self.logger.info("\nXGBoost特征重要性:")
        self.logger.info(importance)
        
        # 绘制特征重要性图
        plt.figure(figsize=(10, 6))
        importance.plot(x='feature', y='importance', kind='bar')
        plt.title('XGBoost Feature Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(self.log_dir / f'feature_importance_{timestamp}.png')
        plt.close()
        
        # 保存特征重要性数据
        importance.to_csv(self.log_dir / f'feature_importance_{timestamp}.csv', index=False)