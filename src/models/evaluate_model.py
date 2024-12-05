import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model
import seaborn as sns
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, lstm_model, xgb_model, data_path, output_dir, sequence_length=10):
        self.lstm_model_path = Path(lstm_model)
        self.xgb_model_path = Path(xgb_model)
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.setup_output_dir()
        
    def setup_output_dir(self):
        """创建输出目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.eval_dir = self.output_dir / f"evaluation_{timestamp}"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.eval_dir / "plots").mkdir(exist_ok=True)
        (self.eval_dir / "metrics").mkdir(exist_ok=True)
        (self.eval_dir / "predictions").mkdir(exist_ok=True)
        
    def prepare_sequences(self, data):
        """准备LSTM的序列数据"""
        sequences = []
        for i in range(len(data) - self.sequence_length):
            sequence = data[i:(i + self.sequence_length)]
            sequences.append(sequence)
        return np.array(sequences)
    
    def denormalize_time(self, normalized_value, mean=45.0, std=30.0):
        """将标准化的时间值转换回实际分钟数"""
        return normalized_value * std + mean

    def minutes_to_time_format(self, minutes):
        """将分钟转换为时分秒格式"""
        # 处理负数时间
        is_negative = minutes < 0
        minutes = abs(minutes)
        
        hours = int(minutes // 60)
        remaining_minutes = int(minutes % 60)
        seconds = int((minutes % 1) * 60)
        
        time_str = f"{hours:02d}:{remaining_minutes:02d}:{seconds:02d}"
        return f"-{time_str}" if is_negative else time_str

    def evaluate_model(self, y_true, y_pred, model_name):
        """计算评估指标并保存结果"""
        mae = np.mean(np.abs(y_pred - y_true))
        mse = np.mean((y_pred - y_true) ** 2)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # 保存评估指标
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            '最大误差': np.max(np.abs(y_pred - y_true)),
            '最小误差': np.min(np.abs(y_pred - y_true)),
            '平均误差': np.mean(y_pred - y_true),
            '误差标准差': np.std(y_pred - y_true)
        }
        
        pd.Series(metrics).to_csv(self.eval_dir / "metrics" / f"{model_name}_metrics.csv")
        
        # 绘制预测vs实际值散点图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Wait Time (minutes)')  # 修改
        plt.ylabel('Predicted Wait Time (minutes)')  # 修改
        plt.title(f'{model_name} Predicted vs Actual')  # 修改
        plt.savefig(self.eval_dir / "plots" / f"{model_name}_scatter.png")
        plt.close()
        
        # 绘制误差分布图
        errors = y_pred - y_true
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=50)
        plt.xlabel('Prediction Error (minutes)')  # 修改
        plt.ylabel('Frequency')  # 修改
        plt.title(f'{model_name} Prediction Error Distribution')  # 修改
        plt.savefig(self.eval_dir / "plots" / f"{model_name}_error_dist.png")
        plt.close()

        # 保存预测样例
        sample_size = min(100, len(y_true))
        sample_indices = np.random.choice(len(y_true), sample_size, replace=False)
        
        # 反标准化并转换为时间格式
        y_true_denorm = [self.denormalize_time(t) for t in y_true[sample_indices]]
        y_pred_denorm = [self.denormalize_time(p) for p in y_pred[sample_indices]]
        
        samples_data = {
            '实际等待时间': [self.minutes_to_time_format(t) for t in y_true_denorm],
            '预测等待时间': [self.minutes_to_time_format(p) for p in y_pred_denorm],
            '预测误差': [self.minutes_to_time_format(p - t) for p, t in zip(y_pred_denorm, y_true_denorm)],
            '相对误差(%)': [((p - t) / t * 100) if t != 0 else float('inf') 
                          for p, t in zip(y_pred_denorm, y_true_denorm)],
            '原始标准化实际值': y_true[sample_indices].round(4),
            '原始标准化预测值': y_pred[sample_indices].round(4)
        }
        
        samples = pd.DataFrame(samples_data)
        samples.to_csv(self.eval_dir / "predictions" / f"{model_name}_samples.csv", index=False)
        
        return metrics
    
    def evaluate_lstm(self):
        """单独评估LSTM模型"""
        try:
            # 检查模型文件是否存在
            if not self.lstm_model_path.exists():
                raise FileNotFoundError(f"找不到LSTM模型文件: {self.lstm_model_path}")
                
            # 加载数据
            logger.info("加载数据...")
            df = pd.read_csv(self.data_path)
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features]
            y_true = df['wait_time'].values
            
            # 准备LSTM数据
            X_seq = self.prepare_sequences(X.values)
            y_true_lstm = y_true[self.sequence_length:]
            
            # 评估LSTM模型
            logger.info(f"加载LSTM模型: {self.lstm_model_path}")
            lstm_model = load_model(self.lstm_model_path)
            y_pred_lstm = lstm_model.predict(X_seq)
            self.evaluate_model(y_true_lstm, y_pred_lstm.flatten(), 'LSTM')
            
        except Exception as e:
            logger.error(f"LSTM模型评估失败: {str(e)}")
            raise
    
    def evaluate_xgboost(self):
        """单独评估XGBoost模型"""
        try:
            # 检查模型文件是否存在
            if not self.xgb_model_path.exists():
                raise FileNotFoundError(f"找不到XGBoost模型文件: {self.xgb_model_path}")
                
            # 加载数据
            logger.info("加载数据...")
            df = pd.read_csv(self.data_path)
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features]
            y_true = df['wait_time'].values
            
            # 评估XGBoost模型
            logger.info(f"加载XGBoost模型: {self.xgb_model_path}")
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(self.xgb_model_path))
            y_pred_xgb = xgb_model.predict(X)
            self.evaluate_model(y_true, y_pred_xgb, 'XGBoost')
            
        except Exception as e:
            logger.error(f"XGBoost模型评估失败: {str(e)}")
            raise
    
    def evaluate_hybrid(self):
        """单独评估混合模型"""
        try:
            # 加载数据
            logger.info("加载数据...")
            df = pd.read_csv(self.data_path)
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features]
            y_true = df['wait_time'].values
            
            # 准备LSTM数据
            X_seq = self.prepare_sequences(X.values)
            y_true_lstm = y_true[self.sequence_length:]
            
            # 加载和预测LSTM
            lstm_model = load_model(self.models_dir / 'best_lstm_model.keras')
            y_pred_lstm = lstm_model.predict(X_seq)
            
            # 加载和预测XGBoost
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(self.models_dir / 'best_xgb_model.json'))
            y_pred_xgb = xgb_model.predict(X)[self.sequence_length:]
            
            # 混合预测
            y_pred_hybrid = (y_pred_lstm.flatten() + y_pred_xgb) / 2
            self.evaluate_model(y_true_lstm, y_pred_hybrid, 'Hybrid')
            
        except Exception as e:
            logger.error(f"混合模型评估失败: {str(e)}")
            raise
    
    def evaluate_all_models(self):
        """评估所有模型"""
        try:
            # 加载数据
            logger.info("加载数据...")
            df = pd.read_csv(self.data_path)
            features = ['rolling_mean_wait', 'coal_type_mean_wait_30min', 
                       'coal_type_mean_wait_rolling', 'coal_type_mean_wait_1h',
                       'wait_time_ratio', 'coal_type_mean_wait', 
                       'coal_type_truck_count', 'hourly_truck_count',
                       'load_period', 'period_truck_count']
            
            X = df[features]
            y_true = df['wait_time'].values
            
            # 准备LSTM数据
            X_seq = self.prepare_sequences(X.values)
            y_true_lstm = y_true[self.sequence_length:]
            
            # 评估LSTM模型
            logger.info("评估LSTM模型...")
            lstm_model = load_model(self.models_dir / 'best_lstm_model.keras')
            y_pred_lstm = lstm_model.predict(X_seq)
            lstm_metrics = self.evaluate_model(y_true_lstm, y_pred_lstm.flatten(), 'LSTM')
            
            # 评估XGBoost模型
            logger.info("评估XGBoost模型...")
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(self.models_dir / 'best_xgb_model.json'))
            y_pred_xgb = xgb_model.predict(X)
            xgb_metrics = self.evaluate_model(y_true, y_pred_xgb, 'XGBoost')
            
            # 评估混合模型
            logger.info("评估混合模型...")
            # 对于重叠部分的数据进行混合预测
            y_pred_hybrid = (y_pred_lstm + y_pred_xgb[self.sequence_length:]) / 2
            hybrid_metrics = self.evaluate_model(y_true_lstm, y_pred_hybrid, 'Hybrid')
            
            # 在评估之前添加维度检查
            print("y_true_lstm shape:", y_true_lstm.shape)
            print("y_pred_hybrid shape:", y_pred_hybrid.shape)
            
            # 生成比较报告
            comparison = pd.DataFrame({
                'LSTM': lstm_metrics,
                'XGBoost': xgb_metrics,
                'Hybrid': hybrid_metrics
            })
            comparison.to_csv(self.eval_dir / "metrics" / "model_comparison.csv")
            
            # 绘制模型比较图
            plt.figure(figsize=(12, 6))
            comparison.loc[['MAE', 'RMSE', 'R2']].plot(kind='bar')
            plt.title('Model Performance Comparison')  # 修改
            plt.ylabel('Metric Value')  # 修改
            plt.xlabel('Metrics')  # 修改
            plt.tight_layout()
            plt.savefig(self.eval_dir / "plots" / "model_comparison.png")
            plt.close()
            
            logger.info(f"评估完成！结果保存在: {self.eval_dir}")
            
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}")
            raise

def main():
    # 设置路径
    models_dir = '/home/lx/LSTMCoal/models/'
    data_path = '/home/lx/LSTMCoal/data/processed/processed_data.csv'
    output_dir = '/home/lx/LSTMCoal/evaluation/'
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator(models_dir, data_path, output_dir)
    evaluator.evaluate_all_models()

if __name__ == "__main__":
    main()