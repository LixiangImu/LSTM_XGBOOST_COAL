import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Union
import logging
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Metrics')

class ModelEvaluator:
    """模型评估类，提供各种评估指标的计算"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, 
                         y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算多个评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        try:
            metrics = {
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred),
                'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            }
            
            logger.info("评估指标计算完成")
            return metrics
            
        except Exception as e:
            logger.error(f"计算评估指标失败: {str(e)}")
            raise
    
    @staticmethod
    def calculate_group_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              groups: np.ndarray) -> pd.DataFrame:
        """
        按组计算评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            groups: 分组标签
            
        Returns:
            pd.DataFrame: 分组评估指标
        """
        try:
            results = []
            unique_groups = np.unique(groups)
            
            for group in unique_groups:
                mask = groups == group
                group_metrics = ModelEvaluator.calculate_metrics(
                    y_true[mask], y_pred[mask]
                )
                group_metrics['group'] = group
                results.append(group_metrics)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"计算分组评估指标失败: {str(e)}")
            raise
    
    @staticmethod
    def calculate_time_window_metrics(y_true: np.ndarray,
                                    y_pred: np.ndarray,
                                    timestamps: np.ndarray,
                                    window: str = 'D') -> pd.DataFrame:
        """
        计算时间窗口评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            timestamps: 时间戳
            window: 时间窗口大小
            
        Returns:
            pd.DataFrame: 时间窗口评估指标
        """
        try:
            df = pd.DataFrame({
                'timestamp': timestamps,
                'y_true': y_true,
                'y_pred': y_pred
            })
            
            results = []
            for name, group in df.groupby(pd.Grouper(key='timestamp', freq=window)):
                metrics = ModelEvaluator.calculate_metrics(
                    group['y_true'].values,
                    group['y_pred'].values
                )
                metrics['timestamp'] = name
                results.append(metrics)
            
            return pd.DataFrame(results)
            
        except Exception as e:
            logger.error(f"计算时间窗口评估指标失败: {str(e)}")
            raise