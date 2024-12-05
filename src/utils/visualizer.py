import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Visualizer')

class Visualizer:
    """可视化工具类，提供各种数据可视化方法"""
    
    def __init__(self, save_dir: str = 'reports/figures'):
        """
        初始化可视化工具
        
        Args:
            save_dir: 图表保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置默认样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_prediction_vs_actual(self,
                                y_true: np.ndarray,
                                y_pred: np.ndarray,
                                title: str = "预测值 vs 实际值",
                                save_name: Optional[str] = None) -> None:
        """
        绘制预测值与实际值的对比图
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            save_name: 保存文件名
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()],
                    [y_true.min(), y_true.max()],
                    'r--', lw=2)
            
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title(title)
            
            if save_name:
                plt.savefig(self.save_dir / save_name)
                logger.info(f"图表已保存: {save_name}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制预测对比图失败: {str(e)}")
            raise
    
    def plot_feature_importance(self,
                              importance_df: pd.DataFrame,
                              top_n: int = 20,
                              title: str = "特征重要性",
                              save_name: Optional[str] = None) -> None:
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性数据框
            top_n: 显示前N个特征
            title: 图表标题
            save_name: 保存文件名
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # 获取前N个特征
            df_plot = importance_df.head(top_n)
            
            sns.barplot(x='importance', y='feature', data=df_plot)
            plt.title(title)
            plt.xlabel('重要性')
            plt.ylabel('特征')
            
            if save_name:
                plt.savefig(self.save_dir / save_name)
                logger.info(f"图表已保存: {save_name}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制特征重要性图失败: {str(e)}")
            raise
    
    def plot_time_series(self,
                        timestamps: np.ndarray,
                        values: Union[np.ndarray, List[np.ndarray]],
                        labels: List[str],
                        title: str = "时间序列图",
                        save_name: Optional[str] = None) -> None:
        """
        绘制时间序列图
        
        Args:
            timestamps: 时间戳
            values: 一个或多个时间序列值
            labels: 图例标签
            title: 图表标题
            save_name: 保存文件名
        """
        try:
            plt.figure(figsize=(15, 6))
            
            if not isinstance(values, list):
                values = [values]
            
            for value, label in zip(values, labels):
                plt.plot(timestamps, value, label=label)
            
            plt.title(title)
            plt.xlabel('时间')
            plt.ylabel('值')
            plt.legend()
            plt.xticks(rotation=45)
            
            if save_name:
                plt.savefig(self.save_dir / save_name)
                logger.info(f"图表已保存: {save_name}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制时间序列图失败: {str(e)}")
            raise