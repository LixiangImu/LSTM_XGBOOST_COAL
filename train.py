import os
from pathlib import Path
import logging
import argparse
from src.models.lstm_model import LSTMPredictor
from src.models.xgboost_model import XGBoostPredictor
from datetime import datetime
import pandas as pd

def setup_logging(model_type):
    """配置日志系统"""
    # 创建logs目录
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{model_type}_training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ModelTraining')

def train_lstm(data_path, output_dir, logger):
    """训练LSTM模型"""
    try:
        # 初始化 LSTM 预测器
        lstm_predictor = LSTMPredictor(
            log_dir=output_dir,
            seq_length=24
        )
        
        # 加载数据
        logger.info("正在加载数据...")
        data = pd.read_csv(data_path)
        
        # 准备特征和目标变量
        y = data['wait_time']  # 使用 wait_time 作为目标变量
        X = data.drop('wait_time', axis=1)  # 移除目标变量列
        
        logger.info(f"数据加载完成。特征数量: {X.shape[1]}, 样本数量: {X.shape[0]}")
        
        # 训练模型
        history = lstm_predictor.train(X, y)
        
        logger.info("LSTM模型训练完成")
        return lstm_predictor
        
    except Exception as e:
        logger.error(f"LSTM模型训练失败: {str(e)}")
        raise

def train_xgboost(data_path, output_dir, logger):
    """训练XGBoost模型"""
    try:
        xgb_output = output_dir / 'xgboost'
        xgb_output.mkdir(exist_ok=True)
        
        # 创建模型特定的日志目录
        model_log_dir = Path('logs/xgboost')
        model_log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("开始XGBoost模型训练...")
        xgb_predictor = XGBoostPredictor(
            log_dir=model_log_dir  # 传递日志目录
        )
        xgb_predictor.train(
            data_path=data_path,
            output_dir=xgb_output
        )
        logger.info("XGBoost模型训练完成")
        
    except Exception as e:
        logger.error(f"XGBoost模型训练失败: {str(e)}")
        raise

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型训练脚本')
    parser.add_argument('--model', type=str, choices=['lstm', 'xgboost', 'both'],
                      required=True, help='选择要训练的模型: lstm, xgboost, 或 both')
    parser.add_argument('--data_path', type=str, 
                      default='/home/lx/LSTMCoal/data/processed/processed_data.csv',
                      help='训练数据的路径')
    parser.add_argument('--output_dir', type=str,
                      default='/home/lx/LSTMCoal/models/',
                      help='模型输出目录')
    parser.add_argument('--gpu', type=str, default='6',
                      help='指定要使用的GPU设备号')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置GPU设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    # 设置日志
    logger = setup_logging(args.model)
    
    try:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 根据选择训练相应的模型
        if args.model in ['lstm', 'both']:
            train_lstm(args.data_path, output_dir, logger)
            
        if args.model in ['xgboost', 'both']:
            train_xgboost(args.data_path, output_dir, logger)
            
        logger.info(f"所选模型训练完成，模型文件保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()