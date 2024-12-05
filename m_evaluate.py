import os
from pathlib import Path
import logging
import argparse
from datetime import datetime
from src.models.evaluate_model import ModelEvaluator

def setup_logging(model_type):
    """配置日志系统"""
    # 创建logs目录
    log_dir = Path('logs/evaluation')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{model_type}_evaluation_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('ModelEvaluation')

def evaluate_model(model_type, data_path, lstm_model, xgb_model, output_dir, logger):
    """评估指定的模型"""
    try:
        evaluator = ModelEvaluator(
            lstm_model=lstm_model,
            xgb_model=xgb_model,
            data_path=data_path,
            output_dir=output_dir,
            sequence_length=10
        )
        
        if model_type == 'lstm':
            logger.info("开始评估LSTM模型...")
            evaluator.evaluate_lstm()
        elif model_type == 'xgboost':
            logger.info("开始评估XGBoost模型...")
            evaluator.evaluate_xgboost()
        elif model_type == 'hybrid':
            logger.info("开始评估混合模型...")
            evaluator.evaluate_hybrid()
        elif model_type == 'all':
            logger.info("开始评估所有模型...")
            evaluator.evaluate_all_models()
            
        logger.info(f"模型评估完成，结果保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"模型评估失败: {str(e)}")
        raise

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='模型评估脚本')
    parser.add_argument('--model', type=str, 
                      choices=['lstm', 'xgboost', 'hybrid', 'all'],
                      required=True, 
                      help='选择要评估的模型: lstm, xgboost, hybrid, 或 all')
    parser.add_argument('--data_path', type=str, 
                      default='/home/lx/LSTMCoal/data/processed/processed_data.csv',
                      help='测试数据的路径')
    parser.add_argument('--lstm_model', type=str,
                      default='/home/lx/LSTMCoal/models/best_lstm_model.keras',
                      help='LSTM模型文件路径')
    parser.add_argument('--xgb_model', type=str,
                      default='/home/lx/LSTMCoal/models/best_xgb_model.json',
                      help='XGBoost模型文件路径')
    parser.add_argument('--output_dir', type=str,
                      default='/home/lx/LSTMCoal/evaluation/',
                      help='评估结果输出目录')
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
        
        # 评估模型
        evaluate_model(
            model_type=args.model,
            data_path=args.data_path,
            lstm_model=args.lstm_model,
            xgb_model=args.xgb_model,
            output_dir=output_dir,
            logger=logger
        )
        
    except Exception as e:
        logger.error(f"评估过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()