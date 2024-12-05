import logging
from pathlib import Path
from typing import Optional

class Logger:
    """日志工具类，提供日志记录功能"""
    
    def __init__(self, name: str, log_dir: Optional[str] = 'logs', log_level: int = logging.INFO):
        """
        初始化日志工具
        
        Args:
            name: 日志记录器名称
            log_dir: 日志文件保存目录
            log_level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # 创建日志目录
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_file = Path(log_dir) / f"{name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取日志记录器"""
        return self.logger