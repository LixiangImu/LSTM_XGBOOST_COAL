from pathlib import Path

class Config:
    """配置类"""
    
    def __init__(self):
        # 项目根目录
        self.ROOT_DIR = Path(__file__).parent.parent.parent
        
        # 数据目录
        self.RAW_DATA_DIR = self.ROOT_DIR / 'data' / 'raw'
        self.PROCESSED_DATA_DIR = self.ROOT_DIR / 'data' / 'processed'
        
        # 确保目录存在
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # 时间戳前缀
        self.TIMESTAMP_PREFIXES = ['打卡', '入口操作', '回皮操作']
        
        # 类别特征
        self.CATEGORICAL_FEATURES = ['煤种编号']