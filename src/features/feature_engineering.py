import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import logging
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        self.cn_holidays = holidays.CN()
        self._setup_logging()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('feature_engineering.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('FeatureEngineer')

    def create_features(self, input_path, output_path='data/featured/'):
        """特征工程主流程"""
        try:
            self.logger.info("开始特征工程...")
            
            # 1. 加载数据
            df = pd.read_csv(input_path)
            initial_shape = df.shape
            
            # 2. 创建基础时间特征
            df = self._create_time_features(df)
            
            # 3. 创建统计特征
            df = self._create_statistical_features(df)
            
            # 4. 创建负载特征
            df = self._create_load_features(df)
            
            # 5. 创建交互特征
            df = self._create_interaction_features(df)
            
            # 6. 保存特征工程结果
            self._save_featured_data(df, output_path)
            
            self.logger.info(f"特征工程完成: {initial_shape} -> {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"特征工程失败: {str(e)}")
            raise

    def _create_time_features(self, df):
        """创建时间相关特征"""
        # 转换时间格式
        df['check_time'] = pd.to_datetime(df['打卡操作日期'] + ' ' + df['打卡操作时间'])
        df['operation_time'] = pd.to_datetime(df['入口操作日期'] + ' ' + df['入口操作时间'])
        df['back_time'] = pd.to_datetime(df['回皮操作日期'] + ' ' + df['回皮操作时间'])
        df['weight_time'] = pd.to_datetime(df['过重操作日期'] + ' ' + df['过重操作时间'])
        df['exit_time'] = pd.to_datetime(df['出口操作日期'] + ' ' + df['出口操作时间'])
        
        # 基础时间特征
        df['hour'] = df['check_time'].dt.hour
        df['minute'] = df['check_time'].dt.minute
        df['weekday'] = df['check_time'].dt.dayofweek
        df['month'] = df['check_time'].dt.month
        df['day'] = df['check_time'].dt.day
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['is_holiday'] = df['check_time'].map(lambda x: int(x in self.cn_holidays))
        
        # 时间段特征
        df['time_period'] = pd.cut(df['hour'], 
                                 bins=[-1, 6, 12, 18, 24], 
                                 labels=['凌晨', '上午', '下午', '晚上'])
        
        # 计算等待时间（分钟）
        df['wait_time'] = (df['operation_time'] - df['check_time']).dt.total_seconds() / 60
        
        # 新增：计算各阶段等待时间
        df['back_wait_time'] = (df['back_time'] - df['operation_time']).dt.total_seconds() / 60
        df['weight_wait_time'] = (df['weight_time'] - df['back_time']).dt.total_seconds() / 60
        df['exit_wait_time'] = (df['exit_time'] - df['weight_time']).dt.total_seconds() / 60
        df['total_process_time'] = (df['exit_time'] - df['check_time']).dt.total_seconds() / 60
        
        return df

    def _create_statistical_features(self, df):
        """创建统计特征"""
        # 按小时统计
        df['hourly_mean_wait'] = df.groupby('hour')['wait_time'].transform('mean')
        df['hourly_std_wait'] = df.groupby('hour')['wait_time'].transform('std')
        
        # 按煤种统计
        df['coal_type_mean_wait'] = df.groupby('煤种编号')['wait_time'].transform('mean')
        df['coal_type_std_wait'] = df.groupby('煤种编号')['wait_time'].transform('std')
        
        # 按时间段统计
        df['period_mean_wait'] = df.groupby('time_period')['wait_time'].transform('mean')
        
        # 滑动窗口统计
        df['rolling_mean_wait'] = df.groupby('煤种编号')['wait_time'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # 新增：各阶段等待时间的统计特征
        for stage in ['back_wait_time', 'weight_wait_time', 'exit_wait_time']:
            df[f'{stage}_mean'] = df.groupby('hour')[stage].transform('mean')
            df[f'{stage}_std'] = df.groupby('hour')[stage].transform('std')
        
        # 新增：总处理时间的统计特征
        df['total_time_mean'] = df.groupby('hour')['total_process_time'].transform('mean')
        df['total_time_std'] = df.groupby('hour')['total_process_time'].transform('std')
        
        return df

    def _create_load_features(self, df):
        """创建负载特征"""
        # 每小时车辆数
        df['hourly_truck_count'] = df.groupby([pd.to_datetime(df['打卡操作日期']).dt.date, 
                                             df['hour']])['提煤单号'].transform('count')
        
        # 每种煤型的车辆数
        df['coal_type_truck_count'] = df.groupby(['煤种编号', 
                                                pd.to_datetime(df['打卡操作日期']).dt.date])['提煤单号'].transform('count')
        
        # 高峰期标记
        df['is_peak_hour'] = df['hour'].isin([8,9,10,14,15,16]).astype(int)
        
        # 新增：时段负载特征
        df['period_truck_count'] = df.groupby([pd.to_datetime(df['打卡操作日期']).dt.date, 
                                             'time_period'])['提煤单号'].transform('count')
        
        df['coal_type_period_count'] = df.groupby(['煤种编号', 
                                                 'time_period'])['提煤单号'].transform('count')
        
        # 新增：累计负载
        df['cumulative_trucks'] = df.groupby(pd.to_datetime(df['打卡操作日期']).dt.date)['提煤单号'].cumcount()
        
        return df

    def _create_interaction_features(self, df):
        """创建交互特征"""
        # 时间与煤种交互
        df['hour_coal_type'] = df['hour'].astype(str) + '_' + df['煤种编号']
        
        # 时间段与煤种交互
        df['period_coal_type'] = df['time_period'].astype(str) + '_' + df['煤种编号']
        
        # 负载与时间段交互
        df['load_period'] = df['hourly_truck_count'] * df['period_mean_wait']
        
        # 新增：时间与负载的交互特征
        df['load_efficiency'] = df['wait_time'] / (df['hourly_truck_count'] + 1)  # 加1避免除零
        df['stage_efficiency'] = df['total_process_time'] / (df['period_truck_count'] + 1)
        
        # 新增：各阶段时间占比
        total_wait = df[['wait_time', 'back_wait_time', 'weight_wait_time', 'exit_wait_time']].sum(axis=1)
        for stage in ['wait_time', 'back_wait_time', 'weight_wait_time', 'exit_wait_time']:
            df[f'{stage}_ratio'] = df[stage] / total_wait
        
        return df

    def _save_featured_data(self, df, output_path):
        """保存特征工程后的数据"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / 'featured_data.csv'
        df.to_csv(output_file, index=False)
        self.logger.info(f"特征工程后的数据已保存至: {output_file}")

def main():
    # 设置路径
    input_file = '/home/lx/LSTMCoal/data/raw/coal_data.csv'
    output_dir = '/home/lx/LSTMCoal/data/featured/'
    
    # 创建特征工程器并处理数据
    engineer = FeatureEngineer()
    featured_df = engineer.create_features(input_file, output_dir)
    
    # 显示特征工程结果
    print("\n=== 特征工程完成 ===")
    print(f"处理后数据形状: {featured_df.shape}")
    print("\n新增特征列表:")
    print([col for col in featured_df.columns if col not in ['提煤单号', '煤种编号', '打卡操作时间', '打卡操作日期']])
    
    # 显示与等待时间的相关性
    correlations = featured_df.corr()['wait_time'].sort_values(ascending=False)
    print("\n与等待时间最相关的前10个特征:")
    print(correlations[:10])

if __name__ == "__main__":
    main()