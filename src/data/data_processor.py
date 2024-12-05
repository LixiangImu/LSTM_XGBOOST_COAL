import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class WaitingTimePreprocessor:
    """等待时间预测数据预处理器"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self._setup_logging()
        self._setup_plot_style()
        
    def _setup_logging(self):
        """配置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('preprocessing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('WaitingTimePreprocessor')
        
    def _setup_plot_style(self):
        """配置绘图样式"""
        plt.style.use('seaborn')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def process_data(self, input_path, output_path='data/processed/'):
        """完整的数据预处理流程"""
        try:
            self.logger.info("开始数据预处理...")
            
            # 1. 加载特征工程后的数据
            df = pd.read_csv(input_path)
            initial_shape = df.shape
            
            # 2. 数据清洗
            df = self._clean_data(df)
            
            # 3. 特征处理
            df = self._process_features(df)
            
            # 4. 特征选择
            df = self._select_features(df)
            
            # 5. 保存处理后的数据
            self._save_processed_data(df, output_path)
            
            # 6. 生成报告和可视化
            self._generate_report(df, output_path)
            self._generate_visualizations(df, output_path)
            
            self.logger.info(f"数据预处理完成: {initial_shape} -> {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {str(e)}")
            raise

    def _clean_data(self, df):
        """清洗数据"""
        initial_len = len(df)
        
        # 1. 删除重复值
        df = df.drop_duplicates()
        
        # 2. 处理异常值
        # 处理等待时间的异常值
        wait_time_cols = ['wait_time', 'back_wait_time', 'weight_wait_time', 'exit_wait_time', 'total_process_time']
        for col in wait_time_cols:
            df = self._remove_outliers(df, col)
        
        # 3. 处理缺失值
        df = df.dropna()
        
        self.logger.info(f"数据清洗: {initial_len} -> {len(df)} 条记录")
        return df
    
    def _remove_outliers(self, df, column, lower_quantile=0.001, upper_quantile=0.999):
        """删除异常值"""
        lower_bound = df[column].quantile(lower_quantile)
        upper_bound = df[column].quantile(upper_quantile)
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def _process_features(self, df):
        """处理特征"""
        # 1. 识别特征类型
        numerical_features = [
            'wait_time', 'back_wait_time', 'weight_wait_time', 'exit_wait_time',
            'total_process_time', 'hourly_mean_wait', 'hourly_std_wait',
            'coal_type_mean_wait', 'coal_type_std_wait', 'period_mean_wait',
            'rolling_mean_wait', 'hourly_truck_count', 'coal_type_truck_count',
            'period_truck_count', 'cumulative_trucks', 'load_period',
            'load_efficiency', 'stage_efficiency'
        ]
        
        ratio_features = [
            'wait_time_ratio', 'back_wait_time_ratio', 
            'weight_wait_time_ratio', 'exit_wait_time_ratio'
        ]
        
        categorical_features = ['煤种编号', 'time_period', 'hour_coal_type', 'period_coal_type']
        binary_features = ['is_weekend', 'is_holiday', 'is_peak_hour']
        
        # 2. 标准化数值特征
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        # 3. 编码分类特征
        for col in categorical_features:
            if col in df.columns:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # 4. 创建统计特征
        df = self._create_statistical_features(df)
        
        return df

    def _create_statistical_features(self, df):
        """创建统计特征"""
        # 确保check_time是datetime类型并设置为索引
        df['check_time'] = pd.to_datetime(df['check_time'])
        df = df.set_index('check_time')
        
        # 使用更小的时间窗口
        df['coal_type_mean_wait_1h'] = df.groupby(['煤种编号'])['wait_time'].transform(
            lambda x: x.rolling(window='1H', min_periods=1).mean()
        )
        df['coal_type_mean_wait_30min'] = df.groupby(['煤种编号'])['wait_time'].transform(
            lambda x: x.rolling(window='30min', min_periods=1).mean()
        )
        
        # 添加动态窗口
        df['coal_type_mean_wait_rolling'] = df.groupby('煤种编号')['wait_time'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
        # 重置索引，将check_time恢复为列
        df = df.reset_index()
        
        return df

    def _select_features(self, df):
        """特征选择"""
        # 1. 保留重要特征
        important_features = [
            'wait_time',  # 目标变量
            'rolling_mean_wait',  # 最相关特征
            'coal_type_mean_wait',
            'coal_type_mean_wait_1h',    # 新增
            'coal_type_mean_wait_30min', # 新增
            'coal_type_mean_wait_rolling', # 新增
            'coal_type_truck_count',
            'hourly_truck_count',
            'load_period',
            'period_truck_count',
            'hourly_mean_wait',
            'back_wait_time',
            'weight_wait_time',
            'exit_wait_time',
            'total_process_time',
            'wait_time_ratio',
            'load_efficiency',
            'stage_efficiency',
            'is_peak_hour',
            'is_weekend',
            'is_holiday',
            'hour',
            'time_period'
        ]
        
        # 2. 检查特征是否存在
        available_features = [col for col in important_features if col in df.columns]
        
        # 3. 检查特征相关性
        correlations = df[available_features].corr()['wait_time'].abs()
        selected_features = correlations[correlations > 0.1].index.tolist()
        
        return df[selected_features]

    def _save_processed_data(self, df, output_path):
        """保存处理后的数据"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / 'processed_data.csv'
        df.to_csv(output_file, index=False)
        self.logger.info(f"处理后的数据已保存至: {output_file}")

    def _generate_report(self, df, output_path):
        """生成数据报告"""
        report = {
            "数据基本信息": {
                "总记录数": len(df),
                "特征数量": len(df.columns),
                "特征列表": df.columns.tolist()
            },
            "等待时间统计(分钟)": {
                "平均等待时间": df['wait_time'].mean(),
                "最短等待时间": df['wait_time'].min(),
                "最长等待时间": df['wait_time'].max(),
                "等待时间标准差": df['wait_time'].std()
            },
            "特征相关性": df.corr()['wait_time'].sort_values(ascending=False).to_dict()
        }
        
        Path(output_path).mkdir(parents=True, exist_ok=True)
        report_file = Path(output_path) / 'preprocessing_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            for section, content in report.items():
                f.write(f"\n=== {section} ===\n")
                f.write(f"{content}\n")

    def _generate_visualizations(self, df, output_path):
        """生成可视化图表"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        # 1. 等待时间分布图
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='wait_time', bins=50)
        plt.title('Waiting Time Distribution')
        plt.xlabel('Waiting Time (Minutes)')
        plt.ylabel('Frequency')
        plt.savefig(Path(output_path) / 'wait_time_distribution.png')
        plt.close()
        
        # 2. 特征相关性热力图
        plt.figure(figsize=(15, 12))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(Path(output_path) / 'correlation_matrix.png')
        plt.close()
        
        # 3. 各阶段等待时间箱线图
        # 检查哪些等待时间列存在
        wait_time_cols = ['wait_time', '入口等待时间', '回皮等待时间', '过重等待时间', '出口等待时间']
        wait_time_labels = ['Entry Wait', 'Back Wait', 'Weight Wait', 'Exit Wait']
        
        available_cols = [col for col in wait_time_cols if col in df.columns]
        if len(available_cols) > 1:  # 至少需要两列才能画箱线图
            plt.figure(figsize=(12, 6))
            df[available_cols].boxplot()
            plt.title('Waiting Time Distribution by Stage')
            plt.ylabel('Time (Minutes)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(Path(output_path) / 'wait_time_stages.png')
            plt.close()
            self.logger.info(f"生成了等待时间阶段分布图，包含以下列: {available_cols}")
        else:
            self.logger.warning("数据中缺少足够的等待时间列，跳过等待时间阶段分布图的生成")

def main():
    # 设置路径
    input_file = '/home/lx/LSTMCoal/data/featured/featured_data.csv'
    output_dir = '/home/lx/LSTMCoal/data/processed/'
    
    # 创建预处理器并处理数据
    preprocessor = WaitingTimePreprocessor()
    processed_df = preprocessor.process_data(input_file, output_dir)
    
    # 显示处理结果摘要
    print("\n=== 数据预处理完成 ===")
    print(f"处理后数据形状: {processed_df.shape}")
    print("\n最终特征列表:")
    print(processed_df.columns.tolist())
    
    print("\n特征相关性:")
    correlations = processed_df.corr()['wait_time'].sort_values(ascending=False)
    print(correlations)

if __name__ == "__main__":
    main()