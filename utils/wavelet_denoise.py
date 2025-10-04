import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
warnings.filterwarnings("ignore")

# 定义全局变量，用于不同类型的数据列
price_columns = ['Open', 'Close', 'High', 'Low', 'Avg']
volume_columns = ['Volume', 'Turnover']
change_columns = ['PricechangeRate', 'TurnoverRate']

# 添加小波降噪函数
def wavelet_denoising(data, wavelet='db4', level=1, threshold_mode='soft', threshold_scale=1.0):
    """
    使用小波变换对时间序列数据进行降噪
    
    参数:
    data: 输入的一维时间序列数据
    wavelet: 小波基函数，默认为'db4'
    level: 分解级别，默认为1
    threshold_mode: 阈值处理模式，'soft'或'hard'
    threshold_scale: 阈值缩放因子，值越小降噪效果越温和
    
    返回:
    降噪后的时间序列数据
    """
    # 确保数据是numpy数组
    data = np.array(data, dtype=float)
    
    # 计算数据的标准差和范围，用于调试输出
    data_std = np.std(data)
    data_range = np.max(data) - np.min(data)
    
    # 进行小波分解
    coeffs = pywt.wavedec(data, wavelet, level=level)
    
    # 计算阈值
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(data))) * threshold_scale
    
    # 打印调试信息
    print(f"小波降噪 - 使用 {wavelet} 小波，级别 {level}，阈值模式 {threshold_mode}")
    print(f"数据范围: {data_range:.2f}, 标准差: {data_std:.2f}")
    print(f"计算的阈值: {threshold:.4f}")
    
    # 应用阈值处理 - 对所有级别的系数应用阈值，而不仅仅是细节系数
    new_coeffs = list(coeffs)
    for i in range(1, len(coeffs)):  # 从1开始，保留近似系数
        # 打印每个级别的系数变化
        before_threshold = np.sum(np.abs(coeffs[i]))
        new_coeffs[i] = pywt.threshold(coeffs[i], threshold, threshold_mode)
        after_threshold = np.sum(np.abs(new_coeffs[i]))
        change_percent = (before_threshold - after_threshold) / before_threshold * 100 if before_threshold > 0 else 0
        print(f"  级别 {i} 系数变化: {change_percent:.2f}%")
    
    # 重构信号
    denoised_data = pywt.waverec(new_coeffs, wavelet)
    
    # 确保输出长度与输入相同
    if len(denoised_data) != len(data):
        denoised_data = denoised_data[:len(data)]
    
    # 计算降噪前后的差异
    diff = np.mean(np.abs(denoised_data - data))
    diff_percent = (diff / np.mean(np.abs(data))) * 100 if np.mean(np.abs(data)) > 0 else 0
    print(f"降噪前后平均绝对差异: {diff:.2f} ({diff_percent:.2f}%)")
    
    return denoised_data

# 添加辅助函数，用于确保数据长度一致
def ensure_length_match(data, target_length, column_name=None):
    """
    确保数据长度与目标长度一致
    
    参数:
    data: 需要调整的数据数组
    target_length: 目标长度
    column_name: 列名，用于打印警告信息
    
    返回:
    调整后的数据数组
    """
    if len(data) != target_length:
        if column_name:
            print(f"警告: {column_name} 的降噪数据长度 ({len(data)}) 与目标长度 ({target_length}) 不匹配，进行调整")
        
        # 如果长度大于目标长度，截断多余部分
        if len(data) > target_length:
            return data[:target_length]
        # 如果长度小于目标长度，通过复制最后一个值来填充
        elif len(data) < target_length:
            padding = np.full(target_length - len(data), data[-1])
            return np.concatenate([data, padding])
    
    return data

# 添加批量小波降噪函数，用于处理DataFrame中的多个列
def apply_wavelet_denoising(df, columns, wavelet='db4', level=1, threshold_mode='soft', threshold_scale=1.0):
    """
    对DataFrame中的多个列应用小波降噪
    
    参数:
    df: 输入的DataFrame
    columns: 需要进行降噪的列名列表
    wavelet: 小波基函数，默认为'db4'
    level: 分解级别，默认为1
    threshold_mode: 阈值处理模式，'soft'或'hard'
    threshold_scale: 阈值缩放因子，值越小降噪效果越温和
    
    返回:
    处理后的DataFrame
    """
    # 引用全局变量
    global price_columns, volume_columns, change_columns
    
    df_denoised = df.copy()
    for col in columns:
        if col in df.columns:
            # 确保数据没有NaN值
            data = df[col].fillna(method='ffill').fillna(method='bfill').values
            
            # 应用小波降噪
            print(f"\n处理列: {col}")
            denoised_data = wavelet_denoising(data, wavelet, level, threshold_mode, threshold_scale)
            
            # 确保降噪后的数据长度与原始数据长度一致
            denoised_data = ensure_length_match(denoised_data, len(df), col)
            
            # 添加保护措施：确保降噪后的数据不会偏离原始数据太多
            # 对于交易量和价格变化率，限制最大偏差
            if col in volume_columns:
                # 限制交易量数据的最大偏差为5%
                max_deviation = 0.05
                original_data = df[col].values
                for i in range(len(denoised_data)):
                    if original_data[i] != 0:  # 避免除以零
                        deviation = abs(denoised_data[i] - original_data[i]) / original_data[i]
                        if deviation > max_deviation:
                            # 将偏差限制在最大允许范围内
                            direction = 1 if denoised_data[i] > original_data[i] else -1
                            denoised_data[i] = original_data[i] * (1 + direction * max_deviation)
            
            elif col in change_columns:
                # 对于价格变化率，使用更严格的限制
                max_deviation = 0.2  # 最大允许偏差为原始值的20%
                original_data = df[col].values
                for i in range(len(denoised_data)):
                    if abs(original_data[i]) > 0.001:  # 避免对接近零的值进行处理
                        deviation = abs(denoised_data[i] - original_data[i]) / abs(original_data[i])
                        if deviation > max_deviation:
                            # 将偏差限制在最大允许范围内
                            direction = 1 if denoised_data[i] > original_data[i] else -1
                            denoised_data[i] = original_data[i] * (1 + direction * max_deviation)
            
            # 更新DataFrame
            df_denoised[col] = denoised_data
    
    return df_denoised

def apply_wavelet_denoising_to_dataframe(df):
    """
    对DataFrame应用小波降噪处理，针对不同类型的数据使用不同的降噪参数
    
    参数:
    df: 输入的DataFrame，包含股票数据
    
    返回:
    处理后的DataFrame
    """
    print("正在应用小波降噪处理...")
    
    # 对价格和交易量数据应用小波降噪
    price_columns = ['Open', 'Close', 'High', 'Low', 'Avg']
    volume_columns = ['Volume', 'Turnover']
    change_columns = ['PricechangeRate', 'TurnoverRate']

    # 对价格数据使用较低级别的分解以保留更多细节
    df = apply_wavelet_denoising(df, price_columns, wavelet='db8', level=3, threshold_mode='soft')
    
    # 对交易量数据使用更温和的降噪参数，减小差异
    df = apply_wavelet_denoising(df, volume_columns, wavelet='sym4', level=1, threshold_mode='soft', threshold_scale=0.2)
    
    # 对变化率数据使用更温和的降噪参数
    df = apply_wavelet_denoising(df, change_columns, wavelet='sym4', level=2, threshold_mode='soft', threshold_scale=0.1)

    print("小波降噪处理完成")
    return df

def apply_wavelet_denoising_to_indicators(df, indicators_dict=None):
    """
    对DataFrame中的技术指标应用小波降噪
    
    参数:
    df: pandas.DataFrame，包含需要降噪的数据
    indicators_dict: 字典，键为指标类型，值为列名列表，默认为None（使用预定义的分类）
    
    返回:
    pandas.DataFrame: 降噪后的DataFrame
    """
    print("正在对技术指标应用小波降噪处理...")
    
    # 创建DataFrame的副本，避免修改原始数据
    df_denoised = df.copy()
    
   # 如果没有提供指标字典，使用默认分类
    if indicators_dict is None:
        indicators_dict = {
            'trend_indicators': ['MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200', 'EMA_12', 'EMA_26', 'TEMA'],
            'momentum_indicators': ['RSI', 'MACD', 'Signal', 'Momentum', 'ROC', 'Williams_R', 'CCI', 'MFI'],
            'volatility_indicators': ['ATR', 'STD', 'Upper_Band', 'Lower_Band', '%B', 'Volatility'],
            'volume_indicators': ['OBV', 'AD', 'AVG_VOL'],
            'oscillator_indicators': ['K', 'D', 'J', 'Stoch_K', 'Stoch_D', 'TRIX'],
            'custom_indicators': ['ProfitRatio', 'LWinnerRatio', 'PPSCM', 'SHCM', 'ZSHTL', 'TTSLKP', 'ZCMPPS', 'ZSHJJ', 'TTSLJJ', 'ZJLRQD', 'SH8', 'lijinz1', 'lijinz4'],
            'chip_indicators': ['Cost90', 'Cost70', 'ProfitRatio', 'LWinnerRatio'],
            'index_indicators': ['open_SZ_index', 'high_SZ_index', 'low_SZ_index', 'close_SZ_index', 'volume_SZ_index'],
            'industry_indicators': ['open_industry_index', 'high_industry_index', 'low_industry_index', 'close_industry_index', 'volume_industry_index', 'turnover_industry_index'],
            'forex_indicators': ['open_forex', 'high_forex', 'low_forex', 'new_forex'],
            'us_market_indicators': ['open_us', 'high_us', 'low_us', 'close_us', 'volume_us']
        }
    
    
    # 对不同类型的指标应用不同参数的小波降噪
    # 趋势指标 - 使用较低级别的降噪，保留长期趋势
    if 'trend_indicators' in indicators_dict:
        trend_cols = [col for col in indicators_dict['trend_indicators'] if col in df.columns]
        if trend_cols:
            print(f"\n处理趋势指标: {', '.join(trend_cols)}")
            for col in trend_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    denoised_data = wavelet_denoising(data, wavelet='sym8', level=2, threshold_mode='soft')
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 动量指标 - 使用中等级别的降噪
    if 'momentum_indicators' in indicators_dict:
        momentum_cols = [col for col in indicators_dict['momentum_indicators'] if col in df.columns]
        if momentum_cols:
            print(f"\n处理动量指标: {', '.join(momentum_cols)}")
            for col in momentum_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    denoised_data = wavelet_denoising(data, wavelet='db4', level=3, threshold_mode='soft')
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 波动性指标 - 使用较高级别的降噪，平滑波动
    if 'volatility_indicators' in indicators_dict:
        volatility_cols = [col for col in indicators_dict['volatility_indicators'] if col in df.columns]
        if volatility_cols:
            print(f"\n处理波动性指标: {', '.join(volatility_cols)}")
            for col in volatility_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    denoised_data = wavelet_denoising(data, wavelet='db8', level=3, threshold_mode='soft')
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 成交量指标 - 使用较高级别的降噪，平滑成交量波动
    if 'volume_indicators' in indicators_dict:
        volume_cols = [col for col in indicators_dict['volume_indicators'] if col in df.columns]
        if volume_cols:
            print(f"\n处理成交量指标: {', '.join(volume_cols)}")
            for col in volume_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    denoised_data = wavelet_denoising(data, wavelet='db8', level=4, threshold_mode='soft')
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 振荡器指标 - 使用中等级别的降噪
    if 'oscillator_indicators' in indicators_dict:
        oscillator_cols = [col for col in indicators_dict['oscillator_indicators'] if col in df.columns]
        if oscillator_cols:
            print(f"\n处理振荡器指标: {', '.join(oscillator_cols)}")
            for col in oscillator_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    denoised_data = wavelet_denoising(data, wavelet='sym4', level=2, threshold_mode='soft')
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 自定义指标 - 使用适合的降噪参数
    if 'custom_indicators' in indicators_dict:
        custom_cols = [col for col in indicators_dict['custom_indicators'] if col in df.columns]
        if custom_cols:
            print(f"\n处理自定义指标: {', '.join(custom_cols)}")
            for col in custom_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    denoised_data = wavelet_denoising(data, wavelet='sym8', level=2, threshold_mode='soft')
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)

     # 筹码分布指标 - 使用特定参数的小波降噪
    if 'chip_indicators' in indicators_dict:
        chip_cols = [col for col in indicators_dict['chip_indicators'] if col in df.columns]
        if chip_cols:
            print(f"\n处理筹码分布指标: {', '.join(chip_cols)}")
            for col in chip_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    # 对筹码分布使用较温和的降噪参数
                    denoised_data = wavelet_denoising(data, wavelet='sym8', level=2, threshold_mode='soft', threshold_scale=0.3)
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 上证指数指标 - 使用特定参数的小波降噪
    if 'index_indicators' in indicators_dict:
        index_cols = [col for col in indicators_dict['index_indicators'] if col in df.columns]
        if index_cols:
            print(f"\n处理上证指数指标: {', '.join(index_cols)}")
            for col in index_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    # 对指数数据使用较低级别的降噪，保留趋势
                    denoised_data = wavelet_denoising(data, wavelet='db8', level=2, threshold_mode='soft', threshold_scale=0.5)
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 行业指数指标 - 使用特定参数的小波降噪
    if 'industry_indicators' in indicators_dict:
        industry_cols = [col for col in indicators_dict['industry_indicators'] if col in df.columns]
        if industry_cols:
            print(f"\n处理行业指数指标: {', '.join(industry_cols)}")
            for col in industry_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    # 对行业指数使用较低级别的降噪，保留趋势
                    denoised_data = wavelet_denoising(data, wavelet='db8', level=2, threshold_mode='soft', threshold_scale=0.5)
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)

    # 美元汇率指标 - 使用特定参数的小波降噪
    if 'forex_indicators' in indicators_dict:
        forex_cols = [col for col in indicators_dict['forex_indicators'] if col in df.columns]
        if forex_cols:
            print(f"\n处理美元汇率指标: {', '.join(forex_cols)}")
            for col in forex_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    # 对汇率数据使用较低级别的降噪，保留趋势
                    denoised_data = wavelet_denoising(data, wavelet='sym8', level=2, threshold_mode='soft', threshold_scale=0.4)
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    # 美股指标 - 使用特定参数的小波降噪
    if 'us_market_indicators' in indicators_dict:
        us_cols = [col for col in indicators_dict['us_market_indicators'] if col in df.columns]
        if us_cols:
            print(f"\n处理美股指标: {', '.join(us_cols)}")
            for col in us_cols:
                if col in df.columns:
                    data = df[col].fillna(0).values
                    # 对美股数据使用适当的降噪参数
                    if 'volume' in col.lower():
                        # 成交量数据使用较高级别的降噪
                        denoised_data = wavelet_denoising(data, wavelet='db8', level=3, threshold_mode='soft', threshold_scale=0.5)
                    else:
                        # 价格数据使用较低级别的降噪
                        denoised_data = wavelet_denoising(data, wavelet='sym8', level=2, threshold_mode='soft', threshold_scale=0.4)
                    df_denoised[col] = ensure_length_match(denoised_data, len(df), col)
    
    return df_denoised

def apply_comprehensive_wavelet_denoising(df):
    """
    对DataFrame中的所有相关数据进行小波降噪处理
    
    参数:
    df: pandas.DataFrame，包含需要降噪的数据
    
    返回:
    pandas.DataFrame: 降噪后的DataFrame
    """
    print("开始全面小波降噪处理...")
    
    # 1. 首先对基本价格和交易量数据进行降噪
    df = apply_wavelet_denoising_to_dataframe(df)
    
    # 2. 然后对技术指标进行降噪
    df = apply_wavelet_denoising_to_indicators(df)
    
    # 3. 检查是否有任何列包含NaN值
    nan_columns = df.columns[df.isna().any()].tolist()
    if nan_columns:
        print(f"警告: 降噪后以下列包含NaN值: {', '.join(nan_columns)}")
        # 填充NaN值
        df = df.fillna(method='ffill').fillna(method='bfill')
    
    print("全面小波降噪处理完成")
    return df

# 增强版可视化降噪效果函数
def plot_denoising_comparison_enhanced(original_df, denoised_df, columns=None, start_date=None, end_date=None, n_samples=200):
    """
    增强版可视化原始数据和降噪后数据的对比
    
    参数:
    original_df: 原始数据DataFrame
    denoised_df: 降噪后的数据DataFrame
    columns: 需要可视化的列名列表，默认为None（使用'Close'）
    start_date: 开始日期，格式为'YYYY-MM-DD'
    end_date: 结束日期，格式为'YYYY-MM-DD'
    n_samples: 如果未指定日期范围，则显示最近的n_samples个样本
    """
    # 设置绘图风格
    try:
        plt.style.use('cyberpunk')
    except:
        plt.style.use('ggplot')  # 如果没有cyberpunk样式，使用ggplot
    
    if columns is None:
        columns = ['Close']
    
    # 为每个列创建一个子图
    n_cols = len(columns)
    fig, axes = plt.subplots(n_cols, 1, figsize=(15, 5*n_cols))
    
    # 如果只有一列，将axes转换为列表以便统一处理
    if n_cols == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        if col in original_df.columns and col in denoised_df.columns:
            # 获取数据
            if start_date and end_date:
                original_data = original_df.loc[start_date:end_date, col]
                denoised_data = denoised_df.loc[start_date:end_date, col]
            else:
                # 使用最近的n_samples个样本
                original_data = original_df[col].iloc[-n_samples:]
                denoised_data = denoised_df[col].iloc[-n_samples:]
            
            # 计算差异
            diff = denoised_data - original_data
            
            # 打印调试信息，检查数据是否相同
            print(f"列 {col} 的原始数据和降噪数据是否完全相同: {np.array_equal(original_data.values, denoised_data.values)}")
            print(f"差异的平均绝对值: {np.mean(np.abs(diff.values))}")
            
            # 绘制原始数据和降噪后的数据，使用不同的线型和颜色以便区分
            axes[i].plot(original_data.index, original_data.values, 'b-', label='raw_data', alpha=0.7, linewidth=2.0)
            axes[i].plot(denoised_data.index, denoised_data.values, 'r--', label='denoise_data', alpha=0.9, linewidth=1.5)
            
            # 添加标题和图例
            axes[i].set_title(f'{col} raw_data vs denoise_data')
            axes[i].legend(loc='upper left')
            axes[i].grid(True, alpha=0.3)
            
            # 在右侧Y轴显示差异
            ax2 = axes[i].twinx()
            ax2.plot(diff.index, diff.values, 'g-', label='差异', alpha=0.5)
            ax2.set_ylabel('差异值', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            ax2.legend(loc='upper right')
            
            # 设置x轴标签
            if i == n_cols - 1:  # 只在最后一个子图上显示x轴标签
                axes[i].set_xlabel('日期')
            
            # 旋转x轴标签以避免重叠
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'enhanced_denoising_visualization_{columns[0]}.png', dpi=300)
    plt.show()
    plt.close()