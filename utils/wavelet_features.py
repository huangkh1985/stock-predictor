import numpy as np
import pywt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_wavelet_decomposition(signal, wavelet='db4', level=3):
    """
    绘制小波分解结果
    
    参数:
    signal: 输入信号
    wavelet: 小波类型
    level: 分解级别
    """
    # 执行小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 绘制原始信号和各级分解结果
    fig, axes = plt.subplots(level + 2, 1, figsize=(12, 9))
    
    # 绘制原始信号
    axes[0].plot(signal)
    axes[0].set_title('原始信号')
    axes[0].set_xticklabels([])
    
    # 绘制近似系数
    axes[1].plot(coeffs[0])
    axes[1].set_title(f'近似系数 (A{level})')
    axes[1].set_xticklabels([])
    
    # 绘制细节系数
    for i in range(level):
        axes[i+2].plot(coeffs[i+1])
        axes[i+2].set_title(f'细节系数 (D{level-i})')
        if i < level - 1:
            axes[i+2].set_xticklabels([])
    
    plt.tight_layout()
    plt.show()

def extract_adaptive_wavelet_features(signal, wavelet_family='db', max_level=5, feature_type='all'):
    """
    自适应小波特征提取 - 根据信号特性选择最佳小波基和分解级别
    
    参数:
    signal: 输入信号
    wavelet_family: 小波族，如'db', 'sym', 'coif'等
    max_level: 最大分解级别
    feature_type: 特征类型 ('stats', 'energy', 'all')
    
    返回:
    特征向量
    """
    # 尝试不同的小波基函数
    wavelet_bases = [f'{wavelet_family}{i}' for i in range(1, 11) if pywt.wavelist(wavelet_family).__contains__(f'{wavelet_family}{i}')]
    
    best_features = None
    best_entropy = float('inf')
    
    # 对每个小波基函数计算信息熵，选择熵最小的（最有信息量的）
    for wavelet in wavelet_bases:
        # 确定最大可能的分解级别
        max_possible_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet).dec_len)
        level = min(max_level, max_possible_level)
        
        # 执行小波分解
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        
        # 计算各系数的信息熵
        entropy = sum([np.sum(-np.abs(coef)**2 * np.log(np.abs(coef)**2 + 1e-10)) for coef in coeffs])
        
        if entropy < best_entropy:
            best_entropy = entropy
            # 提取特征
            features = []
            
            # 对每个系数提取特征
            for i, coeff in enumerate(coeffs):
                if feature_type in ['stats', 'all']:
                    # 统计特征
                    features.extend([
                        np.mean(coeff),      # 均值
                        np.std(coeff),       # 标准差
                        np.max(coeff),       # 最大值
                        np.min(coeff),       # 最小值
                        np.median(coeff),    # 中位数
                        np.percentile(coeff, 25),  # 25分位数
                        np.percentile(coeff, 75),  # 75分位数
                        np.sum(np.abs(coeff) > np.std(coeff))  # 超过标准差的点数
                    ])
                
                if feature_type in ['energy', 'all']:
                    # 能量特征
                    features.append(np.sum(coeff**2))  # 能量
                    
                    # 相对能量
                    total_energy = sum(np.sum(c**2) for c in coeffs)
                    if total_energy > 0:
                        features.append(np.sum(coeff**2) / total_energy)
                    else:
                        features.append(0)
            
            best_features = np.array(features)
    
    return best_features

def prepare_wavelet_features(data, window_size=10, target_col='Close', wavelet='db4', level=3):
    """
    为时间序列数据准备小波特征
    
    参数:
    data: 输入数据DataFrame
    window_size: 滑动窗口大小
    target_col: 目标列名
    wavelet: 小波类型
    level: 分解级别
    
    返回:
    X: 特征矩阵
    y: 目标值
    """
    # 获取目标列数据
    signal = data[target_col].values
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
    
    X = []
    y = []
    
    # 使用滑动窗口提取特征
    for i in range(len(signal_scaled) - window_size):
        # 当前窗口
        window = signal_scaled[i:i+window_size]
        
        # 提取小波特征
        wavelet_features = extract_wavelet_features(window, wavelet, level)
        
        # 添加原始数据作为额外特征
        combined_features = np.concatenate([window, wavelet_features])
        
        X.append(combined_features)
        y.append(signal_scaled[i+window_size])
    
    return np.array(X), np.array(y), scaler


def prepare_multiscale_wavelet_features(df, window_size=10, target_col='Close', wavelet='sym8', levels=[2, 4, 6]):
    """
    提取多尺度小波特征
    
    Args:
        df: 输入数据框
        window_size: 滑动窗口大小
        target_col: 目标列名
        wavelet: 小波基函数
        levels: 多个分解层级
        
    Returns:
        X: 特征矩阵
        y: 目标值
        scaler: 归一化器
    """
    # 获取目标列数据
    data = df[target_col].values
    
    # 创建多尺度特征列表
    multiscale_features = []
    
    # 对每个小波族和层级进行小波分解
    for family in wavelet_families:
        # 选择该族中的一个小波基函数
        wavelet = f"{family}4" if pywt.wavelist(family).__contains__(f"{family}4") else pywt.wavelist(family)[0]
        
        for level in levels:
            # 小波分解
            try:
                coeffs = pywt.wavedec(data, wavelet, level=level)
                
                # 提取每个层级的系数
                for i, coef in enumerate(coeffs):
                    # 对系数进行归一化
                    scaler = MinMaxScaler()
                    coef_scaled = scaler.fit_transform(coef.reshape(-1, 1)).flatten()
                    
                    # 添加到特征列表
                    multiscale_features.append(coef_scaled)
                    
                    # 添加额外的统计特征
                    if i == 0:  # 只对近似系数计算额外特征
                        # 计算滚动统计量
                        window = min(20, len(coef) // 2)  # 动态调整窗口大小
                        if window > 1:
                            # 滚动均值
                            rolling_mean = np.convolve(coef, np.ones(window)/window, mode='valid')
                            multiscale_features.append(scaler.fit_transform(rolling_mean.reshape(-1, 1)).flatten())
                            
                            # 滚动标准差
                            rolling_std = np.array([np.std(coef[max(0, i-window):i+1]) for i in range(len(coef))])
                            multiscale_features.append(scaler.fit_transform(rolling_std.reshape(-1, 1)).flatten())
            except Exception as e:
                print(f"处理小波 {wavelet} 在级别 {level} 时出错: {e}")
                continue
    
    # 确保所有特征长度一致
    min_length = min(len(f) for f in multiscale_features)
    aligned_features = [f[:min_length] for f in multiscale_features]
    
    # 合并所有特征
    all_features = np.column_stack(aligned_features)
    
    # 创建滑动窗口特征
    X, y = [], []
    for i in range(len(all_features) - window_size):
        X.append(all_features[i:i+window_size])
        y.append(data[i+window_size])
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    # 归一化目标值
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y, y_scaler


# 添加增强型多尺度小波特征提取函数
def prepare_enhanced_multiscale_features(df, window_size=10, target_col='Close', 
                                        wavelet_families=['db', 'sym', 'coif'], 
                                        levels=[2, 4]):
    """
    增强型多尺度小波特征提取
    
    Args:
        df: 输入数据框
        window_size: 滑动窗口大小
        target_col: 目标列名
        wavelet_families: 小波族列表
        levels: 多个分解层级
        
    Returns:
        X: 特征矩阵
        y: 目标值
        scaler: 归一化器
    """
    import pywt
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    
    # 获取目标列数据
    data = df[target_col].values
    
    # 创建多尺度特征列表
    multiscale_features = []
    
    # 对每个小波族和层级进行小波分解
    for family in wavelet_families:
        # 选择该族中的一个小波基函数
        wavelet = f"{family}4" if f"{family}4" in pywt.wavelist() else pywt.wavelist(family)[0]
        
        for level in levels:
            # 小波分解
            try:
                coeffs = pywt.wavedec(data, wavelet, level=level)
                
                # 提取每个层级的系数
                for i, coef in enumerate(coeffs):
                    # 对系数进行归一化
                    scaler = MinMaxScaler()
                    coef_scaled = scaler.fit_transform(coef.reshape(-1, 1)).flatten()
                    
                    # 添加到特征列表
                    multiscale_features.append(coef_scaled)
                    
                    # 添加额外的统计特征
                    if i == 0:  # 只对近似系数计算额外特征
                        # 计算滚动统计量
                        window = min(20, len(coef) // 2)  # 动态调整窗口大小
                        if window > 1:
                            # 滚动均值
                            rolling_mean = np.convolve(coef, np.ones(window)/window, mode='valid')
                            multiscale_features.append(scaler.fit_transform(rolling_mean.reshape(-1, 1)).flatten())
                            
                            # 滚动标准差
                            rolling_std = np.array([np.std(coef[max(0, i-window):i+1]) for i in range(len(coef))])
                            multiscale_features.append(scaler.fit_transform(rolling_std.reshape(-1, 1)).flatten())
            except Exception as e:
                print(f"处理小波 {wavelet} 在级别 {level} 时出错: {e}")
                continue
    
    # 确保所有特征长度一致
    min_length = min(len(f) for f in multiscale_features)
    aligned_features = [f[:min_length] for f in multiscale_features]
    
    # 合并所有特征
    all_features = np.column_stack(aligned_features)
    
    # 创建滑动窗口特征
    X, y = [], []
    for i in range(len(all_features) - window_size):
        X.append(all_features[i:i+window_size])
        y.append(data[i+window_size])
    
    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)
    
    # 归一化目标值
    y_scaler = MinMaxScaler()
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y, y_scaler

# 添加特征选择函数
def select_wavelet_features(X, y, method='mutual_info', n_features=None):
    """
    小波特征选择
    
    参数:
    X: 特征矩阵
    y: 目标值
    method: 特征选择方法 ('mutual_info', 'f_regression', 'recursive')
    n_features: 选择的特征数量，默认为None（自动确定）
    
    返回:
    选择后的特征矩阵
    """
    from sklearn.feature_selection import mutual_info_regression, f_regression, RFE
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # 保存原始形状
    original_shape = X.shape
    
    # 如果是3D数据(样本,时间步,特征)，将其重塑为2D(样本,时间步*特征)
    if len(X.shape) == 3:
        X_reshaped = X.reshape(X.shape[0], -1)
    else:
        X_reshaped = X
    
    # 如果未指定特征数量，设为原始特征数的一半
    if n_features is None:
        if len(original_shape) == 3:
            n_features = original_shape[1] * original_shape[2] // 2
        else:
            n_features = original_shape[1] // 2
    
    # 根据方法选择特征
    if method == 'mutual_info':
        # 使用互信息
        mi = mutual_info_regression(X_reshaped, y)
        selected_indices = np.argsort(mi)[-n_features:]
    elif method == 'f_regression':
        # 使用F检验
        f_values, _ = f_regression(X_reshaped, y)
        selected_indices = np.argsort(f_values)[-n_features:]
    elif method == 'recursive':
        # 使用递归特征消除
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
        selector.fit(X_reshaped, y)
        selected_indices = np.where(selector.support_)[0]
    
    # 应用特征选择
    if len(original_shape) == 3:
        # 对于3D数据，需要将选择的索引映射回原始形状
        time_steps = original_shape[1]
        features = original_shape[2]
        
        # 计算每个选定索引对应的时间步和特征
        selected_time_steps = selected_indices // features
        selected_features = selected_indices % features
        
        # 创建掩码
        mask = np.zeros((time_steps, features), dtype=bool)
        for t, f in zip(selected_time_steps, selected_features):
            mask[t, f] = True
        
        # 应用掩码
        X_selected = np.zeros((original_shape[0], np.sum(mask)))
        for i, sample in enumerate(X):
            X_selected[i] = sample[mask]
    else:
        # 对于2D数据，直接选择列
        X_selected = X_reshaped[:, selected_indices]
    
    return X_selected

# 添加缺失的extract_wavelet_features函数
def extract_wavelet_features(signal, wavelet='db4', level=2):
    """
    从信号中提取小波特征
    
    参数:
    signal: 输入信号
    wavelet: 小波类型，默认为'db4'
    level: 分解级别，默认为2
    
    返回:
    特征向量
    """
    import pywt
    import numpy as np
    
    # 执行小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 提取特征
    features = []
    
    # 对每个系数提取统计特征
    for i, coeff in enumerate(coeffs):
        # 统计特征
        features.extend([
            np.mean(coeff),      # 均值
            np.std(coeff),       # 标准差
            np.max(coeff),       # 最大值
            np.min(coeff),       # 最小值
            np.median(coeff),    # 中位数
            np.percentile(coeff, 25),  # 25分位数
            np.percentile(coeff, 75),  # 75分位数
            np.sum(np.abs(coeff) > np.std(coeff))  # 超过标准差的点数
        ])
        
        # 能量特征
        features.append(np.sum(coeff**2))  # 能量
        
        # 相对能量
        total_energy = sum(np.sum(c**2) for c in coeffs)
        if total_energy > 0:
            features.append(np.sum(coeff**2) / total_energy)
        else:
            features.append(0)
    
    return np.array(features)