import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def create_dataset(X, y, time_steps=1, T=1):
    """
    创建时间序列数据集
    
    参数:
    X: 特征数据
    y: 目标数据
    time_steps: 时间步长
    T: 预测步长
    
    返回:
    X_ts: 时间序列特征数据
    y_ts: 时间序列目标数据
    """
    X_ts, y_ts = [], []
    for i in range(len(X) - time_steps - T + 1):
        X_ts.append(X[i:i+time_steps])
        y_ts.append(y[i+time_steps+T-1])
    return np.array(X_ts), np.array(y_ts)

def prepare_time_series_data(df, target_col='Close', time_steps=None,T =None,train_split=0.8,featurerange=(0, 1)):
    """
    准备时间序列数据
    
    参数:
    df: 股票数据DataFrame
    target_col: 目标列名，默认为'Close'
    time_steps: 时间步数，默认为None（使用配置中的值）
    T:预测的时间长度
    
    返回:
    tuple: (X_train, X_test, y_train, y_test, m_in, m_out)
    """
    print("准备时间序列数据...")
    
    # 保存日期列
    if '日期' in df.columns:
        dates = df['日期'].values
        df_values = df.drop(columns=['日期']).values  #.values 方法将 DataFrame 转换为 NumPy 数组
    else:
        dates = np.arange(len(df))
        df_values = df.values
    
    # 数据归一化
    m_in = MinMaxScaler(feature_range=featurerange)
    m_out = MinMaxScaler(feature_range=featurerange)
    
    # 找到目标列的索引
    if target_col in df.columns:
        target_idx = df.columns.get_loc(target_col)
        if '日期' in df.columns and target_idx > 0:
            target_idx -= 1  # 调整索引，因为我们删除了日期列
    else:
        # 如果找不到目标列，默认使用第一列
        target_idx = 0
        print(f"警告: 找不到目标列 '{target_col}'，使用第一列作为目标")
    
    # 先提取目标列
    output_values = df_values[:, target_idx].reshape(-1, 1)
    
    # 然后划分训练集和测试集
    train_size = int(len(df_values) * train_split)  # 80% 作为训练集
    Xtrain, Xtest = df_values[:train_size], df_values[train_size:]
    ytrain, ytest = output_values[:train_size], output_values[train_size:]
    
    # 归一化输入特征
    Xtrain_scaled = m_in.fit_transform(Xtrain)
    Xtest_scaled = m_in.transform(Xtest)

    # 单独归一化输出特征
    ytrain_scaled = m_out.fit_transform(ytrain)
    ytest_scaled = m_out.transform(ytest)

    # 构建监督学习数据集
    X_train, y_train = [], []
    X_test, y_test = [], []

    # 为训练集创建时间序列窗口
    for i in range(time_steps, len(Xtrain_scaled) - T + 1):
        X_train.append(Xtrain_scaled[i-time_steps:i])
        if T == 1:
            y_train.append(ytrain_scaled[i])
        else:
            y_train.append(ytrain_scaled[i:i+T])
    
    # 为测试集创建时间序列窗口
    for i in range(time_steps, len(Xtest_scaled) - T + 1):
        X_test.append(Xtest_scaled[i-time_steps:i])
        if T == 1:
            y_test.append(ytest_scaled[i])
        else:
            y_test.append(ytest_scaled[i:i+T])
    
    # 转换为numpy数组
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print(f"训练标签形状: {y_train.shape}, 测试标签形状: {y_test.shape}")

    return X_train, X_test, y_train, y_test, m_in, m_out, dates


def split_validation_set(X_train, y_train, val_split=0.25):
    """
    从训练集中分割验证集
    
    参数:
    X_train: 训练特征数据
    y_train: 训练目标数据
    val_split: 验证集比例，默认为0.25
    
    返回:
    tuple: (X_train_final, y_train_final, X_val, y_val)
    """
    val_idx = int(X_train.shape[0] * (1 - val_split))
    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]
    X_train_final = X_train[:val_idx]
    y_train_final = y_train[:val_idx]
    
    return X_train_final, y_train_final, X_val, y_val

def prepare_enhanced_dataset(dataset, selected_features, time_steps, target_col='Close', test_split=0.2, val_split=0.2):
    """
    准备增强特征数据集，包括数据归一化、划分训练/验证/测试集
    
    参数:
    dataset: 包含所有特征的数据集
    selected_features: 选择的特征列表
    time_steps: 时间步长
    target_col: 目标列名称
    test_split: 测试集比例
    val_split: 验证集比例
    
    返回:
    tuple: (X_train_final, y_train_final, X_val, y_val, X_test, y_test, m_in, m_out, target_col_idx)
    """
    print("准备增强特征数据集...")
    
    # 排除日期列，只使用数值列作为特征
    enhanced_values = dataset.values[:, 1:]
    
    # 数据归一化
    m_in = MinMaxScaler(feature_range=(0, 1))
    m_out = MinMaxScaler(feature_range=(0, 1))
    
    # 找到目标列在筛选后数据集中的索引
    target_col_idx = selected_features.index(target_col) if target_col in selected_features else 0
    
    # 归一化输入特征
    scaled_features = m_in.fit_transform(enhanced_values)
    
    # 单独归一化输出特征
    output_values = enhanced_values[:, target_col_idx].reshape(-1, 1)
    scaled_output = m_out.fit_transform(output_values)
    
    # 构建监督学习数据集
    X, y = [], []
    for i in range(time_steps, len(scaled_features)):
        X.append(scaled_features[i-time_steps:i])
        y.append(scaled_output[i])
    
    X, y = np.array(X), np.array(y)
    
    # 划分训练集和测试集
    train_size = int(len(X) * (1 - test_split))
    X_train = X[:train_size]
    X_test = X[train_size:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    
    # 划分验证集
    val_idx = int(X_train.shape[0] * (1 - val_split))
    X_val = X_train[val_idx:]
    y_val = y_train[val_idx:]
    X_train_final = X_train[:val_idx]
    y_train_final = y_train[:val_idx]
    
    print(f"训练集形状: {X_train_final.shape}")
    print(f"验证集形状: {X_val.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    return X_train_final, y_train_final, X_val, y_val, X_test, y_test, m_in, m_out

# 定义数据处理函数
def prepare_data_loader(X, y, batch_size=32, shuffle=True, device=None):
    """
    准备数据加载器
    
    参数:
        X: 特征数据
        y: 目标数据
        batch_size: 批量大小
        shuffle: 是否打乱数据
        device: 使用的设备
        
    返回:
        data_loader: 数据加载器
    """
    # 确保X和y是torch.Tensor类型
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    # 创建数据集
    dataset = TensorDataset(X, y)
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,  # 这有助于更快地将数据转移到CUDA设备
        drop_last=False   # 保留最后一个不完整的批次
    )
    
    return data_loader

def split_train_val_test_xlstm(df, target_col='Close', time_steps=10, T=1,
                             train_split=0.6, val_split=0.2, scaler_type='robust'):
    """
    分割数据集并进行特征缩放
    
    参数:
        df: 输入数据框
        target_col: 目标列名
        time_steps: 时间步长
        T: 预测步长
        train_split: 训练集比例
        val_split: 验证集比例
        scaler_type: 缩放器类型 ('standard', 'minmax', 'robust')
    
    返回:
        处理后的训练集、验证集和测试集
    """
    # 1. 数据预处理
    df = df.copy()
    
    # 添加时间特征
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
    
    # 添加技术指标
    df['returns'] = df[target_col].pct_change()
    df['volatility'] = df['returns'].rolling(window=20).std()
    df['ma5'] = df[target_col].rolling(window=5).mean()
    df['ma20'] = df[target_col].rolling(window=20).mean()
    df['rsi'] = calculate_rsi(df[target_col])
    df['macd'], df['macd_signal'] = calculate_macd(df[target_col])
    df['upper_band'], df['lower_band'] = calculate_bollinger_bands(df[target_col])
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 2. 首先划分数据集
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))
    
    # 划分数据集
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    # 3. 对每个数据集分别进行特征工程
    def prepare_sequences(data, time_steps, T):
        X, y = [], []
        for i in range(len(data) - time_steps - T + 1):
            X.append(data.iloc[i:(i + time_steps)].values)
            y.append(data.iloc[i + time_steps + T - 1][target_col])
        return np.array(X), np.array(y)
    
    # 准备特征和目标
    features = df.copy()
    target = df[[target_col]]
    
    # 4. 对训练集进行缩放，并保存缩放器
    if scaler_type == 'standard':
        scaler_feature = StandardScaler()
        scaler_label = StandardScaler()
    elif scaler_type == 'minmax':
        scaler_feature = MinMaxScaler(feature_range=(0, 1))
        scaler_label = MinMaxScaler(feature_range=(0, 1))
    else:  # robust
        scaler_feature = RobustScaler()
        scaler_label = RobustScaler()
    
    # 只使用训练集数据拟合缩放器
    scaler_feature.fit(features.iloc[:train_end])
    scaler_label.fit(target.iloc[:train_end])
    
    # 5. 使用训练集的缩放器转换所有数据集
    train_features = scaler_feature.transform(features.iloc[:train_end])
    train_target = scaler_label.transform(target.iloc[:train_end])
    
    val_features = scaler_feature.transform(features.iloc[train_end:val_end])
    val_target = scaler_label.transform(target.iloc[train_end:val_end])
    
    test_features = scaler_feature.transform(features.iloc[val_end:])
    test_target = scaler_label.transform(target.iloc[val_end:])
    
    # 6. 准备序列数据
    # 创建包含特征和目标的DataFrame
    train_df_processed = pd.DataFrame(
        train_features, 
        columns=features.columns
    )
    train_df_processed[target_col] = train_target
    
    val_df_processed = pd.DataFrame(
        val_features, 
        columns=features.columns
    )
    val_df_processed[target_col] = val_target
    
    test_df_processed = pd.DataFrame(
        test_features, 
        columns=features.columns
    )
    test_df_processed[target_col] = test_target
    
    # 准备序列数据
    train_X, train_y = prepare_sequences(train_df_processed, time_steps, T)
    val_X, val_y = prepare_sequences(val_df_processed, time_steps, T)
    test_X, test_y = prepare_sequences(test_df_processed, time_steps, T)
    
    # 7. 准备日期数据
    train_dates = df.index[time_steps:train_end]
    val_dates = df.index[train_end+time_steps:val_end]
    test_dates = df.index[val_end+time_steps:]
    
    return (train_X, train_y, train_dates, 
            val_X, val_y, val_dates, 
            test_X, test_y, test_dates, 
            scaler_feature, scaler_label)

def calculate_rsi(prices, period=14):
    """计算相对强弱指标(RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """计算布林带"""
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    return upper_band, lower_band

def split_train_test_xlstm(
    dataset: pd.DataFrame,
    target_col: str = 'Close',
    time_steps: int = 10,
    T: int = 1,
    train_split: float = 0.8,
    featurerange: tuple = (0, 1)
) -> tuple:
    """
    准备时间序列数据，包括归一化和划分训练集、验证集、测试集
    
    参数:
        dataset: 包含时间序列数据的DataFrame
        target_col: 目标列名
        time_steps: 用于预测的历史时间步长
        T: 预测未来的时间步长
        train_split: 训练集占总数据的比例
        featurerange: 归一化的范围
        
    返回:
        train_X: 训练集特征
        train_y: 训练集标签
        train_dates: 训练集对应的日期
        test_X: 测试集特征
        test_y: 测试集标签
        test_dates: 测试集对应的日期
        scaler_feature: 输入数据的归一化器
        scaler_label: 输出数据的归一化器
    """


    # 如果输入是numpy数组，则处理为DataFrame
    if isinstance(dataset, np.ndarray):
        df = pd.DataFrame(dataset)
    else:
        # 处理DataFrame输入
        if '日期' in dataset.columns:
            dataset['日期'] = pd.to_datetime(dataset['日期'])
            df = dataset.copy()
        else:
            # 如果找不到日期列，创建一个假的日期索引
            df = pd.DataFrame(dataset)
            df['日期'] = pd.date_range(start='2000-01-01', periods=len(df))
    
    # 检查日期列
    dates = None
    if df.index.dtype.kind in 'Mm':  # 如果索引是日期类型
        dates = df.index
        df_values = df.values
    elif '日期' in df.columns:
        dates = df['日期'].values
        df_values = df.drop(columns=['日期']).values
    else:
        dates = np.arange(len(df))
        df_values = df.values
    
    # 查找目标列索引
    if target_col in df.columns:
        target_idx = df.columns.get_loc(target_col)
    else:
        # 如果找不到目标列，使用第一列
        print(f"警告: 找不到目标列 '{target_col}'，使用第一列作为目标")
        target_idx = 0
        
    # 根据scaler_type选择缩放器
    if scaler_type.lower() == 'minmax':
        scaler_feature = MinMaxScaler(feature_range=featurerange)
        scaler_label = MinMaxScaler(feature_range=featurerange)
    elif scaler_type.lower() == 'standard':
        scaler_feature = StandardScaler()
        scaler_label = StandardScaler()
    elif scaler_type.lower() == 'robust':
        scaler_feature = RobustScaler()
        scaler_label = RobustScaler()
    else:
        print(f"警告: 未知的缩放器类型 '{scaler_type}'，使用 MinMaxScaler 替代")
        scaler_feature = MinMaxScaler(feature_range=featurerange)
        scaler_label = MinMaxScaler(feature_range=featurerange)
        
    # 提取目标列
    if target_idx < df_values.shape[1]:
        output_values = df_values[:, target_idx].reshape(-1, 1)
    else:
        # 如果索引超出范围，使用第一列
        print(f"警告: 目标索引 {target_idx} 超出范围，使用第一列")
        output_values = df_values[:, 0].reshape(-1, 1)
    
    # 缩放特征数据
    scaled_features = scaler_feature.fit_transform(df_values)
    
    # 缩放目标数据
    scaled_target = scaler_label.fit_transform(output_values)
    
    # 创建时间序列数据
    train_X, train_y = create_dataset(scaled_features[:int(len(scaled_features) * train_split)], scaled_target[:int(len(scaled_target) * train_split)])
    test_X, test_y = create_dataset(scaled_features[int(len(scaled_features) * train_split):], scaled_target[int(len(scaled_target) * train_split):])
    
    # 获取对应的日期
    train_dates = dates[:len(train_X)] if dates is not None else np.arange(len(train_X))
    test_dates = dates[len(train_X):] if dates is not None else np.arange(len(test_X)) + len(train_X)
    
    print(f"训练集形状: {train_X.shape}, 测试集形状: {test_X.shape}")
    print(f"训练标签形状: {train_y.shape}, 测试标签形状: {test_y.shape}")
    
    return train_X, train_y, train_dates, test_X, test_y, test_dates, scaler_feature, scaler_label

def split_train_val_test_xlstm_darts(
    dataset: pd.DataFrame,
    target_col: str = 'Close',
    time_steps: int = 10,
    T: int = 1,
    train_split: float = 0.7,
    val_split: float = 0.1,
    
) -> tuple:
    """
    准备时间序列数据，包括归一化和划分训练集、验证集、测试集
    
    参数:
        dataset: 包含时间序列数据的DataFrame
        target_col: 目标列名
        time_steps: 用于预测的历史时间步长
        T: 预测未来的时间步长
        train_split: 训练集占总数据的比例
        val_split: 验证集占总数据的比例
        featurerange: 归一化的范围
        
    返回:
        train_X: 训练集特征
        train_y: 训练集标签
        train_dates: 训练集对应的日期
        val_X: 验证集特征
        val_y: 验证集标签
        val_dates: 验证集对应的日期
        test_X: 测试集特征
        test_y: 测试集标签
        test_dates: 测试集对应的日期
        m_in: 输入数据的归一化器
        m_out: 输出数据的归一化器
    """
    # 确保数据集有日期索引
    if not isinstance(dataset.index, pd.DatetimeIndex):
        if '日期' in dataset.columns:
            dataset['日期'] = pd.to_datetime(dataset['日期'])
            dates = dataset['日期'].copy()
            # 移除日期列，只保留数值列作为特征
            feature_data = dataset.drop(columns=['日期'])
        else:
            # 尝试将第一列转换为日期
            try:
                dates = pd.to_datetime(dataset.iloc[:, 0])
                # 移除第一列（日期列），只保留数值列作为特征
                feature_data = dataset.iloc[:, 1:]
            except:
                # 如果无法转换，创建一个假的日期索引
                dates = pd.date_range(start='2000-01-01', periods=len(dataset))
                feature_data = dataset.copy()
    else:
        dates = dataset.index.copy()
        feature_data = dataset.copy()
    
    # 提取目标列数据用于预测
    target_data = feature_data[target_col].values.reshape(-1, 1)
    
    if scaler:
        return scaler.transform(data)
    else:
        scaler = Scaler() # default uses sklearn's MinMaxScaler
        data = scaler.fit_transform(data)
        return data, scaler
   