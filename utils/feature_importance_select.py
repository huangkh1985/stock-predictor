import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error

# 添加注意力权重分析函数
def analyze_attention_weights(model, X_test, feature_names, y_test, top_n=20):
    """改进后的注意力权重分析函数"""
    try:
        # 查找MultiHeadAttention层
        attention_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.MultiHeadAttention):
                attention_layer = layer
                break
        
        if attention_layer is None:
            raise ValueError("未找到MultiHeadAttention层")
        
        # 获取注意力权重
        attention_weights = []
        for i in range(0, len(X_test), 32):  # 批处理大小为32
            batch = X_test[i:i+32]
            # 直接使用attention_layer计算注意力分数
            _, scores = attention_layer(batch, batch, return_attention_scores=True)
            attention_weights.append(scores.numpy())
        
        # 合并所有批次的注意力权重
        attention_weights = np.concatenate(attention_weights, axis=0)
        
        # 计算平均注意力权重
        if len(attention_weights.shape) == 4:  # [batch_size, num_heads, seq_len, seq_len]
            attention_weights = np.mean(attention_weights, axis=(0, 1))  # 平均所有批次和头
        
        # 计算特征重要性
        feature_importance = np.mean(attention_weights, axis=1)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.repeat(feature_importance, len(feature_names) // len(feature_importance))
        })
        
        # 对特征重要性进行排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df.head(top_n), feature_importance
        
    except Exception as e:
        print(f"注意力分析失败: {str(e)}")
        # 使用扰动法作为备选方法
        print("尝试使用替代方法计算特征重要性...")
        return perturbation_feature_importance(
            model, X_test, y_test, feature_names, n_repeats=3
        ), None

# 添加特征工程指导函数
def guide_feature_engineering(feature_importance, dataset, correlation_threshold=0.7, top_n=20):
    """
    基于注意力权重分析结果指导特征工程
    
    参数:
    feature_importance: 特征重要性字典
    dataset: 原始数据集
    correlation_threshold: 相关性阈值，用于检测冗余特征
    top_n: 考虑前N个重要特征
    
    返回:
    recommendations: 特征工程建议
    """
    print("\n基于注意力权重分析进行特征工程指导...")
    
    # 获取前N个重要特征
    top_features = list(feature_importance.keys())[:top_n]
    
    # 计算特征之间的相关性矩阵
    correlation_matrix = dataset[top_features].corr().abs()
    
    # 找出高度相关的特征对
    high_correlation_pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            if correlation_matrix.iloc[i, j] > correlation_threshold:
                high_correlation_pairs.append((
                    top_features[i], 
                    top_features[j], 
                    correlation_matrix.iloc[i, j]
                ))
    
    # 生成建议
    recommendations = {
        "保留特征": [],
        "可能冗余特征": [],
        "特征交互建议": [],
        "特征变换建议": []
    }
    
    # 处理高度相关的特征对
    processed_features = set()
    for feat1, feat2, corr in high_correlation_pairs:
        # 如果两个特征都已处理，跳过
        if feat1 in processed_features and feat2 in processed_features:
            continue
            
        # 根据重要性决定保留哪个特征
        if feature_importance[feat1] > feature_importance[feat2]:
            keep_feature, remove_feature = feat1, feat2
        else:
            keep_feature, remove_feature = feat2, feat1
            
        if keep_feature not in processed_features:
            recommendations["保留特征"].append(keep_feature)
            processed_features.add(keep_feature)
            
        if remove_feature not in processed_features:
            recommendations["可能冗余特征"].append({
                "特征": remove_feature,
                "与特征相关": keep_feature,
                "相关性": corr
            })
            processed_features.add(remove_feature)
            
        # 建议特征交互
        recommendations["特征交互建议"].append({
            "特征对": (feat1, feat2),
            "建议": f"创建 {feat1} 和 {feat2} 的交互特征"
        })
    
    # 为未处理的重要特征提供建议
    for feature in top_features:
        if feature not in processed_features:
            recommendations["保留特征"].append(feature)
            
            # 检查特征分布，建议适当的变换
            if feature in dataset.columns:
                skewness = dataset[feature].skew()
                if abs(skewness) > 1:
                    if skewness > 0:
                        recommendations["特征变换建议"].append({
                            "特征": feature,
                            "建议": "应用对数变换以减少正偏度",
                            "偏度": skewness
                        })
                    else:
                        recommendations["特征变换建议"].append({
                            "特征": feature,
                            "建议": "应用平方变换以减少负偏度",
                            "偏度": skewness
                        })
    
    # 打印建议
    print("\n特征工程建议:")
    print("\n1. 保留以下重要特征:")
    for i, feature in enumerate(recommendations["保留特征"]):
        print(f"   {i+1}. {feature}")
    
    print("\n2. 考虑移除以下可能冗余的特征:")
    for i, item in enumerate(recommendations["可能冗余特征"]):
        print(f"   {i+1}. {item['特征']} (与 {item['与特征相关']} 相关性: {item['相关性']:.4f})")
    
    print("\n3. 特征交互建议:")
    for i, item in enumerate(recommendations["特征交互建议"][:5]):  # 只显示前5个建议
        print(f"   {i+1}. {item['建议']}")
    
    print("\n4. 特征变换建议:")
    for i, item in enumerate(recommendations["特征变换建议"]):
        print(f"   {i+1}. {item['建议']} (偏度: {item['偏度']:.4f})")
    
    return recommendations

# 添加扰动法特征重要性分析函数
def perturbation_feature_importance(model, X_test, y_test, feature_names, n_repeats=10, random_state=42):
    """
    使用扰动法计算特征重要性
    
    参数:
    model: 训练好的模型
    X_test: 测试数据
    y_test: 测试标签
    feature_names: 特征名称列表
    n_repeats: 重复次数
    random_state: 随机种子
    
    返回:
    feature_importance: 特征重要性字典
    """
    print("开始使用扰动法分析特征重要性...")
    
    # 获取基准预测和误差
    baseline_pred = model.predict(X_test)
    baseline_error = np.mean(np.abs(baseline_pred - y_test))
    
    # 初始化特征重要性字典
    importance_dict = {}
    
    # 对每个特征进行扰动
    for feature_idx in range(X_test.shape[2]):
        feature_name = feature_names[feature_idx]
        print(f"分析特征: {feature_name}")
        
        errors = []
        for _ in range(n_repeats):
            # 创建扰动数据的副本
            X_perturbed = X_test.copy()
            
            # 对特定特征进行随机扰动
            for i in range(X_perturbed.shape[0]):
                for j in range(X_perturbed.shape[1]):  # 对每个时间步
                    # 使用正态分布生成扰动
                    noise = np.random.normal(0, 0.1)
                    X_perturbed[i, j, feature_idx] += noise
            
            # 使用扰动数据进行预测
            perturbed_pred = model.predict(X_perturbed)
            perturbed_error = np.mean(np.abs(perturbed_pred - y_test))
            
            # 计算误差增加量
            error_increase = perturbed_error - baseline_error
            errors.append(error_increase)
        
        # 计算平均误差增加量作为特征重要性
        importance = np.mean(errors)
        importance_dict[feature_name] = float(importance)
    
    # 按重要性排序
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 打印前N个重要特征
    print(f"\n扰动法分析的前{min(len(sorted_importance), 10)}个重要特征:")
    for i, (feature, importance) in enumerate(sorted_importance[:10]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    features = [f[0] for f in sorted_importance[:min(len(sorted_importance), 20)]]
    importances = [f[1] for f in sorted_importance[:min(len(sorted_importance), 20)]]
    
    plt.barh(range(len(features)), importances, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('重要性')
    plt.title('扰动法特征重要性')
    plt.tight_layout()
    plt.savefig('./perturbation_feature_importance.png', dpi=300)
    plt.show()
    
    return dict(sorted_importance)

# 添加结合注意力权重和扰动法的特征组合生成函数
def generate_combined_feature_importance(attention_importance, perturbation_importance, weight_attention=0.6, weight_perturbation=0.4):
    """
    结合注意力权重和扰动法的特征重要性
    
    参数:
    attention_importance: 注意力权重特征重要性字典
    perturbation_importance: 扰动法特征重要性字典
    weight_attention: 注意力权重的权重
    weight_perturbation: 扰动法的权重
    
    返回:
    combined_importance: 结合后的特征重要性字典
    """
    print("结合注意力权重和扰动法的特征重要性...")
    
    # 获取所有特征名称
    all_features = set(list(attention_importance.keys()) + list(perturbation_importance.keys()))
    
    # 初始化结合后的特征重要性字典
    combined_importance = {}
    
    # 对每个特征计算加权平均重要性
    for feature in all_features:
        # 获取注意力权重重要性，如果不存在则为0
        att_imp = attention_importance.get(feature, 0)
        
        # 获取扰动法重要性，如果不存在则为0
        pert_imp = perturbation_importance.get(feature, 0)
        
        # 计算加权平均
        combined_imp = weight_attention * att_imp + weight_perturbation * pert_imp
        combined_importance[feature] = combined_imp
    
    # 按重要性排序
    sorted_importance = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)
    
    # 打印前20个重要特征
    print(f"\n结合后的前20个重要特征:")
    for i, (feature, importance) in enumerate(sorted_importance[:20]):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    features = [f[0] for f in sorted_importance[:20]]
    importances = [f[1] for f in sorted_importance[:20]]
    
    plt.barh(range(len(features)), importances, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('重要性')
    plt.title('结合注意力权重和扰动法的特征重要性')
    plt.tight_layout()
    plt.savefig('./combined_feature_importance.png', dpi=300)
    plt.show()
    
    return dict(sorted_importance)

# 添加特征组合生成函数
def generate_feature_combinations(dataset, combined_importance, top_n=20, interaction_methods=['multiply', 'divide', 'add', 'subtract']):
    """
    基于特征重要性生成新的特征组合
    
    参数:
    dataset: 原始数据集
    combined_importance: 结合后的特征重要性字典
    top_n: 考虑前N个重要特征
    interaction_methods: 特征交互方法列表
    
    返回:
    enhanced_dataset: 增强后的数据集
    """
    print("生成新的特征组合...")
    
    # 创建数据集的副本
    enhanced_dataset = dataset.copy()
    
    # 获取前N个重要特征
    top_features = list(combined_importance.keys())[:top_n]
    
    # 生成特征组合
    new_features = []
    
    # 1. 一阶特征变换
    for feature in top_features:
        if feature not in dataset.columns:
            continue
            
        # 对数变换 (对正值)
        if (dataset[feature] > 0).all():
            enhanced_dataset[f'log_{feature}'] = np.log1p(dataset[feature])
            new_features.append(f'log_{feature}')
            print(f"创建特征: log_{feature}")
        
        # 平方变换
        enhanced_dataset[f'square_{feature}'] = dataset[feature] ** 2
        new_features.append(f'square_{feature}')
        print(f"创建特征: square_{feature}")
        
        # 立方变换
        enhanced_dataset[f'cube_{feature}'] = dataset[feature] ** 3
        new_features.append(f'cube_{feature}')
        print(f"创建特征: cube_{feature}")
        
        # 平方根变换 (对正值)
        if (dataset[feature] >= 0).all():
            enhanced_dataset[f'sqrt_{feature}'] = np.sqrt(dataset[feature])
            new_features.append(f'sqrt_{feature}')
            print(f"创建特征: sqrt_{feature}")
    
    # 2. 二阶特征交互
    for i, feat1 in enumerate(top_features[:10]):  # 只使用前10个特征以避免组合爆炸
        if feat1 not in dataset.columns:
            continue
            
        for j, feat2 in enumerate(top_features[:10]):
            if j <= i or feat2 not in dataset.columns:  # 避免重复组合
                continue
            
            # 特征交互
            for method in interaction_methods:
                if method == 'multiply':
                    enhanced_dataset[f'{feat1}_mul_{feat2}'] = dataset[feat1] * dataset[feat2]
                    new_features.append(f'{feat1}_mul_{feat2}')
                    print(f"创建特征: {feat1}_mul_{feat2}")
                
                elif method == 'divide' and not (dataset[feat2] == 0).any():
                    enhanced_dataset[f'{feat1}_div_{feat2}'] = dataset[feat1] / (dataset[feat2] + 1e-8)  # 避免除零
                    new_features.append(f'{feat1}_div_{feat2}')
                    print(f"创建特征: {feat1}_div_{feat2}")
                
                elif method == 'add':
                    enhanced_dataset[f'{feat1}_add_{feat2}'] = dataset[feat1] + dataset[feat2]
                    new_features.append(f'{feat1}_add_{feat2}')
                    print(f"创建特征: {feat1}_add_{feat2}")
                
                elif method == 'subtract':
                    enhanced_dataset[f'{feat1}_sub_{feat2}'] = dataset[feat1] - dataset[feat2]
                    new_features.append(f'{feat1}_sub_{feat2}')
                    print(f"创建特征: {feat1}_sub_{feat2}")
    
    # 3. 时间序列特征 (移动平均、差分等)
    for feature in top_features[:5]:  # 只对前5个最重要的特征创建时间序列特征
        if feature not in dataset.columns:
            continue
            
        # 移动平均
        for window in [3, 5, 10]:
            enhanced_dataset[f'{feature}_ma{window}'] = dataset[feature].rolling(window=window).mean()
            new_features.append(f'{feature}_ma{window}')
            print(f"创建特征: {feature}_ma{window}")
        
        # 移动标准差
        for window in [5, 10]:
            enhanced_dataset[f'{feature}_std{window}'] = dataset[feature].rolling(window=window).std()
            new_features.append(f'{feature}_std{window}')
            print(f"创建特征: {feature}_std{window}")
        
        # 差分
        enhanced_dataset[f'{feature}_diff1'] = dataset[feature].diff()
        new_features.append(f'{feature}_diff1')
        print(f"创建特征: {feature}_diff1")
        
        # 二阶差分
        enhanced_dataset[f'{feature}_diff2'] = dataset[feature].diff().diff()
        new_features.append(f'{feature}_diff2')
        print(f"创建特征: {feature}_diff2")
    
    # 4. 基于领域知识的特征
    # 例如，如果数据集中有价格和成交量，可以创建成交量加权价格
    if 'Close' in dataset.columns and 'Volume' in dataset.columns:
        enhanced_dataset['volume_weighted_price'] = dataset['Close'] * dataset['Volume']
        new_features.append('volume_weighted_price')
        print("创建特征: volume_weighted_price")
    
    # 如果有高低价，可以创建价格波动范围
    if 'High' in dataset.columns and 'Low' in dataset.columns:
        enhanced_dataset['price_range'] = dataset['High'] - dataset['Low']
        new_features.append('price_range')
        print("创建特征: price_range")
        
        # 价格波动率
        enhanced_dataset['price_volatility'] = enhanced_dataset['price_range'] / dataset['Low']
        new_features.append('price_volatility')
        print("创建特征: price_volatility")
    
    # 填充NaN值
    enhanced_dataset.fillna(method='bfill', inplace=True)
    enhanced_dataset.fillna(method='ffill', inplace=True)
    enhanced_dataset.fillna(0, inplace=True)
    
    print(f"共创建了 {len(new_features)} 个新特征")
    
    return enhanced_dataset, new_features

# 添加特征选择函数
def select_best_features(dataset, combined_importance, new_features, target_col='Close', top_n=50):
    """
    从原始特征和新生成的特征中选择最佳特征子集
    
    参数:
    dataset: 增强后的数据集
    combined_importance: 结合后的特征重要性字典
    new_features: 新生成的特征列表
    target_col: 目标列名
    top_n: 选择的特征数量
    
    返回:
    selected_features: 选择的特征列表
    """
    print("选择最佳特征子集...")
    
    # 1. 首先基于重要性选择原始特征
    original_features = [f for f in combined_importance.keys() if f in dataset.columns]
    top_original_features = original_features[:min(len(original_features), top_n//2)]
    
    # 2. 计算新特征与目标的相关性
    correlations = {}
    for feature in new_features:
        if feature in dataset.columns:
            corr = dataset[feature].corr(dataset[target_col])
            correlations[feature] = abs(corr)  # 使用绝对相关性
    
    # 按相关性排序
    sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    # 选择相关性最高的新特征
    top_new_features = [f[0] for f in sorted_correlations[:min(len(sorted_correlations), top_n//2)]]
    
    # 3. 合并选择的特征
    selected_features = top_original_features + top_new_features
    
    # 4. 检查多重共线性
    if len(selected_features) > 1:
        correlation_matrix = dataset[selected_features].corr().abs()
        
        # 找出高度相关的特征对
        high_correlation_pairs = []
        for i in range(len(selected_features)):
            for j in range(i+1, len(selected_features)):
                if correlation_matrix.iloc[i, j] > 0.95:  # 相关性阈值
                    high_correlation_pairs.append((
                        selected_features[i], 
                        selected_features[j], 
                        correlation_matrix.iloc[i, j]
                    ))
        
        # 移除高度相关的特征
        features_to_remove = set()
        for feat1, feat2, corr in high_correlation_pairs:
            # 保留重要性更高的特征
            if combined_importance.get(feat1, 0) > combined_importance.get(feat2, 0):
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        selected_features = [f for f in selected_features if f not in features_to_remove]
    
    print(f"最终选择了 {len(selected_features)} 个特征")
    print("前10个选择的特征:")
    for i, feature in enumerate(selected_features[:10]):
        print(f"{i+1}. {feature}")
    
    return selected_features

# 添加准备增强数据集的函数
def prepare_enhanced_dataset(dataset, selected_features, time_steps, target_col='Close', train_split=0.8, val_split=0.25):
    """
    准备增强特征的数据集
    
    参数:
    dataset: 数据集
    selected_features: 选择的特征列表
    time_steps: 时间步长
    target_col: 目标列名
    train_split: 训练集比例
    val_split: 验证集比例
    
    返回:
    X_train_final, y_train_final, X_val, y_val, X_test, y_test, m_in, m_out
    """
    # 确保目标列在选择的特征中
    if target_col not in selected_features:
        selected_features.append(target_col)
    
    # 选择特征
    data = dataset[selected_features].values
    
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    m_in = MinMaxScaler(feature_range=(0, 1))
    m_out = MinMaxScaler(feature_range=(0, 1))
    
    # 找到目标列的索引
    target_idx = selected_features.index(target_col)
    
    # 分离特征和目标
    X = data[:, [i for i in range(data.shape[1]) if i != target_idx]]
    y = data[:, target_idx].reshape(-1, 1)
    
    # 归一化
    X = m_in.fit_transform(X)
    y = m_out.fit_transform(y)
    
    # 创建时间序列数据
    X_ts, y_ts = [], []
    for i in range(len(X) - time_steps):
        X_ts.append(X[i:i+time_steps])
        y_ts.append(y[i+time_steps])
    
    X_ts, y_ts = np.array(X_ts), np.array(y_ts)
    
    # 分割训练集和测试集
    train_size = int(len(X_ts) * train_split)
    X_train, X_test = X_ts[:train_size], X_ts[train_size:]
    y_train, y_test = y_ts[:train_size], y_ts[train_size:]
    
    # 分割训练集和验证集
    val_size = int(len(X_train) * val_split)
    X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]
    
    return X_train_final, y_train_final, X_val, y_val, X_test, y_test, m_in, m_out

def analyze_feature_importance(model, X_test, feature_names, top_n=20):
    """分析特征重要性"""
    model.eval()
    with torch.no_grad():
        # 获取基准预测
        baseline_pred = model(X_test)
        # 处理多任务输出
        if isinstance(baseline_pred, dict):
            # 使用主任务的预测结果
            baseline_pred = baseline_pred['main']
        baseline_pred = baseline_pred.cpu().numpy()
    
    # 计算基准误差
    baseline_error = mean_squared_error(y_test, baseline_pred)
    
    # 计算每个特征的重要性
    importance_scores = []
    for i, feature_name in enumerate(feature_names):
        # 创建特征扰动
        X_perturbed = X_test.clone()
        X_perturbed[:, :, i] = torch.randn_like(X_perturbed[:, :, i])
        
        # 获取扰动后的预测
        with torch.no_grad():
            perturbed_pred = model(X_perturbed)
            # 处理多任务输出
            if isinstance(perturbed_pred, dict):
                # 使用主任务的预测结果
                perturbed_pred = perturbed_pred['main']
            perturbed_pred = perturbed_pred.cpu().numpy()
        
        # 计算扰动后的误差
        perturbed_error = mean_squared_error(y_test, perturbed_pred)
        
        # 计算特征重要性（误差增加）
        importance = (perturbed_error - baseline_error) / baseline_error
        importance_scores.append((feature_name, importance))
    
    # 按重要性排序
    importance_scores.sort(key=lambda x: x[1], reverse=True)
    
    # 创建结果DataFrame
    importance_df = pd.DataFrame(importance_scores, columns=['Feature', 'Importance'])
    importance_df = importance_df.head(top_n)
    
    return importance_df