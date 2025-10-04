"""
股票实盘预测模块
使用已训练的模型对输入的股票进行预测
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime, timedelta
import sys

sys.path.append('.')

# 抑制警告
warnings.filterwarnings('ignore')

# 导入数据下载模块
from stock_data_downloader import download_china_stock_enhanced_data

def load_trained_model(model_path='models/trained_model.pkl'):
    """
    加载已训练的模型
    
    参数:
    model_path: 模型文件路径
    
    返回:
    model: 训练好的模型
    """
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在 - {model_path}")
        print("请先运行 stock_statistical_analysis.py 训练并保存模型")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ 成功加载模型: {model_path}")
    return model

def load_all_models(models_path='models/all_trained_models.pkl'):
    """
    加载所有训练的模型（用于对比预测）
    
    参数:
    models_path: 所有模型文件路径
    
    返回:
    all_models_data: 所有模型的字典
    """
    if not os.path.exists(models_path):
        print(f"⚠️  未找到多模型文件: {models_path}")
        print("   将只使用单一最佳模型进行预测")
        return None
    
    with open(models_path, 'rb') as f:
        all_models_data = pickle.load(f)
    
    print(f"✅ 成功加载 {len(all_models_data)} 个模型: {list(all_models_data.keys())}")
    return all_models_data

def load_model_info(info_path='models/model_info.pkl'):
    """
    加载模型信息
    
    参数:
    info_path: 模型信息文件路径
    
    返回:
    model_info: 模型信息字典
    """
    if not os.path.exists(info_path):
        return None
    
    with open(info_path, 'rb') as f:
        model_info = pickle.load(f)
    
    return model_info

def load_feature_list(feature_path='models/feature_list.pkl'):
    """
    加载特征列表
    
    参数:
    feature_path: 特征列表文件路径
    
    返回:
    feature_list: 特征名称列表
    """
    if not os.path.exists(feature_path):
        print(f"❌ 错误：特征列表文件不存在 - {feature_path}")
        return None
    
    with open(feature_path, 'rb') as f:
        feature_list = pickle.load(f)
    
    # 清理特征名称（确保与训练时一致）
    cleaned_feature_list = []
    for col in feature_list:
        clean_col = str(col)
        clean_col = clean_col.replace('[', '_').replace(']', '_')
        clean_col = clean_col.replace('{', '_').replace('}', '_')
        clean_col = clean_col.replace('"', '').replace("'", '')
        clean_col = clean_col.replace(':', '_').replace(',', '_')
        clean_col = clean_col.replace(' ', '_').replace('<', 'lt')
        clean_col = clean_col.replace('>', 'gt').replace('=', 'eq')
        clean_col = clean_col.replace('(', '_').replace(')', '_')
        # 关键：把所有双下划线替换成单下划线
        while '__' in clean_col:
            clean_col = clean_col.replace('__', '_')
        clean_col = clean_col.strip('_')
        cleaned_feature_list.append(clean_col)
    
    print(f"✅ 成功加载特征列表: {len(cleaned_feature_list)} 个特征")
    return cleaned_feature_list

def download_stock_for_prediction(stock_code, days=365):
    """
    下载股票最近的数据用于预测
    
    参数:
    stock_code: 股票代码
    days: 获取最近多少天的数据（默认365天，约1年）
    
    返回:
    stock_data: 股票数据DataFrame
    """
    print(f"\n📥 下载股票 {stock_code} 的最新数据（最近{days}天）...")
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    # 下载数据
    all_data = download_china_stock_enhanced_data(
        [stock_code],
        start_date=start_date_str,
        end_date=end_date_str,
        save_to_file=False
    )
    
    if stock_code not in all_data:
        print(f"❌ 无法下载股票 {stock_code} 的数据")
        return None
    
    stock_data = all_data[stock_code]
    print(f"✅ 成功下载 {len(stock_data)} 条数据记录")
    
    return stock_data

def extract_features_for_prediction(stock_data, window_size=20):
    """
    为预测提取TSFresh特征
    
    参数:
    stock_data: 股票数据DataFrame
    window_size: 滑动窗口大小
    
    返回:
    features_df: 提取的特征DataFrame
    """
    print(f"\n🔧 提取时间序列特征（窗口大小: {window_size}天）...")
    
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    
    # 检查数据长度
    if len(stock_data) < window_size:
        print(f"❌ 数据不足：需要至少 {window_size} 天的数据，当前只有 {len(stock_data)} 天")
        return None
    
    # 选择用于TSFresh分析的特征
    tsfresh_features = [
        'Close', 'Open', 'High', 'Low', 'Volume', 'TurnoverRate', 
        'PriceChangeRate', 'MainNetInflow', 'MainNetInflowRatio'
    ]
    
    # 确保所有特征存在
    for feature in tsfresh_features:
        if feature not in stock_data.columns:
            stock_data[feature] = 0
    
    # 使用最近的窗口数据
    window_data = stock_data.iloc[-window_size:]
    
    # 创建TSFresh格式的数据
    tsfresh_data_list = []
    window_id = f"prediction_window"
    
    for feature in tsfresh_features:
        feature_values = window_data[feature].values
        
        for time_idx, value in enumerate(feature_values):
            tsfresh_data_list.append({
                'id': window_id,
                'time': time_idx,
                'feature_name': feature,
                'value': float(value) if not pd.isna(value) else 0.0
            })
    
    x_df = pd.DataFrame(tsfresh_data_list)
    
    # 提取特征
    print("  提取TSFresh特征...")
    feature_extraction_settings = MinimalFCParameters()
    
    all_extracted_features = []
    
    for feature_type in tsfresh_features:
        feature_df = x_df[x_df['feature_name'] == feature_type].copy()
        feature_df = feature_df.drop('feature_name', axis=1)
        
        try:
            extracted = extract_features(
                feature_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=feature_extraction_settings,
                n_jobs=1,
                disable_progressbar=True
            )
            
            # 重命名列
            extracted.columns = [f"{feature_type}_{col}" for col in extracted.columns]
            all_extracted_features.append(extracted)
            
        except Exception as e:
            print(f"  ⚠️ {feature_type} 特征提取失败: {e}")
            continue
    
    if not all_extracted_features:
        print("❌ 特征提取失败")
        return None
    
    # 合并所有特征
    x_extracted = pd.concat(all_extracted_features, axis=1)
    
    # 处理无限值和缺失值
    x_extracted = impute(x_extracted)
    
    # 立即清理特征名称（与训练时保持一致）
    x_extracted = clean_feature_names(x_extracted)
    
    print(f"✅ 成功提取 {x_extracted.shape[1]} 个特征")
    
    return x_extracted

def clean_feature_names(df):
    """清理特征名称"""
    cleaned_columns = []
    for col in df.columns:
        clean_col = str(col)
        clean_col = clean_col.replace('[', '_').replace(']', '_')
        clean_col = clean_col.replace('{', '_').replace('}', '_')
        clean_col = clean_col.replace('"', '').replace("'", '')
        clean_col = clean_col.replace(':', '_').replace(',', '_')
        clean_col = clean_col.replace(' ', '_').replace('<', 'lt')
        clean_col = clean_col.replace('>', 'gt').replace('=', 'eq')
        clean_col = clean_col.replace('(', '_').replace(')', '_')
        while '__' in clean_col:
            clean_col = clean_col.replace('__', '_')
        clean_col = clean_col.strip('_')
        cleaned_columns.append(clean_col)
    
    df_cleaned = df.copy()
    df_cleaned.columns = cleaned_columns
    return df_cleaned

def align_features(features_df, feature_list):
    """
    对齐特征（确保与训练时特征一致）
    
    参数:
    features_df: 提取的特征DataFrame（已清理特征名称）
    feature_list: 训练时的特征列表（已清理）
    
    返回:
    aligned_df: 对齐后的特征DataFrame
    """
    print(f"\n🔄 对齐特征...")
    
    # 创建一个空DataFrame，包含所有训练时的特征
    aligned_df = pd.DataFrame(0, index=features_df.index, columns=feature_list)
    
    # 填充存在的特征
    matched_count = 0
    matched_features = []
    for col in features_df.columns:
        if col in feature_list:
            aligned_df[col] = features_df[col]
            matched_count += 1
            matched_features.append(col)
    
    missing_in_prediction = [col for col in feature_list if col not in features_df.columns]
    missing_in_training = [col for col in features_df.columns if col not in feature_list]
    
    print(f"  训练特征数: {len(feature_list)}")
    print(f"  提取特征数: {len(features_df.columns)}")
    print(f"  匹配特征数: {matched_count}")
    print(f"  缺失特征数: {len(missing_in_prediction)} (将用0填充)")
    
    if matched_count == 0:
        print(f"\n⚠️  警告：没有匹配的特征！")
        print(f"  训练特征示例: {feature_list[:5]}")
        print(f"  提取特征示例: {list(features_df.columns[:5])}")
    elif matched_count < len(feature_list) * 0.5:
        print(f"\n⚠️  警告：匹配的特征较少（{matched_count}/{len(feature_list)}）")
        print(f"  匹配的特征示例: {matched_features[:3]}")
        if missing_in_prediction:
            print(f"  预测时缺失的特征示例: {missing_in_prediction[:3]}")
        if missing_in_training:
            print(f"  训练时没有的特征示例: {missing_in_training[:3]}")
    else:
        print(f"  ✅ 特征匹配良好")
    
    print(f"✅ 特征对齐完成")
    
    return aligned_df

def predict_stock(model, features_df):
    """
    使用模型进行预测
    
    参数:
    model: 训练好的模型
    features_df: 对齐后的特征DataFrame
    
    返回:
    prediction: 预测类别 (0=强势, 1=弱势)
    probability: 预测概率
    """
    print(f"\n🤖 使用模型进行预测...")
    
    # 预测
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0]
    
    return prediction, probability

def predict_with_all_models(all_models_data, features_df):
    """
    使用所有模型进行预测并对比
    
    参数:
    all_models_data: 所有模型的字典
    features_df: 对齐后的特征DataFrame
    
    返回:
    predictions_dict: 每个模型的预测结果
    """
    print(f"\n🤖 使用 {len(all_models_data)} 个模型进行对比预测...")
    
    predictions_dict = {}
    
    for model_name, model_data in all_models_data.items():
        model = model_data['model']
        optimal_threshold = model_data.get('optimal_threshold', 0.5)
        
        # 预测概率
        probability = model.predict_proba(features_df)[0]
        
        # 使用训练时的最优阈值
        prediction = 1 if probability[1] >= optimal_threshold else 0
        
        predictions_dict[model_name] = {
            'prediction': prediction,
            'probability': probability,
            'prob_strong': probability[0],
            'prob_weak': probability[1],
            'confidence': max(probability),
            'optimal_threshold': optimal_threshold,
            'train_accuracy': model_data.get('accuracy', 0),
            'train_precision': model_data.get('avg_precision', 0)
        }
    
    return predictions_dict

def print_prediction_result(stock_code, stock_data, prediction, probability, model_info=None):
    """
    打印预测结果
    
    参数:
    stock_code: 股票代码
    stock_data: 股票数据
    prediction: 预测类别
    probability: 预测概率
    model_info: 模型信息
    """
    print("\n" + "="*80)
    print(f"📊 股票预测结果 - {stock_code}")
    print("="*80)
    
    # 最新行情
    latest = stock_data.iloc[-1]
    latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
    
    print(f"\n📅 最新数据日期: {latest_date}")
    print(f"💰 最新收盘价: {latest['Close']:.2f} 元")
    
    if 'MA_20' in stock_data.columns:
        ma20 = latest['MA_20']
    elif 'SMA_20' in stock_data.columns:
        ma20 = latest['SMA_20']
    else:
        ma20 = stock_data['Close'].tail(20).mean()
    
    print(f"📈 20日均线(MA20): {ma20:.2f} 元")
    
    # 当前状态
    current_position = "强势 (价格≥MA20)" if latest['Close'] >= ma20 else "弱势 (价格<MA20)"
    position_diff = ((latest['Close'] - ma20) / ma20) * 100
    print(f"📍 当前状态: {current_position} ({position_diff:+.2f}%)")
    
    # 预测结果
    print(f"\n" + "-"*80)
    print("🔮 模型预测（未来5日）")
    print("-"*80)
    
    prediction_label = "强势 (价格≥MA20)" if prediction == 0 else "弱势 (价格<MA20)"
    prob_strong = probability[0]  # 强势概率
    prob_weak = probability[1]    # 弱势概率
    
    if prediction == 0:
        print(f"✅ 预测结果: {prediction_label}")
        print(f"📊 预测置信度: {prob_strong:.1%}")
        print(f"   - 强势概率: {prob_strong:.1%} ⭐")
        print(f"   - 弱势概率: {prob_weak:.1%}")
    else:
        print(f"⚠️  预测结果: {prediction_label}")
        print(f"📊 预测置信度: {prob_weak:.1%}")
        print(f"   - 强势概率: {prob_strong:.1%}")
        print(f"   - 弱势概率: {prob_weak:.1%} ⭐")
    
    # 置信度评级
    confidence = max(prob_strong, prob_weak)
    if confidence >= 0.80:
        confidence_level = "非常高 ⭐⭐⭐⭐⭐"
    elif confidence >= 0.70:
        confidence_level = "高 ⭐⭐⭐⭐"
    elif confidence >= 0.60:
        confidence_level = "中等 ⭐⭐⭐"
    else:
        confidence_level = "较低 ⭐⭐"
    
    print(f"\n🎯 置信度评级: {confidence_level}")
    
    # 模型信息
    if model_info:
        print(f"\n" + "-"*80)
        print("📈 模型信息")
        print("-"*80)
        print(f"模型类型: {model_info.get('model_name', 'Unknown')}")
        print(f"训练时间: {model_info.get('train_date', 'Unknown')}")
        print(f"模型准确率: {model_info.get('accuracy', 0):.2%}")
        print(f"模型精确率: {model_info.get('avg_precision', 0):.2%}")
    
    # 操作建议
    print(f"\n" + "-"*80)
    print("💡 操作建议")
    print("-"*80)
    
    if prediction == 0:  # 预测强势
        if prob_strong >= 0.75:
            print("📈 建议: 可考虑持有或适当增仓")
            print("   理由: 模型预测强势且置信度较高")
        elif prob_strong >= 0.60:
            print("📊 建议: 观察为主，谨慎持有")
            print("   理由: 模型预测强势但置信度中等")
        else:
            print("⚠️  建议: 谨慎对待，建议观望")
            print("   理由: 预测置信度较低")
    else:  # 预测弱势
        if prob_weak >= 0.75:
            print("📉 建议: 考虑减仓或观望")
            print("   理由: 模型预测弱势且置信度较高")
        elif prob_weak >= 0.60:
            print("⚠️  建议: 谨慎持有，控制仓位")
            print("   理由: 模型预测弱势但置信度中等")
        else:
            print("📊 建议: 继续观察，暂不操作")
            print("   理由: 预测置信度较低")
    
    print(f"\n" + "="*80)
    print("⚠️  风险提示:")
    print("   1. 模型预测仅供参考，不构成投资建议")
    print("   2. 股票投资有风险，入市需谨慎")
    print("   3. 请结合基本面、技术面等多方面因素综合判断")
    print("   4. 建议设置止损位，控制风险敞口")
    print("="*80 + "\n")

def print_models_comparison(stock_code, stock_data, predictions_dict, model_info=None):
    """
    打印多模型对比结果
    
    参数:
    stock_code: 股票代码
    stock_data: 股票数据
    predictions_dict: 所有模型的预测结果字典
    model_info: 模型信息
    """
    print("\n" + "="*80)
    print(f"📊 多模型预测对比 - {stock_code}")
    print("="*80)
    
    # 最新行情
    latest = stock_data.iloc[-1]
    latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
    
    print(f"\n📅 最新数据日期: {latest_date}")
    print(f"💰 最新收盘价: {latest['Close']:.2f} 元")
    
    if 'MA_20' in stock_data.columns:
        ma20 = latest['MA_20']
    elif 'SMA_20' in stock_data.columns:
        ma20 = latest['SMA_20']
    else:
        ma20 = stock_data['Close'].tail(20).mean()
    
    print(f"📈 20日均线(MA20): {ma20:.2f} 元")
    
    # 当前状态
    current_position = "强势 (价格≥MA20)" if latest['Close'] >= ma20 else "弱势 (价格<MA20)"
    position_diff = ((latest['Close'] - ma20) / ma20) * 100
    print(f"📍 当前状态: {current_position} ({position_diff:+.2f}%)")
    
    # 模型对比
    print(f"\n" + "-"*80)
    print("🔮 多模型预测对比（未来5日）")
    print("-"*80)
    
    # 统计投票结果
    vote_strong = sum(1 for pred in predictions_dict.values() if pred['prediction'] == 0)
    vote_weak = sum(1 for pred in predictions_dict.values() if pred['prediction'] == 1)
    total_models = len(predictions_dict)
    
    # 按置信度排序
    sorted_models = sorted(
        predictions_dict.items(), 
        key=lambda x: x[1]['confidence'], 
        reverse=True
    )
    
    # 表头
    print(f"\n{'模型名称':<20} {'预测':<10} {'强势概率':<10} {'弱势概率':<10} {'置信度':<8} {'训练精确率':<10}")
    print("-" * 80)
    
    for model_name, pred in sorted_models:
        prediction_label = "强势⭐" if pred['prediction'] == 0 else "弱势⚠️"
        prob_strong = pred['prob_strong']
        prob_weak = pred['prob_weak']
        confidence = pred['confidence']
        train_precision = pred['train_precision']
        
        print(f"{model_name:<20} {prediction_label:<10} {prob_strong:>7.1%}   {prob_weak:>7.1%}   {confidence:>6.1%}  {train_precision:>8.1%}")
    
    # 综合结论
    print(f"\n" + "-"*80)
    print("📊 综合结论")
    print("-"*80)
    
    print(f"\n投票统计:")
    print(f"  预测强势: {vote_strong}/{total_models} 个模型 ({vote_strong/total_models:.1%})")
    print(f"  预测弱势: {vote_weak}/{total_models} 个模型 ({vote_weak/total_models:.1%})")
    
    # 判断一致性
    consensus = max(vote_strong, vote_weak) / total_models
    
    if consensus >= 0.8:
        consensus_level = "非常一致 ⭐⭐⭐⭐⭐"
        consensus_desc = "所有模型意见高度一致"
    elif consensus >= 0.6:
        consensus_level = "比较一致 ⭐⭐⭐⭐"
        consensus_desc = "大多数模型意见一致"
    else:
        consensus_level = "意见分歧 ⭐⭐"
        consensus_desc = "模型意见存在分歧，建议谨慎"
    
    print(f"\n一致性评级: {consensus_level}")
    print(f"说明: {consensus_desc}")
    
    # 最终建议
    print(f"\n" + "-"*80)
    print("💡 综合建议")
    print("-"*80)
    
    if vote_strong > vote_weak:
        print(f"✅ 多数模型预测: 强势")
        if consensus >= 0.8:
            print("📈 建议: 可考虑持有或适当增仓")
            print("   理由: 模型一致性高，预测强势")
        elif consensus >= 0.6:
            print("📊 建议: 观察为主，谨慎持有")
            print("   理由: 多数模型预测强势，但存在一定分歧")
        else:
            print("⚠️  建议: 谨慎对待，建议观望")
            print("   理由: 模型意见分歧较大")
    else:
        print(f"⚠️  多数模型预测: 弱势")
        if consensus >= 0.8:
            print("📉 建议: 考虑减仓或观望")
            print("   理由: 模型一致性高，预测弱势")
        elif consensus >= 0.6:
            print("⚠️  建议: 谨慎持有，控制仓位")
            print("   理由: 多数模型预测弱势，但存在一定分歧")
        else:
            print("📊 建议: 继续观察，暂不操作")
            print("   理由: 模型意见分歧较大")
    
    # 计算平均置信度
    avg_confidence = sum(pred['confidence'] for pred in predictions_dict.values()) / total_models
    print(f"\n平均置信度: {avg_confidence:.1%}")
    
    if avg_confidence >= 0.75:
        print("   评级: 高置信度 ⭐⭐⭐⭐⭐")
    elif avg_confidence >= 0.65:
        print("   评级: 中等置信度 ⭐⭐⭐⭐")
    elif avg_confidence >= 0.55:
        print("   评级: 较低置信度 ⭐⭐⭐")
    else:
        print("   评级: 低置信度 ⭐⭐")
    
    # 模型信息
    if model_info:
        print(f"\n" + "-"*80)
        print("📈 模型训练信息")
        print("-"*80)
        print(f"训练时间: {model_info.get('train_date', 'Unknown')}")
        print(f"最佳模型: {model_info.get('best_model_name', 'Unknown')}")
        print(f"模型数量: {len(predictions_dict)} 个")
    
    # 风险提示
    print(f"\n" + "="*80)
    print("⚠️  风险提示:")
    print("   1. 多模型预测提供更全面的参考，但不保证准确")
    print("   2. 当模型意见分歧时，说明市场处于不确定状态")
    print("   3. 建议结合基本面、技术面等多方面因素综合判断")
    print("   4. 股票投资有风险，入市需谨慎")
    print("="*80 + "\n")

def predict_single_stock(stock_code, window_size=20, use_multi_models=True):
    """
    预测单只股票
    
    参数:
    stock_code: 股票代码
    window_size: 窗口大小
    use_multi_models: 是否使用多模型对比（默认True）
    
    返回:
    success: 是否成功
    """
    print("\n" + "="*80)
    print(f"🔍 开始预测股票: {stock_code}")
    print("="*80)
    
    try:
        # 1. 加载模型
        print("\n步骤1: 加载模型")
        print("-" * 50)
        
        # 尝试加载多模型
        all_models_data = None
        if use_multi_models:
            all_models_data = load_all_models()
        
        # 如果多模型加载失败，使用单一模型
        if all_models_data is None:
            model = load_trained_model()
            if model is None:
                return False
        
        feature_list = load_feature_list()
        if feature_list is None:
            return False
        
        model_info = load_model_info()
        
        # 2. 下载数据
        print("\n步骤2: 下载最新数据")
        print("-" * 50)
        stock_data = download_stock_for_prediction(stock_code, days=365)
        if stock_data is None:
            return False
        
        # 3. 提取特征
        print("\n步骤3: 提取特征")
        print("-" * 50)
        features_df = extract_features_for_prediction(stock_data, window_size=window_size)
        if features_df is None:
            return False
        
        # 4. 对齐特征
        print("\n步骤4: 对齐特征")
        print("-" * 50)
        aligned_features = align_features(features_df, feature_list)
        
        # 5. 预测
        print("\n步骤5: 执行预测")
        print("-" * 50)
        
        # 如果有多个模型，使用多模型对比
        if all_models_data is not None and len(all_models_data) > 1:
            predictions_dict = predict_with_all_models(all_models_data, aligned_features)
            # 6. 输出多模型对比结果
            print_models_comparison(stock_code, stock_data, predictions_dict, model_info)
        else:
            # 使用单一模型预测
            prediction, probability = predict_stock(model, aligned_features)
            # 6. 输出单模型结果
            print_prediction_result(stock_code, stock_data, prediction, probability, model_info)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def predict_multiple_stocks(stock_codes, window_size=20):
    """
    批量预测多只股票
    
    参数:
    stock_codes: 股票代码列表
    window_size: 窗口大小
    """
    print("\n" + "="*80)
    print(f"📊 批量预测 {len(stock_codes)} 只股票")
    print("="*80)
    
    results = []
    
    for i, stock_code in enumerate(stock_codes, 1):
        print(f"\n进度: {i}/{len(stock_codes)}")
        success = predict_single_stock(stock_code, window_size)
        results.append({'stock_code': stock_code, 'success': success})
        
        # 避免请求过快
        if i < len(stock_codes):
            import time
            time.sleep(1)
    
    # 汇总结果
    print("\n" + "="*80)
    print("📊 批量预测汇总")
    print("="*80)
    success_count = sum(1 for r in results if r['success'])
    print(f"成功预测: {success_count}/{len(stock_codes)}")
    
    failed = [r['stock_code'] for r in results if not r['success']]
    if failed:
        print(f"失败股票: {', '.join(failed)}")

def main():
    """
    主函数：实盘预测
    """
    print("="*80)
    print("🔮 股票实盘预测系统")
    print("="*80)
    print("基于已训练的模型对输入股票进行预测")
    print("预测目标：未来5日价格相对于20日均线的位置")
    print("="*80)
    
    # 检查模型是否存在
    if not os.path.exists('models/trained_model.pkl'):
        print("\n❌ 错误：未找到训练好的模型")
        print("请先运行以下命令训练模型:")
        print("  1. python stock_data_downloader.py")
        print("  2. python stock_feature_engineering.py")
        print("  3. python stock_statistical_analysis.py")
        return
    
    print("\n请选择预测模式:")
    print("  1. 单只股票预测")
    print("  2. 批量股票预测")
    print("  3. 使用默认测试股票")
    
    try:
        choice = input("\n请输入选项 (1/2/3): ").strip()
        
        if choice == '1':
            # 单只股票预测
            stock_code = input("请输入股票代码（6位数字）: ").strip()
            if len(stock_code) != 6 or not stock_code.isdigit():
                print("❌ 错误：股票代码格式不正确")
                return
            
            predict_single_stock(stock_code, window_size=20)
            
        elif choice == '2':
            # 批量预测
            print("请输入股票代码（多个代码用逗号或空格分隔）:")
            codes_input = input().strip()
            
            # 解析股票代码
            stock_codes = []
            for code in codes_input.replace(',', ' ').split():
                code = code.strip()
                if len(code) == 6 and code.isdigit():
                    stock_codes.append(code)
            
            if not stock_codes:
                print("❌ 错误：未输入有效的股票代码")
                return
            
            predict_multiple_stocks(stock_codes, window_size=20)
            
        elif choice == '3':
            # 使用默认测试股票
            test_stocks = [
                '600519',  # 贵州茅台
                '000001',  # 平安银行
                '600036',  # 招商银行
            ]
            
            print(f"\n使用默认测试股票: {', '.join(test_stocks)}")
            predict_multiple_stocks(test_stocks, window_size=20)
            
        else:
            print("❌ 错误：无效的选项")
            return
        
        print("\n✅ 预测完成！")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

