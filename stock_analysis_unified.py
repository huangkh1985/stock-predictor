# -*- coding: utf-8 -*-
"""
股票预测统一分析模块（云端优化版）
合并了数据下载、特征工程、模型训练和预测功能
移除了所有文件I/O操作，适用于Streamlit Cloud等只读环境
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = str(mp.cpu_count())

# ============================================================================
# 第一部分：数据下载模块
# ============================================================================

# 修复Streamlit Cloud权限问题：在导入efinance前修补其配置
EFINANCE_AVAILABLE = False
ef = None

try:
    is_streamlit_cloud = os.path.exists('/mount/src')
    
    if is_streamlit_cloud:
        print("[Streamlit Cloud] 检测到云环境，正在修补efinance配置...")
        temp_dir = tempfile.gettempdir()
        cache_dir = os.path.join(temp_dir, 'efinance_cache')
        
        import importlib.util
        import types
        
        mock_config = types.ModuleType('efinance.config')
        mock_config.DATA_DIR = Path(cache_dir)
        mock_config.SEARCH_RESULT_CACHE_PATH = Path(cache_dir) / 'search_cache.json'
        mock_config.MAX_CONNECTIONS = 5
        
        os.makedirs(cache_dir, exist_ok=True)
        sys.modules['efinance.config'] = mock_config
        
        print(f"[OK] efinance缓存目录设置为: {cache_dir}")
    
    import efinance as ef
    EFINANCE_AVAILABLE = True
    print("[OK] efinance导入成功")
    
except Exception as e:
    print(f"[WARNING] efinance导入失败: {e}")
    print("[INFO] 应用将在离线模式下运行")
    EFINANCE_AVAILABLE = False
    ef = None

def add_technical_indicators_inline(df):
    """
    添加技术指标（内联版本）
    """
    if df.empty or 'Close' not in df.columns:
        return df
    
    try:
        # 移动平均线
        for period in [5, 10, 20, 60]:
            df[f'MA_{period}'] = df['Close'].rolling(window=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
    except Exception as e:
        print(f"[WARNING] 技术指标计算失败: {e}")
    
    return df

def download_single_stock_data(stock_code, start_date='20240101', end_date='20250930'):
    """
    下载单只股票数据（内存版本）
    """
    if not EFINANCE_AVAILABLE or ef is None:
        print(f"[ERROR] efinance不可用，无法下载股票 {stock_code}")
        return None
    
    try:
        print(f"[处理] {stock_code}")
        
        # 获取K线数据
        kline_data = ef.stock.get_quote_history(
            stock_codes=[stock_code],
            beg=start_date,
            end=end_date
        )
        
        if not isinstance(kline_data, dict) or stock_code not in kline_data:
            return None
        
        df = kline_data[stock_code]
        if df.empty:
            return None
        
        # 重命名列
        df = df.rename(columns={
            '日期': 'Date',
            '开盘': 'Open',
            '收盘': 'Close',
            '最高': 'High',
            '最低': 'Low',
            '成交量': 'Volume',
            '成交额': 'Amount',
            '涨跌幅': 'PriceChangeRate',
            '涨跌额': 'PriceChangeAmount',
            '换手率': 'TurnoverRate',
            '振幅': 'Amplitude'
        })
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # 获取资金流数据
        try:
            money_flow = ef.stock.get_history_bill(stock_code)
            if isinstance(money_flow, pd.DataFrame) and not money_flow.empty:
                money_flow['日期'] = pd.to_datetime(money_flow['日期'])
                money_flow.set_index('日期', inplace=True)
                
                money_flow_columns = {
                    '主力净流入': 'MainNetInflow',
                    '主力净流入占比': 'MainNetInflowRatio',
                }
                
                for old_col, new_col in money_flow_columns.items():
                    if old_col in money_flow.columns:
                        money_flow[new_col] = money_flow[old_col]
                
                df = df.join(money_flow[[c for c in money_flow_columns.values() if c in money_flow.columns]], how='left')
        except:
            pass
        
        # 填充缺失的资金流列
        for col in ['MainNetInflow', 'MainNetInflowRatio']:
            if col not in df.columns:
                df[col] = 0
        
        # 添加技术指标
        df = add_technical_indicators_inline(df)
        
        # 添加价格特征
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # 数据清理
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(0)
        
        return df
        
    except Exception as e:
        print(f"[ERROR] {stock_code} 失败: {e}")
        return None

def download_multiple_stocks(stock_codes, start_date='20240101', end_date='20250930'):
    """
    批量下载股票数据（内存版本）
    """
    print(f"\n[开始] 下载 {len(stock_codes)} 只股票...")
    all_data = {}
    
    for stock_code in stock_codes:
        df = download_single_stock_data(stock_code, start_date, end_date)
        if df is not None:
            all_data[stock_code] = df
    
    print(f"[完成] 成功下载 {len(all_data)} 只股票")
    return all_data

# ============================================================================
# 第二部分：特征工程模块
# ============================================================================

def create_tsfresh_data(all_data, window_size=20, forecast_horizon=5):
    """
    创建TSFresh格式数据（内存版本）
    """
    print(f"\n[特征工程] 窗口={window_size}天, 预测期={forecast_horizon}天")
    
    tsfresh_data_list = []
    target_list = []
    
    tsfresh_features = [
        'Close', 'Open', 'High', 'Low', 'Volume', 'TurnoverRate',
        'PriceChangeRate', 'MainNetInflow', 'MainNetInflowRatio'
    ]
    
    for stock_code, data in all_data.items():
        for feature in tsfresh_features:
            if feature not in data.columns:
                data[feature] = 0
        
        if len(data) < window_size + forecast_horizon:
            continue
        
        for i in range(window_size, len(data) - forecast_horizon):
            window_id = f"{stock_code}_{i}"
            
            for feature in tsfresh_features:
                feature_window = data[feature].iloc[i - window_size: i]
                
                for time_idx, value in enumerate(feature_window.values):
                    tsfresh_data_list.append({
                        'id': window_id,
                        'time': time_idx,
                        'feature_name': feature,
                        'value': float(value) if not pd.isna(value) else 0.0
                    })
            
            future_close = float(data['Close'].iloc[i + forecast_horizon])
            
            if 'MA_20' in data.columns:
                future_ma20 = float(data['MA_20'].iloc[i + forecast_horizon])
            else:
                future_ma20 = float(data['Close'].iloc[i + forecast_horizon - 19: i + forecast_horizon + 1].mean())
            
            target = 1 if future_close < future_ma20 else 0
            target_list.append({'id': window_id, 'target': target})
    
    x_df = pd.DataFrame(tsfresh_data_list)
    y_df = pd.DataFrame(target_list)
    
    print(f"[完成] 生成 {len(y_df)} 个样本")
    return x_df, y_df

def extract_tsfresh_features(x_df, y_df, use_minimal=True):
    """
    提取TSFresh特征（内存版本）
    """
    print("\n[特征提取] 开始...")
    
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    
    feature_extraction_settings = MinimalFCParameters()
    feature_types = x_df['feature_name'].unique()
    all_extracted_features = []
    
    def extract_single_feature_type(feature_type):
        try:
            feature_df = x_df[x_df['feature_name'] == feature_type].copy()
            feature_df = feature_df.drop('feature_name', axis=1)
            
            extracted = extract_features(
                feature_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=feature_extraction_settings,
                n_jobs=1,
                disable_progressbar=True
            )
            
            extracted.columns = [f"{feature_type}_{col}" for col in extracted.columns]
            return extracted
        except:
            return None
    
    # 并行提取
    max_workers = min(4, mp.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single_feature_type, ft): ft for ft in feature_types}
        
        for future in as_completed(futures):
            extracted = future.result()
            if extracted is not None:
                all_extracted_features.append(extracted)
    
    if not all_extracted_features:
        return None, None
    
    x_extracted = pd.concat(all_extracted_features, axis=1)
    x_extracted = impute(x_extracted)
    
    y_series = y_df.set_index('id')['target']
    x_extracted = x_extracted.loc[y_series.index]
    
    print(f"[完成] 提取 {x_extracted.shape[1]} 个特征")
    return x_extracted, y_series

def select_features(x_extracted, y_series):
    """
    特征选择（内存版本）
    """
    print("\n[特征选择] 开始...")
    
    from tsfresh import select_features as tsfresh_select
    from sklearn.ensemble import RandomForestClassifier
    
    try:
        x_filtered = tsfresh_select(x_extracted, y_series, fdr_level=0.01)
        print(f"[统计筛选] 保留 {x_filtered.shape[1]} 个特征")
    except:
        x_filtered = x_extracted
    
    if x_filtered.shape[1] > 100:
        print("[重要性筛选] 开始...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(x_filtered, y_series)
        
        importances = pd.DataFrame({
            'feature': x_filtered.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        cumsum = importances['importance'].cumsum()
        n_features = (cumsum <= 0.95).sum() + 1
        selected_features = importances.head(n_features)['feature'].tolist()
        x_filtered = x_filtered[selected_features]
        print(f"[重要性筛选] 保留 {len(selected_features)} 个特征")
    
    print(f"[完成] 最终特征数: {x_filtered.shape[1]}")
    return x_filtered

# ============================================================================
# 第三部分：模型训练模块
# ============================================================================

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

def find_optimal_threshold(y_true, y_proba, metric='precision', min_recall=0.3):
    """寻找最优分类阈值"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        min_class_recall = min(recall_0, recall_1)
        
        if min_class_recall >= min_recall:
            avg_precision = (precision_0 + precision_1) / 2
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            balanced_score = (avg_precision + (recall_0 + recall_1) / 2) / 2
            
            if metric == 'precision':
                score = avg_precision
            elif metric == 'f1':
                score = f1
            else:
                score = balanced_score
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    return best_threshold, best_score

def train_models(X_train, X_test, y_train, y_test, use_multi_models=True):
    """
    训练模型（内存版本）
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, classification_report
    
    print("\n[模型训练] 开始...")
    
    # SMOTE过采样
    try:
        from imblearn.combine import SMOTETomek
        smote_tomek = SMOTETomek(random_state=42)
        X_train_use, y_train_use = smote_tomek.fit_resample(X_train, y_train)
        print(f"[SMOTE] 平衡后样本数: {len(y_train_use)}")
    except:
        X_train_use, y_train_use = X_train, y_train
    
    X_train_cleaned = clean_feature_names(X_train_use) if isinstance(X_train_use, pd.DataFrame) else X_train_use
    X_test_cleaned = clean_feature_names(X_test) if isinstance(X_test, pd.DataFrame) else X_test
    
    models_dict = {}
    
    # Random Forest
    print("[训练] Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='log2',
        class_weight={0: 1, 1: 2.5},
        n_jobs=-1
    )
    rf_model.fit(X_train_cleaned, y_train_use)
    models_dict['RandomForest'] = rf_model
    
    # XGBoost
    if use_multi_models:
        try:
            import xgboost as xgb
            print("[训练] XGBoost...")
            neg_count = (y_train_use == 0).sum()
            pos_count = (y_train_use == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_cleaned, y_train_use)
            models_dict['XGBoost'] = xgb_model
        except:
            pass
        
        # LightGBM
        try:
            import lightgbm as lgb
            print("[训练] LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight={0: 1, 1: 2.5},
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train_cleaned, y_train_use)
            models_dict['LightGBM'] = lgb_model
        except:
            pass
    
    # 选择最佳模型
    best_model = None
    best_model_name = None
    best_precision = 0
    all_models_data = {}
    
    for model_name, model in models_dict.items():
        y_proba = model.predict_proba(X_test_cleaned)[:, 1]
        optimal_threshold, _ = find_optimal_threshold(y_test, y_proba, metric='precision', min_recall=0.3)
        y_pred = (y_proba >= optimal_threshold).astype(int)
        
        precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        avg_precision = (precision_0 + precision_1) / 2
        accuracy = accuracy_score(y_test, y_pred)
        
        all_models_data[model_name] = {
            'model': model,
            'optimal_threshold': optimal_threshold,
            'accuracy': accuracy,
            'avg_precision': avg_precision,
            'precision_0': precision_0,
            'precision_1': precision_1
        }
        
        print(f"  {model_name}: 精确率={avg_precision:.2%}, 阈值={optimal_threshold:.3f}")
        
        if avg_precision > best_precision:
            best_precision = avg_precision
            best_model = model
            best_model_name = model_name
    
    print(f"\n[最佳模型] {best_model_name} (精确率={best_precision:.2%})")
    
    # 最终预测
    y_proba = best_model.predict_proba(X_test_cleaned)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_test, y_proba, metric='precision', min_recall=0.3)
    y_pred = (y_proba >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n[性能评估]")
    print(classification_report(y_test, y_pred, target_names=['强势(≥MA20)', '弱势(<MA20)'], zero_division=0))
    
    return best_model, all_models_data, X_test_cleaned.columns.tolist()

# ============================================================================
# 第四部分：预测模块
# ============================================================================

def predict_single_stock_inline(stock_code, model, all_models_data, feature_list,
                                window_size=20, days=365):
    """
    预测单只股票（内存版本）
    """
    print(f"\n[预测] {stock_code}")
    
    # 下载数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    stock_data = download_single_stock_data(stock_code, start_date_str, end_date_str)
    if stock_data is None:
        print(f"[失败] 无法下载数据")
        return None
    
    # 提取特征
    from tsfresh import extract_features
    from tsfresh.feature_extraction import MinimalFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    
    tsfresh_features = [
        'Close', 'Open', 'High', 'Low', 'Volume', 'TurnoverRate',
        'PriceChangeRate', 'MainNetInflow', 'MainNetInflowRatio'
    ]
    
    for feature in tsfresh_features:
        if feature not in stock_data.columns:
            stock_data[feature] = 0
    
    if len(stock_data) < window_size:
        print(f"[失败] 数据不足")
        return None
    
    window_data = stock_data.iloc[-window_size:]
    tsfresh_data_list = []
    window_id = "prediction"
    
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
    
    all_extracted_features = []
    feature_extraction_settings = MinimalFCParameters()
    
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
            extracted.columns = [f"{feature_type}_{col}" for col in extracted.columns]
            all_extracted_features.append(extracted)
        except:
            continue
    
    if not all_extracted_features:
        print(f"[失败] 特征提取失败")
        return None
    
    x_extracted = pd.concat(all_extracted_features, axis=1)
    x_extracted = impute(x_extracted)
    x_extracted = clean_feature_names(x_extracted)
    
    # 对齐特征
    aligned_df = pd.DataFrame(0, index=x_extracted.index, columns=feature_list)
    for col in x_extracted.columns:
        if col in feature_list:
            aligned_df[col] = x_extracted[col]
    
    # 预测
    if all_models_data and len(all_models_data) > 1:
        predictions_dict = {}
        for model_name, model_data in all_models_data.items():
            m = model_data['model']
            threshold = model_data.get('optimal_threshold', 0.5)
            probability = m.predict_proba(aligned_df)[0]
            prediction = 1 if probability[1] >= threshold else 0
            
            predictions_dict[model_name] = {
                'prediction': prediction,
                'probability': probability,
                'prob_strong': probability[0],  # 强势概率
                'prob_weak': probability[1],    # 弱势概率
                'confidence': max(probability),
                'optimal_threshold': threshold,
                'train_accuracy': model_data.get('accuracy', 0),
                'train_precision': model_data.get('avg_precision', 0)
            }
        
        return {
            'stock_code': stock_code,
            'stock_data': stock_data,
            'predictions': predictions_dict,
            'type': 'multi'
        }
    else:
        prediction = model.predict(aligned_df)[0]
        probability = model.predict_proba(aligned_df)[0]
        
        return {
            'stock_code': stock_code,
            'stock_data': stock_data,
            'prediction': prediction,
            'probability': probability,
            'type': 'single'
        }

# ============================================================================
# 主流程函数
# ============================================================================

def train_stock_prediction_model(stock_codes, window_size=20, forecast_horizon=5,
                                 use_multi_models=True):
    """
    完整训练流程（内存版本）
    
    返回：
    - best_model: 最佳模型
    - all_models_data: 所有模型数据
    - feature_list: 特征列表
    """
    print("="*80)
    print("股票预测模型训练（云端优化版）")
    print("="*80)
    
    # 1. 下载数据
    print("\n[步骤1] 下载股票数据")
    all_data = download_multiple_stocks(stock_codes)
    if not all_data:
        print("[错误] 没有成功下载任何数据")
        return None, None, None
    
    # 2. 特征工程
    print("\n[步骤2] 特征工程")
    x_df, y_df = create_tsfresh_data(all_data, window_size, forecast_horizon)
    if x_df.empty:
        print("[错误] 特征数据生成失败")
        return None, None, None
    
    # 3. 提取特征
    print("\n[步骤3] TSFresh特征提取")
    x_extracted, y_series = extract_tsfresh_features(x_df, y_df)
    if x_extracted is None:
        print("[错误] 特征提取失败")
        return None, None, None
    
    # 4. 特征选择
    print("\n[步骤4] 特征选择")
    x_filtered = select_features(x_extracted, y_series)
    x_filtered = clean_feature_names(x_filtered)
    
    # 5. 训练模型
    print("\n[步骤5] 模型训练")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        x_filtered, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    best_model, all_models_data, feature_list = train_models(
        X_train, X_test, y_train, y_test, use_multi_models
    )
    
    print("\n" + "="*80)
    print("[完成] 模型训练完成")
    print(f"  特征数: {len(feature_list)}")
    print(f"  模型数: {len(all_models_data)}")
    print("="*80)
    
    return best_model, all_models_data, feature_list

def predict_stocks_inline(stock_codes, model, all_models_data, feature_list,
                          window_size=20):
    """
    批量预测（内存版本）
    """
    print("\n" + "="*80)
    print(f"股票预测（共{len(stock_codes)}只）")
    print("="*80)
    
    results = []
    for stock_code in stock_codes:
        result = predict_single_stock_inline(
            stock_code, model, all_models_data, feature_list, window_size
        )
        if result:
            results.append(result)
    
    print(f"\n[完成] 成功预测 {len(results)}/{len(stock_codes)} 只股票")
    return results

# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """
    使用示例
    """
    # 示例股票列表
    stock_codes = ['600519', '000001', '600036']
    
    # 训练模型
    best_model, all_models_data, feature_list = train_stock_prediction_model(
        stock_codes=stock_codes,
        window_size=20,
        forecast_horizon=5,
        use_multi_models=True
    )
    
    if best_model is None:
        print("训练失败")
        return
    
    # 预测
    test_stocks = ['600519', '000002']
    results = predict_stocks_inline(
        test_stocks,
        best_model,
        all_models_data,
        feature_list,
        window_size=20
    )
    
    # 显示结果
    for result in results:
        print(f"\n股票: {result['stock_code']}")
        if result['type'] == 'multi':
            for model_name, pred in result['predictions'].items():
                label = "强势" if pred['prediction'] == 0 else "弱势"
                print(f"  {model_name}: {label} (置信度={pred['confidence']:.1%})")
        else:
            label = "强势" if result['prediction'] == 0 else "弱势"
            print(f"  预测: {label} (概率={result['probability']})")

if __name__ == '__main__':
    print("股票预测统一分析模块（云端优化版）")
    print("所有数据处理均在内存中完成，无需文件读写权限")
    print("\n使用方法：")
    print("  from stock_analysis_unified import train_stock_prediction_model, predict_stocks_inline")

