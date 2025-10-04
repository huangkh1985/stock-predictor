"""
股票特征工程模块
负责TSFresh特征提取、特征选择等数据处理工作
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestClassifier
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 抑制joblib/loky在Windows上的CPU核心数警告
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
os.environ['LOKY_MAX_CPU_COUNT'] = str(mp.cpu_count())

def load_stock_data(data_path='data/all_stock_data.pkl'):
    """
    从文件加载股票数据
    
    参数:
    data_path: 数据文件路径
    
    返回:
    all_data: 股票数据字典
    """
    print(f"正在加载股票数据...")
    print(f"数据路径: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在 - {data_path}")
        print("请先运行 stock_data_downloader.py 下载数据")
        return None
    
    try:
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        print(f"成功加载 {len(all_data)} 只股票的数据")
        for stock_code, data in all_data.items():
            print(f"  {stock_code}: {len(data)} 条记录, {data.shape[1]} 个特征")
        
        return all_data
    
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None

def create_multi_feature_tsfresh_data(all_data, window_size=30, forecast_horizon=5):
    """
    创建多特征TSFresh格式数据
    
    参数:
    all_data: 股票数据字典
    window_size: 滑动窗口大小
    forecast_horizon: 预测期限
    
    返回:
    x_df: TSFresh格式的特征数据
    y_df: 目标变量数据
    """
    print(f"正在转换多特征数据为TSFresh格式...")
    print(f"窗口大小: {window_size}天")
    print(f"预测期限: {forecast_horizon}天")
    
    tsfresh_data_list = []
    target_list = []
    
    # 选择用于TSFresh分析的特征
    tsfresh_features = [
        'Close', 'Open', 'High', 'Low', 'Volume', 'TurnoverRate', 
        'PriceChangeRate', 'MainNetInflow', 'MainNetInflowRatio'
    ]

    for stock_code, data in all_data.items():
        print(f"处理股票 {stock_code}...")
        
        # 确保所有必要的列都存在
        for feature in tsfresh_features:
            if feature not in data.columns:
                data[feature] = 0
                
        # 检查数据长度
        if len(data) < window_size + forecast_horizon:
            print(f"警告: {stock_code} 数据点不足，跳过")
            continue

        # 为每个滑动窗口创建样本
        for i in range(window_size, len(data) - forecast_horizon):
            window_id = f"{stock_code}_{i}"
            
            # 提取多个特征的窗口数据
            for feature in tsfresh_features:
                feature_window = data[feature].iloc[i - window_size : i]
                
                # 创建tsfresh格式的DataFrame片段
                for time_idx, value in enumerate(feature_window.values):
                    tsfresh_data_list.append({
                        'id': window_id,
                        'time': time_idx,
                        'feature_name': feature,
                        'value': float(value) if not pd.isna(value) else 0.0
                    })

            # 创建二分类目标变量（基于MA20）
            future_close = float(data['Close'].iloc[i + forecast_horizon])
            
            # 获取未来时点的MA20
            if 'MA_20' in data.columns:
                future_ma20 = float(data['MA_20'].iloc[i + forecast_horizon])
            elif 'SMA_20' in data.columns:
                future_ma20 = float(data['SMA_20'].iloc[i + forecast_horizon])
            else:
                # 如果没有预计算的MA20，临时计算
                future_ma20 = float(data['Close'].iloc[i + forecast_horizon - 19 : i + forecast_horizon + 1].mean())
            
            # 二分类标签：0=收盘价>=MA20(强势), 1=收盘价<MA20(弱势)
            target = 1 if future_close < future_ma20 else 0
            target_list.append({'id': window_id, 'target': target})

    # 合并所有数据
    x_df = pd.DataFrame(tsfresh_data_list)
    y_df = pd.DataFrame(target_list)
    
    print(f"生成的特征数据形状: {x_df.shape}")
    print(f"生成的目标数据形状: {y_df.shape}")
    print(f"包含的特征类型: {x_df['feature_name'].unique()}")

    return x_df, y_df

def get_optimal_thread_count():
    """
    获取最优线程数配置
    
    返回:
    max_workers: 最优线程数
    """
    cpu_count = mp.cpu_count()
    
    # 根据CPU核心数智能配置线程数
    if cpu_count <= 2:
        return 1  # 双核及以下使用单线程
    elif cpu_count <= 4:
        return 2  # 四核使用2线程
    elif cpu_count <= 8:
        return 4  # 八核使用4线程
    else:
        return min(6, cpu_count // 2)  # 高核心数限制最大线程数

def advanced_feature_selection(x_extracted, y_series, method='hybrid'):
    """
    高级特征选择方法
    
    参数:
    x_extracted: 提取的原始特征
    y_series: 目标变量
    method: 特征选择方法 ('statistical', 'importance', 'hybrid', 'recursive')
    
    返回:
    x_filtered: 筛选后的特征
    """
    print(f"\n🔍 执行高级特征选择 (方法: {method})...")
    
    if method == 'statistical':
        # 方法1: 统计显著性筛选
        from tsfresh.feature_selection.relevance import calculate_relevance_table
        relevance_table = calculate_relevance_table(x_extracted, y_series)
        relevant_features = relevance_table[relevance_table.relevant].feature
        x_filtered = x_extracted[relevant_features]
        print(f"  统计筛选后特征数: {len(relevant_features)}")
        
    elif method == 'importance':
        # 方法2: 基于特征重要性的筛选
        print("  训练初步模型评估特征重要性...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(x_extracted, y_series)
        
        importances = pd.DataFrame({
            'feature': x_extracted.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 保留重要性前50%的特征
        threshold = importances['importance'].quantile(0.5)
        selected_features = importances[importances['importance'] >= threshold]['feature'].tolist()
        x_filtered = x_extracted[selected_features]
        print(f"  重要性筛选后特征数: {len(selected_features)}")
        
    elif method == 'recursive':
        # 方法3: 递归特征消除
        from sklearn.feature_selection import RFECV
        print("  执行递归特征消除（可能较慢）...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rfecv = RFECV(estimator=rf, step=10, cv=3, scoring='precision_macro', n_jobs=-1)
        rfecv.fit(x_extracted, y_series)
        x_filtered = x_extracted.iloc[:, rfecv.support_]
        print(f"  递归消除后特征数: {x_filtered.shape[1]}")
        
    else:  # hybrid
        # 方法4: 混合策略（推荐）
        print("  第一阶段: 统计筛选...")
        try:
            x_filtered = select_features(x_extracted, y_series, fdr_level=0.01)
            print(f"    统计筛选保留: {x_filtered.shape[1]} 个特征")
        except:
            print("    统计筛选失败，使用全部特征")
            x_filtered = x_extracted
        
        if x_filtered.shape[1] > 100:
            print("  第二阶段: 重要性筛选...")
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(x_filtered, y_series)
            
            importances = pd.DataFrame({
                'feature': x_filtered.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # 保留累计重要性达到95%的特征
            cumsum = importances['importance'].cumsum()
            n_features = (cumsum <= 0.95).sum() + 1
            selected_features = importances.head(n_features)['feature'].tolist()
            x_filtered = x_filtered[selected_features]
            print(f"    重要性筛选保留: {len(selected_features)} 个特征（累计重要性95%）")
    
    return x_filtered

def extract_enhanced_features(x_df, y_df, use_minimal=True):
    """
    提取增强的TSFresh特征
    
    参数:
    x_df: TSFresh格式的输入数据
    y_df: 目标变量数据
    use_minimal: 是否使用精简的特征集（推荐True，速度快）
    
    返回:
    x_filtered: 筛选后的特征
    y_series: 目标变量Series
    """
    print("开始提取增强TSFresh特征...")
    
    # 特征提取设置
    if use_minimal:
        print("使用精简特征集（速度快，推荐）")
        from tsfresh.feature_extraction import MinimalFCParameters
        feature_extraction_settings = MinimalFCParameters()
    else:
        print("使用完整特征集（特征多，但耗时长）")
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        feature_extraction_settings = ComprehensiveFCParameters()
    
    # 智能并行处理配置
    cpu_count = mp.cpu_count()
    max_workers = get_optimal_thread_count()
    
    print(f"系统CPU核心数: {cpu_count}")
    print(f"使用多线程模式，最优线程数: {max_workers}")
    
    # 为每个特征类型单独进行特征提取
    feature_types = x_df['feature_name'].unique()
    all_extracted_features = []
    
    def extract_single_feature_type(feature_type):
        """提取单个特征类型的TSFresh特征（线程安全）"""
        try:
            print(f"开始提取 {feature_type} 特征...")
            
            feature_df = x_df[x_df['feature_name'] == feature_type].copy()
            feature_df = feature_df.drop('feature_name', axis=1)
            
            extracted = extract_features(
                feature_df,
                column_id="id",
                column_sort="time",
                column_value="value",
                default_fc_parameters=feature_extraction_settings,
                n_jobs=1,  # 每个线程内部使用单进程
                disable_progressbar=True
            )
            
            # 重命名列以包含特征类型
            extracted.columns = [f"{feature_type}_{col}" for col in extracted.columns]
            
            print(f"✅ {feature_type} 特征提取完成，特征数: {extracted.shape[1]}")
            return extracted
            
        except Exception as e:
            print(f"❌ {feature_type} 特征提取失败: {e}")
            return None
    
    # 使用多线程并行提取
    print(f"开始并行提取 {len(feature_types)} 种特征类型...")
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_feature = {
            executor.submit(extract_single_feature_type, feature_type): feature_type 
            for feature_type in feature_types
        }
        
        completed_count = 0
        total_features = 0
        
        for future in as_completed(future_to_feature):
            feature_type = future_to_feature[future]
            try:
                extracted = future.result()
                if extracted is not None:
                    all_extracted_features.append(extracted)
                    total_features += extracted.shape[1]
                    
                completed_count += 1
                elapsed_time = time.time() - start_time
                print(f"进度: {completed_count}/{len(feature_types)} 完成，"
                      f"已提取特征数: {total_features}，"
                      f"耗时: {elapsed_time:.1f}秒")
                      
            except Exception as e:
                print(f"线程执行异常 {feature_type}: {e}")
                completed_count += 1
    
    total_time = time.time() - start_time
    print(f"多线程特征提取完成，总耗时: {total_time:.2f}秒")
    print(f"平均每种特征类型耗时: {total_time/len(feature_types):.2f}秒")
    
    # 合并所有特征
    if all_extracted_features:
        x_extracted = pd.concat(all_extracted_features, axis=1)
    else:
        print("错误: 没有成功提取任何特征")
        return None, None
    
    print(f"提取的原始特征数量: {x_extracted.shape[1]}")
    
    # 处理无限值和缺失值
    print("处理缺失值和无限值...")
    x_extracted = impute(x_extracted)
    
    # 准备目标变量
    y_series = y_df.set_index('id')['target']
    x_extracted = x_extracted.loc[y_series.index]
    
    # 使用高级特征选择
    print("\n" + "="*70)
    print("🚀 V2.4 精确率提升 - 智能特征选择")
    print("="*70)
    x_filtered = advanced_feature_selection(x_extracted, y_series, method='hybrid')
    print(f"✅ 最终特征数量: {x_filtered.shape[1]}")
    print(f"   特征筛选比率: {x_filtered.shape[1]/x_extracted.shape[1]:.2%}")
    
    return x_filtered, y_series

def save_processed_features(x_filtered, y_series, output_dir='data'):
    """
    保存处理后的特征数据
    
    参数:
    x_filtered: 筛选后的特征
    y_series: 目标变量
    output_dir: 输出目录
    """
    print(f"\n保存处理后的特征数据...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为CSV
    x_filtered.to_csv(f'{output_dir}/processed_features.csv')
    y_series.to_csv(f'{output_dir}/processed_targets.csv')
    
    # 保存为pickle（更快）
    with open(f'{output_dir}/processed_features.pkl', 'wb') as f:
        pickle.dump({'X': x_filtered, 'y': y_series}, f)
    
    print(f"✅ 特征数据已保存到: {output_dir}/")
    print(f"   - processed_features.csv ({x_filtered.shape[1]} 个特征)")
    print(f"   - processed_targets.csv ({len(y_series)} 个样本)")
    print(f"   - processed_features.pkl (pickle格式，加载更快)")

def load_processed_features(input_dir='data'):
    """
    加载处理后的特征数据
    
    参数:
    input_dir: 输入目录
    
    返回:
    x_filtered: 特征数据
    y_series: 目标变量
    """
    print(f"正在加载处理后的特征数据...")
    
    pkl_path = f'{input_dir}/processed_features.pkl'
    
    if os.path.exists(pkl_path):
        print(f"从pickle文件加载（更快）...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        x_filtered = data['X']
        y_series = data['y']
        print(f"✅ 加载完成: {x_filtered.shape[1]} 个特征, {len(y_series)} 个样本")
        return x_filtered, y_series
    else:
        print(f"从CSV文件加载...")
        x_filtered = pd.read_csv(f'{input_dir}/processed_features.csv', index_col=0)
        y_series = pd.read_csv(f'{input_dir}/processed_targets.csv', index_col=0, squeeze=True)
        print(f"✅ 加载完成: {x_filtered.shape[1]} 个特征, {len(y_series)} 个样本")
        return x_filtered, y_series

def main():
    """
    特征工程主函数
    """
    from datetime import datetime
    
    print("="*80)
    print("股票特征工程处理")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. 加载股票数据
        print("\n步骤1: 加载股票数据")
        print("-" * 50)
        stock_data = load_stock_data('data/all_stock_data.pkl')
        
        if not stock_data:
            print("数据加载失败，程序退出")
            print("请先运行: python stock_data_downloader.py")
            return
        
        # 2. 转换为TSFresh格式
        print(f"\n步骤2: 转换为TSFresh格式")
        print("-" * 50)
        x_df, y_df = create_multi_feature_tsfresh_data(
            stock_data, 
            window_size=20,  # 可调整：观察窗口大小
            forecast_horizon=5  # 可调整：预测期限
        )
        
        if x_df.empty or y_df.empty:
            print("数据转换失败，程序退出")
            return
        
        # 3. 特征提取和选择
        print(f"\n步骤3: 特征提取和选择")
        print("-" * 50)
        x_filtered, y_series = extract_enhanced_features(
            x_df, y_df, 
            use_minimal=True  # True=快速模式, False=完整特征
        )
        
        if x_filtered is None:
            print("特征提取失败，程序退出")
            return
        
        # 4. 保存处理后的特征
        print(f"\n步骤4: 保存处理后的特征")
        print("-" * 50)
        save_processed_features(x_filtered, y_series, output_dir='data')
        
        # 5. 总结
        print("\n" + "="*80)
        print("特征工程完成！")
        print("="*80)
        print(f"处理股票数量: {len(stock_data)}")
        print(f"生成样本数量: {len(y_series)}")
        print(f"提取特征数量: {x_filtered.shape[1]}")
        
        class_dist = pd.Series(y_series).value_counts().sort_index()
        print(f"\n目标变量分布:")
        print(f"  强势(>=MA20): {class_dist.get(0, 0)} 样本 ({class_dist.get(0, 0)/len(y_series):.2%})")
        print(f"  弱势(<MA20): {class_dist.get(1, 0)} 样本 ({class_dist.get(1, 0)/len(y_series):.2%})")
        
        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n下一步: 运行 stock_statistical_analysis.py 进行统计分析")
        
    except Exception as e:
        print(f"特征工程过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


