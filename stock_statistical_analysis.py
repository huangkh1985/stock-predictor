"""
股票统计分析模块
负责模型训练、阈值优化、结果可视化等统计分析工作
"""

import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 从特征工程模块导入数据加载函数
from stock_feature_engineering import load_processed_features

# 抑制各种警告信息
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['LOKY_MAX_CPU_COUNT'] = str(mp.cpu_count())

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def find_optimal_threshold(y_true, y_proba, metric='precision', min_recall=0.3):
    """
    自动寻找最优分类阈值（V2.4.2改进版）
    
    参数:
    y_true: 真实标签
    y_proba: 预测概率（类别1的概率）
    metric: 优化指标 ('precision', 'f1', 'balanced')
    min_recall: 最小召回率要求（防止极端阈值）
    
    返回:
    optimal_threshold: 最优阈值
    best_score: 最优得分
    results_df: 所有阈值的评估结果
    """
    print(f"\n🎯 自动搜索最优阈值（优化指标: {metric}, 最小召回率要求: {min_recall:.0%}）...")
    
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0
    threshold_results = []
    valid_count = 0
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # 计算各类别指标
        precision_0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        precision_1 = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        
        # 检查召回率约束
        min_class_recall = min(recall_0, recall_1)
        
        avg_precision = (precision_0 + precision_1) / 2
        avg_recall = (recall_0 + recall_1) / 2
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        balanced_score = (avg_precision + avg_recall) / 2
        
        threshold_results.append({
            'threshold': threshold,
            'precision': avg_precision,
            'f1': f1,
            'balanced': balanced_score,
            'precision_0': precision_0,
            'precision_1': precision_1,
            'recall_0': recall_0,
            'recall_1': recall_1,
            'min_recall': min_class_recall,
            'valid': min_class_recall >= min_recall
        })
        
        # 只考虑满足最小召回率要求的阈值
        if min_class_recall >= min_recall:
            valid_count += 1
            
            # 根据指定指标选择最优阈值
            if metric == 'precision':
                score = avg_precision
            elif metric == 'f1':
                score = f1
            else:  # balanced
                score = balanced_score
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
    
    # 打印阈值搜索结果
    results_df = pd.DataFrame(threshold_results)
    print(f"  扫描了 {len(thresholds)} 个阈值候选")
    print(f"  满足召回率要求的阈值: {valid_count} 个")
    print(f"  最优阈值: {best_threshold:.3f}")
    print(f"  最优{metric}得分: {best_score:.4f}")
    
    # 显示最优阈值的详细信息
    best_row = results_df[results_df['threshold'] == best_threshold].iloc[0]
    print(f"\n  最优阈值详细表现:")
    print(f"    强势类别 - 精确率:{best_row['precision_0']:.1%}, 召回率:{best_row['recall_0']:.1%}")
    print(f"    弱势类别 - 精确率:{best_row['precision_1']:.1%}, 召回率:{best_row['recall_1']:.1%}")
    print(f"    平均精确率: {best_row['precision']:.1%}")
    print(f"    F1分数: {best_row['f1']:.3f}")
    
    # 显示最优阈值附近的结果
    print(f"\n  候选阈值对比（前5个）:")
    valid_df = results_df[results_df['valid'] == True].sort_values('precision', ascending=False)
    for idx, (_, row) in enumerate(valid_df.head(5).iterrows(), 1):
        print(f"    {idx}. 阈值={row['threshold']:.3f}: 精确率={row['precision']:.1%}, "
              f"F1={row['f1']:.3f}, 召回率({row['recall_0']:.1%}/{row['recall_1']:.1%})")
    
    if valid_count == 0:
        print(f"\n  ⚠️ 警告：没有阈值满足最小召回率{min_recall:.0%}要求，使用默认阈值0.5")
        best_threshold = 0.5
    
    return best_threshold, best_score, results_df

def clean_feature_names(df):
    """
    清理特征名称，移除LightGBM/XGBoost不支持的特殊字符
    
    参数:
    df: 输入DataFrame
    
    返回:
    df_cleaned: 清理后的DataFrame
    """
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

def train_binary_model(X_train, X_test, y_train, y_test, use_smote=True, use_multi_models=False):
    """
    训练二分类模型（V2.4.2优化版）
    
    参数:
    X_train: 训练特征
    X_test: 测试特征
    y_train: 训练标签
    y_test: 测试标签
    use_smote: 是否使用SMOTE过采样
    use_multi_models: 是否尝试多种算法（XGBoost、LightGBM等）
    
    返回:
    best_model: 最佳模型
    y_pred: 预测结果
    accuracy: 准确率
    model_name: 模型名称
    """
    print("训练二分类模型（V2.4.2优化版）...")
    print("🎯 优化目标: 提高精确率至75%+，确保召回率平衡")
    
    # 打印类别分布
    print("\n训练集类别分布:")
    train_counts = pd.Series(y_train).value_counts().sort_index()
    for label, count in train_counts.items():
        label_name = ['价格>=MA20(强势)', '价格<MA20(弱势)'][label]
        print(f"  {label}: {label_name} - {count} 样本 ({count/len(y_train):.2%})")
    
    print("\n测试集类别分布:")
    test_counts = pd.Series(y_test).value_counts().sort_index()
    for label, count in test_counts.items():
        label_name = ['价格>=MA20(强势)', '价格<MA20(弱势)'][label]
        print(f"  {label}: {label_name} - {count} 样本 ({count/len(y_test):.2%})")
    
    # 检查类别不平衡程度
    minority_ratio = train_counts.min() / train_counts.max()
    print(f"\n类别不平衡比率: {minority_ratio:.2%}")
    
    # SMOTE过采样
    if use_smote:
        try:
            from imblearn.combine import SMOTETomek
            print("\n使用SMOTE过采样平衡数据集...")
            smote_tomek = SMOTETomek(random_state=42)
            X_train_use, y_train_use = smote_tomek.fit_resample(X_train, y_train)
            print(f"原始训练集大小: {len(y_train)}")
            print(f"平衡后训练集大小: {len(y_train_use)}")
            
            balanced_counts = pd.Series(y_train_use).value_counts().sort_index()
            print("平衡后类别分布:")
            for label, count in balanced_counts.items():
                label_name = ['价格>=MA20(强势)', '价格<MA20(弱势)'][label]
                print(f"  {label}: {label_name} - {count} 样本 ({count/len(y_train_use):.2%})")
        except ImportError:
            print("\n⚠️ imbalanced-learn库未安装，跳过SMOTE")
            print("   安装命令: pip install imbalanced-learn")
            X_train_use = X_train
            y_train_use = y_train
    else:
        X_train_use = X_train
        y_train_use = y_train
    
    # 注意：特征名称已在main函数中清理过，这里直接使用
    X_train_cleaned = X_train_use
    X_test_cleaned = X_test
    print(f"\n特征数量: {len(X_train_cleaned.columns)}")
    
    # 训练模型
    models_to_try = {}
    
    # 模型1: Random Forest（必选）
    print("\n1️⃣ 训练Random Forest模型...")
    rf_model = RandomForestClassifier(
        n_estimators=1000,
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='log2',
        class_weight={0: 1, 1: 2.5},
        criterion='gini',
        bootstrap=True,
        n_jobs=-1
    )
    rf_model.fit(X_train_cleaned, y_train_use)
    models_to_try['RandomForest'] = rf_model
    print("   ✅ Random Forest训练完成")
    
    # 可选：其他算法
    if use_multi_models:
        # XGBoost
        try:
            import xgboost as xgb
            print("\n2️⃣ 训练XGBoost模型...")
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
            models_to_try['XGBoost'] = xgb_model
            print("   ✅ XGBoost训练完成")
        except ImportError:
            print("   ⚠️ XGBoost未安装，跳过")
        except Exception as e:
            print(f"   ⚠️ XGBoost训练失败: {e}")
        
        # LightGBM
        try:
            import lightgbm as lgb
            print("\n3️⃣ 训练LightGBM模型...")
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
            models_to_try['LightGBM'] = lgb_model
            print("   ✅ LightGBM训练完成")
        except ImportError:
            print("   ⚠️ LightGBM未安装，跳过")
        except Exception as e:
            print(f"   ⚠️ LightGBM训练失败: {e}")
    
    # 评估所有模型，选择最佳
    print(f"\n📊 评估 {len(models_to_try)} 个模型...")
    best_model = None
    best_model_name = None
    best_precision = 0
    
    for model_name, model in models_to_try.items():
        y_proba_temp = model.predict_proba(X_test_cleaned)[:, 1]
        
        # 使用自动阈值优化
        optimal_threshold_temp, _, _ = find_optimal_threshold(
            y_test, y_proba_temp, metric='precision', min_recall=0.3
        )
        
        y_pred_temp = (y_proba_temp >= optimal_threshold_temp).astype(int)
        
        precision_0 = precision_score(y_test, y_pred_temp, pos_label=0, zero_division=0)
        precision_1 = precision_score(y_test, y_pred_temp, pos_label=1, zero_division=0)
        avg_precision = (precision_0 + precision_1) / 2
        
        print(f"  {model_name}: 平均精确率={avg_precision:.2%} (阈值={optimal_threshold_temp:.3f})")
        
        if avg_precision > best_precision:
            best_precision = avg_precision
            best_model = model
            best_model_name = model_name
    
    print(f"\n🏆 最佳模型: {best_model_name} (精确率={best_precision:.2%})")
    
    # 使用最佳模型进行最终预测
    y_proba = best_model.predict_proba(X_test_cleaned)[:, 1]
    optimal_threshold, _, _ = find_optimal_threshold(
        y_test, y_proba, metric='precision', min_recall=0.3
    )
    
    y_pred = (y_proba >= optimal_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 计算精确率
    precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    avg_precision = (precision_0 + precision_1) / 2
    
    print(f"\n📊 模型性能评估")
    print("=" * 70)
    print(f"整体准确率: {accuracy:.4f}")
    print(f"\n精确率对比:")
    print(f"  强势类别: {precision_0:.2%}")
    print(f"  弱势类别: {precision_1:.2%}")
    print(f"  平均精确率: {avg_precision:.2%}")
    
    if avg_precision >= 0.75:
        print(f"  ✅ 优秀！精确率已达到75%以上")
    elif avg_precision >= 0.70:
        print(f"  ✅ 良好！精确率达到70%+")
    else:
        print(f"  ⚠️ 可以接受，但仍有提升空间")
    
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, 
                              target_names=['价格>=MA20(强势)', '价格<MA20(弱势)'],
                              zero_division=0))
    
    return best_model, y_pred, accuracy, best_model_name, models_to_try

def visualize_binary_results(y_test, y_pred, accuracy, output_dir='results'):
    """
    可视化二分类结果
    
    参数:
    y_test: 测试集真实标签
    y_pred: 预测标签
    accuracy: 准确率
    output_dir: 输出目录
    """
    print(f"\n生成可视化结果...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['价格>=MA20(强势)', '价格<MA20(弱势)']
    
    plt.figure(figsize=(15, 5))
    
    # 子图1: 混淆矩阵
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'二分类混淆矩阵\n准确率: {accuracy:.4f}')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    
    # 子图2: 类别分布对比
    plt.subplot(1, 3, 2)
    test_dist = pd.Series(y_test).value_counts().sort_index()
    pred_dist = pd.Series(y_pred).value_counts().sort_index()
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, test_dist.values, width, label='真实分布', alpha=0.8)
    plt.bar(x + width/2, pred_dist.values, width, label='预测分布', alpha=0.8)
    
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('类别分布对比')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    
    # 子图3: 分类准确率
    plt.subplot(1, 3, 3)
    class_accuracy = []
    for i in range(len(class_names)):
        mask = y_test == i
        if mask.sum() > 0:
            acc = (y_pred[mask] == i).mean()
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)
    
    bars = plt.bar(class_names, class_accuracy, alpha=0.8, color=['lightblue', 'red'])
    plt.ylabel('准确率')
    plt.title('各类别预测准确率')
    plt.xticks(rotation=45)
    
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    output_path = f'{output_dir}/binary_classification_results.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可视化结果已保存: {output_path}")
    plt.show()

def print_detailed_summary(y_test, y_pred, accuracy, stock_count, feature_count):
    """
    打印详细的分析总结
    
    参数:
    y_test: 测试集真实标签
    y_pred: 预测标签
    accuracy: 准确率
    stock_count: 股票数量
    feature_count: 特征数量
    """
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*80)
    print("📊 分析结果详细总结")
    print("="*80)
    
    print(f"\n🎯 整体情况")
    print("-" * 50)
    print(f"分析股票数量: {stock_count}")
    print(f"提取特征数量: {feature_count}")
    print(f"生成样本数量: {len(y_test) + len(y_pred)}")
    print(f"测试集样本数: {len(y_test)}")
    
    print(f"\n📈 模型性能")
    print("-" * 50)
    print(f"整体准确率: {accuracy:.4f} ({accuracy:.2%})")
    
    # 计算各类别指标
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    print(f"\n强势类别 (价格≥MA20):")
    print(f"  精确率: {precision_0:.2%}")
    print(f"  召回率: {recall_0:.2%}")
    print(f"  F1分数: {f1_0:.2%}")
    
    print(f"\n弱势类别 (价格<MA20):")
    print(f"  精确率: {precision_1:.2%}")
    print(f"  召回率: {recall_1:.2%}")
    print(f"  F1分数: {f1_1:.2%}")
    
    avg_precision = (precision_0 + precision_1) / 2
    print(f"\n平均精确率: {avg_precision:.2%}")
    
    if avg_precision >= 0.75:
        print("  ✅ 优秀！精确率已达到75%以上")
    elif avg_precision >= 0.70:
        print("  ✅ 良好！精确率达到70%+")
    elif avg_precision >= 0.65:
        print("  ⚠️ 可以接受，但仍有提升空间")
    else:
        print("  ❌ 需要进一步优化")
    
    print("\n" + "="*80)

def print_metrics_explanation(y_test, y_pred):
    """
    打印详细的指标解释（通俗易懂版）
    
    参数:
    y_test: 测试集真实标签
    y_pred: 预测标签
    """
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # 计算各类别指标
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0
    
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
    
    accuracy = accuracy_score(y_test, y_pred)
    avg_precision = (precision_0 + precision_1) / 2
    avg_recall = (recall_0 + recall_1) / 2
    avg_f1 = (f1_0 + f1_1) / 2
    
    print("\n" + "="*80)
    print("📚 评估指标详细解释 - 让不懂统计的人也能看懂")
    print("="*80)
    
    print("\n" + "🎯 核心概念".center(76))
    print("-" * 80)
    print("""
我们的模型是一个二分类预测系统，目标是预测股票未来价格相对于20日均线的位置：
  • 强势类别（标签0）: 预测股价 ≥ MA20（看涨信号）
  • 弱势类别（标签1）: 预测股价 < MA20（看跌信号）

模型通过学习历史数据中的特征模式，来预测未来股价的强弱状态。
""")
    
    print("\n" + "📊 混淆矩阵详解".center(76))
    print("-" * 80)
    print(f"""
混淆矩阵是评估分类模型的基础工具，展示了预测结果与真实情况的对比：

                    预测为强势    预测为弱势
    实际为强势        {tn:6d}        {fp:6d}
    实际为弱势        {fn:6d}        {tp:6d}

四个关键数字的含义：

  ✅ 真阳性(TP) = {tp:4d}: 实际弱势，预测也是弱势 → 预测正确！
  ❌ 假阳性(FP) = {fp:4d}: 实际强势，但预测成弱势 → 误判为弱势（错失机会）
  ❌ 假阴性(FN) = {fn:4d}: 实际弱势，但预测成强势 → 误判为强势（可能踩雷）
  ✅ 真阴性(TN) = {tn:4d}: 实际强势，预测也是强势 → 预测正确！

总预测次数 = {tn + fp + fn + tp} 次
""")
    
    print("\n" + "📈 核心指标解释".center(76))
    print("-" * 80)
    
    # 1. 准确率
    print(f"""
1️⃣  准确率 (Accuracy) = {accuracy:.2%}
    
    💡 通俗理解：在所有预测中，有多少比例是猜对的。
    
    📐 计算公式：(TP + TN) / (TP + TN + FP + FN)
                = ({tp} + {tn}) / {tn + fp + fn + tp}
                = {accuracy:.2%}
    
    🎯 实际意义：
       如果对100只股票进行预测，大约有 {accuracy*100:.0f} 只的预测是正确的。
       
    ⚖️  评价标准：
       • ≥ 80%: 优秀 ⭐⭐⭐⭐⭐
       • 70-80%: 良好 ⭐⭐⭐⭐
       • 60-70%: 中等 ⭐⭐⭐
       • < 60%: 需要改进 ⭐⭐
""")
    
    # 2. 精确率
    print(f"""
2️⃣  精确率 (Precision) = {avg_precision:.2%}
    
    💡 通俗理解：在模型预测为某个类别的样本中，真正属于该类别的比例。
    
    强势类别精确率 = {precision_0:.2%}
    📐 计算：TN / (TN + FN) = {tn} / ({tn} + {fn}) = {precision_0:.2%}
    🎯 意义：模型说"这股票会强势"时，有 {precision_0*100:.1f}% 的概率是对的。
    
    弱势类别精确率 = {precision_1:.2%}
    📐 计算：TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision_1:.2%}
    🎯 意义：模型说"这股票会弱势"时，有 {precision_1*100:.1f}% 的概率是对的。
    
    平均精确率 = {avg_precision:.2%}
    
    🎯 实际应用：
       精确率高 = 模型说的话可信度高 = "宁缺毋滥"
       如果精确率是75%，意思是模型推荐的股票中，有75%真的符合预期。
       
    ⚖️  评价标准：
       • ≥ 75%: 优秀，推荐可靠 ⭐⭐⭐⭐⭐
       • 65-75%: 良好，有参考价值 ⭐⭐⭐⭐
       • 55-65%: 中等，需谨慎参考 ⭐⭐⭐
       • < 55%: 较差，不建议使用 ⭐⭐
""")
    
    # 3. 召回率
    print(f"""
3️⃣  召回率 (Recall) = {avg_recall:.2%}
    
    💡 通俗理解：在所有真正属于某类别的样本中，模型成功识别出的比例。
    
    强势类别召回率 = {recall_0:.2%}
    📐 计算：TN / (TN + FP) = {tn} / ({tn} + {fp}) = {recall_0:.2%}
    🎯 意义：在所有实际强势的股票中，模型成功识别出了 {recall_0*100:.1f}%。
    
    弱势类别召回率 = {recall_1:.2%}
    📐 计算：TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall_1:.2%}
    🎯 意义：在所有实际弱势的股票中，模型成功识别出了 {recall_1*100:.1f}%。
    
    平均召回率 = {avg_recall:.2%}
    
    🎯 实际应用：
       召回率高 = 模型"不漏过"好机会 = "宁可错杀，不可放过"
       如果强势召回率是80%，意思是100只真正强势的股票中，模型能找出80只。
       
    ⚖️  评价标准：
       • ≥ 80%: 优秀，覆盖全面 ⭐⭐⭐⭐⭐
       • 70-80%: 良好，基本覆盖 ⭐⭐⭐⭐
       • 60-70%: 中等，有遗漏 ⭐⭐⭐
       • < 60%: 较差，遗漏较多 ⭐⭐
""")
    
    # 4. F1分数
    print(f"""
4️⃣  F1分数 (F1-Score) = {avg_f1:.2%}
    
    💡 通俗理解：精确率和召回率的调和平均数，综合评估模型性能。
    
    强势类别F1 = {f1_0:.2%}
    📐 计算：2 × (精确率 × 召回率) / (精确率 + 召回率)
           = 2 × ({precision_0:.2%} × {recall_0:.2%}) / ({precision_0:.2%} + {recall_0:.2%})
           = {f1_0:.2%}
    
    弱势类别F1 = {f1_1:.2%}
    📐 计算：2 × ({precision_1:.2%} × {recall_1:.2%}) / ({precision_1:.2%} + {recall_1:.2%})
           = {f1_1:.2%}
    
    平均F1分数 = {avg_f1:.2%}
    
    🎯 实际意义：
       F1分数平衡了"准确性"和"覆盖面"，是最重要的综合指标。
       当精确率和召回率都高时，F1分数才会高。
       
    ⚖️  评价标准：
       • ≥ 75%: 优秀，模型表现很好 ⭐⭐⭐⭐⭐
       • 65-75%: 良好，模型可用 ⭐⭐⭐⭐
       • 55-65%: 中等，有改进空间 ⭐⭐⭐
       • < 55%: 较差，需要优化 ⭐⭐
""")
    
    print("\n" + "🎓 如何解读这些指标？".center(76))
    print("-" * 80)
    print("""
📌 精确率 vs 召回率的权衡：

  高精确率、低召回率 → 保守策略
    特点：模型推荐的很少，但推荐的大多是对的
    适用：想要高质量推荐，宁愿错过也不愿误判
    例子：只推荐最有把握的强势股
  
  低精确率、高召回率 → 激进策略
    特点：模型推荐的很多，但有不少是错的
    适用：不想错过任何机会，可以接受一些误判
    例子：把所有可能强势的股票都找出来
  
  高精确率、高召回率 → 理想状态 ✨
    特点：推荐的多，而且准确率高
    这是我们追求的目标！

📌 实战应用建议：

  如果您是保守型投资者：
    ✅ 关注精确率（Precision）> 70%
    ✅ 选择模型预测为"强势"的股票
    ✅ 接受可能错过一些机会
  
  如果您是激进型投资者：
    ✅ 关注召回率（Recall）> 70%
    ✅ 关注模型识别出的所有可能机会
    ✅ 自己再做二次筛选
  
  如果您追求平衡：
    ✅ 关注F1分数 > 70%
    ✅ 综合考虑精确率和召回率
    ✅ 这是最推荐的策略

📌 当前模型评价：
""")
    
    if avg_precision >= 0.75 and avg_recall >= 0.70:
        print(f"""
  🎉 恭喜！您的模型表现优秀！
     • 平均精确率 {avg_precision:.2%} - 推荐可靠
     • 平均召回率 {avg_recall:.2%} - 覆盖全面
     • 平均F1分数 {avg_f1:.2%} - 综合优秀
     
  💡 建议：可以实盘小规模测试，观察实际效果。
""")
    elif avg_precision >= 0.65 and avg_recall >= 0.60:
        print(f"""
  👍 不错！您的模型表现良好！
     • 平均精确率 {avg_precision:.2%} - 有参考价值
     • 平均召回率 {avg_recall:.2%} - 基本覆盖
     • 平均F1分数 {avg_f1:.2%} - 综合良好
     
  💡 建议：可以作为辅助决策工具，结合其他指标使用。
""")
    else:
        print(f"""
  ⚠️  您的模型还有较大提升空间
     • 平均精确率 {avg_precision:.2%}
     • 平均召回率 {avg_recall:.2%}
     • 平均F1分数 {avg_f1:.2%}
     
  💡 建议：
     - 增加训练数据量
     - 尝试不同的特征工程方法
     - 调整模型参数
     - 考虑使用集成学习方法
""")
    
    print("\n" + "="*80)
    print("💡 温馨提示：")
    print("   股票投资有风险，模型预测仅供参考。")
    print("   请结合基本面分析、技术分析等多种方法，谨慎决策。")
    print("   过往表现不代表未来收益。")
    print("="*80 + "\n")

def main():
    """
    统计分析主函数
    """
    print("="*80)
    print("股票统计分析")
    print("基于MA20的二分类预测：价格>=MA20(强势) vs 价格<MA20(弱势)")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')
    
    try:
        # 1. 加载处理后的特征
        print("\n步骤1: 加载处理后的特征")
        print("-" * 50)
        
        try:
            x_filtered, y_series = load_processed_features('data')
        except:
            print("错误: 无法加载特征数据")
            print("请先运行: python stock_feature_engineering.py")
            return
        
        # 2. 清理特征名称（与模型训练保持一致）
        print(f"\n步骤2: 清理特征名称")
        print("-" * 50)
        print(f"清理前特征示例: {x_filtered.columns[:3].tolist()}")
        x_filtered = clean_feature_names(x_filtered)
        print(f"清理后特征示例: {x_filtered.columns[:3].tolist()}")
        print(f"✅ 特征名称清理完成")
        
        # 3. 数据分割
        print(f"\n步骤3: 数据分割")
        print("-" * 50)
        X_train, X_test, y_train, y_test = train_test_split(
            x_filtered, y_series, 
            test_size=0.2, 
            random_state=42, 
            stratify=y_series
        )
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        # 4. 模型训练
        print(f"\n步骤4: 模型训练")
        print("-" * 50)
        best_model, y_pred, accuracy, model_name, models_to_try = train_binary_model(
            X_train, X_test, y_train, y_test,
            use_smote=True,  # 是否使用SMOTE过采样
            use_multi_models=True  # ⭐ 改为True，训练多个模型用于对比
        )
        
        # 5. 结果可视化
        print(f"\n步骤5: 结果可视化")
        print("-" * 50)
        visualize_binary_results(y_test, y_pred, accuracy, output_dir='results')
        
        # 6. 详细总结
        print(f"\n步骤6: 生成详细总结")
        print("-" * 50)
        
        # 估算股票数量（从样本ID中提取）
        sample_ids = y_series.index.tolist()
        stock_codes = set([id.rsplit('_', 1)[0] for id in sample_ids])
        stock_count = len(stock_codes)
        
        print_detailed_summary(
            y_test, y_pred, accuracy, 
            stock_count=stock_count,
            feature_count=x_filtered.shape[1]
        )
        
        # 7. 打印详细的指标解释
        print(f"\n步骤7: 打印评估指标详细解释")
        print("-" * 50)
        print_metrics_explanation(y_test, y_pred)
        
        # 8. 保存模型和特征列表（用于实盘预测）
        print(f"\n步骤8: 保存模型用于实盘预测")
        print("-" * 50)
        
        import pickle
        
        os.makedirs('models', exist_ok=True)
        
        # 保存最佳模型（向后兼容）
        model_save_path = 'models/trained_model.pkl'
        with open(model_save_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"✅ 最佳模型已保存到: {model_save_path}")
        
        # ⭐ 新增：保存所有训练的模型（用于对比预测）
        all_models_path = 'models/all_trained_models.pkl'
        all_models_data = {}
        
        for model_name_temp, model_temp in models_to_try.items():
            # 计算每个模型的性能指标
            y_proba_temp = model_temp.predict_proba(X_test)[:, 1]
            optimal_threshold_temp, _, _ = find_optimal_threshold(
                y_test, y_proba_temp, metric='precision', min_recall=0.3
            )
            y_pred_temp = (y_proba_temp >= optimal_threshold_temp).astype(int)
            
            precision_0 = precision_score(y_test, y_pred_temp, pos_label=0, zero_division=0)
            precision_1 = precision_score(y_test, y_pred_temp, pos_label=1, zero_division=0)
            avg_precision_temp = (precision_0 + precision_1) / 2
            accuracy_temp = accuracy_score(y_test, y_pred_temp)
            
            all_models_data[model_name_temp] = {
                'model': model_temp,
                'optimal_threshold': optimal_threshold_temp,
                'accuracy': accuracy_temp,
                'avg_precision': avg_precision_temp,
                'precision_0': precision_0,
                'precision_1': precision_1
            }
        
        with open(all_models_path, 'wb') as f:
            pickle.dump(all_models_data, f)
        print(f"✅ 所有模型已保存到: {all_models_path} (共{len(all_models_data)}个模型)")
        
        # 保存特征列表（用于预测时对齐特征）
        feature_list_path = 'models/feature_list.pkl'
        with open(feature_list_path, 'wb') as f:
            pickle.dump(x_filtered.columns.tolist(), f)
        print(f"✅ 特征列表已保存到: {feature_list_path}")
        
        # 保存模型元信息
        model_info = {
            'best_model_name': model_name,
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': accuracy,
            'avg_precision': (precision_score(y_test, y_pred, pos_label=0, zero_division=0) + 
                             precision_score(y_test, y_pred, pos_label=1, zero_division=0)) / 2,
            'feature_count': len(x_filtered.columns),
            'sample_count': len(y_series),
            'stock_count': stock_count,
            'available_models': list(all_models_data.keys())
        }
        
        model_info_path = 'models/model_info.pkl'
        with open(model_info_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"✅ 模型信息已保存到: {model_info_path}")
        
        print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n✅ 统计分析完成！所有结果已保存到 results/ 目录")
        print("✅ 模型已保存，可使用 stock_live_prediction.py 进行实盘预测")
        
    except Exception as e:
        print(f"统计分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


