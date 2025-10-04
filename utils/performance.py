# utils/performance.py
import torch
import gc
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional

def clear_gpu_memory():
    """清理GPU内存"""
    try:
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"清理GPU内存时发生错误: {e}")

def calculate_directional_accuracy(y_true, y_pred, last_n_days=None):
    """计算价格变动方向的预测准确率"""
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return 0.0, 0, 0 

    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    true_diff = np.diff(y_true_flat)
    pred_diff = np.diff(y_pred_flat)

    true_direction = np.sign(true_diff)
    pred_direction = np.sign(pred_diff)
    
    if last_n_days is not None and last_n_days > 0 and last_n_days <= len(true_direction):
        true_direction = true_direction[-last_n_days:]
        pred_direction = pred_direction[-last_n_days:]
    
    if len(true_direction) == 0:
        return 0.0, 0, 0

    valid_comparisons_mask = true_direction != 0
    if np.sum(valid_comparisons_mask) == 0:
        return 0.0, 0, 0

    correct_predictions_meaningful = np.sum(true_direction[valid_comparisons_mask] == pred_direction[valid_comparisons_mask])
    total_meaningful_comparisons = np.sum(valid_comparisons_mask)

    accuracy = (correct_predictions_meaningful / total_meaningful_comparisons) * 100 if total_meaningful_comparisons > 0 else 0.0
    
    return accuracy, correct_predictions_meaningful, total_meaningful_comparisons

def plot_training_history(history, save_path=None):
    """绘制训练历史"""
    if not history or 'train_loss' not in history or 'val_loss' not in history:
        return

    plt.figure(figsize=(15, 6))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制学习率
    if 'lr' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['lr'], label='学习率')
        plt.title('学习率变化')
        plt.xlabel('轮数')
        plt.ylabel('学习率')
        plt.grid(True, alpha=0.3)
        if any(lr > 0 for lr in history['lr']):
            try:
                plt.yscale('log')
            except ValueError:
                pass
        plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception:
            pass
    
    try:
        plt.show()
        plt.close()
    except Exception:
        pass

def handle_outliers(df, columns, method='zscore', threshold=3.0):
    """处理数据中的异常值"""
    df_clean = df.copy()
    
    for column in columns:
        if column not in df.columns:
            continue
            
        if method == 'zscore':
            # 使用Z分数方法
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            mask = z_scores > threshold
            
            # 使用中位数替换异常值
            median_val = df[column].median()
            df_clean.loc[mask, column] = median_val
            
        elif method == 'percentile':
            # 使用百分位数方法
            lower_bound = df[column].quantile(threshold / 100)
            upper_bound = df[column].quantile(1 - threshold / 100)
            
            # 使用边界值替换异常值
            mask_lower = df[column] < lower_bound
            mask_upper = df[column] > upper_bound
            
            df_clean.loc[mask_lower, column] = lower_bound
            df_clean.loc[mask_upper, column] = upper_bound
            
    return df_clean