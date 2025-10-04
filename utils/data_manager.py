# utils/data_manager.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple, List, Union, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import shap
import joblib
import os
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

class ProgressiveDataLoader:
    """渐进式数据加载器，减少内存占用"""
    
    def __init__(self, df, target_col, time_steps, scaler_type='minmax', 
                batch_size=32, cache_dir='./data_cache'):
        self.df = df
        self.target_col = target_col
        self.time_steps = time_steps
        self.scaler_type = scaler_type
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 初始化缩放器
        self._init_scalers()
        
        # 特征列
        self.feature_cols = [col for col in df.columns if col != target_col]
        
        # 数据长度
        self.data_length = len(df) - time_steps
        
    def _init_scalers(self):
        """初始化缩放器"""
        if self.scaler_type == 'minmax':
            self.feature_scaler = MinMaxScaler()
            self.target_scaler = MinMaxScaler()
        elif self.scaler_type == 'standard':
            self.feature_scaler = StandardScaler()
            self.target_scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.feature_scaler = RobustScaler()
            self.target_scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的缩放器类型: {self.scaler_type}")
    
    def prepare_data(self, train_split=0.7, val_split=0.1):
        """准备数据集划分"""
        # 计算划分点
        train_size = int(self.data_length * train_split)
        val_size = int(self.data_length * val_split)
        
        # 拟合缩放器（仅使用训练数据）
        X_train_fit = self.df[self.feature_cols].iloc[:train_size].values
        y_train_fit = self.df[self.target_col].iloc[:train_size].values.reshape(-1, 1)
        
        self.feature_scaler.fit(X_train_fit)
        self.target_scaler.fit(y_train_fit)
        
        # 保存缩放器
        joblib.dump(self.feature_scaler, f"{self.cache_dir}/feature_scaler.pkl")
        joblib.dump(self.target_scaler, f"{self.cache_dir}/target_scaler.pkl")
        
        # 返回划分信息
        return {
            'train_indices': (0, train_size),
            'val_indices': (train_size, train_size + val_size),
            'test_indices': (train_size + val_size, self.data_length)
        }
    
    def get_dataloader(self, split='train', shuffle=True):
        """获取指定分割的数据加载器"""
        splits = self.prepare_data()
        start_idx, end_idx = splits[f'{split}_indices']
        
        # 创建缓存文件名
        cache_file = f"{self.cache_dir}/{split}_{start_idx}_{end_idx}.pt"
        
        # 如果缓存存在，直接加载
        if os.path.exists(cache_file):
            data = torch.load(cache_file)
            X, y = data['X'], data['y']
        else:
            # 创建序列数据
            X, y = [], []
            
            for i in range(start_idx, end_idx):
                # 获取输入序列
                X_seq = self.df[self.feature_cols].iloc[i:i+self.time_steps].values
                X_seq = self.feature_scaler.transform(X_seq)
                
                # 获取目标值
                y_val = self.df[self.target_col].iloc[i+self.time_steps]
                y_val = self.target_scaler.transform([[y_val]])[0]
                
                X.append(X_seq)
                y.append(y_val)
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            # 保存缓存
            torch.save({'X': torch.tensor(X), 'y': torch.tensor(y)}, cache_file)
            
            X = torch.tensor(X)
            y = torch.tensor(y)
        
        # 创建数据集和数据加载器
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle,
            pin_memory=True,
            prefetch_factor=2,  # 预加载优化
            num_workers=2,      # 使用多进程加载数据
            persistent_workers=True  # 保持worker进程存活
        )
    
    def get_test_data(self):
        """获取测试数据"""
        splits = self.prepare_data()
        start_idx, end_idx = splits['test_indices']
        
        # 创建缓存文件名
        cache_file = f"{self.cache_dir}/test_{start_idx}_{end_idx}.pt"
        
        # 如果缓存存在，直接加载
        if os.path.exists(cache_file):
            data = torch.load(cache_file)
            X, y = data['X'], data['y']
            return X, y
        
        # 创建序列数据
        X, y = [], []
        dates = []
        
        for i in range(start_idx, end_idx):
            # 获取输入序列
            X_seq = self.df[self.feature_cols].iloc[i:i+self.time_steps].values
            X_seq = self.feature_scaler.transform(X_seq)
            
            # 获取目标值
            y_val = self.df[self.target_col].iloc[i+self.time_steps]
            y_val = self.target_scaler.transform([[y_val]])[0]
            
            # 获取日期
            if isinstance(self.df.index, pd.DatetimeIndex):
                dates.append(self.df.index[i+self.time_steps])
            
            X.append(X_seq)
            y.append(y_val)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # 保存缓存
        torch.save({'X': torch.tensor(X), 'y': torch.tensor(y)}, cache_file)
        
        return torch.tensor(X), torch.tensor(y), dates

def select_features_with_shap(model, X, feature_names, top_n=20):
    """使用SHAP值选择最重要的特征"""
    # 创建SHAP解释器
    explainer = shap.DeepExplainer(model, X[:50].to(model.device))
    shap_values = explainer.shap_values(X[:100].to(model.device))
    
    # 计算每个特征的平均绝对SHAP值
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    
    # 处理多维特征
    if len(feature_importance.shape) > 2:
        feature_importance = np.mean(feature_importance, axis=1)
    
    # 获取特征重要性排名
    importance_dict = {}
    for i, name in enumerate(feature_names):
        importance_dict[name] = np.mean(feature_importance[:, i])
    
    # 按重要性排序
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # 返回前N个重要特征
    return [f[0] for f in sorted_features[:top_n]], dict(sorted_features[:top_n])