# 股票预测统一分析模块

## 📋 概述

`stock_analysis_unified.py` 是一个合并了所有功能的单文件模块，专为云端部署（如Streamlit Cloud）优化。

### 🔗 合并的模块
1. ✅ `stock_data_downloader.py` - 数据下载
2. ✅ `stock_feature_engineering.py` - 特征工程
3. ✅ `stock_statistical_analysis.py` - 模型训练
4. ✅ `stock_live_prediction_APP.py` - 实时预测

### 🚀 主要特性

- ✅ **无文件I/O** - 所有数据处理在内存中完成
- ✅ **云端友好** - 适配Streamlit Cloud只读文件系统
- ✅ **efinance修复** - 自动修补权限问题
- ✅ **完整流程** - 从数据下载到模型预测

## 📖 使用方法

### 1. 训练模型

```python
from stock_analysis_unified import train_stock_prediction_model

# 训练模型
best_model, all_models_data, feature_list = train_stock_prediction_model(
    stock_codes=['600519', '000001', '600036'],  # 股票列表
    window_size=20,                              # 窗口大小
    forecast_horizon=5,                          # 预测期限
    use_multi_models=True                        # 是否使用多模型
)
```

### 2. 预测股票

```python
from stock_analysis_unified import predict_stocks_inline

# 预测
results = predict_stocks_inline(
    stock_codes=['600519', '000002'],  # 要预测的股票
    model=best_model,                  # 训练好的模型
    all_models_data=all_models_data,   # 所有模型数据
    feature_list=feature_list,         # 特征列表
    window_size=20                     # 窗口大小
)

# 查看结果
for result in results:
    print(f"股票: {result['stock_code']}")
    if result['type'] == 'multi':
        for model_name, pred in result['predictions'].items():
            label = "强势" if pred['prediction'] == 0 else "弱势"
            print(f"  {model_name}: {label} (置信度={pred['confidence']:.1%})")
```

### 3. 在Streamlit中使用

```python
import streamlit as st
from stock_analysis_unified import train_stock_prediction_model, predict_stocks_inline

# 缓存模型训练
@st.cache_resource
def load_model():
    stock_codes = ['600519', '000001', '600036']
    return train_stock_prediction_model(stock_codes)

# 加载模型
best_model, all_models_data, feature_list = load_model()

# 用户输入
stock_code = st.text_input("输入股票代码", "600519")

if st.button("预测"):
    with st.spinner("预测中..."):
        results = predict_stocks_inline(
            [stock_code], best_model, all_models_data, feature_list
        )
        
        if results:
            result = results[0]
            st.success(f"预测完成！")
            # 显示预测结果...
```

## 🔧 核心函数

### `train_stock_prediction_model()`
完整训练流程

**参数：**
- `stock_codes`: 股票代码列表
- `window_size`: 滑动窗口大小（默认20天）
- `forecast_horizon`: 预测期限（默认5天）
- `use_multi_models`: 是否训练多个模型（默认True）

**返回：**
- `best_model`: 最佳模型
- `all_models_data`: 所有模型的字典
- `feature_list`: 特征名称列表

### `predict_stocks_inline()`
批量预测股票

**参数：**
- `stock_codes`: 要预测的股票列表
- `model`: 训练好的模型
- `all_models_data`: 所有模型数据
- `feature_list`: 特征列表
- `window_size`: 窗口大小

**返回：**
- `results`: 预测结果列表

## ⚠️ 重要说明

### 已移除的功能
- ❌ 所有文件保存操作（pickle.dump, to_csv等）
- ❌ 文件读取操作（pickle.load等）
- ❌ 目录创建操作（os.makedirs等）
- ❌ 图表保存操作（plt.savefig等）

### 保留的功能
- ✅ 数据下载（efinance）
- ✅ 特征提取（TSFresh）
- ✅ 模型训练（sklearn, xgboost, lightgbm）
- ✅ 模型预测
- ✅ 所有内存数据处理

### 依赖项
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
tsfresh>=0.19.0
efinance>=0.3.0
lightgbm>=3.3.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
```

## 💡 云端部署建议

### 1. 数据持久化
由于所有数据在内存中，建议：
- 使用Streamlit的`@st.cache_resource`缓存模型
- 定期重新训练模型
- 或者在本地训练后使用其他方式上传模型

### 2. 性能优化
- 限制股票数量（建议≤20只）
- 使用较小的窗口大小
- 选择精简特征集

### 3. 错误处理
模块已内置错误处理：
- efinance不可用时自动降级
- 特征提取失败时有提示
- 预测失败时返回None

## 🎯 预测结果格式

### 单模型预测
```python
{
    'stock_code': '600519',
    'stock_data': DataFrame,  # 股票数据
    'prediction': 0,          # 0=强势, 1=弱势
    'probability': [0.7, 0.3],
    'type': 'single'
}
```

### 多模型预测
```python
{
    'stock_code': '600519',
    'stock_data': DataFrame,
    'predictions': {
        'RandomForest': {
            'prediction': 0,
            'probability': [0.7, 0.3],
            'confidence': 0.7
        },
        'XGBoost': {...},
        'LightGBM': {...}
    },
    'type': 'multi'
}
```

## 📞 技术支持

如有问题，请参考：
- 原始模块文档
- Streamlit Cloud部署指南（`STREAMLIT_CLOUD_DEPLOYMENT.md`）

---

**版本**: 1.0  
**最后更新**: 2025-10-04  
**状态**: ✅ 已测试

