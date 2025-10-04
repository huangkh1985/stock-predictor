# Streamlit Cloud 部署指南

## 📋 问题说明

### PermissionError 错误原因
在Streamlit Cloud上遇到的`PermissionError`是因为：
1. **Streamlit Cloud使用只读文件系统** - 大部分目录不可写
2. **efinance库在导入时创建缓存目录** - 试图写入系统目录导致权限错误

### 解决方案
已在代码中实现以下修复：

#### 1. 修补efinance配置 (`stock_data_downloader.py`)
```python
# 检测Streamlit Cloud环境
if os.path.exists('/mount/src'):
    # 使用临时目录
    temp_dir = tempfile.gettempdir()
    cache_dir = os.path.join(temp_dir, 'efinance_cache')
    
    # 修补efinance配置模块
    mock_config = types.ModuleType('efinance.config')
    mock_config.DATA_DIR = Path(cache_dir)
    sys.modules['efinance.config'] = mock_config
```

#### 2. 优雅降级处理
如果efinance仍然无法导入，应用会：
- 在离线模式下运行
- 使用已上传的预训练模型
- 显示友好的错误提示

## 🚀 部署步骤

### 1. 准备文件
确保以下文件已提交到GitHub：
- ✅ `stock_app_streamlit.py` - Streamlit应用
- ✅ `stock_live_prediction_APP.py` - 预测模块
- ✅ `stock_data_downloader.py` - 数据下载模块（已修复）
- ✅ `requirements.txt` - 依赖列表
- ✅ `models/*.pkl` - 预训练模型文件
- ✅ `.streamlit/config.toml` - Streamlit配置

### 2. 推送到GitHub
```bash
git add .
git commit -m "Fix: Streamlit Cloud permission error"
git push origin main
```

### 3. 部署到Streamlit Cloud

#### 访问Streamlit Cloud
1. 打开 https://share.streamlit.io
2. 使用GitHub账号登录

#### 创建新应用
1. 点击 "New app"
2. 选择你的GitHub仓库：`stock-predictor`
3. 选择分支：`main`
4. 主文件路径：`stock_app_streamlit.py`
5. 点击 "Deploy!"

#### 等待部署完成
- 首次部署需要5-10分钟
- Streamlit会自动安装`requirements.txt`中的依赖
- 部署成功后会自动打开应用URL

## ⚙️ 配置说明

### requirements.txt
确保包含以下依赖：
```txt
streamlit>=1.20.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
xgboost>=1.5.0
joblib>=1.0.0
python-dateutil>=2.8.0
efinance>=0.3.0
```

### 模型文件大小限制
- Streamlit Cloud对文件大小有限制（通常100MB）
- 如果模型文件过大，考虑：
  1. 使用Git LFS（Large File Storage）
  2. 压缩模型文件
  3. 从外部存储加载模型（如AWS S3）

## 🔍 常见问题

### Q1: 应用启动时显示"efinance导入失败"
**A:** 这是正常的！在Streamlit Cloud上：
- efinance可能因为权限问题无法完全初始化
- 应用会自动切换到离线模式
- 使用预上传的模型文件进行预测
- **不影响核心预测功能**

### Q2: 如何在云端使用实时数据？
**A:** Streamlit Cloud限制：
- 无法直接使用efinance下载实时数据
- **解决方案**：
  1. 本地下载数据并训练模型
  2. 上传模型文件到GitHub
  3. 云端应用使用预训练模型

### Q3: 模型文件太大无法上传
**A:** 使用Git LFS：
```bash
# 安装Git LFS
git lfs install

# 跟踪大文件
git lfs track "models/*.pkl"
git add .gitattributes
git add models/
git commit -m "Add models with Git LFS"
git push
```

### Q4: 部署失败怎么办？
**A:** 检查以下内容：
1. 查看Streamlit Cloud的日志（点击"Manage app" → "Logs"）
2. 确认`requirements.txt`中的依赖版本兼容
3. 确认模型文件已正确上传
4. 检查Python版本（Streamlit Cloud使用Python 3.9-3.11）

## 📱 访问应用

### 获取应用URL
部署成功后，Streamlit Cloud会提供：
- 公开URL: `https://your-app-name.streamlit.app`
- 可以分享给任何人访问

### 手机访问
- 直接在手机浏览器打开应用URL
- 支持响应式布局
- 可以添加到主屏幕作为Web App

## 🔧 维护和更新

### 更新应用
1. 在本地修改代码
2. 提交到GitHub
3. Streamlit Cloud会自动检测更改并重新部署

### 查看日志
在Streamlit Cloud控制台：
- 点击应用名称
- 选择"Manage app"
- 查看"Logs"标签

### 重启应用
如果应用出现问题：
1. 进入"Manage app"
2. 点击"Reboot app"

## 💡 最佳实践

### 1. 使用缓存
```python
@st.cache_resource
def load_model():
    return pickle.load(open('models/trained_model.pkl', 'rb'))
```

### 2. 错误处理
```python
try:
    result = predict_stock(stock_code)
except Exception as e:
    st.error(f"预测失败: {e}")
```

### 3. 显示进度
```python
with st.spinner('正在预测...'):
    result = predict_stock(stock_code)
st.success('预测完成！')
```

## 🎯 性能优化

### 减少应用加载时间
1. 使用`@st.cache_resource`缓存模型
2. 延迟导入大型库
3. 优化数据处理逻辑

### 减少内存使用
1. 及时释放不需要的DataFrame
2. 使用较小的模型文件
3. 限制批量处理的数据量

## 📞 获取帮助

遇到问题？
1. 查看Streamlit官方文档: https://docs.streamlit.io
2. Streamlit社区论坛: https://discuss.streamlit.io
3. 检查本项目的GitHub Issues

---

**更新日期**: 2025-10-04
**版本**: 1.0

