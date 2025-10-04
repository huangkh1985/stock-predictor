@echo off
echo ======================================================================
echo   修复Streamlit Cloud权限错误并部署
echo ======================================================================
echo.
echo [信息] 此脚本将：
echo   1. 添加修复后的文件
echo   2. 提交更改
echo   3. 推送到GitHub
echo   4. Streamlit Cloud会自动重新部署
echo.
echo ======================================================================
pause

echo.
echo [1/3] 添加修改的文件...
git add stock_data_downloader.py
git add stock_live_prediction_APP.py
git add .streamlit/config.toml
git add STREAMLIT_CLOUD_DEPLOYMENT.md
git add deploy_fix.bat

echo.
echo [2/3] 提交更改...
git commit -m "Fix: Streamlit Cloud PermissionError - efinance导入问题修复

- 修复efinance在Streamlit Cloud上的权限错误
- 添加环境检测和配置修补
- 实现优雅降级到离线模式
- 添加Streamlit Cloud部署文档
"

echo.
echo [3/3] 推送到GitHub...
git push origin main

echo.
echo ======================================================================
echo [完成] 修复已推送到GitHub!
echo ======================================================================
echo.
echo [下一步]
echo   1. 访问 https://share.streamlit.io
echo   2. 查看你的应用（Streamlit会自动重新部署）
echo   3. 等待2-3分钟让部署完成
echo   4. 刷新应用页面查看修复结果
echo.
echo [注意]
echo   - efinance可能仍无法下载实时数据
echo   - 应用会使用预训练模型（离线模式）
echo   - 这不影响核心预测功能
echo.
pause

