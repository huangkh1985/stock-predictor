@echo off
echo ======================================================================
echo   启动Streamlit股票预测系统
echo ======================================================================
echo.
echo [信息] 正在启动Streamlit...
echo.
echo [访问地址]
echo   本地: http://localhost:8501
echo   网络: http://你的IP地址:8501
echo.
echo [提示] 按 Ctrl+C 可以停止服务
echo ======================================================================
echo.

streamlit run stock_app_streamlit.py --server.port=8501

pause

