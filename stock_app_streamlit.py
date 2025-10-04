"""
股票预测系统 - Streamlit Web版本
最简单的Android部署方案：Web应用 + PWA/Capacitor封装

运行方式：
streamlit run stock_app_streamlit.py

特点：
- 完全兼容所有Python库（pandas, numpy, sklearn, tsfresh等）
- 无需Linux环境，Windows即可开发
- 可以部署到云端，手机浏览器直接访问
- 可选：用Capacitor封装成原生APP
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 页面配置
st.set_page_config(
    page_title="股票预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.success-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
}
.warning-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
}
.error-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
}
</style>
""", unsafe_allow_html=True)

# 导入预测模块
try:
    from stock_live_prediction_APP import (
        load_trained_model,
        load_all_models,
        load_feature_list,
        load_model_info,
        download_stock_for_prediction,
        extract_features_for_prediction,
        align_features,
        predict_stock,
        predict_with_all_models
    )
    PREDICTION_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ 预测模块导入失败: {e}")
    st.info("请确保所有依赖已正确安装")
    PREDICTION_AVAILABLE = False


def main():
    """主函数"""
    
    # 标题
    st.markdown('<p class="big-font">📊 股票预测系统</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # 预测模式
        use_multi_models = st.checkbox(
            "使用多模型对比",
            value=True,
            help="使用多个模型进行预测并对比结果"
        )
        
        # 窗口大小
        window_size = st.slider(
            "特征窗口大小（天）",
            min_value=10,
            max_value=60,
            value=20,
            step=5,
            help="用于特征提取的历史数据天数"
        )
        
        # 数据天数
        data_days = st.slider(
            "下载数据天数",
            min_value=180,
            max_value=730,
            value=365,
            step=30,
            help="下载多少天的历史数据"
        )
        
        st.markdown("---")
        
        # 模型信息
        st.header("📈 模型信息")
        if os.path.exists('models/model_info.pkl'):
            model_info = load_model_info()
            if model_info:
                st.info(f"**模型类型**: {model_info.get('model_name', 'Unknown')}")
                st.info(f"**训练时间**: {model_info.get('train_date', 'Unknown')}")
                st.info(f"**准确率**: {model_info.get('accuracy', 0):.2%}")
        else:
            st.warning("模型文件未找到")
        
        st.markdown("---")
        
        # 关于
        st.header("ℹ️ 关于")
        st.info("""
        **股票预测系统 v1.0**
        
        基于机器学习的股票走势预测
        
        - 预测目标：未来5日相对MA20的位置
        - 模型：集成学习算法
        - 特征：TSFresh时间序列特征
        """)
    
    # 主内容区
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 股票预测")
        
        # 股票代码输入
        stock_code = st.text_input(
            "请输入股票代码（6位数字）",
            max_chars=6,
            placeholder="例如：600519",
            help="输入A股6位数字代码"
        )
    
    with col2:
        st.header("🎯 快速选择")
        
        # 快速选择按钮
        if st.button("贵州茅台 (600519)", use_container_width=True):
            stock_code = "600519"
            st.session_state['stock_code'] = stock_code
        
        if st.button("平安银行 (000001)", use_container_width=True):
            stock_code = "000001"
            st.session_state['stock_code'] = stock_code
        
        if st.button("招商银行 (600036)", use_container_width=True):
            stock_code = "600036"
            st.session_state['stock_code'] = stock_code
    
    # 使用session_state保持股票代码
    if 'stock_code' in st.session_state:
        stock_code = st.session_state['stock_code']
    
    st.markdown("---")
    
    # 预测按钮
    if st.button("🔮 开始预测", type="primary", use_container_width=True):
        if not PREDICTION_AVAILABLE:
            st.error("❌ 预测功能不可用，请检查依赖安装")
            return
        
        if not stock_code or len(stock_code) != 6 or not stock_code.isdigit():
            st.error("❌ 请输入有效的6位数字股票代码")
            return
        
        # 执行预测
        with st.spinner('📥 正在下载数据...'):
            predict_stock_streamlit(
                stock_code,
                use_multi_models,
                window_size,
                data_days
            )


def predict_stock_streamlit(stock_code, use_multi_models, window_size, data_days):
    """执行预测并显示结果"""
    
    try:
        # 进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1. 加载模型
        status_text.text("⏳ 加载模型...")
        progress_bar.progress(10)
        
        if use_multi_models:
            all_models = load_all_models()
            if all_models is None:
                model = load_trained_model()
                use_multi_models = False
            else:
                model = None
        else:
            model = load_trained_model()
            all_models = None
        
        feature_list = load_feature_list()
        model_info = load_model_info()
        
        if (model is None and all_models is None) or feature_list is None:
            st.error("❌ 模型文件加载失败")
            return
        
        # 2. 下载数据
        status_text.text(f"📥 下载股票 {stock_code} 数据...")
        progress_bar.progress(30)
        
        stock_data = download_stock_for_prediction(stock_code, days=data_days)
        if stock_data is None:
            st.error(f"❌ 无法下载股票 {stock_code} 的数据")
            return
        
        # 3. 提取特征
        status_text.text("🔧 提取时间序列特征...")
        progress_bar.progress(50)
        
        features_df = extract_features_for_prediction(stock_data, window_size=window_size)
        if features_df is None:
            st.error("❌ 特征提取失败")
            return
        
        # 4. 对齐特征
        status_text.text("🔄 对齐特征...")
        progress_bar.progress(70)
        
        aligned_features = align_features(features_df, feature_list)
        
        # 5. 预测
        status_text.text("🤖 执行预测...")
        progress_bar.progress(90)
        
        if use_multi_models and all_models:
            predictions_dict = predict_with_all_models(all_models, aligned_features)
            display_multi_model_result(stock_code, stock_data, predictions_dict, model_info)
        else:
            prediction, probability = predict_stock(model, aligned_features)
            display_single_result(stock_code, stock_data, prediction, probability, model_info)
        
        # 完成
        progress_bar.progress(100)
        status_text.text("✅ 预测完成！")
        
        # 清除进度信息
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        st.error(f"❌ 预测失败：{str(e)}")
        import traceback
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())


def display_single_result(stock_code, stock_data, prediction, probability, model_info):
    """显示单模型预测结果"""
    
    latest = stock_data.iloc[-1]
    latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
    
    # 计算MA20
    if 'MA_20' in stock_data.columns:
        ma20 = latest['MA_20']
    elif 'SMA_20' in stock_data.columns:
        ma20 = latest['SMA_20']
    else:
        ma20 = stock_data['Close'].tail(20).mean()
    
    current_price = latest['Close']
    position_diff = ((current_price - ma20) / ma20) * 100
    
    # 预测结果
    prediction_label = "强势 (价格≥MA20)" if prediction == 0 else "弱势 (价格<MA20)"
    prob_strong = probability[0]
    prob_weak = probability[1]
    confidence = max(prob_strong, prob_weak)
    
    st.markdown("---")
    st.header(f"📊 {stock_code} 预测结果")
    
    # 当前数据
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 数据日期", latest_date)
    with col2:
        st.metric("💰 收盘价", f"{current_price:.2f}元")
    with col3:
        st.metric("📈 MA20", f"{ma20:.2f}元")
    with col4:
        current_status = "强势" if current_price >= ma20 else "弱势"
        st.metric("📍 当前状态", current_status, f"{position_diff:+.2f}%")
    
    st.markdown("---")
    
    # 预测结果
    st.subheader("🔮 预测结果（未来5日）")
    
    if prediction == 0:
        st.success(f"✅ **预测：{prediction_label}**")
    else:
        st.warning(f"⚠️ **预测：{prediction_label}**")
    
    # 概率显示
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("强势概率", f"{prob_strong:.1%}")
    with col2:
        st.metric("弱势概率", f"{prob_weak:.1%}")
    with col3:
        st.metric("置信度", f"{confidence:.1%}")
    
    # 置信度评级
    if confidence >= 0.80:
        confidence_level = "非常高 ⭐⭐⭐⭐⭐"
    elif confidence >= 0.70:
        confidence_level = "高 ⭐⭐⭐⭐"
    elif confidence >= 0.60:
        confidence_level = "中等 ⭐⭐⭐"
    else:
        confidence_level = "较低 ⭐⭐"
    
    st.info(f"**置信度评级**: {confidence_level}")
    
    st.markdown("---")
    
    # 操作建议
    st.subheader("💡 操作建议")
    
    if prediction == 0:
        if prob_strong >= 0.75:
            st.success("""
            📈 **建议：可考虑持有或适当增仓**
            
            理由：模型预测强势且置信度较高
            """)
        else:
            st.info("""
            📊 **建议：观察为主，谨慎持有**
            
            理由：模型预测强势但置信度中等
            """)
    else:
        if prob_weak >= 0.75:
            st.warning("""
            📉 **建议：考虑减仓或观望**
            
            理由：模型预测弱势且置信度较高
            """)
        else:
            st.info("""
            ⚠️ **建议：谨慎持有，控制仓位**
            
            理由：模型预测弱势但置信度中等
            """)
    
    # 风险提示
    st.error("""
    ⚠️ **风险提示**
    
    1. 模型预测仅供参考，不构成投资建议
    2. 股票投资有风险，入市需谨慎
    3. 请结合基本面、技术面等多方面因素综合判断
    4. 建议设置止损位，控制风险敞口
    """)


def display_multi_model_result(stock_code, stock_data, predictions_dict, model_info):
    """显示多模型预测结果"""
    
    latest = stock_data.iloc[-1]
    latest_date = stock_data.index[-1].strftime('%Y-%m-%d')
    
    # 计算MA20
    if 'MA_20' in stock_data.columns:
        ma20 = latest['MA_20']
    elif 'SMA_20' in stock_data.columns:
        ma20 = latest['SMA_20']
    else:
        ma20 = stock_data['Close'].tail(20).mean()
    
    current_price = latest['Close']
    position_diff = ((current_price - ma20) / ma20) * 100
    
    st.markdown("---")
    st.header(f"📊 {stock_code} 多模型预测对比")
    
    # 当前数据
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📅 数据日期", latest_date)
    with col2:
        st.metric("💰 收盘价", f"{current_price:.2f}元")
    with col3:
        st.metric("📈 MA20", f"{ma20:.2f}元")
    with col4:
        current_status = "强势" if current_price >= ma20 else "弱势"
        st.metric("📍 当前状态", current_status, f"{position_diff:+.2f}%")
    
    st.markdown("---")
    
    # 投票统计
    vote_strong = sum(1 for p in predictions_dict.values() if p['prediction'] == 0)
    vote_weak = len(predictions_dict) - vote_strong
    total_models = len(predictions_dict)
    
    st.subheader("🔮 多模型预测对比（未来5日）")
    
    # 综合结论
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("预测强势", f"{vote_strong}/{total_models}", 
                 f"{vote_strong/total_models:.0%}")
    with col2:
        st.metric("预测弱势", f"{vote_weak}/{total_models}",
                 f"{vote_weak/total_models:.0%}")
    with col3:
        avg_conf = sum(p['confidence'] for p in predictions_dict.values()) / total_models
        st.metric("平均置信度", f"{avg_conf:.1%}")
    
    # 一致性评级
    consensus = max(vote_strong, vote_weak) / total_models
    if consensus >= 0.8:
        st.success("**一致性评级**: 非常一致 ⭐⭐⭐⭐⭐ - 所有模型意见高度一致")
    elif consensus >= 0.6:
        st.info("**一致性评级**: 比较一致 ⭐⭐⭐⭐ - 大多数模型意见一致")
    else:
        st.warning("**一致性评级**: 意见分歧 ⭐⭐ - 模型意见存在分歧，建议谨慎")
    
    st.markdown("---")
    
    # 各模型详情
    st.subheader("📈 各模型预测详情")
    
    # 创建DataFrame显示
    model_data = []
    for model_name, pred in predictions_dict.items():
        model_data.append({
            '模型名称': model_name,
            '预测': '强势⭐' if pred['prediction'] == 0 else '弱势⚠️',
            '强势概率': f"{pred['prob_strong']:.1%}",
            '弱势概率': f"{pred['prob_weak']:.1%}",
            '置信度': f"{pred['confidence']:.1%}",
            '训练精确率': f"{pred['train_precision']:.1%}"
        })
    
    df = pd.DataFrame(model_data)
    df = df.sort_values('置信度', ascending=False)
    st.dataframe(df, use_container_width=True)
    
    st.markdown("---")
    
    # 综合建议
    st.subheader("💡 综合建议")
    
    if vote_strong > vote_weak:
        if consensus >= 0.8:
            st.success("""
            ✅ **多数模型预测：强势**
            
            📈 建议：可考虑持有或适当增仓
            
            理由：模型一致性高，预测强势
            """)
        elif consensus >= 0.6:
            st.info("""
            ✅ **多数模型预测：强势**
            
            📊 建议：观察为主，谨慎持有
            
            理由：多数模型预测强势，但存在一定分歧
            """)
        else:
            st.warning("""
            ✅ **多数模型预测：强势**
            
            ⚠️ 建议：谨慎对待，建议观望
            
            理由：模型意见分歧较大
            """)
    else:
        if consensus >= 0.8:
            st.warning("""
            ⚠️ **多数模型预测：弱势**
            
            📉 建议：考虑减仓或观望
            
            理由：模型一致性高，预测弱势
            """)
        elif consensus >= 0.6:
            st.info("""
            ⚠️ **多数模型预测：弱势**
            
            ⚠️ 建议：谨慎持有，控制仓位
            
            理由：多数模型预测弱势，但存在一定分歧
            """)
        else:
            st.info("""
            ⚠️ **多数模型预测：弱势**
            
            📊 建议：继续观察，暂不操作
            
            理由：模型意见分歧较大
            """)
    
    # 风险提示
    st.error("""
    ⚠️ **风险提示**
    
    1. 多模型预测提供更全面的参考，但不保证准确
    2. 当模型意见分歧时，说明市场处于不确定状态
    3. 建议结合基本面、技术面等多方面因素综合判断
    4. 股票投资有风险，入市需谨慎
    """)


if __name__ == '__main__':
    main()


