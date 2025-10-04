import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from matplotlib import rcParams
from datetime import timedelta
import matplotlib.dates as mdates

class StockVisualizer:
    """
    股票数据可视化类，提供各种可视化方法
    """
    def __init__(self, style='default'):
        """
        初始化可视化器
        
        参数:
        style: 可视化样式，可选值为'default', 'dark', 'light'
        """
        self.set_style(style)
    
    def set_style(self, style='default'):
        """
        设置matplotlib的绘图样式
        
        参数:
        style: 可视化样式，可选值为'default', 'dark', 'light'
        """
        # 基础配置
        config = {
            "font.family": 'serif',
            "font.size": 10,
            "mathtext.fontset": 'stix',
            "font.serif": ['Times New Roman'],
            'axes.unicode_minus': False
        }
        
        # 根据样式设置不同的配置
        if style == 'dark':
            plt.style.use('dark_background')
            config.update({
                "figure.facecolor": "#2E2E2E",
                "axes.facecolor": "#2E2E2E",
                "axes.edgecolor": "white",
                "axes.labelcolor": "white",
                "xtick.color": "white",
                "ytick.color": "white",
                "grid.color": "gray",
            })
        elif style == 'light':
            plt.style.use('seaborn-v0_8-whitegrid')
        else:
            plt.style.use('default')
        
        rcParams.update(config)
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_prediction_results(self, predicted_data, actual_data, stock_code, mape_value, 
                               dates=None, save_path=None, show=True, title=None):
        """
        绘制预测结果与实际值的对比图
        
        参数:
        predicted_data: 预测值
        actual_data: 实际值
        stock_code: 股票代码
        mape_value: MAPE评估指标值
        dates: 日期列表（可选）
        save_path: 保存路径（可选）
        show: 是否显示图表，默认为True
        title: 自定义标题，默认为None
        
        返回:
        fig, ax: 图表对象，可用于进一步自定义
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        x = range(1, len(predicted_data) + 1)
        
        # 绘制预测值的折线图
        ax.plot(x, predicted_data, linestyle="--", linewidth=1.2, label='预测收盘价', marker="o", markersize=3)
        
        # 绘制实际值的折线图
        ax.plot(x, actual_data, linestyle="-", linewidth=1.0, label='实际收盘价', marker="x", markersize=3)
        
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("样本点", fontsize=10)
        ax.set_ylabel("股票价格 (元)", fontsize=10)
        
        # 添加更多的x轴刻度，每5个点显示一个
        ax.set_xticks(x[::5])
        
        # 设置标题
        if title:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f"股票 {stock_code} 收盘价预测结果\nMAPE: {mape_value:.2f}%", fontsize=12)
    
        # 添加日期标签（如果提供了日期）
        if dates is not None:
            # 确保日期是字符串格式
            date_labels = [str(d) for d in dates]
        
            # 根据日期数量决定显示间隔
            if len(date_labels) > 20:
                step = len(date_labels) // 10  # 只显示约10个日期标签
                tick_positions = x[::step]
                tick_labels = date_labels[::step]
            else:
                tick_positions = x
                tick_labels = date_labels
        
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, dpi=300)
    
        if show:
            plt.show()
    
        return fig, ax
    
    def plot_model_comparison(self, predicted_data1, predicted_data2, actual_data, stock_code, 
                             mape1, mape2, model1_name='原始模型', model2_name='增强特征模型',
                             save_path=None, show=True, title=None):
        """
        绘制两个模型预测结果的对比图
        
        参数:
        predicted_data1: 模型1的预测值
        predicted_data2: 模型2的预测值
        actual_data: 实际值
        stock_code: 股票代码
        mape1: 模型1的MAPE值
        mape2: 模型2的MAPE值
        model1_name: 模型1的名称，默认为'原始模型'
        model2_name: 模型2的名称，默认为'增强特征模型'
        save_path: 保存路径（可选）
        show: 是否显示图表，默认为True
        title: 自定义标题，默认为None
        
        返回:
        fig, ax: 图表对象，可用于进一步自定义
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        x = range(1, len(predicted_data1) + 1)
        
        ax.plot(x, predicted_data1, linestyle="--", linewidth=1.0, label=f'{model1_name}预测', marker="o", markersize=2, alpha=0.7)
        ax.plot(x, predicted_data2, linestyle="--", linewidth=1.0, label=f'{model2_name}预测', marker="s", markersize=2, alpha=0.7)
        ax.plot(x, actual_data, linestyle="-", linewidth=1.2, label='实际收盘价', marker="x", markersize=3)
        
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlabel("样本点", fontsize=10)
        ax.set_ylabel("股票价格 (元)", fontsize=10)
        ax.set_xticks(x[::5])
        
        # 设置标题
        if title:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f"{model1_name} vs {model2_name} - 股票 {stock_code} 收盘价预测对比\n"
                        f"{model1_name}MAPE: {mape1:.2f}% vs {model2_name}MAPE: {mape2:.2f}%", fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_correlation_scatter(self, actual_data, predicted_data, save_path=None, show=True, title=None):
        """
        绘制预测值与实际值的散点图，查看相关性
        
        参数:
        actual_data: 实际值
        predicted_data: 预测值
        save_path: 保存路径（可选）
        show: 是否显示图表，默认为True
        title: 自定义标题，默认为None
        
        返回:
        fig, ax: 图表对象，可用于进一步自定义
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(actual_data, predicted_data, alpha=0.7)
        ax.plot([min(actual_data), max(actual_data)], [min(actual_data), max(actual_data)], 'r--')
        ax.set_xlabel('实际收盘价')
        ax.set_ylabel('预测收盘价')
        
        # 计算相关系数
        correlation = np.corrcoef(actual_data, predicted_data)[0, 1]
        
        # 设置标题
        if title:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(f'预测值与实际值对比 (相关系数: {correlation:.4f})')
        
        ax.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def create_prediction_comparison_table(self, actual_data, predicted_data, dates, start_idx):
        """
        创建预测结果与实际值对比表格
        
        参数:
        actual_data: 实际值
        predicted_data: 预测值
        dates: 日期列表
        start_idx: 起始索引
        
        返回:
        PrettyTable: 格式化的表格
        """
        comparison_table = PrettyTable(['日期', '实际收盘价', '预测收盘价', '误差(%)', '涨跌'])
        last_n = min(10, len(actual_data))
        
        for i in range(last_n):
            idx = len(actual_data) - last_n + i
            
            # 确保从数组中提取单个数值
            actual_value = float(actual_data[idx][0]) if hasattr(actual_data[idx], '__len__') else float(actual_data[idx])
            pred_value = float(predicted_data[idx][0]) if hasattr(predicted_data[idx], '__len__') else float(predicted_data[idx])
            
            error_pct = abs((pred_value - actual_value) / actual_value * 100)
            
            # 判断涨跌预测是否正确
            if idx > 0:
                prev_actual = float(actual_data[idx-1][0]) if hasattr(actual_data[idx-1], '__len__') else float(actual_data[idx-1])
                prev_pred = float(predicted_data[idx-1][0]) if hasattr(predicted_data[idx-1], '__len__') else float(predicted_data[idx-1])
                
                actual_change = "上涨" if actual_value > prev_actual else "下跌" if actual_value < prev_actual else "持平"
                pred_change = "上涨" if pred_value > prev_pred else "下跌" if pred_value < prev_pred else "持平"
                direction = "✓" if actual_change == pred_change else "✗"
            else:
                direction = "-"
            
            # 确保索引在范围内并使用正确的日期数组
            date_idx = start_idx + idx
            date_str = dates[date_idx] if date_idx < len(dates) else f"样本{idx+1}"
            
            comparison_table.add_row([
                date_str,
                f"{actual_value:.2f}",
                f"{pred_value:.2f}",
                f"{error_pct:.2f}",
                direction
            ])
        
        return comparison_table
    
    def plot_future_predictions(self, historical_dates, historical_prices, future_predictions, 
                               last_date, stock_code, save_path=None, show=True):
        """
        绘制未来预测图表
        
        参数:
        historical_dates: 历史日期
        historical_prices: 历史价格
        future_predictions: 预测结果列表，每个元素为(价格, 涨跌幅)
        last_date: 最后一个已知日期
        stock_code: 股票代码
        save_path: 保存路径（可选）
        show: 是否显示图表，默认为True
        
        返回:
        fig, ax: 图表对象，可用于进一步自定义
        """
        fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
        
        # 历史数据
        historical_dates_dt = pd.to_datetime(historical_dates)
        
        # 未来数据
        future_dates = []
        future_date = pd.to_datetime(last_date)
        for i in range(len(future_predictions)):
            future_date = future_date + timedelta(days=1)
            while future_date.weekday() > 4:  # 跳过周末
                future_date += timedelta(days=1)
            future_dates.append(future_date)
        
        future_prices = [price for price, _ in future_predictions]
        
        # 绘制历史数据
        ax.plot(historical_dates_dt, historical_prices, 'b-', linewidth=1.5, label='历史收盘价')
        
        # 绘制预测数据
        ax.plot(np.concatenate([[historical_dates_dt[-1]], future_dates]), 
                np.concatenate([[historical_prices[-1]], future_prices]), 
                'r--', linewidth=1.5, label='预测收盘价')
        
        # 在预测点添加标记
        for i, ((price, change), date) in enumerate(zip(future_predictions, future_dates)):
            color = 'red' if change > 0 else 'green'
            ax.scatter(date, price, color=color, s=50, zorder=5)
            ax.annotate(f"{change:.1f}%", 
                        (date, price),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        color=color)
        
        # 设置x轴为日期格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # 根据数据量调整x轴刻度
        all_dates = np.concatenate([historical_dates_dt, future_dates])
        if len(all_dates) > 30:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=len(all_dates)//10))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator())
        
        plt.gcf().autofmt_xdate()  # 自动格式化日期标签
        
        ax.set_xlabel('日期', fontsize=10)
        ax.set_ylabel('股票价格 (元)', fontsize=10)
        ax.set_title(f'股票 {stock_code} 历史与未来{len(future_predictions)}天预测', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_feature_importance(self, feature_names, importance_values, top_n=20, 
                               save_path=None, show=True, title=None):
        """
        绘制特征重要性条形图
        
        参数:
        feature_names: 特征名称列表
        importance_values: 重要性值列表
        top_n: 显示前n个特征，默认为20
        save_path: 保存路径（可选）
        show: 是否显示图表，默认为True
        title: 自定义标题，默认为None
        
        返回:
        fig, ax: 图表对象，可用于进一步自定义
        """
        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            '特征': feature_names,
            '重要性': importance_values
        }).sort_values('重要性', ascending=False)
        
        # 只保留前N个特征
        if len(feature_importance) > top_n:
            feature_importance = feature_importance.head(top_n)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        
        # 绘制水平条形图
        bars = ax.barh(feature_importance['特征'], feature_importance['重要性'])
        
        # 为条形图添加数值标签
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else 0
            ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                   va='center', ha='left', fontsize=8)
        
        # 设置轴标签
        ax.set_xlabel('特征重要性')
        
        # 设置标题
        if title:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title('特征重要性分析', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        if show:
            plt.show()
        
        return fig, ax
    
    def plot_metrics_comparison(self, model_names, metrics_dict, save_path=None, show=True):
        """
        绘制多个模型的评估指标对比图
        
        参数:
        model_names: 模型名称列表
        metrics_dict: 指标字典，格式为{'指标名': [模型1值, 模型2值, ...]}
        save_path: 保存路径（可选）
        show: 是否显示图表，默认为True
        
        返回:
        fig, ax: 图表对象，可用于进一步自定义
        """
        # 获取指标名称
        metric_names = list(metrics_dict.keys())
        num_metrics = len(metric_names)
        num_models = len(model_names)
        
        # 创建图形
        fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 6), dpi=300)
        
        # 如果只有一个指标，确保axes是列表
        if num_metrics == 1:
            axes = [axes]
        
        # 设置颜色
        colors = plt.cm.tab10(np.linspace(0, 1, num_models))
        
        # 绘制每个指标的条形图
        for i, metric_name in enumerate(metric_names):
            metric_values = metrics_dict[metric_name]
            
            # 创建条形图
            bars = axes[i].bar(model_names, metric_values, color=colors)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.4f}', ha='center', va='bottom', fontsize=9)
            
            # 设置标题和标签
            axes[i].set_title(f'{metric_name}对比')
            axes[i].set_ylabel(metric_name)
            axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        if show:
            plt.show()
        
        return fig, axes

# 为了向后兼容，保留原始函数
def plot_prediction_results(predicted_data, actual_data, stock_code, mape_value, dates=None, save_path=None):
    """
    绘制预测结果与实际值的对比图（向后兼容函数）
    """
    visualizer = StockVisualizer()
    visualizer.plot_prediction_results(predicted_data, actual_data, stock_code, mape_value, dates, save_path)

def plot_model_comparison(predicted_data1, predicted_data2, actual_data, stock_code, mape1, mape2, save_path=None):
    """
    绘制两个模型预测结果的对比图（向后兼容函数）
    """
    visualizer = StockVisualizer()
    visualizer.plot_model_comparison(predicted_data1, predicted_data2, actual_data, stock_code, mape1, mape2, save_path=save_path)

def plot_correlation_scatter(actual_data, predicted_data, save_path=None):
    """
    绘制预测值与实际值的散点图（向后兼容函数）
    """
    visualizer = StockVisualizer()
    visualizer.plot_correlation_scatter(actual_data, predicted_data, save_path)

def create_prediction_comparison_table(actual_data, predicted_data, dates, start_idx):
    """
    创建预测结果与实际值对比表格（向后兼容函数）
    """
    visualizer = StockVisualizer()
    return visualizer.create_prediction_comparison_table(actual_data, predicted_data, dates, start_idx)

def set_plot_style():
    """
    设置matplotlib的绘图样式（向后兼容函数）
    """
    visualizer = StockVisualizer()
    visualizer.set_style()