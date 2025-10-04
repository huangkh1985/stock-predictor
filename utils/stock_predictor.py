import torch
import numpy as np
import pandas as pd
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta

class StockPredictor:
    """
    股票预测器类，用于预测未来股票价格和涨跌幅
    """
    def __init__(self, model, scaler_feature, scaler_label, device=torch.device('cpu')):
        """
        初始化股票预测器
        
        参数:
            model: 训练好的模型
            scaler_feature: 特征归一化器
            scaler_label: 标签归一化器
            device: 设备(CPU/GPU)
        """
        self.model = model
        self.scaler_feature = scaler_feature
        self.scaler_label = scaler_label
        self.device = device
        self.model.to(device)
        self.model.eval()  # 设置为评估模式
    
    def predict_future(self, last_sequence, last_known_price, future_days=5):
        """
        预测未来几天的股票价格
        
        参数:
            last_sequence: 最后的时间序列数据
            last_known_price: 最后一个已知价格
            future_days: 要预测的未来天数
        
        返回:
            未来几天的预测价格和涨跌百分比
        """
        # 确保输入数据格式正确
        if isinstance(last_sequence, np.ndarray):
            input_seq = last_sequence.copy()
        else:
            input_seq = last_sequence.cpu().numpy().copy()
        
        # 存储预测结果
        future_predictions = []
        
        # 上一个价格，初始为最后一个已知价格
        prev_price = last_known_price
        
        # 逐天预测
        for _ in range(future_days):
            # 准备输入数据
            x_input = torch.FloatTensor(input_seq).to(self.device)
            
            # 如果输入是单个样本，添加批次维度
            if len(x_input.shape) == 2:
                x_input = x_input.unsqueeze(0)
            
            # 进行预测
            with torch.no_grad():
                if hasattr(self.model, 'forward_with_attention'):
                    pred, _ = self.model.forward_with_attention(x_input)
                else:
                    output = self.model(x_input)
                    # 处理模型输出可能是元组的情况
                    if isinstance(output, tuple):
                        pred = output[0]  # 假设第一个元素是预测值
                    else:
                        pred = output
            
            # 转换预测结果为numpy数组
            pred_numpy = pred.cpu().numpy()
            
            # 处理三维数组的情况 [batch_size, seq_len, features]
            if len(pred_numpy.shape) == 3:
                # 取最后一个时间步的预测值
                pred_numpy = pred_numpy[:, -1, :]
            
            # 确保是二维数组 [samples, features]
            if len(pred_numpy.shape) == 1:
                pred_numpy = pred_numpy.reshape(1, -1)
            
            # 反归一化预测结果
            pred_price = float(self.scaler_label.inverse_transform(pred_numpy)[0][0])
            
            # 计算涨跌百分比（与上一个价格比较）
            change_percent = (pred_price - prev_price) / prev_price * 100
            
            # 存储预测结果和涨跌百分比
            future_predictions.append((pred_price, change_percent))
            
            # 更新上一个价格为当前预测价格
            prev_price = pred_price
            
            # 更新输入序列用于下一次预测
            # 移除最早的时间步，添加新预测作为最新的时间步
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1, 0] = pred_numpy[0][0]  # 假设目标变量是第一个特征
        
        return future_predictions
    
    def display_future_predictions(self, future_predictions, last_date, stock_code):
        """
        显示未来预测结果
        
        参数:
            future_predictions: 预测结果列表，每个元素为(价格, 涨跌幅)
            last_date: 最后一个已知日期
            stock_code: 股票代码
        """
        future_days = len(future_predictions)
        
        # 显示预测结果
        print(f"\n未来{future_days}天的股票价格预测:")
        future_table = PrettyTable(['预测日期', '预测收盘价', '涨跌幅(%)', '涨跌'])
        
        # 生成未来日期
        future_dates = []
        future_date = pd.to_datetime(last_date)
        
        for i, (price, change) in enumerate(future_predictions):
            # 计算预测日期（跳过周末）
            future_date = future_date + timedelta(days=1)
            while future_date.weekday() > 4:  # 5和6是周六和周日
                future_date += timedelta(days=1)
            future_dates.append(future_date)
            
            # 判断涨跌
            direction = "上涨" if change > 0 else "下跌" if change < 0 else "持平"
            direction_symbol = "↑" if change > 0 else "↓" if change < 0 else "-"
            
            # 添加到表格
            future_table.add_row([
                future_date.strftime('%Y-%m-%d'),
                f"{price:.2f}",
                f"{change:.2f}",
                f"{direction} {direction_symbol}"
            ])
        
        print(future_table)
        return future_dates
    
    def plot_future_predictions(self, historical_dates, historical_prices, future_predictions, last_date, stock_code):
        """
        绘制未来预测图表
        
        参数:
            historical_dates: 历史日期
            historical_prices: 历史价格
            future_predictions: 预测结果列表，每个元素为(价格, 涨跌幅)
            last_date: 最后一个已知日期
            stock_code: 股票代码
        """
        plt.figure(figsize=(12, 6), dpi=300)
        
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
        plt.plot(historical_dates_dt, historical_prices, 'b-', linewidth=1.5, label='历史收盘价')
        
        # 绘制预测数据
        plt.plot(np.concatenate([[historical_dates_dt[-1]], future_dates]), 
                np.concatenate([[historical_prices[-1]], future_prices]), 
                'r--', linewidth=1.5, label='预测收盘价')
        
        # 在预测点添加标记
        for i, ((price, change), date) in enumerate(zip(future_predictions, future_dates)):
            color = 'red' if change > 0 else 'green'
            plt.scatter(date, price, color=color, s=50, zorder=5)
            plt.annotate(f"{change:.1f}%", 
                        (date, price),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        color=color)
        
        # 设置x轴为日期格式
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # 根据数据量调整x轴刻度
        all_dates = np.concatenate([historical_dates_dt, future_dates])
        if len(all_dates) > 30:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=len(all_dates)//10))
        else:
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        
        plt.gcf().autofmt_xdate()  # 自动格式化日期标签
        
        plt.xlabel('日期', fontsize=10)
        plt.ylabel('股票价格 (元)', fontsize=10)
        plt.title(f'股票 {stock_code} 历史与未来{len(future_predictions)}天预测', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./stock_{stock_code}_future_prediction.png', dpi=300)
        plt.show()