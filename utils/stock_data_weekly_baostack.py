# 添加必要的库导入
import baostock as bs
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def get_stock_data_weekly(stock_code, days):
    # 登陆系统
    bs.login()
    
    # 设置日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 根据股票代码判断前缀
    prefix = 'sh' if stock_code.startswith('6') else 'sz'
    
    # 查询股票数据
    rs = bs.query_history_k_data_plus(
        f"{prefix}.{stock_code}",
        "date,open,high,low,close,preclose,volume,amount,turn,pctChg",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="2"
    )
    
    # 处理结果
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    
    # 创建DataFrame并设置索引
    result = pd.DataFrame(data_list, columns=rs.fields)
    if not result.empty:
        result['date'] = pd.to_datetime(result['date'])
        result.set_index('date', inplace=True)
    # 将字符串列转换为浮点数
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']:
        if col in result.columns:
            result[col] = result[col].astype(float)
    # 按周重采样并聚合数据
    # 假设周线以自然周（周一到周日）为周期，可根据需要调整参数（如'W-FRI'表示周五结束）
    weekly_data = result.resample('W').agg({
        'open': 'first',    # 周一的开盘价
        'high': 'max',      # 当周最高价
        'low': 'min',       # 当周最低价
        'close': 'last',    # 周五的收盘价
        'volume': 'sum',     # 当周成交量总和
        'amount': 'sum',
        'turn': 'sum'     # 当周成交额总和
    }).dropna()  # 删除没有数据的周
    # 计算周涨跌幅
    weekly_data['PricechangeRate'] = (weekly_data['close'] / weekly_data['open'] - 1) * 100
    # 重置索引，将日期恢复为列
    weekly_data.reset_index(inplace=True)
    # 重命名列名为mplfinance所需的格式
    weekly_data.rename(columns={
    'open': 'Open',
    'close': 'Close',
    'high': 'High',
    'low': 'Low',
    'volume': 'Volume',
    'amount': 'Turnover',
    'turn': 'TurnoverRate'
    }, inplace=True)
    weekly_data['Avg']=weekly_data['Turnover']/(weekly_data['Volume']*100)
    # 将索引转换为日期时间格式
    weekly_data['date'] = pd.to_datetime(weekly_data['date'])
    weekly_data.set_index('date', inplace=True)   
    # 登出系统
    bs.logout()
    return weekly_data