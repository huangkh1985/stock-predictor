import pandas as pd
import akshare as ak
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
import time
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings("ignore")

def fetch_forex_data_weekly(start_date, end_date):
    """
    获取美元兑离岸人民币汇率数据，并转换为周频数据
    
    参数:
    start_date: 开始日期，格式为'YYYYMMDD'
    end_date: 结束日期，格式为'YYYYMMDD'
    
    返回:
    pandas.DataFrame: 包含周频汇率数据的DataFrame
    """
    # 获取美元兑离岸人民币汇率数据
    forex_hist_em_df = ak.forex_hist_em(symbol="USDCNH")
    # 将日期列转换为日期格式以便比较
    forex_hist_em_df['日期'] = pd.to_datetime(forex_hist_em_df['日期'])
    # 转换start_date为日期格式 
    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
    # 筛选数据
    forex_hist_df = forex_hist_em_df[forex_hist_em_df['日期'] >= start_date_dt]
    # 使用列表选择多列
    forex_hist_df = forex_hist_df[['日期', '今开', '最高', '最低', '最新价']]

    forex_hist_df.rename(columns={
        '日期': 'date'
    }, inplace=True)
    # 修复：先设置日期索引，再进行重采样
    forex_hist_df.set_index('date', inplace=True)
    # 重采样为每周数据
    forex_hist_df_weekly = forex_hist_df.resample('W').agg({
        '今开': 'first',    # 周一的开盘价
        '最高': 'max',      # 当周最高价
        '最低': 'min',       # 当周最低价
        '最新价': 'last',    # 周五的收盘价     # 当周成交额总和
    }).dropna()  # 删除没有数据的周
    # 重命名列名为mplfinance所需的格式
    forex_hist_df_weekly.rename(columns={
        '今开': 'open_forex',
        '最高': 'high_forex',
        '最低': 'low_forex',
        '最新价': 'new_forex'
    }, inplace=True)

    return forex_hist_df_weekly

def fetch_sh000001_data_weekly(start_date, end_date):
    """
    获取上证指数数据，并转换为周频数据
    
    参数:
    start_date: 开始日期，格式为'YYYYMMDD'
    end_date: 结束日期，格式为'YYYYMMDD'
    
    返回:
    pandas.DataFrame: 包含周频上证指数数据的DataFrame
    """
    # 获取上证指数数据
    stock_zh_index_daily_df = ak.stock_zh_index_daily(symbol="sh000001")
    # 将日期列转换为日期格式以便比较
    stock_zh_index_daily_df['date'] = pd.to_datetime(stock_zh_index_daily_df['date'])
    # 转换start_date为日期格式
    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
    # 筛选数据
    stock_index_sz_df = stock_zh_index_daily_df[stock_zh_index_daily_df['date'] >= start_date_dt]
    # 使用列表选择多列
    stock_index_sz_df = stock_index_sz_df[['date', 'open', 'high', 'low', 'close', 'volume']]
    # 修复：先设置日期索引，再进行重采样
    stock_index_sz_df.set_index('date', inplace=True)
    # 重采样为每周数据
    stock_index_sz_df_weekly = stock_index_sz_df.resample('W').agg({
        'open': 'first',    # 周一的开盘价
        'high': 'max',      # 当周最高价
        'low': 'min',       # 当周最低价
        'close': 'last', 
        'volume': 'sum'   # 周五的收盘价     # 当周成交额总和
    }).dropna()  # 删除没有数据的周
    stock_index_sz_df_weekly.reset_index(inplace=True)
    # 重命名列名为mplfinance所需的格式
    stock_index_sz_df_weekly.rename(columns={
        'date': '日期',
        'open': 'open_SZ_index',
        'high': 'high_SZ_index',
        'low': 'low_SZ_index',
        'close': 'close_SZ_index',
        'volume': 'volume_SZ_index'
        }, inplace=True)
    # 将日期列设置为索引
    return stock_index_sz_df_weekly.set_index('日期')

def fetch_industry_data_weekly(industry_name, start_date, end_date):
    """
    获取行业指数数据，并转换为周频数据
    
    参数:
    industry_name: 行业名称
    start_date: 开始日期，格式为'YYYYMMDD'
    end_date: 结束日期，格式为'YYYYMMDD'
    
    返回:
    pandas.DataFrame: 包含周频行业指数数据的DataFrame
    """
    stock_board_industry_index = ak.stock_board_industry_index_ths(symbol=industry_name, start_date=start_date, end_date=end_date)
    #stock_board_industry_index = ak.stock_board_industry_hist_em(symbol=industry_name, start_date=start_date, end_date=end_date, period="日k", adjust="qfq")
    # 将日期列转换为日期格式以便比较
    stock_board_industry_index['日期'] = pd.to_datetime(stock_board_industry_index['日期'])
    # 转换start_date为日期格式 
    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
    # 筛选数据
    stock_board_industry_df = stock_board_industry_index[stock_board_industry_index['日期'] >= start_date_dt]
    # 使用列表选择多列
    stock_board_industry_df = stock_board_industry_df[['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量','成交额']]
    # 修复：先设置日期索引，再进行重采样
    stock_board_industry_df.set_index('日期', inplace=True)
    # 重采样为每周数据
    stock_board_industry_df_weekly = stock_board_industry_df.resample('W').agg({
        '开盘价': 'first',    # 周一的开盘价
        '最高价': 'max',      # 当周最高价
        '最低价': 'min',       # 当周最低价
        '收盘价': 'last', 
        '成交量': 'sum',
        '成交额': 'sum'   # 周五的收盘价     # 当周成交额总和
    }).dropna()  # 删除没有数据的周
    
    stock_board_industry_df_weekly.rename(columns={
        '开盘价': 'open_industry_index',
        '收盘价': 'close_industry_index',
        '最高价': 'high_industry_index',
        '最低价': 'low_industry_index',
        '成交量': 'volume_industry_index',
        '成交额': 'turnover_industry_index',
        }, inplace=True)
    # 将日期列设置为索引
    return stock_board_industry_df_weekly

def fetch_usd_cnh_data_weekly(start_date, end_date):
    """
    获取美股标普500指数数据，并转换为周频数据
    
    参数:
    start_date: 开始日期，格式为'YYYYMMDD'
    end_date: 结束日期，格式为'YYYYMMDD'
    
    返回:
    pandas.DataFrame: 包含周频美股数据的DataFrame
    """
    # 获取美股数据
    index_us_stock_sina_df = ak.index_us_stock_sina(symbol=".INX")
    # 将日期列转换为日期格式以便比较
    index_us_stock_sina_df['date'] = pd.to_datetime(index_us_stock_sina_df['date'])
    # 转换start_date为日期格式
    start_date_dt = pd.to_datetime(start_date, format='%Y%m%d')
    # 筛选数据
    index_us_stock_df = index_us_stock_sina_df[index_us_stock_sina_df['date'] >= start_date_dt]
    # 使用列表选择多列
    index_us_stock_df = index_us_stock_df[['date', 'open', 'high', 'low', 'close', 'volume']]
    # 修复：先设置日期索引，再进行重采样
    index_us_stock_df.set_index('date', inplace=True)
    # 重采样为每周数据
    index_us_stock_df_weekly = index_us_stock_df.resample('W').agg({
        'open': 'first',    # 周一的开盘价
        'high': 'max',      # 当周最高价
        'low': 'min',       # 当周最低价
        'close': 'last', 
        'volume': 'sum'   # 周五的收盘价     # 当周成交额总和
    }).dropna()  # 删除没有数据的周
    index_us_stock_df_weekly.reset_index(inplace=True)
    # 重命名列名为mplfinance所需的格式
    index_us_stock_df_weekly.rename(columns={
        'date': '日期',
        'open': 'open_us',
        'high': 'high_us',
        'low': 'low_us',
        'close': 'close_us',
        'volume': 'volume_us'
    }, inplace=True)
    # 将日期列设置为索引
    return index_us_stock_df_weekly.set_index('日期')

def get_stock_industry_concept(stock_code):
    """
    获取股票所属行业和概念
    
    参数:
    stock_code: 股票代码
    
    返回:
    dict: 包含行业信息的字典
    """
    try:
        # 获取股票所属行业
        stock_info = ak.stock_individual_info_em(symbol=stock_code)
        industry = None
        for _, row in stock_info.iterrows():
            if row['item'] == '行业':
                industry = row['value']
                break
        
        # 获取股票所属概念
        try:
            print("正在获取行业板块信息...")
            
        except Exception as e:
            print(f"获取行业板块信息时出错: {str(e)}")
        
        return {
            "industry": industry
        }
    except Exception as e:
        print(f"获取股票行业信息时出错: {str(e)}")
        return {"industry": None}