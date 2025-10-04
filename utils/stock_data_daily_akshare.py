import akshare as ak  # 添加akshare库的导入
from datetime import datetime, timedelta  
# 在文件顶部添加pandas导入
import pandas as pd


def fetch_stock_data_daily(stock_code, start_date, end_date):
    """
    获取股票历史数据
    
    参数:
    stock_code: 股票代码
    start_date: 开始日期，格式为'YYYYMMDD'，如果为None则根据days参数计算
    end_date: 结束日期，格式为'YYYYMMDD'，如果为None则使用当前日期
    days: 获取多少天的数据，仅当start_date为None时使用
    
    返回:
    处理后的股票数据DataFrame
    """
    # 设置日期范围
    #if end_date is None:
    #    end_date = datetime.now().strftime('%Y%m%d')
    
    #if start_date is None:
    #    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    #print(f"获取股票 {stock_code} 从 {start_date} 到 {end_date} 的历史数据...")
    
    # 获取股票历史数据
    try:
        # 使用akshare的新API获取股票数据
        stock_zh_a_hist_df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        df = stock_zh_a_hist_df
        print("使用stock_zh_a_hist获取数据成功")
    except Exception as e:
        print(f"使用stock_zh_a_hist获取数据失败: {e}")
        try:
            # 尝试使用另一个API
            df = ak.stock_zh_a_hist_tx(
                symbol=stock_code,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
            print("使用stock_zh_a_hist_tx获取数据成功")
        except Exception as e:
            print(f"使用stock_zh_a_hist_tx获取数据失败: {e}")
            try:
                # 尝试使用腾讯股票数据接口
                df = ak.stock_zh_index_daily_tx(symbol=stock_code)
                print("使用stock_zh_index_daily_tx获取数据成功")
            except Exception as e:
                print(f"使用stock_zh_index_daily_tx获取数据失败: {e}")
                # 如果所有方法都失败，使用示例数据或从本地加载
                print("所有获取数据方法都失败")
                return None
    
    # 使用列表选择多列
    df = df[['日期', '开盘', '收盘','最高', '最低', '成交量','成交额', '涨跌幅', '涨跌额', '换手率']]
    
    # 重命名列名为mplfinance所需的格式
    df.rename(columns={
        '日期': 'date',
        '开盘': 'Open',
        '收盘': 'Close',
        '最高': 'High',
        '最低': 'Low',
        '成交量': 'Volume',
        '成交额': 'Turnover',
        '涨跌幅': 'PricechangeRate',
        '涨跌额': 'PricechangeAmount',
        '换手率': 'TurnoverRate'
    }, inplace=True)
    
    # 计算平均价格
    df['Avg'] = df['Turnover']/(df['Volume']*100)
    
    # 将索引转换为日期时间格式
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    print(df.head())
    
    return df