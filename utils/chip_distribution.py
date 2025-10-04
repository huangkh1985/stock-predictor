import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import seaborn as sns
from scipy import stats
import time
import math

# 忽略警告
warnings.filterwarnings("ignore")

class ChipDistribution():
    def __init__(self):
        self.Chip = {}  # 当前获利盘
        self.ChipList = {}  # 所有的获利盘的
        
    def get_data(self, df):
        """设置数据源"""
        self.data = df.reset_index() 
        
    def calcuJUN(self, dateT, highT, lowT, volT, TurnoverRateT, A, minD):
        """均匀分布计算筹码"""
        x = []
        l = (highT - lowT) / minD
        for i in range(int(l)):
            x.append(round(lowT + i * minD, 2))
        length = len(x)
        eachV = volT/length
        for i in self.Chip:
            self.Chip[i] = self.Chip[i] * (1 - TurnoverRateT * A)
        for i in x:
            if i in self.Chip:
                self.Chip[i] += eachV * (TurnoverRateT * A)
            else:
                self.Chip[i] = eachV * (TurnoverRateT * A)
        self.ChipList[dateT] = copy.deepcopy(self.Chip)
        
    def calcuSin(self, dateT, highT, lowT, avgT, volT, TurnoverRateT, minD, A):
        """正弦分布计算筹码"""
        x = []
        l = (highT - lowT) / minD
        for i in range(int(l)):
            x.append(round(lowT + i * minD, 2))
        length = len(x)
        # 计算仅仅今日的筹码分布
        tmpChip = {}
        eachV = volT/length
        # 极限法分割去逼近
        for i in x:
            x1 = i
            x2 = i + minD
            h = 2 / (highT - lowT)
            s = 0
            if i < avgT:
                y1 = h / (avgT - lowT) * (x1 - lowT)
                y2 = h / (avgT - lowT) * (x2 - lowT)
                s = minD * (y1 + y2) / 2
                s = s * volT
            else:
                y1 = h / (highT - avgT) * (highT - x1)
                y2 = h / (highT - avgT) * (highT - x2)
                s = minD * (y1 + y2) / 2
                s = s * volT
            tmpChip[i] = s
        for i in self.Chip:
            self.Chip[i] = self.Chip[i] * (1 - TurnoverRateT * A)
        for i in tmpChip:
            if i in self.Chip:
                self.Chip[i] += tmpChip[i] * (TurnoverRateT * A)
            else:
                self.Chip[i] = tmpChip[i] * (TurnoverRateT * A)
        self.ChipList[dateT] = copy.deepcopy(self.Chip)
        
    def calcu(self, dateT, highT, lowT, avgT, volT, TurnoverRateT, minD=0.01, flag=1, AC=1):
        """计算筹码分布的统一接口"""
        if flag == 1:
            self.calcuSin(dateT, highT, lowT, avgT, volT, TurnoverRateT, A=AC, minD=minD)
        elif flag == 2:
            self.calcuJUN(dateT, highT, lowT, volT, TurnoverRateT, A=AC, minD=minD)
            
    def calcuChip(self, flag=1, AC=1):  # flag 使用哪个计算方式, AC 衰减系数
        """计算所有日期的筹码分布"""
        low = self.data['Low']
        high = self.data['High']
        vol = self.data['Volume']
        TurnoverRate = self.data['TurnoverRate']
        avg = self.data['Avg']
        date = self.data['日期']
        for i in range(len(date)):
            highT = high[i]
            lowT = low[i]
            volT = vol[i]
            TurnoverRateT = TurnoverRate[i]
            avgT = avg[i]
            dateT = date[i]
            self.calcu(dateT, highT, lowT, avgT, volT, TurnoverRateT/100, flag=flag, AC=AC)  # 东方财富的小数位要注意，兄弟萌。我不除100懵逼了
            
    def winner(self, p=None):
        """计算获利盘比例"""
        Profit = []
        date = self.data['日期']
        if p == None:  # 不输入默认close
            p = self.data['Close']
            count = 0
            for i in self.ChipList:
                # 计算目前的比例
                Chip = self.ChipList[i]
                total = 0
                be = 0
                for i in Chip:
                    total += Chip[i]
                    if i < p[count]:
                        be += Chip[i]
                if total != 0:
                    bili = be / total
                else:
                    bili = 0
                count += 1
                Profit.append(bili)
        else:
            for i in self.ChipList:
                # 计算目前的比例
                Chip = self.ChipList[i]
                total = 0
                be = 0
                for i in Chip:
                    total += Chip[i]
                    if i < p:
                        be += Chip[i]
                if total != 0:
                    bili = be / total
                else:
                    bili = 0
                Profit.append(bili)
        return Profit
        
    def lwinner(self, N=5, p=None):
        """计算滑动窗口获利盘比例"""
        data = copy.deepcopy(self.data)
        date = data['日期']
        # 扩展结果列表，存储更多信息
        ans = []  # 原始的结果列表
        detailed_results = []  # 扩展的结果列表，包含更多信息
        total_dates = len(date)       
        for i in range(total_dates):
            # 只打印进度信息，每10%显示一次
            if i % (total_dates // 10) == 0:
                print(f"处理进度: {i/total_dates*100:.1f}% - 当前日期: {date[i]}")                
            if i < N:
                ans.append(None)
                detailed_results.append(None)
                continue               
            self.data = data[i-N:i]
            self.data.index = range(0, N)
            self.__init__()
            self.calcuChip()    # 使用默认计算方式
            a = self.winner(p)
            win_ratio = a[-1]
            ans.append(win_ratio)            
            # 添加更多分析数据
            if win_ratio is not None:
                # 计算当前日期的其他指标
                current_price = data.iloc[i-1]['Close'] if i > 0 else None
                price_change = data.iloc[i-1]['PricechangeRate'] if i > 0 else None
                
                detailed_results.append({
                    'date': date[i],
                    'win_ratio': win_ratio,  # 获利盘比例
                    'price': current_price,  # 当前价格
                    'price_change': price_change,  # 价格变化率
                    # 可以添加更多指标
                })
            else:
                detailed_results.append(None)         
        print("处理完成: 100%")        
        self.data = data
        return ans, detailed_results  # 返回原始结果和详细结果
        
    def cost(self, N_list=None):
        """
        计算多个成本集中度      
        参数:
        N_list: 成本集中度百分比列表，如[90, 70]表示计算90%和70%的成本集中度    
        返回:
        字典，键为成本集中度百分比，值为对应的成本集中度列表
        """
        if N_list is None:
            N_list = [90]  # 默认计算90%成本集中度
            
        date = self.data['日期']
        result_dict = {}
        
        for N in N_list:
            N_percent = N / 100  # 转换成百分比
            ans = []
            for i in self.ChipList:  # 我的ChipList本身就是有顺序的
                Chip = self.ChipList[i]
                ChipKey = sorted(Chip.keys())  # 排序
                total = 0  # 当前比例
                sumOf = 0  # 所有筹码的总和
                for j in Chip:
                    sumOf += Chip[j]
        
                for j in ChipKey:
                    tmp = Chip[j]
                    tmp = tmp / sumOf
                    total += tmp
                    if total > N_percent:
                        ans.append(j)
                        break
                else:
                    # 如果没有找到满足条件的价格，使用最高价
                    ans.append(ChipKey[-1] if ChipKey else None)
                    
                result_dict[N] = ans
        return result_dict
    
    def add_chip_indicators(self, df):
        """
        计算并添加筹码分布指标到DataFrame
        
        参数:
        df: 包含股票数据的DataFrame
        
        返回:
        添加了筹码分布指标的DataFrame
        """
        # 保存原始数据的副本
        original_df = df.copy(deep=True)
        
        # 设置数据源
        self.get_data(df)
        
        # 计算筹码分布
        self.calcuChip(flag=1, AC=1)
        
        # 计算获利盘比例
        profit_ratios = self.winner()
        
        # 计算多个成本分布
        cost_results = self.cost([90, 70])  # 计算90%和70%的成本集中度
        cost_90 = cost_results[90]
        cost_70 = cost_results[70]
        
        # 计算滑动窗口获利盘比例
        lwinner_ratios, detailed_results = self.lwinner(N=5)
        
        # 创建一个新的DataFrame来存储所有指标
        indicators_df = pd.DataFrame()
        
        # 添加日期列
        indicators_df['日期'] = self.data['日期']
        
        # 添加获利盘比例
        if len(profit_ratios) == len(indicators_df):
            indicators_df['ProfitRatio'] = profit_ratios
        else:
            # 处理长度不匹配的情况
            print(f"警告: 获利盘比例长度 ({len(profit_ratios)}) 与数据集长度 ({len(indicators_df)}) 不匹配")
            # 填充缺失值
            indicators_df['ProfitRatio'] = None
            for i in range(min(len(profit_ratios), len(indicators_df))):
                indicators_df.iloc[i, indicators_df.columns.get_loc('ProfitRatio')] = profit_ratios[i]
        
        # 添加成本分布
        if len(cost_90) == len(indicators_df):
            indicators_df['Cost90'] = cost_90
            indicators_df['Cost70'] = cost_70
        else:
            print(f"警告: 成本分布长度 ({len(cost_90)}) 与数据集长度 ({len(indicators_df)}) 不匹配")
            indicators_df['Cost90'] = None
            indicators_df['Cost70'] = None
            for i in range(min(len(cost_90), len(indicators_df))):
                indicators_df.iloc[i, indicators_df.columns.get_loc('Cost90')] = cost_90[i]
                indicators_df.iloc[i, indicators_df.columns.get_loc('Cost70')] = cost_70[i]
        
        # 添加滑动窗口获利盘比例
        if len(lwinner_ratios) == len(indicators_df):
            indicators_df['LWinnerRatio'] = lwinner_ratios
        else:
            print(f"警告: 滑动窗口获利盘比例长度 ({len(lwinner_ratios)}) 与数据集长度 ({len(indicators_df)}) 不匹配")
            indicators_df['LWinnerRatio'] = None
            for i in range(min(len(lwinner_ratios), len(indicators_df))):
                indicators_df.iloc[i, indicators_df.columns.get_loc('LWinnerRatio')] = lwinner_ratios[i]
        
        # 将指标合并到原始数据集
        result_df = pd.merge(original_df.reset_index(), indicators_df, on='日期', how='left')
        
        return result_df
