import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prettytable import PrettyTable

def mape(y_true, y_pred):
    """
    计算平均绝对百分比误差（MAPE）
    
    参数:
    y_true: 实际值
    y_pred: 预测值
    
    返回:
    float: MAPE值（百分比）
    """
    record = []
    for index in range(len(y_true)):
        temp_mape = np.abs((y_pred[index] - y_true[index]) / y_true[index])
        record.append(temp_mape)
    return np.mean(record) * 100

def evaluate_forecasts(Ytest, predicted_data, n_out):
    """
    评估预测结果的性能
    
    参数:
    Ytest: 实际值
    predicted_data: 预测值
    n_out: 预测步数
    
    返回:
    tuple: (mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table)
    """
    mse_dic = []
    rmse_dic = []
    mae_dic = []
    mape_dic = []
    r2_dic = []
    
    table = PrettyTable(['测试集指标','MSE', 'RMSE', 'MAE', 'MAPE','R2'])
    for i in range(n_out):
        actual = Ytest.flatten() if n_out == 1 else [float(row[i]) for row in Ytest]
        predicted = predicted_data.flatten() if n_out == 1 else [float(row[i]) for row in predicted_data]
        
        # 打印长度，用于调试
        print(f"实际值长度: {len(actual)}, 预测值长度: {len(predicted)}")
        
        mse = mean_squared_error(actual, predicted)
        mse_dic.append(mse)
        
        rmse = sqrt(mean_squared_error(actual, predicted))
        rmse_dic.append(rmse)
        
        mae = mean_absolute_error(actual, predicted)
        mae_dic.append(mae)
        
        MApe = mape(actual, predicted)
        mape_dic.append(MApe)
        
        r2 = r2_score(actual, predicted)
        r2_dic.append(r2)
        
        if n_out == 1:
            strr = '预测结果指标：'
        else:
            strr = '第'+ str(i + 1)+'步预测结果指标：'
        
        table.add_row([strr, mse, rmse, mae, str(MApe)+'%', str(r2*100)+'%'])

    return mse_dic, rmse_dic, mae_dic, mape_dic, r2_dic, table