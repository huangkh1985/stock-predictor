import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


# 修改train_model函数，增加早停机制
def train_model(model, model_name, epochs=30, learning_rate=0.001, train_loader=None, patience=7):
    if train_loader is None:
        raise ValueError("必须提供train_loader参数")
        
    criterion = nn.MSELoss()
    
    # 为不同模型使用不同的优化器和学习率
    if "Attention" in model_name or "Enhanced" in model_name:
        # 添加权重衰减(L2正则化)
        weight_decay = 1e-5
        if hasattr(model, 'weight_decay'):
            weight_decay = model.weight_decay
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 使用更复杂的学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5, min_lr=1e-6, verbose=True
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 如果没有提供特定的train_loader，则使用默认的
    if train_loader is None:
        train_loader = globals()['train_loader']
    
    train_losses = []
    val_losses = []  # 添加验证损失跟踪
    best_loss = float('inf')
    no_improve_count = 0
    best_model_state = None
    
    # 创建验证集
    val_size = int(len(train_loader.dataset) * 0.1)
    train_size = len(train_loader.dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    val_loader = DataLoader(val_dataset, batch_size=train_loader.batch_size, shuffle=False)
    train_subset_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True)
    
    for epoch in tqdm(range(epochs), desc=f'Training {model_name}'):
        # 训练阶段
        model.train()
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(train_subset_loader):
            optimizer.zero_grad()

            # 处理不同模型的输出格式
            if model_name == "LSTM" or model_name == "Wavelet_LSTM" or "Enhanced_Wavelet_LSTM" in model_name:
                outputs, _ = model(inputs)
                # 检查输出维度，如果是3D，则取最后一个时间步
                if len(outputs.shape) == 3:
                    outputs = outputs[:, -1, :1]
                else:
                    outputs = outputs[:, :1]
            elif "Attention" in model_name or "Enhanced" in model_name:
                outputs, _ = model(inputs)
            else:
                outputs, _ = model(inputs)
                outputs = outputs[:, -1, :1]
            
            loss = criterion(outputs, targets)
            
            # 添加L1正则化
            if "Attention" in model_name or "Enhanced" in model_name:
                l1_lambda = 1e-5
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
                
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_subset_loader)
        train_losses.append(avg_epoch_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # 处理不同模型的输出格式，与训练阶段相同
                if model_name == "LSTM" or model_name == "Wavelet_LSTM" or "Enhanced_Wavelet_LSTM" in model_name:
                    outputs, _ = model(inputs)
                    if len(outputs.shape) == 3:
                        outputs = outputs[:, -1, :1]
                    else:
                        outputs = outputs[:, :1]
                elif "Attention" in model_name or "Enhanced" in model_name:
                    outputs, _ = model(inputs)
                else:
                    outputs, _ = model(inputs)
                    outputs = outputs[:, -1, :1]
                
                val_loss += criterion(outputs, targets).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 打印训练和验证损失
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 使用学习率调度器
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)  # 使用验证损失
            else:
                scheduler.step()
        
        # 改进的早停机制，基于验证损失
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_count = 0
            # 保存最佳模型状态
            best_model_state = {k: v.cpu().detach() for k, v in model.state_dict().items()}
        else:
            no_improve_count += 1
            
        if no_improve_count >= patience:
            print(f"早停: {epoch+1}/{epochs} 轮后验证损失没有改进")
            break
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 返回训练和验证损失历史
    return model, {'train': train_losses, 'val': val_losses}

# 添加数据增强函数
def augment_time_series(X, y, noise_level=0.05, shift_range=0.1, scale_range=0.1):
    """
    对时间序列数据进行增强
    
    Args:
        X: 输入特征 [samples, time_steps, features]
        y: 目标值 [samples, 1]
        noise_level: 添加的噪声水平
        shift_range: 时间序列上下移动的范围
        scale_range: 时间序列缩放的范围
        
    Returns:
        增强后的X和y
    """
    X_aug = X.copy()
    y_aug = y.copy()
    
    # 添加高斯噪声
    noise = np.random.normal(0, noise_level, X_aug.shape)
    X_aug += noise
    
    # 随机上下移动
    shift = np.random.uniform(-shift_range, shift_range, (X_aug.shape[0], 1, X_aug.shape[2]))
    X_aug += shift
    
    # 随机缩放
    scale = np.random.uniform(1-scale_range, 1+scale_range, (X_aug.shape[0], 1, X_aug.shape[2]))
    X_aug *= scale
    
    # 对目标值也进行相应调整
    y_aug += np.random.normal(0, noise_level/2, y_aug.shape)
    
    return X_aug, y_aug


def evaluate_model(model, data_loader):
    model.eval()
    predictions = []
    attention_weights = []  # 存储注意力权重
    with torch.no_grad():
        for inputs, _ in data_loader:
            # 处理不同模型的输出格式
            if isinstance(model, nn.LSTM):
                outputs, _ = model(inputs)
                # 检查输出维度，如果是3D，则取最后一个时间步
                if len(outputs.shape) == 3:
                    outputs = outputs[:, -1, :1]
                else:
                    # 如果已经是2D，则直接取第一列
                    outputs = outputs[:, :1]
                attn = None
            elif "attention" in str(model.__class__).lower():  # 修改这里的类型检查
                outputs, attn = model(inputs)
                if attn is not None:
                    attention_weights.append(attn.cpu().numpy())
            else:
                outputs, _ = model(inputs)
                outputs = outputs[:, -1, :1]  # 对其他模型统一处理
                attn = None
                
            predictions.append(outputs.cpu().numpy())
    
    # 使用 np.vstack 而不是 extend，保持正确的维度
    return np.vstack(predictions), attention_weights if attention_weights else None