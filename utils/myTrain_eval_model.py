import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pandas as pd
import scipy.stats as stats
import copy
import torch.nn.functional as F
import random
import math
import os
from utils.tcn_xlstm_atten_rl_pytorch2 import ImprovedRLAgent
warnings.filterwarnings("ignore")

class BaseTrainer:
    """基础训练器类"""
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def _init_optimizer(self, learning_rate, weight_decay=0.01):
        """初始化优化器"""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def _init_scheduler(self, optimizer, scheduler_type='cosine', **kwargs):
        """初始化学习率调度器"""
        schedulers = {
            'cosine': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=10,
                T_mult=2,
                eta_min=kwargs.get('min_lr', 1e-6)
            ),
            'cosine_warmup': torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=kwargs.get('max_lr', 0.001),
                steps_per_epoch=kwargs.get('steps_per_epoch', 100),
                epochs=kwargs.get('epochs', 100),
                pct_start=kwargs.get('warmup_epochs', 5) / kwargs.get('epochs', 100)
            ),
            'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=kwargs.get('min_lr', 1e-6)
            )
        }
        return schedulers.get(scheduler_type, schedulers['reduce_on_plateau'])

class OptimizedTrainer(BaseTrainer):
    """优化后的训练器类"""
    def __init__(self, model, device=None, grad_clip_value=1.0):
        super().__init__(model, device)
        self.grad_clip_value = grad_clip_value
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # 初始化优化器
        self.optimizer = None
        self.scheduler = None
        
        # 初始化RL代理
        self.rl_agent = ImprovedRLAgent(
            state_size=model.input_shape[-1],
            action_size=3,  # 增加、减少、保持学习率
            memory_size=10000,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            learning_rate=0.001
        )
    
    def _init_optimizer(self, learning_rate, weight_decay=0.01):
        """初始化优化器"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def _init_scheduler(self, optimizer, scheduler_type='cosine', **kwargs):
        """初始化学习率调度器"""
        if scheduler_type == 'cosine':
            num_training_steps = len(self.train_loader) * self.epochs
            self.scheduler = self._get_cosine_schedule_with_warmup(
                optimizer, 
                num_training_steps
            )
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=5,
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    def _get_cosine_schedule_with_warmup(self, optimizer, num_training_steps):
        def lr_lambda(current_step):
            if current_step < num_training_steps * 0.1:  # 10% warmup
                return float(current_step) / float(max(1, num_training_steps * 0.1))
            progress = float(current_step - num_training_steps * 0.1) / float(max(1, num_training_steps * 0.9))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def add_noise(self, x, noise_level=0.005):
        """添加高斯噪声进行数据增强"""
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    def mixup_data(self, x, y, alpha=0.05):
        """使用mixup进行数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y
    
    def train(self, train_loader, val_loader, epochs=300, learning_rate=0.0003, patience=15):
        # 保存数据加载器供调度器使用
        self.train_loader = train_loader
        self.epochs = epochs
        
        # 初始化优化器和调度器
        self._init_optimizer(learning_rate)
        self._init_scheduler(self.optimizer, 'cosine')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            # 训练阶段
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 数据增强
                if np.random.random() < 0.5:
                    data = self.add_noise(data)
                if np.random.random() < 0.5:
                    data, target = self.mixup_data(data, target)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.mse_loss(output, target)
                
                # 添加L2正则化损失
                if hasattr(self.model, 'get_l2_reg_loss'):
                    loss += self.model.get_l2_reg_loss()
                
                # 检查损失值是否为nan或inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：检测到损失值为nan或inf，使用默认损失值")
                    # 使用一个合理的默认损失值
                    loss = torch.tensor(1.0, device=self.device, requires_grad=True)
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                
                self.optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            
            # 验证阶段
            val_loss = self._validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录训练历史
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(current_lr)
            
            # 使用RL代理调整学习率
            state = torch.mean(data, dim=0).reshape(1, -1)
            action = self.rl_agent.act(state)
            
            old_lr = current_lr
            if action == 0:  # 增加学习率
                current_lr *= 1.1
            elif action == 1:  # 减少学习率
                current_lr *= 0.9
            
            # 更新优化器的学习率
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # 计算奖励（验证损失的改善）
            # 检查验证损失是否为nan，如果是则设置为一个大的正值
            if torch.isnan(torch.tensor(val_loss)) or val_loss > 1e10:
                val_loss = 1e10
                reward = -1.0  # 给予负奖励
            else:
                reward = self.best_val_loss - val_loss
            
            # 更新最佳验证损失
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
            
            # 存储RL经验
            next_state = torch.mean(data, dim=0).reshape(1, -1)
            done = (self.patience_counter >= patience)
            self.rl_agent.remember(state, action, reward, next_state, done)
            
            # 训练RL代理
            self.rl_agent.replay(min(32, len(self.rl_agent.memory)))
            
            # 打印训练信息
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Learning Rate: {current_lr:.6f}')
            print(f'RL Action: {action}, Reward: {reward:.4f}')
            
            # 早停
            if self.patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        return self.training_history, self.model
    
    def _validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.mse_loss(output, target)
                
                # 检查损失值是否为nan或inf
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告：验证过程中检测到损失值为nan或inf，使用默认损失值")
                    loss = torch.tensor(1.0, device=self.device)
                
                val_loss += loss.item()
                val_steps += 1
        
        # 确保验证损失不是nan
        if np.isnan(val_loss) or np.isinf(val_loss) or val_steps == 0:
            print(f"警告：验证损失为nan或inf，使用默认值")
            return 1.0
        
        return val_loss / val_steps

class ModelEvaluator:
    """模型评估器类"""
    def __init__(self, metrics=None):
        self.metrics = metrics or ['mse', 'rmse', 'mae', 'mape', 'r2', 'direction_accuracy']
    
    def evaluate(self, y_true, y_pred):
        """评估模型性能"""
        results = {}
        
        if 'mse' in self.metrics:
            results['mse'] = mean_squared_error(y_true, y_pred)
        if 'rmse' in self.metrics:
            results['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        if 'mae' in self.metrics:
            results['mae'] = mean_absolute_error(y_true, y_pred)
        if 'mape' in self.metrics:
            results['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        if 'r2' in self.metrics:
            results['r2'] = r2_score(y_true, y_pred)
        if 'direction_accuracy' in self.metrics:
            results['direction_accuracy'] = self._calculate_directional_accuracy(y_true, y_pred)
        
        return results
    
    def _calculate_directional_accuracy(self, y_true, y_pred, last_n_days=None):
        """计算方向准确率"""
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        true_diff = np.diff(y_true_flat)
        pred_diff = np.diff(y_pred_flat)
        
        true_direction = np.sign(true_diff)
        pred_direction = np.sign(pred_diff)
        
        if last_n_days is not None:
            true_direction = true_direction[-last_n_days:]
            pred_direction = pred_direction[-last_n_days:]
        
        valid_comparisons_mask = true_direction != 0
        if np.sum(valid_comparisons_mask) == 0:
            return 0.0
        
        correct_predictions = np.sum(true_direction[valid_comparisons_mask] == pred_direction[valid_comparisons_mask])
        total_meaningful_comparisons = np.sum(valid_comparisons_mask)
        
        return (correct_predictions / total_meaningful_comparisons) * 100 if total_meaningful_comparisons > 0 else 0.0
    
    def plot_results(self, y_true, y_pred, dates=None, title=None, save_path=None):
        """绘制预测结果"""
        plt.figure(figsize=(15, 8))
        
        # 确保数据维度匹配
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        if dates is not None:
            dates = np.array(dates)
            # 确保日期数组长度与数据长度匹配
            if len(dates) != len(y_true):
                print(f"警告：日期数组长度({len(dates)})与数据长度({len(y_true)})不匹配，将截断较长的数组")
                min_len = min(len(dates), len(y_true))
                dates = dates[:min_len]
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
            
            plt.plot(dates, y_true, 'b-', label='实际值', linewidth=2)
            plt.plot(dates, y_pred, 'r--', label='预测值', linewidth=2)
            
            if len(dates) > 10:
                step = len(dates) // 10
                # 将numpy.datetime64转换为字符串格式
                date_labels = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates[::step]]
                plt.xticks(dates[::step], date_labels, rotation=45)
            else:
                date_labels = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates]
                plt.xticks(dates, date_labels, rotation=45)
        else:
            plt.plot(y_true, 'b-', label='实际值', linewidth=2)
            plt.plot(y_pred, 'r--', label='预测值', linewidth=2)
        
        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        dir_acc = self._calculate_directional_accuracy(y_true, y_pred)
        
        # 添加评估指标到标题
        if title is None:
            title = "预测结果对比"
        title += f"\nMSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%, R²: {r2:.4f}, 方向准确率: {dir_acc:.2f}%"
        
        plt.title(title)
        plt.xlabel("时间")
        plt.ylabel("价格")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close()

class ModelEnsemble:
    """模型集成类"""
    def __init__(self, models, scaler_label, weights=None):
        self.models = models
        self.weights = weights or [1.0/len(models)] * len(models)
        self.evaluator = ModelEvaluator()
        self.scaler_label = scaler_label
    
    def predict(self, X, smoothing=True, window_size=3):
        """集成预测"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred.cpu().numpy() * weight)
        
        ensemble_pred = np.sum(predictions, axis=0)
        
        if smoothing:
            smoothed_pred = np.zeros_like(ensemble_pred)
            for i in range(len(ensemble_pred)):
                start = max(0, i - window_size // 2)
                end = min(len(ensemble_pred), i + window_size // 2 + 1)
                smoothed_pred[i] = np.mean(ensemble_pred[start:end], axis=0)
            return smoothed_pred
        
        return ensemble_pred
    
    def evaluate(self, X, y_true):
        """评估集成模型性能"""
        predictions = self.predict(X)
        return self.evaluator.evaluate(y_true, predictions)
    
    def predict_with_uncertainty(self, X, num_samples=30, confidence=0.95):
        """带不确定性的预测
        
        参数:
            X: 输入数据
            num_samples: MC Dropout采样次数
            confidence: 置信区间水平
        """
        all_predictions = []
        
        # 启用dropout进行MC采样
        for model in self.models:
            model.train()  # 设置为训练模式以启用dropout
        
        # 进行多次采样
        for _ in range(num_samples):
            predictions = []
            for model, weight in zip(self.models, self.weights):
                with torch.no_grad():
                    pred = model(X)
                    predictions.append(pred.cpu().numpy() * weight)
            
            ensemble_pred = np.sum(predictions, axis=0)
            all_predictions.append(ensemble_pred)
        
        # 恢复模型为评估模式
        for model in self.models:
            model.eval()
        
        # 计算预测的均值和标准差
        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        
        # 计算每个时间步的标准差
        std_pred = np.std(all_predictions, axis=0)
        
        # 计算置信区间
        z_score = stats.norm.ppf((1 + confidence) / 2)  # 获取对应置信水平的z值
        
        # 使用更激进的标准差调整
        adjusted_std = std_pred * np.sqrt(len(self.models)) * 2.0  # 增加不确定性
        
        # 计算上下界
        lower_bound = mean_pred - z_score * adjusted_std
        upper_bound = mean_pred + z_score * adjusted_std
        
        # 将均值、上下界转换回原始尺度
        if hasattr(self, 'scaler_label'):
            mean_pred_original = self.scaler_label.inverse_transform(mean_pred)
            lower_bound_original = self.scaler_label.inverse_transform(lower_bound)
            upper_bound_original = self.scaler_label.inverse_transform(upper_bound)
        else:
            mean_pred_original = mean_pred
            lower_bound_original = lower_bound
            upper_bound_original = upper_bound
            
        # 使用更合理的边界限制，基于反归一化后的均值
        min_val = np.min(mean_pred_original) * 0.8  # 调整比例，更宽松一些
        max_val = np.max(mean_pred_original) * 1.2  # 调整比例，更宽松一些
        
        lower_bound_original = np.clip(lower_bound_original, min_val, max_val)
        upper_bound_original = np.clip(upper_bound_original, min_val, max_val)
        
        return mean_pred_original, lower_bound_original, upper_bound_original # 返回反归一化后的结果

def prepare_data_loader(X, y, batch_size=32, shuffle=True):
    """准备数据加载器"""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True
    )

def to_numpy(tensor):
    """将PyTorch张量转换为NumPy数组"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def evaluate_forecasts(Y_test, predicted_data, n_out):
    """评估预测性能
    
    参数:
        Y_test: 实际值
        predicted_data: 预测值
        n_out: 预测步长
    
    返回:
        评估指标字典
    """
    # 确保数据是numpy数组
    Y_test = to_numpy(Y_test)
    predicted_data = to_numpy(predicted_data)
    
    # 计算各种评估指标
    mse = mean_squared_error(Y_test, predicted_data)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, predicted_data)
    mape = np.mean(np.abs((Y_test - predicted_data) / Y_test)) * 100
    r2 = r2_score(Y_test, predicted_data)
    
    # 计算方向准确率
    true_diff = np.diff(Y_test.flatten())
    pred_diff = np.diff(predicted_data.flatten())
    true_direction = np.sign(true_diff)
    pred_direction = np.sign(pred_diff)
    direction_accuracy = np.mean(true_direction == pred_direction) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }

def save_model(model, file_path):
    """保存模型到文件
    
    参数:
        model: PyTorch模型
        file_path: 保存路径
    """
    # 创建保存目录
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存模型状态
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': {
            'input_shape': getattr(model, 'input_shape', None),
            'output_size': getattr(model, 'output_size', None),
            'dropout_rate': getattr(model, 'dropout_rate', 0.2),
            'l2_reg': getattr(model, 'l2_reg', 0.001)
        }
    }, file_path)
    print(f"模型已保存到: {file_path}")

def load_model(file_path, device=torch.device('cpu')):
    """从文件加载模型
    
    参数:
        file_path: 模型文件路径
        device: 设备
    
    返回:
        加载的模型
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到模型文件: {file_path}")
    
    # 加载模型状态
    checkpoint = torch.load(file_path, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint['model_config']
    
    # 创建模型实例
    if checkpoint['model_class'] == 'xLSTMAttentionRL':
        from utils.tcn_xlstm_atten_rl_pytorch2 import xLSTMAttentionRL
        model = xLSTMAttentionRL(
            input_shape=model_config['input_shape'],
            output_size=model_config['output_size'],
            dropout_rate=model_config['dropout_rate'],
            l2_reg=model_config['l2_reg']
        )
    else:
        raise ValueError(f"不支持的模型类型: {checkpoint['model_class']}")
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"模型已从 {file_path} 加载")
    return model