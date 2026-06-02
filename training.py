import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# 导入自定义模块
# 确保这些脚本位于同一目录下，或者已添加到Python路径中
from process_data import prepare_bcg_dataloaders, visualize_bcg_samples
from model import BCGHeartbeatDetector, BCGLoss, BCGDataProcessor
from inference import inference

class Config:
    """配置类，管理训练参数"""
    def __init__(self, args=None):
        """初始化配置"""
        DIR="D:\\Code\\ResearchCode\\BCG_Self_One"
        category = "flatten"
        self.train_data_dir=os.path.join(DIR, "data_label","train","data",category)
        self.train_label_dir=os.path.join(DIR, "data_label","train","label",category)

        self.val_data_dir=os.path.join(DIR, "data_label","test","data",category)
        self.val_label_dir=os.path.join(DIR, "data_label","test","label",category)
        # 路径配置
        # self.train_data_dir = './train/data/'
        # self.train_label_dir = './train/label/'
        # self.val_data_dir = './val/data/'
        # self.val_label_dir = './val/label/'
        self.output_dir = './results'
        self.log_dir = './logs'
        self.model_name = 'bcg_heartbeat_detector'
        
        # 数据配置
        self.batch_size = 32
        self.max_peaks = 130
        self.window_size = 91
        self.signal_length = 6000
        self.num_workers = 4
        
        # 模型配置
        self.d_model = 128
        self.n_heads = 8
        self.n_layers = 2
        self.lstm_hidden = 64
        self.dropout = 0.1
        
        # 训练配置
        self.epochs = 100
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.beta = 0.5  # 数量约束损失权重
        self.gamma = 0.5  # 稀疏正则损失权重
        self.patience = 20  # 早停耐心值
        self.clip_grad_norm = 1.0  # 梯度裁剪
        
        # 其他配置
        self.seed = 42
        self.save_interval = 5  # 每多少个epoch保存一次checkpoint
        self.eval_interval = 1  # 每多少个epoch进行一次验证
        self.log_interval = 10  # 每多少个batch记录一次日志
        self.use_gpu = True
        self.gpu_id = 0
        self.resume = False  # 是否从checkpoint恢复训练
        self.checkpoint_path = None  # checkpoint路径
        
        # 从命令行参数更新配置（如果提供）
        if args:
            self.__dict__.update(vars(args))
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 设备设置
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu_id}')
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False
        # 输出 gpu 型号
        if self.use_gpu:
            gpu_name = torch.cuda.get_device_name(self.gpu_id)
            print(f"使用GPU: {gpu_name}")
        else:
            print("使用CPU")
    
    def save(self, path):
        """保存配置到JSON文件"""
        config_dict = {k: str(v) if isinstance(v, torch.device) else v for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path):
        """从JSON文件加载配置"""
        config = cls()
        with open(path, 'r') as f:
            config.__dict__.update(json.load(f))
        return config
    
    def __str__(self):
        """返回配置的字符串表示"""
        config_dict = {k: str(v) if isinstance(v, torch.device) else v for k, v in self.__dict__.items()}
        return json.dumps(config_dict, indent=2)

class Trainer:
    """训练器类，管理模型训练过程"""
    def __init__(self, config):
        """
        初始化训练器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.device
        self.epoch = 0
        self.global_step = 0
        
        # 设置随机种子
        self._set_seed(config.seed)
        
        # 初始化TensorBoard写入器
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # 初始化数据加载器
        self._init_dataloaders()
        
        # 初始化模型
        self._init_model()
        
        # 初始化优化器、调度器和损失函数
        self._init_training_components()
        
        # 创建数据处理器（用于推理）
        self.processor = BCGDataProcessor(
            max_peaks=config.max_peaks,
            window_size=config.window_size,
            signal_length=config.signal_length
        )
        
        # 训练统计
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.patience_counter = 0
        
        # 从checkpoint恢复（如果需要）
        if config.resume and config.checkpoint_path:
            self._resume_from_checkpoint()
    
    def _set_seed(self, seed):
        """设置随机种子以确保可重复性"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _init_dataloaders(self):
        """初始化数据加载器"""
        print("初始化数据加载器...")
        self.train_loader, self.val_loader = prepare_bcg_dataloaders(
            train_data_dir=self.config.train_data_dir,
            train_label_dir=self.config.train_label_dir,
            val_data_dir=self.config.val_data_dir,
            val_label_dir=self.config.val_label_dir,
            batch_size=self.config.batch_size,
            max_peaks=self.config.max_peaks,
            window_size=self.config.window_size,
            signal_length=self.config.signal_length,
            seed=self.config.seed,
            num_workers=self.config.num_workers
        )
        print(f"训练样本数: {len(self.train_loader.dataset)}")
        print(f"验证样本数: {len(self.val_loader.dataset)}")
    
    def _init_model(self):
        """初始化模型"""
        print("初始化模型...")
        self.model = BCGHeartbeatDetector(
            window_size=self.config.window_size,
            max_peaks=self.config.max_peaks,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            lstm_hidden=self.config.lstm_hidden
        ).to(self.device)
        
        # 打印模型结构
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"模型参数数量: {num_params:,}")
        
        # 记录模型结构到TensorBoard
        dummy_input = (
            torch.randn(1, self.config.max_peaks, self.config.window_size).to(self.device),
            torch.rand(1, self.config.max_peaks).to(self.device),
            torch.ones(1, self.config.max_peaks).to(self.device)
        )
        self.writer.add_graph(self.model, dummy_input)
    
    def _init_training_components(self):
        """初始化优化器、学习率调度器和损失函数"""
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 重启周期
            T_mult=2,  # 周期倍乘因子
            eta_min=1e-6  # 最小学习率
        )
        
        # 损失函数
        self.loss_fn = BCGLoss(beta=self.config.beta, gamma=self.config.gamma)
    
    def _resume_from_checkpoint(self):
        """从checkpoint恢复训练"""
        print(f"从checkpoint恢复训练: {self.config.checkpoint_path}")
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
        
        # 恢复模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复优化器和调度器状态
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_mae = checkpoint.get('best_val_mae', float('inf'))
        
        print(f"恢复训练从epoch {self.epoch}，全局步数 {self.global_step}")
    
    def _save_checkpoint(self, path, is_best=False):
        """保存训练checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'config': self.config.__dict__
        }
        
        # 保存checkpoint
        torch.save(checkpoint, path)
        
        # 如果是最佳模型，保存一个副本
        if is_best:
            best_path = os.path.join(os.path.dirname(path), 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到 {best_path}")
    
    def _log_metrics(self, metrics, phase):
        """记录指标到TensorBoard"""
        for key, value in metrics.items():
            self.writer.add_scalar(f"{phase}/{key}", value, self.global_step)
    
    def _log_learning_rate(self):
        """记录当前学习率"""
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(f"lr/group_{i}", param_group['lr'], self.global_step)
    
    def _log_peak_visualizations(self, num_samples=3):
        """记录峰检测可视化结果到TensorBoard"""
        self.model.eval()
        
        # 从验证集中随机选择样本
        sample_indices = np.random.choice(len(self.val_loader.dataset), num_samples, replace=False)
        
        for idx in sample_indices:
            sample = self.val_loader.dataset[idx]
            signal = sample['signal']
            hr_target = sample['hr']
            
            # 推理
            selected_peaks, hr_pred = inference(self.model, sample, self.device)
            
            # 创建可视化图像
            fig = plt.figure(figsize=(10, 5))
            plt.plot(signal, label='BCG Signal')
            plt.scatter(selected_peaks, signal[selected_peaks], color='red', s=50, label='Detected Peaks')
            plt.title(f'Detected Peaks (Pred HR: {hr_pred:.1f}, True HR: {hr_target:.1f}, Predict Peaks: {len(selected_peaks)})')
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # 记录到TensorBoard
            self.writer.add_figure(f"peak_detection/sample_{idx}_{self.global_step}", fig, self.global_step)
            plt.close(fig)
    
    def _log_histogram(self):
        """记录模型参数直方图"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(f"parameters/{name}", param.data, self.global_step)
    
    def train_epoch(self):
        """训练一个轮次"""
        self.model.train()
        epoch_loss = 0.0
        epoch_hr_loss = 0.0
        epoch_count_loss = 0.0
        epoch_sparse_loss = 0.0
        epoch_hr_mae = 0.0
        
        start_time = time.time()
        
        # 进度条
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}/{self.config.epochs} [Train]")
        
        for batch_idx, (segments, positions, masks, hr_targets) in enumerate(pbar):
            # 将数据移至设备
            segments = segments.to(self.device)
            positions = positions.to(self.device)
            masks = masks.to(self.device)
            hr_targets = hr_targets.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            peak_probs, hr_preds = self.model(segments, positions, masks)
            
            # 计算损失
            loss, hr_loss, count_loss, sparse_loss = self.loss_fn(
                peak_probs, hr_preds, hr_targets, masks
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.clip_grad_norm
                )
            
            # 更新参数
            self.optimizer.step()
            
            # 计算MAE
            hr_mae = torch.abs(hr_preds - hr_targets).mean().item()
            
            # 累计损失
            epoch_loss += loss.item()
            epoch_hr_loss += hr_loss.item()
            epoch_count_loss += count_loss.item()
            epoch_sparse_loss += sparse_loss.item()
            epoch_hr_mae += hr_mae
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'hr_mae': f"{hr_mae:.2f}"
            })
            
            # 记录训练批次指标
            if batch_idx % self.config.log_interval == 0:
                # 计算进度百分比
                progress = (self.epoch + batch_idx / len(self.train_loader)) / self.config.epochs
                
                # 记录学习率
                self._log_learning_rate()
                
                # 记录批次指标
                batch_metrics = {
                    'loss': loss.item(),
                    'hr_loss': hr_loss.item(),
                    'count_loss': count_loss.item(),
                    'sparse_loss': sparse_loss.item(),
                    'hr_mae': hr_mae,
                    'progress': progress * 100  # 百分比
                }
                self._log_metrics(batch_metrics, 'train_batch')
            
            # 更新全局步数
            self.global_step += 1
        
        # 计算平均指标
        num_batches = len(self.train_loader)
        epoch_metrics = {
            'loss': epoch_loss / num_batches,
            'hr_loss': epoch_hr_loss / num_batches,
            'count_loss': epoch_count_loss / num_batches,
            'sparse_loss': epoch_sparse_loss / num_batches,
            'hr_mae': epoch_hr_mae / num_batches,
            'epoch_time': time.time() - start_time
        }
        
        # 记录轮次指标
        self._log_metrics(epoch_metrics, 'train_epoch')
        
        # 每轮记录参数直方图
        self._log_histogram()
        
        return epoch_metrics
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        epoch_loss = 0.0
        epoch_hr_loss = 0.0
        epoch_count_loss = 0.0
        epoch_sparse_loss = 0.0
        epoch_hr_mae = 0.0
        
        # 收集预测和真实值
        all_hr_preds = []
        all_hr_targets = []
        
        start_time = time.time()
        
        with torch.no_grad():
            # 进度条
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch}/{self.config.epochs} [Val]")
            
            for segments, positions, masks, hr_targets in pbar:
                # 将数据移至设备
                segments = segments.to(self.device)
                positions = positions.to(self.device)
                masks = masks.to(self.device)
                hr_targets = hr_targets.to(self.device)
                
                # 前向传播
                peak_probs, hr_preds = self.model(segments, positions, masks)
                
                # 计算损失
                loss, hr_loss, count_loss, sparse_loss = self.loss_fn(
                    peak_probs, hr_preds, hr_targets, masks
                )
                
                # 计算MAE
                hr_mae = torch.abs(hr_preds - hr_targets).mean().item()
                
                # 累计损失
                epoch_loss += loss.item()
                epoch_hr_loss += hr_loss.item()
                epoch_count_loss += count_loss.item()
                epoch_sparse_loss += sparse_loss.item()
                epoch_hr_mae += hr_mae
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'hr_mae': f"{hr_mae:.2f}"
                })
                
                # 收集预测和真实值
                all_hr_preds.extend(hr_preds.cpu().numpy())
                all_hr_targets.extend(hr_targets.cpu().numpy())
        
        # 计算平均指标
        num_batches = len(self.val_loader)
        epoch_metrics = {
            'loss': epoch_loss / num_batches,
            'hr_loss': epoch_hr_loss / num_batches,
            'count_loss': epoch_count_loss / num_batches,
            'sparse_loss': epoch_sparse_loss / num_batches,
            'hr_mae': epoch_hr_mae / num_batches,
            'epoch_time': time.time() - start_time
        }
        
        # 计算整体指标
        all_hr_preds = np.array(all_hr_preds)
        all_hr_targets = np.array(all_hr_targets)
        
        # 计算整体MAE和RMSE
        mae = mean_absolute_error(all_hr_targets, all_hr_preds)
        rmse = np.sqrt(mean_squared_error(all_hr_targets, all_hr_preds))
        
        epoch_metrics.update({
            'mae': mae,
            'rmse': rmse
        })
        
        # 记录轮次指标
        self._log_metrics(epoch_metrics, 'val_epoch')
        
        # 绘制回归散点图
        self._plot_regression(all_hr_preds, all_hr_targets)
        
        # 绘制峰检测可视化
        self._log_peak_visualizations()
        
        return epoch_metrics
    
    def _plot_regression(self, hr_preds, hr_targets):
        """绘制回归散点图并记录到TensorBoard"""
        fig = plt.figure(figsize=(8, 8))
        
        # 绘制散点图
        plt.scatter(hr_targets, hr_preds, alpha=0.5)
        
        # 添加对角线（完美预测线）
        min_val = min(min(hr_targets), min(hr_preds))
        max_val = max(max(hr_targets), max(hr_preds))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # 添加标题和标签
        plt.title(f'Heart Rate Prediction vs Ground Truth (Epoch {self.epoch})')
        plt.xlabel('Ground Truth (BPM)')
        plt.ylabel('Prediction (BPM)')
        plt.grid(True)
        
        # 添加指标
        mae = mean_absolute_error(hr_targets, hr_preds)
        rmse = np.sqrt(mean_squared_error(hr_targets, hr_preds))
        plt.annotate(f'MAE: {mae:.2f} BPM\nRMSE: {rmse:.2f} BPM', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        
        # 记录到TensorBoard
        self.writer.add_figure('evaluation/regression', fig, self.global_step)
        plt.close(fig)
    
    def train(self):
        """执行完整的训练循环"""
        print(f"开始训练，总轮次: {self.config.epochs}")
        print(f"使用设备: {self.device}")
        
        # 保存配置
        config_path = os.path.join(self.config.output_dir, 'config.json')
        self.config.save(config_path)
        print(f"保存配置到 {config_path}")
        
        try:
            for epoch in range(self.epoch, self.config.epochs):
                self.epoch = epoch
                print(f"\n===== Epoch {epoch+1}/{self.config.epochs} =====")
                
                # 训练一个轮次
                train_metrics = self.train_epoch()
                print(f"Train - Loss: {train_metrics['loss']:.4f}, HR MAE: {train_metrics['hr_mae']:.2f}")
                
                # 更新学习率
                self.scheduler.step()
                
                # 定期保存checkpoint
                if (epoch + 1) % self.config.save_interval == 0:
                    checkpoint_path = os.path.join(
                        self.config.output_dir, 
                        f"{self.config.model_name}_epoch{epoch+1}.pth"
                    )
                    self._save_checkpoint(checkpoint_path)
                    print(f"保存checkpoint到 {checkpoint_path}")
                
                # 定期验证
                if (epoch + 1) % self.config.eval_interval == 0:
                    val_metrics = self.validate()
                    print(f"Val - Loss: {val_metrics['loss']:.4f}, HR MAE: {val_metrics['hr_mae']:.2f}, "
                          f"MAE: {val_metrics['mae']:.2f}, RMSE: {val_metrics['rmse']:.2f}")
                    
                    # 早停检查
                    if val_metrics['mae'] < self.best_val_mae:
                        self.best_val_loss = val_metrics['loss']
                        self.best_val_mae = val_metrics['mae']
                        
                        # 保存最佳模型
                        best_checkpoint_path = os.path.join(
                            self.config.output_dir, 
                            f"{self.config.model_name}_best.pth"
                        )
                        self._save_checkpoint(best_checkpoint_path, is_best=True)
                        
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.config.patience:
                            print(f"早停: {self.patience_counter} 轮没有改善")
                            break
                
                # 清理内存
                torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            print("训练被用户中断")
        
        finally:
            # 关闭TensorBoard写入器
            self.writer.close()
            
            # 保存最终模型
            final_checkpoint_path = os.path.join(
                self.config.output_dir, 
                f"{self.config.model_name}_final.pth"
            )
            self._save_checkpoint(final_checkpoint_path)
            print(f"保存最终模型到 {final_checkpoint_path}")
            
            # 打印最佳结果
            print("\n===== 训练完成 =====")
            print(f"最佳验证损失: {self.best_val_loss:.4f}")
            print(f"最佳验证MAE: {self.best_val_mae:.2f}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BCG心跳检测训练脚本')
    
    # DIR for default paths
    DIR = "D:\\Code\\ResearchCode\\BCG_Self_One"
    category = "flatten"
    
    # 路径参数
    parser.add_argument('--train_data_dir', type=str, 
                       default=os.path.join(DIR, "data_label","train","data",category),
                       help='训练数据目录')
    parser.add_argument('--train_label_dir', type=str,
                       default=os.path.join(DIR, "data_label","train","label",category),
                       help='训练标签目录')
    parser.add_argument('--val_data_dir', type=str,
                       default=os.path.join(DIR, "data_label","test","data",category),
                       help='验证数据目录')
    parser.add_argument('--val_label_dir', type=str,
                       default=os.path.join(DIR, "data_label","test","label",category),
                       help='验证标签目录')
    parser.add_argument('--output_dir', type=str, default='./results_2', help='输出目录')
    parser.add_argument('--log_dir', type=str, default='./logs_2', help='日志目录')
    parser.add_argument('--model_name', type=str, default='bcg_heartbeat_detector', help='模型名称')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_peaks', type=int, default=150, help='最大候选峰数量')
    parser.add_argument('--window_size', type=int, default=91, help='峰窗口大小')
    parser.add_argument('--signal_length', type=int, default=6000, help='信号长度')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型特征维度')
    parser.add_argument('--n_heads', type=int, default=8, help='多头注意力头数')
    parser.add_argument('--n_layers', type=int, default=2, help='Transformer层数')
    parser.add_argument('--lstm_hidden', type=int, default=64, help='LSTM隐藏层大小')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--beta', type=float, default=0.5, help='数量约束损失权重')
    parser.add_argument('--gamma', type=float, default=0.3, help='稀疏正则损失权重')
    parser.add_argument('--patience', type=int, default=20, help='早停耐心值')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='梯度裁剪')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--save_interval', type=int, default=5, help='保存间隔（轮次）')
    parser.add_argument('--eval_interval', type=int, default=1, help='验证间隔（轮次）')
    parser.add_argument('--log_interval', type=int, default=10, help='日志间隔（批次）')
    parser.add_argument('--no_gpu', action='store_true', help='禁用GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--resume', action='store_true', help='从checkpoint恢复训练')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint路径')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 处理禁用GPU的情况
    if args.no_gpu:
        args.use_gpu = False
    else:
        args.use_gpu = True
    
    # 创建配置
    config = Config(args)
    print("配置:")
    print(config)
    
    # 创建训练器
    trainer = Trainer(config)
    print("训练器初始化完成")
    print("开始训练...")
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()