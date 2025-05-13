# ==============================================
# 训练与评估部分
# ==============================================

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import inference



def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, total_epochs):
    """训练一个轮次"""
    model.train()
    total_loss = 0.0
    hr_loss_sum = 0.0
    count_loss_sum = 0.0
    sparse_loss_sum = 0.0
    hr_mae_sum = 0.0
    
    start_time = time.time()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs} [Train]")
    
    for segments, positions, masks, hr_targets in pbar:
        segments = segments.to(device)
        positions = positions.to(device)
        masks = masks.to(device)
        hr_targets = hr_targets.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        peak_probs, hr_preds = model(segments, positions, masks)
        
        # 计算损失
        loss, hr_loss, count_loss, sparse_loss = loss_fn(peak_probs, hr_preds, hr_targets, masks)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        hr_loss_sum += hr_loss.item()
        count_loss_sum += count_loss.item()
        sparse_loss_sum += sparse_loss.item()
        
        # 计算MAE
        hr_mae = torch.abs(hr_preds - hr_targets).mean().item()
        hr_mae_sum += hr_mae
        
        # 更新进度条信息
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'hr_mae': f"{hr_mae:.2f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
        })
    
    # 计算平均损失
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_hr_loss = hr_loss_sum / num_batches
    avg_count_loss = count_loss_sum / num_batches
    avg_sparse_loss = sparse_loss_sum / num_batches
    avg_hr_mae = hr_mae_sum / num_batches
    
    elapsed_time = time.time() - start_time
    
    print(f"Train Epoch: {epoch}/{total_epochs} | "
          f"Loss: {avg_loss:.4f} | HR MAE: {avg_hr_mae:.2f} | "
          f"Time: {elapsed_time:.2f}s")
    
    return {
        'loss': avg_loss,
        'hr_loss': avg_hr_loss,
        'count_loss': avg_count_loss,
        'sparse_loss': avg_sparse_loss,
        'hr_mae': avg_hr_mae
    }

def validate(model, val_loader, loss_fn, device, epoch, total_epochs):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    hr_loss_sum = 0.0
    count_loss_sum = 0.0
    sparse_loss_sum = 0.0
    hr_mae_sum = 0.0
    
    hr_preds_all = []
    hr_targets_all = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for segments, positions, masks, hr_targets in tqdm(val_loader, desc=f"Epoch {epoch}/{total_epochs} [Val]"):
            segments = segments.to(device)
            positions = positions.to(device)
            masks = masks.to(device)
            hr_targets = hr_targets.to(device)
            
            # 前向传播
            peak_probs, hr_preds = model(segments, positions, masks)
            
            # 计算损失
            loss, hr_loss, count_loss, sparse_loss = loss_fn(peak_probs, hr_preds, hr_targets, masks)
            
            # 累计损失
            total_loss += loss.item()
            hr_loss_sum += hr_loss.item()
            count_loss_sum += count_loss.item()
            sparse_loss_sum += sparse_loss.item()
            
            # 计算MAE
            hr_mae = torch.abs(hr_preds - hr_targets).mean().item()
            hr_mae_sum += hr_mae
            
            # 保存预测和真实值
            hr_preds_all.extend(hr_preds.cpu().numpy())
            hr_targets_all.extend(hr_targets.cpu().numpy())
    
    # 计算平均损失
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_hr_loss = hr_loss_sum / num_batches
    avg_count_loss = count_loss_sum / num_batches
    avg_sparse_loss = sparse_loss_sum / num_batches
    avg_hr_mae = hr_mae_sum / num_batches
    
    # 计算整体指标
    hr_preds_all = np.array(hr_preds_all)
    hr_targets_all = np.array(hr_targets_all)
    
    mae = mean_absolute_error(hr_targets_all, hr_preds_all)
    rmse = np.sqrt(mean_squared_error(hr_targets_all, hr_preds_all))
    
    elapsed_time = time.time() - start_time
    
    print(f"Val Epoch: {epoch}/{total_epochs} | "
          f"Loss: {avg_loss:.4f} | HR MAE: {mae:.2f} | HR RMSE: {rmse:.2f} | "
          f"Time: {elapsed_time:.2f}s")
    
    return {
        'loss': avg_loss,
        'hr_loss': avg_hr_loss,
        'count_loss': avg_count_loss,
        'sparse_loss': avg_sparse_loss,
        'hr_mae': avg_hr_mae,
        'mae': mae,
        'rmse': rmse,
        'hr_preds': hr_preds_all,
        'hr_targets': hr_targets_all
    }

def visualize_peak_detection(model, dataset, indices, device, output_dir):
    """可视化峰检测结果"""
    model.eval()
    
    os.makedirs(os.path.join(output_dir, 'peak_detection'), exist_ok=True)
    
    for idx in indices:
        sample = dataset[idx]
        signal = sample['signal']
        hr_target = sample['hr']
        
        # 推理
        selected_peaks, hr_pred = inference(model, signal, dataset, device)
        
        # 可视化
        plt.figure(figsize=(12, 6))
        
        # 绘制信号
        plt.plot(signal, label='BCG Signal')
        
        # 标记检测到的峰
        plt.scatter(selected_peaks, signal[selected_peaks], color='red', s=50, label='Detected Peaks')
        
        # 标记所有候选峰
        candidate_peaks = sample['original_indices']
        plt.scatter(candidate_peaks, signal[candidate_peaks], color='blue', s=30, alpha=0.5, label='Candidate Peaks')
        
        # 添加标题和标签
        plt.title(f'BCG Peak Detection (Predicted HR: {hr_pred:.2f}, Ground Truth: {hr_target:.2f})')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'peak_detection', f'peak_detection_{idx}.png'))
        plt.close()

def visualize_peak_probabilities(model, dataset, indices, device, output_dir):
    """可视化峰概率"""
    model.eval()
    
    os.makedirs(os.path.join(output_dir, 'peak_probs'), exist_ok=True)
    
    for idx in indices:
        sample = dataset[idx]
        signal = sample['signal']
        hr_target = sample['hr']
        segments = torch.FloatTensor(sample['segments']).unsqueeze(0).to(device)
        positions = torch.FloatTensor(sample['positions']).unsqueeze(0).to(device)
        mask = torch.FloatTensor(sample['mask']).unsqueeze(0).to(device)
        
        # 获取峰概率
        with torch.no_grad():
            peak_probs, hr_pred = model(segments, positions, mask)
        
        peak_probs = peak_probs.squeeze(0).cpu().numpy()
        mask = mask.squeeze(0).cpu().numpy()
        
        # 有效峰索引
        valid_indices = np.where(mask > 0)[0]
        valid_probs = peak_probs[valid_indices]
        
        # 原始峰位置
        original_indices = sample['original_indices']
        
        # 创建峰位置到概率的映射
        peak_idx_to_prob = {}
        for i, idx in enumerate(valid_indices):
            if idx < len(original_indices):
                peak_idx_to_prob[original_indices[idx]] = valid_probs[idx]
        
        # 可视化
        plt.figure(figsize=(12, 8))
        
        # 子图1：信号和峰
        plt.subplot(2, 1, 1)
        plt.plot(signal, label='BCG Signal')
        
        # 绘制带有概率颜色的峰
        for peak_idx, prob in peak_idx_to_prob.items():
            plt.scatter(peak_idx, signal[peak_idx], color=plt.cm.viridis(prob), s=50)
        
        plt.title(f'BCG Signal (Predicted HR: {hr_pred.item():.2f}, Ground Truth: {hr_target:.2f})')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # 子图2：峰概率条形图
        plt.subplot(2, 1, 2)
        peak_indices = list(peak_idx_to_prob.keys())
        peak_probs = list(peak_idx_to_prob.values())
        
        sorted_idx = np.argsort(peak_indices)
        sorted_peaks = [peak_indices[i] for i in sorted_idx]
        sorted_probs = [peak_probs[i] for i in sorted_idx]
        
        plt.bar(range(len(sorted_peaks)), sorted_probs, color=plt.cm.viridis(sorted_probs))
        plt.xticks(range(len(sorted_peaks)), [f"{p}" for p in sorted_peaks], rotation=90)
        plt.title('Peak Probabilities')
        plt.xlabel('Peak Index')
        plt.ylabel('Probability')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'peak_probs', f'peak_probs_{idx}.png'))
        plt.close()

def plot_loss_curves(history, output_dir):
    """绘制损失曲线"""
    plt.figure(figsize=(15, 10))
    
    # 绘制总损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制HR损失
    plt.subplot(2, 2, 2)
    plt.plot(history['train_hr_loss'], label='Training HR Loss')
    plt.plot(history['val_hr_loss'], label='Validation HR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Heart Rate Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制Count损失
    plt.subplot(2, 2, 3)
    plt.plot(history['train_count_loss'], label='Training Count Loss')
    plt.plot(history['val_count_loss'], label='Validation Count Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Count Consistency Loss')
    plt.grid(True)
    plt.legend()
    
    # 绘制Sparse损失
    plt.subplot(2, 2, 4)
    plt.plot(history['train_sparse_loss'], label='Training Sparse Loss')
    plt.plot(history['val_sparse_loss'], label='Validation Sparse Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Sparsity Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    
    # 绘制心率MAE
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_hr_mae'], label='Training HR MAE')
    plt.plot(history['val_hr_mae'], label='Validation HR MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (BPM)')
    plt.title('Heart Rate Mean Absolute Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hr_mae_curve.png'))

def plot_regression(hr_preds, hr_targets, output_dir):
    """绘制回归散点图"""
    plt.figure(figsize=(8, 8))
    
    # 绘制散点图
    plt.scatter(hr_targets, hr_preds, alpha=0.5)
    
    # 添加对角线（完美预测线）
    min_val = min(hr_targets.min(), hr_preds.min())
    max_val = max(hr_targets.max(), hr_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 添加标题和标签
    plt.title('Heart Rate Prediction vs Ground Truth')
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
    plt.savefig(os.path.join(output_dir, 'hr_regression.png'))
