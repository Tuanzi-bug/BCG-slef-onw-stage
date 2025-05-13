import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 导入自定义模块
from process_data import BCGDataLoader
from model import BCGHeartbeatDetector, BCGLoss, non_max_suppression


def inference(model, sample_data, device):
    """
    使用训练好的模型进行推理（修改版）
    
    Args:
        model: 训练好的模型
        sample_data: 由BCGDataLoader处理后的样本数据字典
        device: 设备
        
    Returns:
        selected_peaks: 选择的峰索引
        hr_pred: 预测的心率
    """
    model.eval()
    
    # 从sample_data中提取需要的数据
    segments = torch.FloatTensor(sample_data['segments']).unsqueeze(0).to(device)
    positions = torch.FloatTensor(sample_data['positions']).unsqueeze(0).to(device)
    mask = torch.FloatTensor(sample_data['mask']).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        peak_probs, hr_pred = model(segments, positions, mask)
    
    # 将张量移回CPU并转换为NumPy数组
    peak_probs = peak_probs.squeeze(0).cpu().numpy()
    hr_pred = hr_pred.item()
    mask_np = mask.squeeze(0).cpu().numpy()
    
    # 根据预测心率选择峰
    k = round(hr_pred)
    
    # 给出有效索引（通过掩码）
    valid_indices = np.where(mask_np > 0)[0]
    
    # 获取这些索引对应的概率
    valid_probs = peak_probs[valid_indices]
    
    # 选择概率最高的k个峰
    if len(valid_probs) > k:
        top_k_indices = np.argsort(valid_probs)[-k:]
        selected_valid_indices = valid_indices[top_k_indices]
    else:
        selected_valid_indices = valid_indices
    
    # 映射回原始信号中的索引
    original_indices = sample_data['original_indices']
    selected_peaks = []
    for i in selected_valid_indices:
        if i < len(original_indices):
            selected_peaks.append(original_indices[i])
    
    # 非极大值抑制（NMS）
    if len(selected_peaks) > 0:
        selected_peaks = non_max_suppression(selected_peaks, peak_probs[selected_valid_indices], min_distance=20)
    
    return selected_peaks, hr_pred


class SingleBCGDataset(torch.utils.data.Dataset):
    """单个BCG信号的数据集，用于推理"""
    def __init__(self, signal_path, max_peaks=100, window_size=91, signal_length=6000):
        """
        初始化单样本数据集
        
        Args:
            signal_path: BCG信号文件路径
            max_peaks: 最大候选峰数量
            window_size: 峰周围截取的窗口大小
            signal_length: 信号总长度
        """
        # 创建一个临时的BCGDataLoader来处理单个信号
        self.processor = BCGDataLoader(
            data_dir="",  # 不会被使用
            label_dir="",  # 不会被使用
            transform=False,
            max_peaks=max_peaks,
            window_size=window_size,
            signal_length=signal_length
        )
        
        # 加载信号
        if signal_path.endswith('.npy'):
            self.signal = np.load(signal_path)
        else:
            raise ValueError(f"Unsupported file format: {signal_path}")
        
        # 确保信号长度正确
        if len(self.signal) != signal_length:
            if len(self.signal) > signal_length:
                self.signal = self.signal[:signal_length]
            else:
                padded_signal = np.zeros(signal_length)
                padded_signal[:len(self.signal)] = self.signal
                self.signal = padded_signal
        
        # 处理信号
        self.processed_data = self.processor.process_signal(self.signal)
        self.processed_data['hr'] = 0  # 单文件推理时没有标签
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("Single sample dataset only has one sample")
        return self.processed_data


class InferenceConfig:
    """配置类，管理推理参数"""
    def __init__(self, args=None):
        # 路径配置
        self.test_data_dir = './test/data/'
        self.test_label_dir = './test/label/'
        self.model_path = './results/bcg_heartbeat_detector_best.pth'
        self.output_dir = './inference_results'
        self.log_dir = './inference_logs'
        
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
        
        # 推理配置
        self.save_predictions = True
        self.visualize_samples = 10
        self.use_gpu = True
        self.gpu_id = 0
        
        # 从命令行参数更新配置
        if args:
            self.__dict__.update(vars(args))
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        
        # 设备设置
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu_id}')
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False
    
    @classmethod
    def from_training_checkpoint(cls, checkpoint_path, args=None):
        """从训练checkpoint加载配置"""
        config = cls(args)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'config' in checkpoint:
            for key in ['d_model', 'n_heads', 'n_layers', 'lstm_hidden', 
                       'dropout', 'max_peaks', 'window_size', 'signal_length']:
                if key in checkpoint['config']:
                    setattr(config, key, checkpoint['config'][key])
        
        return config
    
    def save(self, path):
        """保存配置到JSON文件"""
        config_dict = {k: str(v) if isinstance(v, torch.device) else v for k, v in self.__dict__.items()}
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def __str__(self):
        """返回配置的字符串表示"""
        config_dict = {k: str(v) if isinstance(v, torch.device) else v for k, v in self.__dict__.items()}
        return json.dumps(config_dict, indent=2)


class Predictor:
    """推理器类，管理模型推理过程"""
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # 初始化TensorBoard写入器
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # 加载模型
        self._load_model()
        
        # 加载测试数据（如果不是单文件推理）
        if hasattr(config, 'test_data_dir') and config.test_data_dir:
            self._load_test_data()
    
    def _load_model(self):
        """加载预训练模型"""
        print(f"加载模型: {self.config.model_path}")
        
        # 创建模型实例
        self.model = BCGHeartbeatDetector(
            window_size=self.config.window_size,
            max_peaks=self.config.max_peaks,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            lstm_hidden=self.config.lstm_hidden
        ).to(self.device)
        
        # 加载预训练权重
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 设置为评估模式
        self.model.eval()
        
        print("模型加载完成")
    
    def _load_test_data(self):
        """加载测试数据"""
        print("加载测试数据...")
        
        # 创建测试数据集
        self.test_dataset = BCGDataLoader(
            data_dir=self.config.test_data_dir,
            label_dir=self.config.test_label_dir,
            transform=True,
            max_peaks=self.config.max_peaks,
            window_size=self.config.window_size,
            signal_length=self.config.signal_length
        )
        
        # 创建数据加载器
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self._collate_fn
        )
        
        print(f"测试样本数: {len(self.test_dataset)}")
    
    def _collate_fn(self, batch):
        """自定义整理函数"""
        segments = np.stack([item['segments'] for item in batch])
        positions = np.stack([item['positions'] for item in batch])
        masks = np.stack([item['mask'] for item in batch])
        hrs = np.array([item['hr'] for item in batch])
        
        # 保存原始信号和峰索引用于可视化
        signals = [item['signal'] for item in batch]
        original_indices = [item['original_indices'] for item in batch]
        
        # 转换为张量
        segments = torch.FloatTensor(segments)
        positions = torch.FloatTensor(positions)
        masks = torch.FloatTensor(masks)
        hrs = torch.FloatTensor(hrs)
        
        return segments, positions, masks, hrs, signals, original_indices
    
    def run_inference(self):
        """执行模型推理"""
        print("开始推理...")
        
        # 收集预测结果
        all_hr_preds = []
        all_hr_targets = []
        all_peak_indices = []
        all_signals = []
        
        # 记录推理时间
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (segments, positions, masks, hr_targets, signals, batch_original_indices) in enumerate(tqdm(self.test_loader)):
                # 将数据移至设备
                segments = segments.to(self.device)
                positions = positions.to(self.device)
                masks = masks.to(self.device)
                hr_targets = hr_targets.to(self.device)
                
                # 记录开始时间
                start_time = time.time()
                
                # 进行批次推理
                peak_probs, hr_preds = self.model(segments, positions, masks)
                
                # 记录推理时间
                end_time = time.time()
                batch_inference_time = end_time - start_time
                inference_times.append(batch_inference_time)
                
                # 收集批次结果
                batch_hr_preds = hr_preds.cpu().numpy()
                batch_hr_targets = hr_targets.cpu().numpy()
                
                all_hr_preds.extend(batch_hr_preds)
                all_hr_targets.extend(batch_hr_targets)
                
                # 对每个样本进行峰选择
                batch_peak_probs = peak_probs.cpu().numpy()
                batch_masks = masks.cpu().numpy()
                
                batch_peaks = []
                for i in range(len(batch_hr_preds)):
                    # 构建样本数据用于调用inference函数
                    sample_data = {
                        'segments': segments[i].cpu().numpy(),
                        'positions': positions[i].cpu().numpy(),
                        'mask': batch_masks[i],
                        'original_indices': batch_original_indices[i]
                    }
                    
                    # 使用inference函数获取峰位置
                    selected_peaks, _ = inference(self.model, sample_data, self.device)
                    batch_peaks.append(selected_peaks)
                
                all_peak_indices.extend(batch_peaks)
                all_signals.extend(signals)
        
        # 计算平均推理时间
        avg_inference_time = np.mean(inference_times)
        print(f"平均每批次推理时间: {avg_inference_time:.4f} 秒")
        
        # 转换为NumPy数组
        all_hr_preds = np.array(all_hr_preds)
        all_hr_targets = np.array(all_hr_targets)
        
        # 计算性能指标
        results = self._calculate_metrics(all_hr_preds, all_hr_targets)
        results['avg_inference_time'] = avg_inference_time
        
        # 记录到TensorBoard
        self._log_results(results, all_hr_preds, all_hr_targets, all_peak_indices, all_signals)
        
        # 保存预测结果
        if self.config.save_predictions:
            self._save_predictions(all_hr_preds, all_hr_targets, all_peak_indices)
        
        # 可视化部分样本
        if self.config.visualize_samples > 0:
            self._visualize_samples(all_signals, all_peak_indices, all_hr_preds, all_hr_targets)
        
        return results
    
    def run_single_inference(self, signal_path):
        """对单个文件进行推理"""
        print(f"对单个文件进行推理: {signal_path}")
        name=os.path.basename(signal_path).split('.')[0]
        # 创建单样本数据集
        single_dataset = SingleBCGDataset(
            signal_path=signal_path,
            max_peaks=self.config.max_peaks,
            window_size=self.config.window_size,
            signal_length=self.config.signal_length
        )
        
        # 获取处理后的数据
        sample_data = single_dataset[0]
        
        # 推理
        selected_peaks, hr_pred = inference(self.model, sample_data, self.device)
        
        # 打印结果
        print(f"预测心率: {hr_pred:.2f} BPM")
        print(f"检测到的峰索引: {selected_peaks}")
        
        # 可视化结果
        signal = sample_data['signal']
        plt.figure(figsize=(12, 6))
        plt.plot(signal)
        if len(selected_peaks) > 0:
            plt.scatter(selected_peaks, signal[selected_peaks], color='red', s=50)
        plt.title(f'Detected Peaks (Predicted HR: {hr_pred:.2f} BPM)')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        
        # 保存图像
        save_path = os.path.join(self.config.output_dir, name + '.png')
        plt.savefig(save_path)
        print(f"结果图像已保存到: {save_path}")
        plt.close()
        
        return selected_peaks, hr_pred
    
    def _calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        
        rel_error = np.abs(predictions - targets) / targets
        mre = np.mean(rel_error) * 100
        
        ss_total = np.sum((targets - np.mean(targets)) ** 2)
        ss_residual = np.sum((targets - predictions) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        
        accuracy_5 = np.mean(rel_error <= 0.05) * 100
        accuracy_10 = np.mean(rel_error <= 0.1) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mre': mre,
            'r2': r2,
            'accuracy_5': accuracy_5,
            'accuracy_10': accuracy_10
        }
    
    def _log_results(self, results, predictions, targets, peak_indices, signals):
        """记录结果到TensorBoard"""
        for key, value in results.items():
            self.writer.add_scalar(f'metrics/{key}', value)
        
        # 绘制回归散点图
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5)
        
        min_val = min(min(targets), min(predictions))
        max_val = max(max(targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('Heart Rate Prediction vs Ground Truth')
        plt.xlabel('Ground Truth (BPM)')
        plt.ylabel('Prediction (BPM)')
        plt.grid(True)
        
        plt.annotate(f"MAE: {results['mae']:.2f} BPM\n"
                     f"RMSE: {results['rmse']:.2f} BPM\n"
                     f"R²: {results['r2']:.3f}", 
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        self.writer.add_figure('evaluation/regression', fig)
        plt.close(fig)
    
    def _save_predictions(self, predictions, targets, peak_indices):
        """保存预测结果"""
        results = []
        for i in range(len(predictions)):
            peaks = peak_indices[i]
            if isinstance(peaks, np.ndarray):
                peaks = peaks.tolist()
            # For list values, ensure all elements are Python native types
            elif isinstance(peaks, list):
                peaks = [int(x) if isinstance(x, np.integer) else x for x in peaks]
            results.append({
                'sample_idx': i,
                'predicted_hr': float(predictions[i]),
                'target_hr': float(targets[i]),
                'error': float(predictions[i] - targets[i]),
                'relative_error': float(abs(predictions[i] - targets[i]) / targets[i] * 100),
                'detected_peaks': peaks
            })
        
        output_path = os.path.join(self.config.output_dir, 'predictions.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"预测结果已保存到: {output_path}")
    
    def _visualize_samples(self, signals, peak_indices, predictions, targets):
        """可视化部分样本的峰检测结果"""
        num_samples = min(self.config.visualize_samples, len(signals))
        
        if num_samples < len(signals):
            sample_indices = np.random.choice(len(signals), num_samples, replace=False)
        else:
            sample_indices = list(range(num_samples))
        
        for i, idx in enumerate(sample_indices):
            signal = signals[idx]
            peaks = peak_indices[idx]
            pred_hr = predictions[idx]
            target_hr = targets[idx]
            
            # # Convert peaks to list if it's numpy array
            # if isinstance(peaks, np.ndarray):
            #     peaks = peaks.tolist()
            
            # # Ensure peaks is a list
            # if peaks is None:
            #     peaks = []

            fig = plt.figure(figsize=(12, 6))
            
            plt.plot(signal, label='BCG Signal')
            if len(peaks) > 0:
                plt.scatter(peaks, signal[peaks], color='red', s=50, label='Detected Peaks')
            
            plt.title(f'Sample {idx}: Detected Peaks (Pred HR: {pred_hr:.1f}, True HR: {target_hr:.1f})')
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True)
            
            self.writer.add_figure(f'samples/sample_{idx}', fig)
            
            save_path = os.path.join(self.config.output_dir, 'visualizations', f'sample_{idx}.png')
            fig.savefig(save_path)
            plt.close(fig)
        
        print(f"已可视化 {num_samples} 个样本")
    
    def close(self):
        """关闭TensorBoard写入器等资源"""
        self.writer.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BCG心跳检测推理脚本')
    
    # DIR for default paths
    DIR = "D:\\Code\\ResearchCode\\BCG_Self_One"
    category = "flatten"

    parser.add_argument('--test_data_dir', type=str, default=os.path.join(DIR, "data_label","test","data",category), help='测试数据目录')
    parser.add_argument('--test_label_dir', type=str, default=os.path.join(DIR, "data_label","test","label",category), help='测试标签目录')
    parser.add_argument('--model_path', type=str, default="results_2\\bcg_heartbeat_detector_final.pth",help='模型路径')
    parser.add_argument('--output_dir', type=str, default='./inference_results_single', help='输出目录')
    parser.add_argument('--log_dir', type=str, default='./inference_logs_single', help='日志目录')
    
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    
    parser.add_argument('--visualize_samples', type=int, default=10, help='可视化样本数量')
    parser.add_argument('--no_save_predictions', action='store_true', help='不保存预测结果')
    parser.add_argument('--no_gpu', action='store_true', help='禁用GPU')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    
    # 单文件推理
    parser.add_argument('--single_file', type=str, help='要推理的文件路径（包含多条信号）')
    parser.add_argument('--single_label', type=str, help='对应的标签文件路径（可选）')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if args.no_gpu:
        args.use_gpu = False
    else:
        args.use_gpu = True
    
    if args.no_save_predictions:
        args.save_predictions = False
    else:
        args.save_predictions = True
    
    # 创建配置
    if args.model_path:
        config = InferenceConfig.from_training_checkpoint(args.model_path, args)
    else:
        config = InferenceConfig(args)
    
    print("配置:")
    print(config)
    
    # 创建推理器
    predictor = Predictor(config)
    
    try:
        if args.single_file:
            import glob
            single_files = glob.glob(os.path.join(args.single_file, '*.npy'))
            if not single_files:
                print(f"没有找到文件: {args.single_file}")
                return
            # 单文件推理模式
            for single_file in single_files:
                predictor.run_single_inference(single_file)
        else:
            # 批量推理模式
            results = predictor.run_inference()
            
            print("\n===== 推理结果 =====")
            for key, value in results.items():
                print(f"{key}: {value:.4f}")
    
    finally:
        predictor.close()


if __name__ == "__main__":
    main()