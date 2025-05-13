import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class BCGDataLoader(Dataset):
    """加载BCG数据集的类，处理多文件结构"""
    def __init__(self, data_dir, label_dir, distance=45, transform=True, 
                 max_peaks=100, window_size=91, signal_length=6000,
                 seed=42):
        """
        初始化BCG数据加载器
        
        Args:
            data_dir: BCG数据文件目录，包含多个.npy文件
            label_dir: 标签文件目录，包含对应的.npy文件
            distance: 峰之间的最小距离
            transform: 是否进行信号预处理
            max_peaks: 最大候选峰数量
            window_size: 峰周围截取的窗口大小
            signal_length: 信号总长度
            seed: 随机种子，用于数据打乱
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.distance = distance
        self.transform = transform
        self.max_peaks = max_peaks
        self.window_size = window_size
        self.half_window = window_size // 2
        self.signal_length = signal_length
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(seed)
        
        if len(data_dir)!=0:
            # 加载数据和标签
            self.data, self.labels = self._load_all_files()
            print(f"Dataset loaded with {len(self.data)} samples")
    
    def _load_all_files(self):
        """加载所有数据和标签文件，并合并为大数组"""
        # 获取数据目录中的所有.npy文件
        data_files = sorted(glob.glob(os.path.join(self.data_dir, "*.npy")))
        
        if not data_files:
            raise ValueError(f"No .npy files found in {self.data_dir}")
        
        all_data = []
        all_labels = []
        
        print("Loading BCG data files...")
        for data_file in tqdm(data_files):
            # 获取对应的标签文件
            file_name = os.path.basename(data_file)
            label_file = os.path.join(self.label_dir, file_name)
            
            if not os.path.exists(label_file):
                print(f"Warning: No matching label file for {file_name}, skipping.")
                continue
            
            # 加载数据和标签
            data = np.load(data_file)  # 形状(540, 6000)
            labels = np.load(label_file)  # 形状(540, 1)
            
            # 验证形状
            if data.shape[0] != labels.shape[0]:
                print(f"Warning: Sample counts don't match for {file_name}. Data: {data.shape}, Labels: {labels.shape}")
                continue
            
            # 添加到列表
            all_data.append(data)
            all_labels.append(labels)
        
        # 合并所有数据
        if not all_data:
            raise ValueError("No valid data files could be loaded")
        
        merged_data = np.vstack(all_data)  # 形状(40*540, 6000)
        merged_labels = np.vstack(all_labels)  # 形状(40*540, 1)
        
        return merged_data, merged_labels
    
    def _preprocess_signal(self, signal):
        """预处理BCG信号"""
        # # 1. 去除基线漂移（高通滤波）
        # b, a = butter(3, 0.5/50, 'high')  # 0.5 Hz高通滤波器
        # signal = filtfilt(b, a, signal)
        
        # # 2. 去除高频噪声（低通滤波）
        # b, a = butter(3, 20/50, 'low')  # 20 Hz低通滤波器
        # signal = filtfilt(b, a, signal)
        
        # 3. 标准化
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        return signal
    
    def _find_candidate_peaks(self, signal):
        """找到信号中的候选峰"""
        # 使用scipy的find_peaks函数检测峰
        # peak_indices, _ = find_peaks(signal, distance=self.distance)
        import utils
        peak_indices = utils.get_peaks_bestParameters(signal)
        return peak_indices
    
    def _extract_peak_segments(self, signal, peak_indices):
        """截取每个峰周围的子信号"""
        num_peaks = len(peak_indices)
        segments = np.zeros((num_peaks, self.window_size))
        
        for i, peak_idx in enumerate(peak_indices):
            # 计算截取的起始和结束位置
            start_idx = peak_idx - self.half_window
            end_idx = peak_idx + self.half_window + 1
            
            # 处理边界情况
            if start_idx < 0:
                # 左侧需要零填充
                seg_start = 0
                pad_left = -start_idx
                segments[i, pad_left:] = signal[seg_start:end_idx]
            elif end_idx > self.signal_length:
                # 右侧需要零填充
                seg_end = self.signal_length
                pad_right = end_idx - self.signal_length
                segments[i, :(self.window_size-pad_right)] = signal[start_idx:seg_end]
            else:
                # 正常情况，直接截取
                segments[i] = signal[start_idx:end_idx]
        
        return segments
    
    def _normalize_positions(self, peak_indices):
        """将峰的位置归一化"""
        # 归一化位置
        return np.array(peak_indices) / self.signal_length
    
    def _get_positional_encoding(self, positions, d_model=64):
        """生成位置的正余弦编码"""
        num_peaks = len(positions)
        pos_encoding = np.zeros((num_peaks, d_model))
        
        # 计算正余弦位置编码
        for i in range(num_peaks):
            for j in range(0, d_model, 2):
                div_term = np.exp(-(np.log(10000.0) * j) / d_model)
                pos_encoding[i, j] = np.sin(positions[i] * div_term)
                pos_encoding[i, j+1] = np.cos(positions[i] * div_term)
        
        return pos_encoding
    
    def _pad_or_truncate(self, features, positions):
        """将特征和位置信息填充或截断到固定长度"""
        num_peaks = len(positions)
        
        # 创建掩码，1表示有效数据，0表示填充
        mask = np.zeros(self.max_peaks)
        
        if num_peaks >= self.max_peaks:
            # 如果峰数量超过最大值，选择前max_peaks个
            padded_features = features[:self.max_peaks]
            padded_positions = positions[:self.max_peaks]
            mask[:] = 1.0
        else:
            # 如果峰数量不足，进行零填充
            padded_features = np.zeros((self.max_peaks, self.window_size))
            padded_features[:num_peaks] = features
            
            padded_positions = np.zeros(self.max_peaks)
            padded_positions[:num_peaks] = positions
            
            mask[:num_peaks] = 1.0
        
        return padded_features, padded_positions, mask
    
    def process_signal(self, signal):
        """处理单个BCG信号"""
        # 如果需要预处理
        if self.transform:
            signal = self._preprocess_signal(signal)
        
        # 步骤1：寻找候选峰
        peak_indices = self._find_candidate_peaks(signal)
        
        # 步骤2：截取每个峰周围的子信号
        peak_segments = self._extract_peak_segments(signal, peak_indices)
        
        # 步骤3：获取归一化位置
        norm_positions = self._normalize_positions(peak_indices)
        
        # 步骤4：填充或截断到固定长度
        padded_segments, padded_positions, mask = self._pad_or_truncate(peak_segments, norm_positions)
        
        # 步骤5：生成位置编码
        pos_encoding = self._get_positional_encoding(padded_positions)
        
        # 返回处理后的数据
        return {
            'signal': signal,
            'segments': padded_segments,
            'positions': padded_positions,
            'pos_encoding': pos_encoding,
            'mask': mask,
            'original_indices': peak_indices[:min(len(peak_indices), self.max_peaks)]
        }
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        # 获取BCG信号
        signal = self.data[idx]
        
        # 确保信号长度正确
        if len(signal) != self.signal_length:
            # 处理长度不一致的情况
            if len(signal) > self.signal_length:
                signal = signal[:self.signal_length]
            else:
                # 进行零填充
                padded_signal = np.zeros(self.signal_length)
                padded_signal[:len(signal)] = signal
                signal = padded_signal
        
        # 处理信号
        processed = self.process_signal(signal)
        
        # 获取心率标签
        hr = float(self.labels[idx][0])
        
        # 返回处理后的数据和标签
        return {
            'signal': signal,
            'segments': processed['segments'],
            'positions': processed['positions'],
            'pos_encoding': processed['pos_encoding'],
            'mask': processed['mask'],
            'original_indices': processed['original_indices'],
            'hr': hr
        }

def prepare_bcg_dataloaders(train_data_dir, train_label_dir, 
                            val_data_dir, val_label_dir,
                            batch_size=32, transform=False, 
                            max_peaks=150, window_size=91, 
                            signal_length=6000, seed=42,
                            num_workers=4):
    """
    准备BCG数据集的训练和验证数据加载器
    
    Args:
        train_data_dir: 训练数据目录
        train_label_dir: 训练标签目录
        val_data_dir: 验证数据目录
        val_label_dir: 验证标签目录
        batch_size: 批次大小
        transform: 是否进行信号预处理
        max_peaks: 最大候选峰数量
        window_size: 峰周围截取的窗口大小
        signal_length: 信号总长度
        seed: 随机种子
        num_workers: 数据加载线程数
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 创建训练集
    train_dataset = BCGDataLoader(
        data_dir=train_data_dir,
        label_dir=train_label_dir,
        transform=transform,
        max_peaks=max_peaks,
        window_size=window_size,
        signal_length=signal_length,
        seed=seed
    )
    
    # 创建验证集
    val_dataset = BCGDataLoader(
        data_dir=val_data_dir,
        label_dir=val_label_dir,
        transform=transform,
        max_peaks=max_peaks,
        window_size=window_size,
        signal_length=signal_length,
        seed=seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """自定义整理函数，将字典列表转换为批次张量"""
    # 提取每个键的值并创建批次
    signals = np.stack([item['signal'] for item in batch])
    segments = np.stack([item['segments'] for item in batch])
    positions = np.stack([item['positions'] for item in batch])
    masks = np.stack([item['mask'] for item in batch])
    hrs = np.array([item['hr'] for item in batch])
    
    # 转换为张量
    signals = torch.FloatTensor(signals)
    segments = torch.FloatTensor(segments)
    positions = torch.FloatTensor(positions)
    masks = torch.FloatTensor(masks)
    hrs = torch.FloatTensor(hrs)
    
    return segments, positions, masks, hrs

def visualize_bcg_samples(dataset, indices, title=None, save_dir=None):
    """
    可视化BCG样本
    
    Args:
        dataset: BCG数据集
        indices: 要可视化的样本索引列表
        title: 可选的标题前缀
        save_dir: 保存图像的目录，如果为None则直接显示图像
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx in indices:
        sample = dataset[idx]
        signal = sample['signal']
        hr = sample['hr']
        peak_indices = sample['original_indices']
        
        plt.figure(figsize=(12, 6))
        
        # 绘制BCG信号
        plt.plot(signal, label='BCG Signal')
        
        # 标记候选峰
        plt.scatter(peak_indices, signal[peak_indices], color='red', s=50, label='Candidate Peaks')
        
        # 添加标题和标签
        if title:
            plt.title(f"{title}: Sample {idx} (Heart Rate: {hr:.1f} BPM, Peaks: {len(peak_indices)})")
        else:
            plt.title(f"BCG Signal Sample {idx} (Heart Rate: {hr:.1f} BPM, Peaks: {len(peak_indices)})")
        
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f"sample_{idx}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

def explore_dataset_statistics(dataset):
    """
    探索数据集统计信息
    
    Args:
        dataset: BCG数据集
        
    Returns:
        stats: 统计信息字典
    """
    print("Exploring dataset statistics...")
    
    heart_rates = []
    peak_counts = []
    signal_means = []
    signal_stds = []
    
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        heart_rates.append(sample['hr'])
        peak_counts.append(len(sample['original_indices']))
        signal_means.append(np.mean(sample['signal']))
        signal_stds.append(np.std(sample['signal']))
    
    # 计算统计信息
    stats = {
        'hr_min': min(heart_rates),
        'hr_max': max(heart_rates),
        'hr_mean': np.mean(heart_rates),
        'hr_median': np.median(heart_rates),
        'hr_std': np.std(heart_rates),
        'peak_min': min(peak_counts),
        'peak_max': max(peak_counts),
        'peak_mean': np.mean(peak_counts),
        'peak_median': np.median(peak_counts),
        'peak_std': np.std(peak_counts),
        'signal_mean_avg': np.mean(signal_means),
        'signal_std_avg': np.mean(signal_stds)
    }
    
    # 打印统计信息
    print("\nDataset Statistics:")
    print(f"Number of samples: {len(dataset)}")
    print(f"Heart Rate (BPM) - Min: {stats['hr_min']:.1f}, Max: {stats['hr_max']:.1f}, "
          f"Mean: {stats['hr_mean']:.1f}, Median: {stats['hr_median']:.1f}, Std: {stats['hr_std']:.1f}")
    print(f"Peak Counts - Min: {stats['peak_min']}, Max: {stats['peak_max']}, "
          f"Mean: {stats['peak_mean']:.1f}, Median: {stats['peak_median']:.1f}, Std: {stats['peak_std']:.1f}")
    
    # 绘制心率分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(heart_rates, bins=30, alpha=0.7)
    plt.title('Heart Rate Distribution')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 绘制峰数量分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(peak_counts, bins=30, alpha=0.7)
    plt.title('Peak Count Distribution')
    plt.xlabel('Number of Peaks')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 绘制峰数量与心率的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(heart_rates, peak_counts, alpha=0.5)
    plt.title('Heart Rate vs Peak Count')
    plt.xlabel('Heart Rate (BPM)')
    plt.ylabel('Number of Peaks')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return stats

def main():
    """示例主函数"""
    import argparse
    
    # parser = argparse.ArgumentParser(description='BCG数据集加载与处理')
    DIR="D:\\Code\\ResearchCode\\BCG_Self_One"
    category = "flatten"
    train_data_dir=os.path.join(DIR, "data_label","train","data",category)
    train_label_dir=os.path.join(DIR, "data_label","train","label",category)

    val_data_dir=os.path.join(DIR, "data_label","test","data",category)
    val_label_dir=os.path.join(DIR, "data_label","test","label",category)
    # # 数据路径参数
    # parser.add_argument('--train_data_dir', type=str, required=True, help='训练数据目录')
    # parser.add_argument('--train_label_dir', type=str, required=True, help='训练标签目录')
    # parser.add_argument('--val_data_dir', type=str, required=True, help='验证数据目录')
    # parser.add_argument('--val_label_dir', type=str, required=True, help='验证标签目录')
    
    # 数据处理参数
    # parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    # parser.add_argument('--max_peaks', type=int, default=100, help='最大候选峰数量')
    # parser.add_argument('--window_size', type=int, default=91, help='峰窗口大小')
    # parser.add_argument('--signal_length', type=int, default=6000, help='信号长度')
    # parser.add_argument('--seed', type=int, default=42, help='随机种子')
    # parser.add_argument('--visualize', action='store_true', help='是否可视化样本')
    # parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    # parser.add_argument('--explore_stats', action='store_true', help='是否探索数据集统计信息')
    # parser.add_argument('--save_vis_dir', type=str, default=None, help='可视化图像保存目录')
    
    # args = parser.parse_args()
    
    # 准备数据加载器
    train_loader, val_loader = prepare_bcg_dataloaders(
        train_data_dir=train_data_dir,
        train_label_dir=train_label_dir,
        val_data_dir=val_data_dir,
        val_label_dir=val_label_dir,
        # batch_size=args.batch_size,
        # max_peaks=args.max_peaks,
        # window_size=args.window_size,
        # signal_length=args.signal_length,
        # seed=args.seed,
        # num_workers=args.num_workers
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # 获取第一个批次数据
    first_batch = next(iter(train_loader))
    segments, positions, masks, hrs = first_batch
    
    print(f"Batch shapes - Segments: {segments.shape}, Positions: {positions.shape}, "
          f"Masks: {masks.shape}, Heart Rates: {hrs.shape}")
    
    visualize = True
    # 可视化样本（如果请求）
    if visualize:
        save_vis_dir=os.path.join("visualizations")
        # 获取训练集的随机5个样本
        train_indices = random.sample(range(len(train_loader.dataset)), 5)
        visualize_bcg_samples(train_loader.dataset, train_indices, 
                              title="Training Sample", save_dir=save_vis_dir)
        
        # 获取验证集的随机5个样本
        val_indices = random.sample(range(len(val_loader.dataset)), 5)
        visualize_bcg_samples(val_loader.dataset, val_indices, 
                              title="Validation Sample", save_dir=save_vis_dir)
    
    explore_stats=False
    # 探索数据集统计信息（如果请求）
    if explore_stats:
        print("\nExploring training dataset statistics:")
        train_stats = explore_dataset_statistics(train_loader.dataset)
        
        print("\nExploring validation dataset statistics:")
        val_stats = explore_dataset_statistics(val_loader.dataset)


        print(f"Training Stats: {train_stats}")
        print(f"Validation Stats: {val_stats}")

if __name__ == "__main__":
    main()