import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from scipy.signal import find_peaks
from tqdm import tqdm


# 1. 数据预处理部分
class BCGDataProcessor:
    def __init__(self, max_peaks=100, window_size=91, signal_length=6000):
        """
        BCG信号预处理器
        
        Args:
            max_peaks: 最大候选峰数量
            window_size: 峰周围截取的窗口大小（应为奇数）
            signal_length: 信号总长度（1分钟100Hz采样 = 6000点）
        """
        self.max_peaks = max_peaks
        self.window_size = window_size
        self.half_window = window_size // 2
        self.signal_length = signal_length
        
    def find_candidate_peaks(self, signal):
        """
        找到信号中的候选峰
        
        Args:
            signal: 原始BCG信号 [signal_length]
            
        Returns:
            peak_indices: 峰的索引位置列表
        """
        # # 使用scipy的find_peaks函数检测峰
        # peak_indices, _ = find_peaks(signal, distance=45)
        import utils
        peak_indices = utils.get_peaks_bestParameters(signal)
        return peak_indices
    
    def extract_peak_segments(self, signal, peak_indices):
        """
        截取每个峰周围的子信号
        
        Args:
            signal: 原始BCG信号 [signal_length]
            peak_indices: 峰的索引位置列表
            
        Returns:
            segments: 每个峰周围的子信号 [num_peaks, window_size]
        """
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
    
    def normalize_positions(self, peak_indices):
        """
        将峰的位置归一化并创建位置编码
        
        Args:
            peak_indices: 峰的索引位置列表
            
        Returns:
            pos_encoding: 位置编码 [num_peaks, d_model]
        """
        # 归一化位置
        norm_positions = np.array(peak_indices) / self.signal_length
        return norm_positions
    
    def get_positional_encoding(self, positions, d_model=64):
        """
        生成位置的正余弦编码
        
        Args:
            positions: 归一化的位置 [num_peaks]
            d_model: 编码维度
            
        Returns:
            pos_encoding: 位置编码 [num_peaks, d_model]
        """
        num_peaks = len(positions)
        pos_encoding = np.zeros((num_peaks, d_model))
        
        # 计算正余弦位置编码
        for i in range(num_peaks):
            for j in range(0, d_model, 2):
                div_term = np.exp(-(np.log(10000.0) * j) / d_model)
                pos_encoding[i, j] = np.sin(positions[i] * div_term)
                pos_encoding[i, j+1] = np.cos(positions[i] * div_term)
        
        return pos_encoding
    
    def pad_or_truncate(self, features, positions):
        """
        将特征和位置信息填充或截断到固定长度
        
        Args:
            features: 峰特征 [num_peaks, window_size]
            positions: 峰位置 [num_peaks]
            
        Returns:
            padded_features: 填充后的特征 [max_peaks, window_size]
            padded_positions: 填充后的位置 [max_peaks]
            mask: 有效峰的掩码 [max_peaks]
        """
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
    
    def process(self, signal):
        """
        处理单个BCG信号
        
        Args:
            signal: 原始BCG信号 [signal_length]
            
        Returns:
            peak_segments: 峰周围的子信号 [max_peaks, window_size]
            pos_encoding: 位置编码 [max_peaks, d_model]
            mask: 有效峰的掩码 [max_peaks]
            original_indices: 原始峰索引
        """
        # 步骤1：寻找候选峰
        peak_indices = self.find_candidate_peaks(signal)
        
        # 步骤2：截取每个峰周围的子信号
        peak_segments = self.extract_peak_segments(signal, peak_indices)
        
        # 步骤3：获取归一化位置
        norm_positions = self.normalize_positions(peak_indices)
        
        # 步骤4：填充或截断到固定长度
        padded_segments, padded_positions, mask = self.pad_or_truncate(peak_segments, norm_positions)
        
        # 步骤5：生成位置编码
        pos_encoding = self.get_positional_encoding(padded_positions)
        
        # 返回处理后的数据
        return padded_segments, norm_positions, mask, peak_indices[:min(len(peak_indices), self.max_peaks)]

# 2. 模型架构
class PositionalEncoding(nn.Module):
    """优化的Transformer风格位置编码"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # 预计算div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        
    def forward(self, x, positions=None):
        batch_size, seq_len, _ = x.size()
        
        if positions is None:
            positions = torch.arange(0, seq_len, device=x.device).float() / seq_len
            positions = positions.expand(batch_size, seq_len)
        
        # 创建输出张量
        pos_enc = torch.zeros_like(x)
        
        # 向量化计算sin和cos
        pos_expanded = positions.unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # 计算所有偶数位置的sin值
        sin_input = pos_expanded * self.div_term
        pos_enc[:, :, 0::2] = torch.sin(sin_input)
        
        # 计算所有奇数位置的cos值
        pos_enc[:, :, 1::2] = torch.cos(sin_input)
        
        # 应用掩码
        mask = ((positions > 0) | (torch.arange(seq_len, device=x.device).view(1, -1) == 0)).unsqueeze(-1)
        pos_enc = pos_enc * mask.float()
        
        return x + pos_enc
# class PositionalEncoding(nn.Module):
#     """Transformer风格的位置编码"""
#     def __init__(self, d_model, max_len=100):
#         super(PositionalEncoding, self).__init__()
        
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
        
#         self.register_buffer('pe', pe.unsqueeze(0))
        
#     def forward(self, x, positions=None):
#         """
#         Args:
#             x: 输入特征 [batch_size, seq_len, d_model]
#             positions: 归一化位置 [batch_size, seq_len]
            
#         Returns:
#             特征 + 位置编码
#         """
#         if positions is None:
#             return x + self.pe[:, :x.size(1)]
#         else:
#             # 使用指定的位置
#             batch_size, seq_len = positions.shape
#             pos_enc = torch.zeros(batch_size, seq_len, x.size(2), device=x.device)
            
#             for i in tqdm(range(batch_size), desc="Positional Encoding"):
#                 for j in range(seq_len):
#                     pos = positions[i, j]
#                     if pos == 0 and j > 0:  # 假设位置0只会出现在真实序列的开始或者是padding
#                         continue
#                     for k in range(0, x.size(2), 2):
#                         div_term = math.exp(-(math.log(10000.0) * k) / x.size(2))
#                         pos_enc[i, j, k] = math.sin(pos * div_term)
#                         pos_enc[i, j, k+1] = math.cos(pos * div_term)
            
#             return x + pos_enc

class LocalFeatureExtractor(nn.Module):
    """局部特征提取模块（1D-CNN）"""
    def __init__(self, input_size=91, hidden_dims=[32, 64, 128], d_model=128):
        super(LocalFeatureExtractor, self).__init__()
        
        layers = []
        in_channels = 1
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            in_channels = hidden_dim
        
        self.cnn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dims[-1] * input_size, d_model)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征 [batch_size, num_peaks, window_size]
            
        Returns:
            提取的特征 [batch_size, num_peaks, d_model]
        """
        batch_size, num_peaks, window_size = x.size()
        
        # 重塑为CNN可处理的形状
        x = x.view(-1, 1, window_size)  # [batch_size*num_peaks, 1, window_size]
        
        # 通过CNN提取特征
        x = self.cnn(x)  # [batch_size*num_peaks, hidden_dims[-1], window_size]
        
        # 重塑并通过全连接层
        x = x.view(batch_size*num_peaks, -1)  # [batch_size*num_peaks, hidden_dims[-1]*window_size]
        x = self.fc(x)  # [batch_size*num_peaks, d_model]
        
        # 恢复原始批次维度
        x = x.view(batch_size, num_peaks, -1)  # [batch_size, num_peaks, d_model]
        
        return x

class BCGHeartbeatDetector(nn.Module):
    """BCG心跳检测与心率预测模型"""
    def __init__(self, window_size=91, max_peaks=100, d_model=128, n_heads=8, 
                 n_layers=2, dropout=0.1, lstm_hidden=64):
        super(BCGHeartbeatDetector, self).__init__()
        
        # 1. 局部特征提取
        self.feature_extractor = LocalFeatureExtractor(
            input_size=window_size, 
            hidden_dims=[32, 64, 128], 
            d_model=d_model
        )
        
        # 2. 位置编码
        self.positional_encoder = PositionalEncoding(d_model, max_len=max_peaks)
        
        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. 峰概率预测
        self.peak_predictor = nn.Linear(d_model, 1)
        
        # 5. 心率预测
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, lstm_hidden),
            nn.ReLU(),
            nn.Linear(lstm_hidden, 1)
        )
        
    def forward(self, signals, positions, mask=None):
        """
        前向传播
        
        Args:
            signals: 信号片段 [batch_size, num_peaks, window_size]
            positions: 位置编码 [batch_size, num_peaks, d_model] 或 [batch_size, num_peaks]
            mask: 有效峰的掩码 [batch_size, num_peaks]
            
        Returns:
            peak_probs: 峰概率 [batch_size, num_peaks]
            hr_pred: 心率预测 [batch_size]
        """
        batch_size, num_peaks, _ = signals.size()
        
        # 1. 提取局部特征
        local_features = self.feature_extractor(signals)  # [batch_size, num_peaks, d_model]
        # print("local_features.shape:", local_features.shape)
        # print("positions.shape:", positions.shape)
        # 2. 添加位置编码
        if positions.dim() == 2:
            # 如果只给了位置索引，生成完整的位置编码
            features_with_pos = self.positional_encoder(local_features, positions)
        else:
            # 如果已给完整的位置编码，直接相加
            features_with_pos = local_features + positions
        
        # 3. 创建注意力掩码（必要时）
        attn_mask = None
        if mask is not None:
            # 创建对应的掩码 (1表示padding应该被掩盖)
            attn_mask = (1 - mask).bool()  # [batch_size, num_peaks]
        
        # 4. Transformer编码器处理
        # Transformer需要序列维度在前
        features_with_pos = features_with_pos.transpose(0, 1)  # [num_peaks, batch_size, d_model]
        # print("features_with_pos.shape:", features_with_pos.shape)
        if attn_mask is not None:
            context_features = self.transformer_encoder(features_with_pos, src_key_padding_mask=attn_mask)
        else:
            context_features = self.transformer_encoder(features_with_pos)
            
        # 恢复原始维度顺序
        context_features = context_features.transpose(0, 1)  # [batch_size, num_peaks, d_model]
        
        # 5. 峰概率预测
        peak_logits = self.peak_predictor(context_features).squeeze(-1)  # [batch_size, num_peaks]
        peak_probs = torch.sigmoid(peak_logits)
        
        # 应用掩码（如果提供）
        if mask is not None:
            peak_probs = peak_probs * mask
        
        # 6. 加权特征融合
        weighted_features = context_features * peak_probs.unsqueeze(-1)
        
        # 7. 心率预测
        lstm_out, _ = self.lstm(weighted_features)  # [batch_size, num_peaks, lstm_hidden*2]
        
        # 全局平均池化
        global_feature = lstm_out.mean(dim=1)  # [batch_size, lstm_hidden*2]
        
        # 预测心率
        hr_pred = self.fc(global_feature).squeeze(-1)  # [batch_size]
        
        return peak_probs, hr_pred

# 3. 损失函数
class BCGLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, beta=1.0, gamma=0.1):
        super(BCGLoss, self).__init__()
        self.beta = beta    # 数量约束权重
        self.gamma = gamma  # 稀疏正则权重
        self.hr_loss_fn = nn.SmoothL1Loss()
        
    def sparse_loss(self, peak_probs):
        """稀疏正则损失"""
        epsilon = 1e-10  # 防止log(0)
        # 使用二元熵促进稀疏性 (-p*log(p) - (1-p)*log(1-p))
        entropy = -(peak_probs * torch.log(peak_probs + epsilon) + 
                    (1 - peak_probs) * torch.log(1 - peak_probs + epsilon))
        
        return entropy.mean()
    
    def count_consistency_loss(self, peak_probs, hr_targets):
        """数量约束损失"""
        # 概率和应接近心率
        peak_sum = peak_probs.sum(dim=1)  # [batch_size]
        return F.mse_loss(peak_sum, hr_targets)
    
    def forward(self, peak_probs, hr_preds, hr_targets, mask=None):
        """
        计算总损失
        
        Args:
            peak_probs: 峰概率 [batch_size, num_peaks]
            hr_preds: 心率预测 [batch_size]
            hr_targets: 心率真值 [batch_size]
            mask: 有效峰的掩码 [batch_size, num_peaks]
            
        Returns:
            总损失和各部分损失
        """
        # 1. 心率预测损失
        hr_loss = self.hr_loss_fn(hr_preds, hr_targets)
        
        # 应用掩码（如果提供）
        if mask is not None:
            masked_probs = peak_probs * mask
        else:
            masked_probs = peak_probs
        
        # 2. 数量约束损失
        count_loss = self.count_consistency_loss(masked_probs, hr_targets)
        
        # 3. 稀疏正则损失
        if mask is not None:
            # 只对有效部分计算稀疏损失
            valid_probs = masked_probs[mask > 0]
            if len(valid_probs) > 0:
                sparse_loss = self.sparse_loss(valid_probs)
            else:
                sparse_loss = torch.tensor(0.0, device=peak_probs.device)
        else:
            sparse_loss = self.sparse_loss(peak_probs)
        
        # 4. 总损失
        total_loss = hr_loss + self.beta * count_loss + self.gamma * sparse_loss
        
        return total_loss, hr_loss, count_loss, sparse_loss


def non_max_suppression(peak_indices, peak_probs, min_distance=20):
    """
    非极大值抑制
    
    Args:
        peak_indices: 峰索引列表
        peak_probs: 对应的峰概率
        min_distance: 最小距离阈值
        
    Returns:
        nms_peaks: NMS后的峰索引列表
    """
    if len(peak_indices) <= 1:
        return peak_indices
    
    # 将索引和概率配对并按概率排序
    paired = sorted(zip(peak_indices, peak_probs), key=lambda x: x[1], reverse=True)
    sorted_indices, sorted_probs = zip(*paired)
    
    # 初始化NMS结果
    nms_peaks = [sorted_indices[0]]
    
    # 对剩余峰进行NMS
    for i in range(1, len(sorted_indices)):
        # 检查当前峰是否与已选峰距离太近
        too_close = False
        for selected_peak in nms_peaks:
            if abs(sorted_indices[i] - selected_peak) < min_distance:
                too_close = True
                break
        
        if not too_close:
            nms_peaks.append(sorted_indices[i])
    
    return sorted(nms_peaks)

