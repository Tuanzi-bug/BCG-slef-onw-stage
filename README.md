# BCG 心跳检测算法说明

GitHub 地址：

- 代码仓库：[Tuanzi-bug/BCG-slef-onw-stage](https://github.com/Tuanzi-bug/BCG-slef-onw-stage)
- 论文仓库：[Tuanzi-bug/Weakly-Supervised-Learning](https://github.com/Tuanzi-bug/Weakly-Supervised-Learning)，目前是 private，需要有访问权限才能打开。

这个项目主要用于 BCG（ballistocardiogram，心冲击图）信号的心率和心跳峰检测。代码会先从原始 BCG 信号中找出候选峰，再截取每个候选峰附近的一小段信号，最后用模型判断候选峰的概率，并给出心率预测结果。

下面的说明统一按 `window_size=91` 来写，对应现有模型目录 `./results_2` 里的配置和权重文件。

## 和论文的对应关系

这个代码对应的论文在：

```text
Weakly-Supervised-Learning
```

论文主文件是 `main.tex`，当前编译出的 PDF 是 `main.pdf`。论文题目为：

```text
Physiologically Constrained Weakly Supervised Learning for Joint BCG Heart Rate Estimation and J-Peak Localization
```

论文里的核心想法和这份代码是一致的：不用逐点的 J-peak 标注作为训练目标，而是使用每段 1 分钟 BCG 信号的平均心率作为弱监督。模型先生成候选峰，再结合候选峰附近的局部波形特征和整段信号中的节律上下文，预测每个候选峰是不是 J-peak，同时估计整段信号的心率。

训练时主要依赖两个约束：一个是数量一致性，让模型预测出的候选峰数量和平均心率对应的心跳数量接近；另一个是稀疏约束，让模型不要把概率分散到太多不可靠的候选峰上。这样做的目的是在减少标注成本的同时，仍然保留 beat-level 的峰位置信息，方便后续做 IBI 或 HRV 分析。

论文当前 `main.tex` 中报告的摘要结果是：在 40 份 BCG 记录上，该方法的心率估计平均绝对误差为 0.43 beats per minute，并且相比基于重建的基线方法，能更好地保留用于 HRV 分析的心跳时序信息。这里的结果表述以论文当前版本为准。

## 文件说明

- `training.py`：训练入口。它会读取训练集和验证集，初始化模型，完成训练和验证，并保存模型、日志和配置文件。
- `inference.py`：推理入口。它会加载训练好的模型，对测试数据或指定目录中的 `.npy` 文件进行预测，并保存预测结果和可视化图。
- `process_data.py`：数据加载和预处理。主要负责读取 `.npy` 数据与标签，提取候选峰，截取峰附近窗口，并整理成 PyTorch DataLoader。
- `model.py`：模型结构。这里包含 BCG 信号处理器、位置编码、局部特征提取模块、心跳检测模型、损失函数和非极大值抑制等核心逻辑。
- `train_epoch.py`：训练辅助函数。包含单轮训练、验证、损失曲线、回归图和峰检测可视化等函数。
- `utils.py`：通用工具函数。主要包含 Z-score 标准化和候选峰检测函数。

## 数据说明

数据放在 `data_label` 文件夹下。这里主要说明两类数据：

- `flatten`：年轻人数据。
  - 训练数据：`data_label/train/data/flatten`
  - 训练标签：`data_label/train/label/flatten`
  - 测试数据：`data_label/test/data/flatten`
  - 测试标签：`data_label/test/label/flatten`
- `older`：老年人数据。
  - 测试数据：`data_label/test/data/older`
  - 测试标签：`data_label/test/label/older`

已核对的数据形状示例：

- `flatten` 训练数据首个文件形状为 `(432, 6000)`，对应标签形状为 `(432, 1)`。
- `flatten` 测试数据首个文件形状为 `(108, 6000)`，对应标签形状为 `(108, 1)`。
- `older` 测试数据首个文件形状为 `(9, 6000)`，对应标签形状为 `(9, 1)`。

其中每条 BCG 信号长度为 6000 个采样点，标签为对应样本的心率值。

## 运行前准备

先进入项目目录：

```bash
cd /Users/yay/Desktop/deepLearning/BCG-slef-onw-stage
```

代码中还保留了一些 Windows 绝对路径作为默认值，所以运行时建议按下面的命令显式传入相对数据路径。

下面命令中的 `python` 指已经安装好项目依赖的 Python 环境。如果系统提示 `python: command not found`，可以改用 `python3`，或者先进入对应的 Conda/虚拟环境。

如果没有可用 GPU，可以在训练或推理命令后增加 `--no_gpu`。

## 训练命令

使用年轻人 `flatten` 数据训练，输出到 `results_2`：

```bash
python training.py \
  --train_data_dir ./data_label/train/data/flatten \
  --train_label_dir ./data_label/train/label/flatten \
  --val_data_dir ./data_label/test/data/flatten \
  --val_label_dir ./data_label/test/label/flatten \
  --output_dir ./results_2 \
  --log_dir ./logs_2 \
  --max_peaks 130 \
  --window_size 91
```

训练完成后，主要模型文件通常保存在：

- `./results_2/best_model.pth`：验证集表现最好的模型。
- `./results_2/bcg_heartbeat_detector_final.pth`：训练结束时的最终模型。
- `./results_2/config.json`：本次训练使用的配置。

## 推理命令

### 推理年轻人 `flatten` 测试数据

```bash
python inference.py \
  --model_path ./results_2/best_model.pth \
  --test_data_dir ./data_label/test/data/flatten \
  --test_label_dir ./data_label/test/label/flatten \
  --output_dir ./inference_results \
  --log_dir ./inference_logs
```

### 推理老年人 `older` 测试数据

```bash
python inference.py \
  --model_path ./results_2/best_model.pth \
  --test_data_dir ./data_label/test/data/older \
  --test_label_dir ./data_label/test/label/older \
  --output_dir ./inference_older_results \
  --log_dir ./inference_older_log
```

推理完成后，预测结果会保存到对应的 `output_dir` 中，例如 `predictions.json` 和可视化图片；TensorBoard 日志会保存到对应的 `log_dir` 中。

## 常用参数

- `--model_path`：推理时加载的模型权重路径。
- `--train_data_dir`、`--train_label_dir`：训练数据和标签目录。
- `--val_data_dir`、`--val_label_dir`：训练阶段用于验证的数据和标签目录。
- `--test_data_dir`、`--test_label_dir`：推理阶段使用的测试数据和标签目录。
- `--window_size`：候选峰附近截取的窗口长度。本说明默认使用 `91`。
- `--max_peaks`：每条信号最多保留的候选峰数量。本说明默认使用 `130`。
- `--output_dir`：模型或推理结果输出目录。
- `--log_dir`：TensorBoard 日志目录。
- `--no_gpu`：禁用 GPU，使用 CPU 运行。
