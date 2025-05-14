## 训练脚本
```bash
python training.py --output_dir=./results --log_dir=./logs --window_size=71
```

## 推理脚本
```bash
python inference.py --model_path=./results/best_model.pth --output_dir=./inference_win71_results --log_dir=./inference_win71_log 
```
单个文件推理
```bash
python inference.py --model_path=./results/best_model.pth --output_dir=./inference_win71_single_results --log_dir=./inference_win71_single_log --single_file=D:\Code\ResearchCode\BCG_Self_One\data_label\test\data\flatten --single_label=D:\Code\ResearchCode\BCG_Self_One\data_label\test\label\flatten
```