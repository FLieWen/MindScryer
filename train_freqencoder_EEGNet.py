import argparse
from model.MindScryerModels_test import FreqEncoder, EEGNetEncoder
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
import wandb
import pandas as pd
import os

# --- 参数解析器 ---
parser = argparse.ArgumentParser(description="EEG 模型训练脚本")

# 数据参数
parser.add_argument('-ed', '--eeg-dataset', default=r"data/EEG/eeg_5_95_std.pth", help="EEG 数据集路径")
parser.add_argument('-sp', '--splits-path', default=r"data/EEG/block_splits_by_image_all.pth", help="数据划分文件路径")
parser.add_argument('-sn', '--split-num', default=0, type=int, help="划分编号 (通常为0)")
parser.add_argument('-sub','--subject', default=0, type=int, help="选择被试 (1-6), 0 代表所有被试")
parser.add_argument('-tl', '--time_low', default=20, type=int, help="截取 EEG 信号的起始时间点")
parser.add_argument('-th', '--time_high', default=460, type=int, help="截取 EEG 信号的结束时间点")

# 模型参数
parser.add_argument('-mt','--model_type', default='lstm', choices=['lstm', 'eegnet'], 
                   help='指定使用的模型: lstm (FreqEncoder) | eegnet (EEGNetEncoder)')
parser.add_argument('-mp','--model_params', default='', nargs='*', 
                   help='(主要用于lstm) 模型选项 key=value 列表')
parser.add_argument('--pretrained_net', default='', help="预训练模型路径 (用于继续训练)")

# 训练参数
parser.add_argument("-b", "--batch_size", default=128, type=int, help="批处理大小")
parser.add_argument('-o', '--optim', default="Adam", help="优化器")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="学习率")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, 
                   help="学习率衰减因子")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, 
                   help="学习率衰减周期 (epochs)")
parser.add_argument('-dw', '--data-workers', default=3, type=int, help="数据加载进程数")
parser.add_argument('-e', '--epochs', default=10000, type=int, help="训练轮数")

# 保存参数
parser.add_argument('-sc', '--saveCheck', default=20, type=int, help="每隔多少轮保存一次模型")
parser.add_argument('--no-cuda', action='store_true', default=False, help="禁用 CUDA")
parser.add_argument('--wandb-project', default='EEG_Model_Training', help="Wandb项目名称")
parser.add_argument('--run-notes', default='', help="运行说明")

opt = parser.parse_args()
print(opt)

# --- 初始化设置 ---
use_cuda = not opt.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")
if use_cuda:
    cudnn.benchmark = True

# 初始化wandb
wandb.init(project=opt.wandb_project, config=vars(opt), notes=opt.run_notes)
config = wandb.config

# --- 数据集类 ---
class EEGDataset:
    def __init__(self, eeg_signals_path):
        loaded = torch.load(eeg_signals_path)
        if opt.subject != 0:
            self.data = [d for d in loaded['dataset'] if d['subject'] == opt.subject]
            print(f"Loaded data for subject {opt.subject}. Total samples: {len(self.data)}")
            if not self.data:
                raise ValueError(f"No data found for subject {opt.subject}")
        else:
            self.data = loaded['dataset']
            print(f"Loaded data for all subjects. Total samples: {len(self.data)}")

        self.labels = loaded["labels"]
        self.images = loaded["images"]
        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high, :]
        label = self.data[i]["label"]
        return eeg, label

# --- 数据划分器类 ---
class Splitter:
    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        self.dataset = dataset
        loaded = torch.load(split_path)
        
        if split_num < 0 or split_num >= len(loaded["splits"]):
            raise IndexError(f"split_num {split_num} is out of range")
        if split_name not in loaded["splits"][split_num]:
            raise KeyError(f"split_name '{split_name}' not found")

        self.split_idx = loaded["splits"][split_num][split_name]
        print(f"Split '{split_name}' - Original indices: {len(self.split_idx)}")

        # 过滤有效索引
        valid_indices = []
        for i in self.split_idx:
            if 0 <= i < len(self.dataset.data):
                valid_indices.append(i)
            else:
                print(f"Warning: Index {i} out of bounds")
        self.split_idx = valid_indices

        print(f"Split '{split_name}' - Valid indices: {len(self.split_idx)}")
        if not self.split_idx:
            print(f"Warning: Split '{split_name}' is empty")
        self.size = len(self.split_idx)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        eeg, label = self.dataset[self.split_idx[i]]
        return eeg, label

# --- 数据加载 ---
print("Loading dataset...")
dataset = EEGDataset(opt.eeg_dataset)

print("Creating data loaders...")
loaders = {}
for split in ["train", "val", "test"]:
    try:
        split_dataset = Splitter(dataset, opt.splits_path, opt.split_num, split)
        if len(split_dataset) > 0:
            loaders[split] = DataLoader(
                split_dataset,
                batch_size=opt.batch_size,
                drop_last=True,
                shuffle=(split == "train"),
                num_workers=opt.data_workers,
                pin_memory=use_cuda
            )
            print(f"Loader for '{split}' created with {len(split_dataset)} samples")
        else:
            print(f"Skipping empty split '{split}'")
            loaders[split] = None
    except Exception as e:
        print(f"Error creating splitter for '{split}': {e}")
        loaders[split] = None

if loaders["train"] is None:
    raise RuntimeError("Training data loader could not be created")

# --- 模型设置 ---
num_time_steps = opt.time_high - opt.time_low
num_channels = 128  # 根据实际数据调整
num_classes = 40    # 根据实际任务调整
output_feature_dim = 128

print(f"\nSetting up model: {opt.model_type}")
print(f"Input shape: time_steps={num_time_steps}, channels={num_channels}")

if opt.model_type == 'eegnet':
    model = EEGNetEncoder(
        input_channels=num_channels,
        input_time_points=num_time_steps,
        num_classes=num_classes,
        output_feature_dim=output_feature_dim
    )
elif opt.model_type == 'lstm':
    model_options = {
        key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value)
        for (key, value) in [x.split("=") for x in opt.model_params if '=' in x]
    }
    model = FreqEncoder(
        input_size=num_time_steps,
        lstm_size=model_options.get('lstm_size', 128),
        lstm_layers=model_options.get('lstm_layers', 1),
        output_size=output_feature_dim
    )
else:
    raise ValueError(f"Unsupported model_type: {opt.model_type}")

model.to(device)
print(f"Model moved to {device}")

# 加载预训练权重
if opt.pretrained_net:
    try:
        model.load_state_dict(torch.load(opt.pretrained_net, map_location=device))
        print(f"Loaded pre-trained weights from: {opt.pretrained_net}")
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")
        print("Training from scratch")

# 优化器
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)
print(f"Optimizer: {opt.optim}, LR: {opt.learning_rate}")

# --- 训练准备 ---
# 创建结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = Path("results_EEGNet") / f"{opt.model_type}_sub{opt.subject}_{timestamp}"
results_dir.mkdir(parents=True, exist_ok=True)

# 保存配置
config_save_path = results_dir / "config.txt"
with open(config_save_path, 'w') as f:
    for key, value in vars(opt).items():
        f.write(f"{key}: {value}\n")

# 初始化CSV记录
metrics_csv_path = results_dir / "training_metrics.csv"
metrics_df = pd.DataFrame(columns=[
    "epoch", "train_loss", "train_acc", 
    "val_loss", "val_acc", "test_loss", "test_acc"
])

# 训练状态
best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_model_path = results_dir / "best_model.pth"

# --- 训练循环 ---
print("\nStarting training...")
for epoch in range(1, opt.epochs + 1):
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}

    # 学习率调整
    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    for split in ("train", "val", "test"):
        if loaders[split] is None:
            continue

        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)

        for i, (input_batch, target_batch) in enumerate(loaders[split]):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            # 调整输入形状
            if opt.model_type == 'eegnet':
                input_model = input_batch.permute(0, 2, 1)
            else:
                input_model = input_batch

            # 前向传播
            output, _ = model(input_model)
            loss = F.cross_entropy(output, target_batch)
            losses[split] += loss.item()

            # 计算准确率
            _, pred = output.data.max(1)
            correct = pred.eq(target_batch.data).sum().item()
            accuracy = correct / input_batch.size(0)
            accuracies[split] += accuracy
            counts[split] += 1

            # 反向传播
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # 计算epoch指标
    TrL = losses["train"] / counts["train"] if counts["train"] > 0 else 0
    TrA = accuracies["train"] / counts["train"] if counts["train"] > 0 else 0
    VL = losses["val"] / counts["val"] if counts["val"] > 0 else 0
    VA = accuracies["val"] / counts["val"] if counts["val"] > 0 else 0
    TeL = losses["test"] / counts["test"] if counts["test"] > 0 else 0
    TeA = accuracies["test"] / counts["test"] if counts["test"] > 0 else 0

    # 记录到wandb
    wandb.log({
        "epoch": epoch,
        "train_loss": TrL, "train_acc": TrA,
        "val_loss": VL, "val_acc": VA,
        "test_loss": TeL, "test_acc": TeA,
        "best_val_acc": best_accuracy_val,
        "best_test_acc": best_accuracy,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

    # 保存到CSV
    metrics_df.loc[len(metrics_df)] = {
        "epoch": epoch,
        "train_loss": TrL, "train_acc": TrA,
        "val_loss": VL, "val_acc": VA,
        "test_loss": TeL, "test_acc": TeA
    }
    metrics_df.to_csv(metrics_csv_path, index=False)

    # 更新最佳模型（修改后的部分）
    if VA >= best_accuracy_val and counts["val"] > 0:
        best_accuracy_val = VA
        best_accuracy = TeA
        best_epoch = epoch
        
        # 始终保存到同一个文件路径（覆盖前一个最佳模型）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': VA,
            'test_acc': TeA,
            'config': vars(opt)
        }, best_model_path)  # 使用固定文件名
        
        print(f"*** New best val acc: {VA:.4f}, test acc: {TeA:.4f} at epoch {epoch} ***")
        print(f"Updated best model at: {best_model_path}")

    # 打印进度
    print(f"Epoch {epoch}/{opt.epochs}: "
          f"TrL={TrL:.4f}, TrA={TrA:.4f} | "
          f"VL={VL:.4f}, VA={VA:.4f} | "
          f"TeL={TeL:.4f}, TeA={TeA:.4f} | "
          f"Best VA={best_accuracy_val:.4f} @ epoch {best_epoch}")

    # 每10轮保存一次检查点
    if epoch % 20 == 0 or epoch == opt.epochs:
        checkpoint_path = results_dir / f"checkpoint_epoch{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': VA,
            'test_acc': TeA,
            'config': vars(opt)
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

# --- 训练完成 ---
print("\nTraining completed")
print(f"Best validation accuracy: {best_accuracy_val:.4f} at epoch {best_epoch}")
print(f"Corresponding test accuracy: {best_accuracy:.4f}")

# 保存最终结果
final_results = {
    'options': vars(opt),
    'losses': losses_per_epoch,
    'accuracies': accuracies_per_epoch,
    'best_val_acc': best_accuracy_val,
    'best_test_acc': best_accuracy,
    'best_epoch': best_epoch
}

final_results_path = results_dir / "final_results.pth"
torch.save(final_results, final_results_path)
print(f"Saved final results to {final_results_path}")

# 上传文件到wandb
wandb.save(str(best_model_path))
wandb.save(str(metrics_csv_path))
wandb.save(str(final_results_path))
wandb.finish()