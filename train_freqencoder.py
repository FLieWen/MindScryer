# # Define options
# import argparse
# from model.MindScryerModels import FreqEncoder
# from pathlib import Path
# from datetime import datetime

# parser = argparse.ArgumentParser(description="Template")

# parser.add_argument('-ed', '--eeg-dataset', default=r"data/EEG/eeg_5_95_std.pth", help="EEG dataset path") #5-95Hz
# #Splits
# parser.add_argument('-sp', '--splits-path', default=r"data/EEG/block_splits_by_image_all.pth", help="splits path") #All subjects
# ### BLOCK DESIGN ###
# parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number") #leave this always to zero.
# #Subject selecting
# parser.add_argument('-sub','--subject', default= 0   , type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")
# #Time options: select from 20 to 460 samples from EEG data
# parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
# parser.add_argument('-th', '--time_high', default=460,  type=float, help="highest time value")
# # Model type/options
# parser.add_argument('-mt','--model_type', default='lstm', help='specify which generator should be used: lstm|EEGChannelNet')
# parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
# parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")
# # Training options
# parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
# parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
# parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
# parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
# parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
# parser.add_argument('-dw', '--data-workers', default=3, type=int, help="data loading workers")
# parser.add_argument('-e', '--epochs', default=1000, type=int, help="training epochs")
# # Save options
# parser.add_argument('-sc', '--saveCheck', default=20, type=int, help="learning rate")
# # Backend options
# parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")
# # Parse arguments
# opt = parser.parse_args()
# print(opt)

# # Imports
# import torch; torch.utils.backcompat.broadcast_warning.enabled = True
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.optim
# import torch.backends.cudnn as cudnn; cudnn.benchmark = True
# import numpy as np
# import importlib

# # Dataset class
# class EEGDataset:
    
#     # Constructor
#     def __init__(self, eeg_signals_path):
#         # Load EEG signals
#         loaded = torch.load(eeg_signals_path)
#         if opt.subject!=0:
#             self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
#         else:
#             self.data=loaded['dataset']        
#         self.labels = loaded["labels"]
#         self.images = loaded["images"]
        
#         # Compute size
#         self.size = len(self.data)

#     # Get size
#     def __len__(self):
#         return self.size

#     # Get item
#     def __getitem__(self, i):
#         # Process EEG
#         eeg = self.data[i]["eeg"].float().t()
#         eeg = eeg[opt.time_low:opt.time_high,:]

#         if opt.model_type == "model10":
#             eeg = eeg.t()
#             eeg = eeg.view(1,128,opt.time_high-opt.time_low)

#         # # Get label
#         label = self.data[i]["label"]
#         # Return

#         return eeg, label

# # Splitter class
# class Splitter:

#     def __init__(self, dataset, split_path, split_num=0, split_name="train"):
#         # Set EEG dataset
#         self.dataset = dataset
#         # Load split
#         loaded = torch.load(split_path)
#         self.split_idx = loaded["splits"][split_num][split_name]
#         # Filter data
#         self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
#         # Compute size
#         self.size = len(self.split_idx)

#     # Get size
#     def __len__(self):
#         return self.size

#     # Get item
#     def __getitem__(self, i):
#         # Get sample from dataset
#         eeg, label = self.dataset[self.split_idx[i]]
#         # Return
#         return eeg, label

# # Load dataset
# dataset = EEGDataset(opt.eeg_dataset)
# # Create loaders
# loaders = {split: DataLoader(Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}
# train_dataset=Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = "train")
# print(len(train_dataset))

# # Load model

# model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for (key, value) in [x.split("=") for x in opt.model_params]}
# # Create discriminator model/optimizer

# model = FreqEncoder(**model_options)
# optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)
    
# # Setup CUDA
# if not opt.no_cuda:
#     model.cuda()
#     print("Copied to CUDA")

# if opt.pretrained_net != '':
#         model = torch.load(opt.pretrained_net)
#         print(model)

# #initialize training,validation, test losses and accuracy list
# losses_per_epoch={"train":[], "val":[],"test":[]}
# accuracies_per_epoch={"train":[],"val":[],"test":[]}

# best_accuracy = 0
# best_accuracy_val = 0
# best_epoch = 0
# # Start training

# predicted_labels = [] 
# correct_labels = []

# for epoch in range(1, opt.epochs+1):
#     # Initialize loss/accuracy variables
#     losses = {"train": 0, "val": 0, "test": 0}
#     accuracies = {"train": 0, "val": 0, "test": 0}
#     counts = {"train": 0, "val": 0, "test": 0}
#     # Adjust learning rate for SGD
#     if opt.optim == "SGD":
#         lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
#     # Process each split
#     for split in ("train", "val", "test"):
#         # Set network mode
#         if split == "train":
#             model.train()
#             torch.set_grad_enabled(True)
#         else:
#             model.eval()
#             torch.set_grad_enabled(False)
#         # Process all split batches
#         for i, (input, target) in enumerate(loaders[split]):
#             # Check CUDA
#             if not opt.no_cuda:
#                 input = input.to("cuda") 
#                 target = target.to("cuda")

#             #input=input.unsqueeze(1)
#             # Forward
#             # 确认原始输入形状，并调整为 [batch, channels, time_steps]
#             # print("input.shape:", input.shape)
#             # if input.shape[1] == 440 and input.shape[2] == 128:
#             #     input_permuted = input.permute(0, 2, 1) # 调整为 [b, 128, 440]
#             #     # print(f"Debug: Permuted input shape for FreqEncoder: {input_permuted.shape}") # 确认形状
#             # elif input.shape[1] == 128 and input.shape[2] == 440:
#             #     # 如果已经是 [b, 128, 440]，则无需改变
#             #     input_permuted = input
#             #     print(f"Debug: Input shape already correct for FreqEncoder: {input_permuted.shape}") # 确认形状
#             # else:
#             #     print(f"错误: train_freqencoder.py 中输入维度不匹配! 期望 [b, 440, 128] 或 [b, 128, 440], 得到 {input.shape}")
#             #     continue # 跳过此批次或抛出错误
#             output,xa = model(input)
#             #print(np.shape(input))
#             # Compute loss
#             loss = F.cross_entropy(output, target)
#             losses[split] += loss.item()
#             # Compute accuracy
#             _,pred = output.data.max(1)
#             correct = pred.eq(target.data).sum().item()
#             accuracy = correct/input.data.size(0)   
#             accuracies[split] += accuracy
#             counts[split] += 1
#             # Backward and optimize
#             if split == "train":
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
    
#     # Print info at the end of the epoch
#     if accuracies["val"]/counts["val"] >= best_accuracy_val:
#         best_accuracy_val = accuracies["val"]/counts["val"]
#         best_accuracy = accuracies["test"]/counts["test"]
#         best_epoch = epoch
    
#     TrL,TrA,VL,VA,TeL,TeA=  losses["train"]/counts["train"],accuracies["train"]/counts["train"],losses["val"]/counts["val"],accuracies["val"]/counts["val"],losses["test"]/counts["test"],accuracies["test"]/counts["test"]
#     print("Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [{9}-{10} Hz] - Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(epoch,
#                                                                                                          losses["train"]/counts["train"],
#                                                                                                          accuracies["train"]/counts["train"],
#                                                                                                          losses["val"]/counts["val"],
#                                                                                                          accuracies["val"]/counts["val"],
#                                                                                                          losses["test"]/counts["test"],
#                                                                                                          accuracies["test"]/counts["test"],
#                                                                                                          best_accuracy, best_epoch, opt.time_low,opt.time_high, opt.model_type,opt.subject))

#     losses_per_epoch['train'].append(TrL)
#     losses_per_epoch['val'].append(VL)
#     losses_per_epoch['test'].append(TeL)
#     accuracies_per_epoch['train'].append(TrA)
#     accuracies_per_epoch['val'].append(VA)
#     accuracies_per_epoch['test'].append(TeA)

#     # if epoch%opt.saveCheck == 0:
#     #             torch.save(model, '%s__subject%d_epoch_%d.pth' % (opt.model_type, opt.subject,epoch))
#     if epoch % opt.saveCheck == 0:
#         # 获取当前时间戳
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         # 构建保存路径
#         save_dir = Path("test") / f"{opt.model_type}_subject{opt.subject}_{timestamp}"
#         save_dir.mkdir(parents=True, exist_ok=True)
        
#         # 保存模型到该文件夹下
#         model_save_path = save_dir / f"epoch_{epoch}.pth"
#         torch.save(model, model_save_path)
#         print(f"Model saved to {model_save_path}")



# # Define options
# import argparse
# from model.MindScryerModels import FreqEncoder
# from pathlib import Path
# from datetime import datetime
# import wandb
# import csv
# import os

# # 参数解析
# parser = argparse.ArgumentParser(description="Template")

# parser.add_argument('-ed', '--eeg-dataset', default=r"data/EEG/eeg_5_95_std.pth", help="EEG dataset path") #5-95Hz
# #Splits
# parser.add_argument('-sp', '--splits-path', default=r"data/EEG/block_splits_by_image_all.pth", help="splits path") #All subjects
# ### BLOCK DESIGN ###
# parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number") #leave this always to zero.
# #Subject selecting
# parser.add_argument('-sub','--subject', default=0, type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")
# #Time options: select from 20 to 460 samples from EEG data
# parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
# parser.add_argument('-th', '--time_high', default=460, type=float, help="highest time value")
# # Model type/options
# parser.add_argument('-mt','--model_type', default='lstm', help='specify which generator should be used: lstm|EEGChannelNet')
# parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
# parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")
# # Training options
# parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
# parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
# parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
# parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
# parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
# parser.add_argument('-dw', '--data-workers', default=3, type=int, help="data loading workers")
# parser.add_argument('-e', '--epochs', default=1000, type=int, help="training epochs")
# # Save options
# parser.add_argument('-sc', '--saveCheck', default=20, type=int, help="save checkpoint every N epochs")
# # Backend options
# parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")
# # WandB options
# parser.add_argument('--wandb', action='store_true', help="enable wandb logging")
# # Parse arguments
# opt = parser.parse_args()
# print(opt)

# # 初始化wandb
# if opt.wandb:
#     wandb.init(project="EEG-FreqEncoder", config=vars(opt))
#     config = wandb.config

# # Imports
# import torch; torch.utils.backcompat.broadcast_warning.enabled = True
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.optim
# import torch.backends.cudnn as cudnn; cudnn.benchmark = True
# import numpy as np
# import importlib

# # Dataset class
# class EEGDataset:
    
#     # Constructor
#     def __init__(self, eeg_signals_path):
#         # Load EEG signals
#         loaded = torch.load(eeg_signals_path)
#         if opt.subject!=0:
#             self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if loaded['dataset'][i]['subject']==opt.subject]
#         else:
#             self.data=loaded['dataset']        
#         self.labels = loaded["labels"]
#         self.images = loaded["images"]
        
#         # Compute size
#         self.size = len(self.data)

#     # Get size
#     def __len__(self):
#         return self.size

#     # Get item
#     def __getitem__(self, i):
#         # Process EEG
#         eeg = self.data[i]["eeg"].float().t()
#         eeg = eeg[opt.time_low:opt.time_high,:]

#         if opt.model_type == "model10":
#             eeg = eeg.t()
#             eeg = eeg.view(1,128,opt.time_high-opt.time_low)

#         # Get label
#         label = self.data[i]["label"]
#         return eeg, label

# # Splitter class
# class Splitter:
#     def __init__(self, dataset, split_path, split_num=0, split_name="train"):
#         # Set EEG dataset
#         self.dataset = dataset
#         # Load split
#         loaded = torch.load(split_path)
#         self.split_idx = loaded["splits"][split_num][split_name]
#         # Filter data
#         self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
#         # Compute size
#         self.size = len(self.split_idx)

#     # Get size
#     def __len__(self):
#         return self.size

#     # Get item
#     def __getitem__(self, i):
#         # Get sample from dataset
#         eeg, label = self.dataset[self.split_idx[i]]
#         return eeg, label

# # 创建保存目录
# model_save_dir = Path("saved_models") / f"{opt.model_type}_subject{opt.subject}"
# model_save_dir.mkdir(parents=True, exist_ok=True)

# # 初始化CSV日志文件
# csv_file = model_save_dir / "training_metrics.csv"
# with open(csv_file, mode='w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc', 'best_val_acc', 'best_test_acc'])

# # Load dataset
# dataset = EEGDataset(opt.eeg_dataset)
# # Create loaders
# loaders = {split: DataLoader(Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name=split), 
#                             batch_size=opt.batch_size, drop_last=True, shuffle=True) for split in ["train", "val", "test"]}
# train_dataset = Splitter(dataset, split_path=opt.splits_path, split_num=opt.split_num, split_name="train")
# print(f"训练集样本数量: {len(train_dataset)}")

# # Load model
# model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) 
#                 for (key, value) in [x.split("=") for x in opt.model_params]}
# model = FreqEncoder(**model_options)
# optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr=opt.learning_rate)
    
# # Setup CUDA
# if not opt.no_cuda:
#     model.cuda()
#     print("Copied to CUDA")

# if opt.pretrained_net != '':
#         model = torch.load(opt.pretrained_net)
#         print(model)

# # 初始化训练指标
# losses_per_epoch = {"train": [], "val": [], "test": []}
# accuracies_per_epoch = {"train": [], "val": [], "test": []}
# best_accuracy = 0
# best_accuracy_val = 0
# best_epoch = 0

# # 训练循环
# for epoch in range(1, opt.epochs+1):
#     # 初始化损失和准确率
#     losses = {"train": 0, "val": 0, "test": 0}
#     accuracies = {"train": 0, "val": 0, "test": 0}
#     counts = {"train": 0, "val": 0, "test": 0}
    
#     # 调整学习率(SGD)
#     if opt.optim == "SGD":
#         lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
    
#     # 处理每个数据分割
#     for split in ("train", "val", "test"):
#         # 设置模型模式
#         if split == "train":
#             model.train()
#             torch.set_grad_enabled(True)
#         else:
#             model.eval()
#             torch.set_grad_enabled(False)
        
#         # 处理所有批次
#         for i, (input, target) in enumerate(loaders[split]):
#             if not opt.no_cuda:
#                 input = input.to("cuda")
#                 target = target.to("cuda")
            
#             # 前向传播
#             output, xa = model(input)
            
#             # 计算损失
#             loss = F.cross_entropy(output, target)
#             losses[split] += loss.item()
            
#             # 计算准确率
#             _, pred = output.data.max(1)
#             correct = pred.eq(target.data).sum().item()
#             accuracy = correct / input.data.size(0)
#             accuracies[split] += accuracy
#             counts[split] += 1
            
#             # 反向传播和优化
#             if split == "train":
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
    
#     # 计算各指标
#     TrL, TrA = losses["train"]/counts["train"], accuracies["train"]/counts["train"]
#     VL, VA = losses["val"]/counts["val"], accuracies["val"]/counts["val"]
#     TeL, TeA = losses["test"]/counts["test"], accuracies["test"]/counts["test"]
    
#     # 更新最佳准确率
#     if VA > best_accuracy_val:
#         best_accuracy_val = VA
#         best_accuracy = TeA
#         best_epoch = epoch
#         # 保存最佳模型
#         best_model_path = model_save_dir / "best_model.pth"
#         torch.save(model.state_dict(), best_model_path)
#         print(f"新的最佳模型已保存，验证准确率: {VA:.4f}")
    
#     # 打印信息
#     print(f"模型: {opt.model_type} - 受试者 {opt.subject} - 时间区间: [{opt.time_low}-{opt.time_high}]")
#     print(f"Epoch {epoch}: 训练损失={TrL:.4f}, 训练准确率={TrA:.4f}, 验证损失={VL:.4f}, 验证准确率={VA:.4f}")
#     print(f"测试损失={TeL:.4f}, 测试准确率={TeA:.4f}, 最佳验证准确率={best_accuracy_val:.4f} (epoch {best_epoch})")
    
#     # 记录指标
#     losses_per_epoch['train'].append(TrL)
#     losses_per_epoch['val'].append(VL)
#     losses_per_epoch['test'].append(TeL)
#     accuracies_per_epoch['train'].append(TrA)
#     accuracies_per_epoch['val'].append(VA)
#     accuracies_per_epoch['test'].append(TeA)
    
#     # 保存到CSV
#     with open(csv_file, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow([epoch, TrL, TrA, VL, VA, TeL, TeA, best_accuracy_val, best_accuracy])
    
#     # 记录到wandb
#     if opt.wandb:
#         wandb.log({
#             "epoch": epoch,
#             "train_loss": TrL,
#             "train_acc": TrA,
#             "val_loss": VL,
#             "val_acc": VA,
#             "test_loss": TeL,
#             "test_acc": TeA,
#             "best_val_acc": best_accuracy_val,
#             "best_test_acc": best_accuracy,
#             "learning_rate": optimizer.param_groups[0]['lr']
#         })
    
#     # 定期保存检查点
#     if epoch % opt.saveCheck == 0:
#         checkpoint_path = model_save_dir / f"epoch_{epoch}.pth"
#         torch.save(model.state_dict(), checkpoint_path)
#         print(f"检查点已保存: {checkpoint_path}")

# # 训练结束
# if opt.wandb:
#     wandb.finish()

# print(f"训练完成，最佳验证准确率: {best_accuracy_val:.4f} (epoch {best_epoch})")
# print(f"对应测试准确率: {best_accuracy:.4f}")
# print(f"所有指标已保存到: {csv_file}")
# print(f"最佳模型已保存到: {model_save_dir/'best_model.pth'}")



# Define options
import argparse
from model.MindScryerModels import FreqEncoder
parser = argparse.ArgumentParser(description="Template")

parser.add_argument('-ed', '--eeg-dataset', default=r"data/EEG/eeg_5_95_std.pth", help="EEG dataset path") #5-95Hz
#Splits
parser.add_argument('-sp', '--splits-path', default=r"data/EEG/block_splits_by_image_all.pth", help="splits path") #All subjects
### BLOCK DESIGN ###
parser.add_argument('-sn', '--split-num', default=0, type=int, help="split number") #leave this always to zero.
#Subject selecting
parser.add_argument('-sub','--subject', default= 0   , type=int, help="choose a subject from 1 to 6, default is 0 (all subjects)")
#Time options: select from 20 to 460 samples from EEG data
parser.add_argument('-tl', '--time_low', default=20, type=float, help="lowest time value")
parser.add_argument('-th', '--time_high', default=460,  type=float, help="highest time value")
# Model type/options
parser.add_argument('-mt','--model_type', default='lstm', help='specify which generator should be used: lstm|EEGChannelNet')
parser.add_argument('-mp','--model_params', default='', nargs='*', help='list of key=value pairs of model options')
parser.add_argument('--pretrained_net', default='', help="path to pre-trained net (to continue training)")
# Training options
parser.add_argument("-b", "--batch_size", default=128, type=int, help="batch size")
parser.add_argument('-o', '--optim', default="Adam", help="optimizer")
parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, help="learning rate")
parser.add_argument('-lrdb', '--learning-rate-decay-by', default=0.5, type=float, help="learning rate decay factor")
parser.add_argument('-lrde', '--learning-rate-decay-every', default=10, type=int, help="learning rate decay period")
parser.add_argument('-dw', '--data-workers', default=3, type=int, help="data loading workers")
parser.add_argument('-e', '--epochs', default=1000, type=int, help="training epochs")
# Save options
parser.add_argument('-sc', '--saveCheck', default=20, type=int, help="learning rate")
# Backend options
parser.add_argument('--no-cuda', default=False, help="disable CUDA", action="store_true")
# Parse arguments
opt = parser.parse_args()
print(opt)

# Imports
import torch; torch.utils.backcompat.broadcast_warning.enabled = True
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import numpy as np
import importlib

# Dataset class
class EEGDataset:
    
    # Constructor
    def __init__(self, eeg_signals_path):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if opt.subject!=0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset']) ) if loaded['dataset'][i]['subject']==opt.subject]
        else:
            self.data=loaded['dataset']        
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float().t()
        eeg = eeg[opt.time_low:opt.time_high,:]

        if opt.model_type == "model10":
            eeg = eeg.t()
            eeg = eeg.view(1,128,opt.time_high-opt.time_low)

        # # Get label
        label = self.data[i]["label"]
        # Return

        return eeg, label

# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label

# Load dataset
dataset = EEGDataset(opt.eeg_dataset)
# Create loaders
loaders = {split: DataLoader(Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = split), batch_size = opt.batch_size, drop_last = True, shuffle = True) for split in ["train", "val", "test"]}
train_dataset=Splitter(dataset, split_path = opt.splits_path, split_num = opt.split_num, split_name = "train")
print(len(train_dataset))

# Load model

model_options = {key: int(value) if value.isdigit() else (float(value) if value[0].isdigit() else value) for (key, value) in [x.split("=") for x in opt.model_params]}
# Create discriminator model/optimizer
model = FreqEncoder(**model_options)
optimizer = getattr(torch.optim, opt.optim)(model.parameters(), lr = opt.learning_rate)
    
# Setup CUDA
if not opt.no_cuda:
    model.cuda()
    print("Copied to CUDA")

if opt.pretrained_net != '':
        model = torch.load(opt.pretrained_net)
        print(model)

#initialize training,validation, test losses and accuracy list
losses_per_epoch={"train":[], "val":[],"test":[]}
accuracies_per_epoch={"train":[],"val":[],"test":[]}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
# Start training

predicted_labels = [] 
correct_labels = []

for epoch in range(1, opt.epochs+1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Adjust learning rate for SGD
    if opt.optim == "SGD":
        lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]):
            # Check CUDA
            if not opt.no_cuda:
                input = input.to("cuda") 
                target = target.to("cuda")

            #input=input.unsqueeze(1)
            # Forward
            output,xa = model(input)
            #print(np.shape(input))
            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _,pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct/input.data.size(0)   
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    # Print info at the end of the epoch
    if accuracies["val"]/counts["val"] >= best_accuracy_val:
        best_accuracy_val = accuracies["val"]/counts["val"]
        best_accuracy = accuracies["test"]/counts["test"]
        best_epoch = epoch
    
    TrL,TrA,VL,VA,TeL,TeA=  losses["train"]/counts["train"],accuracies["train"]/counts["train"],losses["val"]/counts["val"],accuracies["val"]/counts["val"],losses["test"]/counts["test"],accuracies["test"]/counts["test"]
    print("Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [{9}-{10} Hz] - Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(epoch,
                                                                                                         losses["train"]/counts["train"],
                                                                                                         accuracies["train"]/counts["train"],
                                                                                                         losses["val"]/counts["val"],
                                                                                                         accuracies["val"]/counts["val"],
                                                                                                         losses["test"]/counts["test"],
                                                                                                         accuracies["test"]/counts["test"],
                                                                                                         best_accuracy, best_epoch, opt.time_low,opt.time_high, opt.model_type,opt.subject))

    losses_per_epoch['train'].append(TrL)
    losses_per_epoch['val'].append(VL)
    losses_per_epoch['test'].append(TeL)
    accuracies_per_epoch['train'].append(TrA)
    accuracies_per_epoch['val'].append(VA)
    accuracies_per_epoch['test'].append(TeA)

    if epoch%opt.saveCheck == 0:
                torch.save(model, '%s__subject%d_epoch_%d.pth' % (opt.model_type, opt.subject,epoch))
