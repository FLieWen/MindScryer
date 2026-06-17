import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.models import ViT_H_14_Weights, vit_h_14
import numpy as np
from torchmetrics.functional import accuracy
import os
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
from torchvision.models import inception_v3
from torchvision.transforms import ToTensor, Normalize
from skimage import io, color

# 结构相似度指标函数
def ssim_metric(img1, img2):
    img1 = np.array(img1.squeeze(0).cpu())
    img2 = np.array(img2.squeeze(0).cpu())
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = np.transpose(img2, (1, 2, 0))
    return ssim(img1, img2, data_range=255, channel_axis=-1)

# Top-k 精度评估函数
def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    pick_range = [i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way - 1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
        acc = accuracy(pred_picked.unsqueeze(0), torch.tensor([0], device=pred.device), task="multiclass", num_classes=50, top_k=top_k)
        acc_list.append(acc.item())
    return np.mean(acc_list), np.std(acc_list)

# 图像预处理
def preprocess_images(images):
    transform = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor_images = torch.stack([ToTensor()(img) for img in images])
    normalized_images = transform(tensor_images)
    return normalized_images

# 加载 ViT 模型
weights = ViT_H_14_Weights.DEFAULT
model = vit_h_14(weights=weights)
preprocess = weights.transforms()
model = model.to("cuda")
model = model.eval()

# 设置参数
n_way = 50
num_trials = 50
top_k = 1

# 路径配置
gt_folder = "./my_picture-gene-onlygt_finetune/"
gene_folder = "./my_picture-gene_finetune/"
gt_images_name = os.listdir(gt_folder)
gt_images_name.sort()

final_acc_list = []

# 主评估循环
for gt_name in gt_images_name:
    print(gt_name)

    # 加载 GT 图像
    real_image = Image.open(os.path.join(gt_folder, gt_name)).convert('RGB')

    # 获取对应的 4 张生成图像名称
    base = "_".join(gt_name.split('_')[:6])
    gene_image_name = [f"{base}_{i}.png" for i in range(1, 5)]

    # GT 图像分类
    gt = preprocess(real_image).unsqueeze(0).to("cuda")
    gt_class_id = model(gt).squeeze(0).softmax(0).argmax().item()

    # 当前 GT 对应的 4 张生成图准确率列表
    cur_acc_list = []

    for gene_name in gene_image_name:
        generated_image = Image.open(os.path.join(gene_folder, gene_name)).convert('RGB')
        pred = preprocess(generated_image).unsqueeze(0).to("cuda")
        pred_out = model(pred).squeeze(0).softmax(0).detach()

        acc, std = n_way_top_k_acc(pred_out, gt_class_id, n_way, num_trials, top_k)
        cur_acc_list.append(acc)

    mean_acc_per_gt = np.mean(cur_acc_list)
    final_acc_list.append(mean_acc_per_gt)

    print(f"   mean_acc_for_{gt_name}: {mean_acc_per_gt:.4f}")

# 打印整体评估结果
print("==== Final Result ====")
print("Average Top-1 Accuracy across GT images:", np.mean(final_acc_list))
print("Standard Deviation:", np.std(final_acc_list))

    


