import torch
import numpy as np
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. 选择指定的 GPU
gpu_id = 1  # 更改为你希望使用的 GPU 编号
torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}"

# 加载 CLIP 模型和处理器
model_path = "clip-vit-large-patch14"
model = CLIPModel.from_pretrained(model_path).to(device)
processor = CLIPProcessor.from_pretrained(model_path)

# 输入和输出文件夹
input_folder = "data/image_test"
output_folder = "data/image_CLIP/"
os.makedirs(output_folder, exist_ok=True)

# 获取所有 .png 图片
image_files = [f for f in os.listdir(input_folder) if f.endswith('.JPEG')]

for idx, image_file in enumerate(image_files):
    print(f"Processing {idx + 1}/{len(image_files)}: {image_file}")
    
    # 读取并预处理图片
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    
    # 转换为 NumPy 数组并保存为 CSV
    image_features = image_features.cpu().numpy()
    output_path = os.path.join(output_folder, image_file.replace('.JPEG', '.csv'))
    np.savetxt(output_path, image_features, delimiter=',')

print("Processing complete!")