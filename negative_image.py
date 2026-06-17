
import os
import torch
from diffusers import StableDiffusionPipeline

# 1. 选择指定的 GPU
gpu_id = 1  # 更改为你希望使用的 GPU 编号
torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}"

# 2. 加载 Stable Diffusion 模型
custom_cache_dir = "pretrained_model/model"
model2 = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir=custom_cache_dir,
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)
# pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
# pipe.to(device)

# 3. 定义文本描述文件夹路径
text_folder = "data/negative_text"
output_folder = "data/negative_image"

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 生成图像的数量
num_samples = 4  # 可以修改这个值以生成多张图像

# 4. 遍历所有文本文件
for filename in os.listdir(text_folder):
    if filename.endswith(".txt"):
        text_file_path = os.path.join(text_folder, filename)

        # 读取文本描述
        with open(text_file_path, "r") as file:
            description = file.read().strip()

        # 生成 num_samples 张图像
        for idx in range(num_samples):
            output_image = model2(prompt=description, strength=0.85, guidance_scale=7.5, num_inference_steps=100).images[0]

            # 生成输出文件路径
            output_image_path = os.path.join(output_folder, f"{filename.replace('.txt', '')}_sample{idx+1}.png")

            # 保存图像
            output_image.save(output_image_path)
            print(f"已生成并保存: {output_image_path}")
