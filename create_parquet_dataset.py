
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
import hashlib # 用于生成文件名中的哈希部分（可选）

def create_dataset_from_folders(
    image_folder_path,
    text_folder_path,
    output_dataset_root_dir, # 主数据集的根目录，例如 "my_custom_dataset"
    dataset_config_name="default" # 用于 dataset_infos.json 中的顶级键
):
    """
    从包含图片和文本文件的文件夹创建 Parquet 数据集和 dataset_infos.json。
    """
    # --- 1. 创建主输出目录和 data 子目录 ---
    data_subdir = os.path.join(output_dataset_root_dir, "data")
    os.makedirs(data_subdir, exist_ok=True)

    image_data_list = []
    text_contents = []

    print(f"正在从 {image_folder_path} 读取图片和从 {text_folder_path} 读取文本...")
    image_filenames = sorted(os.listdir(image_folder_path))

    for img_filename in tqdm(image_filenames):
        base_filename, img_ext = os.path.splitext(img_filename)

        if img_ext.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
            print(f"跳过非图片文件: {img_filename}")
            continue

        img_path = os.path.join(image_folder_path, img_filename)
        txt_filename = base_filename + ".txt"
        txt_path = os.path.join(text_folder_path, txt_filename)

        if os.path.exists(txt_path):
            try:
                with open(img_path, 'rb') as f_img:
                    image_bytes = f_img.read()
                image_data_list.append({'bytes': image_bytes})

                with open(txt_path, 'r', encoding='utf-8') as f_txt:
                    text = f_txt.read().strip()
                text_contents.append(text)
            except Exception as e:
                print(f"处理文件 {img_filename} 或 {txt_filename} 时出错: {e}。跳过此项。")
        else:
            print(f"警告: 找到了图片 {img_filename} 但未找到对应的文本文件 {txt_filename}。跳过此项。")

    if not image_data_list or not text_contents:
        print("没有找到有效的图片和文本对。无法创建 DataFrame。")
        return

    df = pd.DataFrame({'image': image_data_list, 'text': text_contents})
    num_examples = len(df)

    # --- 2. 保存 Parquet 文件 (单个分片示例) ---
    # 生成一个简单的哈希作为文件名的一部分，模仿原始文件名
    # 你也可以使用固定的字符串或者更复杂的哈希
    file_hash = hashlib.md5(str(df.sample(5).to_dict()).encode()).hexdigest()[:16]
    parquet_filename = f"train-00000-of-00001-{file_hash}.parquet" # 假设只有一个分片
    output_parquet_path = os.path.join(data_subdir, parquet_filename)

    try:
        df.to_parquet(output_parquet_path, index=False)
        print(f"\n数据集已成功保存到: {output_parquet_path}")
        parquet_file_size = os.path.getsize(output_parquet_path)
    except Exception as e:
        print(f"\n保存 Parquet 文件时出错: {e}")
        return

    # --- 3. 创建 dataset_infos.json ---
    dataset_infos_content = {
        dataset_config_name: { # 使用配置名称作为顶级键
            "description": "My custom image-text dataset.",
            "citation": "",
            "homepage": "",
            "license": "",
            "features": {
                "image": {"decode": True, "id": None, "_type": "Image"},
                "text": {"dtype": "string", "id": None, "_type": "Value"}
            },
            "post_processed": None,
            "supervised_keys": None,
            "task_templates": None,
            "builder_name": None, # 可以是 "parquet" 或自定义
            "config_name": dataset_config_name, # 与顶级键一致
            "version": {"version_str": "1.0.0", "major": 1, "minor": 0, "patch": 0}, # 示例版本
            "splits": {
                "train": {
                    "name": "train",
                    "num_bytes": parquet_file_size,
                    "num_examples": num_examples,
                    # "dataset_name": dataset_config_name # 指向父配置名
                    # 对于本地数据集，通常 datasets 库会自动处理，有时 dataset_name 字段可以省略或设为None
                    # 或者你可以将其设置为 builder_name (如果定义了)
                    "dataset_name": "parquet" # 或者更具体的名字
                }
            },
            "download_checksums": None, # 本地数据集通常为None
            "download_size": parquet_file_size, # 对于本地数据集，通常等于dataset_size
            "post_processing_size": None,
            "dataset_size": parquet_file_size,
            "size_in_bytes": parquet_file_size # 总大小
        }
    }
    dataset_infos_path = os.path.join(output_dataset_root_dir, "dataset_infos.json")
    with open(dataset_infos_path, 'w', encoding='utf-8') as f_json:
        json.dump(dataset_infos_content, f_json, indent=4)
    print(f"dataset_infos.json 已保存到: {dataset_infos_path}")

    # --- (可选) 创建一个简单的 README.md ---
    readme_content = f"""
---
dataset_info:
  features:
  - name: image
    dtype: image
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: {parquet_file_size}
    num_examples: {num_examples}
  download_size: {parquet_file_size}
  dataset_size: {parquet_file_size}
---

# My Custom Dataset ({dataset_config_name})

This is a custom dataset prepared for fine-tuning.
    """
    readme_path = os.path.join(output_dataset_root_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f_readme:
        f_readme.write(readme_content.strip())
    print(f"README.md 已保存到: {readme_path}")


if __name__ == "__main__":
    # --- 配置你的路径 ---
    # 你的原始图片和文本文件夹
    input_image_folder = "./data/image_test" # 修改为你的图片文件夹
    input_text_folder = "./data/image_text"   # 修改为你的文本文件夹

    # 你希望创建的新数据集的根目录名称
    # 这将是你在 load_dataset() 中使用的路径
    custom_dataset_name = "my_naruto_style_dataset"
    output_root = f"./{custom_dataset_name}" # 例如 ./my_naruto_style_dataset

    # 创建主输出目录 (如果不存在)
    os.makedirs(output_root, exist_ok=True)

    create_dataset_from_folders(
        input_image_folder,
        input_text_folder,
        output_root,
        dataset_config_name=custom_dataset_name # 使用数据集文件夹名作为配置名
    )

    print(f"\n数据集构建完成。现在你可以在训练脚本中使用路径 '{output_root}' 作为 dataset_name。")
    print(f"例如：load_dataset('{output_root}')")

    # 验证 (可选)
    print("\n--- 尝试使用 datasets 库加载自定义数据集 ---")
    try:
        from datasets import load_dataset
        # 注意：这里加载的是 output_root，而不是 output_root/data
        loaded_ds = load_dataset(output_root, split="train")
        print("成功使用 datasets 库加载！")
        print(loaded_ds)
        print("第一条数据:")
        print(loaded_ds[0])
    except Exception as e:
        print(f"使用 datasets 库加载时出错: {e}")
        print("确保 datasets 库已安装: pip install datasets")