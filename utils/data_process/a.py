import os
import shutil
import glob
from pathlib import Path
import argparse

def split_and_copy_images(source_path, save_base_path, num_parts=4):
    # 获取所有图像并按名称排序
    images = sorted(glob.glob(os.path.join(source_path, '*.jpg')))
    total_images = len(images)
    
    # 计算每个部分的大小（平均分配）
    base_size = total_images // num_parts
    remainder = total_images % num_parts
    part_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_parts)]
    
    # 创建保存目录并复制图像
    start_idx = 0
    for part_idx, part_size in enumerate(part_sizes, 1):
        # 创建保存目录
        save_dir = os.path.join(save_base_path, f'tmp_{part_idx}')
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取当前部分的图像
        end_idx = start_idx + part_size
        part_images = images[start_idx:end_idx]
        
        # 复制图像，保持原始文件名
        for img_path in part_images:
            shutil.copy2(img_path, os.path.join(save_dir, os.path.basename(img_path)))
        
        print(f"第 {part_idx} 部分完成，共 {len(part_images)} 张图像")
        start_idx = end_idx

def parse_args():
    parser = argparse.ArgumentParser(description='将图像数据集平均分割成多个部分')
    
    # 路径参数
    parser.add_argument('--source_path', type=str, 
                       default='/share/Data/data_20250519_v2/images/visible/',
                       help='源图像路径')
    parser.add_argument('--save_base_path', type=str,
                       default='/share/Data/data_20250519_v2/images/split_visible/',
                       help='保存路径')
    
    # 处理参数
    parser.add_argument('--num_parts', type=int, default=3,
                       help='要分割的部分数量')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # 创建基础保存目录
    os.makedirs(args.save_base_path, exist_ok=True)
    
    # 执行分割和复制
    split_and_copy_images(
        source_path=args.source_path,
        save_base_path=args.save_base_path,
        num_parts=args.num_parts
    )