import os
import shutil
import re
from pathlib import Path
import glob
import argparse

def extract_frame_number(filename):
    """从文件名中提取帧号"""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else None

def process_dataset(source_path, visible_save_path, infrared_save_path, 
                   initial_interval=100, min_interval=10, interval_step=10, 
                   min_images=2000):
    # 创建保存目录
    os.makedirs(visible_save_path, exist_ok=True)
    os.makedirs(infrared_save_path, exist_ok=True)
    
    # 获取所有包含tmp的文件夹
    tmp_folders = glob.glob(os.path.join(source_path, '*tmp*'))
    
    # 统计所有xyz文件夹下的图像
    all_visible_images = []
    for tmp_folder in tmp_folders:
        xyz_path = os.path.join(tmp_folder, 'xyz')
        if os.path.exists(xyz_path):
            images = glob.glob(os.path.join(xyz_path, '*.jpg'))
            all_visible_images.extend(images)
    
    print(f"总图像数量: {len(all_visible_images)}")
    
    # 按文件夹分组处理图像
    selected_images = []
    frame_interval = initial_interval  # 初始帧间隔
    
    while True:
        selected_images = []
        for folder_idx, tmp_folder in enumerate(tmp_folders):
            xyz_path = os.path.join(tmp_folder, 'xyz')
            if not os.path.exists(xyz_path):
                continue
                
            # 获取当前文件夹下的所有图像
            images = glob.glob(os.path.join(xyz_path, '*.jpg'))
            if not images:
                continue
                
            # 提取帧号并排序，添加文件夹索引作为唯一标识
            frame_images = []
            for img in images:
                frame = extract_frame_number(os.path.basename(img))
                if frame is not None:
                    # 使用文件夹索引和帧号组合作为唯一标识
                    unique_frame = folder_idx * 1000000 + frame
                    frame_images.append((unique_frame, frame, img))
            
            frame_images.sort(key=lambda x: x[0])
            
            # 筛选间隔大于等于frame_interval的图像
            selected = []
            last_frame = None
            for unique_frame, frame, img in frame_images:
                if last_frame is None or frame - last_frame >= frame_interval:
                    selected.append((unique_frame, frame, img))
                    last_frame = frame
            
            selected_images.extend(selected)
        
        print(f"当前帧间隔: {frame_interval}, 筛选后的图像数量: {len(selected_images)}")
        
        if len(selected_images) >= min_images:
            break
        elif frame_interval <= min_interval:  # 设置最小帧间隔
            print(f"警告：即使使用最小帧间隔 {min_interval}，图像数量仍然不足 {min_images}！")
            return
        else:
            frame_interval -= interval_step
            print(f"降低帧间隔至: {frame_interval}")
    
    # 复制图像到目标目录
    for idx, (unique_frame, frame, img_path) in enumerate(selected_images, 1):
        # 复制可见光图像
        new_visible_name = f"{idx:06d}.jpg"
        shutil.copy2(img_path, os.path.join(visible_save_path, new_visible_name))
        
        # 复制对应的红外图像
        m3juvc_path = img_path.replace('xyz', 'M3JUVC')
        if os.path.exists(m3juvc_path):
            new_infrared_name = f"{idx:06d}.jpg"
            shutil.copy2(m3juvc_path, os.path.join(infrared_save_path, new_infrared_name))
        else:
            print(f"警告：找不到对应的红外图像: {m3juvc_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='处理数据集图像')
    
    # 路径参数
    parser.add_argument('--source_path', type=str, 
                       default='/share/Data/Casualty_dataset/',
                       help='源数据路径')
    parser.add_argument('--visible_save_path', type=str,
                       default='/share/Data/Casualty_dataset/images/visible/',
                       help='可见光图像保存路径')
    parser.add_argument('--infrared_save_path', type=str,
                       default='/share/Data/Casualty_dataset/images/infrared/',
                       help='红外图像保存路径')
    
    # 处理参数
    parser.add_argument('--initial_interval', type=int, default=100,
                       help='初始帧间隔')
    parser.add_argument('--min_interval', type=int, default=10,
                       help='最小帧间隔')
    parser.add_argument('--interval_step', type=int, default=10,
                       help='帧间隔递减步长')
    parser.add_argument('--min_images', type=int, default=2000,
                       help='最小所需图像数量')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_dataset(
        source_path=args.source_path,
        visible_save_path=args.visible_save_path,
        infrared_save_path=args.infrared_save_path,
        initial_interval=args.initial_interval,
        min_interval=args.min_interval,
        interval_step=args.interval_step,
        min_images=args.min_images
    )
