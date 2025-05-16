import os
import argparse
import random
import shutil
from tqdm import tqdm

def parse_opt():
    parser = argparse.ArgumentParser(description='Split dataset labels and corresponding images')
    parser.add_argument('--label_path', type=str, default='data/LLVIP/labels/train', help='Folder containing label txt files')
    parser.add_argument('--infrared_image_path', type=str, default='data/LLVIP/images/infrared/train', help='Folder containing infrared images')
    parser.add_argument('--visible_image_path', type=str, default='data/LLVIP/images/visible/train', help='Folder containing visible images')
    parser.add_argument('--txt_save_path', type=str, default='data/LLVIP/labels/val', help='Destination folder to save selected txt files')
    parser.add_argument('--infrared_save_path', type=str, default='data/LLVIP/images/infrared/val', help='Destination folder to save selected infrared images')
    parser.add_argument('--visible_save_path', type=str, default='data/LLVIP/images/visible/val', help='Destination folder to save selected visible images')
    parser.add_argument('--split_ratio', type=float, default=0.2, help='Ratio of samples to move (default: 0.2)')
    opt = parser.parse_args()
    return opt

def find_image_file(base_name, image_dir):
    """
    根据基准文件名查找对应的图像文件（支持多种图像格式）。
    
    Args:
        base_name (str): 文件的基准名称，不包含扩展名。
        image_dir (str): 图像所在目录。
    
    Returns:
        str or None: 找到的图像文件路径，如果未找到则返回 None。
    """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for ext in supported_extensions:
        image_file = f"{base_name}{ext}"
        image_path = os.path.join(image_dir, image_file)
        if os.path.isfile(image_path):
            return image_path
    return None

def split_labels_and_images(label_path, txt_save_path, infrared_image_path, infrared_save_path, visible_image_path, visible_save_path, split_ratio=0.2):
    # 确保目标目录存在
    os.makedirs(txt_save_path, exist_ok=True)
    os.makedirs(infrared_save_path, exist_ok=True)
    os.makedirs(visible_save_path, exist_ok=True)

    # 获取所有 .txt 文件
    txt_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    if total_files == 0:
        print(f"在 {label_path} 目录下未找到任何 .txt 文件。")
        return

    # 计算要移动的文件数量
    num_to_move = int(total_files * split_ratio)
    if num_to_move == 0:
        print(f"根据 split_ratio={split_ratio}，没有文件需要移动。")
        return

    # 随机选择文件
    selected_files = random.sample(txt_files, num_to_move)

    # 移动文件及对应图像
    for file_name in tqdm(selected_files, desc="移动 .txt 文件及对应图像"):
        # 移动标签文件
        src_txt = os.path.join(label_path, file_name)
        dst_txt = os.path.join(txt_save_path, file_name)
        shutil.move(src_txt, dst_txt)

        # 获取基准文件名（不含扩展名）
        base_name = os.path.splitext(file_name)[0]

        # 移动可见光图像
        src_vis_image = find_image_file(base_name, visible_image_path)
        if src_vis_image:
            dst_vis_image = os.path.join(visible_save_path, os.path.basename(src_vis_image))
            shutil.move(src_vis_image, dst_vis_image)
        else:
            print(f"警告: 未找到对应的可见光图像 for {file_name}")

        # 移动红外图像
        src_infrared_image = find_image_file(base_name, infrared_image_path)
        if src_infrared_image:
            dst_infrared_image = os.path.join(infrared_save_path, os.path.basename(src_infrared_image))
            shutil.move(src_infrared_image, dst_infrared_image)
        else:
            print(f"警告: 未找到对应的红外图像 for {file_name}")

    print(f"成功移动 {num_to_move} 个文件及对应图像到验证集路径。")

def main():
    opt = parse_opt()
    split_labels_and_images(
        label_path=opt.label_path,
        txt_save_path=opt.txt_save_path,
        infrared_image_path=opt.infrared_image_path,
        infrared_save_path=opt.infrared_save_path,
        visible_image_path=opt.visible_image_path,
        visible_save_path=opt.visible_save_path,
        split_ratio=opt.split_ratio
    )

if __name__ == "__main__":
    main()