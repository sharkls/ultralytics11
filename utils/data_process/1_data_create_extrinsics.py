import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def create_identity_homography_files(input_dir, output_dir, homography_path=None):
    """
    读取指定目录下的所有.txt文件，并为每个文件创建对应的单应性矩阵文件
    如果没有.txt文件，则根据 images/visible/*.jpg 生成同名txt
    Args:
        input_dir (str): 输入目录路径，包含原始.txt文件
        output_dir (str): 输出目录路径，用于保存生成的单应性矩阵文件
        homography_path (str, optional): 单应性矩阵.npy文件的路径，如果提供则使用该矩阵
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    # 检查输入目录是否存在
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # 加载单应性矩阵
    if homography_path is not None:
        homography_path = Path(homography_path)
        if not homography_path.exists():
            raise FileNotFoundError(f"单应性矩阵文件不存在: {homography_path}")
        try:
            homography_matrix = np.load(homography_path)
            if homography_matrix.shape != (3, 3):
                raise ValueError(f"单应性矩阵维度不正确，应为(3,3)，实际为{homography_matrix.shape}")
            print(f"已加载单应性矩阵文件: {homography_path}")
        except Exception as e:
            raise Exception(f"加载单应性矩阵文件时出错: {e}")
    else:
        homography_matrix = np.eye(3)
        print("使用单位矩阵作为默认单应性矩阵")
    # 获取输入目录中的所有.txt文件
    txt_files = list(input_dir.glob("*.txt"))
    # 如果没有txt文件，尝试直接在该目录下查找 .jpg 生成
    if not txt_files:
        print(f"{input_dir} 中未找到.txt文件，直接在该目录下查找 .jpg 生成...")
        images_dir = input_dir
        jpg_files = list(images_dir.glob('*.jpg'))
        if not jpg_files:
            print(f"警告: 在 {images_dir} 中未找到.jpg文件，无法生成单应性矩阵文件")
            return
        txt_files = [Path(f.stem + '.txt') for f in jpg_files]
        print(f"将为 {len(txt_files)} 张图片生成单应性矩阵文件")
        for txt_file in tqdm(txt_files, desc=f"处理 {images_dir.name} 目录"): 
            output_file = output_dir / txt_file.name
            try:
                np.savetxt(output_file, homography_matrix, fmt='%.6f')
            except Exception as e:
                print(f"处理文件 {txt_file.name} 时出错: {e}")
        print(f"已生成 {len(txt_files)} 个单应性矩阵文件到 {output_dir}")
        return
    # 正常流程：有txt文件
    print(f"找到 {len(txt_files)} 个.txt文件")
    for txt_file in tqdm(txt_files, desc=f"处理 {input_dir.name} 目录"):
        try:
            output_file = output_dir / txt_file.name
            np.savetxt(output_file, homography_matrix, fmt='%.6f')
        except Exception as e:
            print(f"处理文件 {txt_file.name} 时出错: {e}")
    print(f"已生成 {len(txt_files)} 个单应性矩阵文件到 {output_dir}")

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='创建单应性矩阵文件')
    
    # 添加命令行参数
    parser.add_argument('--input_dir', type=str, 
                      default="data/Data/0528-test/images/visible",
                      help='输入目录路径，包含原始.txt文件')
    parser.add_argument('--output_dir', type=str,
                      default="data/Data/0528-test/extrinsics",
                      help='输出目录路径，用于保存生成的单应性矩阵文件')
    parser.add_argument('--homography_path', type=str, default="runs/mapping_matrix-v3/manual_homography_matrix-v3.npy",
                      help='单应性矩阵.npy文件的路径（可选）')
    parser.add_argument('--subdirs', type=str, nargs='+',
                      default=['train'],
                      help='要处理的子目录列表，默认为 train val test')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    try:
        # 使用命令行参数
        base_input_dir = Path(args.input_dir)
        base_output_dir = Path(args.output_dir)
        
        # 检查基础输入目录是否存在
        if not base_input_dir.exists():
            raise FileNotFoundError(f"基础输入目录不存在: {base_input_dir}")
        
        # 统计总数
        total_processed = 0
        
        for subdir in args.subdirs:
            input_dir = base_input_dir / subdir
            output_dir = base_output_dir / subdir
            
            print(f"\n处理 {subdir} 目录...")
            
            try:
                create_identity_homography_files(input_dir, output_dir, args.homography_path)
                # 计算该目录下处理的文件数
                processed = len(list(output_dir.glob("*.txt")))
                total_processed += processed
                print(f"{subdir} 目录处理完成: {processed} 个文件")
            except Exception as e:
                print(f"处理 {subdir} 目录时出错: {e}")
        
        print(f"\n所有处理完成! 总共生成 {total_processed} 个单应性矩阵文件")
        
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()
