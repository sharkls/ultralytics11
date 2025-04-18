import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_identity_homography_files(input_dir, output_dir):
    """
    读取指定目录下的所有.txt文件，并为每个文件创建对应的单位矩阵单应性矩阵文件
    
    Args:
        input_dir (str): 输入目录路径，包含原始.txt文件
        output_dir (str): 输出目录路径，用于保存生成的单应性矩阵文件
    """
    # 转换为Path对象，更好的路径处理
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建单位矩阵
    identity_matrix = np.eye(3)
    
    # 获取输入目录中的所有.txt文件
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"警告: 在 {input_dir} 中未找到.txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个.txt文件")
    
    # 使用tqdm显示进度条
    for txt_file in tqdm(txt_files, desc=f"处理 {input_dir.name} 目录"):
        try:
            # 构建输出文件路径
            output_file = output_dir / txt_file.name
            
            # 保存单位矩阵到文件
            np.savetxt(output_file, identity_matrix, fmt='%.1f')
        except Exception as e:
            print(f"处理文件 {txt_file.name} 时出错: {e}")
    
    print(f"已生成 {len(txt_files)} 个单应性矩阵文件到 {output_dir}")

def main():
    try:
        # 基础路径
        base_input_dir = Path("/ultralytics/data/LLVIP/labels/visible")
        base_output_dir = Path("/ultralytics/data/LLVIP/extrinsics")
        
        # 检查基础输入目录是否存在
        if not base_input_dir.exists():
            raise FileNotFoundError(f"基础输入目录不存在: {base_input_dir}")
        
        # 处理的子目录列表
        subdirs = ['train', 'val', 'test']
        
        # 统计总数
        total_processed = 0
        
        for subdir in subdirs:
            input_dir = base_input_dir / subdir
            output_dir = base_output_dir / subdir
            
            print(f"\n处理 {subdir} 目录...")
            
            try:
                create_identity_homography_files(input_dir, output_dir)
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
