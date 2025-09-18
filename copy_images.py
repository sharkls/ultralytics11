#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片复制和重命名脚本
将 /ultralytics/data/Data/saved_files_1756457235172/XYZ_color 目录下的图片
复制到 /ultralytics/c++/Data/Test 目录，并按照 1.jpg, 2.jpg, 3.jpg... 的格式重命名
"""

import os
import shutil
import glob
from pathlib import Path

def copy_and_rename_images():
    # 源目录和目标目录
    source_dir = "/ultralytics/data/Data/saved_files_1756457235172/XYZ_color"
    target_dir = "/ultralytics/c++/Data/Test"
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"错误：源目录不存在: {source_dir}")
        return False
    
    # 检查目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"已创建目标目录: {target_dir}")
    
    # 获取所有jpg图片文件
    image_files = glob.glob(os.path.join(source_dir, "*.jpg"))
    
    if not image_files:
        print(f"在源目录中没有找到jpg图片: {source_dir}")
        return False
    
    # 按文件名排序，确保复制顺序一致
    image_files.sort()
    
    print(f"找到 {len(image_files)} 张图片")
    print(f"开始复制图片从 {source_dir} 到 {target_dir}")
    
    success_count = 0
    error_count = 0
    
    for index, image_file in enumerate(image_files, 1):
        # 新的文件名：1.jpg, 2.jpg, 3.jpg...
        new_filename = f"{index}.jpg"
        target_path = os.path.join(target_dir, new_filename)
        
        try:
            # 复制文件
            shutil.copy2(image_file, target_path)
            print(f"已复制: {os.path.basename(image_file)} -> {new_filename}")
            success_count += 1
            
        except Exception as e:
            print(f"复制失败 {os.path.basename(image_file)} -> {new_filename}: {str(e)}")
            error_count += 1
    
    print(f"\n复制完成！")
    print(f"成功: {success_count} 张图片")
    if error_count > 0:
        print(f"失败: {error_count} 张图片")
    
    return True

def main():
    print("=" * 60)
    print("图片复制和重命名脚本")
    print("=" * 60)
    
    try:
        success = copy_and_rename_images()
        if success:
            print("\n脚本执行成功！")
        else:
            print("\n脚本执行失败！")
            
    except Exception as e:
        print(f"\n脚本执行出错: {str(e)}")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 