import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description='将图片序列转换为视频')
    
    # 输入输出参数
    parser.add_argument('--input-dir', type=str, default="runs/torch_val",
                      help='输入图片目录路径')
    parser.add_argument('--output-path', type=str, default="runs/torch_val/video_60.mp4",
                      help='输出视频文件路径')
    
    # 视频参数
    parser.add_argument('--fps', type=int, default=10,
                      help='视频帧率')
    parser.add_argument('--frame-delay', type=int, default=5,
                      help='每帧之间的延迟帧数')
    parser.add_argument('--duration', type=float, default=60,
                      help='视频最大时长（秒），超过此时长将停止合成')
    
    return parser.parse_args()

def generate_video(args):
    """生成视频"""
    # 获取所有图片文件
    input_dir = Path(args.input_dir)
    image_files = sorted([f for f in input_dir.glob('*.jpg')])
    
    if not image_files:
        print(f"错误：在目录 {args.input_dir} 中没有找到jpg图片")
        return
    
    # 读取第一张图片获取尺寸
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print(f"错误：无法读取图片 {image_files[0]}")
        return
    
    # 获取原始图像尺寸
    height, width = first_img.shape[:2]
    
    # 设置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        args.output_path,
        fourcc,
        args.fps,
        (width, height)  # 使用原始图像尺寸
    )
    
    print(f"开始生成视频...")
    print(f"总图片数: {len(image_files)}")
    print(f"输出视频: {args.output_path}")
    print(f"视频尺寸: {width}x{height}")
    print(f"帧率: {args.fps}")
    print(f"帧间延迟: {args.frame_delay} 帧")
    if args.duration:
        print(f"最大时长: {args.duration} 秒")
    
    # 计算每张图片的实际显示时间（秒）
    frame_time = (1 + args.frame_delay) / args.fps
    
    # 处理每张图片
    current_time = 0.0
    processed_images = 0
    
    for img_path in tqdm(image_files, desc="处理图片"):
        # 检查是否达到时长限制
        if args.duration and current_time >= args.duration:
            print(f"\n达到指定时长 {args.duration} 秒，停止合成")
            break
            
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告：无法读取图片 {img_path}，跳过")
            continue
        
        # 检查图片尺寸是否一致
        if img.shape[:2] != (height, width):
            print(f"警告：图片 {img_path} 尺寸不一致，将被调整到 {width}x{height}")
            img = cv2.resize(img, (width, height))
        
        # 写入原始帧
        out.write(img)
        
        # 添加延迟帧
        for _ in range(args.frame_delay):
            out.write(img)
        
        # 更新当前时间
        current_time += frame_time
        processed_images += 1
    
    # 释放资源
    out.release()
    
    # 打印最终信息
    print(f"\n视频生成完成！")
    print(f"保存路径: {args.output_path}")
    print(f"实际时长: {current_time:.2f} 秒")
    print(f"处理图片数: {processed_images}/{len(image_files)}")
    if args.duration:
        print(f"目标时长: {args.duration} 秒")

def main():
    args = parse_args()
    generate_video(args)

if __name__ == '__main__':
    main() 