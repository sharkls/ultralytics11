import cv2
import os

def extract_single_frame(video_path, frame_number, output_dir, modality):
    """
    从视频中提取单帧
    
    参数:
        video_path: 视频文件路径
        frame_number: 要提取的帧号
        output_dir: 输出目录
        modality: 模态类型（'visible' 或 'infrared'）
    """
    # 创建模态特定的输出目录
    modal_dir = os.path.join(output_dir, modality)
    os.makedirs(modal_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False
    
    # 设置要提取的帧的位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 读取该帧
    ret, frame = cap.read()
    
    if ret:
        # 构建输出文件路径（统一命名格式）
        output_path = os.path.join(modal_dir, f"frame{frame_number:03d}.jpg")
        # 保存帧
        cv2.imwrite(output_path, frame)
        print(f"已将{modality}第 {frame_number} 帧保存到: {output_path}")
    else:
        print(f"无法读取第 {frame_number} 帧")
    
    # 释放视频对象
    cap.release()
    return ret

def extract_frames(video_path, start_frame, end_frame, output_dir, modality):
    """
    从视频中提取指定范围的帧
    
    参数:
        video_path: 视频文件路径
        start_frame: 起始帧（包含）
        end_frame: 结束帧（包含）
        output_dir: 输出目录
        modality: 模态类型（'visible' 或 'infrared'）
    """
    # 创建模态特定的输出目录
    modal_dir = os.path.join(output_dir, modality)
    os.makedirs(modal_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False
    
    # 对指定范围的每一帧进行处理
    for frame_idx in range(start_frame, end_frame + 1):
        # 设置要提取的帧的位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # 读取该帧
        ret, frame = cap.read()
        
        if ret:
            # 构建输出文件路径（统一命名格式）
            output_path = os.path.join(modal_dir, f"frame{frame_idx:03d}.jpg")
            # 保存帧
            cv2.imwrite(output_path, frame)
            print(f"已将{modality}第 {frame_idx} 帧保存到: {output_path}")
        else:
            print(f"无法读取第 {frame_idx} 帧")
    
    # 释放视频对象
    cap.release()
    return True

def extract_video_frames(video_path, output_dir, modality, frame_number=None, start_frame=None, end_frame=None):
    """
    统一的视频帧提取接口
    
    参数:
        video_path: 视频文件路径
        output_dir: 输出目录
        modality: 模态类型（'visible' 或 'infrared'）
        frame_number: 单帧提取时的帧号
        start_frame: 多帧提取时的起始帧号
        end_frame: 多帧提取时的结束帧号
    """
    if frame_number is not None:
        # 提取单帧
        return extract_single_frame(video_path, frame_number, output_dir, modality)
    elif start_frame is not None and end_frame is not None:
        # 提取多帧
        return extract_frames(video_path, start_frame, end_frame, output_dir, modality)
    else:
        print("请指定要提取的帧号或帧范围")
        return False

# 使用示例
if __name__ == "__main__":
    # 定义视频路径
    video1_path = "data/LLVIP/visible_01.mp4"
    video2_path = "data/LLVIP/infrared_01.mp4"
    
    # 定义输出目录
    output_dir = "runs/extract_frames"
    
    # # 示例1：提取单帧（第10帧）
    # print("\n提取单帧示例：")
    # extract_video_frames(video1_path, output_dir, modality='visible', frame_number=10)
    # extract_video_frames(video2_path, output_dir, modality='infrared', frame_number=10)
    
    # 示例2：提取多帧（9-20帧）
    print("\n提取多帧示例：")
    extract_video_frames(video1_path, output_dir, modality='visible', start_frame=9, end_frame=20)
    extract_video_frames(video2_path, output_dir, modality='infrared', start_frame=9, end_frame=20)