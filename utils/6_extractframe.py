import cv2

def extract_frame(video_path, frame_number, output_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return False
    
    # 设置要提取的帧的位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    
    # 读取该帧
    ret, frame = cap.read()
    
    if ret:
        # 将帧保存为jpg图片
        cv2.imwrite(output_path, frame)
        print(f"已将第 {frame_number} 帧保存到: {output_path}")
    else:
        print(f"无法读取第 {frame_number} 帧")
    
    # 释放视频对象
    cap.release()
    return ret

# 使用示例
video1_path = "data/LLVIP/visible_01.mp4"
video2_path = "data/LLVIP/infrared_01.mp4"

# 提取两个视频的第10帧（注意：帧的计数从0开始，所以第10帧的索引是9）
extract_frame(video1_path, 9, "runs/extract_frame/visible_frame10.jpg")
extract_frame(video2_path, 9, "runs/extract_frame/infrared_frame10.jpg")