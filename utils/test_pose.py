import cv2
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO("ckpt/yolo11m-pose.pt")

# 读取图像
image_path = "data/coco8-pose/images/val/000000000113.jpg"
image = cv2.imread(image_path)

# 进行预测
results = model(image)

# 提取关键点
keypoints = results[0].keypoints  # 假设结果中包含关键点

# 定义关键点和骨架连接
# 定义关键点和骨架连接
keypoint_colors = [(0, 255, 0)] * 17  # 绿色，针对17个关键点
skeleton_pairs = [
    (0, 1), (0, 2),  # Nose to Eyes
    (1, 3), (2, 4),  # Eyes to Ears
    (5, 6),          # Shoulders
    (5, 7), (6, 8),  # Shoulders to Elbows
    (7, 9), (8, 10), # Elbows to Wrists
    (5, 11), (6, 12),# Shoulders to Hips
    (11, 13), (12, 14), # Hips to Knees
    (13, 15), (14, 16)  # Knees to Ankles
]

# 打印keypoints以检查其结构
# print(keypoints)

# # 假设keypoints的每个元素是一个数组，包含x, y, conf
# for i, keypoint in enumerate(keypoints.data[0]):
#     if len(keypoint) >= 3:  # 确保有足够的值
#         x, y, conf = keypoint[0], keypoint[1], keypoint[2]
#         if conf > 0.5:  # 只绘制置信度高的关键点
#             cv2.circle(image, (int(x), int(y)), 5, keypoint_colors[i], -1)

# # 绘制骨架
# for pair in skeleton_pairs:
#     if pair[0] < len(keypoints.data[0]) and pair[1] < len(keypoints.data[0]):  # 确保索引在范围内
#         start = keypoints.data[0][pair[0]]
#         end = keypoints.data[0][pair[1]]
#         if len(start) >= 2 and len(end) >= 2:  # 确保有足够的值
#             if len(start) > 2 and len(end) > 2:  # 检查是否有置信度
#                 if start[2] > 0.5 and end[2] > 0.5:  # 只绘制置信度高的骨架
#                     cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 2)
#             else:
#                 # 如果没有置信度信息，可以选择绘制
#                 cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 2)

# 假设keypoints的每个元素是一个数组，包含x, y, conf
for obj_keypoints in keypoints.data:  # 遍历每个检测到的目标
    # keypoints = result.keypoints  # 获取当前目标的关键点
    for i, keypoint in enumerate(obj_keypoints):
        if len(keypoint) >= 3:  # 确保有足够的值
            x, y, conf = keypoint[0], keypoint[1], keypoint[2]
            if conf > 0.5:  # 只绘制置信度高的关键点
                cv2.circle(image, (int(x), int(y)), 5, keypoint_colors[i], -1)

    # 绘制骨架
    for pair in skeleton_pairs:
        if pair[0] < len(obj_keypoints) and pair[1] < len(obj_keypoints):  # 确保索引在范围内
            start = obj_keypoints[pair[0]]
            end = obj_keypoints[pair[1]]
            if len(start) >= 2 and len(end) >= 2:  # 确保有足够的值
                if len(start) > 2 and len(end) > 2:  # 检查是否有置信度
                    if start[2] > 0.5 and end[2] > 0.5:  # 只绘制置信度高的骨架
                        cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 2)
                else:
                    # 如果没有置信度信息，可以选择绘制
                    cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 0, 0), 2)


# 保存结果
# output_path = "/share/Code/ultralytics/results/pose_estimation_result.jpg"  # 指定保存路径
# cv2.imwrite(output_path, image)  # 保存图像

# # 显示结果
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()