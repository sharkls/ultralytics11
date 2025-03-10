from ultralytics import YOLO
import cv2

# # Load a model
# model = YOLO("ckpt/yolo11m-pose.pt")  # load an official model
# # model = YOLO("path/to/best.pt")  # load a custom model

# # Validate the model
# metrics = model.val()  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category


# 加载模型
model = YOLO("ckpt/yolo11m-pose.pt")

# 读取图像
image_path = "data/coco8-pose/images/val/000000000113.jpg"
image = cv2.imread(image_path)

# 进行预测
results = model(image)

# 方案1：使用OpenCV显示
result_image = results[0].plot()  # 返回numpy数组
cv2.imshow('Result', result_image[:, :, ::-1])  # BGR转RGB
cv2.waitKey(0)
cv2.destroyAllWindows()