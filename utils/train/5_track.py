from ultralytics import YOLO, YOLOMultimodal

# Load an official or custom model
# model = YOLO("yolo11n.pt")  # Load an official Detect model
# model = YOLO("yolo11n-seg.pt")  # Load an official Segment model
# model = YOLO("yolo11n-pose.pt")  # Load an official Pose model
# model = YOLO("path/to/best.pt")  # Load a custom trained model

# # Perform tracking with the model
# results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
# results = model.track(
#     source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml"
# )  # Tracking with ByteTrack tracker

# 常规检测+跟踪
# model = YOLO("yolo11n.pt")  # Load an official Detect model
# results = model.track(source='data/LLVIP/visible_01.mp4', show=True, tracker="bytetrack.yaml")

# 多模态检测+跟踪(EnhancedFMEDA)
# model = YOLOMultimodal("yolo11n-EnhancedFMDEA.yaml", task="multimodal").load('/ultralytics/runs/multimodal/train8/weights/last.pt')  # pretrained YOLO11n model
# results = model.track(source=[['data/LLVIP/visible_01.mp4', 'data/LLVIP/infrared_01.mp4']], show=True, tracker="bytetrack.yaml")

# 多模态检测+跟踪(EnhancedFMEDA)
model = YOLOMultimodal("yolo11s-EnhancedFMDEA.yaml", task="multimodal").load('runs/multimodal/multimodal0317/0319/last.pt')  # pretrained YOLO11n model
results = model.track(source=[['data/LLVIP/visible_01.mp4', 'data/LLVIP/infrared_01.mp4']], show=True, tracker="bytetrack.yaml")

# # 多模态检测+跟踪(EDA)
# model = YOLOMultimodal("yolo11s-DEYOLO.yaml", task="multimodal").load('/ultralytics/runs/multimodal/multimodal0317/best.pt')  # pretrained YOLO11n model
# results = model.track(source=[['data/LLVIP/visible_01.mp4', 'data/LLVIP/infrared_01.mp4']], show=True, tracker="bytetrack.yaml",device=0)
