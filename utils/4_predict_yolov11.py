# from ultralytics import YOLOFusion

# # Load a model
# model = YOLOFusion("yolo11n-DEYOLO.yaml", task="detect").load('/ultralytics/runs/detect/train8/weights/last.pt')  # pretrained YOLO11n model

# # Run batched inference on a list of images
# results = model(["image1.jpg", "image2.jpg"])  # return a list of Results objects

# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk


from ultralytics import YOLOMultimodal
import torch

# Load a model
# model = YOLOMultimodal("yolo11s-DEYOLO.yaml", task="multimodal").load('/ultralytics/runs/multimodal/multimodal0317/best.pt')  # pretrained YOLO11n model
# model.predict([["/ultralytics/data/LLVIP/images/visible/test/190001.jpg", "/ultralytics/data/LLVIP/images/infrared/test/190001.jpg"], # corresponding image pair
#               ["/ultralytics/data/LLVIP/images/visible/test/190002.jpg", "/ultralytics/data/LLVIP/images/infrared/test/190002.jpg"]], 
#               save=True, imgsz=640, conf=0.5, device=0)


# 测试 EnhancedMultimodal
# mapping_matrix = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0.873, 1]])
# model = YOLOMultimodal("yolo11s-EnhancedFMDEA.yaml", task="multimodal").load('runs/multimodal/multimodal0317/0319/last.pt')  # pretrained YOLO11n model
# # model.predict(source=[['data/LLVIP/visible_01.mp4', 'data/LLVIP/infrared_01.mp4']], save=True, imgsz=640, conf=0.5, device=0) # corresponding image pair
# # model.predict(source=[['runs/extract_frame/visible_frame10.jpg', 'runs/extract_frame/infrared_frame10.jpg']], save=True, imgsz=640, conf=0.5, device=0) # corresponding image pair
# model.predict(source=[['data/LLVIP/images/visible/test/190001.jpg', 'data/LLVIP/images/infrared/test/190001.jpg', mapping_matrix]], save=True, imgsz=640, conf=0.5, device=0) # corresponding image pair

# 未配准数据测试
H = torch.tensor([
    [1.06503400e+00, 4.47383490e-02, 2.51326881e+02],
    [1.06740213e-02, 1.49526420e+00, 2.54558633e+01],
    [2.44125413e-05, 3.28032519e-05, 1.00000000e+00]
])
model = YOLOMultimodal("yolo11s-EnhancedFMDEA.yaml", task="multimodal").load('runs/multimodal/multimodal0317/0319/last.pt')  # pretrained YOLO11n model
model.predict(source=[['runs/extract_frame/visible_frame10.jpg', 'runs/extract_frame/infrared_frame10.jpg', H]], save=True, imgsz=640, conf=0.5, device=0) # corresponding image pair
